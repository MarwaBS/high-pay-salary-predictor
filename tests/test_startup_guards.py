"""The fail-loud startup guards must actually refuse to serve.

Each guard below crashes the liveness probe rather than serving a model that
would be silently wrong. Testing the helpers in isolation leaves the *wiring*
unproven — a guard whose result is computed and then ignored looks identical to
one that is enforced. These drive the real lifespan so deleting any guard turns
a test red.
"""

import json
import math

import pytest
from fastapi.testclient import TestClient

import api.main as m
from api.drift import DriftMonitor
from pipeline import load_metrics


def _metrics_without(real_load_metrics, artefact):
    """Real metrics with one artefact's digest removed."""

    def _loader(path):
        metrics = dict(real_load_metrics(path))
        metrics["artifact_sha256"] = {k: v for k, v in metrics["artifact_sha256"].items() if k != artefact}
        return metrics

    return _loader


def _metrics_with_flipped(real_load_metrics, artefact):
    """Real metrics with one artefact's digest replaced by a value no file hashes to."""

    def _loader(path):
        metrics = dict(real_load_metrics(path))
        metrics["artifact_sha256"] = {**metrics["artifact_sha256"], artefact: "0" * 64}
        return metrics

    return _loader


PAYLOAD = {
    "state": "CA",
    "education_level": "Bachelor's degree",
    "gender": "Female",
    "age": 32,
}


class TestStartupRefusesToServe:
    def test_artifact_digest_mismatch_aborts_startup(self, monkeypatch):
        """Serving bytes that differ from the recorded digests is a hard stop."""
        monkeypatch.setattr(
            m, "artifact_mismatches", lambda _files, _recorded: ["model (x.ubj): loaded a != recorded b"]
        )
        with pytest.raises(RuntimeError, match="Artifact integrity check failed"):
            with TestClient(m.app):
                pass

    def test_point_estimate_model_aborts_startup(self, monkeypatch):
        """A non-quantile model would collapse every 80% band to a single value."""
        monkeypatch.setattr(m, "is_quantile_model", lambda _model: False)
        with pytest.raises(RuntimeError, match="not a multi-quantile model"):
            with TestClient(m.app):
                pass

    def test_classifier_threshold_mismatch_aborts_startup(self, monkeypatch):
        """A config threshold the classifier was never fitted against is a stop.

        Serving it would advertise a ``premium_threshold`` that does not match
        the boundary the model actually learned.
        """
        real_load_metrics = m.load_metrics

        def _shifted_threshold(path):
            metrics = dict(real_load_metrics(path))
            metrics["classifier_threshold"] = 999_999
            return metrics

        monkeypatch.setattr(m, "load_metrics", _shifted_threshold)
        with pytest.raises(RuntimeError, match="Classifier threshold mismatch"):
            with TestClient(m.app):
                pass

    def test_healthy_artefacts_start_normally(self):
        """The guards above must not fire on the committed artefacts."""
        with TestClient(m.app) as client:
            assert client.get("/health").status_code == 200


ARTEFACTS = ["model", "classifier", "features", "group_means", "baseline_stats", "conformal"]


class TestEveryLoadedArtefactIsVerified:
    """An artefact the recorded digests do not cover must stop startup.

    Verifying only the keys the metrics file happens to name would leave the
    rest served unchecked, which is what the digests exist to prevent.
    """

    def test_absent_metrics_file_loads_as_no_digests(self, tmp_path):
        """Ties the stubbed metrics below to the real file: absent loads as {}."""
        assert load_metrics(str(tmp_path / "model_metrics.json")) == {}

    @pytest.mark.parametrize(
        "metrics",
        [{}, {"artifact_sha256": {}}, {"artifact_sha256": None}, {"model_version": "2.0.0"}],
        ids=["file-absent", "empty-map", "null-map", "key-missing"],
    )
    def test_startup_refuses_without_digests(self, monkeypatch, metrics):
        monkeypatch.setattr(m, "load_metrics", lambda _path: dict(metrics))
        with pytest.raises(RuntimeError, match="no recorded digest in artifact_sha256"):
            with TestClient(m.app):
                pass

    @pytest.mark.parametrize("artefact", ARTEFACTS)
    def test_startup_refuses_when_one_digest_is_unrecorded(self, monkeypatch, artefact):
        """A map covering everything but one artefact leaves that one unverified."""
        monkeypatch.setattr(m, "load_metrics", _metrics_without(m.load_metrics, artefact))
        with pytest.raises(RuntimeError, match=f"{artefact} .*no recorded digest"):
            with TestClient(m.app):
                pass

    def test_health_omits_the_digest_of_an_artefact_that_did_not_load(self, monkeypatch):
        """The classifier is optional; advertising its digest would claim it is served."""

        def _absent(_path):
            raise FileNotFoundError("no classifier artefact")

        monkeypatch.setattr(m, "load_classifier", _absent)
        monkeypatch.setattr(m.state, "classifier", None)
        recorded_classifier = m.load_metrics(str(m.ROOT / m.VALIDATED_CFG.model.metrics_path))["artifact_sha256"][
            "classifier"
        ]
        with TestClient(m.app) as client:
            digests = client.get("/health").json()["artifact_sha256"]
            namespace = m.cache.version
        assert "classifier" not in digests
        assert "model" in digests
        # A pod serving no classifier must not share a cache namespace with one
        # that does, or it poisons a shared Redis with null probabilities.
        assert recorded_classifier[:12] not in namespace

    @pytest.mark.parametrize("artefact", ARTEFACTS)
    def test_every_recorded_digest_is_compared_against_its_file(self, monkeypatch, artefact):
        """Each artefact is hashed and compared, not just the ones that happen to be checked."""
        monkeypatch.setattr(m, "load_metrics", _metrics_with_flipped(m.load_metrics, artefact))
        with pytest.raises(RuntimeError, match=f"Artifact integrity check failed.*{artefact}"):
            with TestClient(m.app):
                pass


class TestCacheNamespaceBindsToTheServedModel:
    """Cached predictions must be addressed by the model that produced them.

    ``model_version`` is built from the git SHA and the input CSV hash, so a
    retrain that only edits hyperparameters leaves it identical while the model
    bytes change. Namespacing on it alone lets a shared Redis serve the previous
    model's predictions for a full TTL.
    """

    def test_namespace_binds_every_artefact_that_shapes_a_cached_response(self):
        """A cached body carries the regressor's quantiles, the classifier's
        probability and the conformal-widened interval, so a retrain of any one
        of them alone must not reuse the previous namespace."""
        with TestClient(m.app):
            assert m.state.model_version in m.cache.version
            for key in m._CACHE_KEYED_ARTEFACTS:
                digest = m.state.artifact_sha256.get(key)
                if digest is None:
                    continue  # optional artefact this process did not load
                assert digest[:12] in m.cache.version, f"namespace does not bind to the {key} bytes"

    def test_the_keyed_set_names_every_response_shaping_artefact(self):
        """Named literally on purpose. The test above iterates the constant, so
        deriving this expectation from it too would let the constant shrink and
        take both assertions with it."""
        assert {"model", "classifier", "conformal"} <= set(m._CACHE_KEYED_ARTEFACTS)


class TestClassifierHeadIsActuallyServed:
    """The premium-tier probability is an advertised response field.

    A missing classifier artefact degrades it to ``None`` on every response, so
    without this the feature can disappear entirely under a green suite.
    """

    def test_predict_returns_a_premium_probability(self):
        with TestClient(m.app) as client:
            payload = {**PAYLOAD, "occupation": m.state.occupations[0]}
            body = client.post("/predict", json=payload).json()
            assert body["p_above_premium_threshold"] is not None
            assert 0.0 <= body["p_above_premium_threshold"] <= 1.0
            assert body["premium_threshold"] == m.state.premium_threshold

    def test_batch_returns_a_premium_probability(self):
        with TestClient(m.app) as client:
            payload = {**PAYLOAD, "occupation": m.state.occupations[0]}
            body = client.post("/predict/batch", json={"items": [payload]}).json()
            assert body["items"][0]["p_above_premium_threshold"] is not None


class TestServedCoverageWiring:
    """The coverage the API reports must come through the helper that treats a
    recorded 0.0 as a real measurement, not as a missing value."""

    def test_a_recorded_zero_coverage_survives_startup(self, monkeypatch):
        real_load_metrics = m.load_metrics

        def _zero_conformal(path):
            metrics = dict(real_load_metrics(path))
            metrics["conformal_coverage_80"] = 0.0
            metrics["quantile_coverage_80"] = 0.8
            return metrics

        monkeypatch.setattr(m, "load_metrics", _zero_conformal)
        with TestClient(m.app):
            assert m.state.quantile_coverage_80 == 0.0

    def test_the_helper_prefers_the_conformalized_number(self):
        assert m._served_interval_coverage({"conformal_coverage_80": 0.0, "quantile_coverage_80": 0.8}) == 0.0
        assert m._served_interval_coverage({"quantile_coverage_80": 0.8}) == 0.8


class TestConfiguredArtefactPaths:
    def test_the_api_reads_the_configured_baseline_path(self, monkeypatch):
        """Renaming the artefact in config must move where startup looks, or the
        key is decorative and the real path is hardcoded."""
        renamed = m.VALIDATED_CFG.model.model_copy(update={"baseline_stats_path": "models/renamed_baseline.json"})
        monkeypatch.setattr(m.VALIDATED_CFG, "model", renamed)
        with pytest.raises(RuntimeError, match="renamed_baseline.json"):
            with TestClient(m.app):
                pass

    def test_the_monitor_reads_the_artefact_whose_digest_startup_verified(self):
        """Startup verifies ``baseline_stats``, then builds the monitor. Reading
        the config a second time lets those be different files, and the one that
        aborts on a mismatch is not the one the detector opened."""
        declared = json.loads((m.ROOT / m.VALIDATED_CFG.model.baseline_stats_path).read_text(encoding="utf-8"))
        with TestClient(m.app) as client:
            assert client.get("/health").status_code == 200
            assert m.state.drift_monitor.baseline == declared


class TestTheConfiguredDriftWindowReachesTheMonitor:
    """``config.yaml::drift.window`` has to be what the served monitor runs on.

    Startup is the only place the two meet, so a monitor built with anything
    else — a literal, or a default re-added to ``DriftMonitor`` — leaves the
    config key decorative while every other drift test still passes.
    """

    def test_the_served_monitor_runs_on_the_configured_window(self):
        with TestClient(m.app):
            assert m.state.drift_monitor.window == m.VALIDATED_CFG.drift.window

    def test_changing_the_configured_window_moves_the_served_one(self, monkeypatch):
        """Equality against the config alone would also hold for a hardcoded 500."""
        moved = m.VALIDATED_CFG.drift.model_copy(update={"window": m.VALIDATED_CFG.drift.window + 137})
        monkeypatch.setattr(m.VALIDATED_CFG, "drift", moved)
        with TestClient(m.app):
            assert m.state.drift_monitor.window == moved.window

    def test_the_monitor_refuses_to_pick_a_window_for_its_caller(self):
        """A default would let a caller that forgets the config still start."""
        with pytest.raises(TypeError):
            DriftMonitor(baseline_stats={"Age": {"mean": 40.0, "std": 10.0, "min": 19.0, "max": 94.0}})

    def test_a_window_under_the_handover_aborts_startup(self, monkeypatch):
        """Serving it would advertise a sensitivity the window cannot deliver."""
        monitor = m.DriftMonitor({"Age": {"mean": 40.0, "std": 10.0, "min": 19.0, "max": 94.0}}, window=1)
        too_small = monitor.effect_floor_handover() - 1
        monkeypatch.setattr(m.VALIDATED_CFG, "drift", m.VALIDATED_CFG.drift.model_copy(update={"window": too_small}))
        with pytest.raises(RuntimeError, match="is below"):
            with TestClient(m.app):
                pass

    def test_a_window_exactly_at_the_handover_is_accepted(self, monkeypatch):
        """The handover is the first sufficient window, so the guard is ``<``."""
        monitor = m.DriftMonitor({"Age": {"mean": 40.0, "std": 10.0, "min": 19.0, "max": 94.0}}, window=1)
        exact = monitor.effect_floor_handover()
        monkeypatch.setattr(m.VALIDATED_CFG, "drift", m.VALIDATED_CFG.drift.model_copy(update={"window": exact}))
        with TestClient(m.app) as client:
            assert client.get("/health").status_code == 200

    @pytest.mark.parametrize("knob", ["min_effect_size", "alert_threshold"])
    def test_retuning_either_knob_moves_the_window_the_guard_demands(self, monkeypatch, knob):
        """The bound is a function of both; a literal in place of either would
        keep demanding the window the shipped tuning happened to need.

        Each tuning is derived from the configured window so that it is one the
        window cannot satisfy — a fixed pair would also fail whenever someone
        raised the window past it, which the config says they may.
        """
        configured = m.VALIDATED_CFG.drift.window
        alert, effect = 2.0, 0.2
        if knob == "min_effect_size":
            effect = alert * math.sqrt(2.0 / configured) * 0.9
        else:
            alert = effect * math.sqrt(configured / 2.0) * 1.1
        monkeypatch.setattr(m.DriftMonitor.__init__, "__defaults__", (alert, effect, None))
        with pytest.raises(RuntimeError, match="is below"):
            with TestClient(m.app):
                pass

    def test_the_guard_reads_the_verdict_floor_rather_than_a_literal(self, monkeypatch):
        """Moving the floor must move what the guard demands."""
        monkeypatch.setattr("api.main.MIN_WINDOW_FOR_VERDICT", m.VALIDATED_CFG.drift.window + 1)
        with pytest.raises(RuntimeError, match="is below"):
            with TestClient(m.app):
                pass

    def test_a_window_under_the_verdict_floor_aborts_even_when_the_handover_is_lower(self, monkeypatch):
        """A loose ``min_effect_size`` drops the handover under the verdict floor,
        at which point the floor is the binding bound and the handover is not."""
        monkeypatch.setattr(m.DriftMonitor.__init__, "__defaults__", (2.0, 1.0, None))
        below = m.MIN_WINDOW_FOR_VERDICT - 1
        monkeypatch.setattr(m.VALIDATED_CFG, "drift", m.VALIDATED_CFG.drift.model_copy(update={"window": below}))
        with pytest.raises(RuntimeError, match="is below"):
            with TestClient(m.app):
                pass
