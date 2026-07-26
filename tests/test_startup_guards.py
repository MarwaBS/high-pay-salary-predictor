"""The fail-loud startup guards must actually refuse to serve.

Each guard below crashes the liveness probe rather than serving a model that
would be silently wrong. Testing the helpers in isolation leaves the *wiring*
unproven — a guard whose result is computed and then ignored looks identical to
one that is enforced. These drive the real lifespan so deleting any guard turns
a test red.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as m
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
            m, "_artifact_mismatches", lambda _files, _recorded: ["model (x.ubj): loaded a != recorded b"]
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
        with TestClient(m.app) as client:
            digests = client.get("/health").json()["artifact_sha256"]
        assert "classifier" not in digests
        assert "model" in digests

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

    def test_namespace_includes_the_served_artefact_digest(self):
        with TestClient(m.app):
            model_digest = m.state.artifact_sha256["model"]
            assert m.state.model_version in m.cache.version
            assert model_digest[:12] in m.cache.version, (
                "cache namespace does not bind to the model bytes, so a "
                "hyperparameter-only retrain would reuse the previous namespace"
            )


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
