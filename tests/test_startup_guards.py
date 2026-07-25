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
