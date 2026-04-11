"""
API endpoint tests.
Uses FastAPI's TestClient (synchronous, no server needed).
Run: pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Start the app once for all tests in this module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def base_payload(client):
    """Return a valid predict payload using the first available occupation from /meta.

    Fetching occupation dynamically prevents the test from breaking when the
    dataset changes and 'Software Developers' is no longer present.
    """
    occupations = client.get("/meta").json()["occupations"]
    return {
        "state": "CA",
        "occupation": occupations[0],
        "education_level": "Bachelor's degree",
        "gender": "Female",
        "age": 32,
    }


# ── Health & Meta ──────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_dataset_rows(self, client):
        data = client.get("/health").json()
        assert data["dataset_rows"] > 1000

    def test_root_returns_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data


class TestMeta:
    def test_meta_200(self, client):
        assert client.get("/meta").status_code == 200

    def test_meta_has_50_states(self, client):
        data = client.get("/meta").json()
        assert len(data["states"]) == 50

    def test_meta_has_occupations(self, client):
        data = client.get("/meta").json()
        assert len(data["occupations"]) > 10

    def test_meta_education_levels(self, client):
        data = client.get("/meta").json()
        assert set(data["education_levels"]) == {
            "Bachelor's degree",
            "Master's degree",
            "Professional degree",
            "Doctoral degree",
        }


# ── Prediction ────────────────────────────────────────────────────────────────


class TestPredict:
    def test_predict_200(self, client, base_payload):
        r = client.post("/predict", json=base_payload)
        assert r.status_code == 200, r.text

    def test_predict_returns_salary(self, client, base_payload):
        data = client.post("/predict", json=base_payload).json()
        assert "predicted_salary" in data
        assert data["predicted_salary"] > 0

    def test_predict_salary_above_threshold(self, client, base_payload):
        """Predicted salary should be above the $100K minimum."""
        data = client.post("/predict", json=base_payload).json()
        assert data["predicted_salary"] >= 50_000  # generous lower bound for model

    def test_predict_percentile_range(self, client, base_payload):
        data = client.post("/predict", json=base_payload).json()
        assert 0 <= data["percentile_in_group"] <= 100

    def test_predict_group_benchmarks_present(self, client, base_payload):
        data = client.post("/predict", json=base_payload).json()
        for field in ("group_median", "group_mean", "group_size"):
            assert field in data, f"Missing field: {field}"

    def test_predict_returns_prediction_interval(self, client, base_payload):
        data = client.post("/predict", json=base_payload).json()
        assert "prediction_interval_low" in data
        assert "prediction_interval_high" in data

    def test_predict_interval_ordered(self, client, base_payload):
        """Lower PI bound must be less than the upper PI bound."""
        data = client.post("/predict", json=base_payload).json()
        assert data["prediction_interval_low"] < data["prediction_interval_high"]

    def test_predict_interval_contains_prediction(self, client, base_payload):
        """Predicted salary should lie within the empirical 80% PI."""
        data = client.post("/predict", json=base_payload).json()
        assert data["prediction_interval_low"] <= data["predicted_salary"] <= data["prediction_interval_high"]

    def test_predict_echoes_inputs(self, client, base_payload):
        data = client.post("/predict", json=base_payload).json()
        assert data["state"] == base_payload["state"]
        assert data["occupation"] == base_payload["occupation"]
        assert data["gender"] == base_payload["gender"]
        assert data["age"] == base_payload["age"]

    def test_predict_with_optional_bls_fields(self, client, base_payload):
        """Optional BLS fields (employment, lq, jobs_per_1000, hourly_mean) can be supplied."""
        payload = {
            **base_payload,
            "hourly_mean": 75.0,
            "employment": 5000,
            "location_quotient": 1.2,
            "jobs_per_1000": 3.5,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_gender_case_insensitive(self, client, base_payload):
        payload = {**base_payload, "gender": "male"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["gender"] == "Male"

    def test_predict_state_case_insensitive(self, client, base_payload):
        payload = {**base_payload, "state": "ca"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_unknown_state_422(self, client):
        payload = {
            "state": "ZZ",
            "occupation": "Software Developers",
            "education_level": "Bachelor's degree",
            "gender": "Female",
            "age": 32,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_unknown_occupation_422(self, client):
        payload = {
            "state": "CA",
            "occupation": "Unicorn Wrangler",
            "education_level": "Bachelor's degree",
            "gender": "Female",
            "age": 32,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_gender_422(self, client):
        payload = {
            "state": "CA",
            "occupation": "Software Developers",
            "education_level": "Bachelor's degree",
            "gender": "Robot",
            "age": 32,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_age_below_minimum_422(self, client):
        payload = {
            "state": "CA",
            "occupation": "Software Developers",
            "education_level": "Bachelor's degree",
            "gender": "Male",
            "age": 10,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_missing_required_field_422(self, client):
        r = client.post("/predict", json={"state": "CA"})
        assert r.status_code == 422


# ── Prediction Cache ─────────────────────────────────────────────────────────


class TestPredictionCache:
    """Verify that /predict consults the cache before inference and writes
    results back on a miss. Uses a MagicMock for ``api.main.cache`` so no
    live Redis is required.
    """

    def test_cache_miss_then_hit(self, client, base_payload, monkeypatch):
        """First request: cache miss → cache.set called. Second request:
        cache hit → response comes from cache and cache.set is not called
        again with the same key."""
        fake_cache = MagicMock()
        # First call: miss. Second call: hit with a canned response.
        cached_response = {
            "predicted_salary": 123456.78,
            "predicted_p10": 90000.0,
            "predicted_p50": 123456.78,
            "predicted_p90": 200000.0,
            "prediction_interval_low": 90000.0,
            "prediction_interval_high": 200000.0,
            "percentile_in_group": 55.0,
            "group_median": 150000.0,
            "group_mean": 160000.0,
            "group_size": 42,
            "state": base_payload["state"],
            "occupation": base_payload["occupation"],
            "education_level": base_payload["education_level"],
            "gender": base_payload["gender"],
            "age": base_payload["age"],
        }
        fake_cache.get.side_effect = [None, cached_response]
        monkeypatch.setattr(api_main, "cache", fake_cache)

        # Miss — model runs, cache.set called
        r1 = client.post("/predict", json=base_payload)
        assert r1.status_code == 200, r1.text
        assert fake_cache.get.call_count == 1
        assert fake_cache.set.call_count == 1

        # Hit — response echoes cached payload, cache.set NOT called again
        r2 = client.post("/predict", json=base_payload)
        assert r2.status_code == 200, r2.text
        assert r2.json()["predicted_salary"] == 123456.78
        assert r2.json()["group_size"] == 42
        assert fake_cache.get.call_count == 2
        assert fake_cache.set.call_count == 1  # unchanged — no set on hit

    def test_cache_get_called_with_validated_payload(self, client, base_payload, monkeypatch):
        """Cache key must be built from the Pydantic-normalised request, so
        case-insensitive inputs ('ca' → 'CA', 'male' → 'Male') hit the same
        cache entry. Proves the cache key is stable under input normalisation.
        """
        fake_cache = MagicMock()
        fake_cache.get.return_value = None
        monkeypatch.setattr(api_main, "cache", fake_cache)

        payload_lower = {**base_payload, "state": "ca", "gender": "male"}
        r = client.post("/predict", json=payload_lower)
        assert r.status_code == 200

        # The key passed to cache.get must contain the normalised values.
        key_arg = fake_cache.get.call_args[0][0]
        assert key_arg["state"] == "CA"
        assert key_arg["gender"] == "Male"

    def test_default_cache_is_disabled_noop(self):
        """With REDIS_URL unset (default), the real cache instance is a
        silent no-op and does not crash the app."""
        # The module-level ``cache`` is a real PredictionCache; when
        # REDIS_URL is empty it should report enabled=False and return
        # None on all lookups.
        assert api_main.cache.enabled is False
        assert api_main.cache.get({"state": "CA"}) is None
        # .set() must not raise even when disabled
        api_main.cache.set({"state": "CA"}, {"predicted_salary": 1.0})


# ── Batch Prediction ─────────────────────────────────────────────────────────


class TestPredictBatch:
    """Verify /predict/batch happy path, ordering, validation errors, and limits."""

    def test_batch_200(self, client, base_payload):
        resp = client.post("/predict/batch", json={"items": [base_payload, base_payload]})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["items"]) == 2

    def test_batch_preserves_order(self, client, base_payload):
        payload_a = {**base_payload, "age": 25}
        payload_b = {**base_payload, "age": 55}
        resp = client.post("/predict/batch", json={"items": [payload_a, payload_b]})
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert items[0]["age"] == 25
        assert items[1]["age"] == 55

    def test_batch_validation_error_on_bad_item(self, client, base_payload):
        bad = {**base_payload, "state": "ZZ"}
        resp = client.post("/predict/batch", json={"items": [base_payload, bad]})
        assert resp.status_code == 422
        # The error message should identify the offending index.
        assert "Item 1" in resp.text

    def test_batch_empty_list_rejected(self, client):
        resp = client.post("/predict/batch", json={"items": []})
        assert resp.status_code == 422

    def test_batch_quantile_fields_present(self, client, base_payload):
        resp = client.post("/predict/batch", json={"items": [base_payload]})
        item = resp.json()["items"][0]
        assert "predicted_p10" in item
        assert "predicted_p50" in item
        assert "predicted_p90" in item
        assert item["predicted_p10"] <= item["predicted_p50"] <= item["predicted_p90"]


# ── Drift Endpoint ───────────────────────────────────────────────────────────


class TestDriftEndpoint:
    def test_drift_200(self, client):
        """GET /drift should return 200 regardless of observation count."""
        r = client.get("/drift")
        assert r.status_code == 200

    def test_drift_reports_after_predictions(self, client, base_payload):
        """After making predictions, /drift should report observations."""
        for _ in range(35):
            client.post("/predict", json=base_payload)
        r = client.get("/drift")
        data = r.json()
        assert data.get("observations", 0) >= 35
        assert "features" in data
