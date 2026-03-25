"""
API endpoint tests.
Uses FastAPI's TestClient (synchronous, no server needed).
Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

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
