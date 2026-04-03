"""
Performance benchmark tests.
Ensures API prediction latency stays within SLO bounds.
Run: pytest tests/test_performance.py -v
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def base_payload(client):
    """Valid predict payload using the first available occupation from /meta."""
    occupations = client.get("/meta").json()["occupations"]
    assert len(occupations) > 0, "No occupations returned by /meta"
    return {
        "state": "CA",
        "occupation": occupations[0],
        "education_level": "Bachelor's degree",
        "gender": "Female",
        "age": 32,
    }


class TestLatency:
    """SLO: single prediction must complete under 200ms (p99)."""

    def test_predict_single_under_200ms(self, client, base_payload):
        """A single /predict call must respond within 200ms."""
        start = time.perf_counter()
        resp = client.post("/predict", json=base_payload)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 0.200, f"Single prediction took {elapsed:.3f}s, exceeding 200ms SLO"

    def test_predict_p99_under_200ms(self, client, base_payload):
        """p99 of 50 sequential predictions must stay under 200ms."""
        times = []
        for _ in range(50):
            start = time.perf_counter()
            resp = client.post("/predict", json=base_payload)
            elapsed = time.perf_counter() - start
            assert resp.status_code == 200
            times.append(elapsed)

        times.sort()
        p50 = times[24]
        p99 = times[49]
        assert p99 < 0.200, f"p99 latency {p99:.3f}s exceeds 200ms SLO (p50={p50:.3f}s)"

    def test_health_under_50ms(self, client):
        """Health endpoint must respond within 50ms."""
        start = time.perf_counter()
        resp = client.get("/health")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 0.050, f"/health took {elapsed:.3f}s, exceeding 50ms"

    def test_meta_under_100ms(self, client):
        """/meta endpoint must respond within 100ms."""
        start = time.perf_counter()
        resp = client.get("/meta")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 0.100, f"/meta took {elapsed:.3f}s, exceeding 100ms"


class TestThroughput:
    """Baseline throughput: 50 sequential predictions under 5 seconds."""

    def test_50_predictions_under_5s(self, client, base_payload):
        start = time.perf_counter()
        for _ in range(50):
            resp = client.post("/predict", json=base_payload)
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"50 predictions took {elapsed:.1f}s, exceeding 5s budget"
