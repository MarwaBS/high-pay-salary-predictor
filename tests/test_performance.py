"""
Performance benchmark tests.
Ensures API prediction latency stays within SLO bounds.

Note: rate limiting is disabled for performance tests by using the app's
limiter.reset() to avoid false failures from the 60/minute cap.
Run: pytest tests/test_performance.py -v
"""

import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, limiter

REPO_ROOT = Path(__file__).parent.parent

#: The budgets this module enforces; the README publishes them.
PREDICT_BUDGET_S = 0.200
P99_SAMPLES = 100
THROUGHPUT_CALLS = 50
THROUGHPUT_BUDGET_S = 5.0


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


@pytest.fixture(autouse=True)
def _disable_rate_limit():
    """Disable rate limiting for performance benchmarks."""
    limiter.enabled = False
    yield
    limiter.enabled = True


class TestLatency:
    """SLO: single prediction must complete under 200ms (p99)."""

    def test_predict_single_under_200ms(self, client, base_payload):
        """A single /predict call must respond within 200ms."""
        start = time.perf_counter()
        resp = client.post("/predict", json=base_payload)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < PREDICT_BUDGET_S, f"Single prediction took {elapsed:.3f}s, over the {PREDICT_BUDGET_S}s SLO"

    def test_predict_p99_under_200ms(self, client, base_payload):
        """p99 of 100 sequential predictions must stay under 200ms.

        Uses enough samples that the 99th percentile is a genuine percentile
        rather than the max of a shorter run.
        """
        times = []
        for _ in range(P99_SAMPLES):
            start = time.perf_counter()
            resp = client.post("/predict", json=base_payload)
            elapsed = time.perf_counter() - start
            assert resp.status_code == 200
            times.append(elapsed)

        times.sort()
        p50 = times[math.ceil(0.50 * P99_SAMPLES) - 1]
        p99 = times[math.ceil(0.99 * P99_SAMPLES) - 1]  # nearest rank
        assert p99 < PREDICT_BUDGET_S, f"p99 latency {p99:.3f}s over the {PREDICT_BUDGET_S}s SLO (p50={p50:.3f}s)"

    def test_health_under_100ms(self, client):
        """Health endpoint must respond within 100ms (warm)."""
        # Warm-up call (first call may include Prometheus init overhead)
        client.get("/health")
        start = time.perf_counter()
        resp = client.get("/health")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 0.100, f"/health took {elapsed:.3f}s, exceeding 100ms"

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
        for _ in range(THROUGHPUT_CALLS):
            resp = client.post("/predict", json=base_payload)
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start
        assert elapsed < THROUGHPUT_BUDGET_S, (
            f"{THROUGHPUT_CALLS} predictions took {elapsed:.1f}s, over the {THROUGHPUT_BUDGET_S}s budget"
        )


def _slo_paragraph() -> str:
    """The README's SLO bullet, continuation lines included."""
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if "**Enforced SLO:**" in line]
    assert len(starts) == 1, f"the SLO claim is anchored {len(starts)} times, expected 1"
    end = next((i for i in range(starts[0] + 1, len(lines)) if not lines[i].strip()), len(lines))
    return " ".join(lines[starts[0] : end])


def test_the_published_slo_prints_the_budgets_enforced_here():
    """Every figure the README's SLO claim prints is one of the budgets above."""
    published = _slo_paragraph()
    expected = {
        f"{PREDICT_BUDGET_S * 1000:.0f}ms",
        f"{P99_SAMPLES} sequential",
        f"{THROUGHPUT_CALLS} predictions",
        f"{THROUGHPUT_BUDGET_S:.0f}s",
    }
    missing = sorted(token for token in expected if token not in published)
    assert not missing, f"README's SLO claim does not print {missing}: {published.strip()!r}"
