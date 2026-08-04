"""
Performance benchmark tests.
Ensures API prediction latency stays within SLO bounds.

Note: rate limiting is disabled for performance tests by using the app's
limiter.reset() to avoid false failures from the 60/minute cap.
Run: pytest tests/test_performance.py -v
"""

import math
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, limiter

REPO_ROOT = Path(__file__).parent.parent

#: The budgets this module enforces; the README publishes them.
PREDICT_BUDGET_S = 0.200
PERCENTILE = 0.99
LATENCY_SAMPLES = 100
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
    """Latency held against the budgets above."""

    def test_predict_single_within_budget(self, client, base_payload):
        """A single /predict call must respond inside the latency budget."""
        start = time.perf_counter()
        resp = client.post("/predict", json=base_payload)
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < PREDICT_BUDGET_S, f"Single prediction took {elapsed:.3f}s, over the {PREDICT_BUDGET_S}s SLO"

    def test_predict_p99_within_budget(self, client, base_payload):
        """The p99 of a run of sequential predictions must stay inside the budget.

        Uses enough samples that the 99th percentile is a genuine percentile
        rather than the max of a shorter run.
        """
        times = []
        for _ in range(LATENCY_SAMPLES):
            start = time.perf_counter()
            resp = client.post("/predict", json=base_payload)
            elapsed = time.perf_counter() - start
            assert resp.status_code == 200
            times.append(elapsed)

        times.sort()
        p50 = times[math.ceil(0.50 * LATENCY_SAMPLES) - 1]
        p99 = times[math.ceil(PERCENTILE * LATENCY_SAMPLES) - 1]  # nearest rank
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
    """Sequential throughput held against the budgets above."""

    def test_throughput_within_budget(self, client, base_payload):
        start = time.perf_counter()
        for _ in range(THROUGHPUT_CALLS):
            resp = client.post("/predict", json=base_payload)
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start
        assert elapsed < THROUGHPUT_BUDGET_S, (
            f"{THROUGHPUT_CALLS} predictions took {elapsed:.1f}s, over the {THROUGHPUT_BUDGET_S}s budget"
        )


def _paragraphs_citing_this_module() -> list[str]:
    """Every README paragraph that names this file as the thing enforcing a budget."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    citing = [" ".join(p.split()) for p in re.split(r"\r?\n\s*\r?\n", text) if "tests/test_performance.py" in p]
    assert len(citing) >= 2, f"only {len(citing)} README paragraphs cite this module — the scan has gone stale"
    return citing


def test_every_published_slo_states_the_budgets_enforced_here():
    """Each paragraph states both budgets in full, and prints no figure this module does not decide."""
    percentile = f"{PERCENTILE * 100:.0f}"
    milliseconds = f"{PREDICT_BUDGET_S * 1000:.0f}"
    seconds = f"{THROUGHPUT_BUDGET_S:.0f}"
    clauses = (
        f"p{percentile} < {milliseconds}ms over {LATENCY_SAMPLES} sequential",
        f"{THROUGHPUT_CALLS} predictions inside {seconds}s",
    )
    sourced = {percentile, milliseconds, str(LATENCY_SAMPLES), str(THROUGHPUT_CALLS), seconds}
    for published in _paragraphs_citing_this_module():
        missing = [clause for clause in clauses if clause not in published]
        assert not missing, f"a README paragraph does not state {missing}: {published!r}"
        printed = set(re.findall(r"\d+(?:\.\d+)?", published))
        assert printed == sourced, (
            f"a README paragraph prints {sorted(printed - sourced)} that nothing here decides "
            f"and omits {sorted(sourced - printed)}: {published!r}"
        )
