"""
Tests for the drift detection module (api/drift.py).
Verifies detection logic, edge cases, rolling window, and persistence.
Run: pytest tests/test_drift.py -v
"""

import json

import numpy as np
import pytest

from api.drift import DriftMonitor, save_baseline_stats


@pytest.fixture
def baseline_stats():
    """Baseline with known distributions for two features."""
    return {
        "Age": {"mean": 40.0, "std": 10.0, "min": 18.0, "max": 80.0},
        "Education_Ord": {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 4.0},
    }


@pytest.fixture
def monitor(baseline_stats):
    return DriftMonitor(baseline_stats=baseline_stats, window=100, alert_threshold=2.0)


# ── Detection Logic ──────────────────────────────────────────────────────────


class TestDriftDetection:
    def test_no_drift_on_normal_observations(self, monitor):
        """Observations matching baseline should not trigger drift."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            monitor.observe({"Age": float(rng.normal(40, 10)), "Education_Ord": float(rng.normal(2, 1))})
        report = monitor.check_drift()
        assert not report["any_drifted"]

    def test_drift_detected_on_shifted_mean(self, monitor):
        """Large mean shift (3 std) should flag drift."""
        for _ in range(50):
            monitor.observe({"Age": 70.0, "Education_Ord": 2.0})  # Age shifted by 3 std
        report = monitor.check_drift()
        assert report["any_drifted"]
        assert report["features"]["Age"]["drifted"] is True
        assert report["features"]["Age"]["z_score"] > 2.0
        # Education_Ord should NOT be flagged (unchanged)
        assert report["features"]["Education_Ord"]["drifted"] is False

    def test_drift_clears_after_normal_observations(self, baseline_stats):
        """Drift flag should clear when window fills with normal data."""
        mon = DriftMonitor(baseline_stats, window=50, alert_threshold=2.0)
        # Fill with drifted data
        for _ in range(50):
            mon.observe({"Age": 70.0, "Education_Ord": 2.0})
        assert mon.check_drift()["any_drifted"]
        # Overwrite with normal data
        for _ in range(50):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        assert not mon.check_drift()["any_drifted"]


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestDriftEdgeCases:
    def test_insufficient_observations(self, monitor):
        """Under 30 observations should return empty features with message."""
        for _ in range(10):
            monitor.observe({"Age": 70.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
        assert report["features"] == {}
        assert "Need at least 30" in report.get("message", "")
        assert not report["any_drifted"]

    def test_exactly_30_observations_reports(self, monitor):
        """Exactly 30 observations should produce a drift report."""
        for _ in range(30):
            monitor.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
        assert "features" in report
        assert len(report["features"]) == 2

    def test_zero_std_feature(self):
        """Feature with zero std in baseline should not crash."""
        stats = {"Constant": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0}}
        mon = DriftMonitor(baseline_stats=stats, window=50)
        for _ in range(35):
            mon.observe({"Constant": 5.0})
        report = mon.check_drift()
        assert report["features"]["Constant"]["z_score"] == 0.0
        assert not report["features"]["Constant"]["drifted"]

    def test_observation_count_exceeds_window(self, monitor):
        """Total observation count should track beyond window size."""
        for _ in range(200):
            monitor.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
        assert report["observations"] == 200
        assert report["window_size"] == 100  # capped at window


# ── Persistence ──────────────────────────────────────────────────────────────


class _FakeRedisPipeline:
    """Minimal fake Redis pipeline recording lpush/ltrim/incr calls."""

    def __init__(self, store: dict):
        self._store = store
        self._ops: list = []

    def lpush(self, key, value):
        self._ops.append(("lpush", key, value))
        return self

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", key, start, end))
        return self

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def execute(self):
        for op in self._ops:
            if op[0] == "lpush":
                _, key, value = op
                self._store.setdefault(key, []).insert(0, value)
            elif op[0] == "ltrim":
                _, key, start, end = op
                self._store[key] = self._store.get(key, [])[start : end + 1]
            elif op[0] == "incr":
                _, key = op
                self._store[key] = str(int(self._store.get(key, "0")) + 1)
        self._ops = []


class _FakeRedis:
    """Minimal fake Redis client with just the methods DriftMonitor uses."""

    def __init__(self):
        self._store: dict = {}

    def ping(self):
        return True

    def pipeline(self):
        return _FakeRedisPipeline(self._store)

    def lrange(self, key, start, end):
        data = self._store.get(key, [])
        if end == -1:
            return data[start:]
        return data[start : end + 1]

    def get(self, key):
        return self._store.get(key)


class TestDriftMonitorRedisBackend:
    """Verify the Redis-backed path uses the shared list and aggregates
    across multiple monitor instances (simulating multi-replica pods)."""

    def test_redis_backend_is_selected_when_client_supplied(self, baseline_stats):
        fake = _FakeRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake)
        mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = mon.check_drift()
        assert report["backend"] == "redis"
        assert report["observations"] == 1

    def test_two_replicas_share_window(self, baseline_stats):
        """Two DriftMonitor instances pointed at the same fake Redis should
        aggregate their observations into a single cluster-wide window."""
        fake = _FakeRedis()  # shared backend
        pod_a = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake)
        pod_b = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake)

        for _ in range(20):
            pod_a.observe({"Age": 40.0, "Education_Ord": 2.0})
        for _ in range(20):
            pod_b.observe({"Age": 40.0, "Education_Ord": 2.0})

        # Either pod should see all 40 observations from the shared list.
        report_a = pod_a.check_drift()
        report_b = pod_b.check_drift()
        assert report_a["window_size"] == 40
        assert report_b["window_size"] == 40
        assert report_a["observations"] == 40
        assert report_b["observations"] == 40

    def test_redis_window_is_trimmed_to_cap(self, baseline_stats):
        fake = _FakeRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=50, redis_client=fake)
        for _ in range(200):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = mon.check_drift()
        assert report["window_size"] == 50  # trimmed
        assert report["observations"] == 200  # counter not trimmed

    def test_redis_drift_detection_still_works(self, baseline_stats):
        fake = _FakeRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake, alert_threshold=2.0)
        for _ in range(50):
            mon.observe({"Age": 70.0, "Education_Ord": 2.0})  # 3-sigma shift
        report = mon.check_drift()
        assert report["any_drifted"]
        assert report["features"]["Age"]["drifted"] is True


class TestBaselinePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        """save_baseline_stats() output should be loadable by DriftMonitor."""
        data = {"Age": [30.0, 40.0, 50.0], "Education_Ord": [1.0, 2.0, 3.0]}
        path = tmp_path / "baseline.json"
        save_baseline_stats(data, str(path))

        monitor = DriftMonitor.from_baseline(str(path))
        assert "Age" in monitor.baseline
        assert monitor.baseline["Age"]["mean"] == pytest.approx(40.0, abs=0.01)
        assert monitor.baseline["Age"]["std"] == pytest.approx(8.1650, abs=0.01)

    def test_save_creates_parent_directories(self, tmp_path):
        """save_baseline_stats should create missing parent dirs."""
        path = tmp_path / "nested" / "dir" / "baseline.json"
        save_baseline_stats({"Age": [1.0, 2.0]}, str(path))
        assert path.exists()
        with open(path) as f:
            stats = json.load(f)
        assert "Age" in stats

    def test_from_baseline_missing_file_raises(self, tmp_path):
        """Loading a nonexistent baseline should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DriftMonitor.from_baseline(str(tmp_path / "nonexistent.json"))
