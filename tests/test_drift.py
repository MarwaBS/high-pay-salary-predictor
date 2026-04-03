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
