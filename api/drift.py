"""
Lightweight online drift detector.

Tracks incoming prediction features in a rolling window and compares
them against training-time baseline statistics using z-score deviation.
Alerts when feature distributions shift significantly from training data.

Usage:
    monitor = DriftMonitor.from_baseline("models/baseline_stats.json")
    monitor.observe({"Age": 42, "Education_Ord": 2, ...})
    report = monitor.check_drift()  # {"Age": 0.12, "Education_Ord": 2.3, ...}
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class DriftMonitor:
    """Rolling-window drift detector using z-score deviation from baseline."""

    def __init__(
        self,
        baseline_stats: dict[str, dict[str, float]],
        window: int = 500,
        alert_threshold: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        baseline_stats : per-feature statistics from training set
                         {feature: {mean, std, min, max}}
        window         : number of recent observations to keep
        alert_threshold: z-score above which a feature is flagged as drifted
        """
        self.baseline = baseline_stats
        self.window = window
        self.alert_threshold = alert_threshold
        self.buffer: deque[dict[str, float]] = deque(maxlen=window)
        self._observation_count = 0

    @classmethod
    def from_baseline(cls, path: str | Path, **kwargs) -> DriftMonitor:
        """Load baseline statistics from JSON."""
        with open(path) as f:
            stats = json.load(f)
        return cls(baseline_stats=stats, **kwargs)

    def observe(self, features: dict[str, float]) -> None:
        """Record a single observation (feature dict from one prediction)."""
        self.buffer.append(features)
        self._observation_count += 1

    def check_drift(self) -> dict:
        """Compare current window against baseline.

        Returns
        -------
        dict with keys:
            observations : total observations recorded
            window_size  : current buffer length
            features     : {feature: {z_score, current_mean, baseline_mean, drifted}}
            any_drifted  : True if any feature exceeds alert_threshold
        """
        result: dict[str, dict] = {}
        if len(self.buffer) < 30:
            return {
                "observations": self._observation_count,
                "window_size": len(self.buffer),
                "features": {},
                "any_drifted": False,
                "message": f"Need at least 30 observations (have {len(self.buffer)})",
            }

        for feat, stats in self.baseline.items():
            baseline_mean = stats["mean"]
            baseline_std = stats["std"]
            values = [obs.get(feat, 0.0) for obs in self.buffer]
            current_mean = float(np.mean(values))

            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
            else:
                z_score = 0.0

            result[feat] = {
                "z_score": round(z_score, 3),
                "current_mean": round(current_mean, 2),
                "baseline_mean": round(baseline_mean, 2),
                "drifted": z_score > self.alert_threshold,
            }

        return {
            "observations": self._observation_count,
            "window_size": len(self.buffer),
            "features": result,
            "any_drifted": any(v["drifted"] for v in result.values()),
        }


def save_baseline_stats(
    feature_data: dict[str, list[float]],
    path: str | Path,
) -> None:
    """Compute and save per-feature baseline statistics from training data.

    Call this in train_model.py after training to persist the baseline
    that the drift monitor compares against.
    """
    stats = {}
    for feat, values in feature_data.items():
        arr = np.array(values, dtype=float)
        stats[feat] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
