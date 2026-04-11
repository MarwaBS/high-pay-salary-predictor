"""
Lightweight online drift detector.

Tracks incoming prediction features in a rolling window and compares them
against training-time baseline statistics using z-score deviation. Alerts
when feature distributions shift significantly from training data.

Storage backends
----------------
- **Redis (preferred for multi-replica deployments)**: observations are
  pushed onto a shared Redis list and trimmed to the window size. Every
  replica reads and writes the same window, so `/drift` returns a single
  cluster-wide view regardless of which pod handled a given prediction.
- **In-memory (fallback)**: if Redis is unreachable or ``REDIS_URL`` is
  unset, observations are stored in a process-local deque. Drift in this
  mode is per-replica and therefore only meaningful when replicas=1.

The backend is selected automatically by ``DriftMonitor.__init__`` based on
whether a Redis client is provided / available.

Usage
-----
    monitor = DriftMonitor.from_baseline("models/baseline_stats.json")
    monitor.observe({"Age": 42, "Education_Ord": 2, ...})
    report = monitor.check_drift()  # {"features": {...}, "any_drifted": bool}
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Redis list key holding the rolling observation window. Every replica
#: writes to and reads from the same key.
REDIS_DRIFT_KEY = "drift:observations"


class DriftMonitor:
    """Rolling-window drift detector using z-score deviation from baseline.

    Backend selection
    -----------------
    If a redis client is supplied (or ``REDIS_URL`` resolves at init time)
    the monitor stores observations in a shared Redis list. Otherwise it
    falls back to an in-process deque — correct for single-replica
    deployments, non-deterministic for multi-replica.
    """

    def __init__(
        self,
        baseline_stats: dict[str, dict[str, float]],
        window: int = 500,
        alert_threshold: float = 2.0,
        redis_client: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        baseline_stats : per-feature statistics from training set
                         {feature: {mean, std, min, max}}
        window         : number of recent observations to keep
        alert_threshold: z-score above which a feature is flagged as drifted
        redis_client   : optional Redis client. If provided (or discoverable
                         from REDIS_URL), observations are stored in a
                         shared list so multi-replica Deployments aggregate
                         correctly.
        """
        self.baseline = baseline_stats
        self.window = window
        self.alert_threshold = alert_threshold
        self.buffer: deque[dict[str, float]] = deque(maxlen=window)
        self._observation_count = 0
        self._redis = redis_client or self._discover_redis()
        if self._redis is not None:
            logger.info("DriftMonitor using Redis-backed shared window (key=%s)", REDIS_DRIFT_KEY)
        else:
            logger.info("DriftMonitor using in-process deque (single-replica mode)")

    # ------------------------------------------------------------------ #
    # Backend discovery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _discover_redis() -> Any | None:
        """Try to create a Redis client from ``REDIS_URL``. Returns None on
        any failure (missing env var, missing redis dep, connection error)."""
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return None
        try:
            import redis  # type: ignore

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:
            logger.warning("DriftMonitor: Redis unreachable at %s (%s) — falling back to in-memory", redis_url, exc)
            return None

    @classmethod
    def from_baseline(cls, path: str | Path, **kwargs: Any) -> DriftMonitor:
        """Load baseline statistics from JSON."""
        with open(path) as f:
            stats = json.load(f)
        return cls(baseline_stats=stats, **kwargs)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def observe(self, features: dict[str, float]) -> None:
        """Record a single observation (feature dict from one prediction)."""
        if self._redis is not None:
            try:
                payload = json.dumps(features, default=float)
                # Atomic LPUSH + LTRIM to keep the shared list capped at window.
                pipe = self._redis.pipeline()
                pipe.lpush(REDIS_DRIFT_KEY, payload)
                pipe.ltrim(REDIS_DRIFT_KEY, 0, self.window - 1)
                # Observation counter (monotonic across all replicas).
                pipe.incr(f"{REDIS_DRIFT_KEY}:count")
                pipe.execute()
                return
            except Exception as exc:
                logger.warning("DriftMonitor Redis write failed (%s) — falling back to in-memory for this request", exc)
                # Fall through to in-memory path on transient Redis errors.

        self.buffer.append(features)
        self._observation_count += 1

    # ------------------------------------------------------------------ #
    # Read side
    # ------------------------------------------------------------------ #

    def _read_window(self) -> tuple[list[dict[str, float]], int]:
        """Return (observations, total_count) from whichever backend is active."""
        if self._redis is not None:
            try:
                raw = self._redis.lrange(REDIS_DRIFT_KEY, 0, -1)
                observations = [json.loads(item) for item in raw]
                count_raw = self._redis.get(f"{REDIS_DRIFT_KEY}:count") or "0"
                return observations, int(count_raw)
            except Exception as exc:
                logger.warning("DriftMonitor Redis read failed (%s) — returning in-memory window", exc)

        return list(self.buffer), self._observation_count

    def check_drift(self) -> dict:
        """Compare current window against baseline.

        Returns
        -------
        dict with keys:
            observations : total observations recorded (cluster-wide in Redis mode)
            window_size  : current buffer length
            backend      : "redis" | "memory"
            features     : {feature: {z_score, current_mean, baseline_mean, drifted}}
            any_drifted  : True if any feature exceeds alert_threshold
        """
        observations, total_count = self._read_window()
        backend = "redis" if self._redis is not None else "memory"

        if len(observations) < 30:
            return {
                "observations": total_count,
                "window_size": len(observations),
                "backend": backend,
                "features": {},
                "any_drifted": False,
                "message": f"Need at least 30 observations (have {len(observations)})",
            }

        result: dict[str, dict] = {}
        for feat, stats in self.baseline.items():
            baseline_mean = stats["mean"]
            baseline_std = stats["std"]
            values = [obs.get(feat, 0.0) for obs in observations]
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
            "observations": total_count,
            "window_size": len(observations),
            "backend": backend,
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
