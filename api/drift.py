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
- **In-memory (fallback)**: if ``REDIS_URL`` is unset, or Redis is
  unreachable when the monitor is constructed, observations are stored in
  a process-local deque. Drift in this mode is per-replica and therefore
  only meaningful when replicas=1.

Once Redis is the selected backend it stays selected: observations lost to a
later failure are dropped rather than diverted to the deque, and ``check_drift``
withholds its verdict until as many fresh ones have landed.

The backend is selected automatically by ``DriftMonitor.__init__`` based on
whether a Redis client is provided / available.

Usage
-----
    monitor = DriftMonitor.from_baseline("models/baseline_stats.json")
    monitor.observe({"Age": 42, "Education_Ord": 2, ...})
    report = monitor.check_drift()  # {"features": {...}, "any_drifted": bool | None}
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Redis list key holding the rolling observation window. Every replica
#: writes to and reads from the same key.
REDIS_DRIFT_KEY = "drift:observations"

#: Smallest window ``check_drift`` will rule on. Its p-values come off the
#: normal tail, and every observation carries every baseline feature (``api``
#: imputes the BLS defaults before observing), so the window length is also the
#: per-feature sample size behind that tail. 30 is the conventional floor for
#: the approximation; a shorter window would report clean verdicts from one.
MIN_WINDOW_FOR_VERDICT = 30


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
        window: int,
        alert_threshold: float = 2.0,
        min_effect_size: float = 0.2,
        redis_client: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        baseline_stats : per-feature statistics from training set
                         {feature: {mean, std, min, max}}
        window         : number of recent observations to keep
        alert_threshold: base standard-error z-score setting the FAMILYWISE
                         significance level: alpha_family =
                         erfc(alert_threshold/sqrt(2)) (two-sided normal tail;
                         default 2.0 -> ~4.6%). Around 10 features are tested
                         per report, so the per-feature cut is Sidak-adjusted
                         from this familywise alpha (see ``check_drift``).
                         Testing each feature at the raw cut instead lets the
                         union of k tests alarm at 1-(1-alpha)^k — an order of
                         magnitude above the headline rate.
        min_effect_size: minimum |mean shift| in baseline-std units required ON
                         TOP OF statistical significance. Significance alone
                         scales with n — at window=500 a ~0.09-std wobble
                         clears it — so never-quite-i.i.d. production traffic
                         would alarm forever. Requiring a practical effect too
                         (Cohen's d small = 0.2) keeps an i.i.d. window silent
                         while a consistent >=0.2-std shift still fires. The
                         fixed floor is vacuous while the window is filling, so
                         the effective floor is ramp-scaled to
                         max(min_effect_size, alert_threshold*sqrt(2/n)).
                         See ``check_drift``.
        redis_client   : optional Redis client. If provided (or discoverable
                         from REDIS_URL), observations are stored in a
                         shared list so multi-replica Deployments aggregate
                         correctly.
        """
        if window < 1:
            # Would cap the dropped-write backlog at zero: a permanent all-clear.
            raise ValueError(f"window must be >= 1, got {window}")
        self.baseline = baseline_stats
        self.window = window
        self.alert_threshold = alert_threshold
        self.min_effect_size = min_effect_size
        self.buffer: deque[dict[str, float]] = deque(maxlen=window)
        self._observation_count = 0
        #: Backlog of observations lost to failed Redis writes, per process.
        #: Non-zero means the shared window is still missing live traffic.
        self._dropped_writes = 0
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
                # One landed observation replaces one lost, so the verdict
                # resumes only once the window has been made whole.
                self._dropped_writes = max(0, self._dropped_writes - 1)
                return
            except Exception as exc:
                # Dropped, not diverted to the deque: on a Redis deployment
                # reads come from the shared list, so a partial local window
                # would only mislead a later read failure. Capped at the window
                # size, past which no pre-outage data can remain.
                self._dropped_writes = min(self._dropped_writes + 1, self.window)
                logger.warning("DriftMonitor Redis write failed (%s) — observation dropped", exc)
                return

        self.buffer.append(features)
        self._observation_count += 1

    # ------------------------------------------------------------------ #
    # Read side
    # ------------------------------------------------------------------ #

    def _read_window(self) -> tuple[list[dict[str, float]], int, str, bool]:
        """Return (observations, total_count, backend_used, degraded).

        ``backend_used`` names the path that actually served this read, not the
        one configured. ``degraded`` is True only when a configured Redis
        backend was unreachable: the shared window lives in Redis, so the local
        deque is empty on a Redis deployment and the returned window is NOT
        authoritative — the caller must withhold any drift verdict.
        """
        if self._redis is not None:
            try:
                raw = self._redis.lrange(REDIS_DRIFT_KEY, 0, -1)
                observations = [json.loads(item) for item in raw]
                count_raw = self._redis.get(f"{REDIS_DRIFT_KEY}:count") or "0"
                return observations, int(count_raw), "redis", False
            except Exception as exc:
                logger.warning("DriftMonitor Redis read failed (%s) — window unavailable", exc)
                return list(self.buffer), self._observation_count, "memory", True

        return list(self.buffer), self._observation_count, "memory", False

    def check_drift(self) -> dict:
        """Compare current window against baseline.

        Returns
        -------
        dict with keys:
            observations : total observations recorded (cluster-wide in Redis mode)
            window_size  : current buffer length
            backend      : "redis" | "memory" — the path that actually served
                           this read, not merely the configured backend
            degraded     : True if a configured Redis window could not be loaded,
                           or observations were dropped before reaching it
            dropped_observations : outstanding backlog of dropped observations,
                           capped at ``window`` — how many more must land before
                           the verdict resumes, not how many were lost
            features     : {feature: {z_score, effect_size, p_value, current_mean,
                            baseline_mean, n_observed, drifted}}
            any_drifted  : True if any feature is BOTH statistically significant
                           (Sidak-corrected across all tested features) AND above
                           the (ramp-scaled) practical effect-size floor. ``None``
                           whenever no verdict can be reached — a degraded
                           window, fewer than ``MIN_WINDOW_FOR_VERDICT``
                           observations, or no observed feature matching the
                           baseline. Never a clean False from a window that
                           could not be tested; ``message`` says which.
        """
        observations, total_count, backend, degraded = self._read_window()
        dropped = self._dropped_writes

        if degraded or dropped:
            # Unreadable or incomplete, so ``any_drifted: False`` would be a
            # clean bill of health the data does not support. Dropped writes are
            # the subtler case: reads still succeed and the window looks fine.
            reason = (
                "Redis backend unreachable"
                if degraded
                else f"{dropped} more observation(s) must land to replace ones dropped by failed Redis writes"
            )
            return {
                "observations": total_count,
                "window_size": len(observations),
                "backend": backend,
                "degraded": True,
                "status": "unavailable",
                "features": {},
                "any_drifted": None,
                "dropped_observations": dropped,
                "message": f"Drift window unavailable: {reason} — verdict withheld",
            }

        if len(observations) < MIN_WINDOW_FOR_VERDICT:
            return {
                "observations": total_count,
                "window_size": len(observations),
                "backend": backend,
                "degraded": False,
                "features": {},
                "any_drifted": None,
                "dropped_observations": 0,
                "message": f"Need at least {MIN_WINDOW_FOR_VERDICT} observations (have {len(observations)})",
            }

        # Collect per-feature samples FIRST so the number of tests actually
        # performed (k) is known before any drift decision is made — the
        # familywise correction below needs it.
        feature_values = {
            # Count only observations that carry the feature; folding in absent
            # features as 0.0 would drag the mean toward zero and manufacture
            # phantom drift.
            feat: [obs[feat] for obs in observations if feat in obs]
            for feat in self.baseline
        }
        n_tested = sum(1 for vals in feature_values.values() if vals)

        # Nothing testable: a renamed or absent feature set would otherwise score
        # every feature n_observed=0, drifted=False and report a clean pass.
        if not n_tested:
            return {
                "observations": total_count,
                "window_size": len(observations),
                "backend": backend,
                "degraded": False,
                "features": {},
                "any_drifted": None,
                "dropped_observations": 0,
                "message": "No observed feature matches a baseline feature — verdict withheld",
            }

        # ── Familywise error control (Sidak) ──────────────────────────────
        # ``any_drifted`` is the union of k per-feature tests (~10 in
        # production). At a per-feature two-sided cut of z > 2 (alpha_1 =
        # erfc(2/sqrt 2) ~= 4.55%) the familywise false-alarm probability on a
        # perfectly stationary window is 1 - (1 - 0.0455)^10 ~= 37%, and during
        # ramp-up nothing else gates the decision (see the effect-floor note
        # below). Sidak inverts that union bound exactly for independent tests:
        # testing each feature at
        #     alpha_k = 1 - (1 - alpha_family)^(1/k)
        # gives P(any false alarm) = alpha_family regardless of k. Features
        # here are only weakly correlated, so alpha_family is a tight upper
        # bound. ``alert_threshold`` keeps its z-score interface but now sets
        # the FAMILYWISE level: alpha_family = erfc(threshold/sqrt 2) (~4.6%
        # at the default 2.0). Decisions compare two-sided p-values
        # (erfc(z/sqrt 2)) against alpha_k — equivalent to raising the
        # per-feature z cut to ~2.8 at k=10, without needing an inverse-CDF.
        alpha_family = math.erfc(self.alert_threshold / math.sqrt(2.0))
        alpha_per_feature = 1.0 - (1.0 - alpha_family) ** (1.0 / n_tested)

        result: dict[str, dict] = {}
        for feat, stats in self.baseline.items():
            baseline_mean = stats["mean"]
            baseline_std = stats["std"]
            values = feature_values[feat]
            n = len(values)

            if n == 0:
                # Feature never observed in this window — can't assess drift.
                result[feat] = {
                    "z_score": 0.0,
                    "effect_size": 0.0,
                    "p_value": 1.0,
                    "current_mean": None,
                    "baseline_mean": round(baseline_mean, 2),
                    "n_observed": 0,
                    "drifted": False,
                }
                continue

            current_mean = float(np.mean(values))

            # Test whether the *mean* of n observations differs from the
            # baseline mean: the correct yardstick is the standard error of
            # the mean (σ/√n), not the population σ. Dividing by the population
            # σ alone would require a full multi-σ shift in the raw feature to
            # alert — statistically near-deaf. With the
            # standard error, ``alert_threshold`` is in standard-error units
            # (default 2.0 ≈ a 95% confidence bound on the mean).
            if baseline_std > 0:
                standard_error = baseline_std / np.sqrt(n)
                z_score = float(abs(current_mean - baseline_mean) / standard_error)
                effect_size = float(abs(current_mean - baseline_mean) / baseline_std)
            else:
                z_score = 0.0
                effect_size = 0.0

            # Drift requires BOTH a statistically real mean shift (Sidak-
            # corrected p-value, see above) AND a practically meaningful one
            # (effect size). The significance test alone makes the alarm
            # n-sensitive — at window=500 a ~0.09-std wobble clears it — so
            # always-noisy production traffic alarms forever. The effect-size
            # floor is the practical-significance gate.
            #
            # ── Ramp-scaled effect floor ──────────────────────────────────
            # The fixed floor d = min_effect_size only binds once it exceeds
            # the shift implied by significance alone (z_crit/sqrt(n)), i.e.
            # for n > (z_crit/d)^2 — with z_crit ~ 2 and d = 0.2 that is
            # n > 100. Below that, "significant" implies "above the floor",
            # the floor adds nothing, and every feature runs at its full
            # per-test alpha — the ~37% union-bound regime above. Scaling the floor as
            #     floor_n = max(d, alert_threshold * sqrt(2/n))
            # keeps the practical gate a factor sqrt(2) ABOVE the base
            # significance bound in z-units (2.0 -> 2.83 SE; per-feature tail
            # 0.47% vs 4.55%) for the whole ramp-up, decays as 1/sqrt(n), and
            # hands over to the fixed Cohen's-d floor continuously at
            # n = 2*(alert_threshold/d)^2 = 200 (defaults) — so behaviour at
            # the full window=500 operating point is unchanged.
            p_value = math.erfc(z_score / math.sqrt(2.0))
            effect_floor = max(self.min_effect_size, self.alert_threshold * math.sqrt(2.0 / n))
            drifted = bool(p_value < alpha_per_feature and effect_size > effect_floor)

            result[feat] = {
                "z_score": round(z_score, 3),
                "effect_size": round(effect_size, 3),
                "p_value": round(p_value, 6),
                "current_mean": round(current_mean, 2),
                "baseline_mean": round(baseline_mean, 2),
                "n_observed": n,
                "drifted": drifted,
            }

        return {
            "observations": total_count,
            "window_size": len(observations),
            "backend": backend,
            "degraded": False,
            "features": result,
            "any_drifted": any(v["drifted"] for v in result.values()),
            "dropped_observations": 0,
        }


def save_baseline_stats(
    feature_data: dict[str, list[float]],
    path: str | Path,
) -> None:
    """Compute and save per-feature baseline statistics from training data.

    Call this from ``scripts/train_quantile.py`` after training to
    persist the baseline that the drift monitor compares against.
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
    with open(path, "w", newline="\n") as f:
        json.dump(stats, f, indent=2)
