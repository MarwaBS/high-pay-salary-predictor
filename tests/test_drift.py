"""
Tests for the drift detection module (api/drift.py).
Verifies detection logic, edge cases, rolling window, and persistence.
Run: pytest tests/test_drift.py -v
"""

import json
import math
from pathlib import Path

import pytest

from api.drift import MIN_WINDOW_FOR_VERDICT, DriftMonitor, save_baseline_stats
from config_schema import ProjectConfig

CONFIGURED_WINDOW = ProjectConfig.from_yaml(Path(__file__).parent.parent / "config.yaml").drift.window


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
        """Observations centred on the baseline mean should not trigger drift.

        Uses a deterministic spread symmetric about each baseline mean so the
        sample mean equals the baseline mean exactly — the z-score is then 0
        regardless of the (now standard-error-based) sensitivity. Random draws
        from the baseline distribution would, correctly, trip the 2-SE bound
        ~5% of the time, which is expected statistics, not a bug."""
        for _ in range(25):
            monitor.observe({"Age": 30.0, "Education_Ord": 1.0})
            monitor.observe({"Age": 50.0, "Education_Ord": 3.0})  # symmetric → mean stays at baseline
        report = monitor.check_drift()
        assert not report["any_drifted"]
        assert report["features"]["Age"]["z_score"] == pytest.approx(0.0, abs=1e-6)

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

    def test_small_consistent_mean_shift_alerts_once_window_filled(self, baseline_stats):
        """A consistent sub-σ shift in the mean must alert once enough data has
        accumulated. Age→43 is only 0.3σ of the raw feature — a σ-scaled z-score
        (0.3) never crosses threshold, but the standard-error z-score flags it
        (≈4.7 SE at n=250). During ramp-up the detector is deliberately
        conservative (ramp-scaled effect floor + Šidák correction — an
        uncorrected z>2 cut leaves the ~10-feature union at ≈37%),
        so a 0.3σ shift is below the n=50 floor (2·√(2/50) ≈ 0.4σ) but well
        above the settled floor (0.2σ) once n ≥ 2·(z/d)² = 200."""
        mon = DriftMonitor(baseline_stats, window=300, alert_threshold=2.0)
        for _ in range(50):
            mon.observe({"Age": 43.0, "Education_Ord": 2.0})
        early = mon.check_drift()
        assert early["features"]["Age"]["drifted"] is False, "0.3σ is below the n=50 ramp floor (≈0.4σ)"
        for _ in range(200):
            mon.observe({"Age": 43.0, "Education_Ord": 2.0})
        report = mon.check_drift()
        assert report["features"]["Age"]["drifted"] is True
        assert report["features"]["Age"]["z_score"] > 2.0
        assert report["features"]["Education_Ord"]["drifted"] is False

    def test_absent_feature_does_not_manufacture_drift(self, monitor):
        """A feature missing from observations must not be imputed as 0.0 —
        imputing 0.0 drags its mean far from baseline and manufactures drift."""
        for _ in range(40):
            monitor.observe({"Age": 40.0})  # Education_Ord never observed
        report = monitor.check_drift()
        assert report["features"]["Education_Ord"]["n_observed"] == 0
        assert report["features"]["Education_Ord"]["drifted"] is False
        assert report["features"]["Education_Ord"]["current_mean"] is None
        assert report["any_drifted"] is False

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
        """Below the floor: no features scored, and the message names the shortfall."""
        for _ in range(10):
            monitor.observe({"Age": 70.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
        assert report["features"] == {}
        assert f"Need at least {MIN_WINDOW_FOR_VERDICT}" in report.get("message", "")
        assert report["any_drifted"] is None

    def test_exactly_at_the_floor_reports(self, monitor):
        """At the floor a verdict is issued — the gate is ``<``, not ``<=``."""
        for _ in range(MIN_WINDOW_FOR_VERDICT):
            monitor.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
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


# ── Sensitivity at the real operating window ─────────────────────────────────


class TestDriftSensitivityAtTheConfiguredWindow:
    """The effect-size floor must suppress benign wobble at the real window.

    With the SE z-score alone, the alarm fires whenever the window mean shifts by
    more than ``alert_threshold/sqrt(window)`` std — ~0.09 std at the configured
    500. Production traffic is never i.i.d. from the training baseline, so
    significance alone alarms on benign sampling wobble; ``min_effect_size`` gates
    it with a practical effect-size floor on TOP of significance.
    """

    BASELINE = {
        "Age": {"mean": 40.0, "std": 10.0, "min": 18.0, "max": 80.0},
        "Education_Ord": {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 4.0},
        "Region_Code": {"mean": 1.85, "std": 1.03, "min": 0.0, "max": 3.0},
        "Hourly_Mean": {"mean": 65.7, "std": 13.7, "min": 48.0, "max": 123.0},
    }

    def test_iid_from_baseline_at_the_configured_window_does_not_alarm(self):
        """Drawing a full window straight from the baseline distribution — i.e.
        NO real drift — must not alarm. Without the effect-size floor these four
        tests alarm on ≈4.5% of windows even with the Šidák correction in place,
        and on 1 - (1 - 0.0455)^4 ≈ 17% if the correction goes too."""
        import numpy as np

        rng = np.random.default_rng(20260617)
        false_alarms = 0
        trials = 25
        for _ in range(trials):
            mon = DriftMonitor(self.BASELINE, window=CONFIGURED_WINDOW, alert_threshold=2.0)
            for _ in range(CONFIGURED_WINDOW):
                mon.observe({f: float(rng.normal(s["mean"], s["std"])) for f, s in self.BASELINE.items()})
            if mon.check_drift()["any_drifted"]:
                false_alarms += 1
        # Allow a hair of slack for the rare tail, but it must be near-zero.
        assert false_alarms <= 1, f"{false_alarms}/{trials} i.i.d. windows alarmed"

    def test_genuine_consistent_shift_still_alarms_at_the_configured_window(self):
        """A consistent >= min_effect_size shift in a feature mean must still be
        caught at the real window — the fix must not make the monitor deaf."""
        import numpy as np

        rng = np.random.default_rng(7)
        mon = DriftMonitor(self.BASELINE, window=CONFIGURED_WINDOW, alert_threshold=2.0)
        for _ in range(CONFIGURED_WINDOW):
            obs = {f: float(rng.normal(s["mean"], s["std"])) for f, s in self.BASELINE.items()}
            obs["Age"] = 40.0 + 0.5 * 10.0  # consistent +0.5 std shift on Age
            mon.observe(obs)
        report = mon.check_drift()
        assert report["any_drifted"] is True
        assert report["features"]["Age"]["drifted"] is True
        assert report["features"]["Age"]["effect_size"] >= 0.2

    def test_significant_but_trivial_effect_does_not_alarm(self):
        """The precise failure mode: a statistically significant (SE z > 2) but
        practically trivial mean shift must NOT alarm at the real window."""
        mon = DriftMonitor(self.BASELINE, window=CONFIGURED_WINDOW, alert_threshold=2.0)
        # Halfway between what significance alone detects and the effect floor,
        # so the shift is significant and trivial at whatever window is configured
        # — a fixed 0.1 would stop being significant below n=401 and fail there
        # for a reason that has nothing to do with what this test checks.
        significant_from = mon.alert_threshold / math.sqrt(CONFIGURED_WINDOW)
        effect = (significant_from + mon.min_effect_size) / 2
        assert significant_from < effect < mon.min_effect_size, "no trivial-yet-significant shift exists here"
        for _ in range(CONFIGURED_WINDOW):
            mon.observe({"Age": self.BASELINE["Age"]["mean"] + effect * self.BASELINE["Age"]["std"]})
        report = mon.check_drift()
        age = report["features"]["Age"]
        assert age["z_score"] > mon.alert_threshold, "shift should be statistically significant"
        assert age["effect_size"] == pytest.approx(effect, abs=1e-3), "reported effect is rounded to 3dp"
        assert age["drifted"] is False, "trivial effect must not alarm"


# ── Ramp-up behaviour (window still filling) ─────────────────────────────────


class TestDriftRampUpFalseAlarms:
    """The familywise false-alarm rate must stay bounded while the window fills.

    ``any_drifted`` is the union of ~10 per-feature tests. An UNcorrected
    per-feature cut at z > 2 (per-test α ≈ 4.55%) leaves the union ungated, and
    below n = (z/d)² = 100 a fixed 0.2σ effect floor is implied by significance
    alone — so up to that fill a perfectly stationary window false-alarms with
    probability 1 - (1 - 0.0455)^10 ≈ 37%, and less once the floor starts to bind.
    The monitor therefore Šidák-corrects the per-feature α across the k tested
    features AND ramp-scales the effect floor (max(0.2, z·√(2/n))), bounding the
    familywise false-alarm rate at ≈ erfc(2/√2) ≈ 4.6% at ANY window fill —
    without touching the configured operating point (see TestDriftSensitivityAtTheConfiguredWindow).
    """

    # Ten features, mirroring the width of the production baseline — the
    # familywise failure mode only shows at realistic k.
    BASELINE = {
        "Age": {"mean": 40.0, "std": 10.0, "min": 18.0, "max": 80.0},
        "Education_Ord": {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 4.0},
        "Gender_Bin": {"mean": 0.42, "std": 0.49, "min": 0.0, "max": 1.0},
        "Region_Code": {"mean": 1.85, "std": 1.03, "min": 0.0, "max": 3.0},
        "Employment": {"mean": 30000.0, "std": 60000.0, "min": 40.0, "max": 600000.0},
        "Location Quotient": {"mean": 1.0, "std": 0.6, "min": 0.01, "max": 8.0},
        "Jobs per 1000": {"mean": 4.0, "std": 6.0, "min": 0.01, "max": 90.0},
        "Hourly Mean": {"mean": 65.7, "std": 13.7, "min": 48.0, "max": 123.0},
        "Occ_Mean_Income": {"mean": 136000.0, "std": 25000.0, "min": 100000.0, "max": 230000.0},
        "State_Mean_Income": {"mean": 136000.0, "std": 9000.0, "min": 115000.0, "max": 160000.0},
    }

    def _familywise_false_alarm_rate(self, n_obs: int, trials: int, seed: int) -> float:
        import numpy as np

        rng = np.random.default_rng(seed)
        false_alarms = 0
        for _ in range(trials):
            mon = DriftMonitor(self.BASELINE, window=500)  # shipped tuning, not a fixed threshold
            for _ in range(n_obs):
                mon.observe({f: float(rng.normal(s["mean"], s["std"])) for f, s in self.BASELINE.items()})
            if mon.check_drift()["any_drifted"]:
                false_alarms += 1
        return false_alarms / trials

    #: Familywise level the detector is allowed to design for. A drift alarm is
    #: a page: at the conventional 5% one stationary window in twenty wakes
    #: someone, and looser than that the endpoint is noise.
    DESIGN_CEILING = 0.05

    def _bound(self, trials: int) -> float:
        """Two binomial standard deviations above the level the shipped tuning
        designs for — a bound on the measurement, not a second design choice."""
        designed = math.erfc(DriftMonitor(self.BASELINE, window=1).alert_threshold / math.sqrt(2.0))
        assert designed <= self.DESIGN_CEILING, f"the shipped alert_threshold designs for {designed:.1%}"
        return designed + 2 * math.sqrt(designed * (1 - designed) / trials)

    def test_stationary_at_the_floor_familywise_false_alarm_rate_bounded(self):
        """At the reporting floor, i.i.d.-from-baseline windows (NO real drift)
        must false-alarm at ≲ the designed familywise level."""
        rate = self._familywise_false_alarm_rate(n_obs=MIN_WINDOW_FOR_VERDICT, trials=150, seed=20260704)
        assert rate <= self._bound(150), f"familywise FA rate {rate:.1%} at the floor exceeds {self._bound(150):.1%}"

    def test_stationary_n100_familywise_false_alarm_rate_bounded(self):
        """Same bound at n=100 — the stress point where the fixed 0.2σ floor
        exactly coincides with the uncorrected z>2 bound, so the effect floor
        adds no protection and only the Šidák α-correction bounds the union."""
        rate = self._familywise_false_alarm_rate(n_obs=100, trials=150, seed=20260705)
        assert rate <= self._bound(150), f"familywise FA rate {rate:.1%} at n=100 exceeds {self._bound(150):.1%}"

    def test_mid_window_real_drift_still_fires(self):
        """Deaf-check: the ramp-up conservatism must NOT silence real drift
        mid-fill. Age +5 years (0.5σ) at n=150 — floor(150) ≈ 0.23σ, expected
        per-trial power ≈ 99.9% — must fire on every one of 25 trials."""
        import numpy as np

        rng = np.random.default_rng(20260706)
        detections = 0
        trials = 25
        for _ in range(trials):
            mon = DriftMonitor(self.BASELINE, window=500)
            for _ in range(150):
                obs = {f: float(rng.normal(s["mean"], s["std"])) for f, s in self.BASELINE.items()}
                obs["Age"] += 5.0  # +0.5 baseline std
                mon.observe(obs)
            report = mon.check_drift()
            if report["any_drifted"] and report["features"]["Age"]["drifted"]:
                detections += 1
        assert detections == trials, f"only {detections}/{trials} mid-window drift trials fired"


class TestDriftMechanismIsolation:
    """Each false-alarm control is load-bearing ON ITS OWN, not just as a pair.

    The Šidák per-feature correction and the ramp-scaled effect floor are
    REDUNDANT at the production width (k≈10): there, clearing the ramp floor
    already implies z > 2√2 ≈ 2.83, which is ~the Šidák per-feature cut, so a
    stationary-traffic false-alarm test cannot tell them apart — deleting either
    one alone leaves the familywise rate at ≈5% and every ramp-up test still
    green. That is a maintenance trap: someone could silently drop the Šidák
    correction (which the docstrings call load-bearing) and CI would stay green.

    These two tests break the k=10 coincidence so each mechanism is isolated —
    removing just that one mechanism flips the single assertion to red. They are
    fully deterministic (constant observations, zero sample variance), so the
    boundary values are exact, not statistical.
    """

    def test_sidak_correction_is_applied_not_just_familywise_alpha(self):
        """With many features, a per-feature shift that is significant at the
        UNcorrected familywise α but NOT at the Šidák-corrected per-feature α —
        while clearing the effect floor — must NOT alarm.

        k=100 features, n=200 (effect floor at its fixed 0.2σ handover). One
        feature is shifted to z=3.3 (p≈9.7e-4): that is far below α_family
        (0.0455) so an uncorrected detector fires, but ABOVE the Šidák per-feature
        cut α_k≈4.7e-4, so the corrected detector stays silent. Its effect size
        (0.233σ) clears the 0.2σ floor, so ONLY the Šidák correction is what
        holds the alarm — delete it and this feature drifts. All other features
        sit exactly on the baseline mean (z=0)."""
        k, n, z_target = 100, 200, 3.3
        value = z_target / (n**0.5)  # std=1 → this constant gives mean-shift z=z_target
        baseline = {f"f{i:03d}": {"mean": 0.0, "std": 1.0, "min": -9.0, "max": 9.0} for i in range(k)}
        mon = DriftMonitor(baseline, window=500, alert_threshold=2.0, min_effect_size=0.2)
        for _ in range(n):
            obs = {f: 0.0 for f in baseline}
            obs["f000"] = value
            mon.observe(obs)
        report = mon.check_drift()
        target = report["features"]["f000"]
        # Pin the counterfactual: it IS significant at the familywise level and
        # DOES clear the fixed floor, so the ONLY thing keeping it silent is the
        # Šidák tightening of the per-feature α across k tests.
        assert target["p_value"] < math.erfc(2.0 / math.sqrt(2.0)), "must be significant at α_family"
        assert target["effect_size"] > 0.2, "must clear the fixed effect floor"
        assert report["any_drifted"] is False, "Šidák correction must suppress this familywise-only signal"

    def test_ramp_scaled_floor_binds_above_the_fixed_floor(self):
        """A mid-fill shift that is statistically significant AND above the FIXED
        0.2σ floor, but below the RAMP-scaled floor, must NOT alarm.

        Single feature (k=1, so the Šidák term is a no-op and cannot be what
        gates), n=50: ramp floor = max(0.2, 2·√(2/50)) = 0.4σ. A constant +0.3σ
        shift is significant (z=0.3·√50≈2.12 > 2) and clears the fixed 0.2σ floor,
        but 0.3 < 0.4, so the ramp-scaled floor is the ONLY thing holding the
        alarm — drop the ramp scaling (fixed 0.2σ only) and this drifts."""
        baseline = {"f": {"mean": 0.0, "std": 1.0, "min": -9.0, "max": 9.0}}
        mon = DriftMonitor(baseline, window=500, alert_threshold=2.0, min_effect_size=0.2)
        for _ in range(50):
            mon.observe({"f": 0.3})
        report = mon.check_drift()
        feat = report["features"]["f"]
        assert feat["z_score"] > 2.0, "must be statistically significant"
        assert feat["effect_size"] > 0.2, "must clear the FIXED 0.2σ floor"
        assert feat["drifted"] is False, "ramp-scaled floor must suppress a sub-floor mid-fill shift"


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


class _ReadFailingRedis(_FakeRedis):
    """Writes succeed (shared list populates) but reads raise — models a Redis
    partition where the authoritative window cannot be loaded."""

    def lrange(self, key, start, end):
        raise ConnectionError("simulated redis read failure")


class _WriteFailingRedisPipeline(_FakeRedisPipeline):
    def execute(self):
        raise ConnectionError("simulated redis write failure")


class _WriteFailingRedis(_FakeRedis):
    def pipeline(self):
        return _WriteFailingRedisPipeline(self._store)


class _WriteFailingReadsWorkingRedis(_FakeRedis):
    """Rejects writes while still serving reads — a Redis at ``maxmemory`` or a
    replica promoted read-only. The stored window keeps loading cleanly while
    live observations are discarded."""

    def __init__(self):
        super().__init__()
        self.writes_ok = True

    def pipeline(self):
        if self.writes_ok:
            return super().pipeline()
        return _WriteFailingRedisPipeline(self._store)


class TestDriftBackendFailureIsLoud:
    """A configured Redis backend that fails must never yield a clean bill of
    health from a window it could not read, nor mislabel which path served it."""

    def test_read_failure_reports_unavailable_not_clean(self, baseline_stats):
        fake = _ReadFailingRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=200, redis_client=fake)
        # 200 extreme-drift observations land in Redis; the local deque stays empty.
        for _ in range(200):
            mon.observe({"Age": 999.0, "Education_Ord": 2.0})
        report = mon.check_drift()
        # The read fell back to the empty local deque: the verdict must be
        # withheld, not a confident "no drift".
        assert report["degraded"] is True
        assert report["status"] == "unavailable"
        assert report["backend"] == "memory", "must name the path that actually served the read"
        assert report["any_drifted"] is None, "never a clean False from an unloadable window"
        assert report["features"] == {}

    def test_healthy_redis_read_is_not_degraded(self, baseline_stats):
        fake = _FakeRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake)
        for _ in range(40):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = mon.check_drift()
        assert report["backend"] == "redis"
        assert report["degraded"] is False
        assert report["any_drifted"] is False

    def test_write_failure_does_not_populate_local_deque(self, baseline_stats):
        fake = _WriteFailingRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=100, redis_client=fake)
        for _ in range(50):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})  # every write fails
        # Dropped, not mixed into the local plane — deque stays empty.
        assert len(mon.buffer) == 0

    def test_dropped_writes_withhold_the_verdict(self, baseline_stats):
        """Observations lost to failed writes make the window unrepresentative.

        Reads still succeed here, so the stored window looks healthy while live
        traffic is being discarded — the verdict must be withheld rather than
        reported clean from a window the dropped traffic never reached.
        """
        fake = _WriteFailingReadsWorkingRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=500, redis_client=fake)

        for _ in range(50):  # healthy traffic lands in the shared window
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        assert mon.check_drift()["degraded"] is False, "healthy writes must not degrade"

        fake.writes_ok = False
        for _ in range(200):  # heavily drifted traffic is dropped
            mon.observe({"Age": 90.0, "Education_Ord": 2.0})

        report = mon.check_drift()
        assert report["degraded"] is True
        assert report["status"] == "unavailable"
        assert report["any_drifted"] is None, "never a clean False while observations are being dropped"
        assert report["dropped_observations"] == 200

    def test_verdict_resumes_only_after_the_lost_traffic_is_replaced(self, baseline_stats):
        """Recovery is proportional to what was lost, not to the first success.

        A single landed observation does not make a window that is missing many
        of them representative again.
        """
        fake = _WriteFailingReadsWorkingRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=500, redis_client=fake)

        fake.writes_ok = False
        for _ in range(10):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        assert mon.check_drift()["degraded"] is True

        fake.writes_ok = True
        mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        partial = mon.check_drift()
        assert partial["degraded"] is True, "one landed write must not clear a ten-observation gap"
        assert partial["dropped_observations"] == 9

        for _ in range(40):
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})

        report = mon.check_drift()
        assert report["degraded"] is False
        assert report["any_drifted"] is False
        assert report["dropped_observations"] == 0

    def test_zero_window_is_refused(self, baseline_stats):
        """A zero window would cap the backlog at zero and never withhold.

        The dropped-write guard is bounded by the window size, so a window of 0
        turns it into a permanent all-clear — the exact failure it exists to
        prevent.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            DriftMonitor(baseline_stats=baseline_stats, window=0)

    def test_backlog_never_exceeds_the_window(self, baseline_stats):
        """A long outage must not withhold the verdict longer than the window.

        The shared list is trimmed to ``window``, so once that many fresh
        observations have landed none of the pre-outage data remains and there
        is nothing further to wait for. Without the cap the monitor stays dark
        for as many requests as it dropped, however long the outage ran.
        """
        window = 50
        fake = _WriteFailingReadsWorkingRedis()
        mon = DriftMonitor(baseline_stats=baseline_stats, window=window, redis_client=fake)

        fake.writes_ok = False
        for _ in range(10_000):  # a long outage
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})
        assert mon.check_drift()["dropped_observations"] == window

        fake.writes_ok = True
        for _ in range(window):  # exactly one full window of fresh traffic
            mon.observe({"Age": 40.0, "Education_Ord": 2.0})

        report = mon.check_drift()
        assert report["dropped_observations"] == 0
        assert report["degraded"] is False


class TestBaselinePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        """save_baseline_stats() output should be loadable by DriftMonitor."""
        data = {"Age": [30.0, 40.0, 50.0], "Education_Ord": [1.0, 2.0, 3.0]}
        path = tmp_path / "baseline.json"
        save_baseline_stats(data, str(path))

        monitor = DriftMonitor.from_baseline(str(path), window=100)
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
            DriftMonitor.from_baseline(str(tmp_path / "nonexistent.json"), window=100)


class TestVerdictWithheldWhenNothingTestable:
    """A window that could not be tested must not read as a clean pass.

    Both branches return ``any_drifted: None``; asserting ``not any_drifted``
    would pass on ``False`` too, so these assert identity.
    """

    def test_one_short_of_the_floor_withholds_the_verdict(self, monitor):
        for _ in range(MIN_WINDOW_FOR_VERDICT - 1):
            monitor.observe({"Age": 40.0, "Education_Ord": 2.0})
        report = monitor.check_drift()
        assert report["any_drifted"] is None
        assert report["degraded"] is False

    def test_no_observed_feature_matching_the_baseline_withholds_the_verdict(self, monitor):
        # A renamed feature is the realistic trigger: the window fills, but no
        # key matches, so every feature would otherwise score n_observed=0.
        for _ in range(40):
            monitor.observe({"Age_renamed": 40.0, "Education_Ord_renamed": 2.0})
        report = monitor.check_drift()
        assert report["any_drifted"] is None
        assert report["degraded"] is False


# ── Configured window vs the ramp-scaled effect floor ────────────────────────


class TestConfiguredWindowClearsTheEffectFloorHandover:
    """``config.yaml::drift.window`` must put normal operation past the ramp.

    Below the handover the ramp term rules and the advertised
    ``min_effect_size`` sensitivity is unreachable — a shift just over the
    advertised floor goes unreported. The probe is exactly that limiting shift,
    so the window at which it stops being masked IS the handover; a bigger probe
    would clear the ramp early and pass at windows the bound forbids.
    """

    BASELINE = {"Age": {"mean": 40.0, "std": 10.0, "min": 18.0, "max": 80.0}}

    def _monitor(self, window: int, **tuning: float) -> DriftMonitor:
        """Built the way ``api.main`` builds it: tuning comes from the defaults."""
        return DriftMonitor(self.BASELINE, window=window, **tuning)

    @property
    def handover(self) -> int:
        return self._monitor(1).effect_floor_handover()

    def _verdict_at(self, window: int, **tuning: float) -> bool:
        mon = self._monitor(window, **tuning)
        handover = mon.effect_floor_handover()
        # Halfway between the advertised floor and the ramp one observation short
        # of the handover — the only band that is masked below the handover and
        # reported at it, whatever the tuning makes those two values.
        ramp_below = mon.alert_threshold * math.sqrt(2.0 / (handover - 1))
        probe = (mon.min_effect_size + ramp_below) / 2
        for _ in range(window):
            mon.observe({"Age": self.BASELINE["Age"]["mean"] + probe * self.BASELINE["Age"]["std"]})
        return bool(mon.check_drift()["any_drifted"])

    def test_the_advertised_sensitivity_is_reached_at_the_handover(self):
        assert self._verdict_at(self.handover) is True

    def test_one_observation_short_of_the_handover_still_masks_it(self):
        """Pins where the ramp stops binding, so the bound below is a real line."""
        assert self._verdict_at(self.handover - 1) is False

    def test_the_configured_window_is_not_below_the_handover(self):
        assert CONFIGURED_WINDOW >= self.handover, (
            f"config.yaml::drift.window={CONFIGURED_WINDOW} cannot reach the advertised "
            f"sensitivity; the ramp binds until {self.handover}"
        )

    def test_the_configured_window_reports_that_shift(self):
        assert self._verdict_at(CONFIGURED_WINDOW) is True

    def test_the_handover_rounds_up_to_a_window_that_works(self):
        """A tuning whose handover is fractional (2*(2.0/0.3)**2 = 88.9). Rounding
        down would name a window at which the advertised shift is still masked."""
        tuning = {"min_effect_size": 0.3}
        handover = self._monitor(1, **tuning).effect_floor_handover()
        assert handover == 89, "the handover must round up, not to the nearest integer"
        assert self._verdict_at(handover - 1, **tuning) is False
        assert self._verdict_at(handover, **tuning) is True


class TestVerdictFloorStaysInTheNormalApproximation:
    def test_the_floor_is_not_lowered_below_the_clt_rule_of_thumb(self):
        """A minimum, not an equality: raising it only withholds more verdicts,
        while lowering it rules on windows too small for the normal tail."""
        assert MIN_WINDOW_FOR_VERDICT >= 30
