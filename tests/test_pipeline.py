"""
Unit tests for the High-Paying Jobs analysis pipeline.

Covers: config validation, raw-data schema, feature engineering, and
model predictions. Fixtures are provided by tests/conftest.py.

Run: pytest tests/ -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import FEATURES_FULL, REGION_CODES, engineer_features


def _require(condition: bool, reason: str) -> None:
    """Skip locally when an artefact is absent, but FAIL under CI.

    In CI the committed artefacts are always present, so a skip there would be
    a silent green — the metric-band gates would certify nothing on a deleted
    or wrong-format metrics file. Locally, a skip keeps the suite runnable
    before the first ``python -m scripts.train_quantile``.
    """
    if condition:
        return
    if os.getenv("CI"):
        pytest.fail(reason)
    pytest.skip(reason)


# ── Config Tests ──────────────────────────────────────────────────────────────


class TestConfig:
    def test_config_loads(self, cfg):
        assert cfg is not None

    def test_required_keys(self, cfg):
        for key in ("data", "thresholds", "model", "education_order", "regions"):
            assert key in cfg, f"Missing config key: {key}"

    def test_income_threshold(self, cfg):
        assert cfg["thresholds"]["min_annual_income"] == 100_000

    def test_hourly_threshold(self, cfg):
        expected = 100_000 / 2_080
        assert abs(cfg["thresholds"]["min_hourly_mean"] - expected) < 0.01

    def test_education_order_is_ordinal(self, cfg):
        values = list(cfg["education_order"].values())
        assert values == sorted(values), "Education order must be strictly ascending"

    def test_all_50_states_covered(self, cfg):
        all_states = [s for states in cfg["regions"].values() for s in states]
        assert len(all_states) == 50, f"Expected 50 states, got {len(all_states)}"

    def test_model_split_valid(self, cfg):
        assert 0 < cfg["model"]["test_size"] < 1

    def test_region_no_overlap(self, cfg):
        all_states = [s for states in cfg["regions"].values() for s in states]
        assert len(all_states) == len(set(all_states)), "Duplicate states found in regions"


# ── Data Schema Tests ─────────────────────────────────────────────────────────


class TestDataSchema:
    REQUIRED_COLUMNS = [
        "State Abbreviation",
        "State",
        "Gender",
        "Age",
        "Education Code",
        "Education Level",
        "Degree Field",
        "Occupation Code",
        "Occupation",
        "Annual Income",
        "Employment",
        "Location Quotient",
        "Jobs per 1000",
        "Hourly Mean",
        "Annual Mean Wage",
    ]

    def test_columns_present(self, df):
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self, df):
        missing = df.isnull().sum()
        assert missing.sum() == 0, f"Unexpected NaNs:\n{missing[missing > 0]}"

    def test_row_count_reasonable(self, df):
        assert 5_000 <= len(df) <= 50_000, f"Unexpected row count: {len(df)}"

    def test_50_states_present(self, df):
        n_states = df["State Abbreviation"].nunique()
        assert n_states == 50, f"Expected 50 states, found {n_states}"

    def test_income_floor(self, df):
        assert df["Annual Income"].min() >= 100_000, "Income below $100K threshold found"

    def test_income_no_negatives(self, df):
        assert (df["Annual Income"] > 0).all()

    def test_age_range(self, df):
        assert df["Age"].between(16, 100).all(), "Age values outside plausible range"

    def test_gender_values(self, df):
        assert set(df["Gender"].unique()).issubset({"Male", "Female"})

    def test_education_levels(self, df, cfg):
        expected = set(cfg["education_order"].keys())
        actual = set(df["Education Level"].unique())
        assert actual.issubset(expected), f"Unexpected education levels: {actual - expected}"

    def test_location_quotient_positive(self, df):
        assert (df["Location Quotient"] > 0).all()

    def test_employment_positive(self, df):
        assert (df["Employment"] > 0).all()

    def test_annual_mean_wage_positive(self, df):
        assert (df["Annual Mean Wage"] > 0).all()


# ── Feature Engineering Tests ─────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_education_ordinal_no_nulls(self, df_engineered):
        assert df_engineered["Education_Ord"].isnull().sum() == 0

    def test_education_ordinal_range(self, df_engineered, cfg):
        lo, hi = min(cfg["education_order"].values()), max(cfg["education_order"].values())
        assert df_engineered["Education_Ord"].between(lo, hi).all()

    def test_gender_binary_values(self, df_engineered):
        assert set(df_engineered["Gender_Bin"].unique()).issubset({0, 1})

    def test_gender_binary_male_is_1(self, df_engineered):
        male_rows = df_engineered[df_engineered["Gender"] == "Male"]
        assert (male_rows["Gender_Bin"] == 1).all()

    def test_region_no_nulls(self, df_engineered):
        nulls = df_engineered["Region"].isnull().sum()
        assert nulls == 0, f"{nulls} states not mapped to a region"

    def test_region_valid_values(self, df_engineered):
        assert set(df_engineered["Region"].unique()).issubset(set(REGION_CODES.keys()))

    def test_region_code_valid_values(self, df_engineered):
        assert set(df_engineered["Region_Code"].unique()).issubset(set(REGION_CODES.values()))

    def test_region_code_no_nulls(self, df_engineered):
        assert df_engineered["Region_Code"].isnull().sum() == 0

    def test_occ_mean_income_positive(self, df_engineered):
        assert (df_engineered["Occ_Mean_Income"] > 0).all()

    def test_state_mean_income_positive(self, df_engineered):
        assert (df_engineered["State_Mean_Income"] > 0).all()

    def test_occ_mean_income_no_nulls(self, df_engineered):
        assert df_engineered["Occ_Mean_Income"].isnull().sum() == 0

    def test_state_mean_income_no_nulls(self, df_engineered):
        assert df_engineered["State_Mean_Income"].isnull().sum() == 0

    def test_features_full_all_present(self, df_engineered):
        """All columns in FEATURES_FULL must exist after engineering."""
        for col in FEATURES_FULL:
            assert col in df_engineered.columns, f"Missing engineered column: {col}"


# ── Pipeline constants ────────────────────────────────────────────────────────


class TestPipelineConstants:
    def test_region_codes_cover_four_regions(self):
        assert set(REGION_CODES.keys()) == {"Midwest", "Northeast", "South", "West"}

    def test_region_codes_unique_integers(self):
        vals = list(REGION_CODES.values())
        assert len(vals) == len(set(vals)), "REGION_CODES values must be unique"

    def test_features_full_length(self):
        assert len(FEATURES_FULL) == 10, f"Expected 10 features, got {len(FEATURES_FULL)}"

    def test_features_full_has_region_code(self):
        assert "Region_Code" in FEATURES_FULL


# ── Model Prediction Tests ─────────────────────────────────────────────────────


class TestModelPrediction:
    """Tests against the production model loaded from disk.

    The model is trained by scripts/train_quantile.py (run via 'make model').
    The artefacts are committed, so a fresh checkout — and CI — always has them.
    Testing the production model (rather than re-training a toy one) catches
    hyperparameter regressions and artefact-format changes.
    """

    def test_model_outputs_quantile_triple(self, production_model, df_engineered):
        """Multi-quantile XGBoost emits (n, 3) — P10, P50, P90 per row."""
        from pipeline import is_quantile_model, predict_quantiles

        assert is_quantile_model(production_model), (
            "production model must be multi-quantile (refused otherwise at startup)"
        )
        row = df_engineered[FEATURES_FULL].iloc[[0]]
        p10, p50, p90 = predict_quantiles(production_model, row)
        assert isinstance(p50, float)
        assert p10 <= p50 <= p90

    def test_prediction_above_zero(self, production_model, df_engineered):
        X = df_engineered[FEATURES_FULL].head(50)
        preds = production_model.predict(X)
        assert np.asarray(preds).min() > 0

    def test_prediction_plausible_range(self, production_model, df_engineered):
        """Back-transformed P50 predictions must be in a plausible dollar range."""
        from pipeline import predict_quantiles

        X = df_engineered[FEATURES_FULL].head(200)
        p50_arr = np.asarray([predict_quantiles(production_model, X.iloc[[i]])[1] for i in range(len(X))])
        assert p50_arr.min() > 10_000, "Predictions unrealistically low"
        assert p50_arr.max() < 5_000_000, "Predictions unrealistically high"

    def test_predict_quantiles_batch_refuses_non_triple_output(self):
        """A non-(n, 3) model output must raise, not silently collapse
        to a degenerate (p, p, p) interval. Single source of truth for both the
        single-row and batch prediction paths."""
        from pipeline import predict_quantiles, predict_quantiles_batch

        class _PointModel:
            def predict(self, rows):
                return np.zeros(len(rows))  # 1-D point output

        row = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            predict_quantiles_batch(_PointModel(), row)
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            predict_quantiles(_PointModel(), row)

    def test_conformal_delta_widens_interval_and_preserves_p50(self, production_model, df_engineered):
        """The conformal margin widens P10/P90 symmetrically in log space
        (so the dollar interval grows) while leaving the P50 point untouched."""
        from pipeline import predict_quantiles_batch

        X = df_engineered[FEATURES_FULL].head(100)
        raw = predict_quantiles_batch(production_model, X, conformal_delta=0.0)
        conf = predict_quantiles_batch(production_model, X, conformal_delta=0.02)
        assert np.allclose(raw[:, 1], conf[:, 1]), "P50 must not move under conformal widening"
        assert (conf[:, 0] <= raw[:, 0]).all() and (conf[:, 2] >= raw[:, 2]).all()
        assert np.median(conf[:, 2] - conf[:, 0]) > np.median(raw[:, 2] - raw[:, 0])

    def test_load_conformal_delta_raises_on_missing(self, tmp_path):
        """A configured-but-absent margin is a deploy error: the served interval
        claims a coverage that only holds with the margin, so loading must fail
        loud rather than silently fall back to the under-covering raw interval."""
        from pipeline import load_conformal_delta

        with pytest.raises(FileNotFoundError):
            load_conformal_delta(str(tmp_path / "does_not_exist.json"))

    def test_saved_metrics_within_expected_range(self, cfg):
        """Saved model metrics must fall inside explicit regression windows.

        Reads the frozen ``model_metrics.json`` and checks both the P50
        point metrics AND the quantile-specific metrics (coverage,
        crossings). Point-estimate bands are intentionally wide because
        P50 under a quantile objective is the median-minimiser, not the
        mean-minimiser, so R² is a weak fit-statistic for this model —
        the real SLO is the quantile coverage and crossings band below.
        """
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / cfg["model"]["metrics_path"]
        _require(metrics_path.exists(), "model_metrics.json not found — run scripts/train_quantile.py first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        r2 = metrics["r2"]
        mae = metrics["mae"]
        rmse = metrics["rmse"]

        # Point-estimate bands: wide — see docstring.
        assert 0.00 <= r2 <= 0.40, f"P50 R² {r2:.4f} outside expected band [0.00, 0.40]"
        assert 30_000 <= mae <= 90_000, f"P50 MAE ${mae:,.0f} outside expected band"
        assert 60_000 <= rmse <= 160_000, f"P50 RMSE ${rmse:,.0f} outside expected band"

        # Quantile-specific guards (skip gracefully for legacy point models).
        if "quantile_coverage_80" in metrics:
            coverage = metrics["quantile_coverage_80"]
            crossings = metrics.get("quantile_crossings", 0)
            # 80% PI should empirically cover ~80% of test targets (band [0.72, 0.88]).
            assert 0.72 <= coverage <= 0.88, (
                f"Quantile 80% coverage {coverage:.3f} outside [0.72, 0.88] — quantile calibration has drifted"
            )
            assert crossings == 0, (
                f"{crossings} quantile crossings detected — P10>P50 or P50>P90. Check model training."
            )

        # Cross-conformal calibration: the served (conformalized) interval must
        # land near its nominal target and beat the raw interval's coverage.
        if "conformal_coverage_80" in metrics:
            raw_cov = metrics["quantile_coverage_80"]
            conf_cov = metrics["conformal_coverage_80"]
            target = metrics["conformal_target_coverage"]
            assert metrics["conformal_delta"] > 0, "conformal margin must be positive to widen the interval"
            assert abs(conf_cov - target) <= 0.03, (
                f"Conformalized coverage {conf_cov:.3f} not within 0.03 of target {target:.2f}"
            )
            assert conf_cov >= raw_cov, (
                f"Conformalized coverage {conf_cov:.3f} should not under-cover the raw interval {raw_cov:.3f}"
            )

    def test_saved_cv_matches_test(self, cfg):
        """CV R² and Test R² must agree within ~0.15.

        Both metrics are computed in dollar space on train-only folds
        (CV) and the held-out test split (Test), so they should be
        close. A spurious gap would indicate CV leaked test rows or was
        computed in a different transformed space from the test metric.

        Metrics files that lack the ``cv_space`` flag are skipped with a
        clear retrain message.
        """
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / cfg["model"]["metrics_path"]
        _require(metrics_path.exists(), "model_metrics.json not found — run scripts/train_quantile.py first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        _require(
            metrics.get("cv_space") == "dollar",
            "model_metrics.json predates the dollar-space CV change (no cv_space flag). "
            "Re-run `python -m scripts.train_quantile` to regenerate metrics with train-only, dollar-space CV.",
        )

        gap = abs(metrics["cv_r2_mean"] - metrics["r2"])
        assert gap <= 0.15, (
            f"CV/Test R² mismatch too large ({gap:.4f}). "
            f"cv_r2_mean={metrics['cv_r2_mean']:.4f} vs r2={metrics['r2']:.4f}."
        )

    def test_subgroup_coverage_within_band(self, cfg):
        """Every per-gender / per-region subgroup must stay within a
        calibration band around the cohort-wide target of 0.80.

        The floor at 0.60 is generous but catches a catastrophic
        subgroup collapse — e.g. the female cohort dropping from
        ~0.77 to 0.50 — that would indicate the quantile model has
        stopped being calibrated for that population.
        """
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / cfg["model"]["metrics_path"]
        _require(metrics_path.exists(), "model_metrics.json not found — run scripts/train_quantile.py first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        subgroup_coverage = metrics.get("subgroup_coverage_80")
        _require(
            bool(subgroup_coverage),
            "model_metrics.json predates the subgroup_coverage_80 field. "
            "Re-run `python -m scripts.train_quantile` to regenerate metrics.",
        )

        bad = {k: v for k, v in subgroup_coverage.items() if not (0.60 <= v <= 0.95)}
        assert not bad, (
            f"Subgroup coverage outside [0.60, 0.95]: {bad}. "
            f"Quantile model calibration has drifted for these subgroups."
        )

    def test_feature_count_matches(self, production_model):
        assert production_model.n_features_in_ == len(FEATURES_FULL)

    def test_metric_gates_fail_not_skip_under_ci(self, monkeypatch):
        """A missing/wrong-format metrics file must FAIL under CI, not
        skip — a skip there would let the metric-band gates certify nothing."""
        monkeypatch.setenv("CI", "1")
        with pytest.raises(pytest.fail.Exception):
            _require(False, "missing metrics")

    def test_metric_gates_skip_locally(self, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        with pytest.raises(pytest.skip.Exception):
            _require(False, "missing metrics")


# ── Feature-engineering guards ──────────────────────────────────────────────────


class TestEngineerFeaturesGuards:
    """engineer_features must fail loud on unmapped categoricals rather than
    encode them as a silent NaN (education/gender) or a Region_Code-0 collision
    (state) — a config typo must surface, not ship a quietly-degraded model."""

    EDU = {"Bachelor's degree": 1, "Master's degree": 2}
    REGION = {"CA": "West", "NY": "Northeast"}

    def _frame(self, **overrides) -> pd.DataFrame:
        base = {
            "Education Level": "Bachelor's degree",
            "Gender": "Male",
            "State Abbreviation": "CA",
            "Occupation": "Engineer",
            "Annual Income": 150_000.0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_clean_frame_encodes(self):
        out = engineer_features(self._frame(), self.EDU, self.REGION)
        assert out["Education_Ord"].iloc[0] == 1
        assert out["Gender_Bin"].iloc[0] == 1
        assert out["Region_Code"].iloc[0] == REGION_CODES["West"]

    def test_region_absent_from_region_codes_raises_naming_it(self):
        # Renaming a region in config.yaml otherwise dies later as an opaque
        # IntCastingNaNError from the Region_Code cast.
        with pytest.raises(ValueError) as exc:
            engineer_features(self._frame(), self.EDU, {"CA": "Pacific", "NY": "Northeast"})
        assert "Pacific" in str(exc.value) and "REGION_CODES" in str(exc.value)

    def test_unmapped_education_raises_naming_the_label(self):
        with pytest.raises(ValueError) as exc:
            engineer_features(self._frame(**{"Education Level": "Some College"}), self.EDU, self.REGION)
        assert "Education Level" in str(exc.value) and "Some College" in str(exc.value)

    def test_unknown_gender_raises(self):
        # Wrong case must not silently fold into the Female bucket.
        with pytest.raises(ValueError, match="Gender"):
            engineer_features(self._frame(Gender="male"), self.EDU, self.REGION)

    def test_unmapped_state_raises(self):
        with pytest.raises(ValueError, match="State Abbreviation"):
            engineer_features(self._frame(**{"State Abbreviation": "ZZ"}), self.EDU, self.REGION)


def test_compute_fallback_means_raises_on_empty():
    """Averaging an empty group-means dict yields NaN, which the API
    would inject as the fallback feature for every unseen occupation/state.
    Fail loud instead."""
    from pipeline import compute_fallback_means

    with pytest.raises(ValueError, match="empty"):
        compute_fallback_means({"occ_means": {}, "state_means": {"CA": 1.0}})
    occ, state = compute_fallback_means({"occ_means": {"a": 100.0, "b": 200.0}, "state_means": {"CA": 150.0}})
    assert occ == 150.0 and state == 150.0


# ── Config Schema Validation ─────────────────────────────────────────────────


class TestArtefactWritersRequestLf:
    """Content-addressed artefacts must be byte-identical across platforms.

    Their SHA-256 digests are recorded at training time and re-verified at
    startup and in CI, so a writer left on the platform default would produce a
    different digest on Windows than on Linux for identical content.

    Asserting on the output bytes would only catch this on Windows: with
    ``newline=None`` Python translates to ``os.linesep``, which on the Linux CI
    runner already is ``\\n``. So the check is on the call itself — every
    artefact writer must open its file with an explicit ``newline="\\n"``,
    which fails identically on every platform.
    """

    def test_all_json_artefact_writers_request_lf(self, tmp_path, monkeypatch):
        from api.drift import save_baseline_stats
        from pipeline import save_conformal, save_features, save_group_means, save_metrics

        real_open = open
        newline_by_path: dict[str, object] = {}

        def recording_open(file, mode="r", *args, **kwargs):
            if "w" in str(mode):
                newline_by_path[str(file)] = kwargs.get("newline", "<default>")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", recording_open)

        writers = {
            "features": (save_features, (["Age", "Education_Ord"], str(tmp_path / "features.json")), {}),
            "metrics": (save_metrics, ({"r2": 0.5, "nested": {"a": 1}}, str(tmp_path / "metrics.json")), {}),
            "group_means": (
                save_group_means,
                ({"occ_means": {"Dev": 1.0}, "state_means": {"CA": 2.0}}, str(tmp_path / "gm.json")),
                {},
            ),
            "conformal": (
                save_conformal,
                (0.01, str(tmp_path / "conformal.json")),
                {"target_coverage": 0.8, "n_scores": 100},
            ),
            "baseline_stats": (
                save_baseline_stats,
                ({"Age": [30.0, 40.0, 50.0]}, str(tmp_path / "baseline.json")),
                {},
            ),
        }
        for name, (writer, args, kwargs) in writers.items():
            writer(*args, **kwargs)
            requested = newline_by_path.get(str(args[1]))
            assert requested == "\n", (
                f"{name} writer opened its artefact with newline={requested!r}; "
                "on a CRLF platform its recorded digest would not reproduce"
            )
            assert b"\r\n" not in Path(args[1]).read_bytes()


class TestConfigSchema:
    """Verify that config_schema.py catches invalid configurations."""

    def test_valid_config_passes(self, cfg):
        """The production config.yaml should pass Pydantic validation."""
        from config_schema import ProjectConfig

        config = ProjectConfig(**cfg)
        assert config.thresholds.min_annual_income == 100_000
        assert len(config.education_order) == 4

    def test_unknown_model_key_rejected(self, cfg):
        """A mistyped optional knob must fail loudly, not fall back to a default.

        Optional settings are the dangerous case: ``stabilty_seeds`` silently
        ignored leaves the trainer using a value nobody configured.
        """
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        broken["model"]["stabilty_seeds"] = [1, 2]
        with pytest.raises(ValidationError):
            ProjectConfig(**broken)

    def test_classifier_path_without_hyperparameters_rejected(self, cfg):
        """A configured classifier needs every setting the trainer relies on."""
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        del broken["model"]["classifier_n_estimators"]
        with pytest.raises(ValidationError, match="classifier_n_estimators"):
            ProjectConfig(**broken)

    def test_missing_section_raises(self, cfg):
        """Missing a required top-level key should fail validation."""
        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = {k: v for k, v in cfg.items() if k != "thresholds"}
        with pytest.raises(ValidationError):
            ProjectConfig(**broken)

    def test_missing_state_raises(self, cfg):
        """49 states with no repeat, so only the count check can raise."""
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        broken["regions"]["West"].remove("WY")
        with pytest.raises(ValidationError, match="exactly 50 states, got 49"):
            ProjectConfig(**broken)

    def test_duplicate_state_raises(self, cfg):
        """A repeated state with the total still at 50, so only the duplicate check can raise."""
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        broken["regions"]["West"].remove("WY")
        broken["regions"]["Midwest"].append("CA")  # CA is already in West
        with pytest.raises(ValidationError, match="duplicate states in regions"):
            ProjectConfig(**broken)

    def test_non_ordinal_education_raises(self, cfg):
        """Education values that aren't 1..N should fail validation."""
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        broken["education_order"] = {"Bachelor's degree": 1, "Master's degree": 5}
        with pytest.raises(ValidationError):
            ProjectConfig(**broken)
