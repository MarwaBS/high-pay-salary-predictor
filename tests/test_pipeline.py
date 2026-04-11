"""
Unit tests for the High-Paying Jobs analysis pipeline.

Covers: config validation, raw-data schema, feature engineering, and
model predictions. Fixtures are provided by tests/conftest.py.

Run: pytest tests/ -v
"""

import json

import numpy as np
import pytest

from pipeline import FEATURES_FULL, REGION_CODES

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

    The model is trained by scripts/train_model.py (run via 'make model').
    CI runs that step before pytest so the artefact is always present.
    Testing the production model (rather than re-training a toy one) catches
    hyperparameter regressions and artefact-format changes.
    """

    def test_model_outputs_quantile_triple(self, production_model, df_engineered):
        """Multi-quantile XGBoost emits (n, 3) — P10, P50, P90 per row."""
        from pipeline import is_quantile_model, predict_quantiles

        row = df_engineered[FEATURES_FULL].iloc[[0]]
        if is_quantile_model(production_model):
            p10, p50, p90 = predict_quantiles(production_model, row)
            assert isinstance(p50, float)
            assert p10 <= p50 <= p90
        else:
            # Legacy point model — shape (n,)
            pred = production_model.predict(row)
            assert isinstance(pred[0], (float, np.floating))

    def test_prediction_above_zero(self, production_model, df_engineered):
        X = df_engineered[FEATURES_FULL].head(50)
        preds = production_model.predict(X)
        # For quantile model (n,3), all cells must be positive. For point (n,), all entries.
        assert np.asarray(preds).min() > 0

    def test_prediction_plausible_range(self, production_model, df_engineered):
        """Back-transformed P50 predictions must be in a plausible dollar range."""
        from pipeline import is_quantile_model, predict_quantiles

        X = df_engineered[FEATURES_FULL].head(200)
        if is_quantile_model(production_model):
            p50s = [predict_quantiles(production_model, X.iloc[[i]])[1] for i in range(len(X))]
            p50_arr = np.asarray(p50s)
        else:
            p50_arr = np.expm1(production_model.predict(X))

        assert p50_arr.min() > 10_000, "Predictions unrealistically low"
        assert p50_arr.max() < 5_000_000, "Predictions unrealistically high"

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
        if not metrics_path.exists():
            pytest.skip("model_metrics.json not found — run scripts/train_quantile.py first")

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
            # 80% PI should empirically cover ~80% of test targets (±5%).
            assert 0.72 <= coverage <= 0.88, (
                f"Quantile 80% coverage {coverage:.3f} outside [0.72, 0.88] — quantile calibration has drifted"
            )
            assert crossings == 0, (
                f"{crossings} quantile crossings detected — P10>P50 or P50>P90. Check model training."
            )

    def test_saved_cv_matches_test(self, cfg):
        """CV R² and Test R² must agree within ~0.15.

        Both metrics are computed in dollar space on train-only folds
        (CV) and the held-out test split (Test), so they should be
        close. A spurious gap would indicate CV leaked test rows or was
        computed in a different transformed space from the test metric.

        Metrics files written before the dollar-space CV change do not
        carry the ``cv_space`` flag and are skipped with a clear retrain
        message.
        """
        from pathlib import Path

        metrics_path = Path(__file__).parent.parent / cfg["model"]["metrics_path"]
        if not metrics_path.exists():
            pytest.skip("model_metrics.json not found — run scripts/train_quantile.py first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        if metrics.get("cv_space") != "dollar":
            pytest.skip(
                "model_metrics.json predates the dollar-space CV change "
                "(no cv_space flag). Re-run `python -m scripts.train_quantile` "
                "to regenerate metrics with train-only, dollar-space CV."
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
        if not metrics_path.exists():
            pytest.skip("model_metrics.json not found — run scripts/train_quantile.py first")

        with open(metrics_path) as f:
            metrics = json.load(f)

        subgroup_coverage = metrics.get("subgroup_coverage_80")
        if not subgroup_coverage:
            pytest.skip(
                "model_metrics.json predates the subgroup_coverage_80 field. "
                "Re-run `python -m scripts.train_quantile` to regenerate metrics."
            )

        bad = {k: v for k, v in subgroup_coverage.items() if not (0.60 <= v <= 0.95)}
        assert not bad, (
            f"Subgroup coverage outside [0.60, 0.95]: {bad}. "
            f"Quantile model calibration has drifted for these subgroups."
        )

    def test_feature_count_matches(self, production_model):
        assert production_model.n_features_in_ == len(FEATURES_FULL)


# ── Config Schema Validation ─────────────────────────────────────────────────


class TestConfigSchema:
    """Verify that config_schema.py catches invalid configurations."""

    def test_valid_config_passes(self, cfg):
        """The production config.yaml should pass Pydantic validation."""
        from config_schema import ProjectConfig

        config = ProjectConfig(**cfg)
        assert config.thresholds.min_annual_income == 100_000
        assert len(config.education_order) == 4

    def test_missing_section_raises(self, cfg):
        """Missing a required top-level key should fail validation."""
        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = {k: v for k, v in cfg.items() if k != "thresholds"}
        with pytest.raises(ValidationError):
            ProjectConfig(**broken)

    def test_duplicate_state_raises(self, cfg):
        """A state appearing in two regions should fail the 50-state check."""
        import copy

        from pydantic import ValidationError

        from config_schema import ProjectConfig

        broken = copy.deepcopy(cfg)
        broken["regions"]["West"].append("CA")  # CA already in West — now 51 entries
        with pytest.raises(ValidationError):
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
