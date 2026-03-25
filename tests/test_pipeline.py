"""
Unit tests for the High-Paying Jobs analysis pipeline.

Covers: config validation, raw-data schema, feature engineering, and
model predictions. Fixtures are provided by tests/conftest.py.

Run: pytest tests/ -v
"""

import numpy as np

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

    def test_model_outputs_float(self, production_model, df_engineered):
        row = df_engineered[FEATURES_FULL].iloc[[0]]
        pred = production_model.predict(row)
        assert isinstance(pred[0], (float, np.floating))

    def test_prediction_above_zero(self, production_model, df_engineered):
        X = df_engineered[FEATURES_FULL].head(50)
        preds = production_model.predict(X)
        assert (preds > 0).all()

    def test_prediction_plausible_range(self, production_model, df_engineered):
        # Model predicts log1p(income); back-transform with expm1 for dollar check
        X = df_engineered[FEATURES_FULL].head(200)
        preds = np.expm1(production_model.predict(X))
        assert preds.min() > 10_000, "Predictions unrealistically low"
        assert preds.max() < 5_000_000, "Predictions unrealistically high"

    def test_r2_above_floor(self, production_model, df_engineered):
        """Production model should explain at least 5% of variance on the test set.

        Model is trained on log1p(Annual Income); predictions are back-transformed
        with expm1 before computing R². With log transform + Optuna HPO + fixed
        target encoding, the production model achieves R² ≈ 0.077 on this dataset.
        The 0.05 floor guards against hyperparameter regressions while staying
        safely below the expected value.
        """
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        X = df_engineered[FEATURES_FULL]
        y = df_engineered["Annual Income"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = np.expm1(production_model.predict(X_test))
        r2 = r2_score(y_test, preds)
        assert r2 > 0.05, f"R² too low: {r2:.4f}"

    def test_feature_count_matches(self, production_model):
        assert production_model.n_features_in_ == len(FEATURES_FULL)
