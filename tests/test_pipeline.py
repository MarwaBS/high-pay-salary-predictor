"""
Unit tests for the High-Paying Jobs analysis pipeline.
Run: pytest tests/ -v
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Load project configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def df(cfg):
    """Load cleaned dataset once for all tests."""
    data_path = Path(__file__).parent.parent / cfg["data"]["cleaned"]
    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def df_engineered(df, cfg):
    """Return dataframe with engineered features."""
    edu_order = cfg["education_order"]
    region_map = {
        state: region
        for region, states in cfg["regions"].items()
        for state in states
    }
    out = df.copy()
    out["Education_Ord"] = out["Education Level"].map(edu_order)
    out["Gender_Bin"] = (out["Gender"] == "Male").astype(int)
    out["Region"] = out["State Abbreviation"].map(region_map)
    out["Occ_Mean_Income"] = out.groupby("Occupation")["Annual Income"].transform("mean")
    out["State_Mean_Income"] = out.groupby("State Abbreviation")["Annual Income"].transform("mean")
    return out


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
        # 100000 / 2080 hours ≈ 48.08
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

    def test_education_ordinal_range(self, df_engineered):
        assert df_engineered["Education_Ord"].between(1, 4).all()

    def test_gender_binary_values(self, df_engineered):
        assert set(df_engineered["Gender_Bin"].unique()).issubset({0, 1})

    def test_gender_binary_male_is_1(self, df_engineered):
        male_rows = df_engineered[df_engineered["Gender"] == "Male"]
        assert (male_rows["Gender_Bin"] == 1).all()

    def test_region_no_nulls(self, df_engineered):
        nulls = df_engineered["Region"].isnull().sum()
        assert nulls == 0, f"{nulls} states not mapped to a region"

    def test_region_valid_values(self, df_engineered):
        valid = {"Northeast", "Midwest", "South", "West"}
        assert set(df_engineered["Region"].unique()).issubset(valid)

    def test_occ_mean_income_positive(self, df_engineered):
        assert (df_engineered["Occ_Mean_Income"] > 0).all()

    def test_state_mean_income_positive(self, df_engineered):
        assert (df_engineered["State_Mean_Income"] > 0).all()

    def test_occ_mean_income_no_nulls(self, df_engineered):
        assert df_engineered["Occ_Mean_Income"].isnull().sum() == 0

    def test_state_mean_income_no_nulls(self, df_engineered):
        assert df_engineered["State_Mean_Income"].isnull().sum() == 0


# ── Model Prediction Tests ─────────────────────────────────────────────────────

class TestModelPrediction:
    FEATURES = [
        "Age",
        "Education_Ord",
        "Gender_Bin",
        "Employment",
        "Location Quotient",
        "Jobs per 1000",
        "Hourly Mean",
        "Annual Mean Wage",
        "Occ_Mean_Income",
        "State_Mean_Income",
    ]

    @pytest.fixture
    def trained_model(self, df_engineered):
        """Train a regularized model for prediction tests.

        Uses shallow trees and L2 regularization to reduce overfitting.
        Individual Census income within the $100K+ cohort has very high
        within-occupation variance, so a deep unregularized model memorizes
        training data but generalizes poorly.
        """
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        X = df_engineered[self.FEATURES]
        y = df_engineered["Annual Income"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_lambda=10.0,
            reg_alpha=1.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    def test_model_outputs_float(self, trained_model, df_engineered):
        row = df_engineered[self.FEATURES].iloc[[0]]
        pred = trained_model.predict(row)
        assert isinstance(pred[0], (float, np.floating))

    def test_prediction_above_zero(self, trained_model, df_engineered):
        X = df_engineered[self.FEATURES].head(50)
        preds = trained_model.predict(X)
        assert (preds > 0).all()

    def test_prediction_plausible_range(self, trained_model, df_engineered):
        X = df_engineered[self.FEATURES].head(200)
        preds = trained_model.predict(X)
        assert preds.min() > 10_000, "Predictions unrealistically low"
        assert preds.max() < 5_000_000, "Predictions unrealistically high"

    def test_r2_above_floor(self, trained_model, df_engineered):
        """Model should explain at least 8% of variance on the test set.

        Individual Census income within the $100K+ cohort has extremely high
        within-occupation variance ($100K to $1M+). The available features
        (BLS occupation wages, demographics) explain occupation-level means
        but not individual variation. Empirically, well-regularized models
        achieve R² ≈ 0.10-0.12 on this data. The 0.08 floor ensures the
        model is non-trivially better than predicting the mean (R²=0).
        """
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        X = df_engineered[self.FEATURES]
        y = df_engineered["Annual Income"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        r2 = r2_score(y_test, trained_model.predict(X_test))
        assert r2 > 0.08, f"R² too low: {r2:.4f}"

    def test_feature_count_matches(self, trained_model):
        assert trained_model.n_features_in_ == len(self.FEATURES)
