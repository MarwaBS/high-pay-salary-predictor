"""
Integration tests — full pipeline path.

Each test exercises a complete data flow rather than a single unit:
  raw CSV → split → compute_group_means → engineer_features → train → predict

These guard against regressions where each unit test passes in isolation
but the composed pipeline breaks (e.g. feature-name mismatch, wrong
encoding, log/dollar-scale confusion).

Run: pytest tests/test_integration.py -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pipeline import (
    FEATURES_FULL,
    build_feature_row,
    compute_group_means,
    engineer_features,
    load_group_means,
    save_group_means,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_split(df_raw: pd.DataFrame, edu_order: dict, region_map: dict):
    """Split raw data, compute group means from train, engineer both splits."""
    train_raw, test_raw = train_test_split(df_raw, test_size=0.2, random_state=42)
    gm = compute_group_means(train_raw)
    df_train = engineer_features(train_raw, edu_order, region_map,
                                 occ_means=gm["occ_means"],
                                 state_means=gm["state_means"])
    df_test  = engineer_features(test_raw,  edu_order, region_map,
                                 occ_means=gm["occ_means"],
                                 state_means=gm["state_means"])
    return df_train, df_test, gm


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSplitThenEngineer:
    """Group means must be derived from train split only."""

    def test_train_test_sizes(self, df, edu_order, region_map):
        df_train, df_test, _ = _make_split(df, edu_order, region_map)
        n = len(df)
        assert abs(len(df_train) / n - 0.8) < 0.02
        assert abs(len(df_test)  / n - 0.2) < 0.02

    def test_no_occ_mean_leakage(self, df, edu_order, region_map):
        """Training-set occ means must not equal full-dataset means (leakage check)."""
        gm_train = compute_group_means(
            train_test_split(df, test_size=0.2, random_state=42)[0]
        )
        gm_full  = compute_group_means(df)
        # At least one occupation's mean should differ after splitting
        shared = set(gm_train["occ_means"]) & set(gm_full["occ_means"])
        diffs = [
            abs(gm_train["occ_means"][k] - gm_full["occ_means"][k])
            for k in shared
        ]
        assert max(diffs) > 100, (
            "Train-only group means are identical to full-dataset means — "
            "leakage may not have been eliminated."
        )

    def test_features_present_after_split_engineer(self, df, edu_order, region_map):
        df_train, df_test, _ = _make_split(df, edu_order, region_map)
        for col in FEATURES_FULL:
            assert col in df_train.columns
            assert col in df_test.columns

    def test_no_nulls_after_split_engineer(self, df, edu_order, region_map):
        df_train, df_test, _ = _make_split(df, edu_order, region_map)
        assert df_train[FEATURES_FULL].isnull().sum().sum() == 0
        assert df_test[FEATURES_FULL].isnull().sum().sum() == 0


class TestGroupMeansPersistence:
    """save/load round-trip must preserve values exactly."""

    def test_round_trip(self, df, tmp_path):
        gm = compute_group_means(df)
        path = str(tmp_path / "gm.json")
        save_group_means(gm, path)
        loaded = load_group_means(path)
        for occ, val in gm["occ_means"].items():
            assert abs(loaded["occ_means"][occ] - val) < 1e-3, f"Mismatch for {occ}"
        for st, val in gm["state_means"].items():
            assert abs(loaded["state_means"][st] - val) < 1e-3, f"Mismatch for {st}"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_group_means(str(tmp_path / "nonexistent.json"))


class TestProductionModelEndToEnd:
    """Production artefacts wired together: load → engineer with saved means → predict."""

    def test_production_encoding_consistent(self, production_model, df, cfg, edu_order, region_map):
        """Model trained with training-set means must get same feature names as saved means."""
        from pathlib import Path
        gm = load_group_means(str(Path(__file__).parent.parent / cfg["model"]["group_means_path"]))
        df_eng = engineer_features(df, edu_order, region_map,
                                   occ_means=gm["occ_means"],
                                   state_means=gm["state_means"])
        X = df_eng[FEATURES_FULL].head(10)
        preds_log = production_model.predict(X)
        preds_dollar = np.expm1(preds_log)
        assert (preds_dollar > 0).all()
        assert preds_dollar.min() > 50_000

    def test_build_feature_row_matches_training_features(self, production_model):
        """build_feature_row must produce exactly the columns the model expects."""
        row = build_feature_row(
            age=35, edu_ord=2, gender_bin=1, region_code=1,
            employment=1000.0, lq=1.0, jobs_k=2.0, hourly_mean=60.0,
            occ_mean_income=130_000.0, state_mean_income=140_000.0,
        )
        assert list(row.columns) == FEATURES_FULL
        assert row.shape == (1, len(FEATURES_FULL))
        pred = float(np.expm1(production_model.predict(row)[0]))
        assert pred > 50_000

    def test_r2_with_saved_group_means(self, production_model, df, cfg, edu_order, region_map):
        """End-to-end R² with saved training group means must exceed 0.05."""
        from pathlib import Path
        gm = load_group_means(str(Path(__file__).parent.parent / cfg["model"]["group_means_path"]))
        df_eng = engineer_features(df, edu_order, region_map,
                                   occ_means=gm["occ_means"],
                                   state_means=gm["state_means"])
        X = df_eng[FEATURES_FULL]
        y = df_eng["Annual Income"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        r2 = r2_score(y_test, np.expm1(production_model.predict(X_test)))
        assert r2 > 0.05, f"End-to-end R² too low: {r2:.4f}"

    def test_prediction_interval_ordered(self, production_model, df, cfg, edu_order, region_map):
        """pi_offset_10 < 0 and pi_offset_90 > 0 so PI always contains prediction."""
        from pathlib import Path

        from pipeline import load_metrics
        metrics = load_metrics(str(Path(__file__).parent.parent / cfg["model"]["metrics_path"]))
        assert metrics["pi_offset_10"] < 0, "Lower PI offset must be negative (residual below prediction)"
        assert metrics["pi_offset_90"] > 0, "Upper PI offset must be positive (residual above prediction)"
        assert metrics["pi_offset_10"] < metrics["pi_offset_90"]
