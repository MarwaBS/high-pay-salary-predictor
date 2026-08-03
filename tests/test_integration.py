"""
Integration tests — full pipeline path.

Each test exercises a complete data flow rather than a single unit:
  raw CSV → split → compute_group_means → engineer_features → train → predict

These guard against regressions where each unit test passes in isolation
but the composed pipeline breaks (e.g. feature-name mismatch, wrong
encoding, log/dollar-scale confusion).

Run: pytest tests/test_integration.py -v
"""

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from pipeline import (
    FEATURES_FULL,
    compute_group_means,
    engineer_features,
    load_group_means,
    save_group_means,
    train_test_positions,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _split_rows(df_raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The project's one split, on the configured parameters.

    Re-deriving it here would let these tests score rows the trainer trained on
    the moment either parameter changes, while still passing.
    """
    train_pos, test_pos = train_test_positions(
        len(df_raw), test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
    )
    return df_raw.iloc[train_pos], df_raw.iloc[test_pos]


def _make_split(df_raw: pd.DataFrame, edu_order: dict, region_map: dict, cfg: dict):
    """Split raw data, compute group means from train, engineer both splits."""
    train_raw, test_raw = _split_rows(df_raw, cfg)
    gm = compute_group_means(train_raw)
    df_train = engineer_features(
        train_raw, edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
    )
    df_test = engineer_features(
        test_raw, edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
    )
    return df_train, df_test, gm


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSplitThenEngineer:
    """Group means must be derived from train split only."""

    def test_train_test_sizes(self, df, edu_order, region_map, cfg):
        df_train, df_test, _ = _make_split(df, edu_order, region_map, cfg)
        n = len(df)
        held_out = cfg["model"]["test_size"]
        assert abs(len(df_test) / n - held_out) < 0.02
        assert abs(len(df_train) / n - (1 - held_out)) < 0.02

    def test_no_occ_mean_leakage(self, df, cfg):
        """Training-set occ means must not equal full-dataset means (leakage check)."""
        gm_train = compute_group_means(_split_rows(df, cfg)[0])
        gm_full = compute_group_means(df)
        # At least one occupation's mean should differ after splitting
        shared = set(gm_train["occ_means"]) & set(gm_full["occ_means"])
        diffs = [abs(gm_train["occ_means"][k] - gm_full["occ_means"][k]) for k in shared]
        assert max(diffs) > 100, (
            "Train-only group means are identical to full-dataset means — leakage may not have been eliminated."
        )

    def test_features_present_after_split_engineer(self, df, edu_order, region_map, cfg):
        df_train, df_test, _ = _make_split(df, edu_order, region_map, cfg)
        for col in FEATURES_FULL:
            assert col in df_train.columns
            assert col in df_test.columns

    def test_no_nulls_after_split_engineer(self, df, edu_order, region_map, cfg):
        df_train, df_test, _ = _make_split(df, edu_order, region_map, cfg)
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

        from pipeline import predict_quantiles_batch

        gm = load_group_means(str(Path(__file__).parent.parent / cfg["model"]["group_means_path"]))
        df_eng = engineer_features(df, edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"])
        X = df_eng[FEATURES_FULL].head(10)
        p50s = predict_quantiles_batch(production_model, X)[:, 1]
        assert all(p > 0 for p in p50s)
        assert min(p50s) > 50_000

    def test_p50_r2_with_saved_group_means(self, production_model, df, cfg, edu_order, region_map):
        """End-to-end P50 R² with saved training group means must be non-negative.

        P50 under the quantile objective is the median-minimiser, not the
        mean-minimiser, so R² is a weak fit-statistic for this model. The
        real SLO lives in
        ``tests/test_pipeline.py::test_saved_metrics_within_expected_range``
        (coverage + crossings). This test only guards against a catastrophic
        regression where predictions become uncorrelated with the target.
        """
        from pathlib import Path

        from pipeline import predict_quantiles_batch

        gm = load_group_means(str(Path(__file__).parent.parent / cfg["model"]["group_means_path"]))
        df_eng = engineer_features(df, edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"])
        _, test_pos = train_test_positions(
            len(df_eng), test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
        )
        X_test = df_eng[FEATURES_FULL].iloc[test_pos]
        y_test = df_eng["Annual Income"].iloc[test_pos]

        p50_preds = predict_quantiles_batch(production_model, X_test)[:, 1]
        r2 = r2_score(y_test, p50_preds)
        assert r2 > -0.05, f"End-to-end P50 R² {r2:.4f} is implausibly negative"

    def test_quantile_metrics_shape(self, cfg):
        """Metrics file must expose the quantile calibration fields.

        Asserts on the real SLO for the multi-quantile model: empirical
        80% coverage within a reasonable band, zero quantile crossings,
        and per-subgroup coverage present for at least one subgroup.
        """
        from pathlib import Path

        from pipeline import load_metrics

        metrics = load_metrics(str(Path(__file__).parent.parent / cfg["model"]["metrics_path"]))
        assert "quantile_coverage_80" in metrics, "metrics file must expose quantile_coverage_80"
        assert 0.65 <= metrics["quantile_coverage_80"] <= 0.92, (
            f"quantile_coverage_80 {metrics['quantile_coverage_80']:.3f} outside [0.65, 0.92]"
        )
        assert metrics.get("quantile_crossings", -1) == 0, (
            f"quantile_crossings must be 0, got {metrics.get('quantile_crossings')}"
        )
        assert "subgroup_coverage_80" in metrics, "metrics file must expose subgroup_coverage_80 dict"
        assert len(metrics["subgroup_coverage_80"]) > 0, "subgroup_coverage_80 must contain at least one subgroup"
