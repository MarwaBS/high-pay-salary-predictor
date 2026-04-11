"""
scripts/train_quantile.py
-------------------------
Train an XGBoost multi-quantile model predicting P10 / P50 / P90 of the
``log1p(Annual Income)`` target in a single pass.

Why a quantile model?
---------------------
Within the $100K+ cohort, individual income has extreme within-group
variance driven by unobserved factors (equity, bonuses, tenure, specific
employer). No point estimator can resolve that — so instead of pretending
to, the model returns a calibrated interval. "Given this profile, what
P10 / P50 / P90 income range should the caller expect?" is a useful
answer even when an exact-dollar prediction would not be.

Data caveat: the cleaning notebook double-filters the cohort — Census
rows with ``INCTOT >= 100_000`` inner-joined against BLS cells where
``A_MEAN >= 100_000``. This truncates the training set and puts a hard
ceiling on any point-estimator metric on this data. Re-prepping from the
full Census dataset with a binary ``>= $100K`` classifier is the right
structural fix and is intentionally out of scope for this trainer.

No MLflow / Optuna dependencies — this trainer is deliberately lean so
it can run on a CI worker or a dev machine without pulling the full
experiment-tracking stack. For HPO + tracking use ``scripts/train_model.py``.

Artefacts saved
---------------
  models/xgb_salary_model.ubj       multi-quantile XGBoost model
                                    (primary path, loaded by the API and
                                    dashboard via config.yaml::model.model_path)
  models/model_metrics.json         quantile metrics (coverage, pinball
                                    losses, crossings) + point-estimate
                                    metrics (P50 R²/MAE/RMSE, back-compat)
  models/baseline_stats.json        drift-monitor baseline
  models/group_means.json           target-encoding lookup
  models/feature_names.json         feature list

Usage
-----
    python scripts/train_quantile.py
    python scripts/train_quantile.py --config config.yaml
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from api.drift import save_baseline_stats
from pipeline import (
    FEATURES_FULL,
    compute_group_means,
    engineer_features,
    save_features,
    save_group_means,
    save_metrics,
    save_model,
)

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The three quantiles we predict. Must stay aligned with the API schema
# (``predicted_p10``, ``predicted_p50``, ``predicted_p90``) and with any
# downstream consumers.
QUANTILE_ALPHAS: list[float] = [0.10, 0.50, 0.90]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the multi-quantile salary predictor.")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    return p.parse_args()


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss — the scoring rule for quantile models.

    Lower is better. At alpha=0.5 this reduces to ``0.5 * mean(|error|)``.
    """
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    edu_order = cfg["education_order"]
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}
    random_state = model_cfg["random_state"]

    data_path = ROOT / cfg["data"]["cleaned"]
    logger.info("Loading dataset from %s", data_path)
    df_raw = pd.read_csv(data_path)
    logger.info("Raw dataset: %d rows × %d cols", *df_raw.shape)

    # ── Train / test split ───────────────────────────────────────────────────
    df_train_raw, df_test_raw = train_test_split(df_raw, test_size=model_cfg["test_size"], random_state=random_state)
    logger.info("Split: %d train / %d test rows", len(df_train_raw), len(df_test_raw))

    # ── Group means from TRAINING SET ONLY (no leakage) ──────────────────────
    group_means = compute_group_means(df_train_raw)

    # ── Engineer features ────────────────────────────────────────────────────
    df_train = engineer_features(
        df_train_raw,
        edu_order,
        region_map,
        occ_means=group_means["occ_means"],
        state_means=group_means["state_means"],
    )
    df_test = engineer_features(
        df_test_raw,
        edu_order,
        region_map,
        occ_means=group_means["occ_means"],
        state_means=group_means["state_means"],
    )

    X_train, y_train = df_train[FEATURES_FULL], df_train["Annual Income"]
    X_test, y_test = df_test[FEATURES_FULL], df_test["Annual Income"]
    y_train_log = np.log1p(y_train)
    logger.info("Target log1p-transformed for training")

    # ── Train multi-quantile XGBoost ────────────────────────────────────────
    # reg:quantileerror with quantile_alpha=[...] produces one model that
    # outputs all three quantiles simultaneously. Requires xgboost >= 2.0.
    params = {
        "n_estimators": model_cfg["n_estimators"],
        "max_depth": model_cfg["max_depth"],
        "learning_rate": model_cfg["learning_rate"],
        "subsample": model_cfg["subsample"],
        "colsample_bytree": model_cfg["colsample_bytree"],
        "reg_lambda": model_cfg["reg_lambda"],
    }
    logger.info("Training XGBoost quantile model (alphas=%s)…", QUANTILE_ALPHAS)
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_ALPHAS,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train_log)

    # ── Evaluate on test set (dollar space) ─────────────────────────────────
    preds_log = model.predict(X_test)  # shape (n_test, 3)
    preds_dollar = np.expm1(preds_log)

    p10_dollar = preds_dollar[:, 0]
    p50_dollar = preds_dollar[:, 1]
    p90_dollar = preds_dollar[:, 2]

    # Point-estimate backwards-compatibility metrics (P50 as the point pred)
    r2 = float(r2_score(y_test, p50_dollar))
    mae = float(mean_absolute_error(y_test, p50_dollar))
    rmse = float(mean_squared_error(y_test, p50_dollar) ** 0.5)

    # Quantile-specific metrics: pinball loss + empirical coverage
    pinballs_dollar = {
        f"p{int(alpha * 100)}_pinball": round(pinball_loss(y_test.values, preds_dollar[:, i], alpha), 2)
        for i, alpha in enumerate(QUANTILE_ALPHAS)
    }
    coverage_80 = float(((y_test.values >= p10_dollar) & (y_test.values <= p90_dollar)).mean())
    interval_width_median = float(np.median(p90_dollar - p10_dollar))

    # Quantile crossing check — preds should satisfy p10 <= p50 <= p90
    crossings = int(((p10_dollar > p50_dollar) | (p50_dollar > p90_dollar) | (p10_dollar > p90_dollar)).sum())

    logger.info(
        "Quantile metrics: coverage_80=%.1f%% width_median=$%d crossings=%d/%d",
        coverage_80 * 100,
        int(interval_width_median),
        crossings,
        len(y_test),
    )
    logger.info("Point (P50 back-compat): R²=%.4f RMSE=$%d MAE=$%d", r2, int(rmse), int(mae))

    # ── 5-fold CV on training set only, dollar-space P50 R² ─────────────────
    # CV and test R² are computed in the same (dollar) space so the numbers
    # are directly comparable.
    kf = KFold(n_splits=model_cfg.get("cv_folds", 5), shuffle=True, random_state=random_state)
    cv_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        fold_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=QUANTILE_ALPHAS,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
        fold_model.fit(X_train.iloc[tr_idx], y_train_log.iloc[tr_idx])
        fold_preds = np.expm1(fold_model.predict(X_train.iloc[va_idx]))[:, 1]  # P50
        fold_r2 = float(r2_score(y_train.iloc[va_idx], fold_preds))
        cv_scores.append(fold_r2)
        logger.info("  fold %d: P50 R²=%.4f", fold_idx, fold_r2)
    cv_r2_mean = float(np.mean(cv_scores))
    cv_r2_std = float(np.std(cv_scores))
    logger.info("CV R² (P50, dollar, train-only) = %.4f ± %.4f", cv_r2_mean, cv_r2_std)

    # ── Save artefacts ───────────────────────────────────────────────────────
    # Single write to the path declared in config.yaml::model.model_path.
    # The API, dashboard, and tests all load from that path and pick up
    # the multi-quantile output shape via ``pipeline.predict_quantiles``
    # + ``is_quantile_model``.
    primary_model_path = ROOT / cfg["model"]["model_path"]
    save_model(model, str(primary_model_path))

    save_features(FEATURES_FULL, str(ROOT / cfg["model"]["features_path"]))
    save_group_means(group_means, str(ROOT / cfg["model"]["group_means_path"]))

    metrics = {
        "r2": round(r2, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std": round(cv_r2_std, 4),
        "cv_space": "dollar",
        "cv_train_only": True,
        # Quantile-specific metrics
        "quantile_alphas": QUANTILE_ALPHAS,
        "quantile_coverage_80": round(coverage_80, 4),
        "quantile_width_median": round(interval_width_median, 2),
        "quantile_crossings": crossings,
        **pinballs_dollar,
        # PI offsets kept for backward compat with the old API path — now
        # derived from actual quantile predictions rather than residuals,
        # so the /predict response's prediction_interval_low/high stay
        # populated with a meaningful range.
        "pi_offset_10": round(float(np.mean(p10_dollar - p50_dollar)), 2),
        "pi_offset_90": round(float(np.mean(p90_dollar - p50_dollar)), 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(FEATURES_FULL),
        "train_date": datetime.date.today().isoformat(),
        "hyperparameters": params,
        "log_transform": True,
        "fixed_group_means": True,
        "objective": "reg:quantileerror",
        "framing": (
            "Quantile regression (P10/P50/P90) within the $100K+ cohort. "
            "The previous point-estimate framing was killed because the "
            "cohort variance is dominated by unobserved factors — no point "
            "estimator can exceed a ceiling of R² ≈ 0.10. The quantile "
            "output is useful where the point estimate is not: it tells "
            "a caller the realistic income range for their profile."
        ),
        "data_prep_caveat": (
            "Training cohort is double-filtered by notebook 1: Census "
            "rows with INCTOT >= 100K inner-joined against BLS cells "
            "where A_MEAN >= 100K. A future gap should re-prep using "
            "the full Census dataset with a binary >= $100K classifier."
        ),
    }
    save_metrics(metrics, str(ROOT / cfg["model"]["metrics_path"]))

    # ── Drift baseline from training features ───────────────────────────────
    baseline_data = {feat: X_train[feat].tolist() for feat in FEATURES_FULL}
    save_baseline_stats(baseline_data, str(ROOT / "models" / "baseline_stats.json"))

    logger.info("Artefacts saved:")
    logger.info("  Model       : %s", primary_model_path)
    logger.info("  Features    : %s", ROOT / cfg["model"]["features_path"])
    logger.info("  Group means : %s", ROOT / cfg["model"]["group_means_path"])
    logger.info("  Metrics     : %s", ROOT / cfg["model"]["metrics_path"])
    logger.info("  Drift base  : %s", ROOT / "models" / "baseline_stats.json")
    logger.info("Done — Test P50 R²=%.4f  coverage_80=%.1f%%", r2, coverage_80 * 100)


if __name__ == "__main__":
    main()
