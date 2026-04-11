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
it can run on a CI worker or a dev machine without pulling an
experiment-tracking stack. Hyper-parameters are pinned in ``config.yaml``
and were selected by a prior offline Optuna search against the
``reg:squarederror`` baseline; re-tuning against the quantile objective
is out of scope for this trainer.

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
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from api import __version__ as SERVICE_VERSION
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


def _resolve_git_sha() -> str:
    """Return the short git SHA of HEAD, or ``"unknown"`` if git isn't available.

    CI workflows set ``GITHUB_SHA``; honour that first so scheduled runs
    record the exact SHA that triggered the workflow even when the
    checkout is shallow or in a detached-HEAD state. Falls back to a
    local ``git rev-parse`` for developer runs, and finally to
    ``"unknown"`` so the trainer never crashes on a bare tarball.
    """
    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha[:12]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _hash_training_data(path: Path) -> str:
    """Return the first 12 hex chars of the SHA-256 of the training CSV.

    Binding the model version to the data content means two runs on the
    same code against the same CSV produce the same ``MODEL_VERSION``,
    and two runs on the same code against *different* CSVs do not.
    That is what makes the version string a real reproducibility
    primitive and not just a timestamp.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def build_model_version(data_path: Path) -> str:
    """Build the canonical model version string.

    Shape: ``{service_version}+{git_sha}.{data_sha256_prefix}``. The
    ``+`` separator keeps this a valid semver build-metadata suffix so
    tooling that parses semver strings (release automation, dependency
    managers) continues to work. The two prefixed fragments are each
    12 hex chars — enough to disambiguate without bloating logs.

    Examples::

        2.0.0+a1b2c3d4e5f6.9e8d7c6b5a40
        2.0.0+unknown.9e8d7c6b5a40     # offline build, no git
    """
    git_sha = _resolve_git_sha()
    data_sha = _hash_training_data(data_path)
    return f"{SERVICE_VERSION}+{git_sha}.{data_sha}"


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

    # ── Subgroup quantile calibration ───────────────────────────────────────
    # Coverage is computed per Gender and per Region so fairness drift is
    # surfaced as a first-class metric. A sudden collapse in one subgroup's
    # coverage (e.g. women dropping from 0.77 to 0.50) would indicate the
    # model has stopped being calibrated for that population.
    subgroup_coverage: dict[str, float] = {}
    for col in ("Gender", "Region"):
        if col not in df_test.columns:
            continue
        for val in sorted(df_test[col].dropna().unique()):
            mask = (df_test[col] == val).to_numpy()
            if mask.sum() < 30:
                continue
            subgroup_hit = (y_test.values[mask] >= p10_dollar[mask]) & (y_test.values[mask] <= p90_dollar[mask])
            cov = float(subgroup_hit.mean())
            subgroup_coverage[f"{col}={val}"] = round(cov, 4)
            logger.info("  subgroup coverage_80 %-20s n=%4d cov=%.3f", f"{col}={val}", int(mask.sum()), cov)

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

    # ── Model provenance: service version + code SHA + data SHA ────────────
    # The composite version is the reproducibility primitive: any operator
    # investigating a production incident can recover the exact training
    # state (code, data) from the three fragments. It is also the string
    # the scheduled release workflow uses to tag GitHub Releases.
    model_version = build_model_version(data_path)
    logger.info("Model version: %s", model_version)

    metrics = {
        "model_version": model_version,
        "service_version": SERVICE_VERSION,
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
        "subgroup_coverage_80": subgroup_coverage,
        **pinballs_dollar,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(FEATURES_FULL),
        "train_date": datetime.date.today().isoformat(),
        "hyperparameters": params,
        "log_transform": True,
        "fixed_group_means": True,
        "objective": "reg:quantileerror",
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
