"""Train the multi-quantile regressor and the premium-tier classifier in one pass.

The regressor predicts P10/P50/P90 of ``log1p(Annual Income)`` via
``reg:quantileerror``; the classifier predicts
``P(Annual Income >= config.yaml::model.premium_threshold)`` on the same feature
matrix. Both read their hyper-parameters from ``config.yaml::model`` with no
fallback, and write to the paths declared there. See DESIGN_DECISIONS.md D-004
for why there are two heads and why the classifier's label is scoped to the
high-pay cohort.

Usage: ``python -m scripts.train_quantile [--config config.yaml]``
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from api import __version__ as SERVICE_VERSION
from api.drift import save_baseline_stats
from config_schema import ProjectConfig
from pipeline import (
    FEATURES_FULL,
    compute_group_means,
    engineer_features,
    save_classifier,
    save_conformal,
    save_features,
    save_group_means,
    save_metrics,
    save_model,
    sha256_file,
    train_test_positions,
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

# Smallest slice that gets its own published fairness metric — subgroup
# coverage below, classifier AUC further down. At n=30 a coverage rate near 0.8
# already carries a 95% sampling interval of ±0.14, so thinner slices would
# publish their own sampling noise as unfairness.
MIN_SUBGROUP_SIZE = 30


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


def _library_versions() -> dict[str, str]:
    """Versions of the libraries that determine the trained artifact bytes.

    Stamped into the metrics file so it states the environment its numbers were
    produced under — the reproducibility claim rests on the lock, and this
    records what the lock actually resolved to at train time.
    """
    import sklearn
    import xgboost

    return {"xgboost": xgboost.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}


def _prepare_split(
    df_raw: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
    edu_order: dict[str, int],
    region_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Split, then engineer features using TRAIN-only group means (no leakage).

    Also returns the raw train frame so cross-validation can re-derive
    per-fold target-encoding means from raw rows (see :func:`_cross_val_r2`).
    """
    train_pos, test_pos = train_test_positions(len(df_raw), test_size=test_size, random_state=seed)
    df_train_raw, df_test_raw = df_raw.iloc[train_pos], df_raw.iloc[test_pos]
    group_means = compute_group_means(df_train_raw)
    df_train = engineer_features(
        df_train_raw, edu_order, region_map, occ_means=group_means["occ_means"], state_means=group_means["state_means"]
    )
    df_test = engineer_features(
        df_test_raw, edu_order, region_map, occ_means=group_means["occ_means"], state_means=group_means["state_means"]
    )
    return df_train, df_test, df_train_raw, group_means


def _cross_val_r2(
    df_train_raw: pd.DataFrame,
    *,
    seed: int,
    n_splits: int,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    params: dict,
) -> tuple[float, float]:
    """K-fold CV P50 R² with per-fold target encoding (leakage-free).

    Group means are recomputed from each fold's TRAIN rows only, so a
    validation row is never encoded with a mean that saw its own target.
    Encoding once from all of train and folding over that matrix leaks:
    every validation fold's targets sit inside the means baked into its own
    ``Occ_Mean_Income`` / ``State_Mean_Income`` features. Returns (mean, std)
    of the per-fold dollar-space R².
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw = df_train_raw.reset_index(drop=True)
    cv_scores: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(raw)):
        gm = compute_group_means(raw.iloc[tr_idx])
        fold_train = engineer_features(
            raw.iloc[tr_idx], edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
        )
        fold_val = engineer_features(
            raw.iloc[va_idx], edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
        )
        fold_model = _train_quantile_regressor(
            fold_train[FEATURES_FULL], np.log1p(fold_train["Annual Income"]), params=params, seed=seed
        )
        fold_preds = np.expm1(fold_model.predict(fold_val[FEATURES_FULL]))[:, 1]  # P50
        fold_r2 = float(r2_score(fold_val["Annual Income"], fold_preds))
        cv_scores.append(fold_r2)
        logger.info("  fold %d: P50 R²=%.4f", fold_idx, fold_r2)
    return float(np.mean(cv_scores)), float(np.std(cv_scores))


def _cross_conformal_delta(
    df_train_raw: pd.DataFrame,
    *,
    seed: int,
    n_splits: int,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    params: dict,
    target_coverage: float,
) -> tuple[float, int]:
    """Cross-conformal (CQR) interval margin in log1p space.

    Each fold trains the quantile model on the fold's TRAIN rows (per-fold
    target-encoding means, leakage-free — same protocol as :func:`_cross_val_r2`)
    and scores the held-out rows with the CQR conformity score
    ``max(q_lo - y, y - q_hi)``. The margin is the 0.80 empirical quantile of
    the pooled scores (with the standard ``(n+1)`` small-sample lift); because
    the fold models differ from the served full-data model the coverage is
    approximate, validated at ~0.80 on the held-out test set. The shipped model still trains on
    ALL of train, so its bytes are unchanged; this only estimates how far to
    widen its raw P10/P90 interval — which under-covers by a couple of points —
    to reach the nominal coverage. Returns (delta, n_scores).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw = df_train_raw.reset_index(drop=True)
    scores: list[np.ndarray] = []
    for tr_idx, va_idx in kf.split(raw):
        gm = compute_group_means(raw.iloc[tr_idx])
        fold_train = engineer_features(
            raw.iloc[tr_idx], edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
        )
        fold_val = engineer_features(
            raw.iloc[va_idx], edu_order, region_map, occ_means=gm["occ_means"], state_means=gm["state_means"]
        )
        fold_model = _train_quantile_regressor(
            fold_train[FEATURES_FULL], np.log1p(fold_train["Annual Income"]), params=params, seed=seed
        )
        q = fold_model.predict(fold_val[FEATURES_FULL])
        y_log = np.log1p(fold_val["Annual Income"].to_numpy())
        scores.append(np.maximum(q[:, 0] - y_log, y_log - q[:, 2]))
    pooled = np.concatenate(scores)
    n = len(pooled)
    k = min(int(np.ceil((n + 1) * target_coverage)), n)
    delta = float(np.sort(pooled)[k - 1])
    return delta, n


def _train_quantile_regressor(
    X_train: pd.DataFrame, y_train_log: pd.Series, *, params: dict, seed: int
) -> XGBRegressor:
    """Fit the multi-quantile regressor (P10/P50/P90 in one model)."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_ALPHAS,
        tree_method="hist",
        random_state=seed,
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train_log)
    return model


def _train_premium_classifier(
    X_train: pd.DataFrame, y_train_clf: pd.Series, *, params: dict, seed: int
) -> XGBClassifier:
    """Fit the premium-tier head with the ``classifier_*`` config hyper-parameters.

    No ``scale_pos_weight``: at the ~40/60 class balance of this cohort the
    imbalance is mild, and the head is served as a probability
    (``p_above_premium_threshold``) rather than a ranking. The weighted variant
    was not run, so this follows from the class balance, not from a measurement.
    """
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
        **params,
    )
    clf.fit(X_train, y_train_clf)
    return clf


def _headline_metrics_for_seed(
    df_raw: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    params: dict,
    clf_params: dict,
    premium_threshold: int,
) -> dict[str, float]:
    """Train both heads on one seed's split and return the headline metrics.

    The stability loop calls this across several seeds so the reported
    numbers carry a mean±std, not a single-split point estimate — the
    difference between "R²=0.82" and "R²=0.82±0.01 over 5 seeds".
    """
    df_train, df_test, _, _ = _prepare_split(
        df_raw, seed=seed, test_size=test_size, edu_order=edu_order, region_map=region_map
    )
    X_train, y_train = df_train[FEATURES_FULL], df_train["Annual Income"]
    X_test, y_test = df_test[FEATURES_FULL], df_test["Annual Income"]

    model = _train_quantile_regressor(X_train, np.log1p(y_train), params=params, seed=seed)
    preds = np.expm1(model.predict(X_test))
    coverage = float(((y_test.to_numpy() >= preds[:, 0]) & (y_test.to_numpy() <= preds[:, 2])).mean())

    y_test_clf = (y_test >= premium_threshold).astype(int)
    y_train_clf = (y_train >= premium_threshold).astype(int)
    clf = _train_premium_classifier(X_train, y_train_clf, params=clf_params, seed=seed)
    proba = clf.predict_proba(X_test)[:, 1]

    # The logistic reference refits on this seed's split too. Scoring it once on
    # the shipped split while the head carries a mean over five would compare a
    # point estimate against a distribution, which cannot settle which ranks better.
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    logreg.fit(X_train, y_train_clf)
    baseline_proba = logreg.predict_proba(X_test)[:, 1]

    return {
        "p50_r2": float(r2_score(y_test, preds[:, 1])),
        "coverage_80": coverage,
        "clf_roc_auc": float(roc_auc_score(y_test_clf, proba)),
        "clf_brier": float(brier_score_loss(y_test_clf, proba)),
        "clf_baseline_logreg_roc_auc": float(roc_auc_score(y_test_clf, baseline_proba)),
    }


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Validate before training: the trainer produces the artefacts the API
    # loads, so a config the API would refuse must not reach a model file.
    ProjectConfig.model_validate(cfg)

    model_cfg = cfg["model"]
    edu_order = cfg["education_order"]
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}
    random_state = model_cfg["random_state"]
    n_jobs = int(model_cfg["n_jobs"])

    data_path = ROOT / cfg["data"]["cleaned"]
    logger.info("Loading dataset from %s", data_path)
    df_raw = pd.read_csv(data_path)
    logger.info("Raw dataset: %d rows × %d cols", *df_raw.shape)

    # ── Train / test split + leakage-safe feature engineering ────────────────
    df_train, df_test, df_train_raw, group_means = _prepare_split(
        df_raw,
        seed=random_state,
        test_size=model_cfg["test_size"],
        edu_order=edu_order,
        region_map=region_map,
    )
    logger.info("Split: %d train / %d test rows", len(df_train), len(df_test))

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
        "n_jobs": n_jobs,
    }
    logger.info("Training XGBoost quantile model (alphas=%s)…", QUANTILE_ALPHAS)
    model = _train_quantile_regressor(X_train, y_train_log, params=params, seed=random_state)

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
        f"p{int(alpha * 100)}_pinball": round(pinball_loss(y_test.to_numpy(), preds_dollar[:, i], alpha), 2)
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
            if mask.sum() < MIN_SUBGROUP_SIZE:
                continue
            subgroup_hit = (y_test.values[mask] >= p10_dollar[mask]) & (y_test.values[mask] <= p90_dollar[mask])
            cov = float(subgroup_hit.mean())
            subgroup_coverage[f"{col}={val}"] = round(cov, 4)
            logger.info("  subgroup coverage_80 %-20s n=%4d cov=%.3f", f"{col}={val}", int(mask.sum()), cov)

    # ── 5-fold CV on training set only, dollar-space P50 R² ─────────────────
    # Per-fold target-encoding means (leakage-free) — see _cross_val_r2. CV and
    # test R² are computed in the same dollar space so the numbers compare.
    cv_r2_mean, cv_r2_std = _cross_val_r2(
        df_train_raw,
        seed=random_state,
        n_splits=model_cfg["cv_folds"],
        edu_order=edu_order,
        region_map=region_map,
        params=params,
    )
    logger.info("CV R² (P50, dollar, train-only, per-fold means) = %.4f ± %.4f", cv_r2_mean, cv_r2_std)

    # ── Cross-conformal interval calibration (CQR) ──────────────────────────
    # The raw P10/P90 interval under-covers its nominal 80% by ~2 points. A
    # cross-conformal margin, estimated from train-only folds so the shipped
    # model's bytes are untouched, widens the served interval to target.
    target_coverage = float(QUANTILE_ALPHAS[2] - QUANTILE_ALPHAS[0])
    conformal_delta, n_conf_scores = _cross_conformal_delta(
        df_train_raw,
        seed=random_state,
        n_splits=model_cfg["cv_folds"],
        edu_order=edu_order,
        region_map=region_map,
        params=params,
        target_coverage=target_coverage,
    )
    p10_conf = np.expm1(preds_log[:, 0] - conformal_delta)
    p90_conf = np.expm1(preds_log[:, 2] + conformal_delta)
    coverage_80_conformal = float(((y_test.values >= p10_conf) & (y_test.values <= p90_conf)).mean())
    width_median_conformal = float(np.median(p90_conf - p10_conf))
    logger.info(
        "Conformal: delta=%.4f target=%.2f test_coverage=%.3f width_median=$%d",
        conformal_delta,
        target_coverage,
        coverage_80_conformal,
        int(width_median_conformal),
    )

    # ── Premium-tier classifier head ────────────────────────────────────────
    # Binary XGBoost classifier trained on the same engineered feature
    # matrix as the quantile regressor. Label: Annual Income >= the
    # premium threshold configured in config.yaml; hyper-parameters come
    # from config.yaml::model.classifier_* alongside the regressor's.
    premium_threshold = int(model_cfg["premium_threshold"])
    clf_params = {
        "n_estimators": model_cfg["classifier_n_estimators"],
        "max_depth": model_cfg["classifier_max_depth"],
        "learning_rate": model_cfg["classifier_learning_rate"],
        "subsample": model_cfg["classifier_subsample"],
        "colsample_bytree": model_cfg["classifier_colsample_bytree"],
        "reg_lambda": model_cfg["classifier_reg_lambda"],
        "n_jobs": n_jobs,
    }
    y_train_clf = (y_train >= premium_threshold).astype(int)
    y_test_clf = (y_test >= premium_threshold).astype(int)
    pos_rate_train = float(y_train_clf.mean())
    pos_rate_test = float(y_test_clf.mean())
    logger.info(
        "Classifier label: Annual Income >= $%d  (positives: train=%.1f%% / test=%.1f%%)",
        premium_threshold,
        pos_rate_train * 100,
        pos_rate_test * 100,
    )
    classifier = _train_premium_classifier(X_train, y_train_clf, params=clf_params, seed=random_state)

    clf_proba_test = classifier.predict_proba(X_test)[:, 1]
    clf_pred_test = (clf_proba_test >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(y_test_clf, clf_proba_test))
    pr_auc = float(average_precision_score(y_test_clf, clf_proba_test))
    accuracy = float((clf_pred_test == y_test_clf.values).mean())
    # True positive rate, precision, recall at the default 0.5 threshold
    tp = int(((clf_pred_test == 1) & (y_test_clf.values == 1)).sum())
    fp = int(((clf_pred_test == 1) & (y_test_clf.values == 0)).sum())
    fn = int(((clf_pred_test == 0) & (y_test_clf.values == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    logger.info(
        "Classifier metrics: ROC-AUC=%.4f  PR-AUC=%.4f  acc=%.4f  precision=%.4f  recall=%.4f  F1=%.4f",
        roc_auc,
        pr_auc,
        accuracy,
        precision,
        recall,
        f1,
    )

    # ── Baseline comparisons + calibration (Brier) ──────────────────────────
    # Majority-class accuracy and a scaled logistic-regression ROC-AUC make the
    # XGB head's lift — or shortfall — explicit rather than assumed. Brier
    # (lower=better) scores the served probabilities against the base-rate
    # constant predictor.
    baseline_majority_acc = float(max(pos_rate_test, 1.0 - pos_rate_test))
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))
    logreg.fit(X_train, y_train_clf)
    baseline_logreg_roc_auc = float(roc_auc_score(y_test_clf, logreg.predict_proba(X_test)[:, 1]))
    clf_brier = float(brier_score_loss(y_test_clf, clf_proba_test))
    baseline_brier = float(brier_score_loss(y_test_clf, np.full(len(y_test_clf), pos_rate_test)))
    logger.info(
        "Baselines: majority-acc=%.4f  logreg ROC-AUC=%.4f | Brier: model=%.4f base-rate=%.4f",
        baseline_majority_acc,
        baseline_logreg_roc_auc,
        clf_brier,
        baseline_brier,
    )

    # Subgroup ROC-AUC — fairness guardrail on the classifier head.
    # A collapse in one subgroup's AUC (relative to the global AUC)
    # is the drift signal the fairness test locks in.
    clf_subgroup_roc_auc: dict[str, float] = {}
    for col in ("Gender", "Region"):
        if col not in df_test.columns:
            continue
        for val in sorted(df_test[col].dropna().unique()):
            mask = (df_test[col] == val).to_numpy()
            if mask.sum() < MIN_SUBGROUP_SIZE:
                continue
            y_sub = y_test_clf.values[mask]
            # Skip degenerate slices (all pos or all neg) — AUC is undefined
            if len(np.unique(y_sub)) < 2:
                continue
            sub_auc = float(roc_auc_score(y_sub, clf_proba_test[mask]))
            clf_subgroup_roc_auc[f"{col}={val}"] = round(sub_auc, 4)
            logger.info("  subgroup clf ROC-AUC %-20s n=%4d auc=%.3f", f"{col}={val}", int(mask.sum()), sub_auc)

    # ── Stability across seeds (mean±std, not a single split) ───────────────
    # The headline test numbers above come from one split. A staff-level
    # submission reports the spread, so re-run both heads across several seeds
    # and record mean±std. Cheap on a 10K-row set; turns "R²=0.82" into
    # "R²=0.82±0.01", which is the difference between a lucky split and a
    # stable model.
    stability_seeds = list(model_cfg["stability_seeds"])
    logger.info("Stability eval over %d seeds: %s", len(stability_seeds), stability_seeds)
    stab_runs = [
        _headline_metrics_for_seed(
            df_raw,
            seed=s,
            test_size=model_cfg["test_size"],
            edu_order=edu_order,
            region_map=region_map,
            params=params,
            clf_params=clf_params,
            premium_threshold=premium_threshold,
        )
        for s in stability_seeds
    ]
    stability_metrics: dict[str, float] = {}
    for key in ("p50_r2", "coverage_80", "clf_roc_auc", "clf_brier", "clf_baseline_logreg_roc_auc"):
        vals = [run[key] for run in stab_runs]
        stability_metrics[f"stability_{key}_mean"] = round(float(np.mean(vals)), 4)
        stability_metrics[f"stability_{key}_std"] = round(float(np.std(vals)), 4)
    logger.info(
        "Stability: P50 R²=%.4f±%.4f  cov80=%.4f±%.4f  clf AUC=%.4f±%.4f  Brier=%.4f±%.4f"
        "  | logreg AUC=%.4f±%.4f over the same splits",
        stability_metrics["stability_p50_r2_mean"],
        stability_metrics["stability_p50_r2_std"],
        stability_metrics["stability_coverage_80_mean"],
        stability_metrics["stability_coverage_80_std"],
        stability_metrics["stability_clf_roc_auc_mean"],
        stability_metrics["stability_clf_roc_auc_std"],
        stability_metrics["stability_clf_brier_mean"],
        stability_metrics["stability_clf_brier_std"],
        stability_metrics["stability_clf_baseline_logreg_roc_auc_mean"],
        stability_metrics["stability_clf_baseline_logreg_roc_auc_std"],
    )

    # ── Save artefacts ───────────────────────────────────────────────────────
    # Single write to the path declared in config.yaml::model.model_path.
    # The API, dashboard, and tests all load from that path and pick up
    # the multi-quantile output shape via ``pipeline.predict_quantiles``
    # + ``is_quantile_model``.
    primary_model_path = ROOT / cfg["model"]["model_path"]
    save_model(model, str(primary_model_path))

    classifier_path = ROOT / cfg["model"]["classifier_path"]
    save_classifier(classifier, str(classifier_path))

    save_features(FEATURES_FULL, str(ROOT / cfg["model"]["features_path"]))
    save_group_means(group_means, str(ROOT / cfg["model"]["group_means_path"]))

    conformal_path = ROOT / cfg["model"]["conformal_path"]
    save_conformal(conformal_delta, str(conformal_path), target_coverage=target_coverage, n_scores=n_conf_scores)

    # ── Drift baseline from training features ───────────────────────────────
    # Written before the metrics file so its bytes can be content-addressed
    # alongside the other artefacts below.
    features_path = ROOT / cfg["model"]["features_path"]
    group_means_path = ROOT / cfg["model"]["group_means_path"]
    baseline_path = ROOT / cfg["model"]["baseline_stats_path"]
    baseline_data = {feat: X_train[feat].tolist() for feat in FEATURES_FULL}
    save_baseline_stats(baseline_data, str(baseline_path))

    # ── Model provenance: service version + code SHA + data SHA ────────────
    # The composite version is the reproducibility primitive: any operator
    # investigating a production incident can recover the exact training
    # state (code, data) from the three fragments. It is also the string
    # the scheduled release workflow uses to tag GitHub Releases.
    model_version = build_model_version(data_path)
    logger.info("Model version: %s", model_version)

    # Content-address every served artefact. The API re-hashes on load and
    # crashes on mismatch, and CI verifies committed bytes against these — so a
    # corrupt or desynced artefact cannot ship under green.
    artifact_sha256 = {
        "model": sha256_file(primary_model_path),
        "classifier": sha256_file(classifier_path),
        "features": sha256_file(features_path),
        "group_means": sha256_file(group_means_path),
        "baseline_stats": sha256_file(baseline_path),
        "conformal": sha256_file(conformal_path),
    }

    metrics = {
        "model_version": model_version,
        "service_version": SERVICE_VERSION,
        "artifact_sha256": artifact_sha256,
        "library_versions": _library_versions(),
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
        # Cross-conformal calibration: the raw interval under-covers, so the API
        # serves the conformalized bounds. Both raw and conformalized coverage
        # are recorded so the gain is auditable.
        "conformal_delta": round(conformal_delta, 6),
        "conformal_target_coverage": round(target_coverage, 4),
        "conformal_coverage_80": round(coverage_80_conformal, 4),
        "conformal_width_median": round(width_median_conformal, 2),
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
        # Premium-tier classifier head
        "classifier_objective": "binary:logistic",
        "classifier_threshold": premium_threshold,
        "classifier_positive_rate_train": round(pos_rate_train, 4),
        "classifier_positive_rate_test": round(pos_rate_test, 4),
        "classifier_roc_auc": round(roc_auc, 4),
        "classifier_pr_auc": round(pr_auc, 4),
        "classifier_accuracy": round(accuracy, 4),
        "classifier_precision": round(precision, 4),
        "classifier_recall": round(recall, 4),
        "classifier_f1": round(f1, 4),
        # Calibration + baselines: the served probability is honest (Brier)
        # and the head beats the majority / base-rate baselines (it trails the
        # logistic on ROC-AUC).
        "classifier_brier": round(clf_brier, 4),
        "classifier_brier_base_rate": round(baseline_brier, 4),
        "classifier_baseline_majority_acc": round(baseline_majority_acc, 4),
        "classifier_baseline_logreg_roc_auc": round(baseline_logreg_roc_auc, 4),
        "classifier_subgroup_roc_auc": clf_subgroup_roc_auc,
        # Stability across seeds (mean±std of the headline metrics)
        "stability_seeds": stability_seeds,
        **stability_metrics,
    }
    save_metrics(metrics, str(ROOT / cfg["model"]["metrics_path"]))

    logger.info("Artefacts saved:")
    logger.info("  Model       : %s", primary_model_path)
    logger.info("  Classifier  : %s", classifier_path)
    logger.info("  Features    : %s", ROOT / cfg["model"]["features_path"])
    logger.info("  Group means : %s", ROOT / cfg["model"]["group_means_path"])
    logger.info("  Metrics     : %s", ROOT / cfg["model"]["metrics_path"])
    logger.info("  Drift base  : %s", baseline_path)
    logger.info(
        "Done — Test P50 R²=%.4f  coverage_80=%.1f%%  clf ROC-AUC=%.4f",
        r2,
        coverage_80 * 100,
        roc_auc,
    )


if __name__ == "__main__":
    main()
