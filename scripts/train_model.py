"""
scripts/train_model.py
----------------------
Train the XGBoost salary prediction model and save artefacts to disk.

This script replaces the fragile inline Python one-liner that was previously
embedded in the Makefile `model` target.  It is the single canonical place
for production model training outside of the notebook environment.

Usage
-----
    # From repo root (uses .venv automatically via make model):
    make model

    # Or directly:
    python scripts/train_model.py
    python scripts/train_model.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

# Allow running from any working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import FEATURES_FULL, engineer_features  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost salary prediction model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml (default: repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and engineer features but skip training/saving.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    edu_order  = cfg["education_order"]
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}

    # ── Load & engineer features ─────────────────────────────────────────────
    data_path = ROOT / cfg["data"]["cleaned"]
    logger.info("Loading dataset from %s", data_path)
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw, edu_order, region_map)
    logger.info("Dataset shape: %s", df.shape)

    X = df[FEATURES_FULL]
    y = df["Annual Income"]

    if args.dry_run:
        logger.info("Dry run complete — skipping training.")
        return

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
    )
    logger.info(
        "Split: %d train / %d test  (features: %d)",
        len(X_train), len(X_test), X_train.shape[1],
    )

    # ── Fit model ────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["learning_rate"],
        subsample=cfg["model"]["subsample"],
        colsample_bytree=cfg["model"]["colsample_bytree"],
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f s", elapsed)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    logger.info(
        "Test metrics — R²: %.4f  MAE: $%,.0f  RMSE: $%,.0f",
        r2, mae, rmse,
    )

    # ── Cross-validation ─────────────────────────────────────────────────────
    n_folds = cfg["model"].get("cv_folds", 5)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg["model"]["random_state"])
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    logger.info(
        "%d-fold CV R²: %.4f ± %.4f  (min: %.4f  max: %.4f)",
        n_folds, cv_scores.mean(), cv_scores.std(), cv_scores.min(), cv_scores.max(),
    )

    # ── Save artefacts ───────────────────────────────────────────────────────
    model_path    = ROOT / cfg["model"]["model_path"]
    features_path = ROOT / cfg["model"]["features_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(features_path, "wb") as f:
        pickle.dump(FEATURES_FULL, f)

    logger.info("Model saved to    %s", model_path)
    logger.info("Features saved to %s", features_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
