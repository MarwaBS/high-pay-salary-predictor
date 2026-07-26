"""Choose the XGBoost quantile hyper-parameters that ``config.yaml`` ships.

The values in ``config.yaml::model`` had no producer: nothing in the repo
emitted ``n_estimators: 169``, ``learning_rate: 0.045``, ``subsample: 0.741`` or
``colsample_bytree: 0.829``, so they could not be re-derived, defended, or
re-run against new data. This script is that producer.

It scores candidates by mean pinball loss — the proper scoring rule for a
quantile model, and the loss the shipped objective already minimises — under
leakage-free K-fold CV on the **train split only**. The test split is never
touched: selecting on it would make every downstream test metric optimistic.
The incumbent configuration is scored under the identical protocol, so the study
records whether tuning actually beat what was already shipped.

Run: ``python -m scripts.tune [--trials N] [--seed S]``
Writes: ``models/tuning_study.json``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from pipeline import FEATURES_FULL, compute_group_means, engineer_features, train_test_positions
from scripts.train_quantile import (
    QUANTILE_ALPHAS,
    _hash_training_data,
    _library_versions,
    _train_quantile_regressor,
    pinball_loss,
)

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# Each range brackets the incumbent value so the study can confirm it or replace
# it; a space that excluded the shipped value could only ever replace it.
# ``max_depth`` stops at 8 because the cohort is 8k rows — deeper trees memorise
# the target encodings rather than the features.
SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "n_estimators": ("int", 50, 600),
    "max_depth": ("int", 2, 8),
    "learning_rate": ("logfloat", 0.01, 0.30),
    "subsample": ("float", 0.50, 1.00),
    "colsample_bytree": ("float", 0.50, 1.00),
    "reg_lambda": ("logfloat", 0.10, 50.0),
}

TUNED_KEYS = tuple(SEARCH_SPACE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None, help="defaults to config model.random_state")
    parser.add_argument("--out", default=str(ROOT / "models" / "tuning_study.json"))
    return parser.parse_args()


def _sample(rng: np.random.Generator) -> dict[str, Any]:
    """One candidate from SEARCH_SPACE. Log-uniform where the scale is multiplicative."""
    candidate: dict[str, Any] = {}
    for name, (kind, low, high) in SEARCH_SPACE.items():
        if kind == "int":
            candidate[name] = int(rng.integers(int(low), int(high) + 1))
        elif kind == "logfloat":
            candidate[name] = round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 4)
        else:
            candidate[name] = round(float(rng.uniform(low, high)), 4)
    return candidate


def training_frame(raw: pd.DataFrame, *, test_size: float, seed: int) -> pd.DataFrame:
    """The trainer's TRAIN rows, and only those.

    Selection must never see the held-out split, so this returns the same
    positions ``scripts.train_quantile`` trains on; scoring candidates on any
    other slice would make every downstream test metric optimistic.
    """
    train_pos, _ = train_test_positions(len(raw), test_size=test_size, random_state=seed)
    return raw.iloc[train_pos].reset_index(drop=True)


def cv_pinball(
    train_raw: pd.DataFrame,
    params: dict[str, Any],
    *,
    seed: int,
    folds: int,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    n_jobs: int,
) -> float:
    """Mean dollar-space pinball loss over the three quantiles, K-fold.

    Group means are recomputed from each fold's train rows, so a validation row
    is never encoded with a mean that saw its own target.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_losses = []
    for tr_idx, va_idx in kf.split(train_raw):
        means = compute_group_means(train_raw.iloc[tr_idx])
        encode = dict(occ_means=means["occ_means"], state_means=means["state_means"])
        fold_train = engineer_features(train_raw.iloc[tr_idx], edu_order, region_map, **encode)
        fold_val = engineer_features(train_raw.iloc[va_idx], edu_order, region_map, **encode)
        model = _train_quantile_regressor(
            fold_train[FEATURES_FULL],
            np.log1p(fold_train["Annual Income"]),
            params={**params, "n_jobs": n_jobs},
            seed=seed,
        )
        preds = np.expm1(model.predict(fold_val[FEATURES_FULL]))
        truth = fold_val["Annual Income"].to_numpy()
        fold_losses.append(np.mean([pinball_loss(truth, preds[:, i], a) for i, a in enumerate(QUANTILE_ALPHAS)]))
    return float(np.mean(fold_losses))


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = cfg["model"]
    seed = args.seed if args.seed is not None else int(model_cfg["random_state"])
    folds = int(model_cfg["cv_folds"])
    n_jobs = int(model_cfg["n_jobs"])
    edu_order = cfg["education_order"]
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}

    data_path = ROOT / cfg["data"]["cleaned"]
    raw = pd.read_csv(data_path)
    train_raw = training_frame(raw, test_size=model_cfg["test_size"], seed=seed)
    logger.info("Tuning on %d train rows, %d-fold CV, %d trials, seed=%d", len(train_raw), folds, args.trials, seed)

    def score(params: dict[str, Any]) -> float:
        return cv_pinball(
            train_raw, params, seed=seed, folds=folds, edu_order=edu_order, region_map=region_map, n_jobs=n_jobs
        )

    incumbent = {key: model_cfg[key] for key in TUNED_KEYS}
    incumbent_loss = score(incumbent)
    logger.info("Incumbent mean pinball: %.2f", incumbent_loss)

    rng = np.random.default_rng(seed)
    trials: list[tuple[int, dict[str, Any], float]] = []
    for index in range(args.trials):
        params = _sample(rng)
        loss = score(params)
        trials.append((index, params, loss))
        logger.info("trial %3d  pinball=%.2f  %s", index, loss, params)

    best_index, best_params, best_loss = min(trials, key=lambda trial: trial[2])
    improvement = incumbent_loss - best_loss
    logger.info("Best trial %d: %.2f (%.2f better than incumbent)", best_index, best_loss, improvement)

    study = {
        "seed": seed,
        "trials": args.trials,
        "cv_folds": folds,
        "n_jobs": n_jobs,
        "objective": "mean dollar-space pinball loss over alphas [0.10, 0.50, 0.90], train split only",
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "library_versions": _library_versions(),
        "data_sha256": _hash_training_data(data_path),
        "incumbent": {"params": incumbent, "cv_pinball": incumbent_loss},
        "best": {"trial": best_index, "params": best_params, "cv_pinball": best_loss},
        "improvement_vs_incumbent": improvement,
        "all_trials": [{"trial": i, "params": p, "cv_pinball": loss} for i, p, loss in trials],
    }
    out = Path(args.out)
    out.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8", newline="\n")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
