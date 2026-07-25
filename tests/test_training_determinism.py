"""Training must be reproducible on a machine of any size.

XGBoost's ``hist`` tree method sums gradient histograms in thread-partition
order, so the fitted bytes depend on how many threads ran. A seed and a pinned
library set are not enough: at ``n_jobs=-1`` the same code and data yield
different models on a 2-core runner and a 24-core workstation, along with
different published metrics. The thread count is therefore a configured input.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.train_quantile as tq
from pipeline import FEATURES_FULL, compute_group_means, engineer_features, train_test_positions


def test_thread_count_is_pinned_to_one(cfg):
    """``n_jobs`` must be an explicit single thread, never the -1 core count."""
    assert cfg["model"]["n_jobs"] == 1


def test_trainers_take_the_thread_count_from_config():
    """Neither trainer may hardcode a thread count past the configured one."""
    source = Path(tq.__file__).read_text(encoding="utf-8")
    assert "n_jobs=-1" not in source, "an all-cores default makes the fitted bytes machine-specific"
    assert source.count('"n_jobs": n_jobs') == 2, "both the regressor and classifier params must carry n_jobs"


def _production_params(cfg: dict, n_jobs: int) -> dict:
    """The shipped hyperparameters — thread sensitivity depends on tree shape,
    so a reduced stand-in would not exercise the real training path."""
    model_cfg = cfg["model"]
    return {
        "n_estimators": model_cfg["n_estimators"],
        "max_depth": model_cfg["max_depth"],
        "learning_rate": model_cfg["learning_rate"],
        "subsample": model_cfg["subsample"],
        "colsample_bytree": model_cfg["colsample_bytree"],
        "reg_lambda": model_cfg["reg_lambda"],
        "n_jobs": n_jobs,
    }


def _fit_digest(x_train: pd.DataFrame, y_train: pd.Series, params: dict, seed: int) -> str:
    model = tq._train_quantile_regressor(x_train, y_train, params=params, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.ubj"
        model.save_model(str(path))
        return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_matrix(df: pd.DataFrame, cfg: dict, edu_order: dict, region_map: dict):
    train_pos, _ = train_test_positions(
        len(df), test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"]
    )
    raw_train = df.iloc[train_pos]
    means = compute_group_means(raw_train)
    encoded = engineer_features(
        raw_train, edu_order, region_map, occ_means=means["occ_means"], state_means=means["state_means"]
    )
    return encoded[FEATURES_FULL], np.log1p(encoded["Annual Income"])


def test_configured_fit_is_bit_identical_across_runs(df, cfg, edu_order, region_map):
    """Two fits under the shipped settings must produce identical bytes.

    This is the property the reproducibility claim rests on: it is what lets a
    recorded artefact digest be reproduced rather than merely asserted.
    """
    x_train, y_train = _training_matrix(df, cfg, edu_order, region_map)
    params = _production_params(cfg, cfg["model"]["n_jobs"])
    seed = cfg["model"]["random_state"]

    assert _fit_digest(x_train, y_train, params, seed) == _fit_digest(x_train, y_train, params, seed)
