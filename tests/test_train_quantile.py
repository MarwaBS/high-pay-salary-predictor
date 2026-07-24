"""
End-to-end trainer test for ``scripts/train_quantile.py``.

The trainer is the critical artefact-producing path: the API, dashboard, and
every model test load what it writes, but nothing asserted anything about the
trainer's own code, so under coverage it read 0% — a 205-statement
blind spot on the most important script in the repo.

This test runs ``main()`` end-to-end against the real dataset but into a
throwaically-isolated output tree (``ROOT`` monkeypatched to ``tmp_path``) so it
never touches the committed ``models/`` artefacts, with the stability sweep cut
to a single seed for speed. It then asserts the artefacts exist and the metrics
are structurally sane — locking the trainer's contract, not just its coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import scripts.train_quantile as tq
from pipeline import FEATURES_FULL, compute_group_means, engineer_features

REPO_ROOT = Path(tq.__file__).resolve().parent.parent


@pytest.fixture
def trained_in_tmp(tmp_path, monkeypatch):
    """Run the trainer once into an isolated tmp tree; return its models dir."""
    # Real config, with the data path pinned to an absolute location (so it
    # resolves regardless of the monkeypatched ROOT) and the stability sweep
    # reduced to one seed to keep the test fast.
    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["cleaned"] = str((REPO_ROOT / cfg["data"]["cleaned"]).resolve())
    cfg["model"]["stability_seeds"] = [11]

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    # ROOT drives every *output* path (model/classifier/metrics + the
    # hardcoded baseline_stats.json), so redirecting it isolates the run.
    monkeypatch.setattr(tq, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_quantile.py", "--config", str(cfg_path)])

    tq.main()
    return tmp_path / "models"


def test_trainer_writes_all_artifacts(trained_in_tmp):
    models = trained_in_tmp
    for name in (
        "xgb_salary_model.ubj",
        "xgb_premium_classifier.ubj",
        "model_metrics.json",
        "baseline_stats.json",
        "group_means.json",
        "feature_names.json",
    ):
        artefact = models / name
        assert artefact.exists(), f"trainer did not write {name}"
        assert artefact.stat().st_size > 0, f"{name} is empty"


def test_trainer_metrics_are_sane(trained_in_tmp):
    metrics = json.loads((trained_in_tmp / "model_metrics.json").read_text())

    # Provenance + core regression metrics.
    assert metrics["model_version"], "empty model_version provenance string"
    assert metrics["objective"] == "reg:quantileerror"
    assert -1.0 < metrics["r2"] <= 1.0
    assert 0.0 < metrics["quantile_coverage_80"] <= 1.0

    # Classifier head must beat a coin flip and carry calibration metrics.
    assert metrics["classifier_objective"] == "binary:logistic"
    assert metrics["classifier_roc_auc"] > 0.5
    assert 0.0 <= metrics["classifier_brier"] <= 1.0

    # Stability sweep ran and recorded mean±std for the headline metrics.
    assert metrics["stability_seeds"] == [11]
    for key in ("stability_p50_r2_mean", "stability_clf_roc_auc_mean"):
        assert key in metrics


def test_build_model_version_is_composite():
    """Provenance string is ``{service}+{git_sha}.{data_sha}`` — the operator
    recovery primitive. Lock its shape directly (fast, no training)."""
    data_path = REPO_ROOT / "Data" / "cleaned_high_pay_data.csv"
    version = tq.build_model_version(data_path)
    assert "+" in version and "." in version.split("+", 1)[1]


def test_trainer_test_split_matches_shared_primitive():
    """The dashboard scores its residual plot on the test rows selected by
    ``pipeline.train_test_positions``. The trainer must derive its own test set the
    same way, or the dashboard silently reports residuals on a train-contaminated
    split. Red if ``_prepare_split`` stops using the shared split primitive (e.g.
    someone adds stratification to only one of the two)."""
    from pipeline import train_test_positions

    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    df_raw = pd.read_csv(REPO_ROOT / cfg["data"]["cleaned"])
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}
    seed = cfg["model"]["random_state"]
    test_size = cfg["model"]["test_size"]

    _, df_test, _, _ = tq._prepare_split(
        df_raw, seed=seed, test_size=test_size, edu_order=cfg["education_order"], region_map=region_map
    )
    _, test_pos = train_test_positions(len(df_raw), test_size=test_size, random_state=seed)
    assert list(df_test.index) == list(df_raw.iloc[test_pos].index), (
        "trainer test set diverged from the shared split primitive the dashboard uses"
    )


# ── Regression: CV must not leak the validation target ──────────────────────────

_TINY_PARAMS = {
    "n_estimators": 40,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
}
_EDU = {"Bachelor's degree": 1}
_REGION = {"CA": "West"}


def _leak_frame(seed: int = 0) -> pd.DataFrame:
    """Frame whose only signal is a high-cardinality target encoding.

    120 occupations × 2 rows: the global ``Occ_Mean_Income`` of a row carries
    50% of that row's own target, so encoding once over all of train and then
    folding leaks. Every other feature is target-independent, so an honest
    per-fold encoding has nothing left to predict.
    """
    rng = np.random.default_rng(seed)
    rows = [
        {
            "Age": 40,
            "Employment": 1000.0,
            "Location Quotient": 1.0,
            "Jobs per 1000": 5.0,
            "Hourly Mean": 75.0,
            "Education Level": "Bachelor's degree",
            "Gender": "Male" if rng.random() < 0.5 else "Female",
            "State Abbreviation": "CA",
            "Occupation": f"occ_{occ}",
            "Annual Income": float(150_000 + rng.normal(0, 40_000)),
        }
        for occ in range(120)
        for _ in range(2)
    ]
    return pd.DataFrame(rows)


def _leaky_cv_reference(df_raw: pd.DataFrame) -> float:
    """Pre-fix behaviour: encode once from all of train, then fold over it."""
    gm = compute_group_means(df_raw)
    df = engineer_features(df_raw, _EDU, _REGION, occ_means=gm["occ_means"], state_means=gm["state_means"]).reset_index(
        drop=True
    )
    X, y = df[FEATURES_FULL], df["Annual Income"]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = [
        r2_score(
            y.iloc[va],
            np.expm1(
                tq._train_quantile_regressor(X.iloc[tr], np.log1p(y.iloc[tr]), params=_TINY_PARAMS, seed=0).predict(
                    X.iloc[va]
                )
            )[:, 1],
        )
        for tr, va in kf.split(X)
    ]
    return float(np.mean(scores))


def test_cross_val_r2_is_leakage_free():
    """Per-fold target encoding must not leak the validation target.

    On a frame whose only signal is a high-cardinality encoding, encoding once
    over all of train (the pre-fix path) inflates CV R²; the honest per-fold
    encoding scores far lower. Red if ``_cross_val_r2`` regresses to global
    encoding.
    """
    df = _leak_frame()
    honest, _ = tq._cross_val_r2(df, seed=0, n_splits=5, edu_order=_EDU, region_map=_REGION, params=_TINY_PARAMS)
    leaky = _leaky_cv_reference(df)
    assert leaky - honest > 0.05, (
        f"no leakage separation (leaky={leaky:.4f}, honest={honest:.4f}) — "
        "per-fold encoding may have regressed to global encoding"
    )
    assert honest < 0.10, f"honest CV R² {honest:.4f} too high for a pure-noise frame — encoding still leaks"
