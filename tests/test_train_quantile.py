"""
End-to-end trainer test for ``scripts/train_quantile.py``.

The trainer is the critical artefact-producing path: the API, dashboard, and
every model test load what it writes. CI *runs* it before pytest, but nothing
asserted anything about it, so under coverage it read 0% — a 205-statement
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

import pytest
import yaml

import scripts.train_quantile as tq

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
