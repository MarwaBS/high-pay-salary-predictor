"""
End-to-end trainer test for ``scripts/train_quantile.py``.

The trainer is the critical artefact-producing path: the API, dashboard, and
every model test load what it writes. This test locks the trainer's own
contract end-to-end.

This test runs ``main()`` end-to-end against the real dataset but into a
throwaway-isolated output tree (``ROOT`` monkeypatched to ``tmp_path``) so it
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
from pydantic import ValidationError
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import scripts.train_quantile as tq
from config_schema import ProjectConfig
from pipeline import FEATURES_FULL, compute_group_means, engineer_features

REPO_ROOT = Path(tq.__file__).resolve().parent.parent


#: Parameters each trainer was actually called with during ``trained_in_tmp``.
#: Populated by the fixture so a test can assert on what ``main()`` handed the
#: trainers, not merely on what the source text says.
_TRAINER_CALLS: dict[str, list[dict]] = {"regressor": [], "classifier": []}


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
    # ROOT drives every *output* path, so redirecting it isolates the run.
    monkeypatch.setattr(tq, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_quantile.py", "--config", str(cfg_path)])

    # Record what each trainer is actually called with, then delegate.
    _TRAINER_CALLS["regressor"].clear()
    _TRAINER_CALLS["classifier"].clear()
    for name, bucket in (("_train_quantile_regressor", "regressor"), ("_train_premium_classifier", "classifier")):
        real = getattr(tq, name)

        def recorder(*args, _real=real, _bucket=bucket, **kwargs):
            fitted = _real(*args, **kwargs)
            # Record what the estimator was actually built with, not just what
            # was passed in: a trainer that overrode the value internally would
            # leave the incoming params untouched.
            _TRAINER_CALLS[_bucket].append({"passed": dict(kwargs.get("params", {})), "fitted": fitted.get_params()})
            return fitted

        monkeypatch.setattr(tq, name, recorder)

    tq.main()
    return tmp_path / "models"


def test_trainers_are_called_with_the_configured_thread_count(trained_in_tmp):
    """Both heads must be fitted with the thread count config declares.

    The fitted bytes depend on it, so a hardcoded value anywhere between the
    config and the trainers silently restores machine-specific artefacts. A
    source-text check cannot see that; this asserts on the calls themselves.
    """
    with open(REPO_ROOT / "config.yaml") as f:
        configured = yaml.safe_load(f)["model"]["n_jobs"]

    assert _TRAINER_CALLS["regressor"], "the regressor was never called"
    assert _TRAINER_CALLS["classifier"], "the classifier was never called"
    for head, calls in _TRAINER_CALLS.items():
        for call in calls:
            assert call["passed"].get("n_jobs") == configured, (
                f"{head} was handed n_jobs={call['passed'].get('n_jobs')!r}"
            )
            assert call["fitted"].get("n_jobs") == configured, (
                f"{head} was fitted with n_jobs={call['fitted'].get('n_jobs')!r} despite the configured value"
            )

    # The recorded provenance must agree with what was actually used.
    metrics = json.loads((trained_in_tmp / "model_metrics.json").read_text())
    assert metrics["hyperparameters"]["n_jobs"] == configured


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
    """Leaky reference: encode once over all of train, then fold over it."""
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
    over all of train (the leaky path) inflates CV R²; the honest per-fold
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


def _config_with(tmp_path, drop=(), overrides=None):
    """Real config with keys dropped or values overridden, written to tmp."""
    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["cleaned"] = str((REPO_ROOT / cfg["data"]["cleaned"]).resolve())
    for key in drop:
        del cfg["model"][key]
    cfg["model"].update(overrides or {})
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.mark.parametrize(
    ("drop", "overrides"),
    [(["cv_folds"], None), (["premium_threshold"], None), ((), {"test_size": 0.9})],
    ids=["required-key-absent", "half-declared-classifier", "value-out-of-range"],
)
def test_a_config_the_api_would_refuse_never_reaches_a_model_file(tmp_path, monkeypatch, drop, overrides):
    """Only the schema knows a value is out of range, so this expects
    ValidationError specifically: a KeyError would mean the trainer tripped over
    the key on its own and the validation call proved nothing.
    """
    cfg_path = _config_with(tmp_path, drop, overrides)
    monkeypatch.setattr(tq, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_quantile.py", "--config", str(cfg_path)])
    with pytest.raises(ValidationError):
        tq.main()
    assert not list(tmp_path.glob("**/*.ubj")), "training ran before the config was rejected"


def test_a_schema_valid_config_missing_a_key_the_trainer_reads_still_stops(tmp_path, monkeypatch):
    """Dropping the classifier makes ``premium_threshold`` optional to the schema,
    so nothing but the trainer's own unguarded read can stop it here."""
    cfg_path = _config_with(tmp_path, drop=["premium_threshold", "classifier_path"])
    monkeypatch.setattr(tq, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_quantile.py", "--config", str(cfg_path)])
    with pytest.raises(KeyError):
        tq.main()
    assert not list(tmp_path.glob("**/*.ubj")), "an unconfigured threshold reached a shipped model"


def test_a_config_without_a_drift_block_is_refused():
    """Giving the block a default would restore the buried window this replaced."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    del cfg["drift"]
    with pytest.raises(ValidationError):
        ProjectConfig(**cfg)


def test_a_mistyped_drift_knob_is_refused(tmp_path):
    """Silently ignored, the monitor would run on the default the typo replaced."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    cfg["drift"]["windwo"] = 200
    with pytest.raises(ValidationError):
        ProjectConfig(**cfg)


def test_a_non_positive_drift_window_is_refused(tmp_path):
    """A zero window caps the dropped-write backlog at zero: a permanent all-clear."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    cfg["drift"]["window"] = 0
    with pytest.raises(ValidationError):
        ProjectConfig(**cfg)


#: The test's own statement of the floor, deliberately not read from
#: ``tq.MIN_SUBGROUP_SIZE``: sharing it would let the implementation move the
#: floor and carry this expectation along with it.
_MEANINGFUL_SLICE = 30


def _test_split_slices() -> tuple[dict[str, int], dict[str, bool], dict]:
    """Per-subgroup test-row counts and two-class flags, from the shared split."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    df_raw = pd.read_csv(REPO_ROOT / cfg["data"]["cleaned"])
    region_map = {s: r for r, states in cfg["regions"].items() for s in states}
    _, df_test, _, _ = tq._prepare_split(
        df_raw,
        seed=cfg["model"]["random_state"],
        test_size=cfg["model"]["test_size"],
        edu_order=cfg["education_order"],
        region_map=region_map,
    )
    premium = df_test["Annual Income"] >= cfg["model"]["premium_threshold"]
    sizes, two_class = {}, {}
    for col in ("Gender", "Region"):
        for val in sorted(df_test[col].dropna().unique()):
            mask = df_test[col] == val
            sizes[f"{col}={val}"] = int(mask.to_numpy().sum())
            two_class[f"{col}={val}"] = bool(premium[mask].nunique() == 2)
    return sizes, two_class, cfg


def test_every_slice_big_enough_to_score_is_in_the_published_fairness_table():
    """Set equality in both directions, on BOTH published tables. Raising the
    trainer's floor drops a real subgroup — even after a retrain, which is when
    it would otherwise pass unnoticed — and lowering it publishes a slice whose
    rate is mostly its own sampling error. The AUC table carries one further
    exclusion, single-class slices, so that is computed rather than waived."""
    sizes, two_class, cfg = _test_split_slices()
    expected = {name for name, n in sizes.items() if n >= _MEANINGFUL_SLICE}
    metrics = json.loads((REPO_ROOT / cfg["model"]["metrics_path"]).read_text())
    assert set(metrics["subgroup_coverage_80"]) == expected
    assert set(metrics["classifier_subgroup_roc_auc"]) == {n for n in expected if two_class[n]}


def test_the_trainer_publishes_at_exactly_the_floor_the_policy_states():
    """A coverage rate near the 0.80 target carries a 95% sampling interval of
    about ±0.14 at n=30. Equality, not a minimum: a lower floor publishes a rate
    that is mostly its own sampling noise, and a higher one hides a real
    subgroup — including the worst-covered one, which is the one that matters.
    """
    assert tq.MIN_SUBGROUP_SIZE == _MEANINGFUL_SLICE


def test_the_subgroup_gates_read_the_constant_rather_than_a_literal(tmp_path, monkeypatch):
    """Trains with the floor raised past the smallest slice and checks the table
    shrinks to match. Gates comparing against a literal would publish the same
    table whatever the constant says, leaving the pin above asserting nothing.
    """
    sizes, two_class, _ = _test_split_slices()
    raised = max(sizes.values()) // 2
    expected = {name for name, n in sizes.items() if n >= raised}
    assert expected < set(sizes), "floor is not high enough to exclude anything — the check would be vacuous"

    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    cfg["data"]["cleaned"] = str((REPO_ROOT / cfg["data"]["cleaned"]).resolve())
    cfg["model"]["stability_seeds"] = [11]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tq, "ROOT", tmp_path)
    monkeypatch.setattr(tq, "MIN_SUBGROUP_SIZE", raised)
    monkeypatch.setattr(sys, "argv", ["train_quantile.py", "--config", str(cfg_path)])
    tq.main()

    published = json.loads((tmp_path / "models" / "model_metrics.json").read_text())
    assert set(published["subgroup_coverage_80"]) == expected
    assert set(published["classifier_subgroup_roc_auc"]) == {n for n in expected if two_class[n]}
