"""
tests/test_classifier.py
------------------------
Regression guards for the premium-tier classifier head (Gap 1 Phase 1).

The classifier is trained alongside the quantile regressor by
``scripts/train_quantile.py`` and persisted to
``models/xgb_premium_classifier.ubj``. These tests lock in:

1. The saved artefact exists and loads as an ``XGBClassifier``.
2. ``predict_proba`` returns calibrated [0, 1] probabilities for every
   feature row.
3. ``model_metrics.json`` records the classifier-specific metrics
   (``classifier_objective``, ``classifier_roc_auc``, threshold,
   subgroup AUCs) with sensible values — ROC-AUC must clear the
   no-skill baseline and stay under the perfect-score ceiling so a
   regression like "classifier trained on the wrong label" fails loudly.
4. The FastAPI ``/predict`` endpoint surfaces
   ``p_above_premium_threshold`` and ``premium_threshold`` on every
   response as either a calibrated probability or ``None`` (legacy
   artefacts) — never a wrong type.
5. ``/predict/batch`` preserves the same contract item-for-item so bulk
   callers get the same payload shape as single-row consumers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.main import app
from pipeline import load_classifier

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def base_payload(client: TestClient) -> dict:
    occupations = client.get("/meta").json()["occupations"]
    return {
        "state": "CA",
        "occupation": occupations[0],
        "education_level": "Bachelor's degree",
        "gender": "Female",
        "age": 32,
    }


@pytest.fixture(scope="module")
def metrics() -> dict:
    path = ROOT / "models" / "model_metrics.json"
    with open(path) as f:
        return json.load(f)


class TestClassifierArtefact:
    """The saved classifier is present, loadable, and metrically sane."""

    def test_classifier_artefact_exists(self) -> None:
        path = ROOT / "models" / "xgb_premium_classifier.ubj"
        assert path.exists(), "Classifier artefact missing. Run `python -m scripts.train_quantile` to generate it."

    def test_classifier_loads_as_xgb_classifier(self) -> None:
        path = ROOT / "models" / "xgb_premium_classifier.ubj"
        clf = load_classifier(str(path))
        assert isinstance(clf, XGBClassifier)

    def test_classifier_predict_proba_shape_and_range(self, df_engineered) -> None:
        from pipeline import FEATURES_FULL

        path = ROOT / "models" / "xgb_premium_classifier.ubj"
        clf = load_classifier(str(path))
        X = df_engineered[FEATURES_FULL].head(32)
        proba = clf.predict_proba(X)
        assert proba.shape == (32, 2)
        p_pos = proba[:, 1]
        assert (p_pos >= 0).all() and (p_pos <= 1).all()

    def test_metrics_record_classifier_fields(self, metrics: dict) -> None:
        required = {
            "classifier_objective",
            "classifier_threshold",
            "classifier_roc_auc",
            "classifier_pr_auc",
            "classifier_precision",
            "classifier_recall",
            "classifier_f1",
            "classifier_subgroup_roc_auc",
        }
        missing = required - metrics.keys()
        assert not missing, f"Missing classifier metrics: {missing}"
        assert metrics["classifier_objective"] == "binary:logistic"

    def test_classifier_roc_auc_above_no_skill(self, metrics: dict) -> None:
        # A classifier with ROC-AUC <= 0.55 is no better than coin-flipping
        # on this 40/60 class balance — surfaces a broken training label.
        # >= 0.99 is suspect (leakage / overfitting to the test split).
        auc = metrics["classifier_roc_auc"]
        assert 0.55 <= auc < 0.99, f"Unexpected classifier ROC-AUC: {auc}"

    def test_classifier_subgroup_auc_sane(self, metrics: dict) -> None:
        # Every subgroup in the fairness guardrail must be inside (0.5, 0.95).
        # A subgroup AUC of 0.50 means the classifier is no better than a
        # coin on that slice — the kind of fairness collapse the guard exists
        # to catch.
        subgroup = metrics["classifier_subgroup_roc_auc"]
        assert len(subgroup) > 0, "Subgroup AUCs missing"
        for key, auc in subgroup.items():
            assert 0.50 < auc < 0.95, f"Subgroup {key} AUC out of range: {auc}"


class TestPredictExposesClassifierProbability:
    """API responses carry the classifier probability through every path."""

    def test_predict_single_surfaces_p_above_premium_threshold(self, client: TestClient, base_payload: dict) -> None:
        r = client.post("/predict", json=base_payload)
        assert r.status_code == 200
        data = r.json()
        assert "p_above_premium_threshold" in data
        assert "premium_threshold" in data
        p = data["p_above_premium_threshold"]
        threshold = data["premium_threshold"]
        # Classifier loaded → must be a probability. Not loaded → must be
        # None with a null threshold, and never any other type.
        if p is None:
            assert threshold is None
        else:
            assert 0.0 <= p <= 1.0
            assert isinstance(threshold, int)
            assert threshold >= 100_000

    def test_predict_batch_surfaces_p_above_premium_threshold(self, client: TestClient, base_payload: dict) -> None:
        alt = dict(base_payload)
        alt["gender"] = "Male"
        alt["age"] = 45
        r = client.post("/predict/batch", json={"items": [base_payload, alt]})
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 2
        for item in items:
            assert "p_above_premium_threshold" in item
            assert "premium_threshold" in item
            p = item["p_above_premium_threshold"]
            if p is not None:
                assert 0.0 <= p <= 1.0
