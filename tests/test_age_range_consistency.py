"""The accepted age range must be the model's training support.

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. The support is recorded as ``Age`` min/max in
``models/baseline_stats.json``, so a retrain that moves it turns these red.

Checked where a caller meets the bounds — through the running app, not only
through the ``Field`` metadata, which a narrowing added in a validator or a
route guard would never appear in.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as m
from api.schemas import PredictRequest

REPO_ROOT = Path(__file__).parent.parent

PAYLOAD = {
    "state": "CA",
    "occupation": "Software Developers",
    "education_level": "Bachelor's degree",
    "gender": "Female",
}


def _training_support() -> tuple[int, int]:
    stats = json.loads((REPO_ROOT / "models" / "baseline_stats.json").read_text(encoding="utf-8"))
    age = stats["Age"]
    return int(age["min"]), int(age["max"])


def _schema_bounds() -> tuple[int, int]:
    metadata = PredictRequest.model_fields["age"].metadata
    low = next(c.ge for c in metadata if hasattr(c, "ge"))
    high = next(c.le for c in metadata if hasattr(c, "le"))
    return int(low), int(high)


@pytest.fixture(scope="module")
def client():
    with TestClient(m.app) as c:
        yield c


def test_schema_bounds_equal_the_training_support():
    assert _schema_bounds() == _training_support()


def test_the_published_description_states_the_enforced_range():
    """The bounds and the prose the OpenAPI schema publishes must not disagree."""
    low, high = _schema_bounds()
    description = PredictRequest.model_fields["age"].description or ""
    numbers = [int(n) for n in re.findall(r"\d+", description)]
    assert numbers == [low, high], f"description {description!r} does not state the enforced {low}-{high}"


class TestTheServedRangeIsTheTrainingSupport:
    """Driven through the app, so a narrowing outside the ``Field`` bounds fails."""

    def test_both_ends_of_the_support_are_served(self, client):
        low, high = _training_support()
        payload = {**PAYLOAD, "occupation": m.state.occupations[0]}
        for age in (low, high):
            r = client.post("/predict", json={**payload, "age": age})
            assert r.status_code == 200, f"age {age} is inside the training support but was refused"

    def test_just_outside_the_support_is_refused_at_both_ends(self, client):
        low, high = _training_support()
        payload = {**PAYLOAD, "occupation": m.state.occupations[0]}
        for age in (low - 1, high + 1):
            r = client.post("/predict", json={**payload, "age": age})
            assert r.status_code == 422, f"age {age} is outside the training support but was served"


def test_the_dashboard_holds_no_second_copy_of_the_bounds():
    """``streamlit_app`` must derive its Age slider from the artefact.

    Only the absence of a literal is checkable statically — the rendered widget
    is not. Imported bounds are impossible here: ``requirements-dashboard.txt``
    ships no pydantic, so the dashboard cannot read ``api.schemas``.
    """
    tree = ast.parse((REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    literal_bounds = [
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"slider", "number_input", "select_slider"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and re.search(r"\bage\b", str(node.args[0].value), re.IGNORECASE)
        for kw in node.keywords
        if kw.arg in {"min_value", "max_value"} and isinstance(kw.value, ast.Constant)
    ]
    assert not literal_bounds, f"age widget hardcodes {literal_bounds} instead of reading baseline_stats.json"
