"""The accepted age range must be the model's training support.

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. The support is recorded as ``Age`` min/max in
``models/baseline_stats.json``, so a retrain that moves it turns these red.

Checked where a caller meets the bounds — through the running app on both
serving routes, not through the ``Field`` metadata, which a narrowing added in a
validator or a route guard would never appear in.

The dashboard holds no bounds of its own and is checked statically, by where its
widget's bounds come from. The range it actually renders is beyond any static
check, so that is not claimed here.
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
from pipeline import training_age_support, typical_training_age

REPO_ROOT = Path(__file__).parent.parent
BASELINE_STATS = REPO_ROOT / "models" / "baseline_stats.json"

PAYLOAD = {
    "state": "CA",
    "occupation": "Software Developers",
    "education_level": "Bachelor's degree",
    "gender": "Female",
}


def _schema_bounds() -> tuple[int, int]:
    metadata = PredictRequest.model_fields["age"].metadata
    low = next(c.ge for c in metadata if hasattr(c, "ge"))
    high = next(c.le for c in metadata if hasattr(c, "le"))
    return int(low), int(high)


@pytest.fixture(scope="module")
def client():
    """Own client IP, so these requests draw on their own rate-limit bucket.

    ``/predict/batch`` carries a fixed 10/minute budget keyed by IP; sharing the
    default one would spend another module's allowance and 429 it.
    """
    with TestClient(m.app, client=("10.0.0.99", 5000)) as c:
        yield c


def test_the_support_helper_reports_what_the_artefact_records():
    """Every consumer bounds itself by this helper, so it is pinned to the file
    rather than to whatever the helper happens to return."""
    age = json.loads(BASELINE_STATS.read_text(encoding="utf-8"))["Age"]
    assert training_age_support(BASELINE_STATS) == (int(age["min"]), int(age["max"]))


def test_the_dashboards_opening_age_sits_inside_the_support_and_near_its_centre():
    """A default outside the bounds would break the widget; one at the midpoint
    of the support would open on its 90th percentile."""
    low, high = training_age_support(BASELINE_STATS)
    typical = typical_training_age(BASELINE_STATS)
    assert low <= typical <= high
    assert abs(typical - (low + high) / 2) > 1, "the default drifted back to the midpoint of the support"


def test_schema_bounds_equal_the_training_support():
    assert _schema_bounds() == training_age_support(BASELINE_STATS)


def test_the_published_description_states_the_enforced_range():
    """The bounds and the prose the OpenAPI schema publishes must not disagree."""
    low, high = _schema_bounds()
    description = PredictRequest.model_fields["age"].description or ""
    numbers = [int(n) for n in re.findall(r"\d+", description)]
    assert numbers == [low, high], f"description {description!r} does not state the enforced {low}-{high}"


class TestEveryServingRouteServesExactlyTheSupport:
    """Both routes, both ends. A guard on one route only leaves the other open."""

    def _post(self, client, route, age):
        payload = {**PAYLOAD, "occupation": m.state.occupations[0], "age": age}
        body = payload if route == "/predict" else {"items": [payload]}
        return client.post(route, json=body).status_code

    @pytest.mark.parametrize("route", ["/predict", "/predict/batch"])
    def test_both_ends_of_the_support_are_served(self, client, route):
        for age in training_age_support(BASELINE_STATS):
            assert self._post(client, route, age) == 200, f"{route} refused age {age}, inside the support"

    @pytest.mark.parametrize("route", ["/predict", "/predict/batch"])
    def test_just_outside_the_support_is_refused(self, client, route):
        low, high = training_age_support(BASELINE_STATS)
        for age in (low - 1, high + 1):
            assert self._post(client, route, age) == 422, f"{route} served age {age}, outside the support"


_WIDGETS = {"slider", "number_input", "select_slider"}


def _widget_label(call: ast.Call) -> str:
    positional = call.args[0] if call.args else None
    keyword = next((kw.value for kw in call.keywords if kw.arg == "label"), None)
    node = positional if isinstance(positional, ast.Constant) else keyword
    return str(node.value) if isinstance(node, ast.Constant) else ""


def _widget_argument(call: ast.Call, names: set[str], position: slice) -> list[ast.expr]:
    """A widget's arguments by name, falling back to the positions they occupy."""
    by_keyword = [kw.value for kw in call.keywords if kw.arg in names]
    return by_keyword or list(call.args[position])


def _derives_from(node: ast.expr, function: str, tree: ast.AST) -> bool:
    """True if ``node`` is that function's result, directly or through one name."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function:
        return True
    return isinstance(node, ast.Name) and node.id in _names_assigned_from(tree, function)


def _names_assigned_from(tree: ast.AST, function: str) -> set[str]:
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == function
        for element in node.targets
        for target in (element.elts if isinstance(element, ast.Tuple) else [element])
        if isinstance(target, ast.Name)
    }


def _assignments_of(tree: ast.AST, name: str) -> int:
    return sum(
        isinstance(target, ast.Name) and target.id == name
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for element in node.targets
        for target in (element.elts if isinstance(element, ast.Tuple) else [element])
    )


def test_the_dashboards_age_bounds_come_from_the_shared_derivation():
    """Every number the age widget is built from — both bounds and the value it
    opens on — must come from the artefact, not be written beside it.

    Checks where each argument comes from, so how it is passed makes no
    difference; and each derived name must be assigned once, so the derivation
    cannot be run and then overwritten. What it cannot see: a transform applied
    to the widget's own result, or a label carrying no form of the word age.
    """
    tree = ast.parse((REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    age_widgets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WIDGETS
        and re.search(r"\bage\b", _widget_label(node), re.IGNORECASE)
    ]
    assert len(age_widgets) == 1, f"expected one age widget in streamlit_app.py, found {len(age_widgets)}"
    widget = age_widgets[0]

    for names, position, source in (
        ({"min_value", "max_value"}, slice(1, 3), "training_age_support"),
        ({"value"}, slice(3, 4), "typical_training_age"),
    ):
        given = _widget_argument(widget, names, position)
        assert given, f"the age widget passes no {sorted(names)}"
        underived = [ast.unparse(arg) for arg in given if not _derives_from(arg, source, tree)]
        assert not underived, f"age widget takes {underived} for {sorted(names)} instead of {source}'s result"
        rebound = [name for name in _names_assigned_from(tree, source) if _assignments_of(tree, name) > 1]
        assert not rebound, f"{rebound} reassigned after {source}, so the widget need not see the artefact"
