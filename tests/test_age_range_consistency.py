"""The accepted age range must be the model's training support.

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. The support is recorded as ``Age`` min/max in
``models/baseline_stats.json``, so a retrain that moves it turns these red.

Checked where a caller meets the bounds — through the running app on both
serving routes, not through the ``Field`` metadata, which a narrowing added in a
validator or a route guard would never appear in.

The dashboard holds no bounds of its own and is checked statically, by where its
widget's bounds come from. What no static check can see is the range the widget
actually renders, so that is not claimed.
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
from pipeline import training_age_support

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


def _bound_arguments(call: ast.Call) -> list[ast.expr]:
    """The ``min_value``/``max_value`` a widget was given, however they were passed."""
    by_keyword = [kw.value for kw in call.keywords if kw.arg in {"min_value", "max_value"}]
    return by_keyword or list(call.args[1:3])


def test_the_dashboards_age_bounds_come_from_the_shared_derivation():
    """The bounds the age widget is built from must be the values
    ``training_age_support`` returned, not numbers written beside it.

    Checks where the values come from, so passing them positionally, by keyword,
    or to a different widget makes no difference. Relabelling the widget away
    from "Age" does defeat it, and nothing static can see the rendered range —
    the module docstring says so rather than implying otherwise.
    """
    tree = ast.parse((REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    derived = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "training_age_support"
        for element in node.targets
        for target in (element.elts if isinstance(element, ast.Tuple) else [element])
        if isinstance(target, ast.Name)
    }
    age_widgets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WIDGETS
        and re.search(r"\bage\b", _widget_label(node), re.IGNORECASE)
    ]
    assert len(age_widgets) == 1, f"expected one age widget in streamlit_app.py, found {len(age_widgets)}"
    given = _bound_arguments(age_widgets[0])
    assert given, "the age widget declares no bounds at all"
    underived = [ast.unparse(arg) for arg in given if not (isinstance(arg, ast.Name) and arg.id in derived)]
    assert not underived, f"age widget bounded by {underived} instead of training_age_support's result"
