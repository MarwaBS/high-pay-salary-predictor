"""
Regression guard: the accepted age range must equal the model's training
support — recorded as ``Age`` min/max in ``models/baseline_stats.json`` — and
must agree across:

1. the ``ge``/``le`` bounds on ``PredictRequest.age`` (``api/schemas.py``)
2. the ``description`` those bounds ship with, published in the OpenAPI schema
3. the dashboard's Age slider (``streamlit_app.py``)

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. Each site holds its own copy of the range,
so a retrain that moves the support would leave them silently wrong. Pinning
every copy to the artefact makes that a red test instead.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from api.schemas import PredictRequest

REPO_ROOT = Path(__file__).parent.parent


def _training_support() -> tuple[int, int]:
    """Age min/max the model was fitted on, from the drift baseline artefact."""
    stats = json.loads((REPO_ROOT / "models" / "baseline_stats.json").read_text(encoding="utf-8"))
    age = stats["Age"]
    return int(age["min"]), int(age["max"])


def _schema_bounds() -> tuple[int, int]:
    """The ``ge``/``le`` actually enforced on ``PredictRequest.age``."""
    metadata = PredictRequest.model_fields["age"].metadata
    low = next(c.ge for c in metadata if hasattr(c, "ge"))
    high = next(c.le for c in metadata if hasattr(c, "le"))
    return int(low), int(high)


def _slider_bounds() -> tuple[int, int]:
    """``min_value``/``max_value`` of the dashboard's Age slider, read statically.

    Parsed rather than imported: ``requirements-dashboard.txt`` is a separate
    dependency set, so ``streamlit_app`` is not importable in the API test env.
    """
    tree = ast.parse((REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8"))
    found = [
        {kw.arg: kw.value.value for kw in node.keywords if isinstance(kw.value, ast.Constant)}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "slider"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Age"
    ]
    assert len(found) == 1, f"expected exactly one Age slider in streamlit_app.py, found {len(found)}"
    return int(found[0]["min_value"]), int(found[0]["max_value"])


def test_schema_bounds_equal_the_training_support():
    assert _schema_bounds() == _training_support()


def test_the_published_description_states_the_enforced_range():
    """The bounds and the prose a caller reads must not disagree."""
    low, high = _schema_bounds()
    description = PredictRequest.model_fields["age"].description or ""
    numbers = [int(n) for n in re.findall(r"\d+", description)]
    assert numbers == [low, high], f"description {description!r} does not state the enforced {low}-{high}"


def test_the_dashboard_slider_offers_the_enforced_range():
    assert _slider_bounds() == _schema_bounds()
