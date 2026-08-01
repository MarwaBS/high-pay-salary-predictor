"""The accepted age range must be the model's training support.

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. The support is recorded as ``Age`` min/max in
``models/baseline_stats.json``, so a retrain that moves it turns these red.

Checked where a caller meets the bounds — through the running app on both
serving routes, not through the ``Field`` metadata, which a narrowing added in a
validator or a route guard would never appear in.

The dashboard is driven rather than parsed: its Age slider is rendered and the
arguments it was built with are compared against the same artefact.
"""

from __future__ import annotations

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


def test_the_typical_age_helper_reports_the_mean_of_the_training_ages():
    """What a UI opens on. The midpoint of the support would be its 90th
    percentile, so the first thing a visitor sees would be an atypical profile."""
    age = json.loads(BASELINE_STATS.read_text(encoding="utf-8"))["Age"]
    assert typical_training_age(BASELINE_STATS) == round(float(age["mean"]))


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


class TestTheDashboardWidgetOffersTheSupport:
    """Drives the real widget rather than reading the source.

    Every static version of this check was defeated by a spelling it did not
    anticipate — a positional argument, a rebound name, a different widget. What
    the widget is actually constructed with is not a matter of syntax, so it is
    captured from a run instead.
    """

    def test_ci_installs_what_these_tests_need_to_run(self):
        """The renders below skip without streamlit, which would read as green.

        ``requirements.txt`` is what CI installs, so declaring it there is what
        keeps that skip out of the pipeline.
        """
        declared = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("streamlit") for line in declared), "CI would skip the dashboard render tests"

    def _render(self, monkeypatch, picks=None):
        """Render the tab, returning the Age slider's arguments and the age sent.

        ``picks`` chooses what the user moves the slider to; the payload is
        intercepted at the HTTP boundary, so anything applied between the widget
        and the request shows up as a difference between the two.
        """
        streamlit_app = pytest.importorskip("streamlit_app")
        captured: dict[str, dict] = {}
        sent: dict[str, object] = {}

        def record(label, *args, **kwargs):
            kwargs = {**dict(zip(("min_value", "max_value", "value"), args, strict=False)), **kwargs}
            captured[label] = kwargs
            return picks(kwargs) if picks and label == "Age" else kwargs.get("value")

        monkeypatch.setattr(streamlit_app.st, "slider", record)
        monkeypatch.setattr(streamlit_app.st, "button", lambda *a, **k: picks is not None)
        monkeypatch.setattr(streamlit_app, "_call_predict_api", lambda payload: sent.update(payload) or None)
        streamlit_app.tab_predictor(streamlit_app.load_data())
        assert "Age" in captured, f"no Age slider rendered; saw {sorted(captured)}"
        return captured["Age"], sent.get("age")

    def test_the_widget_spans_exactly_the_training_support(self, monkeypatch):
        widget, _ = self._render(monkeypatch)
        low, high = training_age_support(BASELINE_STATS)
        assert (widget["min_value"], widget["max_value"]) == (low, high)

    def test_the_widget_opens_on_the_mean_of_the_training_ages(self, monkeypatch):
        widget, _ = self._render(monkeypatch)
        assert widget["value"] == typical_training_age(BASELINE_STATS)

    @pytest.mark.parametrize("end", ["min_value", "max_value"])
    def test_an_age_picked_at_either_end_reaches_the_request_unchanged(self, monkeypatch, end):
        """A clamp between the widget and the request is invisible to the widget."""
        widget, age_sent = self._render(monkeypatch, picks=lambda kwargs: kwargs[end])
        assert age_sent == widget[end], f"age was altered between the slider and the request ({end})"
