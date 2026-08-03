"""The accepted age range must be the model's training support.

Accepting an age outside the support serves extrapolation; refusing one inside
it declines an answerable request. The support is recorded as ``Age`` min/max in
``models/baseline_stats.json``, so a retrain that moves it turns these red.

Checked where a caller meets the bounds — through the running app on both
serving routes, not through the ``Field`` metadata, which a narrowing added in a
validator or a route guard would never appear in.

The dashboard is driven rather than parsed: its age control is rendered and the
set of ages it makes selectable is compared against the same artefact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as m
import streamlit_app
from api.schemas import PredictRequest
from pipeline import training_age_support, typical_training_age

REPO_ROOT = Path(__file__).parent.parent
BASELINE_STATS = REPO_ROOT / "models" / "baseline_stats.json"
#: Widget labels naming an age. Word-bounded so "Wage" is not one.
_AGE_LABEL = re.compile(r"\bage\b", re.IGNORECASE)

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
    with TestClient(m.app) as c:
        yield c


def test_the_helpers_read_the_artefact_they_are_given(tmp_path):
    """Against numbers the shipped artefact does not carry, so a helper that
    returned today's values as literals would still be caught."""
    probe = tmp_path / "baseline_stats.json"
    probe.write_text(json.dumps({"Age": {"mean": 47.6, "std": 9.0, "min": 22.0, "max": 71.0}}), encoding="utf-8")
    assert training_age_support(probe) == (22, 71)
    assert typical_training_age(probe) == 48


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


class TestTheDashboardOffersTheWholeSupport:
    """Drives the widget rather than reading the source.

    What a caller can pick follows from the arguments the widget is built with,
    not from how they are written, so they are captured from a render.
    """

    def _render(self, monkeypatch, picks=None, advanced=False):
        """Render the tab; return the age controls it drew and the age it sent.

        ``picks`` chooses what the user moves the slider to. The request is
        intercepted where it leaves the process, so anything applied between the
        widget and the wire — including inside the dashboard's own API helper —
        shows up as a difference between the two. ``advanced`` opens the
        optional-inputs branch, which is otherwise never drawn.
        """
        controls: dict[str, dict] = {}
        sent: dict[str, object] = {}

        def record(label, *args, **kwargs):
            kwargs = {**dict(zip(("min_value", "max_value", "value"), args, strict=False)), **kwargs}
            controls[label] = kwargs
            return picks(kwargs) if picks and label == "Age" else kwargs.get("value")

        monkeypatch.setattr(streamlit_app.st, "slider", record)
        monkeypatch.setattr(streamlit_app.st, "number_input", record)
        monkeypatch.setattr(streamlit_app.st, "checkbox", lambda *_a, **_k: advanced)
        monkeypatch.setattr(streamlit_app.st, "button", lambda *_a, **_k: picks is not None)

        def intercept(_url, json=None, **_kwargs):
            sent.update(json or {})
            raise httpx.ConnectError("intercepted before the wire")

        monkeypatch.setattr(httpx, "post", intercept)
        streamlit_app.tab_predictor(streamlit_app.load_data())
        return controls, sent.get("age")

    def _age_control(self, controls):
        matches = [kwargs for label, kwargs in controls.items() if _AGE_LABEL.search(label)]
        assert len(matches) == 1, f"expected one age control, found {len(matches)} in {sorted(controls)}"
        return matches[0]

    @pytest.mark.parametrize("advanced", [False, True], ids=["basic", "advanced-inputs-open"])
    def test_every_age_in_the_support_can_be_selected(self, monkeypatch, advanced):
        """Reachability, not just the endpoints: a step would leave the top of
        the support unselectable while both bounds still read correctly."""
        controls, _ = self._render(monkeypatch, advanced=advanced)
        widget = self._age_control(controls)
        low, high = training_age_support(BASELINE_STATS)
        selectable = set(range(widget["min_value"], widget["max_value"] + 1, widget.get("step") or 1))
        assert not widget.get("disabled"), "the age control is disabled, so no age is selectable"
        assert selectable == set(range(low, high + 1))

    def test_the_widget_follows_the_artefact_rather_than_repeating_its_values(self, monkeypatch):
        """Against numbers the shipped artefact does not carry. Every other check
        here compares the widget to that artefact, so literals equal to today's
        values would satisfy all of them."""
        monkeypatch.setattr(streamlit_app, "training_age_support", lambda _path: (22, 71))
        monkeypatch.setattr(streamlit_app, "typical_training_age", lambda _path: 48)
        streamlit_app.get_age_range.clear()  # the derivation is cached per session
        try:
            controls, _ = self._render(monkeypatch)
        finally:
            streamlit_app.get_age_range.clear()
        widget = self._age_control(controls)
        assert (widget["min_value"], widget["max_value"], widget["value"]) == (22, 71, 48)

    def test_the_widget_opens_on_the_mean_of_the_training_ages(self, monkeypatch):
        controls, _ = self._render(monkeypatch)
        assert self._age_control(controls)["value"] == typical_training_age(BASELINE_STATS)

    @pytest.mark.parametrize("advanced", [False, True], ids=["basic", "advanced-inputs-open"])
    @pytest.mark.parametrize("end", ["min_value", "max_value"])
    def test_an_age_picked_at_either_end_reaches_the_request_unchanged(self, monkeypatch, end, advanced):
        """A clamp between the widget and the request is invisible to the widget,
        and the optional branch is a second place one could sit."""
        controls, age_sent = self._render(monkeypatch, picks=lambda kwargs: kwargs[end], advanced=advanced)
        assert age_sent == self._age_control(controls)[end], f"age was altered before the request ({end})"
