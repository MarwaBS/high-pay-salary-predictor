"""Pin the served interval to the committed conformal margin.

The margin is applied at the route, so /predict and /predict/batch each have a
call site that can drop it, and dropping one serves the raw quantiles — a
narrower interval whose coverage is not the calibrated figure the metrics record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from pipeline import load_conformal_delta

ARTEFACT = Path(__file__).parent.parent / "models" / "conformal_delta.json"
# Probed at two magnitudes. One probe cannot separate a route that drops small
# margins from one that drops large ones: rounding the shipped 0.010 to one
# decimal serves 0.0, while a 0.5 probe survives the same rounding intact.
_PROBE_DELTA = 0.5
_LOWER_BOUNDS = ("predicted_p10", "prediction_interval_low")
_UPPER_BOUNDS = ("predicted_p90", "prediction_interval_high")
_POINT_ESTIMATES = ("predicted_p50", "predicted_salary")
# The model predicts in float32 and expm1 amplifies the log-space error, so
# reconstructing one bound from another lands a few times float32 eps (1.19e-07)
# away — still three orders tighter than the ~1e-2 a dropped margin produces.
_REL_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def payload(client):
    occupations = client.get("/meta").json()["occupations"]
    return {
        "state": "CA",
        "occupation": occupations[0],
        "education_level": "Bachelor's degree",
        "gender": "Female",
        "age": 32,
    }


def _committed_delta() -> float:
    return float(json.loads(ARTEFACT.read_text(encoding="utf-8"))["conformal_delta"])


def _post(client, url, body, monkeypatch, *, delta, tag):
    """Send one request with ``delta`` in effect, on a cache namespace of its own.

    Both routes answer from cache before reaching the model, so without the
    private namespace a repeated payload returns the previous delta's response.
    """
    monkeypatch.setattr(api_main.state, "conformal_delta", delta)
    monkeypatch.setattr(api_main.cache, "version", f"conformal-serving-{tag}")
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_margin_applied(raw, wide, delta):
    """Each bound must equal the raw one shifted by exactly ``delta`` in log space."""
    for field in _LOWER_BOUNDS:
        expected = math.expm1(math.log1p(raw[field]) - delta)
        assert wide[field] == pytest.approx(expected, rel=_REL_TOLERANCE), field
    for field in _UPPER_BOUNDS:
        expected = math.expm1(math.log1p(raw[field]) + delta)
        assert wide[field] == pytest.approx(expected, rel=_REL_TOLERANCE), field
    for field in _POINT_ESTIMATES:
        assert wide[field] == pytest.approx(raw[field]), f"{field} moved"


def _assert_route_carries_margin(client, url, body, monkeypatch, unwrap):
    raw = unwrap(_post(client, url, body, monkeypatch, delta=0.0, tag=f"{url}-raw"))
    for delta in (_committed_delta(), _PROBE_DELTA):
        wide = unwrap(_post(client, url, body, monkeypatch, delta=delta, tag=f"{url}-{delta}"))
        _assert_margin_applied(raw, wide, delta)


def test_startup_loads_the_committed_margin(client):
    """Compared against the file, not ``state``: state compared to itself agrees at zero."""
    committed = _committed_delta()
    assert committed > 0.0, "a zero margin would make every assertion below vacuous"
    assert api_main.state.conformal_delta == pytest.approx(committed)


def test_loader_returns_the_margin_it_was_given(tmp_path):
    """A loader ignoring its file would match today's artefact and rot at the next retrain."""
    probe = tmp_path / "conformal_delta.json"
    probe.write_text(json.dumps({"conformal_delta": 0.1234}), encoding="utf-8")
    assert load_conformal_delta(str(probe)) == pytest.approx(0.1234)


def test_predict_interval_carries_the_loaded_margin(client, payload, monkeypatch):
    """/predict must pass the loaded margin through rather than a literal."""
    _assert_route_carries_margin(client, "/predict", payload, monkeypatch, lambda body: body)


def test_batch_interval_carries_the_loaded_margin(client, payload, monkeypatch):
    """/predict/batch applies the margin at its own call site, so it needs its own check."""
    _assert_route_carries_margin(
        client, "/predict/batch", {"items": [payload]}, monkeypatch, lambda body: body["items"][0]
    )
