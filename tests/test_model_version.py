"""
Regression guard for the model-registry / provenance story.

Background
----------
``scripts/train_quantile.py`` emits a composite ``model_version`` string of
the form::

    {service_version}+{git_sha}.{data_sha256_prefix}

into ``models/model_metrics.json``. The string is the reproducibility
primitive for the model registry: any operator looking at a live API can
recover the exact training state (code + data) from the three fragments.
The scheduled ``train.yml`` workflow uses it to tag GitHub Releases.

These tests lock in three invariants so the provenance contract cannot
silently regress:

1. The trained artefact on disk has a ``model_version`` field.
2. The string matches the shape the release workflow parses.
3. The running API surfaces the same value on ``/health`` — i.e. the
   model loaded into the FastAPI process advertises the same provenance
   string that was written at training time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from api import __version__ as SERVICE_VERSION
from api.main import app, state

REPO_ROOT = Path(__file__).parent.parent
METRICS_PATH = REPO_ROOT / "models" / "model_metrics.json"

# ``2.0.0+a1b2c3d4e5f6.9e8d7c6b5a40`` — allow either 12 hex chars or the
# literal ``unknown`` fragment for the git SHA (bare-tarball builds) but
# require the service version to be semver and the data SHA to be hex.
MODEL_VERSION_RE = re.compile(
    r"^(?P<service>\d+\.\d+\.\d+)"
    r"\+(?P<git>[0-9a-f]{12}|unknown)"
    r"\.(?P<data>[0-9a-f]{12})$"
)


def _load_metrics() -> dict:
    assert METRICS_PATH.exists(), (
        f"{METRICS_PATH} is missing. Run `python -m scripts.train_quantile` "
        f"first — the provenance tests need a trained artefact on disk."
    )
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_metrics_json_contains_model_version():
    """``model_version`` must be present in saved metrics.

    Guards against a future refactor that accidentally drops the field
    and silently breaks the scheduled release workflow (which parses
    this field to build the release tag).
    """
    metrics = _load_metrics()
    assert "model_version" in metrics, (
        "models/model_metrics.json has no 'model_version' field. "
        "scripts/train_quantile.py must call build_model_version() and "
        "emit the result into the metrics dict — this is how the "
        "scheduled train.yml workflow names GitHub Releases."
    )
    assert isinstance(metrics["model_version"], str)
    assert metrics["model_version"] != ""


def test_model_version_matches_expected_shape():
    """``model_version`` must match ``{semver}+{git}.{data}``.

    If this regex fails, the scheduled release workflow will fail to
    produce a tag (``tag_name: model-${{ env.MODEL_VERSION }}``) or
    will produce a tag the git server rejects. Catch it at train time.
    """
    metrics = _load_metrics()
    version = metrics["model_version"]
    match = MODEL_VERSION_RE.match(version)
    assert match is not None, (
        f"model_version={version!r} does not match the expected shape "
        f"'{{semver}}+{{git_sha|unknown}}.{{data_sha}}'. This is the "
        f"tag name format the scheduled train.yml workflow uses — "
        f"breaking it silently breaks model releases."
    )
    assert match["service"] == SERVICE_VERSION, (
        f"model_version service fragment={match['service']!r} does not "
        f"match api.__version__={SERVICE_VERSION!r}. The trainer must "
        f"import the same __version__ constant the FastAPI app uses."
    )


def test_metrics_json_contains_service_version():
    """``service_version`` is duplicated as a top-level field for clarity."""
    metrics = _load_metrics()
    assert metrics.get("service_version") == SERVICE_VERSION


def test_health_endpoint_surfaces_model_version():
    """``GET /health`` must return the same ``model_version`` the trainer wrote.

    The API is the operator's single source of truth for "what is
    live?". If /health says a different ``model_version`` than the
    artefact on disk, either the API didn't reload after a retrain or
    something is loading a stale metrics file — both are incidents.
    """
    metrics = _load_metrics()
    # Force the lifespan to run so state.model_version is populated. The
    # TestClient context-manager does this automatically.
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body, (
        "GET /health response is missing 'model_version'. Check "
        "HealthResponse schema and the /health route in api/main.py."
    )
    assert body["model_version"] == metrics["model_version"], (
        f"GET /health returned model_version={body['model_version']!r} "
        f"but models/model_metrics.json has {metrics['model_version']!r}. "
        f"This is the 'live artefact drift' scenario — the API and the "
        f"on-disk metrics have fallen out of sync."
    )
    # Sanity: the AppState singleton should also reflect it.
    assert state.model_version == metrics["model_version"]
