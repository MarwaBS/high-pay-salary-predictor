"""
Regression guard for audit gap F-03: the service version string must come
from a single source of truth (``api.__version__``) and must match across:

1. ``FastAPI(version=...)`` declared in ``api/main.py``
2. The ``GET /`` root endpoint JSON body
3. The default on ``HealthResponse.version`` in ``api/schemas.py``
4. The ``[project].version`` field in ``pyproject.toml``

Prior to F-03 the root endpoint returned a hardcoded ``"1.0.0"`` while
the FastAPI app advertised ``"2.0.0"`` — silent version skew across
endpoints in the same service.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from api import __version__
from api.main import app
from api.schemas import HealthResponse

REPO_ROOT = Path(__file__).parent.parent


def test_api_version_constant_is_set():
    """``api.__version__`` must be a non-empty semver-looking string."""
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), f"api.__version__ must be semver X.Y.Z, got {__version__!r}"


def test_fastapi_app_version_matches_constant():
    """The mounted FastAPI app must advertise ``__version__`` at `.version`."""
    assert app.version == __version__


def test_root_endpoint_version_matches_constant():
    """``GET /`` must return the same version as ``api.__version__``."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == __version__, (
        f"GET / returned version={body.get('version')!r} but "
        f"api.__version__={__version__!r}. This is audit gap F-03 — "
        f"the root endpoint was hardcoded to a stale version string."
    )


def test_health_response_default_version_matches_constant():
    """``HealthResponse.version`` default must come from ``api.__version__``."""
    resp = HealthResponse(model_loaded=True, dataset_rows=0)
    assert resp.version == __version__


def test_pyproject_version_matches_constant():
    """``pyproject.toml::[project].version`` must match ``api.__version__``.

    Parses the file line by line rather than importing ``tomllib`` so the
    test has no Python 3.11+ dependency footgun and no extra deps.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    in_project = False
    found_version: str | None = None
    for raw_line in pyproject.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version"):
            # ``version         = "2.0.0"`` — take the quoted value.
            match = re.search(r'"([^"]+)"', line)
            if match:
                found_version = match.group(1)
                break
    assert found_version is not None, "pyproject.toml [project].version not found"
    assert found_version == __version__, (
        f"pyproject.toml version={found_version!r} does not match api.__version__={__version__!r}. Bump both together."
    )
