"""
Static sanity checks on the root Dockerfile's API stage.

These tests don't build or run Docker — they parse the Dockerfile text
and the source of api/main.py to catch the exact class of bug that
shipped in commit 390382a: api/main.py imports a top-level module
(config_schema) that was never COPY'd into the API image, so the CI
smoke test crashes with ModuleNotFoundError the first time the
container tries to start.

Guard is intentionally narrow: only top-level modules api/main.py
imports by bare name (`from <name> import ...` where <name> is not
`api.*`, not a stdlib module, and resolves to a `.py` file at the
repo root) must appear in a COPY directive inside the `AS api` stage.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
API_MAIN = REPO_ROOT / "api" / "main.py"


def _top_level_bare_imports(source_path: Path) -> set[str]:
    """Return top-level module names imported from bare names.

    Filters out:
      - stdlib modules (sys.stdlib_module_names)
      - dotted imports like `api.cache` (handled by `COPY api/`)
      - names that don't resolve to a `<repo_root>/<name>.py` file
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in sys.stdlib_module_names:
                continue
            if "." in node.module:
                # dotted — handled by a directory COPY elsewhere
                continue
            if (REPO_ROOT / f"{root}.py").is_file():
                bare.add(root)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in sys.stdlib_module_names:
                    continue
                if (REPO_ROOT / f"{root}.py").is_file():
                    bare.add(root)
    return bare


def _api_stage_copy_targets() -> set[str]:
    """Return the set of source paths COPY'd inside the `AS api` stage.

    Parses `Dockerfile` text, finds the `FROM ... AS api` marker, then
    collects every `COPY <src> <dst>` source (ignoring `--from=` stage
    copies, which bring in built dependency trees, not source files).
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    # Slice from `FROM ... AS api` onward. The `(?![\w-])` tail stops
    # the match from also hitting `AS api-builder` — `\b` would be wrong
    # because `-` is a non-word character and still counts as a boundary.
    match = re.search(r"^FROM\s+\S+\s+AS\s+api(?![\w-])", text, flags=re.MULTILINE)
    assert match is not None, "Dockerfile must declare a `FROM ... AS api` stage"
    rel_from_end = match.end() - match.start()
    api_stage = text[match.start() :]
    # Stop at the next `FROM` that starts a new stage, if any.
    next_from = re.search(r"^FROM\s+", api_stage[rel_from_end:], flags=re.MULTILINE)
    if next_from is not None:
        api_stage = api_stage[: rel_from_end + next_from.start()]

    copies: set[str] = set()
    for line in api_stage.splitlines():
        line = line.strip()
        if not line.startswith("COPY"):
            continue
        if "--from=" in line:
            # multi-stage copy of built wheels — not a repo source file
            continue
        # Strip the leading `COPY` and tokenize. Last token is the dest.
        tokens = line.split()
        if len(tokens) < 3:
            continue
        # Allow flags like `--chown=...` between COPY and the first src.
        srcs = [t for t in tokens[1:-1] if not t.startswith("--")]
        copies.update(srcs)
    return copies


def _parse_requirements(path: Path) -> set[str]:
    """Return the set of lowercased distribution names pinned in *path*.

    Strips comments and blank lines, splits on the first version specifier
    (``==``, ``>=``, ``<=``, ``~=``, ``!=``, ``>``, ``<``), and lowercases
    the resulting name. Extras (``uvicorn[standard]``) are flattened to
    the bare name. Good enough for a "is it in the list" assertion; does
    not attempt to be a full PEP 508 parser.
    """
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Split off any version specifier.
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in line:
                line = line.split(sep, 1)[0]
                break
        # Drop extras: uvicorn[standard] -> uvicorn
        line = line.split("[", 1)[0]
        names.add(line.strip().lower())
    return names


def test_api_requirements_pins_sklearn_for_xgboost_wrapper():
    """Regression guard for the 2026-04-11 smoke-test crash.

    ``pipeline.load_model`` instantiates ``XGBRegressor()`` (and
    ``load_classifier`` instantiates ``XGBClassifier()``), and in xgboost
    3.x both constructors call into ``sklearn.base`` and raise
    ``ImportError: sklearn needs to be installed in order to use this
    module`` if scikit-learn is not present. The API container must
    therefore ship scikit-learn, not just xgboost.

    Historically the failure mode was silent: the dev environment picked
    up sklearn transitively through other packages in ``requirements.txt``,
    so the bug only surfaced inside the minimal ``requirements-api.txt``
    image at container start-up.
    """
    api_reqs = _parse_requirements(REPO_ROOT / "requirements-api.txt")
    assert "scikit-learn" in api_reqs, (
        "requirements-api.txt must pin scikit-learn. xgboost 3.x's "
        "XGBRegressor / XGBClassifier constructors hard-require sklearn, "
        "and pipeline.load_model / load_classifier instantiate both at "
        "API startup. Without this pin the API container crashes with "
        "'ImportError: sklearn needs to be installed in order to use "
        "this module' the first time the lifespan handler runs."
    )
    assert "xgboost" in api_reqs, "requirements-api.txt must pin xgboost"


def test_api_main_bare_imports_all_copied_into_api_stage():
    """Regression guard for the 390382a CI smoke-test crash.

    Every top-level `.py` module that `api/main.py` imports by bare name
    (e.g. `from config_schema import ...`, `from pipeline import ...`)
    must be COPY'd into the API image. Otherwise `uvicorn api.main:app`
    raises `ModuleNotFoundError` the instant the container starts.
    """
    required = _top_level_bare_imports(API_MAIN)
    copied = _api_stage_copy_targets()

    # A `<name>.py` is satisfied if any COPY source equals `<name>.py`
    # exactly. Directory copies like `api/` don't satisfy bare top-level
    # module imports — those are for the `api/` package itself.
    missing = {name for name in required if f"{name}.py" not in copied}

    assert not missing, (
        f"api/main.py imports top-level modules {sorted(missing)} by bare name, "
        f"but the Dockerfile `AS api` stage never COPY's them into the image. "
        f"This reproduces the 390382a smoke-test crash "
        f"(`ModuleNotFoundError: No module named '{next(iter(missing))}'`). "
        f"Add `COPY {next(iter(missing))}.py ./{next(iter(missing))}.py` to the API stage."
    )
