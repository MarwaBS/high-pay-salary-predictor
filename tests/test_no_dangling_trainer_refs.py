"""Guard against dangling references to the deleted legacy trainer.

``scripts/train_model.py`` (the v1 point-estimate trainer with MLflow +
Optuna) was deleted in commit ``cd1037d`` as part of closing final-audit
gap F-01 (two trainers sharing artefact paths). The file is gone from
disk, but docstrings and user-facing error messages elsewhere in the
repo kept mentioning it, so a user who hit the ``/drift`` disabled path
was told to run a file that no longer exists.

This test walks the repo and asserts the string ``train_model.py``
appears nowhere in tracked, user-visible source — tests, docs, reports,
Python modules, YAML workflows, Dockerfiles, and Makefile. Anything
under ``private/`` or ``.git/`` is exempt (the private audit directory
intentionally preserves historical audit notes).

The guard is intentionally a plain substring search rather than an AST
walk because the prior failure modes were in **comments and docstrings**
(``api/main.py:646``'s JSON response body, ``api/drift.py:215``'s
save_baseline_stats docstring, etc.) — an AST walk would have silently
missed all of them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Directories / files that are allowed to still mention `train_model.py`.
# - private/                 : audit notes tracking historical decisions
# - .git/                    : git internals
# - .venv/, venv/            : developer virtualenvs
# - __pycache__/             : compiled caches
# - tests/test_single_trainer.py : this file's sibling, which intentionally
#                              asserts the legacy trainer does not exist
#                              and therefore references it by name inside
#                              its own guard assertion message.
# - tests/test_no_dangling_trainer_refs.py : this file itself, which
#                              mentions the string in its docstring.
_EXCLUDED_DIR_PARTS = {"private", ".git", ".venv", "venv", "__pycache__", "node_modules"}
# Any path segment ending in these suffixes is treated as a build-tool
# artefact (pip egg-info, setuptools dist-info, mypy cache, etc.). These
# are generated files, usually gitignored, and often stale by design —
# scanning them produces noise, not signal.
_EXCLUDED_DIR_SUFFIXES = (".egg-info", ".dist-info", ".mypy_cache", ".pytest_cache", ".ruff_cache")
_EXCLUDED_FILES = {
    Path("tests") / "test_single_trainer.py",
    Path("tests") / "test_no_dangling_trainer_refs.py",
}

# File extensions the guard actually scans. Skipping binaries, images,
# notebooks (they embed ANSI output that can produce false positives),
# and lockfiles.
_SCAN_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".sh",
    "",  # Dockerfile, Makefile
}


def _iter_tracked_files():
    """Yield every repo file eligible for the dangling-ref scan."""
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in _EXCLUDED_DIR_PARTS for part in parts):
            continue
        if any(part.endswith(_EXCLUDED_DIR_SUFFIXES) for part in parts):
            continue
        rel = path.relative_to(REPO_ROOT)
        if rel in _EXCLUDED_FILES:
            continue
        # Match by full filename (Dockerfile, Makefile) or suffix.
        if path.suffix not in _SCAN_SUFFIXES and path.name not in {"Dockerfile", "Makefile"}:
            continue
        yield path, rel


def test_no_dangling_train_model_references():
    """Every mention of ``train_model.py`` is dead code — fail if one reappears."""
    offenders: list[str] = []
    for path, rel in _iter_tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        if "train_model.py" in text:
            # Find the line numbers so the failure message is actionable.
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "train_model.py" in line:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "The legacy trainer ``scripts/train_model.py`` was deleted in "
        "cd1037d but the following files still reference it. Update "
        "each to point at ``scripts/train_quantile.py`` (or delete the "
        "stale line entirely):\n  " + "\n  ".join(offenders)
    )
