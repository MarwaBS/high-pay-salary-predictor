"""Guard against dangling references to the deleted legacy trainer.

``scripts/train_model.py`` (the v1 point-estimate trainer with MLflow +
Optuna) no longer exists; a reference to it in a docstring or user-facing
error message would tell a user to run a file that isn't there.

This test walks the repo and asserts the string ``train_model.py``
appears nowhere in user-visible source — tests, docs, Python modules,
YAML workflows, Dockerfiles, and Makefile.

The guard is a plain substring search rather than an AST walk because the
references it must catch live in **comments and docstrings** (a JSON
response body, a save_baseline_stats docstring), which an AST walk would
silently miss.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Both guard files name the legacy trainer on purpose, inside their own assertions.
_EXCLUDED_FILES = {
    Path("tests") / "test_single_trainer.py",
    Path("tests") / "test_no_dangling_trainer_refs.py",
}

# Notebooks are skipped because they embed ANSI output that produces false
# positives; binaries and images have nothing to match.
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
    """Yield every tracked file eligible for the scan — only tracked files ship."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    for name in filter(None, listed):
        rel = Path(name)
        if rel in _EXCLUDED_FILES:
            continue
        # Match by full filename (Dockerfile, Makefile) or suffix.
        if rel.suffix not in _SCAN_SUFFIXES and rel.name not in {"Dockerfile", "Makefile"}:
            continue
        path = REPO_ROOT / rel
        if path.is_file():
            yield path, rel


def test_no_dangling_train_model_references():
    """Every mention of ``train_model.py`` is dead code — fail if one reappears."""
    offenders: list[str] = []
    scanned = 0
    for path, rel in _iter_tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        scanned += 1
        if "train_model.py" in text:
            # Find the line numbers so the failure message is actionable.
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "train_model.py" in line:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert scanned > 1, f"only {scanned} file(s) scanned — an empty sweep passes on any repo"
    assert not offenders, (
        "The legacy trainer ``scripts/train_model.py`` no longer exists, "
        "but the following files still reference it. Update each to point "
        "at ``scripts/train_quantile.py`` (or delete the stale line "
        "entirely):\n  " + "\n  ".join(offenders)
    )
