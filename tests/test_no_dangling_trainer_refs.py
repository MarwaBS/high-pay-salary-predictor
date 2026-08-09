"""``train_model.py`` may not be named outside the two guards that assert on it.

A substring search rather than an AST walk: the references that matter live in
comments, docstrings and response bodies, which an AST walk cannot see. Scope is
tracked files whose suffix is in ``_SCAN_SUFFIXES``, minus ``_EXCLUDED_FILES``;
both sets say beside themselves what they leave out.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.conftest import SUBPROCESS_TIMEOUT_S

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
    "",  # extensionless: Dockerfile, Makefile
}


def _eligible_tracked_count() -> int:
    """How many tracked files the sweep must reach, counted without its filters."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=SUBPROCESS_TIMEOUT_S
    ).stdout
    names = [Path(n) for n in out.splitlines() if n.strip()]
    eligible = [rel for rel in names if rel.suffix in _SCAN_SUFFIXES and rel not in _EXCLUDED_FILES]
    assert eligible, "no eligible tracked files found — this counter is stale"
    return len(eligible)


def _iter_tracked_files():
    """Yield every tracked file eligible for the scan — only tracked files ship."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    ).stdout.split("\0")
    for name in filter(None, listed):
        rel = Path(name)
        if rel in _EXCLUDED_FILES:
            continue
        if rel.suffix not in _SCAN_SUFFIXES:
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

    assert scanned >= _eligible_tracked_count(), (
        f"only {scanned} of {_eligible_tracked_count()} eligible tracked files were read — "
        f"a narrowed sweep passes while a dangling reference survives outside it"
    )
    assert not offenders, (
        "The legacy trainer ``scripts/train_model.py`` no longer exists, "
        "but the following files still reference it. Update each to point "
        "at ``scripts/train_quantile.py`` (or delete the stale line "
        "entirely):\n  " + "\n  ".join(offenders)
    )


def test_every_subprocess_call_is_bounded_by_a_timeout() -> None:
    """A child with no deadline hangs the job to its six-hour ceiling.

    All twelve call sites here were unbounded, one of them in the training
    script rather than a test, so this is a fix and a floor at once. Read from
    the parse tree, because the spawn can be spelled `run`, `check_output` or
    `Popen`.
    """
    import ast

    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    spawns = {"subprocess.run", "subprocess.check_output", "subprocess.Popen"}
    unbounded = []
    for name in [n for n in listed.stdout.split("\0") if n]:
        tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in spawns:
                if not any(word.arg == "timeout" for word in node.keywords):
                    unbounded.append(f"{name}:{node.lineno}")
    assert not unbounded, unbounded
