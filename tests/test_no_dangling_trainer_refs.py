"""``train_model.py`` may not be named outside the two guards that assert on it.

A substring search rather than an AST walk: the references that matter live in
comments, docstrings and response bodies, which an AST walk cannot see. Scope is
tracked files whose suffix is in ``_SCAN_SUFFIXES``, minus ``_EXCLUDED_FILES``;
both sets say beside themselves what they leave out.
"""

from __future__ import annotations

import ast
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


def _spellings(tree: ast.Module, exported: set[str]) -> set[str]:
    """How this file can name those members. Read from its own imports, so an
    aliased module or a bare `from subprocess import run` is not missed."""
    modules: set[str] = {"subprocess"}
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            bare.update(a.asname or a.name for a in node.names if a.name in exported)
    return {f"{m}.{n}" for m in modules for n in exported} | bare


def _has_deadline(call: ast.Call) -> bool:
    """`timeout=None` restores the unbounded default and `timeout=0` is a
    deadline no child can meet, so neither is a real bound and the
    keyword's presence is not the property wanted.

    A `**kwargs` splat carries no `arg`, so a spawn is flagged even when
    the mapping holds a timeout; write the deadline at the call site."""
    for word in call.keywords:
        if word.arg == "timeout":
            return not (isinstance(word.value, ast.Constant) and not word.value.value)
    return False


def _names_bound_to(tree: ast.Module, spawns: set[str]) -> set[str]:
    """Local names holding a spawn, from a parameter default or an assignment.

    One hop, and simple targets only. A tuple-unpacked target, a name bound
    from another name or a lambda default is not followed.
    """
    held = (ast.Attribute, ast.Name)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            taking = node.args.posonlyargs + node.args.args
            given = node.args.defaults
            pairs = list(zip(taking[len(taking) - len(given) :], given, strict=True))
            pairs += [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True) if d is not None]
            names.update(a.arg for a, d in pairs if isinstance(d, held) and ast.unparse(d) in spawns)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            bound = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, held) and ast.unparse(node.value) in spawns:
                names.update(t.id for t in bound if isinstance(t, ast.Name))
    return names


def test_every_subprocess_call_is_bounded_by_a_timeout() -> None:
    """A child with no deadline hangs the job to its six-hour ceiling.

    Of the seven spawns `subprocess` exports, `getoutput`, `getstatusoutput`
    and `Popen` accept no `timeout`, so they are refused rather than waved
    through by a rule they cannot satisfy. A spawn reaching its call site as a
    value is read through the name it is bound to.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    takes_timeout = {"run", "call", "check_call", "check_output"}
    takes_none = {"getoutput", "getstatusoutput", "Popen"}
    unbounded: list[str] = []
    refused: list[str] = []
    for name in [n for n in listed.stdout.split("\0") if n]:
        tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
        bounded = _spellings(tree, takes_timeout)
        unboundable = _spellings(tree, takes_none)
        aliases = _names_bound_to(tree, bounded)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if (callee in bounded or callee in aliases) and not _has_deadline(node):
                    unbounded.append(f"{name}:{node.lineno} {callee}")
            elif isinstance(node, ast.Name | ast.Attribute):
                if ast.unparse(node) in unboundable:
                    refused.append(f"{name}:{node.lineno} {ast.unparse(node)}")
    assert not unbounded, unbounded
    assert not refused, refused
