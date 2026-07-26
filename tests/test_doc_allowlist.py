"""Guard the ``.gitignore`` doc allowlist.

``.gitignore`` ignores every ``*.md`` and re-allows a fixed set of anchored
paths. The failure mode is silent: ``git add .`` skips a doc that is not on the
list, so its edits simply never land. These assertions read ``git check-ignore``
exit codes, because its ``-v`` output prints for negation matches too and so
cannot distinguish "ignored" from "explicitly allowed".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Paths that must stay ignored: private notes at the root, an allowlisted name
# cloned into a subdirectory, and markdown under .github/, which ships none.
_MUST_BE_IGNORED = [
    "notes.md",
    "scratch.md",
    "todo.md",
    "AUDIT_findings.md",
    "docs/SECURITY.md",
    "a/b/README.md",
    ".github/AUDIT_findings.md",
    ".github/ISSUE_TEMPLATE/notes.md",
]


def _is_ignored(rel_path: str) -> bool:
    """True when the ignore rules match ``rel_path``.

    ``--no-index`` is required: without it git short-circuits on any path that
    is already tracked and reports it as not-ignored whatever the rules say,
    which would make the tracked-file assertion below unable to fail.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def test_every_tracked_markdown_file_stays_committable():
    """A shipped doc that falls off the allowlist would silently stop updating."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert tracked, "expected the repo to track markdown files"

    ignored = [path for path in tracked if _is_ignored(path)]
    assert not ignored, (
        "These tracked docs are now ignored, so `git add .` will skip their "
        "edits. Add an anchored `!/<path>` line to .gitignore:\n  " + "\n  ".join(ignored)
    )


def test_private_note_names_stay_ignored():
    """The allowlist is anchored, so an allowed name elsewhere is still private."""
    committable = [path for path in _MUST_BE_IGNORED if not _is_ignored(path)]
    assert not committable, (
        "These paths are committable, so a private note under one of these "
        "names could be published:\n  " + "\n  ".join(committable)
    )
