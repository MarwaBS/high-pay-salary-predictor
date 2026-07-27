"""No private repository name may appear in a tracked file.

Two of them reached `.gitignore` and `.trivyignore` once and survived three
reviews, because nothing checked. This is that check. It compares SHA-256
digests from ``.private-name-hashes`` rather than plaintext patterns, so the
guard cannot become the leak it exists to prevent.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
HASH_FILE = REPO_ROOT / ".private-name-hashes"
# Names are matched as whole tokens: the same split the guard file's own
# instructions assume, so a hash added there is found by this sweep.
TOKEN = re.compile(r"[A-Za-z0-9_-]+")


def _banned_digests() -> set[str]:
    lines = HASH_FILE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _tracked_text_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    paths = []
    for name in filter(None, listed):
        path = REPO_ROOT / name
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: a name would not be legible there anyway
        paths.append(path)
    return paths


def _offending_tokens(text: str, banned: set[str]) -> set[str]:
    return {tok for tok in TOKEN.findall(text) if hashlib.sha256(tok.lower().encode()).hexdigest() in banned}


def test_the_guard_file_carries_hashes_and_no_plaintext():
    digests = _banned_digests()
    assert digests, "no digests recorded — this gate would pass on anything"
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests), "a non-digest line would never match a token"


def test_the_sweep_fires_on_a_planted_name():
    """Positive control. Without it a zero-hit result proves only that the sweep ran."""
    digests = _banned_digests()
    planted = "zzz-not-a-real-name"
    control = digests | {hashlib.sha256(planted.encode()).hexdigest()}
    assert _offending_tokens(f"see the {planted} repo", control) == {planted}
    assert not _offending_tokens("an ordinary sentence about salaries", control)


@pytest.mark.parametrize("path", _tracked_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_tracked_file_names_no_private_repository(path):
    found = _offending_tokens(path.read_text(encoding="utf-8"), _banned_digests())
    assert not found, f"{path.relative_to(REPO_ROOT)} names a private repository"
