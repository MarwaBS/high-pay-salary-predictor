"""No private repository name may appear in a tracked file.

Two of them reached `.gitignore` and `.trivyignore` once and survived three
reviews, because nothing checked. This is that check. It compares SHA-256
digests from ``.private-name-hashes`` rather than plaintext patterns, so the
guard cannot become the leak it exists to prevent.

Matching is on a normalised form — lowercase, alphanumerics only — so a name
split by punctuation, suffixed, or run into a neighbouring word reduces to the
same digest as the plain form.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
HASH_FILE = REPO_ROOT / ".private-name-hashes"
EXPECTED_DIGESTS = 2  # narrowing the list is itself a regression; see the count test
TOKEN = re.compile(r"[A-Za-z0-9]+")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _digest(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


def _banned_digests() -> set[str]:
    lines = HASH_FILE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _candidates(text: str) -> set[str]:
    """Every fragment a name could hide in: tokens, their joins, and pairs.

    A name split by punctuation or camel-cased into a longer word is the same
    disclosure as the bare name, so both have to reduce to the same digest.
    """
    tokens = TOKEN.findall(text)
    forms = set(tokens)
    forms |= {a + b for a, b in zip(tokens, tokens[1:], strict=False)}
    forms |= {"".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}
    return forms


def _offending(text: str, banned: set[str]) -> set[str]:
    return {form for form in _candidates(text) if _digest(form) in banned}


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


def test_the_guard_file_carries_hashes_and_no_plaintext():
    digests = _banned_digests()
    assert digests, "no digests recorded — this gate would pass on anything"
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests), "a non-digest line would never match"


def test_no_digest_has_been_dropped():
    """Silently shortening the list would disarm the gate without failing it."""
    assert len(_banned_digests()) == EXPECTED_DIGESTS


@pytest.mark.parametrize(
    "variant",
    ["{n}", "{n}-v2", "my-{n}", "the {n} repo", "see {n}.git", "[{n}]"],
    ids=["bare", "suffixed", "prefixed", "in-prose", "with-extension", "bracketed"],
)
def test_the_sweep_fires_on_a_planted_name(variant):
    """Positive control. A zero-hit run otherwise proves only that the sweep ran."""
    planted = "zzznotarealname"
    control = _banned_digests() | {_digest(planted)}
    assert _offending(variant.format(n=planted), control), f"missed {variant!r}"
    assert not _offending("an ordinary sentence about salaries", control)


def test_a_separated_name_reduces_to_the_same_digest():
    """Punctuation between the words is not a different name."""
    assert _digest("Two-Words") == _digest("TwoWords") == _digest("two words")


@pytest.mark.parametrize("path", _tracked_text_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_tracked_file_names_no_private_repository(path):
    found = _offending(path.read_text(encoding="utf-8"), _banned_digests())
    assert not found, f"{path.relative_to(REPO_ROOT)} names a private repository"
