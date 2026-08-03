"""No private repository name may appear in a tracked file.

Compares SHA-256 digests from ``.private-name-hashes`` rather than plaintext
patterns, so the guard cannot become the leak it exists to prevent. Matching is
on a normalised form — lowercase, alphanumerics only — so a name split by
punctuation, suffixed, or run into a neighbouring word reduces to the same
digest as the plain form.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
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


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def _commit_messages() -> list[tuple[str, str]]:
    """Every commit reachable from HEAD, so a branch is swept before it is merged.

    Sweeping the default branch instead would only fire once a leak is published,
    when removing it needs a force-push.
    """
    assert _git("rev-parse", "--is-shallow-repository").stdout.strip() == "false", (
        "shallow clone: a sweep of a truncated history passes for the wrong reason "
        "(CI needs actions/checkout with fetch-depth: 0)"
    )
    out = _git("log", "--format=%H%x1f%B%x1e", "HEAD").stdout
    records = []
    for record in out.split("\x1e"):
        sha_and_body = record.strip().split("\x1f")
        if len(sha_and_body) == 2:
            records.append((sha_and_body[0], sha_and_body[1]))
    return records


def test_no_commit_message_names_a_private_repository():
    """`git ls-files` never reaches history, so published messages need their own sweep."""
    banned = _banned_digests()
    messages = _commit_messages()
    assert messages, "no commits scanned — this would pass on an empty history"
    offenders = {sha for sha, body in messages if _offending(body, banned)}
    assert not offenders, f"commit messages name a private repository: {offenders}"


def test_the_message_sweep_reads_real_bodies():
    """Bodies parsed as empty would make the sweep above pass on any history."""
    messages = _commit_messages()
    tokens = Counter(token for _, body in messages for token in TOKEN.findall(body))
    assert tokens, "no tokens parsed from any commit body"
    most_common, _ = tokens.most_common(1)[0]
    offenders = {sha for sha, body in messages if _offending(body, {_digest(most_common)})}
    assert offenders, "found nothing for a token taken from the history itself"


def test_the_sweep_covers_this_branch_not_only_the_default():
    """Sweeping the default branch would only catch a leak once it is published.

    On a feature branch the two ref choices differ, so this fails if the sweep
    is ever pointed back at ``origin/main``.
    """
    swept = {sha for sha, _ in _commit_messages()}
    assert swept == set(_git("rev-list", "HEAD").stdout.split())
