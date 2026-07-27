"""The published test count must equal the number of tests that actually collect.

A floor such as "460+" stays true while it rots, so the figure is pinned exactly:
adding or removing a test fails here until the README is updated with it.
Collection runs in a subprocess because a suite cannot collect itself.

Scope: only a count written directly before the word "tests" is compared. A subset
count such as "99 unit tests" asserts something this cannot check.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"

# The trailing "+" is matched, not excluded: a floor must be pinned, not skipped.
# The digits are taken as one token, separators included, so "1,506" is read as
# 1506 rather than as the 506 that a word boundary after the comma would find.
CLAIM = re.compile(r"(?<![0-9,])([0-9][0-9,]{1,})\+?\s+tests\b")


@functools.cache
def collected_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"([0-9]+) tests collected", result.stdout)
    assert match, f"could not read a collection count from:\n{result.stdout[-500:]}"
    return int(match.group(1))


def claiming_lines() -> list[tuple[int, str, int]]:
    lines = README.read_text(encoding="utf-8").splitlines()
    return [
        (number, line, int(match.group(1).replace(",", "")))
        for number, line in enumerate(lines, start=1)
        if (match := CLAIM.search(line))
    ]


def test_the_readme_states_a_test_count_at_all():
    """A claim that has been deleted cannot be compared, so absence is a failure."""
    assert claiming_lines(), "no README line states a test count"


@pytest.mark.parametrize("line_number, line, claimed", claiming_lines(), ids=lambda v: str(v)[:20])
def test_every_published_test_count_equals_the_collected_count(line_number, line, claimed):
    assert claimed == collected_count(), f"README.md:{line_number} claims {claimed}: {line.strip()[:80]}"
