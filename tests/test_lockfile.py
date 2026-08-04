"""``requirements-lock.txt`` is the freeze the pip-audit gate runs against.

Two checks: every API runtime package is in it, so the audited surface is not
smaller than what ships; and no named maintainer-only tool is, so it does not
grow to cover software the project never runs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tools that only ever exist on a maintainer's machine — never a runtime, test,
# or CI dependency of this project. Their presence in a `pip freeze` means the
# freeze captured an environment polluted with dev CLIs.
_STRAY_TOOLS = {"git-filter-repo"}


def _pkg_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-").split("[")[0])
    return names


def test_lock_covers_the_api_runtime_it_claims_to_lock() -> None:
    api = _pkg_names(REPO_ROOT / "requirements-api.txt")
    lock = _pkg_names(REPO_ROOT / "requirements-lock.txt")
    missing = api - lock
    assert not missing, (
        f"requirements-lock.txt is missing API-runtime package(s) {sorted(missing)} "
        f"— it claims to pin the audited runtime, so every requirements-api.txt "
        f"package must appear in it."
    )


def test_lock_has_no_maintainer_only_tools() -> None:
    lock = _pkg_names(REPO_ROOT / "requirements-lock.txt")
    leaked = lock & _STRAY_TOOLS
    assert not leaked, (
        f"requirements-lock.txt contains maintainer-only tool(s) {sorted(leaked)} "
        f"that leaked from a dev machine's pip freeze — remove them so the lock "
        f"reflects only what the project runs."
    )
