"""Guards for requirements-lock.txt honesty.

The README describes the lock as the exact-version freeze of the API runtime +
CI/security tooling (the pip-audit target). Two ways that claim rots:

1. A package the API runtime needs drops out of the freeze, so the "audited
   runtime" no longer covers what actually ships.
2. A maintainer-machine-only tool leaks into the freeze (this happened:
   ``git-filter-repo``, a history-rewrite CLI, was pinned in the lock), which
   both pollutes the reproducibility story and expands the audited surface with
   software the project never runs.

These check both, by name, so the corrected README claim stays true.
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
