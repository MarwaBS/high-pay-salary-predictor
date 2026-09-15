"""The committed requirement freezes are the sets the pip-audit gate runs against.

Checks: every API runtime package is in ``requirements-lock.txt``, so the
audited surface is not smaller than what ships; no named maintainer-only tool
is, so it does not grow to cover software the project never runs; and the
dashboard freeze cannot put GitPython back below the patched floor Streamlit
pulls in.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tools that only ever exist on a maintainer's machine - never a runtime, test,
# or CI dependency of this project. Their presence in a `pip freeze` means the
# freeze captured an environment polluted with dev CLIs.
_STRAY_TOOLS = {"git-filter-repo"}

# First patched GitPython on the open alerts against this freeze. streamlit
# allows gitpython<4, so the dashboard pin can take the floor without a new
# ignore. A later `pip freeze` must not write 3.1.58 back.
_GITPYTHON_FLOOR = (3, 1, 59)


def _pkg_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-").split("[")[0])
    return names


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^;\s]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_lock_covers_the_api_runtime_it_claims_to_lock() -> None:
    api = _pkg_names(REPO_ROOT / "requirements-api.txt")
    lock = _pkg_names(REPO_ROOT / "requirements-lock.txt")
    missing = api - lock
    assert not missing, (
        f"requirements-lock.txt is missing API-runtime package(s) {sorted(missing)} "
        f"- it claims to pin the audited runtime, so every requirements-api.txt "
        f"package must appear in it."
    )


def test_lock_has_no_maintainer_only_tools() -> None:
    lock = _pkg_names(REPO_ROOT / "requirements-lock.txt")
    leaked = lock & _STRAY_TOOLS
    assert not leaked, (
        f"requirements-lock.txt contains maintainer-only tool(s) {sorted(leaked)} "
        f"that leaked from a dev machine's pip freeze - remove them so the lock "
        f"reflects only what the project runs."
    )


def test_dashboard_freeze_resolves_patched_gitpython() -> None:
    """CI pip-audits ``requirements-dashboard.txt`` on the 3.12 job.

    Streamlit pulls GitPython into that image. 3.1.58 is the last vulnerable
    release named by the alerts against this freeze; 3.1.59 is the first
    patched. This test holds the floor so a later freeze cannot put 3.1.58
    back.
    """
    version = _pins(REPO_ROOT / "requirements-dashboard.txt").get("gitpython")
    assert version is not None, "gitpython missing from requirements-dashboard.txt"
    major, minor, patch = (int(p) for p in version.split(".")[:3])
    assert (major, minor, patch) >= _GITPYTHON_FLOOR, (
        f"GitPython=={version} in requirements-dashboard.txt is below the "
        f"{'.'.join(str(p) for p in _GITPYTHON_FLOOR)} security floor"
    )
