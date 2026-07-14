"""The assembled HF Space snapshot must satisfy the Space Dockerfile's build.

deploy/huggingface/assemble.sh overlays the runtime snapshot onto a clone of
the Space repo; the Space then runs `docker build` on deploy/huggingface/
Dockerfile with the Space root as context. If assemble.sh omits (or misplaces)
any path the Dockerfile `COPY`s, the Space build fails — and nothing catches it,
because the weekly drift guard only diffs files, it never builds. This exact bug
shipped once: assemble.sh copied start.sh to the Space ROOT while the Dockerfile
`COPY deploy/huggingface/start.sh`, so the build would have failed on a missing
COPY source.

This test runs assemble.sh into a throwaway dir and asserts every COPY *source*
in the Space Dockerfile exists in the output — a static stand-in for the build.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DIR = REPO_ROOT / "deploy" / "huggingface"


def _dockerfile_copy_sources() -> list[str]:
    """Source paths of every `COPY` in the Space Dockerfile (build-context-relative)."""
    sources: list[str] = []
    for raw in (HF_DIR / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("COPY"):
            continue
        # tokens after COPY, minus flags (--chown=…); src is the first, dst last.
        toks = [t for t in line.split()[1:] if not t.startswith("--")]
        if len(toks) >= 2:
            sources.append(toks[0])
    assert sources, "no COPY sources parsed from the Space Dockerfile"
    return sources


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable to run assemble.sh")
def test_assembled_snapshot_satisfies_every_dockerfile_copy(tmp_path: Path) -> None:
    space = tmp_path / "space"
    space.mkdir()
    result = subprocess.run(
        ["bash", str(HF_DIR / "assemble.sh"), str(REPO_ROOT), str(space)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"assemble.sh failed:\n{result.stderr}"

    missing = [src for src in _dockerfile_copy_sources() if not (space / src.rstrip("/")).exists()]
    assert not missing, (
        f"assemble.sh did not produce Dockerfile COPY source(s) {missing} — the "
        f"Space `docker build` would fail on the missing path(s). Fix assemble.sh "
        f"so every COPY source lands where the Dockerfile expects it."
    )
