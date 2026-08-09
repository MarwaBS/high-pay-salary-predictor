"""Every artefact ``config.yaml`` declares must survive a registry deploy.

The GitHub Release ``train.yml`` publishes is the model registry, and the k8s
initContainers stage from it into an ``emptyDir`` on every pod start. An artefact
missing from either list is present in git and absent at runtime.
"""

from __future__ import annotations

import ast
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

from tests.conftest import SUBPROCESS_TIMEOUT_S

REPO_ROOT = Path(__file__).resolve().parents[1]

# The directory a serving stage's relative COPY destinations resolve against,
# and so the one each app derives its artefact paths from. The k8s mountPaths
# are written for it; a stage that moves leaves them pointing nowhere.
IMAGE_WORKDIR = PurePosixPath("/app")


def _config_serving_artifacts() -> set[str]:
    """Basenames of every artefact config.yaml's model block tells the API to load."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    model = cfg["model"]
    arts = {Path(v).name for k, v in model.items() if isinstance(v, str) and v.endswith((".ubj", ".json"))}
    assert arts, "no model artefact paths found in config.yaml — parser drift"
    return arts


def _required_serving_artifacts() -> set[str]:
    """Every served artefact is discoverable from the model config alone."""
    return _config_serving_artifacts()


def _release_artifacts() -> set[str]:
    """Basenames in the softprops/action-gh-release `files:` list of train.yml."""
    wf = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "train.yml").read_text(encoding="utf-8"))
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            if "action-gh-release" in str(step.get("uses", "")):
                files = step["with"]["files"]
                return {Path(line.strip()).name for line in files.splitlines() if line.strip()}
    raise AssertionError("no action-gh-release step found in train.yml")


def _app_model_dir() -> PurePosixPath:
    """Absolute directory the app resolves config.yaml's model artefacts from."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    parents = {
        PurePosixPath(v).parent for v in cfg["model"].values() if isinstance(v, str) and v.endswith((".ubj", ".json"))
    }
    assert len(parents) == 1, f"model artefacts span several directories: {sorted(map(str, parents))}"
    return IMAGE_WORKDIR / parents.pop()


def _staged_for_the_app(manifest: str) -> set[str]:
    """Basenames a deployment's initContainer leaves where the app reads them.

    Follows the volume rather than the text: a `-o` inside a shell comment
    writes nothing, and a file written to a volume the app mounts elsewhere is
    staged into a directory the app never opens.
    """
    spec = yaml.safe_load((REPO_ROOT / "k8s" / manifest).read_text(encoding="utf-8"))["spec"]["template"]["spec"]
    (init,) = spec["initContainers"]
    assert init["command"] == ["sh", "-c"], f"k8s/{manifest} stages artefacts some other way — this check is stale"

    staging = {PurePosixPath(m["mountPath"]): m["name"] for m in init["volumeMounts"]}
    app_dir = _app_model_dir()
    readable = {
        m["name"]
        for c in spec["containers"]
        for m in c.get("volumeMounts", [])
        if PurePosixPath(m["mountPath"]) == app_dir
    }

    staged = set()
    for line in "\n".join(init["args"]).splitlines():
        tokens = shlex.split(line, comments=True)
        for flag, target in zip(tokens, tokens[1:], strict=False):
            written = PurePosixPath(target)
            if flag == "-o" and staging.get(written.parent) in readable:
                staged.add(written.name)
    return staged


def test_release_publishes_every_serving_artifact() -> None:
    required = _required_serving_artifacts()
    published = _release_artifacts()
    missing = required - published
    assert not missing, (
        f"train.yml release omits serving artefact(s) {sorted(missing)} — a "
        f"registry rollback would ship without them and the API would degrade. "
        f"Add them to the `files:` list."
    )


@pytest.mark.parametrize("manifest", ["api-deployment.yaml", "dashboard-deployment.yaml"])
def test_every_pod_stages_every_declared_serving_artifact(manifest: str) -> None:
    """Both pods stage the whole declared set, not the subset each is thought to need.

    A per-pod list has to be kept in step with what that pod's code loads, which
    no check can read off the source. The volume is an emptyDir, so an artefact
    absent from the list is absent at runtime; fetching a few unused files costs
    one download, guessing wrong crashes a pod.
    """
    missing = _required_serving_artifacts() - _staged_for_the_app(manifest)
    assert not missing, (
        f"k8s {manifest} does not leave {sorted(missing)} in {_app_model_dir()} — the pod "
        f"mounts an emptyDir, so anything the initContainer does not write there is absent "
        f"at runtime. Add a curl for each, and check both mountPaths still agree."
    )


def _stage_workdirs() -> dict[str, str]:
    """WORKDIR in effect at the end of each named Dockerfile stage."""
    stage = None
    workdirs: dict[str, str] = {}
    for line in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if named := re.match(r"^FROM\s+\S+\s+AS\s+(\S+)", line):
            stage = named.group(1)
        elif (workdir := re.match(r"^WORKDIR\s+(\S+)", line)) and stage:
            workdirs[stage] = workdir.group(1)
    assert workdirs, "no named build stages found in the Dockerfile — this parser is stale"
    return workdirs


def _served_stage(manifest: str) -> str:
    """The Dockerfile stage a deployment's app container runs."""
    spec = yaml.safe_load((REPO_ROOT / "k8s" / manifest).read_text(encoding="utf-8"))["spec"]["template"]["spec"]
    (image,) = {c["image"] for c in spec["containers"]}
    return PurePosixPath(image.rsplit(":", 1)[0]).name


@pytest.mark.parametrize("manifest", ["api-deployment.yaml", "dashboard-deployment.yaml"])
def test_the_image_each_pod_runs_works_from_where_its_mountpaths_assume(manifest: str) -> None:
    """Both images are checked, because both manifests mount for both.

    Reading one stage covers one image, and the pod running the other fails at
    startup with the manifest and the suite both looking correct.
    """
    stage = _served_stage(manifest)
    workdirs = _stage_workdirs()
    assert stage in workdirs, f"k8s/{manifest} runs an image with no matching Dockerfile stage: {stage!r}"
    assert workdirs[stage] == str(IMAGE_WORKDIR), (
        f"Dockerfile stage {stage!r} now works from {workdirs[stage]}, not {IMAGE_WORKDIR} — "
        f"k8s/{manifest} mounts the artefacts at the old directory and the app will not find them."
    )


def _configmap_urls() -> dict[str, str]:
    """model-url / data-url from the salary-api ConfigMap."""
    cm = yaml.safe_load((REPO_ROOT / "k8s" / "api-configmap.yaml").read_text(encoding="utf-8"))
    return cm["data"]


def test_configmap_points_at_the_gated_release_not_a_placeholder() -> None:
    """The initContainer must pull from the gated GitHub Release (the model
    registry train.yml publishes AFTER its test + integrity gate), not the dead
    ``artifacts.example.com/v1.0.0`` placeholder host — a placeholder URL means
    every pod start ImagePull/curl-fails or, worse, serves
    whatever an attacker parks at the example host."""
    urls = _configmap_urls()
    for key in ("model-url", "data-url"):
        url = urls[key]
        assert "example.com" not in url, f"{key} still points at the placeholder host: {url}"
        assert "github.com/MarwaBS/high-pay-salary-predictor/releases/latest/download/" in url, (
            f"{key} must resolve to the latest gated release: {url}"
        )


def test_dataset_is_published_in_the_release() -> None:
    """The configmap's data-url resolves to a release asset, so the dataset it
    names must actually be in the release ``files:`` list — otherwise the pod's
    data fetch 404s on every start."""
    data_asset = Path(_configmap_urls()["data-url"]).name
    published = _release_artifacts()
    assert data_asset in published, (
        f"data-url names {data_asset!r} but train.yml does not publish it — add it to the release `files:` list."
    )


def test_k8s_images_use_the_ghcr_path_ci_actually_pushes() -> None:
    """The k8s manifests must reference the GHCR path CI actually pushes to
    (``ghcr.io/<repo>`` = ``high-pay-salary-predictor``), not the stale
    ``high_pay_analysis_us`` path — a mismatch ImagePullBackOffs on a bare
    apply. Pin the live path and forbid the dead one."""
    for manifest in ("api-deployment.yaml", "dashboard-deployment.yaml"):
        text = (REPO_ROOT / "k8s" / manifest).read_text(encoding="utf-8")
        assert "high_pay_analysis_us" not in text, (
            f"{manifest} still references the dead GHCR path 'high_pay_analysis_us'"
        )
        assert "ghcr.io/marwabs/high-pay-salary-predictor/" in text, (
            f"{manifest} must use the GHCR path CI publishes to (ghcr.io/marwabs/high-pay-salary-predictor/*)"
        )


def _shipped_modules() -> list[str]:
    """Every shipped module, from what git tracks — a directory list misses
    nested packages that packaging still ships."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    ).stdout.split("\0")
    found = sorted(name for name in filter(None, listed) if not name.startswith("tests/"))
    assert found, "no shipped modules discovered — the enumeration rotted"
    return found


@pytest.mark.parametrize("rel", _shipped_modules())
def test_no_module_names_an_artefact_file(rel: str) -> None:
    """Artefact paths come from config.yaml, and the release and k8s gates above
    derive their coverage from it. A module naming a file directly drops that
    artefact out of both gates while they stay green."""
    declared = _config_serving_artifacts()
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    named = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and Path(node.value).name in declared
    }
    assert not named, f"{rel} names artefact file(s) {sorted(named)} instead of reading config.yaml"
