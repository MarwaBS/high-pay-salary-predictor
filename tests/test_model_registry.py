"""The model registry must ship every artefact the API loads.

The GitHub Release published by ``.github/workflows/train.yml`` IS the model
registry: the k8s initContainer (``k8s/api-deployment.yaml``) downloads the
serving artefacts from it on every deploy/rollback. If an artefact the API
loads at startup is absent from the release (or from the initContainer's
download list), a registry-based deploy silently ships without it: an artefact
present in git and baked into the Space image but absent from the release and
the k8s download list is lost on any rollback, so the premium-tier classifier
head disappears and every ``p_above_premium_threshold`` returns null.

These tests machine-check that the release list and the initContainer download
list both cover the artefacts config.yaml tells the API to load, so the three
can never drift apart again.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path, PurePosixPath

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# The runtime image's working directory (Dockerfile). config.yaml names model
# artefacts relative, so this is what makes them absolute at runtime — and
# therefore the only mountPath at which the staged volume is reachable.
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


def test_the_image_workdir_the_mountpaths_are_written_for_is_the_one_it_uses() -> None:
    """Every mountPath above is only correct because the image works from here."""
    workdirs = re.findall(r"^WORKDIR\s+(\S+)", (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"), re.M)
    assert workdirs[-1] == str(IMAGE_WORKDIR), (
        f"the runtime image now works from {workdirs[-1]}, not {IMAGE_WORKDIR} — the k8s "
        f"mountPaths point at the old directory and the app will not find its artefacts."
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
    """Every shipped module. Listing consumers by hand makes the gate a
    whitelist, and the file it forgets is the one that hardcodes a path."""
    roots = (REPO_ROOT, REPO_ROOT / "api", REPO_ROOT / "scripts")
    found = sorted(p.relative_to(REPO_ROOT).as_posix() for root in roots for p in root.glob("*.py"))
    assert found, "no shipped modules discovered — the glob rotted"
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
