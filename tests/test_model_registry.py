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
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _k8s_download_artifacts() -> set[str]:
    """Basenames the api-deployment initContainer curls into /shared-models."""
    text = (REPO_ROOT / "k8s" / "api-deployment.yaml").read_text(encoding="utf-8")
    # The initContainer stages artefacts with `curl ... -o /shared-models/<name>`.
    names = set(re.findall(r"-o\s+/shared-models/(\S+)", text))
    assert names, "no /shared-models downloads found in k8s/api-deployment.yaml"
    return names


def test_release_publishes_every_serving_artifact() -> None:
    required = _required_serving_artifacts()
    published = _release_artifacts()
    missing = required - published
    assert not missing, (
        f"train.yml release omits serving artefact(s) {sorted(missing)} — a "
        f"registry rollback would ship without them and the API would degrade. "
        f"Add them to the `files:` list."
    )


def test_k8s_initcontainer_downloads_every_serving_artifact() -> None:
    required = _required_serving_artifacts()
    downloaded = _k8s_download_artifacts()
    missing = required - downloaded
    assert not missing, (
        f"k8s initContainer does not download {sorted(missing)} — the API pod "
        f"would start without them. Add a curl for each to api-deployment.yaml."
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
