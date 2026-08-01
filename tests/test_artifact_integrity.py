"""Committed serving artefacts must match the SHA-256 digests training recorded.

verify() content-addresses them against models/model_metrics.json::artifact_sha256.
"""

from pathlib import Path

import yaml

import pipeline
import scripts.verify_artifacts as va

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_artifacts_match_recorded_digests():
    problems = va.verify()
    assert not problems, "artifact integrity: " + "; ".join(problems)


def test_integrity_check_can_go_red(monkeypatch):
    """Prove the gate can fail: force every recomputed digest to a wrong value
    and confirm verify() flags a mismatch rather than passing silently."""
    monkeypatch.setattr(pipeline, "sha256_file", lambda _p: "0" * 64)
    problems = va.verify()
    assert problems, "integrity check did not flag mismatched digests — the gate cannot fail"


def test_serving_mismatch_detector(tmp_path):
    """The serving-side check (crashes startup on mismatch) must flag a swapped
    artefact and pass a matching one."""
    artefact = tmp_path / "model.ubj"
    artefact.write_bytes(b"trained-bytes")
    good = {"model": pipeline.sha256_file(artefact)}
    assert pipeline.artifact_mismatches({"model": artefact}, good) == []
    assert pipeline.artifact_mismatches({"model": artefact}, {"model": "0" * 64})


def test_both_callers_share_one_implementation():
    """Two copies of the loop can check different things while both stay green."""
    import api.main

    assert api.main.artifact_mismatches is pipeline.artifact_mismatches
    assert va.artifact_mismatches is pipeline.artifact_mismatches


def test_both_callers_check_the_same_artefacts(monkeypatch):
    """One shared loop still leaves the two key sets free to diverge: the CI
    script builds a static dict, the API a dynamic one from what actually loaded.
    """
    from fastapi.testclient import TestClient

    import api.main

    captured: dict[str, object] = {}

    def _capture(files, _recorded):
        captured.update(files)
        return []

    monkeypatch.setattr(va, "artifact_mismatches", _capture)
    va.verify()

    with TestClient(api.main.app):
        served = set(api.main.state.artifact_sha256)

    assert set(captured) == served, "the CI integrity script and the API check different artefacts"


def test_the_ci_script_reads_the_configured_baseline_path(tmp_path):
    """Renaming the artefact in config must move where the check looks, or the
    key is decorative and the path is really hardcoded."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["baseline_stats_path"] = "models/renamed_baseline.json"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    problems = va.verify(cfg_path)
    assert any("renamed_baseline.json" in p for p in problems), (
        f"verify() did not look at the configured path; reported {problems}"
    )
