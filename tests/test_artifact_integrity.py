"""Committed serving artefacts must match the SHA-256 digests training recorded.

Without this, the committed artefacts the live demo serves are never checked
against the digests training recorded. verify() content-addresses them against
models/model_metrics.json::artifact_sha256.
"""

import scripts.verify_artifacts as va


def test_committed_artifacts_match_recorded_digests():
    problems = va.verify()
    assert not problems, "artifact integrity: " + "; ".join(problems)


def test_integrity_check_can_go_red(monkeypatch):
    """Prove the gate can fail: force every recomputed digest to a wrong value
    and confirm verify() flags a mismatch rather than passing silently."""
    monkeypatch.setattr(va, "sha256_file", lambda _p: "0" * 64)
    problems = va.verify()
    assert problems, "integrity check did not flag mismatched digests — the gate cannot fail"


def test_serving_mismatch_detector(tmp_path):
    """The serving-side check (crashes startup on mismatch) must flag a swapped
    artefact and pass a matching one."""
    from api.main import _artifact_mismatches
    from pipeline import sha256_file

    artefact = tmp_path / "model.ubj"
    artefact.write_bytes(b"trained-bytes")
    good = {"model": sha256_file(artefact)}
    assert _artifact_mismatches({"model": artefact}, good) == []
    assert _artifact_mismatches({"model": artefact}, {"model": "0" * 64})
