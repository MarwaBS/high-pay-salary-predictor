"""Verify committed model artefacts against the SHA-256 digests training recorded.

The live demo serves the artefacts committed to ``models/``, but nothing tied
those committed bytes to the digests training recorded, so a corrupt or
desynced committed artefact could ship under a fully green pipeline. This
script closes that gap: it recomputes the SHA-256 of each committed artefact
and compares it to ``models/model_metrics.json::artifact_sha256``.

Run it in CI and locally. Exit 0 when every artefact matches; exit 1 (naming
the mismatches) otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from pipeline import artifact_mismatches

ROOT = Path(__file__).resolve().parent.parent


def verify(config_path: str | Path | None = None) -> list[str]:
    """Return a list of integrity problems (empty when everything matches)."""
    with open(config_path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    metrics_path = ROOT / model_cfg["metrics_path"]
    metrics = json.loads(metrics_path.read_text())

    recorded = metrics.get("artifact_sha256")
    if not recorded:
        return [f"{metrics_path.name} has no artifact_sha256 block — retrain to record it"]

    paths = {
        "model": ROOT / model_cfg["model_path"],
        "classifier": ROOT / model_cfg["classifier_path"],
        "features": ROOT / model_cfg["features_path"],
        "group_means": ROOT / model_cfg["group_means_path"],
        "baseline_stats": ROOT / model_cfg["baseline_stats_path"],
        "conformal": ROOT / model_cfg["conformal_path"],
    }

    return artifact_mismatches(paths, recorded)


def main() -> int:
    problems = verify()
    if problems:
        print("Artifact integrity check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Artifact integrity OK: every committed artefact matches its recorded SHA-256 digest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
