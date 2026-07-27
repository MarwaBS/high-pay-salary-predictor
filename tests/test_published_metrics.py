"""Every metric published in a doc must equal ``models/model_metrics.json``.

Each claim is pinned to one line by an anchor that must match exactly once, so
editing the number, moving the row, or deleting it turns this red rather than
passing quietly. A claim is compared at the precision it prints itself: a doc
writing ``~0.77`` must match the recorded value rounded to two places, and one
writing ``~0.782`` to three.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
METRICS = json.loads((REPO_ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))

# A printed figure. Bounded rather than a character class so it stops at the
# sentence period in "R² ≈ 0.03." instead of capturing an unparseable "0.03.".
N = r"([0-9]+(?:\.[0-9]+)?)"


@dataclass(frozen=True)
class Claim:
    doc: str
    anchor: str  # substring identifying the claiming line; must be unique in the file
    pattern: str  # regex over that line with one group capturing the printed number
    key: str  # key in model_metrics.json
    scale: float = 1.0  # recorded value is divided by this before comparing

    @property
    def id(self) -> str:
        return f"{self.doc}:{self.key}:{self.anchor[:28]}"


CLAIMS = [
    # ── Interval quality ────────────────────────────────────────────────────
    Claim("MODEL_CARD.md", "raw quantiles | ~", rf"\| ~{N} \|", "quantile_coverage_80"),
    Claim("MODEL_CARD.md", "served (cross-conformal)", rf"\*\*~{N}\*\*", "conformal_coverage_80"),
    Claim("MODEL_CARD.md", "| Median PI width", rf"~\${N}K", "conformal_width_median", 1000),
    Claim("MODEL_CARD.md", "| Conformal margin (log space)", rf"\| ~{N} \|", "conformal_delta"),
    Claim("MODEL_CARD.md", "| P10 pinball loss", rf"~\${N}K", "p10_pinball", 1000),
    Claim("MODEL_CARD.md", "| P50 pinball loss", rf"~\${N}K", "p50_pinball", 1000),
    Claim("MODEL_CARD.md", "| P90 pinball loss", rf"~\${N}K", "p90_pinball", 1000),
    # ── Point-estimate fit ──────────────────────────────────────────────────
    Claim("MODEL_CARD.md", "| Test R² (P50)", rf"\| ~{N} \|", "r2"),
    Claim("MODEL_CARD.md", "| CV R² (5-fold, train only", rf"~{N} ±", "cv_r2_mean"),
    Claim("MODEL_CARD.md", "| CV R² (5-fold, train only", rf"± {N}", "cv_r2_std"),
    # ── Seed stability ──────────────────────────────────────────────────────
    Claim("MODEL_CARD.md", "| P50 R² | ~", rf"~{N} ±", "stability_p50_r2_mean"),
    Claim("MODEL_CARD.md", "| P50 R² | ~", rf"± {N}", "stability_p50_r2_std"),
    Claim("MODEL_CARD.md", "| 80% coverage | ~", rf"~{N} ±", "stability_coverage_80_mean"),
    Claim("MODEL_CARD.md", "| 80% coverage | ~", rf"± {N}", "stability_coverage_80_std"),
    Claim("MODEL_CARD.md", "| Classifier ROC-AUC | ~", rf"~{N} ±", "stability_clf_roc_auc_mean"),
    Claim("MODEL_CARD.md", "| Classifier ROC-AUC | ~", rf"± {N}", "stability_clf_roc_auc_std"),
    Claim("MODEL_CARD.md", "| Classifier Brier | ~", rf"~{N} ±", "stability_clf_brier_mean"),
    Claim("MODEL_CARD.md", "| Classifier Brier | ~", rf"± {N}", "stability_clf_brier_std"),
    # ── Classifier head ─────────────────────────────────────────────────────
    Claim("MODEL_CARD.md", "| ROC-AUC | ~", rf"\| ~{N} \|", "classifier_roc_auc"),
    Claim("MODEL_CARD.md", "| PR-AUC | ~", rf"\| ~{N} \|", "classifier_pr_auc"),
    Claim("MODEL_CARD.md", "| F1 @ 0.5 | ~", rf"\| ~{N} \|", "classifier_f1"),
    Claim("MODEL_CARD.md", "| **Brier score**", rf"\*\*~{N}\*\* vs", "classifier_brier"),
    Claim("MODEL_CARD.md", "| **Brier score**", rf"vs \*\*{N}\*\*", "classifier_brier_base_rate"),
    Claim("MODEL_CARD.md", "| Majority-class accuracy", rf"\| ~{N} \|", "classifier_baseline_majority_acc"),
    Claim("MODEL_CARD.md", "| Majority-class accuracy", rf"accuracy \(~{N}\)", "classifier_accuracy"),
    Claim("MODEL_CARD.md", "| Logistic-regression ROC", rf"\| ~{N} \|", "classifier_baseline_logreg_roc_auc"),
    Claim("MODEL_CARD.md", "| Logistic-regression ROC", rf"\({N} vs", "classifier_roc_auc"),
    # ── README summary ──────────────────────────────────────────────────────
    Claim("README.md", "); point-estimate R²", rf"≈ \${N}K", "conformal_width_median", 1000),
    Claim("README.md", "); point-estimate R²", rf"R² ≈ {N}", "r2"),
    Claim("README.md", "> (AUC ", rf"AUC {N} vs", "classifier_roc_auc"),
    Claim("README.md", "> (AUC ", rf"vs ≈ {N},", "classifier_baseline_logreg_roc_auc"),
    Claim("README.md", "> (AUC ", rf"Brier ≈ {N}", "classifier_brier"),
    Claim("README.md", "raw quantiles | ~", rf"\| ~{N} \|", "quantile_coverage_80"),
    Claim("README.md", "served (conformal)", rf"\*\*~{N}\*\*", "conformal_coverage_80"),
    Claim("README.md", "| Median PI width (served)", rf"~\${N}K", "conformal_width_median", 1000),
    Claim("README.md", "| P50 R² (backward-compat", rf"\| ~{N} \|", "r2"),
    Claim("README.md", "| CV R² (5-fold, train-only", rf"~{N} ±", "cv_r2_mean"),
    Claim("README.md", "| CV R² (5-fold, train-only", rf"± {N}", "cv_r2_std"),
    # ── CHANGELOG ───────────────────────────────────────────────────────────
    Claim("CHANGELOG.md", "to the honest", rf"honest {N}", "cv_r2_mean"),
]


def _claiming_line(claim: Claim) -> str:
    """The one line carrying this claim; a missing or ambiguous anchor is a failure."""
    lines = (REPO_ROOT / claim.doc).read_text(encoding="utf-8").splitlines()
    hits = [line for line in lines if claim.anchor in line]
    assert len(hits) == 1, f"anchor {claim.anchor!r} matched {len(hits)} lines in {claim.doc}, expected 1"
    return hits[0]


def _rounded(value: float, places: int) -> Decimal:
    """Round the way the doc author would, not the way IEEE ties-to-even does."""
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
def test_published_metric_matches_the_artefact(claim: Claim):
    line = _claiming_line(claim)
    match = re.search(claim.pattern, line)
    assert match, f"{claim.doc}: {claim.pattern!r} found no number in {line.strip()!r}"

    printed = match.group(1)
    places = len(printed.split(".")[1]) if "." in printed else 0
    recorded = _rounded(METRICS[claim.key] / claim.scale, places)

    assert Decimal(printed) == recorded, (
        f"{claim.doc} publishes {printed} for {claim.key}; "
        f"model_metrics.json records {METRICS[claim.key]} -> {recorded} at {places} dp"
    )


def test_every_claim_names_a_recorded_metric():
    """A pin against a key the trainer stopped emitting would never compare anything."""
    missing = sorted({c.key for c in CLAIMS} - set(METRICS))
    assert not missing, f"pinned keys absent from model_metrics.json: {missing}"
