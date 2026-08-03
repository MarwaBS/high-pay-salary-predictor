"""Every metric published in a doc must equal ``models/model_metrics.json``.

Each claim is pinned to one line by an anchor that must match exactly once, so
editing the number, moving the row, or deleting it turns this red rather than
passing quietly. A claim is compared at the precision it prints itself: a doc
writing ``~0.77`` must match the recorded value rounded to two places, and one
writing ``~0.782`` to three.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from api.drift import DriftMonitor

REPO_ROOT = Path(__file__).parent.parent
METRICS = json.loads((REPO_ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))

# A printed figure. Bounded rather than a character class so it stops at the
# sentence period in "R² ≈ 0.03." instead of capturing an unparseable "0.03.".
N = r"([0-9]+(?:\.[0-9]+)?)"
# Row counts print with thousands separators; stripped before comparison.
NC = r"([0-9][0-9,]*)"


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
    Claim("MODEL_CARD.md", "point estimator reaches R²", rf"R² ≈ {N}", "r2"),
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
    Claim("README.md", "on held-out test)", rf"\({N} on held-out test\)", "conformal_coverage_80"),
    Claim("MODEL_CARD.md", "coverage 0.7", rf"coverage {N} against", "conformal_coverage_80"),
    Claim("MODEL_CARD.md", "| Test MAE |", rf"~\${N}K", "mae", 1000),
    Claim("MODEL_CARD.md", "| Test RMSE |", rf"~\${N}K", "rmse", 1000),
    Claim("README.md", "| Train / test |", rf"\| {NC} / ", "n_train"),
    Claim("README.md", "| Train / test |", rf"/ {NC} \|", "n_test"),
    Claim("MODEL_CARD.md", "| Positive rate (test)", rf"\| ~{N} \|", "classifier_positive_rate_test"),
    Claim("README.md", "| Quantile crossings |", rf"\*\*{N}\*\*", "quantile_crossings"),
    Claim("MODEL_CARD.md", "| Quantile crossings |", rf"\*\*{N}\*\*", "quantile_crossings"),
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

    printed = match.group(1).replace(",", "")
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


def test_subgroup_roc_auc_range_matches_the_artefact():
    """The published range is a min and a max over the recorded subgroup map."""
    recorded = METRICS["classifier_subgroup_roc_auc"].values()
    line = _claiming_line(Claim("MODEL_CARD.md", "| Subgroup ROC-AUC |", "", "classifier_roc_auc"))
    low, high = (float(x) for x in re.findall(r"min ([0-9.]+), max ([0-9.]+)", line)[0])
    assert _rounded(min(recorded), 2) == Decimal(str(low))
    assert _rounded(max(recorded), 2) == Decimal(str(high))


# Headings whose tables publish measured values. A numeric row under one of these
# must be pinned above; rows elsewhere (ports, dtypes, Make targets) need not be.
METRIC_SECTIONS = (
    "Quantile metrics",
    "Point-estimate metrics",
    "Stability across seeds",
    "Classifier",
    "Baselines",
    "Model performance",
)
# A heading this tuple does not match scans nothing, and a scan of nothing passes
# whatever it is pointed at, so each doc must be shown to have rows to check.
MINIMUM_ROWS_SCANNED = {"README.md": 7, "MODEL_CARD.md": 25}
# Rows pinned by a standalone assertion rather than a Claim entry, because the
# published figure is derived from a nested map rather than a single key.
ANCHORS_PINNED_ELSEWHERE = ("| Subgroup ROC-AUC |",)
# Rows under a metric heading that carry a number but assert no measurement.
EXEMPT_ROW_PREFIXES = (
    "| Metric ",
    "| Percentile ",
    "| --",
    "|---",
)


def _rows_under_metric_sections(doc: str) -> list[str]:
    lines = (REPO_ROOT / doc).read_text(encoding="utf-8").splitlines()
    inside, rows = False, []
    for line in lines:
        if line.startswith("#"):
            inside = any(section.lower() in line.lower() for section in METRIC_SECTIONS)
        elif inside and line.startswith("| ") and re.search(r"\d", line):
            rows.append(line)
    return rows


@pytest.mark.parametrize("doc", ["README.md", "MODEL_CARD.md"])
def test_every_metric_row_is_pinned(doc):
    """The reverse of the pins above: a published figure with no pin is the gap.

    Checking only that each pin resolves would let a new row be added and never
    compared to anything, which is how the stale set accumulated in the first
    place.
    """
    rows = _rows_under_metric_sections(doc)
    assert len(rows) >= MINIMUM_ROWS_SCANNED[doc], (
        f"only {len(rows)} metric rows found in {doc}; a renamed heading leaves this check "
        f"scanning nothing, which passes for the wrong reason"
    )
    anchors = [c.anchor for c in CLAIMS if c.doc == doc] + list(ANCHORS_PINNED_ELSEWHERE)
    unpinned = [row for row in rows if not any(a in row for a in anchors) and not row.startswith(EXEMPT_ROW_PREFIXES)]
    detail = "\n".join(f"  {row[:100]}" for row in unpinned)
    assert not unpinned, f"unpinned metric rows in {doc}:\n{detail}"


def _alarm_paragraph() -> str:
    """The whole alarm bullet, continuation lines included.

    A markdown bullet runs until the next one, so a figure added on a following
    line is published in the same paragraph and has to be read as part of it.
    """
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if "Statistically controlled alarms" in line]
    assert len(starts) == 1, f"the alarm bullet is anchored {len(starts)} times, expected 1"
    end = next(
        (i for i in range(starts[0] + 1, len(lines)) if lines[i].startswith(("-", "#")) or not lines[i].strip()),
        len(lines),
    )
    return " ".join(lines[starts[0] : end])


def _alarm_figures() -> dict[str, tuple[str, str]]:
    """Each figure the bullet may print, keyed by the claim it stands for.

    Recomputed from the detector's tuning and the baseline it monitors, and
    paired with the wording that has to carry it — a number matching some other
    claim in the same sentence would otherwise satisfy a check on the set alone.
    """
    baseline = json.loads((REPO_ROOT / "models" / "baseline_stats.json").read_text(encoding="utf-8"))
    mon = DriftMonitor(baseline, window=1)
    designed = math.erfc(mon.alert_threshold / math.sqrt(2.0))
    return {
        "familywise design level": (f"{designed:.1%}".rstrip("%"), r"rate holds at ≈([\d.]+)%"),
        "uncorrected union": (f"{1 - (1 - designed) ** len(baseline):.0%}".rstrip("%"), r"instead of the ≈(\d+)%"),
        "features monitored": (str(len(baseline)), r"union of ~(\d+) per-feature"),
        "effect floor": (str(mon.min_effect_size), r"effect ≥ ([\d.]+) baseline"),
        "ramp z-multiple": (str(round(mon.alert_threshold)), r"max\(0\.2, (\d+)·"),
        "where the fixed floor starts to bind": (
            str(round((mon.alert_threshold / mon.min_effect_size) ** 2)),
            r"\(z/d\)² = (\d+)",
        ),
    }


def test_every_figure_in_the_alarm_paragraph_is_one_the_code_decides():
    """Membership, not a list of the figures someone happened to check: every
    number the bullet prints is recomputed, and it may print nothing else."""
    printed = set(re.findall(r"\d+(?:\.\d+)?", _alarm_paragraph()))
    sourced = {value for value, _ in _alarm_figures().values()}
    unaccounted = printed - sourced
    assert not unaccounted, f"the bullet prints {sorted(unaccounted)}, which nothing in the code decides"


@pytest.mark.parametrize("claim", sorted(_alarm_figures()))
def test_each_alarm_figure_stands_where_its_claim_does(claim: str):
    """Bound to the wording, so two sourced figures cannot swap roles and leave
    the paragraph stating the opposite of what the code does."""
    expected, pattern = _alarm_figures()[claim]
    match = re.search(pattern, _alarm_paragraph())
    assert match, f"the bullet no longer states the {claim}"
    assert match.group(1) == expected, f"the bullet gives {match.group(1)} for the {claim}, code says {expected}"
