"""
Inference helpers for the salary prediction API.

Each helper has a single responsibility so the FastAPI ``/predict`` route
in ``api/main.py`` stays thin and every step can be unit-tested without
spinning up a TestClient.

The pipeline is: ``encode_feature_values`` → ``build_feature_frame`` →
``run_model`` → ``lookup_benchmarks`` → ``build_response``.
``build_benchmark_lookup`` is
called once at startup to build an O(log n) lookup table of per-(state,
education) salary benchmarks so the route never scans the full frame at
request time.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd

from api.schemas import PredictRequest, PredictResponse
from pipeline import (
    FEATURES_FULL,
    REGION_CODES,
    predict_quantiles,
)


class BlsDefaults(TypedDict):
    """Precomputed BLS context defaults for one (state, occupation) cell.

    These are the values ``encode_features`` falls back to when the
    caller does not supply the optional BLS context fields on the
    ``PredictRequest``. Precomputing them at startup removes the last
    per-request DataFrame mask from the hot path.
    """

    employment: float
    location_quotient: float
    jobs_per_1000: float
    hourly_mean: float


class GroupStats(TypedDict):
    """Precomputed benchmark statistics for one (state, education) cell.

    ``sorted_incomes`` holds the Annual Income values sorted ascending so
    ``build_response`` can compute the exact percentile of a prediction
    via ``np.searchsorted`` (O(log n) per request, identical output to
    the previous ``(group < predicted).mean()`` scan).
    """

    median: float
    mean: float
    count: int
    sorted_incomes: np.ndarray


# Sentinel key used when a (state, education) cell has no rows in the
# dataset. Callers fall back to global stats in that case.
_GLOBAL_KEY = ("__global__", "__global__")


def build_benchmark_lookup(df: pd.DataFrame) -> dict[tuple[str, str], GroupStats]:
    """Precompute median/mean/count/sorted-incomes per (state, education).

    Called once at API startup so ``/predict`` becomes an O(log n) lookup
    (one dict get + one binary search over a small sorted array) instead
    of a full-DataFrame mask per request. At the current dataset size
    (10,255 rows × 50 states × 4 education levels ≈ 200 keys) the memory
    footprint is <1 MB.

    Returns
    -------
    dict keyed by ``(State Abbreviation, Education Level)`` plus a
    ``_GLOBAL_KEY`` entry holding dataset-wide fallback stats. Each value
    is a :class:`GroupStats` with a sorted numpy array of incomes attached.
    """
    lookup: dict[tuple[str, str], GroupStats] = {}

    grouped = df.groupby(["State Abbreviation", "Education Level"])["Annual Income"]
    for (state, education), series in grouped:
        values = np.sort(series.to_numpy(dtype=float))
        lookup[(state, education)] = GroupStats(
            median=float(np.median(values)),
            mean=float(np.mean(values)),
            count=int(len(values)),
            sorted_incomes=values,
        )

    # Global fallback for unseen cells (count=0 signals fallback to callers).
    all_income = np.sort(df["Annual Income"].to_numpy(dtype=float))
    lookup[_GLOBAL_KEY] = GroupStats(
        median=float(np.median(all_income)),
        mean=float(np.mean(all_income)),
        count=0,
        sorted_incomes=all_income,
    )
    return lookup


def lookup_benchmarks(
    lookup: dict[tuple[str, str], GroupStats],
    state: str,
    education: str,
) -> GroupStats:
    """Single-hit benchmark lookup with automatic global fallback."""
    return lookup.get((state, education), lookup[_GLOBAL_KEY])


def build_bls_defaults_lookup(df: pd.DataFrame) -> dict[tuple[str, str], BlsDefaults]:
    """Precompute per-(state, occupation) BLS context default values.

    Same idea as :func:`build_benchmark_lookup` but for the BLS context
    fields (``Employment``, ``Location Quotient``, ``Jobs per 1000``,
    ``Hourly Mean``). Called once at startup so ``encode_features`` does
    a dict get instead of scanning the full DataFrame. Contains a
    ``_GLOBAL_KEY`` entry with dataset-wide fallbacks for unseen pairs,
    plus state-only fallbacks (keyed on ``(state, _GLOBAL_KEY[1])``).
    """
    lookup: dict[tuple[str, str], BlsDefaults] = {}

    # Cell-level: (state, occupation) → medians within the cell
    for (state, occupation), sub in df.groupby(["State Abbreviation", "Occupation"]):
        lookup[(state, occupation)] = BlsDefaults(
            employment=float(sub["Employment"].median()),
            location_quotient=float(sub["Location Quotient"].median()),
            jobs_per_1000=float(sub["Jobs per 1000"].median()),
            hourly_mean=float(sub["Hourly Mean"].median()),
        )

    # State-level fallback: (state, "") → state medians
    for state, sub in df.groupby("State Abbreviation"):
        lookup[(state, "")] = BlsDefaults(
            employment=float(sub["Employment"].median()),
            location_quotient=float(sub["Location Quotient"].median()),
            jobs_per_1000=float(sub["Jobs per 1000"].median()),
            hourly_mean=float(sub["Hourly Mean"].median()),
        )

    # Global fallback
    lookup[_GLOBAL_KEY] = BlsDefaults(
        employment=float(df["Employment"].median()),
        location_quotient=float(df["Location Quotient"].median()),
        jobs_per_1000=float(df["Jobs per 1000"].median()),
        hourly_mean=float(df["Hourly Mean"].median()),
    )
    return lookup


def _lookup_bls(
    lookup: dict[tuple[str, str], BlsDefaults],
    state: str,
    occupation: str,
) -> BlsDefaults:
    """Lookup with progressive fallback: (state, occupation) → (state, "") → global."""
    if (state, occupation) in lookup:
        return lookup[(state, occupation)]
    if (state, "") in lookup:
        return lookup[(state, "")]
    return lookup[_GLOBAL_KEY]


def encode_feature_values(
    req: PredictRequest,
    *,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    region_codes: dict[str, int],
    occ_means: dict[str, float],
    state_means: dict[str, float],
    occ_fallback: float,
    state_fallback: float,
    bls_defaults_lookup: dict[tuple[str, str], BlsDefaults],
) -> dict[str, float]:
    """Turn a validated request into an ordered feature dict (one per model column).

    Pure and allocation-light: the occupation/state fallback means are
    precomputed once at startup and passed in, so no per-request reduction
    over the group-mean dicts happens here. Keys are the model's feature
    names, so a caller can both build a frame *and* feed the drift monitor
    from this dict — no DataFrame round-trip needed.
    """
    bls = _lookup_bls(bls_defaults_lookup, req.state, req.occupation)
    region = region_map.get(req.state, "South")
    return {
        "Age": req.age,
        "Education_Ord": edu_order[req.education_level],
        "Gender_Bin": 1 if req.gender == "Male" else 0,
        "Region_Code": region_codes.get(region, 0),
        "Employment": req.employment if req.employment is not None else bls["employment"],
        "Location Quotient": (
            req.location_quotient if req.location_quotient is not None else bls["location_quotient"]
        ),
        "Jobs per 1000": req.jobs_per_1000 if req.jobs_per_1000 is not None else bls["jobs_per_1000"],
        "Hourly Mean": req.hourly_mean if req.hourly_mean is not None else bls["hourly_mean"],
        "Occ_Mean_Income": occ_means.get(req.occupation, occ_fallback),
        "State_Mean_Income": state_means.get(req.state, state_fallback),
    }


def build_feature_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    """Pack one or more feature dicts into a model-ready DataFrame.

    Column order is pinned to ``FEATURES_FULL`` regardless of dict insertion
    order, so a single construction replaces the previous
    list-of-one-row-frames + ``pd.concat`` pattern on the batch path.
    """
    return pd.DataFrame(rows, columns=FEATURES_FULL)


def quantiles_crossed(p10: float, p50: float, p90: float) -> bool:
    """True if the raw quantile trio violates monotonicity (p10<=p50<=p90).

    ``build_response`` clamps crossings so callers never see a degenerate
    interval, but a *rising* crossing rate is a model-health signal (the
    booster drifting near decision boundaries). The route increments a
    Prometheus counter on this so the clamp is observable, not silent.
    """
    return p10 > p50 or p50 > p90


def run_model(model: Any, row: pd.DataFrame) -> tuple[float, float, float]:
    """Invoke the (multi-quantile) model and return (p10, p50, p90) dollars.

    Falls back to (point, point, point) if a legacy point-estimate model is
    loaded. See ``pipeline.predict_quantiles`` for the details.
    """
    return predict_quantiles(model, row)


def build_response(
    req: PredictRequest,
    *,
    p10: float,
    p50: float,
    p90: float,
    group_stats: GroupStats,
    p_above_premium_threshold: float | None = None,
    premium_threshold: int | None = None,
) -> PredictResponse:
    """Assemble the final PredictResponse from the model's quantile trio.

    The percentile matches the previous behaviour exactly — it is
    ``(group < predicted).mean() * 100`` — but computed via a binary
    search on the precomputed sorted array instead of a per-request
    DataFrame mask. Both methods count strictly-less-than values.

    ``p_above_premium_threshold`` and ``premium_threshold`` come from the
    binary classifier head (Gap 1 Phase 1). Both default to ``None`` so
    pre-Phase-1 deployments keep a stable response shape.
    """
    # Defensive quantile ordering — XGBoost occasionally emits tiny
    # crossings near decision boundaries. Force non-decreasing.
    p10_ord = min(p10, p50, p90)
    p90_ord = max(p10, p50, p90)
    p50_ord = min(max(p50, p10_ord), p90_ord)

    sorted_incomes = group_stats["sorted_incomes"]
    if group_stats["count"] > 0 and len(sorted_incomes) > 0:
        strictly_below = int(np.searchsorted(sorted_incomes, p50_ord, side="left"))
        percentile = float(strictly_below / len(sorted_incomes) * 100.0)
    else:
        percentile = 50.0

    return PredictResponse(
        predicted_salary=round(p50_ord, 2),  # backward-compat alias
        predicted_p10=round(p10_ord, 2),
        predicted_p50=round(p50_ord, 2),
        predicted_p90=round(p90_ord, 2),
        prediction_interval_low=round(p10_ord, 2),
        prediction_interval_high=round(p90_ord, 2),
        percentile_in_group=round(percentile, 1),
        group_median=round(group_stats["median"], 2),
        group_mean=round(group_stats["mean"], 2),
        group_size=group_stats["count"],
        state=req.state,
        occupation=req.occupation,
        education_level=req.education_level,
        gender=req.gender,
        age=req.age,
        p_above_premium_threshold=(
            round(float(p_above_premium_threshold), 4) if p_above_premium_threshold is not None else None
        ),
        premium_threshold=premium_threshold,
    )


__all__ = [
    "BlsDefaults",
    "GroupStats",
    "build_benchmark_lookup",
    "build_bls_defaults_lookup",
    "lookup_benchmarks",
    "encode_feature_values",
    "build_feature_frame",
    "quantiles_crossed",
    "run_model",
    "build_response",
    "REGION_CODES",
]
