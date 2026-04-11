"""
Inference helpers for the salary prediction API.

Each helper has a single responsibility so the FastAPI ``/predict`` route
in ``api/main.py`` stays thin and every step can be unit-tested without
spinning up a TestClient.

The pipeline is: ``encode_features`` → ``run_model`` →
``lookup_benchmarks`` → ``build_response``. ``build_benchmark_lookup`` is
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
    REGION_CODES,
    build_feature_row,
    compute_fallback_means,
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


def encode_features(
    req: PredictRequest,
    *,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    region_codes: dict[str, int],
    occ_means: dict[str, float],
    state_means: dict[str, float],
    bls_defaults_lookup: dict[tuple[str, str], BlsDefaults],
) -> pd.DataFrame:
    """Turn a validated request into a model-ready single-row DataFrame.

    Reads BLS context defaults from the precomputed ``bls_defaults_lookup``
    (O(1) dict get) instead of scanning the full DataFrame per request.
    """
    bls = _lookup_bls(bls_defaults_lookup, req.state, req.occupation)
    employment = req.employment if req.employment is not None else bls["employment"]
    lq = req.location_quotient if req.location_quotient is not None else bls["location_quotient"]
    jobs_k = req.jobs_per_1000 if req.jobs_per_1000 is not None else bls["jobs_per_1000"]
    hourly_mean = req.hourly_mean if req.hourly_mean is not None else bls["hourly_mean"]

    edu_ord = edu_order[req.education_level]
    gender_bin = 1 if req.gender == "Male" else 0
    region = region_map.get(req.state, "South")
    region_code = region_codes.get(region, 0)

    occ_fallback, state_fallback = compute_fallback_means({"occ_means": occ_means, "state_means": state_means})
    occ_mean_income = occ_means.get(req.occupation, occ_fallback)
    state_mean_income = state_means.get(req.state, state_fallback)

    return build_feature_row(
        age=req.age,
        edu_ord=edu_ord,
        gender_bin=gender_bin,
        region_code=region_code,
        employment=employment,
        lq=lq,
        jobs_k=jobs_k,
        hourly_mean=hourly_mean,
        occ_mean_income=occ_mean_income,
        state_mean_income=state_mean_income,
    )


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
    "encode_features",
    "run_model",
    "build_response",
    "REGION_CODES",
]
