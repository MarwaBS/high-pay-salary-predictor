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
    get_bls_defaults,
    predict_quantiles,
)


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


def encode_features(
    req: PredictRequest,
    df: pd.DataFrame,
    *,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    region_codes: dict[str, int],
    occ_means: dict[str, float],
    state_means: dict[str, float],
) -> pd.DataFrame:
    """Turn a validated request into a model-ready single-row DataFrame."""
    # Optional BLS context defaults from dataset medians
    bls = get_bls_defaults(df, req.state, req.occupation)
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
) -> PredictResponse:
    """Assemble the final PredictResponse from the model's quantile trio.

    The percentile matches the previous behaviour exactly — it is
    ``(group < predicted).mean() * 100`` — but computed via a binary
    search on the precomputed sorted array instead of a per-request
    DataFrame mask. Both methods count strictly-less-than values.
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
    )


__all__ = [
    "GroupStats",
    "build_benchmark_lookup",
    "lookup_benchmarks",
    "encode_features",
    "run_model",
    "build_response",
    "REGION_CODES",
]
