"""Unit tests for api/inference.py — the helpers extracted from /predict.

These tests exercise the helpers in isolation without spinning up FastAPI
or loading the full app. They guarantee that ``build_benchmark_lookup``
→ ``lookup_benchmarks`` → ``build_response`` reproduces the exact
percentile semantics of the previous DataFrame-scan implementation.

Run: pytest tests/test_inference.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.inference import (
    build_benchmark_lookup,
    build_bls_defaults_lookup,
    build_response,
    lookup_benchmarks,
)
from api.schemas import PredictRequest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small frame with two (state, education) cells and a global spread."""
    return pd.DataFrame(
        {
            "State Abbreviation": ["CA", "CA", "CA", "NY", "NY", "NY", "NY"],
            "Education Level": [
                "Bachelor's degree",
                "Bachelor's degree",
                "Bachelor's degree",
                "Master's degree",
                "Master's degree",
                "Master's degree",
                "Master's degree",
            ],
            "Annual Income": [100_000, 150_000, 200_000, 120_000, 180_000, 240_000, 300_000],
        }
    )


@pytest.fixture
def sample_bls_df() -> pd.DataFrame:
    """A small frame with two (state, occupation) cells for BLS default lookup tests."""
    return pd.DataFrame(
        {
            "State Abbreviation": ["CA", "CA", "CA", "NY", "NY"],
            "Occupation": [
                "Software Developers",
                "Software Developers",
                "Software Developers",
                "Physicians",
                "Physicians",
            ],
            "Employment": [5000, 6000, 7000, 2000, 2500],
            "Location Quotient": [1.5, 1.6, 1.4, 0.8, 0.9],
            "Jobs per 1000": [3.0, 3.1, 3.2, 1.5, 1.6],
            "Hourly Mean": [75.0, 76.0, 74.0, 100.0, 105.0],
        }
    )


class TestBlsDefaultsLookup:
    """Verify the precomputed (state, occupation) BLS lookup table."""

    def test_lookup_has_cell_entries(self, sample_bls_df):
        lookup = build_bls_defaults_lookup(sample_bls_df)
        assert ("CA", "Software Developers") in lookup
        assert ("NY", "Physicians") in lookup

    def test_cell_values_are_medians(self, sample_bls_df):
        lookup = build_bls_defaults_lookup(sample_bls_df)
        ca_sd = lookup[("CA", "Software Developers")]
        assert ca_sd["employment"] == pytest.approx(6000.0)  # median of [5000, 6000, 7000]
        assert ca_sd["hourly_mean"] == pytest.approx(75.0)  # median of [75, 76, 74]

    def test_state_level_fallback_present(self, sample_bls_df):
        lookup = build_bls_defaults_lookup(sample_bls_df)
        # A state-level entry keyed on ("CA", "") should exist with CA-wide medians.
        assert ("CA", "") in lookup

    def test_global_fallback_present(self, sample_bls_df):
        lookup = build_bls_defaults_lookup(sample_bls_df)
        # The global fallback key (sentinel used by _lookup_bls).
        fallback_keys = [k for k in lookup if k[0].startswith("__")]
        assert len(fallback_keys) >= 1


class TestBenchmarkLookup:
    def test_lookup_has_expected_cells(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        assert ("CA", "Bachelor's degree") in lookup
        assert ("NY", "Master's degree") in lookup

    def test_cell_stats_are_correct(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        ca_ba = lookup[("CA", "Bachelor's degree")]
        assert ca_ba["count"] == 3
        assert ca_ba["median"] == 150_000.0
        assert ca_ba["mean"] == pytest.approx(150_000.0)
        assert list(ca_ba["sorted_incomes"]) == [100_000.0, 150_000.0, 200_000.0]

    def test_sorted_incomes_are_sorted(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        for key, stats in lookup.items():
            values = stats["sorted_incomes"]
            assert np.all(np.diff(values) >= 0), f"{key} not sorted"

    def test_global_fallback_present(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        # Missing cell → lookup_benchmarks should return global stats
        fallback = lookup_benchmarks(lookup, "ZZ", "Unknown degree")
        assert fallback["count"] == 0  # signals fallback
        assert fallback["median"] == pytest.approx(180_000.0)  # median of all 7


class TestBuildResponsePercentile:
    """Verify the sorted-array percentile matches the original DataFrame
    mask semantics: ``(group < predicted).mean() * 100``, using P50 as
    the anchor for the percentile calculation."""

    def _make_req(self) -> PredictRequest:
        return PredictRequest(
            state="CA",
            occupation="Software Developers",
            education_level="Bachelor's degree",
            gender="Female",
            age=32,
        )

    def test_percentile_below_all_is_zero(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        group_stats = lookup[("CA", "Bachelor's degree")]
        resp = build_response(
            self._make_req(),
            p10=40_000.0,
            p50=50_000.0,  # below every row
            p90=60_000.0,
            group_stats=group_stats,
        )
        assert resp.percentile_in_group == 0.0

    def test_percentile_above_all_is_hundred(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        group_stats = lookup[("CA", "Bachelor's degree")]
        resp = build_response(
            self._make_req(),
            p10=400_000.0,
            p50=500_000.0,  # above every row
            p90=600_000.0,
            group_stats=group_stats,
        )
        assert resp.percentile_in_group == 100.0

    def test_percentile_equals_scan_result(self, sample_df):
        """For a 3-row cell [100K, 150K, 200K] a P50 of 175K should
        give the same percentile as the original ``(group < pred).mean()``:
        2 of 3 rows strictly below → 66.6666... → rounds to 66.7."""
        lookup = build_benchmark_lookup(sample_df)
        group_stats = lookup[("CA", "Bachelor's degree")]
        resp = build_response(
            self._make_req(),
            p10=150_000.0,
            p50=175_000.0,
            p90=200_000.0,
            group_stats=group_stats,
        )
        assert resp.percentile_in_group == pytest.approx(66.7, abs=0.1)

    def test_fallback_group_uses_50th_percentile_sentinel(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        fallback = lookup_benchmarks(lookup, "ZZ", "Unknown")
        resp = build_response(
            self._make_req(),
            p10=175_000.0,
            p50=200_000.0,
            p90=225_000.0,
            group_stats=fallback,
        )
        # count=0 triggers the hard-coded 50.0 fallback
        assert resp.percentile_in_group == 50.0
        assert resp.group_size == 0

    def test_quantile_fields_populated(self, sample_df):
        lookup = build_benchmark_lookup(sample_df)
        group_stats = lookup[("CA", "Bachelor's degree")]
        resp = build_response(
            self._make_req(),
            p10=120_000.0,
            p50=150_000.0,
            p90=190_000.0,
            group_stats=group_stats,
        )
        assert resp.predicted_p10 == pytest.approx(120_000.0)
        assert resp.predicted_p50 == pytest.approx(150_000.0)
        assert resp.predicted_p90 == pytest.approx(190_000.0)
        # Backward-compat alias
        assert resp.predicted_salary == resp.predicted_p50
        # PI bounds are now the direct quantiles
        assert resp.prediction_interval_low == pytest.approx(120_000.0)
        assert resp.prediction_interval_high == pytest.approx(190_000.0)

    def test_quantile_crossings_are_reordered(self, sample_df):
        """If the model emits p10 > p90 (a crossing), build_response must
        reorder them so the API never returns an inverted interval."""
        lookup = build_benchmark_lookup(sample_df)
        group_stats = lookup[("CA", "Bachelor's degree")]
        resp = build_response(
            self._make_req(),
            p10=200_000.0,  # crossing: larger than p50
            p50=150_000.0,
            p90=130_000.0,  # crossing: smaller than p50
            group_stats=group_stats,
        )
        assert resp.predicted_p10 <= resp.predicted_p50 <= resp.predicted_p90
