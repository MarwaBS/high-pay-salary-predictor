"""Every Key Findings claim must re-derive from the committed data.

A pin on the figure alone is not enough: the sentence around it names the
education tiers, the region, the feature and the direction of the comparison,
and each of those is a separate way for the claim to become false while the
number stays right. Every assertion below therefore checks the subject and the
direction as well as the value, against
``Data/cleaned_high_pay_data.csv`` or, for the served interval, the committed
model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from xgboost import XGBRegressor

from pipeline import FEATURES_FULL, engineer_features, predict_quantiles_batch

REPO_ROOT = Path(__file__).parent.parent
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()


def _claiming_line(anchor: str, *, in_table: bool = False) -> str:
    """The one README line carrying this claim.

    ``in_table`` additionally requires it to be a Key Findings row, so the claim
    cannot be satisfied by the same text hidden in a comment or moved elsewhere.
    """
    hits = [line for line in README if anchor in line]
    assert len(hits) == 1, f"anchor {anchor!r} matched {len(hits)} README lines, expected 1"
    if in_table:
        assert hits[0].startswith("| **"), f"claim left the Key Findings table: {hits[0][:70]!r}"
    return hits[0]


def _number(line: str, pattern: str) -> float:
    match = re.search(pattern, line)
    assert match, f"{pattern!r} found no number in {line.strip()!r}"
    return float(match.group(1))


def _requires(line: str, *phrases: str) -> None:
    for phrase in phrases:
        assert phrase in line, f"missing {phrase!r} in {line.strip()!r}"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df(cfg):
    frame = pd.read_csv(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")
    region_of = {s: r for r, states in cfg["regions"].items() for s in states}
    frame["Region"] = frame["State Abbreviation"].map(region_of)
    return frame


@pytest.fixture(scope="module")
def engineered(df, cfg):
    region_of = {s: r for r, states in cfg["regions"].items() for s in states}
    return engineer_features(df, cfg["education_order"], region_of)


@pytest.fixture(scope="module")
def within_cell(df):
    """Sample-weighted mean Cohen's d per grouping, computed once."""
    return {
        key: _within_cell_d(df, list(key))
        for key in (("Occupation", "State Abbreviation"), ("Education Level", "Region"))
    }


def _within_cell_d(df: pd.DataFrame, keys: list[str]) -> float:
    effects, weights = [], []
    for _, cell in df.groupby(keys):
        male = cell[cell["Gender"] == "Male"]["Annual Income"]
        female = cell[cell["Gender"] == "Female"]["Annual Income"]
        if len(male) < 2 or len(female) < 2:
            continue
        pooled = math.sqrt(
            ((len(male) - 1) * male.var(ddof=1) + (len(female) - 1) * female.var(ddof=1))
            / (len(male) + len(female) - 2)
        )
        if pooled > 0:
            effects.append((male.mean() - female.mean()) / pooled)
            weights.append(len(cell))
    return float(np.average(effects, weights=weights))


class TestEducationPremium:
    @staticmethod
    def _medians(df, cfg):
        order = sorted(cfg["education_order"], key=cfg["education_order"].get)
        return order, df.groupby("Education Level")["Annual Income"].median()

    def test_headline_names_both_tiers_and_both_steps(self, df, cfg):
        order, med = self._medians(df, cfg)
        line = _claiming_line("median jump", in_table=True)
        biggest = max(order[1:], key=lambda lvl: med[lvl] - med[order[0]])

        _requires(line, f"{order[0]} → {order[-1]}", f"{order[0]} → {biggest}", "not the largest step")
        assert round((med[order[-1]] - med[order[0]]) / 1000, 1) == _number(line, r"~\$([0-9.]+)K median jump")
        assert round((med[biggest] - med[order[0]]) / 1000, 1) == _number(line, r"is ~\$([0-9.]+)K")
        assert med[biggest] > med[order[-1]], "the named largest step is not actually the largest"

    @staticmethod
    def _state_premiums(df):
        """Advanced-tier minus lower-tier mean income per state, ranked."""
        advanced = {"Doctoral degree", "Professional degree", "Master's degree"}
        rows = []
        for state, cell in df.groupby("State"):
            top = cell[cell["Education Level"].isin(advanced)]["Annual Income"]
            rest = cell[~cell["Education Level"].isin(advanced)]["Annual Income"]
            if len(top) > 1 and len(rest) > 1:
                rows.append((state, top.mean() - rest.mean()))
        return pd.DataFrame(rows, columns=["state", "premium"]).sort_values("premium", ascending=False)

    def test_the_state_premium_spread_and_both_named_states(self, df):
        """The gallery paragraph names the real extremes, median and the rank of each tech state."""
        prem = self._state_premiums(df).reset_index(drop=True)
        rank = {row.state: i + 1 for i, row in enumerate(prem.itertuples())}
        line = _claiming_line("The education premium spans")
        _requires(
            line,
            f"spans −${abs(round(prem.premium.min() / 1000))}K to +${round(prem.premium.max() / 1000)}K",
            f"across the {len(prem)} states",
            f"median of ${round(prem.premium.median() / 1000)}K",
            f"Washington sits {rank['Washington']}th",
            f"California {rank['California']}th",
        )

    def test_narrative_states_the_same_gap_and_its_non_monotonicity(self, df, cfg):
        order, med = self._medians(df, cfg)
        line = _claiming_line("The steps are modest")
        steps = [med[b] - med[a] for a, b in zip(order, order[1:], strict=False)]

        assert min(steps) < 0, "if every ordinal step now gains, this sentence must be rewritten"
        _requires(line, "not monotone", f"{order[-2]} out-earns {order[-1]}")
        assert round((med[order[-1]] - med[order[0]]) / 1000, 1) == _number(line, r"~\$([0-9.]+)K in medians")
        assert round(-min(steps) / 1000, 1) == _number(line, r"by ~\$([0-9.]+)K")


class TestRegionalDisparity:
    def test_headline_names_the_leader_the_runner_up_and_both_means(self, df):
        means = df.groupby("Region")["Annual Income"].mean().sort_values(ascending=False)
        line = _claiming_line("workers earn the most", in_table=True)
        leader, second = means.index[0], means.index[1]

        assert line.startswith(f"| **Regional disparity** | {leader} workers earn the most")
        _requires(line, f"in the {second}")
        assert round(means.iloc[0] / 1000, 1) == _number(line, r"mean \$([0-9.]+)K")
        assert round(means.iloc[1] / 1000, 1) == _number(line, r"ahead of \$([0-9.]+)K")

    def test_narrative_names_the_spread_extremes(self, df):
        """The gallery sentence names the same widest and narrowest regions as the data."""
        spread = df.groupby("Region")["Annual Income"].std()
        line = _claiming_line("carries the widest spread")
        _requires(line, f"The {spread.idxmax()} carries the widest spread", f"the {spread.idxmin()} the narrowest")

    def test_narrowest_served_interval_region(self, df, engineered):
        """The interval half of the same claim, from the model actually shipped."""
        model = XGBRegressor()
        model.load_model(str(REPO_ROOT / "models" / "xgb_salary_model.ubj"))
        delta = json.loads((REPO_ROOT / "models" / "conformal_delta.json").read_text())["conformal_delta"]
        preds = predict_quantiles_batch(model, engineered[FEATURES_FULL], conformal_delta=delta)
        spread = pd.Series(preds[:, 2] - preds[:, 0]).groupby(df["Region"].values).median()

        line = _claiming_line("served interval band", in_table=True)
        _requires(line, f"band is narrowest in the {spread.idxmin()}")
        # The gallery paragraph restates it, so it has to move with the model too.
        gallery = _claiming_line("served interval being narrowest")
        _requires(gallery, f"narrowest in the {spread.idxmin()}")


class TestGenderGap:
    def test_headline_reports_the_pooled_effect_and_the_welch_t(self, df):
        from scipy import stats

        male = df[df["Gender"] == "Male"]["Annual Income"]
        female = df[df["Gender"] == "Female"]["Annual Income"]
        pooled = math.sqrt(
            ((len(male) - 1) * male.var(ddof=1) + (len(female) - 1) * female.var(ddof=1))
            / (len(male) + len(female) - 2)
        )
        line = _claiming_line("Pooled Cohen's", in_table=True)

        _requires(line, "survives conditioning")
        assert round((male.mean() - female.mean()) / pooled, 2) == _number(line, r"\*d\* ≈ ([0-9.]+)")
        assert round(float(stats.ttest_ind(male, female, equal_var=False).statistic), 1) == _number(
            line, r"Welch t = ([0-9.]+)"
        )

    @pytest.mark.parametrize(
        ("keys", "pattern"),
        [
            (("Occupation", "State Abbreviation"), r"≈ ([0-9.]+) within the same occupation and state"),
            (("Education Level", "Region"), r"≈ ([0-9.]+) within education and region"),
        ],
        ids=["occupation-state", "education-region"],
    )
    def test_headline_within_cell_effects(self, within_cell, keys, pattern):
        line = _claiming_line("Pooled Cohen's", in_table=True)
        assert round(within_cell[keys], 2) == _number(line, pattern)

    def test_narrative_states_the_within_cell_effect(self, within_cell):
        line = _claiming_line("persists *within* occupation-state cells")
        _requires(line, "sample-weighted")
        assert round(within_cell[("Occupation", "State Abbreviation")], 2) == _number(line, r"\*d\* = ([0-9.]+)")


class TestAgeSignal:
    @staticmethod
    def _ranked(engineered):
        rho = {f: abs(engineered[f].corr(engineered["Annual Income"], method="spearman")) for f in FEATURES_FULL}
        return sorted(rho, key=rho.get, reverse=True), rho

    def test_headline_names_the_leader_runner_up_and_both_coefficients(self, engineered):
        ranked, rho = self._ranked(engineered)
        line = _claiming_line("Spearman", in_table=True)

        _requires(line, f"{ranked[0]} has the strongest rank correlation", f"next is `{ranked[1]}`")
        assert f"the {len(FEATURES_FULL)} model features" in line
        assert round(rho[ranked[0]], 2) == _number(line, r"ρ = \+([0-9.]+)")
        assert round(rho[ranked[1]], 2) == _number(line, r"at \+([0-9.]+)")

    def test_headline_gain_share_matches_the_shipped_model(self):
        """The row is a claim about the model, so it cites a model statistic too."""
        model = XGBRegressor()
        model.load_model(str(REPO_ROOT / "models" / "xgb_salary_model.ubj"))
        scores = model.get_booster().get_score(importance_type="total_gain")
        total = sum(scores.values())
        ranked = sorted(scores, key=scores.get, reverse=True)
        line = _claiming_line("Spearman", in_table=True)

        assert ranked[0] == "Age", f"model's top feature by total gain is {ranked[0]}"
        assert round(100 * scores[ranked[0]] / total, 1) == _number(line, r"([0-9.]+)% of total gain")
        assert round(100 * scores[ranked[1]] / total, 1) == _number(line, r"vs ([0-9.]+)%")

    def test_headline_age_income_endpoints(self, engineered):
        line = _claiming_line("Spearman", in_table=True)
        buckets = pd.cut(engineered["Age"], [17, 29, 39, 49, 64, 100])
        medians = engineered.groupby(buckets, observed=True)["Annual Income"].median()

        assert medians.is_monotonic_increasing, "if income no longer climbs throughout, rewrite the claim"
        assert round(medians.iloc[0] / 1000) == _number(line, r"\$([0-9]+)K at 18–29")
        assert round(medians.iloc[-1] / 1000) == _number(line, r"\$([0-9]+)K at 65\+")

    def test_narrative_names_the_same_leader_and_runner_up(self, engineered):
        ranked, _ = self._ranked(engineered)
        line = _claiming_line("BLS wage signals")
        _requires(line, f"{ranked[0]} carries the **strongest", f"above `{ranked[1]}`")
        assert f"the {len(FEATURES_FULL)} model features" in line

    def test_narrative_repeats_the_age_endpoints(self, engineered):
        """The narrative restates the table's figures, so it carries the same pins."""
        line = _claiming_line("climbs across every age bucket")
        buckets = pd.cut(engineered["Age"], [17, 29, 39, 49, 64, 100])
        medians = engineered.groupby(buckets, observed=True)["Annual Income"].median()

        assert medians.is_monotonic_increasing
        assert round(medians.iloc[0] / 1000) == _number(line, r"\$([0-9]+)K at 18–29")
        assert round(medians.iloc[-1] / 1000) == _number(line, r"\$([0-9]+)K at 65\+")


class TestDataPrepCeiling:
    def test_both_filters_really_bind_at_100k(self, df):
        line = _claiming_line("double-filters the cohort", in_table=True)
        _requires(line, "INCTOT ≥ 100K", "A_MEAN ≥ 100K")
        assert df["Annual Income"].min() == 100_000
        assert df["Annual Mean Wage"].min() == 100_000
