"""
High-Paying Jobs in the US — Interactive Dashboard
===================================================
Streamlit app: EDA explorer + ML salary predictor.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from xgboost import XGBRegressor

from pipeline import (
    FEATURES_FULL,
    engineer_features,
    load_group_means,
    load_metrics,
    load_model,
    predict_quantiles_batch,
    train_test_positions,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US High-Pay Jobs Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CFG_PATH = Path(__file__).parent / "config.yaml"
with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)

ROOT = Path(__file__).parent  # project root — resolve all paths relative to here

EDU_ORDER = CFG["education_order"]
REGION_MAP = {state: region for region, states in CFG["regions"].items() for state in states}

# ── Data & Model Loading ──────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading group means...")
def get_group_means() -> dict:
    return load_group_means(str(ROOT / CFG["model"]["group_means_path"]))


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    gm = get_group_means()
    df = pd.read_csv(ROOT / CFG["data"]["cleaned"])
    return engineer_features(df, EDU_ORDER, REGION_MAP, occ_means=gm["occ_means"], state_means=gm["state_means"])


@st.cache_resource(show_spinner="Loading model...")
def get_model() -> XGBRegressor:
    try:
        return load_model(str(ROOT / CFG["model"]["model_path"]))
    except FileNotFoundError:
        st.error("Model not found. Run `make model` (or `python -m scripts.train_quantile`) first.")
        st.stop()
        raise  # unreachable, keeps mypy happy


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    """Load pre-computed model metrics from training artefacts."""
    return load_metrics(str(ROOT / CFG["model"]["metrics_path"]))


# ── Sidebar ───────────────────────────────────────────────────────────────────


def sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("💼 Dashboard Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Filters")
    selected_regions = st.sidebar.multiselect(
        "Region(s)",
        options=sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique()),
    )
    selected_edu = st.sidebar.multiselect(
        "Education Level(s)",
        options=list(EDU_ORDER.keys()),
        default=list(EDU_ORDER.keys()),
    )
    income_range = st.sidebar.slider(
        "Annual Income Range ($)",
        min_value=int(df["Annual Income"].min()),
        max_value=int(df["Annual Income"].max()),
        value=(int(df["Annual Income"].min()), int(df["Annual Income"].max())),
        step=10000,
        format="$%d",
    )

    mask = (
        df["Region"].isin(selected_regions)
        & df["Education Level"].isin(selected_edu)
        & df["Annual Income"].between(*income_range)
    )
    return df[mask]


# ── Tab: Overview ─────────────────────────────────────────────────────────────


def tab_overview(df: pd.DataFrame) -> None:
    """Render the Overview EDA tab: key metrics, top occupations, distributions."""
    st.header("Overview")

    top_occ = df["Occupation"].value_counts().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Annual Income", f"${df['Annual Income'].mean():,.0f}")
    col3.metric(
        "Top State (Volume)",
        df.groupby("State Abbreviation").size().idxmax(),
    )
    col4.metric(
        "Top Occupation (Volume)",
        top_occ[:30] + ("..." if len(top_occ) > 30 else ""),
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        top_occ = (
            df.groupby("Occupation")["Annual Income"]
            .mean()
            .nlargest(15)
            .reset_index()
            .rename(columns={"Annual Income": "Avg Annual Income"})
        )
        fig = px.bar(
            top_occ,
            x="Avg Annual Income",
            y="Occupation",
            orientation="h",
            title="Top 15 Occupations by Avg Income",
            color="Avg Annual Income",
            color_continuous_scale="Blues",
            labels={"Avg Annual Income": "Avg Income ($)"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        edu_income = (
            df.groupby("Education Level")["Annual Income"]
            .mean()
            .reindex(EDU_ORDER.keys())
            .reset_index()
            .rename(columns={"Annual Income": "Avg Annual Income"})
        )
        fig = px.bar(
            edu_income,
            x="Education Level",
            y="Avg Annual Income",
            title="Avg Income by Education Level",
            color="Avg Annual Income",
            color_continuous_scale="Blues",
            text_auto=".2s",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        gender_edu = df.groupby(["Education Level", "Gender"]).size().reset_index(name="Count")
        fig = px.bar(
            gender_edu,
            x="Education Level",
            y="Count",
            color="Gender",
            barmode="group",
            title="Gender Distribution by Education Level",
            color_discrete_map={
                "Male": CFG["visualization"]["colors"]["gender_male"],
                "Female": CFG["visualization"]["colors"]["gender_female"],
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        fig = px.violin(
            df,
            y="Annual Income",
            x="Education Level",
            color="Gender",
            box=True,
            title="Income Distribution by Education & Gender",
            color_discrete_map={
                "Male": CFG["visualization"]["colors"]["gender_male"],
                "Female": CFG["visualization"]["colors"]["gender_female"],
            },
            category_orders={"Education Level": list(EDU_ORDER.keys())},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab: Geographic ───────────────────────────────────────────────────────────


def tab_geographic(df: pd.DataFrame) -> None:
    """Render the Geographic Analysis tab: state-level choropleths and rankings."""
    st.header("Geographic Analysis")

    metric = st.selectbox(
        "Map Metric",
        ["Avg Annual Income", "Job Count", "Avg Location Quotient"],
    )

    state_agg = (
        df.groupby("State Abbreviation")
        .agg(
            avg_income=("Annual Income", "mean"),
            job_count=("Annual Income", "count"),
            avg_lq=("Location Quotient", "mean"),
        )
        .reset_index()
    )

    metric_map = {
        "Avg Annual Income": ("avg_income", "Average Annual Income ($)", "Blues"),
        "Job Count": ("job_count", "Number of High-Pay Records", "Greens"),
        "Avg Location Quotient": ("avg_lq", "Avg Location Quotient", "Oranges"),
    }
    col, label, palette = metric_map[metric]

    fig = px.choropleth(
        state_agg,
        locations="State Abbreviation",
        locationmode="USA-states",
        color=col,
        scope="usa",
        title=f"{label} by State",
        color_continuous_scale=palette,
        labels={col: label},
        hover_data={"avg_income": ":$,.0f", "job_count": ":,", "avg_lq": ":.2f"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_states = state_agg.nlargest(20, "avg_income")
        fig2 = px.bar(
            top_states,
            x="avg_income",
            y="State Abbreviation",
            orientation="h",
            title="Top 20 States — Avg Income",
            color="avg_income",
            color_continuous_scale="Blues",
            text_auto=".2s",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        region_income = (
            df.groupby("Region")["Annual Income"]
            .describe()[["mean", "50%", "std"]]
            .reset_index()
            .rename(columns={"mean": "Mean", "50%": "Median", "std": "Std Dev"})
        )
        fig3 = px.bar(
            region_income,
            x="Region",
            y="Mean",
            error_y="Std Dev",
            title="Regional Income — Mean ± Std Dev",
            color="Mean",
            color_continuous_scale="Blues",
            text_auto=".2s",
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab: Salary Predictor ─────────────────────────────────────────────────────


#: Base URL for the FastAPI service. Set via ``API_BASE_URL`` env var so
#: the dashboard talks to the correct API in local / docker-compose / k8s.
#: Default assumes docker-compose where the API service is named ``api``.
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")


def _call_predict_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST to the FastAPI /predict endpoint. Returns None on network failure."""
    import httpx

    try:
        response = httpx.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        return None
    except httpx.RequestError as exc:
        st.error(
            f"Could not reach the API at {API_BASE_URL}: {exc}.  \n"
            "Set the `API_BASE_URL` env var, or start the API with "
            "`make api` / `docker compose up api`."
        )
        return None


def tab_predictor(df: pd.DataFrame) -> None:
    """Render the Salary Predictor form.

    Delegates inference to the FastAPI ``/predict`` endpoint via ``httpx``
    rather than calling the in-process model. This keeps the dashboard
    and API on a single prediction path — cache hits, rate limiting,
    drift monitoring, and benchmark lookups all flow through one code
    path, so API changes never silently diverge from dashboard
    behaviour.
    """
    st.header("Salary Predictor")
    st.markdown(
        "Enter a profile and the dashboard will call the FastAPI "
        f"`/predict` endpoint at `{API_BASE_URL}` to score it. The API "
        "handles caching, rate limiting, drift tracking, and benchmark "
        "lookups — one source of truth."
    )

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", sorted(df["State Abbreviation"].unique()))
        occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
        education = st.selectbox("Education Level", list(EDU_ORDER.keys()))
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    with col2:
        age = st.slider("Age", min_value=18, max_value=80, value=35)  # match the API's accepted range
        show_adv = st.checkbox("Show advanced inputs (BLS context)")
        if show_adv:
            employment = st.number_input("State-Occupation Employment", value=1000, min_value=0)
            lq = st.number_input("Location Quotient", value=1.0, min_value=0.0, step=0.1)
            jobs_k = st.number_input("Jobs per 1,000", value=2.0, min_value=0.0, step=0.1)
            hourly_mean = st.number_input("BLS Hourly Mean Wage ($)", value=60.0, min_value=0.0, step=1.0)
        else:
            employment = None
            lq = None
            jobs_k = None
            hourly_mean = None

    if st.button("Predict Salary", type="primary"):
        payload: dict[str, Any] = {
            "state": state,
            "occupation": occupation,
            "education_level": education,
            "gender": gender,
            "age": age,
        }
        # Only include BLS context fields if the user supplied them —
        # otherwise let the API fill them from its precomputed defaults.
        if show_adv:
            payload.update(
                {
                    "employment": employment,
                    "location_quotient": lq,
                    "jobs_per_1000": jobs_k,
                    "hourly_mean": hourly_mean,
                }
            )

        result = _call_predict_api(payload)
        if result is None:
            return  # Error already surfaced by _call_predict_api

        p10 = result["predicted_p10"]
        p50 = result["predicted_p50"]
        p90 = result["predicted_p90"]
        pct = result["percentile_in_group"]
        group_size = result["group_size"]

        st.success(f"Median estimate (P50): **${p50:,.0f}**")
        st.info(
            f"**80% prediction interval**: ${p10:,.0f} — ${p90:,.0f}  \n"
            "_Interval comes from a multi-quantile XGBoost model, widened by a "
            "cross-conformal margin so it reaches ~80% empirical coverage on the "
            "held-out test set, and served by the FastAPI `/predict` endpoint. "
            "See the Model Insights tab and `MODEL_CARD.md` for details._"
        )

        if group_size > 0:
            st.markdown(
                f"This estimate is higher than **{pct:.1f}%** of "
                f"{education} earners in {state} ({group_size} comparable records)."
            )
        else:
            st.caption(
                "No records for this exact (state, education) cell, so the "
                f"percentile (**{pct:.1f}%**) is computed against the full dataset."
            )

        # ── Premium-tier probability (classifier head) ────
        # The classifier is optional on the API side — older deployments
        # return ``None`` and the dashboard silently skips the tile.
        p_premium = result.get("p_above_premium_threshold")
        premium_threshold_resp = result.get("premium_threshold")
        if p_premium is not None and premium_threshold_resp is not None:
            st.markdown("---")
            st.metric(
                f"Probability of earning ≥ ${premium_threshold_resp:,}",
                f"{p_premium * 100:.1f}%",
            )
            st.caption(
                "From a separate XGBoost binary classifier trained alongside "
                "the quantile regressor on the same features. Answers a "
                "different question than the quantile interval: _how likely "
                "is this profile to cross the premium threshold at all?_"
            )


# ── Tab: Model Insights ───────────────────────────────────────────────────────


def tab_model(df: pd.DataFrame, model: XGBRegressor, metrics: dict[str, Any]) -> None:
    """Render Model Insights tab: feature importance, residuals, subgroup analysis."""
    st.header("Model Performance & Feature Importance")

    # ── Key metric tiles ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Test R²", f"{metrics.get('r2', 0):.4f}")
    col2.metric("CV R²", f"{metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
    col3.metric("MAE", f"${metrics.get('mae', 0):,.0f}")
    col4.metric("RMSE", f"${metrics.get('rmse', 0):,.0f}")
    col5.metric("Train / Test", f"{metrics.get('n_train', 0):,} / {metrics.get('n_test', 0):,}")

    # ── R² interpretation note ────────────────────────────────────────────────
    st.info(
        "**Note on R²**  \n"
        "P50 under a multi-quantile objective is the median-minimiser, not "
        "the mean-minimiser, so R² is a weak fit-statistic for this model. "
        "The real SLO is empirical quantile coverage — see MODEL_CARD.md."
    )

    # ── 80% prediction interval info (served = conformal-widened) ────────────
    pi_w = metrics.get("conformal_width_median", metrics.get("quantile_width_median", 0))
    pi_cov = metrics.get("conformal_coverage_80", metrics.get("quantile_coverage_80", 0))
    raw_cov = metrics.get("quantile_coverage_80", 0)
    st.markdown(
        f"**80% prediction interval (as served)** — median width **${pi_w:,.0f}**, "
        f"empirical coverage **{pi_cov * 100:.1f}%** on the held-out test set.  \n"
        f"_The API widens the model's raw P10/P90 band (raw coverage "
        f"{raw_cov * 100:.1f}%) by a cross-conformal margin so the served "
        "interval reaches the 80% target — the same interval the Predictor tab shows._"
    )

    st.markdown("---")

    # ── Feature importance + residuals ───────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        feat_imp = pd.DataFrame({"Feature": FEATURES_FULL, "Importance": model.feature_importances_}).sort_values(
            "Importance", ascending=True
        )
        fig = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="XGBoost Feature Importance (Gain)",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        X = df[FEATURES_FULL]
        y = df["Annual Income"]
        # Use the trainer's split (shared primitive) so the residual plot scores
        # the SAME held-out rows the model was evaluated on — not an independently
        # re-derived split that would silently diverge if the trainer's changed.
        _, test_pos = train_test_positions(
            len(df),
            test_size=CFG["model"]["test_size"],
            random_state=CFG["model"]["random_state"],
        )
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]
        # P50 column of the multi-quantile output, already in dollars;
        # predict_quantiles_batch raises on any non-(n, 3) model output.
        y_pred_dollar = predict_quantiles_batch(model, X_test)[:, 1]
        residuals = y_test.values - y_pred_dollar

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_pred_dollar,
                y=residuals,
                mode="markers",
                marker={"opacity": 0.4, "size": 4, "color": "#2196F3"},
                name="Residuals",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual Plot — Dollar Space (Predicted vs Residual)",
            xaxis_title="Predicted Annual Income ($)",
            yaxis_title="Residual ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        x=y_test.values,
        y=y_pred_dollar,
        opacity=0.4,
        labels={"x": "Actual Income ($)", "y": "Predicted Income ($)"},
        title="Actual vs Predicted Annual Income",
    )
    max_val = max(y_test.values.max(), y_pred_dollar.max())
    fig2.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect Prediction",
        )
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("💼 High-Paying Jobs in the US")
    st.markdown(
        "Interactive analysis of high-paying occupations (≥ $100K/yr) "
        "integrating **BLS OEWS** and **US Census** microdata. "
        "Use the sidebar to filter data."
    )

    df = load_data()
    filtered_df = sidebar(df)
    model = get_model()
    metrics = get_metrics()

    if len(filtered_df) == 0:
        st.warning("No data matches current filters. Adjust the sidebar selections.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Geographic Analysis", "Salary Predictor", "Model Insights"])

    with tab1:
        tab_overview(filtered_df)
    with tab2:
        tab_geographic(filtered_df)
    with tab3:
        tab_predictor(df)
    with tab4:
        tab_model(df, model, metrics)


if __name__ == "__main__":
    main()
