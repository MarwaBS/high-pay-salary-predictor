"""
High-Paying Jobs in the US — Interactive Dashboard
===================================================
Streamlit app: EDA explorer + ML salary predictor.
Run: streamlit run streamlit_app.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from pipeline import (
    FEATURES_FULL,
    REGION_CODES,
    build_feature_row,
    compute_fallback_means,
    engineer_features,
    get_bls_defaults,
    load_group_means,
    load_metrics,
    load_model,
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
def get_model():
    return load_model(str(ROOT / CFG["model"]["model_path"]))


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
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Annual Income", f"${df['Annual Income'].mean():,.0f}")
    col3.metric(
        "Top State (Volume)",
        df.groupby("State Abbreviation").size().idxmax(),
    )
    col4.metric(
        "Top Occupation (Volume)",
        df["Occupation"].value_counts().idxmax().split()[0] + "...",
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


def tab_predictor(df: pd.DataFrame, model, metrics: dict) -> None:
    st.header("Salary Predictor")
    st.markdown("Enter individual profile details to estimate annual income using the trained XGBoost model.")

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", sorted(df["State Abbreviation"].unique()))
        occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
        education = st.selectbox("Education Level", list(EDU_ORDER.keys()))
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    with col2:
        age = st.slider("Age", min_value=22, max_value=75, value=35)
        show_adv = st.checkbox("Show advanced inputs (BLS context)")
        if show_adv:
            employment = st.number_input("State-Occupation Employment", value=1000, min_value=0)
            lq = st.number_input("Location Quotient", value=1.0, min_value=0.0, step=0.1)
            jobs_k = st.number_input("Jobs per 1,000", value=2.0, min_value=0.0, step=0.1)
            hourly_mean = st.number_input("BLS Hourly Mean Wage ($)", value=60.0, min_value=0.0, step=1.0)
        else:
            bls = get_bls_defaults(df, state, occupation)
            employment = bls["employment"]
            lq = bls["location_quotient"]
            jobs_k = bls["jobs_per_1000"]
            hourly_mean = bls["hourly_mean"]

    if st.button("Predict Salary", type="primary"):
        edu_ord = EDU_ORDER[education]
        gender_bin = 1 if gender == "Male" else 0
        region = REGION_MAP.get(state, "South")
        region_code = REGION_CODES.get(region, 0)
        gm = get_group_means()
        occ_fallback, state_fallback = compute_fallback_means(gm)
        occ_mean = gm["occ_means"].get(occupation, occ_fallback)
        state_mean_val = gm["state_means"].get(state, state_fallback)

        row = build_feature_row(
            age=age,
            edu_ord=edu_ord,
            gender_bin=gender_bin,
            region_code=region_code,
            employment=employment,
            lq=lq,
            jobs_k=jobs_k,
            hourly_mean=hourly_mean,
            occ_mean_income=occ_mean,
            state_mean_income=state_mean_val,
        )
        prediction = float(np.expm1(model.predict(row)[0]))

        # Empirical 80% prediction interval from training-time residual offsets
        pi_low = prediction + metrics.get("pi_offset_10", 0.0)
        pi_high = prediction + metrics.get("pi_offset_90", 0.0)

        st.success(f"Estimated Annual Income: **${prediction:,.0f}**")
        st.info(
            f"**Empirical 80% prediction interval**: ${pi_low:,.0f} — ${pi_high:,.0f}  \n"
            "_Interval is approximate (heteroscedastic income residuals); "
            "treat as a directional range, not a precise bound._"
        )

        comparable = df[(df["Education Level"] == education) & (df["State Abbreviation"] == state)]["Annual Income"]
        if len(comparable) > 0:
            pct = (comparable < prediction).mean() * 100
            st.markdown(
                f"This estimate is higher than **{pct:.1f}%** of {education} earners in {state} in the dataset."
            )


# ── Tab: Model Insights ───────────────────────────────────────────────────────


def tab_model(df: pd.DataFrame, model, metrics: dict) -> None:
    st.header("Model Performance & Feature Importance")

    # ── Key metric tiles ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Test R²", f"{metrics.get('r2', 0):.4f}")
    col2.metric("CV R²", f"{metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
    col3.metric("MAE", f"${metrics.get('mae', 0):,.0f}")
    col4.metric("RMSE", f"${metrics.get('rmse', 0):,.0f}")
    col5.metric("Train / Test", f"{metrics.get('n_train', 0):,} / {metrics.get('n_test', 0):,}")

    # ── Honest R² context ─────────────────────────────────────────────────────
    r2_ctx = metrics.get("r2_context", "")
    if r2_ctx:
        st.info(f"**Why is R² low?**  {r2_ctx}")

    # ── PI info ───────────────────────────────────────────────────────────────
    pi_w = metrics.get("pi_width", 0)
    pi_cov = metrics.get("pi_coverage", 0)
    st.markdown(
        f"**Empirical 80% prediction interval** — median width **${pi_w:,.0f}**, "
        f"empirical coverage **{pi_cov * 100:.1f}%** on the held-out test set.  \n"
        "_Interval derived from the 10th/90th percentiles of test-set residuals; "
        "broader than ±RMSE because income residuals are right-skewed._"
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
        from sklearn.model_selection import train_test_split

        X = df[FEATURES_FULL]
        y = df["Annual Income"]
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=CFG["model"]["test_size"],
            random_state=CFG["model"]["random_state"],
        )
        # Model predicts log1p(income) — back-transform to dollar space before
        # computing residuals (dollar − log would be meaningless)
        y_pred_dollar = np.expm1(model.predict(X_test))
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

    st.markdown("---")

    # ── Permutation importance (from training artefacts) ──────────────────────
    perm_imp = metrics.get("permutation_importance", {})
    if perm_imp:
        perm_df = pd.DataFrame(
            [
                {"Feature": feat, "Mean ΔR²": v["mean"], "Std": v["std"]}
                for feat, v in sorted(perm_imp.items(), key=lambda x: x[1]["mean"], reverse=True)
            ]
        )
        fig_perm = px.bar(
            perm_df.sort_values("Mean ΔR²"),
            x="Mean ΔR²",
            y="Feature",
            orientation="h",
            error_x="Std",
            title="Permutation Importance (Mean Decrease in R², 50 repeats)",
            color="Mean ΔR²",
            color_continuous_scale="Oranges",
        )
        fig_perm.update_layout(showlegend=False)
        st.plotly_chart(fig_perm, use_container_width=True)
        st.caption(
            "Permutation importance shuffles each feature and measures R² drop — "
            "more trustworthy than gain-based importance for correlated features."
        )

    st.markdown("---")

    # ── Subgroup performance ───────────────────────────────────────────────────
    subgroup = metrics.get("subgroup_metrics", {})
    if subgroup:
        st.subheader("Subgroup Performance (held-out test set)")
        sg_df = pd.DataFrame(
            [
                {"Subgroup": k, "n": v["n"], "R²": round(v["r2"], 4), "MAE ($)": int(v["mae"])}
                for k, v in subgroup.items()
            ]
        )
        col_g, col_r = st.columns(2)
        with col_g:
            gender_df = sg_df[sg_df["Subgroup"].str.startswith("Gender")]
            fig_g = px.bar(
                gender_df,
                x="Subgroup",
                y="R²",
                title="R² by Gender",
                color="Subgroup",
                color_discrete_map={"Gender=Male": "#2196F3", "Gender=Female": "#E91E63"},
                text="R²",
            )
            fig_g.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_g.update_layout(showlegend=False, yaxis_range=[0, sg_df["R²"].max() * 1.3])
            st.plotly_chart(fig_g, use_container_width=True)
        with col_r:
            region_df = sg_df[sg_df["Subgroup"].str.startswith("Region")]
            fig_r = px.bar(
                region_df,
                x="Subgroup",
                y="R²",
                title="R² by Region",
                color="R²",
                color_continuous_scale="Blues",
                text="R²",
            )
            fig_r.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_r.update_layout(showlegend=False, yaxis_range=[0, sg_df["R²"].max() * 1.3])
            st.plotly_chart(fig_r, use_container_width=True)
        st.dataframe(sg_df, use_container_width=True, hide_index=True)
        st.caption(
            "Lower female R² reflects smaller sample size and higher within-cohort income variance. "
            "Gender is encoded as binary (Census CPS limitation)."
        )


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
        tab_predictor(df, model, metrics)
    with tab4:
        tab_model(df, model, metrics)


if __name__ == "__main__":
    main()
