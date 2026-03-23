"""
High-Paying Jobs in the US — Interactive Dashboard
===================================================
Streamlit app: EDA explorer + ML salary predictor.
Run: streamlit run streamlit_app.py
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from pipeline import FEATURES_FULL, REGION_CODES, engineer_features

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US High-Pay Jobs Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

EDU_ORDER = CFG["education_order"]
REGION_MAP = {
    state: region
    for region, states in CFG["regions"].items()
    for state in states
}

# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CFG["data"]["cleaned"])
    return engineer_features(df, EDU_ORDER, REGION_MAP)


@st.cache_resource(show_spinner="Training model (first run)...")
def get_model(df: pd.DataFrame) -> XGBRegressor:
    model_path = CFG["model"]["model_path"]
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        X = df[FEATURES_FULL]
        y = df["Annual Income"]
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=CFG["model"]["test_size"], random_state=CFG["model"]["random_state"]
        )
        model = XGBRegressor(
            n_estimators=CFG["model"]["n_estimators"],
            max_depth=CFG["model"]["max_depth"],
            learning_rate=CFG["model"]["learning_rate"],
            subsample=CFG["model"]["subsample"],
            colsample_bytree=CFG["model"]["colsample_bytree"],
            random_state=CFG["model"]["random_state"],
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        os.makedirs(CFG["data"]["models_dir"], exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model


@st.cache_data(show_spinner=False)
def get_model_metrics(_model: XGBRegressor, df: pd.DataFrame) -> dict[str, object]:
    X = df[FEATURES_FULL]
    y = df["Annual Income"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=CFG["model"]["test_size"], random_state=CFG["model"]["random_state"]
    )
    y_pred = _model.predict(X_test)
    return {
        "R²": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 0),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 0),
        "n_test": len(y_test),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────


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


# ── Tab: Overview ────────────────────────────────────────────────────────────


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
        gender_edu = (
            df.groupby(["Education Level", "Gender"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            gender_edu,
            x="Education Level",
            y="Count",
            color="Gender",
            barmode="group",
            title="Gender Distribution by Education Level",
            color_discrete_map={"Male": CFG["visualization"]["colors"]["gender_male"],
                                 "Female": CFG["visualization"]["colors"]["gender_female"]},
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
            color_discrete_map={"Male": CFG["visualization"]["colors"]["gender_male"],
                                 "Female": CFG["visualization"]["colors"]["gender_female"]},
            category_orders={"Education Level": list(EDU_ORDER.keys())},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab: Geographic ──────────────────────────────────────────────────────────


def tab_geographic(df: pd.DataFrame) -> None:
    st.header("Geographic Analysis")

    metric = st.selectbox(
        "Map Metric",
        ["Avg Annual Income", "Job Count", "Avg Location Quotient"],
    )

    state_agg = df.groupby("State Abbreviation").agg(
        avg_income=("Annual Income", "mean"),
        job_count=("Annual Income", "count"),
        avg_lq=("Location Quotient", "mean"),
    ).reset_index()

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


def tab_predictor(df: pd.DataFrame, model: XGBRegressor) -> None:
    st.header("Salary Predictor")
    st.markdown(
        "Enter individual profile details to estimate annual income "
        "using the trained XGBoost model."
    )

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", sorted(df["State Abbreviation"].unique()))
        occupation = st.selectbox(
            "Occupation", sorted(df["Occupation"].unique())
        )
        education = st.selectbox("Education Level", list(EDU_ORDER.keys()))
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    with col2:
        age = st.slider("Age", min_value=22, max_value=75, value=35)
        show_adv = st.checkbox("Show advanced inputs (BLS context)")
        if show_adv:
            employment = st.number_input(
                "State-Occupation Employment", value=1000, min_value=0
            )
            lq = st.number_input("Location Quotient", value=1.0, min_value=0.0, step=0.1)
            jobs_k = st.number_input("Jobs per 1,000", value=2.0, min_value=0.0, step=0.1)
            hourly_mean = st.number_input(
                "BLS Hourly Mean Wage ($)", value=60.0, min_value=0.0, step=1.0
            )
            annual_mean = st.number_input(
                "BLS Annual Mean Wage ($)", value=124000, min_value=0, step=1000
            )
        else:
            mask = (df["State Abbreviation"] == state) & (df["Occupation"] == occupation)
            subset = df[mask] if mask.sum() > 0 else df[df["State Abbreviation"] == state]
            if len(subset) == 0:
                subset = df
            employment = float(subset["Employment"].median())
            lq = float(subset["Location Quotient"].median())
            jobs_k = float(subset["Jobs per 1000"].median())
            hourly_mean = float(subset["Hourly Mean"].median())
            annual_mean = float(subset["Annual Mean Wage"].median())

    if st.button("Predict Salary", type="primary"):
        edu_ord = EDU_ORDER[education]
        gender_bin = 1 if gender == "Male" else 0
        region = REGION_MAP.get(state, "South")
        region_code = REGION_CODES.get(region, 0)
        occ_mean = df[df["Occupation"] == occupation]["Annual Income"].mean()
        state_mean = df[df["State Abbreviation"] == state]["Annual Income"].mean()

        row = pd.DataFrame(
            [[age, edu_ord, gender_bin, region_code, employment, lq,
              jobs_k, hourly_mean, annual_mean, occ_mean, state_mean]],
            columns=FEATURES_FULL,
        )
        prediction = model.predict(row)[0]

        st.success(f"Estimated Annual Income: **${prediction:,.0f}**")

        comparable = df[
            (df["Education Level"] == education)
            & (df["State Abbreviation"] == state)
        ]["Annual Income"]
        if len(comparable) > 0:
            pct = (comparable < prediction).mean() * 100
            st.info(
                f"This is higher than **{pct:.1f}%** of {education} earners in {state}."
            )


# ── Tab: Model Insights ──────────────────────────────────────────────────────


def tab_model(df: pd.DataFrame, model: XGBRegressor) -> None:
    st.header("Model Performance & Feature Importance")

    metrics = get_model_metrics(model, df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{metrics['R²']:.4f}")
    col2.metric("MAE", f"${metrics['MAE']:,.0f}")
    col3.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    col4.metric("Test Samples", f"{metrics['n_test']:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        feat_imp = pd.DataFrame(
            {"Feature": FEATURES_FULL, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=True)
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
        residuals = metrics["y_test"] - metrics["y_pred"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["y_pred"],
                y=residuals,
                mode="markers",
                marker={"opacity": 0.4, "size": 4, "color": "#2196F3"},
                name="Residuals",
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual Plot (Predicted vs Residual)",
            xaxis_title="Predicted Annual Income ($)",
            yaxis_title="Residual ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        x=metrics["y_test"],
        y=metrics["y_pred"],
        opacity=0.4,
        labels={"x": "Actual Income ($)", "y": "Predicted Income ($)"},
        title="Actual vs Predicted Annual Income",
    )
    max_val = max(metrics["y_test"].max(), metrics["y_pred"].max())
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


# ── Main App ─────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("💼 High-Paying Jobs in the US")
    st.markdown(
        "Interactive analysis of high-paying occupations (≥ $100K/yr) "
        "integrating **BLS OEWS** and **US Census** microdata. "
        "Use the sidebar to filter data."
    )

    df = load_data()
    filtered_df = sidebar(df)
    model = get_model(df)  # always train on full dataset

    if len(filtered_df) == 0:
        st.warning("No data matches current filters. Adjust the sidebar selections.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Geographic Analysis", "Salary Predictor", "Model Insights"]
    )

    with tab1:
        tab_overview(filtered_df)
    with tab2:
        tab_geographic(filtered_df)
    with tab3:
        tab_predictor(df, model)
    with tab4:
        tab_model(df, model)


if __name__ == "__main__":
    main()
