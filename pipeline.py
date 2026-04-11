"""
pipeline.py
-----------
Single source of truth for:
  - Feature constants (FEATURES_FULL, FEATURES_DEMO, REGION_CODES)
  - Feature-engineering function (engineer_features)
  - Group-means helpers (compute_group_means, save/load_group_means)
  - Model save / load helpers (no pickle — XGBoost native + JSON)
  - build_feature_row helper (shared by API + dashboard)

Design notes
------------
* ``Annual Mean Wage`` was removed from FEATURES_FULL / FEATURES_DEMO because
  it is a near-perfect linear transformation of ``Hourly Mean`` (×2080,
  corr ≈ 1.0000, VIF ≈ 5.4×10⁸).  Keeping both distorts feature-importance
  scores and wastes a feature slot with zero new information.

* ``Occ_Mean_Income`` and ``State_Mean_Income`` are computed from the **training
  set only** during model training (see scripts/train_quantile.py) and saved as
  ``models/group_means.json``.  At inference time the API loads those saved
  means so the encoding is consistent with training. This eliminates the
  target-encoding leakage that arises from computing group means on the full
  dataset (including the test split) before the train/test split.

* The model is trained on ``log1p(Annual Income)`` and predicts in log space;
  callers must ``numpy.expm1()`` the raw output to get dollar predictions.

Shared across the entire project:

  - api/main.py
  - streamlit_app.py
  - scripts/train_quantile.py
  - tests/test_pipeline.py
  - 04_salary_prediction_model.ipynb (historical v1 EDA)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

#: Full feature vector used by the production XGBoost model.
#: ``Annual Mean Wage`` is intentionally excluded — it is a near-perfect linear
#: transform of ``Hourly Mean`` (correlation 0.9999, VIF ≈ 5.4×10⁸).
FEATURES_FULL: list[str] = [
    "Age",
    "Education_Ord",
    "Gender_Bin",
    "Region_Code",
    "Employment",
    "Location Quotient",
    "Jobs per 1000",
    "Hourly Mean",
    "Occ_Mean_Income",
    "State_Mean_Income",
]

#: Demographic-only feature vector (no BLS context) used in the
#: "fairness / demographic gap" model in notebook 4.
#: ``Annual Mean Wage`` also excluded here for the same collinearity reason.
FEATURES_DEMO: list[str] = [
    "Age",
    "Education_Ord",
    "Gender_Bin",
    "Employment",
    "Location Quotient",
    "Jobs per 1000",
    "Hourly Mean",
    "Occ_Mean_Income",
    "State_Mean_Income",
]

# ---------------------------------------------------------------------------
# Deterministic region → integer encoding
# ---------------------------------------------------------------------------
# Alphabetical order matches both pd.Categorical default and the API's
# enumerate(sorted(...)) approach — guaranteeing consistent encoding across
# training (notebook), serving (API), and the dashboard (Streamlit).

REGION_CODES: dict[str, int] = {
    "Midwest": 0,
    "Northeast": 1,
    "South": 2,
    "West": 3,
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "Education Level",
    "Gender",
    "State Abbreviation",
    "Occupation",
    "Annual Income",
]


def engineer_features(
    df: pd.DataFrame,
    edu_order: dict[str, int],
    region_map: dict[str, str],
    occ_means: dict[str, float] | None = None,
    state_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return *df* with all model-ready derived columns appended.

    Added columns
    -------------
    Education_Ord     : int   ordinal encoding of Education Level (1–4)
    Gender_Bin        : int   1 = Male, 0 = Female
    Region            : str   US Census four-region label
    Region_Code       : int   deterministic integer from REGION_CODES
    Occ_Mean_Income   : float mean Annual Income for that Occupation
    State_Mean_Income : float mean Annual Income for that State

    Parameters
    ----------
    df          : raw or cleaned dataset (must contain the standard columns)
    edu_order   : mapping from education label → ordinal integer (from config.yaml)
    region_map  : mapping from state abbreviation → region label (from config.yaml)
    occ_means   : precomputed occupation→mean_income mapping (from training set).
                  If *None*, means are computed from *df* (suitable for the full
                  deployed dataset at API startup; not for model evaluation).
    state_means : precomputed state→mean_income mapping (from training set).
                  Same semantics as *occ_means*.

    Raises
    ------
    ValueError  if any required column is missing from *df*.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"engineer_features: missing required columns: {missing}")

    out = df.copy()
    out["Education_Ord"] = out["Education Level"].map(edu_order)
    out["Gender_Bin"] = (out["Gender"] == "Male").astype(int)
    out["Region"] = out["State Abbreviation"].map(region_map)
    out["Region_Code"] = out["Region"].map(REGION_CODES).fillna(0).astype(int)

    if occ_means is not None:
        out["Occ_Mean_Income"] = out["Occupation"].map(occ_means).fillna(pd.Series(occ_means).mean())
    else:
        out["Occ_Mean_Income"] = out.groupby("Occupation")["Annual Income"].transform("mean")

    if state_means is not None:
        out["State_Mean_Income"] = out["State Abbreviation"].map(state_means).fillna(pd.Series(state_means).mean())
    else:
        out["State_Mean_Income"] = out.groupby("State Abbreviation")["Annual Income"].transform("mean")

    return out


def compute_group_means(df_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute occupation and state mean incomes from the *training set only*.

    Call this **after** the train/test split to avoid target-encoding leakage.
    Save the result with :func:`save_group_means` and pass it back into
    :func:`engineer_features` for both the train and test sets.

    Returns
    -------
    dict with keys ``"occ_means"`` and ``"state_means"``.
    """
    return {
        "occ_means": df_train.groupby("Occupation")["Annual Income"].mean().to_dict(),
        "state_means": df_train.groupby("State Abbreviation")["Annual Income"].mean().to_dict(),
    }


# ---------------------------------------------------------------------------
# Shared prediction helper — eliminates duplication between API and dashboard
# ---------------------------------------------------------------------------


def build_feature_row(
    *,
    age: int,
    edu_ord: int,
    gender_bin: int,
    region_code: int,
    employment: float,
    lq: float,
    jobs_k: float,
    hourly_mean: float,
    occ_mean_income: float,
    state_mean_income: float,
) -> pd.DataFrame:
    """Return a single-row DataFrame ready for model.predict().

    All callers (api/main.py, streamlit_app.py) must go through this
    function so the column order always matches FEATURES_FULL.

    Note: the model is trained on log1p(Annual Income).  Callers must
    apply ``numpy.expm1()`` to the raw prediction to get dollar values.
    """
    return pd.DataFrame(
        [
            [
                age,
                edu_ord,
                gender_bin,
                region_code,
                employment,
                lq,
                jobs_k,
                hourly_mean,
                occ_mean_income,
                state_mean_income,
            ]
        ],
        columns=FEATURES_FULL,
    )


# ---------------------------------------------------------------------------
# Shared fallback helpers — eliminates duplication between API and dashboard
# ---------------------------------------------------------------------------


def compute_fallback_means(
    group_means: dict[str, dict[str, float]],
) -> tuple[float, float]:
    """Return (occ_fallback, state_fallback) as averages of all group means.

    Used when a specific occupation or state has no entry in the training-set
    group means (e.g. unseen at training time).
    """
    occ_fallback = float(np.mean(list(group_means["occ_means"].values())))
    state_fallback = float(np.mean(list(group_means["state_means"].values())))
    return occ_fallback, state_fallback


def get_bls_defaults(
    df: pd.DataFrame,
    state: str,
    occupation: str,
    state_col: str = "State Abbreviation",
) -> dict[str, float]:
    """Return median BLS context for a state+occupation pair with progressive fallback.

    Lookup order: (state AND occupation) → (state only) → (global).
    Returns dict with keys: employment, location_quotient, jobs_per_1000, hourly_mean.
    """
    mask_both = (df[state_col] == state) & (df["Occupation"] == occupation)
    subset = df[mask_both] if mask_both.sum() > 0 else df[df[state_col] == state]
    if len(subset) == 0:
        subset = df  # final fallback: global medians

    return {
        "employment": float(subset["Employment"].median()),
        "location_quotient": float(subset["Location Quotient"].median()),
        "jobs_per_1000": float(subset["Jobs per 1000"].median()),
        "hourly_mean": float(subset["Hourly Mean"].median()),
    }


# ---------------------------------------------------------------------------
# Model persistence — no pickle
# ---------------------------------------------------------------------------
# Pickle is Python-version-sensitive and can execute arbitrary code on load.
# We use XGBoost's native binary format (.ubj) for models and plain JSON
# for the feature list and metrics, making artefacts portable and auditable.


def save_model(model: XGBRegressor, path: str) -> None:
    """Save an XGBoost model using its native binary format (.ubj)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: str) -> XGBRegressor:
    """Load an XGBoost model from its native binary format.

    Raises
    ------
    FileNotFoundError  if *path* does not exist (e.g. model not yet trained).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model artefact not found: {p}. Run 'make model' (or 'python -m scripts.train_quantile') to generate it."
        )
    m = XGBRegressor()
    m.load_model(str(p))
    return m


def save_features(features: list[str], path: str) -> None:
    """Persist the feature name list as plain JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(features, f, indent=2)


def load_features(path: str) -> list[str]:
    """Load the feature name list from JSON."""
    with open(path) as f:
        return json.load(f)


def save_metrics(metrics: dict, path: str) -> None:
    """Persist model evaluation metrics as plain JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    """Load model evaluation metrics from JSON. Returns empty dict on missing file."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_group_means(group_means: dict, path: str) -> None:
    """Persist occupation and state mean-income mappings as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(group_means, f, indent=2)


def load_group_means(path: str) -> dict[str, dict[str, float]]:
    """Load occupation and state mean-income mappings from JSON.

    Raises
    ------
    FileNotFoundError  if *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Group means artefact not found: {p}. "
            "Run 'make model' (or 'python -m scripts.train_quantile') to generate it."
        )
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Quantile prediction helpers
# ---------------------------------------------------------------------------
# The production model is trained with ``objective="reg:quantileerror"`` and
# ``quantile_alpha=[0.1, 0.5, 0.9]`` by scripts/train_quantile.py. At inference
# it returns a (n, 3) array where columns are P10, P50, P90 in log1p space.
#
# If a legacy point-estimate model is loaded (1-D output), ``predict_quantiles``
# gracefully falls back to returning the same value for all three quantiles
# so callers never crash — the API additionally surfaces a flag indicating
# whether the range is real or a degenerate fallback.


def predict_quantiles(model: XGBRegressor, row: pd.DataFrame) -> tuple[float, float, float]:
    """Return (p10, p50, p90) dollar predictions for a single-row input.

    Works with both the new multi-quantile model (preferred) and the legacy
    point estimator (fallback — returns the point value for all three).
    """
    raw = np.asarray(model.predict(row))
    if raw.ndim == 2 and raw.shape[1] == 3:
        p10, p50, p90 = raw[0]
    elif raw.ndim == 1:
        # Legacy point model — degenerate interval
        p50 = float(raw[0])
        p10, p90 = p50, p50
    else:
        raise ValueError(f"Unexpected model.predict() shape: {raw.shape}")

    return float(np.expm1(p10)), float(np.expm1(p50)), float(np.expm1(p90))


def is_quantile_model(model: XGBRegressor) -> bool:
    """True if the model was trained with the multi-quantile objective."""
    params = model.get_params()
    return params.get("objective") == "reg:quantileerror"
