"""
pipeline.py
-----------
Single source of truth for feature constants and the feature-engineering
function shared across the entire project:

  - api/main.py
  - streamlit_app.py
  - scripts/train_model.py
  - tests/test_pipeline.py
  - 04_salary_prediction_model.ipynb

Keeping this here eliminates the 4-way duplication that previously existed
and ensures every layer of the stack uses an identical feature set.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

#: Full feature vector used by the production XGBoost model.
FEATURES_FULL: list[str] = [
    "Age",
    "Education_Ord",
    "Gender_Bin",
    "Region_Code",
    "Employment",
    "Location Quotient",
    "Jobs per 1000",
    "Hourly Mean",
    "Annual Mean Wage",
    "Occ_Mean_Income",
    "State_Mean_Income",
]

#: Demographic-only feature vector (no BLS context) used in the
#: "fairness / demographic gap" model in notebook 4.
FEATURES_DEMO: list[str] = [
    "Age",
    "Education_Ord",
    "Gender_Bin",
    "Employment",
    "Location Quotient",
    "Jobs per 1000",
    "Hourly Mean",
    "Annual Mean Wage",
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

def engineer_features(
    df: pd.DataFrame,
    edu_order: dict[str, int],
    region_map: dict[str, str],
) -> pd.DataFrame:
    """Return *df* with all model-ready derived columns appended.

    Added columns
    -------------
    Education_Ord   : int   ordinal encoding of Education Level (1–4)
    Gender_Bin      : int   1 = Male, 0 = Female
    Region          : str   US Census four-region label
    Region_Code     : int   deterministic integer from REGION_CODES
    Occ_Mean_Income : float mean Annual Income for that Occupation
    State_Mean_Income: float mean Annual Income for that State

    Parameters
    ----------
    df         : raw or cleaned dataset (must contain the standard columns)
    edu_order  : mapping from education label → ordinal integer (from config.yaml)
    region_map : mapping from state abbreviation → region label (from config.yaml)
    """
    out = df.copy()
    out["Education_Ord"] = out["Education Level"].map(edu_order)
    out["Gender_Bin"] = (out["Gender"] == "Male").astype(int)
    out["Region"] = out["State Abbreviation"].map(region_map)
    out["Region_Code"] = (
        out["Region"].map(REGION_CODES).fillna(0).astype(int)
    )
    out["Occ_Mean_Income"] = (
        out.groupby("Occupation")["Annual Income"].transform("mean")
    )
    out["State_Mean_Income"] = (
        out.groupby("State Abbreviation")["Annual Income"].transform("mean")
    )
    return out
