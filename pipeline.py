"""
pipeline.py
-----------
Single source of truth for:
  - Feature constants (FEATURES_FULL, FEATURES_DEMO, REGION_CODES)
  - Feature-engineering function (engineer_features)
  - Group-means helpers (compute_group_means, save/load_group_means)
  - Model save / load helpers (no pickle — XGBoost native + JSON)
  - Quantile prediction helpers (predict_quantiles, predict_quantiles_batch)

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
  - notebooks/04_salary_prediction_model.ipynb (historical v1 EDA)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


def sha256_file(path: str | Path) -> str:
    """Full SHA-256 hex digest of a file's bytes.

    The content address used to pin a served artefact to the exact bytes
    training recorded: the trainer stamps these into model_metrics.json, the
    API re-hashes on load and crashes on mismatch, and CI verifies committed
    bytes against them.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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

#: Gender_Bin is a binary Male/Female encoding — the only two values in the
#: training distribution. Anything else is rejected rather than silently folded
#: into the Female bucket by the ``== "Male"`` comparison.
_KNOWN_GENDERS: frozenset[str] = frozenset({"Male", "Female"})


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
    ValueError  if any required column is missing, or if any Education Level,
                Gender, or State Abbreviation value has no mapping — encoding
                an unknown category silently (NaN, or a Region_Code 0 collision)
                would ship a quietly-degraded model on a config typo.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"engineer_features: missing required columns: {missing}")

    bad_edu = sorted(set(df["Education Level"].unique()) - set(edu_order), key=str)
    if bad_edu:
        raise ValueError(
            f"engineer_features: unmapped Education Level values {bad_edu}; expected one of {sorted(edu_order)}"
        )
    bad_gender = sorted(set(df["Gender"].unique()) - _KNOWN_GENDERS, key=str)
    if bad_gender:
        raise ValueError(
            f"engineer_features: unexpected Gender values {bad_gender}; expected one of {sorted(_KNOWN_GENDERS)}"
        )
    bad_states = sorted(set(df["State Abbreviation"].unique()) - set(region_map), key=str)
    if bad_states:
        raise ValueError(f"engineer_features: unmapped State Abbreviation values {bad_states}")

    out = df.copy()
    out["Education_Ord"] = out["Education Level"].map(edu_order)
    out["Gender_Bin"] = (out["Gender"] == "Male").astype(int)
    out["Region"] = out["State Abbreviation"].map(region_map)
    out["Region_Code"] = out["Region"].map(REGION_CODES).astype(int)

    if occ_means is not None:
        occ_fallback = float(np.mean(list(occ_means.values()))) if occ_means else 0.0
        out["Occ_Mean_Income"] = out["Occupation"].map(occ_means).fillna(occ_fallback)
    else:
        out["Occ_Mean_Income"] = out.groupby("Occupation")["Annual Income"].transform("mean")

    if state_means is not None:
        state_fallback = float(np.mean(list(state_means.values()))) if state_means else 0.0
        out["State_Mean_Income"] = out["State Abbreviation"].map(state_means).fillna(state_fallback)
    else:
        out["State_Mean_Income"] = out.groupby("State Abbreviation")["Annual Income"].transform("mean")

    return out


def train_test_positions(n_rows: int, *, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) positional row indices for the project's single split.

    The trainer and the Streamlit dashboard both derive *which rows are test*
    from this one function, so a change to the split (adding stratification, a
    different seed) moves them together — the dashboard can never silently
    report residuals on a train-contaminated "test" set because it re-derived
    the split a second, diverging way. sklearn is imported lazily so importing
    ``pipeline`` on the serving hot path does not pull it in.
    """
    from sklearn.model_selection import train_test_split

    train_pos, test_pos = train_test_split(np.arange(n_rows), test_size=test_size, random_state=random_state)
    return np.asarray(train_pos), np.asarray(test_pos)


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
        "occ_means": {str(k): float(v) for k, v in df_train.groupby("Occupation")["Annual Income"].mean().items()},
        "state_means": {
            str(k): float(v) for k, v in df_train.groupby("State Abbreviation")["Annual Income"].mean().items()
        },
    }


# ---------------------------------------------------------------------------
# Shared fallback helpers — eliminates duplication between API and dashboard
# ---------------------------------------------------------------------------


def compute_fallback_means(
    group_means: dict[str, dict[str, float]],
) -> tuple[float, float]:
    """Return (occ_fallback, state_fallback) as averages of all group means.

    Used when a specific occupation or state has no entry in the training-set
    group means (e.g. unseen at training time).

    Raises
    ------
    ValueError  if either group-mean dict is empty — averaging an empty set
                yields NaN, which the API would then inject as the fallback
                feature value for every unseen occupation/state. Fail at
                startup instead of serving silent NaNs.
    """
    occ = list(group_means["occ_means"].values())
    state = list(group_means["state_means"].values())
    if not occ or not state:
        raise ValueError(
            "compute_fallback_means: group_means artefact has empty occ_means or state_means "
            "— retrain (`python -m scripts.train_quantile`) to produce non-empty means."
        )
    return float(np.mean(occ)), float(np.mean(state))


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


def save_classifier(model: XGBClassifier, path: str) -> None:
    """Save an XGBoost binary classifier using its native ``.ubj`` format.

    Separate helper from ``save_model`` so the type annotation on the
    caller side makes the intent clear — regressor artefacts and
    classifier artefacts live at different paths and must not collide.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_classifier(path: str) -> XGBClassifier:
    """Load an XGBoost binary classifier from ``.ubj``.

    Raises
    ------
    FileNotFoundError  if *path* does not exist — meaning the classifier
        head has not been trained yet. Callers that want backwards
        compatibility with pre-Phase-1 artefacts should catch this
        exception and degrade gracefully (the API does exactly that:
        the ``p_above_premium_threshold`` field becomes ``None``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Classifier artefact not found: {p}. Run 'make model' (or 'python -m scripts.train_quantile') to generate it."
        )
    m = XGBClassifier()
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
        features: list[str] = json.load(f)
    return features


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
        metrics: dict = json.load(f)
    return metrics


def save_group_means(group_means: dict, path: str) -> None:
    """Persist occupation and state mean-income mappings as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(group_means, f, indent=2)


def save_conformal(delta: float, path: str, *, target_coverage: float, n_scores: int) -> None:
    """Persist the split-conformal interval margin (log space) plus its provenance."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "conformal_delta": delta,
        "target_coverage": target_coverage,
        "method": "cross-conformal (5-fold CQR, log1p space)",
        "n_scores": n_scores,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_conformal_delta(path: str) -> float:
    """Load the split-conformal interval margin.

    Raises
    ------
    FileNotFoundError  if *path* does not exist. The served interval claims a
    calibrated coverage that only holds with this margin applied, so a missing
    artefact is a hard failure, not a silent fall-back to the raw interval.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Conformal margin artefact not found: {p}. "
            "Run 'make model' (or 'python -m scripts.train_quantile') to generate it."
        )
    with open(p) as f:
        payload = json.load(f)
    return float(payload["conformal_delta"])


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
        group_means: dict[str, dict[str, float]] = json.load(f)
    return group_means


# ---------------------------------------------------------------------------
# Quantile prediction helpers
# ---------------------------------------------------------------------------
# The production model is trained with ``objective="reg:quantileerror"`` and
# ``quantile_alpha=[0.1, 0.5, 0.9]`` by scripts/train_quantile.py, so it emits a
# (n, 3) array of P10/P50/P90 in log1p space. A non-quantile model is refused at
# API startup (``is_quantile_model``), so these helpers require the (n, 3) shape
# and raise on anything else rather than silently degrading to a (p, p, p) point.


def predict_quantiles_batch(model: XGBRegressor, rows: pd.DataFrame, *, conformal_delta: float = 0.0) -> np.ndarray:
    """Return an (n, 3) array of (p10, p50, p90) dollar predictions for a frame.

    Single source of truth for parsing the multi-quantile output: expm1's the
    (n, 3) log-space prediction back to dollars. Raises on any other shape — a
    legacy point model is refused at startup, so a non-(n, 3) output here is a
    real fault, not a fallback.

    ``conformal_delta`` widens the P10/P90 bounds symmetrically in log space by
    the split-conformal margin (see ``scripts.train_quantile`` cross-conformal
    calibration) so the served interval reaches its nominal coverage; the raw
    quantiles under-cover by a couple of points. P50 is never shifted. The
    default 0.0 leaves the raw interval unchanged.
    """
    raw = np.asarray(model.predict(rows))
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"Expected multi-quantile model output (n, 3), got shape {raw.shape}")
    if conformal_delta:
        raw = raw.copy()
        raw[:, 0] -= conformal_delta
        raw[:, 2] += conformal_delta
    dollars: np.ndarray = np.expm1(raw)
    return dollars


def predict_quantiles(
    model: XGBRegressor, row: pd.DataFrame, *, conformal_delta: float = 0.0
) -> tuple[float, float, float]:
    """Return (p10, p50, p90) dollar predictions for a single-row input."""
    p10, p50, p90 = predict_quantiles_batch(model, row, conformal_delta=conformal_delta)[0]
    return float(p10), float(p50), float(p90)


def is_quantile_model(model: XGBRegressor) -> bool:
    """True if the model was trained with the multi-quantile objective."""
    params = model.get_params()
    return params.get("objective") == "reg:quantileerror"
