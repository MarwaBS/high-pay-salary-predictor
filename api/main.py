"""
High-Paying Jobs US — Salary Prediction API
============================================
FastAPI service wrapping the trained XGBoost model.

Endpoints:
  GET  /            — API info
  GET  /health      — liveness probe
  GET  /meta        — valid states, occupations, education levels
  POST /predict     — salary prediction with contextual benchmarks

Run locally:
  uvicorn api.main:app --reload --port 8000

Docker (via docker-compose):
  docker compose up api

Environment variables:
  CORS_ORIGINS   Comma-separated list of allowed origins.
                 Defaults to "*" (open) — restrict in production.
                 Example: CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
"""
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np  # noqa: F401 — kept for type narrowing in callers
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import HealthResponse, MetaResponse, PredictRequest, PredictResponse
from pipeline import FEATURES_FULL, REGION_CODES, engineer_features

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

EDU_ORDER = CFG["education_order"]
REGION_MAP = {s: r for r, states in CFG["regions"].items() for s in states}

VALID_EDUCATION = list(EDU_ORDER.keys())
VALID_STATES = list({s for states in CFG["regions"].values() for s in states})

# ── Application state (loaded once at startup) ────────────────────────────────


class AppState:
    df: pd.DataFrame = None
    model = None
    occupations: list[str] = []
    region_codes: dict[str, int] = {}


state = AppState()


def _load_or_train_model(df: pd.DataFrame):
    """Load saved model or train a fresh one if pkl not found."""
    model_path = ROOT / CFG["model"]["model_path"]

    if model_path.exists():
        logger.info("Loading model from %s", model_path)
        with open(model_path, "rb") as f:
            return pickle.load(f)

    logger.info("No saved model found — training from scratch (first run)")
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    X = df[FEATURES_FULL]
    y = df["Annual Income"]
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=CFG["model"]["test_size"],
        random_state=CFG["model"]["random_state"],
    )
    model = XGBRegressor(
        n_estimators=CFG["model"]["n_estimators"],
        max_depth=CFG["model"]["max_depth"],
        learning_rate=CFG["model"]["learning_rate"],
        subsample=CFG["model"]["subsample"],
        colsample_bytree=CFG["model"]["colsample_bytree"],
        random_state=CFG["model"]["random_state"],
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Model trained and saved to %s", model_path)
    return model


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    logger.info("Starting up: loading dataset and model…")
    df_raw = pd.read_csv(ROOT / CFG["data"]["cleaned"])
    df_eng = engineer_features(df_raw, EDU_ORDER, REGION_MAP)

    state.df = df_eng
    state.model = _load_or_train_model(df_eng)
    state.occupations = sorted(df_eng["Occupation"].unique().tolist())
    # Use the same deterministic mapping as pipeline.REGION_CODES
    state.region_codes = REGION_CODES

    logger.info(
        "Ready — dataset rows: %d, occupations: %d, model features: %d",
        len(df_eng),
        len(state.occupations),
        state.model.n_features_in_,
    )

    yield
    # ── shutdown ──
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

# Allow CORS origins to be configured via environment variable so the same
# Docker image works in dev ("*") and production (explicit allow-list).
_raw_origins = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app = FastAPI(
    title="US High-Pay Salary Predictor",
    description=(
        "Predicts annual income for high-paying ($100K+) US jobs using an "
        "XGBoost model trained on integrated BLS OEWS + US Census microdata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "US High-Pay Salary Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness probe — returns model load status and dataset size."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        dataset_rows=len(state.df) if state.df is not None else 0,
    )


@app.get("/meta", response_model=MetaResponse, tags=["Meta"])
async def meta():
    """Return all valid values for state, occupation, and education_level fields."""
    return MetaResponse(
        states=sorted(VALID_STATES),
        occupations=state.occupations,
        education_levels=VALID_EDUCATION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    """
    Predict annual income for a given demographic + occupational profile.

    Required: `state`, `occupation`, `education_level`, `gender`, `age`.
    Optional BLS context fields default to dataset medians for the given
    state/occupation combination when omitted.

    Returns the predicted salary alongside percentile rank and group benchmarks
    (median and mean for same state + education level).
    """
    df = state.df

    # ── Validate inputs ───────────────────────────────────────────────────────
    if req.state not in VALID_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown state '{req.state}'. Use /meta to see valid values.",
        )
    if req.occupation not in state.occupations:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown occupation '{req.occupation}'. Use /meta to see valid values.",
        )
    if req.education_level not in VALID_EDUCATION:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown education_level '{req.education_level}'. Valid: {VALID_EDUCATION}",
        )

    # ── Derive BLS context defaults from dataset medians ─────────────────────
    mask = (df["State Abbreviation"] == req.state) & (df["Occupation"] == req.occupation)
    subset = df[mask] if mask.sum() > 0 else df[df["State Abbreviation"] == req.state]
    if len(subset) == 0:
        subset = df  # final fallback: global medians

    employment    = req.employment           if req.employment           is not None else float(subset["Employment"].median())
    lq            = req.location_quotient    if req.location_quotient    is not None else float(subset["Location Quotient"].median())
    jobs_k        = req.jobs_per_1000        if req.jobs_per_1000        is not None else float(subset["Jobs per 1000"].median())
    hourly_mean   = req.hourly_mean          if req.hourly_mean          is not None else float(subset["Hourly Mean"].median())
    annual_mean_w = req.annual_mean_wage     if req.annual_mean_wage     is not None else float(subset["Annual Mean Wage"].median())

    # ── Encode categorical inputs ─────────────────────────────────────────────
    edu_ord     = EDU_ORDER[req.education_level]
    gender_bin  = 1 if req.gender == "Male" else 0
    region      = REGION_MAP.get(req.state, "South")
    region_code = state.region_codes.get(region, 0)

    occ_mean   = float(df[df["Occupation"] == req.occupation]["Annual Income"].mean())
    state_mean = float(df[df["State Abbreviation"] == req.state]["Annual Income"].mean())

    # ── Predict ───────────────────────────────────────────────────────────────
    row = pd.DataFrame(
        [[req.age, edu_ord, gender_bin, region_code, employment, lq,
          jobs_k, hourly_mean, annual_mean_w, occ_mean, state_mean]],
        columns=FEATURES_FULL,
    )
    predicted = float(state.model.predict(row)[0])
    logger.debug(
        "Prediction: state=%s occ=%s edu=%s gender=%s age=%d → $%.0f",
        req.state, req.occupation, req.education_level, req.gender, req.age, predicted,
    )

    # ── Contextual benchmarks ─────────────────────────────────────────────────
    group = df[
        (df["State Abbreviation"] == req.state) &
        (df["Education Level"] == req.education_level)
    ]["Annual Income"]

    if len(group) > 0:
        percentile   = float((group < predicted).mean() * 100)
        group_median = float(group.median())
        group_mean   = float(group.mean())
        group_size   = len(group)
    else:
        percentile   = 50.0
        group_median = float(df["Annual Income"].median())
        group_mean   = float(df["Annual Income"].mean())
        group_size   = 0

    return PredictResponse(
        predicted_salary=round(predicted, 2),
        percentile_in_group=round(percentile, 1),
        group_median=round(group_median, 2),
        group_mean=round(group_mean, 2),
        group_size=group_size,
        state=req.state,
        occupation=req.occupation,
        education_level=req.education_level,
        gender=req.gender,
        age=req.age,
    )
