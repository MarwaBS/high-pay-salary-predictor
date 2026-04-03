"""
High-Paying Jobs US — Salary Prediction API
============================================
FastAPI service wrapping the trained XGBoost model.

Endpoints:
  GET  /            — API info
  GET  /health      — liveness probe
  GET  /meta        — valid states, occupations, education levels
  GET  /metrics     — Prometheus metrics (auto-instrumented)
  POST /predict     — salary prediction with contextual benchmarks and PI

Run locally:
  uvicorn api.main:app --reload --port 8000

Docker (via docker-compose):
  docker compose up api

Environment variables:
  CORS_ORIGINS   Comma-separated list of allowed origins.
                 Defaults to empty (rejects cross-origin requests).
                 Set to "*" for local dev or an explicit allow-list for production.
                 Example: CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
  API_KEY        If set, all /predict requests require X-API-Key header.
                 Unset = dev mode (no auth required).
  RATE_LIMIT     Per-IP rate limit for /predict (default: "60/minute").
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.drift import DriftMonitor
from api.schemas import HealthResponse, MetaResponse, PredictRequest, PredictResponse
from pipeline import (
    REGION_CODES,
    build_feature_row,
    compute_fallback_means,
    engineer_features,
    get_bls_defaults,
    load_group_means,
    load_metrics,
    load_model,
)

# ── Structured JSON Logging ──────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """Emit logs as single-line JSON for machine parsing and log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ── API Key Auth ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(_api_key_header)) -> str | None:
    """Validate API key if API_KEY is configured; skip in dev mode (unset)."""
    if not API_KEY:
        return None  # dev mode: no auth required
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Rate Limiting ────────────────────────────────────────────────────────────

RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

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
    pi_offset_10: float = 0.0
    pi_offset_90: float = 0.0
    occupations: list[str] = []
    region_codes: dict[str, int] = {}
    occ_means: dict[str, float] = {}
    state_means: dict[str, float] = {}
    drift_monitor: DriftMonitor | None = None


state = AppState()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    logger.info("Starting up: loading dataset, model, group means, and metrics…")

    # Load training-set group means for consistent target encoding at inference
    group_means = load_group_means(str(ROOT / CFG["model"]["group_means_path"]))
    state.occ_means = group_means["occ_means"]
    state.state_means = group_means["state_means"]

    # Engineer features using saved training means (no leakage at inference)
    df_raw = pd.read_csv(ROOT / CFG["data"]["cleaned"])
    df_eng = engineer_features(
        df_raw,
        EDU_ORDER,
        REGION_MAP,
        occ_means=state.occ_means,
        state_means=state.state_means,
    )

    state.df = df_eng
    state.model = load_model(str(ROOT / CFG["model"]["model_path"]))
    state.occupations = sorted(df_eng["Occupation"].unique().tolist())
    state.region_codes = REGION_CODES

    # Load empirical prediction-interval offsets from training artefacts
    # (offsets are in dollar space — applied after expm1 back-transform)
    metrics = load_metrics(str(ROOT / CFG["model"]["metrics_path"]))
    state.pi_offset_10 = metrics.get("pi_offset_10", 0.0)
    state.pi_offset_90 = metrics.get("pi_offset_90", 0.0)

    # Load drift baseline (optional — created by train_model.py)
    baseline_path = ROOT / "models" / "baseline_stats.json"
    if baseline_path.exists():
        state.drift_monitor = DriftMonitor.from_baseline(str(baseline_path))
        logger.info("Drift monitor loaded from %s", baseline_path)
    else:
        logger.warning("No baseline_stats.json found — drift monitoring disabled")

    logger.info(
        "Ready — dataset rows: %d, occupations: %d, model features: %d, 80%% PI offsets: [%d, %+d]",
        len(df_eng),
        len(state.occupations),
        state.model.n_features_in_,
        int(state.pi_offset_10),
        int(state.pi_offset_90),
    )

    yield
    # ── shutdown ──
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

# CORS: default to closed (empty). Set CORS_ORIGINS="*" for local dev or an
# explicit comma-separated list for production.
_raw_origins = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins else []
if not CORS_ORIGINS:
    logger.warning("CORS_ORIGINS not set — cross-origin requests will be rejected")

app = FastAPI(
    title="US High-Pay Salary Predictor",
    description=(
        "Predicts annual income for high-paying ($100K+) US jobs using an "
        "XGBoost model trained on integrated BLS OEWS + US Census microdata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)


# ── Request ID + Logging Middleware ──────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(elapsed * 1000, 1),
        },
    )
    return response


# ── Prometheus Metrics ───────────────────────────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


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
@limiter.limit(RATE_LIMIT)
def predict(request: Request, req: PredictRequest, _key: str | None = Depends(verify_api_key)):
    """
    Predict annual income for a given demographic + occupational profile.

    Required: `state`, `occupation`, `education_level`, `gender`, `age`.
    Optional BLS context fields default to dataset medians for the given
    state/occupation combination when omitted.

    Returns the predicted salary alongside an empirical 80% prediction interval,
    percentile rank, and group benchmarks (median and mean for same state +
    education level).

    **Model notes**:
    - Trained on log1p(Annual Income); back-transformed with expm1 internally.
    - Group means (Occ_Mean_Income, State_Mean_Income) use training-set values
      for consistent encoding with no leakage.

    **Prediction interval**: derived from the 10th/90th percentiles of
    test-set residuals (in dollar space) at training time.  The interval is
    approximate — income residuals are heteroscedastic — but is clearly
    labelled as such in the response.
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
    bls = get_bls_defaults(df, req.state, req.occupation)
    employment = req.employment if req.employment is not None else bls["employment"]
    lq = req.location_quotient if req.location_quotient is not None else bls["location_quotient"]
    jobs_k = req.jobs_per_1000 if req.jobs_per_1000 is not None else bls["jobs_per_1000"]
    hourly_mean = req.hourly_mean if req.hourly_mean is not None else bls["hourly_mean"]

    # ── Encode categorical inputs ─────────────────────────────────────────────
    edu_ord = EDU_ORDER[req.education_level]
    gender_bin = 1 if req.gender == "Male" else 0
    region = REGION_MAP.get(req.state, "South")
    region_code = state.region_codes.get(region, 0)

    # Use training-set group means for consistent encoding with training
    occ_fallback, state_fallback = compute_fallback_means(
        {"occ_means": state.occ_means, "state_means": state.state_means}
    )
    occ_mean = state.occ_means.get(req.occupation, occ_fallback)
    state_mean_income = state.state_means.get(req.state, state_fallback)

    # ── Predict (model trained on log1p scale — back-transform with expm1) ───
    row = build_feature_row(
        age=req.age,
        edu_ord=edu_ord,
        gender_bin=gender_bin,
        region_code=region_code,
        employment=employment,
        lq=lq,
        jobs_k=jobs_k,
        hourly_mean=hourly_mean,
        occ_mean_income=occ_mean,
        state_mean_income=state_mean_income,
    )
    # Feed features to drift monitor (if baseline loaded)
    if state.drift_monitor is not None:
        state.drift_monitor.observe(row.iloc[0].to_dict())

    predicted = float(np.expm1(state.model.predict(row)[0]))
    logger.debug(
        "Prediction: state=%s occ=%s edu=%s gender=%s age=%d → $%.0f",
        req.state,
        req.occupation,
        req.education_level,
        req.gender,
        req.age,
        predicted,
    )

    # ── Empirical 80% prediction interval (dollar-space offsets) ─────────────
    pi_low = round(predicted + state.pi_offset_10, 2)
    pi_high = round(predicted + state.pi_offset_90, 2)

    # ── Contextual benchmarks ─────────────────────────────────────────────────
    group = df[(df["State Abbreviation"] == req.state) & (df["Education Level"] == req.education_level)][
        "Annual Income"
    ]

    if len(group) > 0:
        percentile = float((group < predicted).mean() * 100)
        group_median = float(group.median())
        group_mean = float(group.mean())
        group_size = len(group)
    else:
        percentile = 50.0
        group_median = float(df["Annual Income"].median())
        group_mean = float(df["Annual Income"].mean())
        group_size = 0

    return PredictResponse(
        predicted_salary=round(predicted, 2),
        prediction_interval_low=pi_low,
        prediction_interval_high=pi_high,
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


@app.get("/drift", tags=["Monitoring"])
async def drift_report():
    """Return feature drift report comparing recent predictions to training baseline.

    Requires ``models/baseline_stats.json`` (generated by ``train_model.py``).
    Returns z-score deviation per feature and a boolean ``any_drifted`` flag.
    """
    if state.drift_monitor is None:
        return {"status": "disabled", "message": "No baseline_stats.json — run train_model.py to generate it"}
    return state.drift_monitor.check_drift()
