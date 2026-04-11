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
  GET  /drift       — feature drift report (cluster-wide with Redis)

Run locally:
  uvicorn api.main:app --reload --port 8000

Docker (via docker-compose):
  docker compose up api

Environment variables:
  CORS_ORIGINS          Comma-separated list of allowed origins. Defaults to
                        empty (rejects cross-origin requests). Set to "*"
                        for local dev or an explicit allow-list for prod.
  API_KEY               If set, all /predict requests require X-API-Key.
                        Unset = dev mode (no auth).
  RATE_LIMIT            Per-IP rate limit for /predict (default: "60/minute").
  TRUSTED_PROXY_HOPS    Number of reverse proxies in front of the API. The
                        rate limiter and logging read the Nth-from-last
                        entry of X-Forwarded-For. Default: 0 (bind to the
                        direct client.host — dev / no proxy).
  REDIS_URL             Optional. Enables the PredictionCache and the
                        shared drift monitor window. Default: no-op.
  CACHE_TTL             Prediction cache TTL in seconds. Default: 3600.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from api.cache import PredictionCache
from api.drift import DriftMonitor
from api.inference import (
    GroupStats,
    build_benchmark_lookup,
    build_response,
    encode_features,
    lookup_benchmarks,
    run_model,
)
from api.schemas import HealthResponse, MetaResponse, PredictRequest, PredictResponse
from config_schema import ProjectConfig
from pipeline import (
    REGION_CODES,
    engineer_features,
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


# ── Rate Limiting (proxy-aware) ──────────────────────────────────────────────
#
# ``slowapi.util.get_remote_address`` reads ``request.client.host`` which is
# the IP of whatever spoke directly to uvicorn — behind any ingress, that
# is a single internal IP, collapsing every caller onto one bucket. We
# read ``X-Forwarded-For`` instead and peel off ``TRUSTED_PROXY_HOPS`` entries
# from the right (right-most entries are the ones added by trusted hops).

RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))


def _client_ip(request: Request) -> str:
    """Return the client IP, respecting TRUSTED_PROXY_HOPS for X-Forwarded-For."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff and TRUSTED_PROXY_HOPS > 0:
        # XFF is "client, proxy1, proxy2, ..." — peel TRUSTED_PROXY_HOPS
        # proxies off the right end and take the next entry as the client.
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        idx = max(0, len(hops) - 1 - TRUSTED_PROXY_HOPS)
        return hops[idx]
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip, default_limits=[RATE_LIMIT])

# ── Prediction Cache ─────────────────────────────────────────────────────────
# Redis-backed deterministic prediction cache. Consulted inside predict()
# after domain validation. Graceful no-op when REDIS_URL is unset; tests
# monkeypatch ``api.main.cache`` (see tests/test_api.py::TestPredictionCache).

cache = PredictionCache()

# ── Config (validated at import) ─────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

# Parse + validate through the Pydantic schema in one step. Fail fast on
# typos or invalid values at import time so the k8s liveness probe catches
# broken config before traffic hits the pod.
VALIDATED_CFG = ProjectConfig.from_yaml(ROOT / "config.yaml")
CFG: dict = yaml.safe_load((ROOT / "config.yaml").read_text())

EDU_ORDER = VALIDATED_CFG.education_order
REGION_MAP = {s: r for r, states in VALIDATED_CFG.regions.items() for s in states}

VALID_EDUCATION = list(EDU_ORDER.keys())
VALID_STATES = sorted({s for states in VALIDATED_CFG.regions.values() for s in states})

# ── Application state (loaded once at startup) ────────────────────────────────


class AppState:
    """Module-global singleton holding state loaded at startup."""

    df: pd.DataFrame | None = None
    model = None
    pi_offset_10: float = 0.0
    pi_offset_90: float = 0.0
    occupations: list[str] = []
    region_codes: dict[str, int] = {}
    occ_means: dict[str, float] = {}
    state_means: dict[str, float] = {}
    drift_monitor: DriftMonitor | None = None
    benchmark_lookup: dict[tuple[str, str], GroupStats] = {}


state = AppState()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    logger.info("Starting up: loading dataset, model, group means, and metrics…")

    # Load training-set group means for consistent target encoding at inference
    group_means = load_group_means(str(ROOT / VALIDATED_CFG.model.group_means_path))
    state.occ_means = group_means["occ_means"]
    state.state_means = group_means["state_means"]

    # Engineer features using saved training means (no leakage at inference)
    df_raw = pd.read_csv(ROOT / VALIDATED_CFG.data.cleaned)
    df_eng = engineer_features(
        df_raw,
        EDU_ORDER,
        REGION_MAP,
        occ_means=state.occ_means,
        state_means=state.state_means,
    )

    state.df = df_eng
    state.model = load_model(str(ROOT / VALIDATED_CFG.model.model_path))
    state.occupations = sorted(df_eng["Occupation"].unique().tolist())
    state.region_codes = REGION_CODES

    # Precompute (state, education) benchmark lookup so /predict becomes
    # an O(log n) dict get + binary search instead of a per-request
    # full-DataFrame mask.
    state.benchmark_lookup = build_benchmark_lookup(df_eng)
    logger.info("Benchmark lookup built with %d (state, education) cells", len(state.benchmark_lookup))

    # Load empirical prediction-interval offsets from training artefacts
    metrics = load_metrics(str(ROOT / VALIDATED_CFG.model.metrics_path))
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
    # Explicit header list — mixing "*" with explicit values would be
    # meaningless because "*" already matches everything.
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
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


# ── Validation helper ────────────────────────────────────────────────────────


def _validate_domain(req: PredictRequest) -> None:
    """Domain validation against loaded data. Raises 422 on unknown values."""
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
        states=VALID_STATES,
        occupations=state.occupations,
        education_levels=VALID_EDUCATION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@limiter.limit(RATE_LIMIT)
def predict(request: Request, req: PredictRequest, _key: str | None = Depends(verify_api_key)):
    """Predict annual income for a given demographic + occupational profile.

    Required: ``state``, ``occupation``, ``education_level``, ``gender``, ``age``.
    Optional BLS context fields default to dataset medians for the given
    state/occupation combination when omitted.

    Returns the predicted salary alongside an empirical 80% prediction
    interval, percentile rank, and group benchmarks (median and mean for
    same state + education level).

    Notes
    -----
    - Trained on ``log1p(Annual Income)``; back-transformed internally.
    - Group means (``Occ_Mean_Income``, ``State_Mean_Income``) use
      training-set values for consistent encoding with no leakage.
    - Benchmark stats for the (state, education) group are precomputed at
      startup so this route is O(log n) per request, not O(dataset rows).
    """
    _validate_domain(req)

    # ── Cache lookup (keyed on validated request payload) ────────────────────
    cache_key = req.model_dump()
    cached = cache.get(cache_key)
    if cached is not None:
        return PredictResponse(**cached)

    # ── Feature encoding → inference → response ─────────────────────────────
    row = encode_features(
        req,
        state.df,
        edu_order=EDU_ORDER,
        region_map=REGION_MAP,
        region_codes=state.region_codes,
        occ_means=state.occ_means,
        state_means=state.state_means,
    )

    if state.drift_monitor is not None:
        state.drift_monitor.observe(row.iloc[0].to_dict())

    p10, p50, p90 = run_model(state.model, row)
    group_stats = lookup_benchmarks(state.benchmark_lookup, req.state, req.education_level)
    response = build_response(
        req,
        p10=p10,
        p50=p50,
        p90=p90,
        group_stats=group_stats,
    )

    # Persist to cache for subsequent identical requests (no-op if disabled).
    cache.set(cache_key, response.model_dump())
    return response


@app.get("/drift", tags=["Monitoring"])
async def drift_report():
    """Return feature drift report comparing recent predictions to training baseline.

    Requires ``models/baseline_stats.json`` (generated by ``train_model.py``).
    With ``REDIS_URL`` set, the observation window is shared across all
    replicas — the report is cluster-wide. Without Redis, the report is
    per-pod.
    """
    if state.drift_monitor is None:
        return {"status": "disabled", "message": "No baseline_stats.json — run train_model.py to generate it"}
    return state.drift_monitor.check_drift()
