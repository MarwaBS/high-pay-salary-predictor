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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from api import __version__
from api.cache import PredictionCache
from api.drift import DriftMonitor
from api.inference import (
    BlsDefaults,
    GroupStats,
    build_benchmark_lookup,
    build_bls_defaults_lookup,
    build_response,
    encode_features,
    lookup_benchmarks,
    run_model,
)
from api.schemas import (
    HealthResponse,
    MetaResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from config_schema import ProjectConfig
from pipeline import (
    REGION_CODES,
    engineer_features,
    load_classifier,
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

EDU_ORDER = VALIDATED_CFG.education_order
REGION_MAP = {s: r for r, states in VALIDATED_CFG.regions.items() for s in states}

VALID_EDUCATION = list(EDU_ORDER.keys())
VALID_STATES = sorted({s for states in VALIDATED_CFG.regions.values() for s in states})

# ── Application state (loaded once at startup) ────────────────────────────────


@dataclass
class AppState:
    """Module-global singleton holding state loaded at startup.

    Uses ``@dataclass`` instead of class-level mutable defaults so fields
    have a proper per-instance lifetime and mypy reasons about them
    correctly.
    """

    df: pd.DataFrame | None = None
    model: Any = None
    classifier: Any = None
    premium_threshold: int | None = None
    occupations: list[str] = field(default_factory=list)
    region_codes: dict[str, int] = field(default_factory=dict)
    occ_means: dict[str, float] = field(default_factory=dict)
    state_means: dict[str, float] = field(default_factory=dict)
    drift_monitor: DriftMonitor | None = None
    benchmark_lookup: dict[tuple[str, str], GroupStats] = field(default_factory=dict)
    bls_defaults_lookup: dict[tuple[str, str], BlsDefaults] = field(default_factory=dict)
    quantile_coverage_80: float = 0.0
    model_version: str = "unknown"


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

    # ── Premium-tier classifier head (Gap 1 Phase 1) ────────────────────────
    # Optional on purpose: pre-Phase-1 artefacts (any model trained before
    # the classifier was added) do not ship a classifier, and the API must
    # keep running against them. Missing artefact → ``p_above_premium_threshold``
    # becomes ``None`` on every response, the rest of the pipeline is
    # unaffected. Any *other* exception is a real fault and should crash
    # the probe — do not silently swallow it.
    classifier_cfg_path = VALIDATED_CFG.model.classifier_path
    premium_threshold_cfg = VALIDATED_CFG.model.premium_threshold
    if classifier_cfg_path and premium_threshold_cfg is not None:
        try:
            state.classifier = load_classifier(str(ROOT / classifier_cfg_path))
            state.premium_threshold = int(premium_threshold_cfg)
            logger.info(
                "Premium-tier classifier loaded (threshold=$%d)",
                state.premium_threshold,
            )
        except FileNotFoundError:
            logger.warning(
                "No classifier artefact at %s — premium-tier probability will be None",
                classifier_cfg_path,
            )
    else:
        logger.info("Classifier not configured — premium-tier probability disabled")

    # Precompute (state, education) benchmark lookup so /predict becomes
    # an O(log n) dict get + binary search instead of a per-request
    # full-DataFrame mask.
    state.benchmark_lookup = build_benchmark_lookup(df_eng)
    logger.info("Benchmark lookup built with %d (state, education) cells", len(state.benchmark_lookup))

    # Precompute (state, occupation) BLS context defaults so encode_features
    # becomes an O(1) dict lookup. Eliminates the last per-request
    # DataFrame mask on the hot path.
    state.bls_defaults_lookup = build_bls_defaults_lookup(df_eng)
    logger.info("BLS defaults lookup built with %d (state, occupation) cells", len(state.bls_defaults_lookup))

    # Load model metrics — only the quantile coverage is surfaced at startup
    # for a quick operator sanity check. The /predict route no longer
    # reads residual-based PI offsets; intervals come from the model's
    # quantile output directly.
    metrics = load_metrics(str(ROOT / VALIDATED_CFG.model.metrics_path))
    state.quantile_coverage_80 = float(metrics.get("quantile_coverage_80", 0.0))
    # Model provenance string (``{service_version}+{git_sha}.{data_sha}``)
    # emitted by scripts/train_quantile.py. Falls back to "unknown" on
    # pre-provenance artefacts so the API is backwards-compatible.
    state.model_version = str(metrics.get("model_version", "unknown"))

    # Classifier ↔ config threshold consistency check. The trainer writes
    # the exact ``classifier_threshold`` it was fitted against into
    # ``model_metrics.json`` (see scripts/train_quantile.py). If an
    # operator edits ``config.yaml::model.premium_threshold`` without
    # re-training, the loaded classifier is still calibrated against
    # the old label distribution, and every ``/predict`` response
    # would advertise a ``premium_threshold`` that does not match the
    # boundary the classifier actually learned. That is a silent
    # correctness bug, so we crash the liveness probe on mismatch —
    # the operator sees the failure immediately and either rolls back
    # the config edit or retrains.
    if state.classifier is not None:
        trained_threshold = metrics.get("classifier_threshold")
        if trained_threshold is not None and int(trained_threshold) != state.premium_threshold:
            raise RuntimeError(
                "Classifier threshold mismatch: the loaded classifier was "
                f"trained at premium_threshold=${int(trained_threshold):,} "
                f"(per models/model_metrics.json) but config.yaml declares "
                f"premium_threshold=${state.premium_threshold:,}. Re-train "
                "the classifier (`python -m scripts.train_quantile`) so the "
                "advertised threshold matches the model's decision boundary."
            )

    # Load drift baseline (optional — produced by the training script)
    baseline_path = ROOT / "models" / "baseline_stats.json"
    if baseline_path.exists():
        state.drift_monitor = DriftMonitor.from_baseline(str(baseline_path))
        logger.info("Drift monitor loaded from %s", baseline_path)
    else:
        logger.warning("No baseline_stats.json found — drift monitoring disabled")

    logger.info(
        "Ready — dataset rows: %d, occupations: %d, model features: %d, quantile 80%% coverage: %.3f, model_version: %s",
        len(df_eng),
        len(state.occupations),
        state.model.n_features_in_,
        state.quantile_coverage_80,
        state.model_version,
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
    version=__version__,
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter

# ── Request body size limit ──────────────────────────────────────────────────
# Reject requests with a body larger than MAX_BODY_BYTES. Batch endpoint
# payloads are bounded by PredictBatchRequest.items max_length, but this
# middleware is belt-and-braces against very large payloads that would
# otherwise consume memory before Pydantic validation runs.
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(512 * 1024)))  # 512 KiB default


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"},
                    )
            except ValueError:
                pass
        return await call_next(request)


app.add_middleware(_BodySizeLimitMiddleware)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Scrub unhandled exceptions: log the stack trace server-side, return a
    generic 500 body with the request ID so operators can correlate without
    leaking internal details to the caller.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled exception", extra={"request_id": request_id, "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_exception_handler(Exception, _global_exception_handler)

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
        "version": __version__,
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
        model_version=state.model_version,
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
        edu_order=EDU_ORDER,
        region_map=REGION_MAP,
        region_codes=state.region_codes,
        occ_means=state.occ_means,
        state_means=state.state_means,
        bls_defaults_lookup=state.bls_defaults_lookup,
    )

    if state.drift_monitor is not None:
        state.drift_monitor.observe(row.iloc[0].to_dict())

    p10, p50, p90 = run_model(state.model, row)
    group_stats = lookup_benchmarks(state.benchmark_lookup, req.state, req.education_level)

    # Premium-tier classifier probability (Gap 1 Phase 1). ``None`` when
    # no classifier is loaded so older deployments keep returning the
    # same payload shape.
    p_premium: float | None = None
    if state.classifier is not None:
        p_premium = float(state.classifier.predict_proba(row)[0, 1])

    response = build_response(
        req,
        p10=p10,
        p50=p50,
        p90=p90,
        group_stats=group_stats,
        p_above_premium_threshold=p_premium,
        premium_threshold=state.premium_threshold,
    )

    # Persist to cache for subsequent identical requests (no-op if disabled).
    cache.set(cache_key, response.model_dump())
    return response


@app.post("/predict/batch", response_model=PredictBatchResponse, tags=["Prediction"])
@limiter.limit("10/minute")
def predict_batch(
    request: Request,
    req: PredictBatchRequest,
    _key: str | None = Depends(verify_api_key),
):
    """Score a batch of profiles in a single request.

    Bulk callers (e.g. a consumer scoring a CSV of candidates) should use
    this endpoint instead of calling ``/predict`` in a loop: validation
    runs once, the cache is consulted per-item, and XGBoost scores the
    un-cached rows in a single ``model.predict`` call so per-request
    overhead is amortised across the batch.

    Items that fail domain validation raise 422 for the whole batch.
    """
    # 1. Validate every item up-front so a bad item at position N doesn't
    #    waste inference work on items 0..N-1.
    for idx, item in enumerate(req.items):
        try:
            _validate_domain(item)
        except HTTPException as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Item {idx}: {exc.detail}",
            ) from exc

    responses: list[PredictResponse | None] = [None] * len(req.items)
    rows_to_score: list[tuple[int, PredictRequest]] = []

    # 2. Cache pass — return hits without touching the model.
    for idx, item in enumerate(req.items):
        cache_key = item.model_dump()
        cached = cache.get(cache_key)
        if cached is not None:
            responses[idx] = PredictResponse(**cached)
        else:
            rows_to_score.append((idx, item))

    # 3. Single vectorised model call for the un-cached items.
    if rows_to_score:
        encoded = [
            encode_features(
                item,
                edu_order=EDU_ORDER,
                region_map=REGION_MAP,
                region_codes=state.region_codes,
                occ_means=state.occ_means,
                state_means=state.state_means,
                bls_defaults_lookup=state.bls_defaults_lookup,
            )
            for _, item in rows_to_score
        ]
        batch_df = pd.concat(encoded, ignore_index=True)

        if state.drift_monitor is not None:
            for _, row in batch_df.iterrows():
                state.drift_monitor.observe(row.to_dict())

        raw = np.asarray(state.model.predict(batch_df))
        if raw.ndim != 2 or raw.shape[1] != 3:
            # Legacy point model fallback — degenerate (p, p, p) trio per row.
            raw = np.column_stack([raw, raw, raw])
        preds_dollar = np.expm1(raw)

        # Batched classifier call — one predict_proba for the whole batch
        # keeps overhead amortised. ``None`` when the classifier isn't
        # loaded, same graceful-degradation contract as /predict.
        if state.classifier is not None:
            clf_proba = state.classifier.predict_proba(batch_df)[:, 1]
        else:
            clf_proba = None

        for local_idx, (global_idx, item) in enumerate(rows_to_score):
            p10, p50, p90 = (float(x) for x in preds_dollar[local_idx])
            group_stats = lookup_benchmarks(state.benchmark_lookup, item.state, item.education_level)
            p_premium = float(clf_proba[local_idx]) if clf_proba is not None else None
            resp = build_response(
                item,
                p10=p10,
                p50=p50,
                p90=p90,
                group_stats=group_stats,
                p_above_premium_threshold=p_premium,
                premium_threshold=state.premium_threshold,
            )
            cache.set(item.model_dump(), resp.model_dump())
            responses[global_idx] = resp

    return PredictBatchResponse(items=[r for r in responses if r is not None])


@app.get("/drift", tags=["Monitoring"])
async def drift_report():
    """Return feature drift report comparing recent predictions to training baseline.

    Requires ``models/baseline_stats.json`` (generated by
    ``scripts/train_quantile.py``). With ``REDIS_URL`` set, the
    observation window is shared across all replicas — the report is
    cluster-wide. Without Redis, the report is per-pod.
    """
    if state.drift_monitor is None:
        return {
            "status": "disabled",
            "message": "No baseline_stats.json — run 'python -m scripts.train_quantile' to generate it",
        }
    return state.drift_monitor.check_drift()
