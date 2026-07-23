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
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import Counter
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
    build_feature_frame,
    build_response,
    encode_feature_values,
    lookup_benchmarks,
    quantiles_crossed,
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
    compute_fallback_means,
    engineer_features,
    is_quantile_model,
    load_classifier,
    load_conformal_delta,
    load_group_means,
    load_metrics,
    load_model,
    predict_quantiles_batch,
    sha256_file,
)

# ── Structured JSON Logging ──────────────────────────────────────────────────


class _JSONFormatter(logging.Formatter):
    """Emit logs as single-line JSON for machine parsing and log aggregation.

    Any non-standard attributes attached via ``logger.info(..., extra={...})``
    (request_id, method, path, status, duration_ms, …) are merged into the
    JSON object. Without this, every structured field is silently dropped and
    access logs collapse to ``{"message": "request completed"}`` — making the
    request-correlation story the README advertises impossible.
    """

    #: Attributes the stdlib sets on every LogRecord; everything else on the
    #: record was supplied by the caller via ``extra=`` and must be emitted.
    _RESERVED = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge caller-supplied structured fields (extra={...}).
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ── API Key Auth ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Per-IP throttle for FAILED auth attempts. The main rate limiter runs inside the
# route function, after dependency resolution, so a 401 raised by verify_api_key
# never reaches it — leaving X-API-Key guessing unthrottled. This caps failures
# per IP independently (in-process per replica; defence-in-depth, not a substitute
# for a high-entropy key).
AUTH_FAILURE_LIMIT = int(os.getenv("AUTH_FAILURE_LIMIT", "10"))
AUTH_FAILURE_WINDOW_S = float(os.getenv("AUTH_FAILURE_WINDOW_S", "60"))


class _AuthFailureThrottle:
    """Sliding-window count of failed auth attempts per client IP."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _sweep_stale(self, cutoff: float) -> None:
        """Drop IPs whose most-recent failure has aged out. Called at most once
        per window so the map can't grow without bound as attackers rotate source
        IPs — a per-IP deque is never emptied in place (the just-appended hit keeps
        it non-empty), so eviction has to happen here, across keys."""
        stale = [ip for ip, dq in self._hits.items() if not dq or dq[-1] < cutoff]
        for ip in stale:
            del self._hits[ip]

    def record_failure(self, ip: str, now: float) -> bool:
        """Record one failure for ``ip`` at time ``now``; return True if the IP is
        still within budget, False once it has exceeded ``limit`` in the window."""
        with self._lock:
            cutoff = now - self._window
            if now - self._last_sweep >= self._window:
                self._sweep_stale(cutoff)
                self._last_sweep = now
            dq = self._hits[ip]
            while dq and dq[0] < cutoff:
                dq.popleft()
            dq.append(now)
            return len(dq) <= self._limit


_auth_throttle = _AuthFailureThrottle(AUTH_FAILURE_LIMIT, AUTH_FAILURE_WINDOW_S)


async def verify_api_key(request: Request, key: str | None = Security(_api_key_header)) -> str | None:
    """Validate API key if API_KEY is configured; skip in dev mode (unset)."""
    if not API_KEY:
        return None  # dev mode: no auth required
    # Constant-time comparison so a timing side-channel can't be used to
    # recover the key byte by byte.
    if key is None or not secrets.compare_digest(key, API_KEY):
        # Throttle brute-force key guessing per IP — the route-level limiter
        # never sees this request because the 401 short-circuits before it.
        within_budget = _auth_throttle.record_failure(_client_ip(request), time.monotonic())
        if not within_budget:
            raise HTTPException(status_code=429, detail="Too many failed authentication attempts")
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
    """Return the client IP, respecting TRUSTED_PROXY_HOPS for X-Forwarded-For.

    Security-critical. Each of our own ``TRUSTED_PROXY_HOPS`` reverse proxies
    appends exactly one entry to the *right* end of ``X-Forwarded-For``, so the
    right-most ``TRUSTED_PROXY_HOPS`` entries are the trustworthy ones and the
    real client is the left-most of those — index ``-TRUSTED_PROXY_HOPS``
    (werkzeug ``ProxyFix`` semantics). Every entry further left is
    client-supplied and therefore spoofable: reading it would let an attacker
    forge ``X-Forwarded-For`` to mint a fresh rate-limit bucket on every
    request. The previous ``len - 1 - hops`` index was off by one and returned
    exactly that attacker-controlled entry. If the header carries fewer entries
    than we have trusted proxies, it cannot have come through our proxy chain,
    so we ignore it and bind to the direct peer.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff and TRUSTED_PROXY_HOPS > 0:
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        if len(hops) >= TRUSTED_PROXY_HOPS:
            return hops[-TRUSTED_PROXY_HOPS]
        return request.client.host if request.client else "unknown"
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
# Sorted lists above drive the /meta response; these frozensets back the
# per-request domain checks so membership is O(1), not a list scan.
VALID_EDUCATION_SET = frozenset(VALID_EDUCATION)
VALID_STATES_SET = frozenset(VALID_STATES)

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
    occupation_set: frozenset[str] = field(default_factory=frozenset)
    region_codes: dict[str, int] = field(default_factory=dict)
    occ_means: dict[str, float] = field(default_factory=dict)
    state_means: dict[str, float] = field(default_factory=dict)
    # Occupation/state fallback means, precomputed once at startup so the
    # hot path never reduces over the full group-mean dicts per request.
    occ_fallback: float = 0.0
    state_fallback: float = 0.0
    drift_monitor: DriftMonitor | None = None
    benchmark_lookup: dict[tuple[str, str], GroupStats] = field(default_factory=dict)
    bls_defaults_lookup: dict[tuple[str, str], BlsDefaults] = field(default_factory=dict)
    quantile_coverage_80: float = 0.0
    model_version: str = "unknown"
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    # Split-conformal interval margin (log space). 0.0 ⇒ raw interval.
    conformal_delta: float = 0.0


state = AppState()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


def _artifact_mismatches(artefact_files: dict[str, Path], recorded: dict[str, str]) -> list[str]:
    """Return a mismatch string for every artefact whose bytes differ from the
    recorded SHA-256. Empty list ⇒ every present, recorded artefact verified."""
    problems = []
    for key, path in artefact_files.items():
        want = recorded.get(key)
        if want and path.exists():
            got = sha256_file(path)
            if got != want:
                problems.append(f"{key} ({path.name}): loaded {got[:12]} != recorded {want[:12]}")
    return problems


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    logger.info("Starting up: loading dataset, model, group means, and metrics…")

    # Load training-set group means for consistent target encoding at inference
    group_means = load_group_means(str(ROOT / VALIDATED_CFG.model.group_means_path))
    state.occ_means = group_means["occ_means"]
    state.state_means = group_means["state_means"]
    # Reduce over the group-mean dicts exactly once here, not per request.
    state.occ_fallback, state.state_fallback = compute_fallback_means(group_means)

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
    # Refuse a non-quantile (legacy point) model rather than silently serving a
    # degenerate (p, p, p) interval as if it were an 80% band. The whole product
    # is calibrated uncertainty; a point model here is a deploy error, not a
    # graceful fallback.
    if not is_quantile_model(state.model):
        raise RuntimeError(
            "Loaded model is not a multi-quantile model (objective != 'reg:quantileerror'). "
            "The API serves P10/P50/P90 intervals; refusing to start with a point-estimate "
            "model that would collapse every interval to a single value."
        )
    state.occupations = sorted(df_eng["Occupation"].unique().tolist())
    state.occupation_set = frozenset(state.occupations)
    state.region_codes = REGION_CODES

    # Split-conformal interval margin. Configured ⇒ required: the served
    # interval claims a calibrated 80% coverage that only holds with this margin
    # applied, so a missing file is a hard startup failure (load_conformal_delta
    # raises), not a silent fall-back to the under-covering raw interval. A
    # config without the key (legacy) leaves the margin at 0.0.
    if VALIDATED_CFG.model.conformal_path:
        state.conformal_delta = load_conformal_delta(str(ROOT / VALIDATED_CFG.model.conformal_path))
        logger.info("Conformal interval margin loaded (delta=%.4f, log space)", state.conformal_delta)

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
    # Report the coverage of the interval the API actually serves. With a
    # conformal margin applied that is the conformalized coverage, not the raw
    # quantile coverage — surfacing the raw number would understate the served
    # interval. Falls back to the raw coverage when no margin is calibrated.
    state.quantile_coverage_80 = float(metrics.get("conformal_coverage_80") or metrics.get("quantile_coverage_80", 0.0))
    # Model provenance string (``{service_version}+{git_sha}.{data_sha}``)
    # emitted by scripts/train_quantile.py. Falls back to "unknown" on
    # pre-provenance artefacts so the API is backwards-compatible.
    state.model_version = str(metrics.get("model_version", "unknown"))
    # Namespace the prediction cache by model version so a retrain never
    # serves a previous model's cached predictions from a shared Redis.
    cache.version = state.model_version

    # ── Artifact integrity: served bytes must match what training recorded ───
    # /health reports model_version from the metrics sidecar, but nothing tied
    # that string to the actual .ubj/.json files on disk — a swapped or
    # partially-copied artefact would serve under a /health that still claims
    # the trained version. Re-hash each loaded artefact against the recorded
    # SHA-256 and crash the probe on mismatch, same fail-loud posture as the
    # threshold check below.
    state.artifact_sha256 = dict(metrics.get("artifact_sha256") or {})
    if state.artifact_sha256:
        artefact_files = {
            "model": ROOT / VALIDATED_CFG.model.model_path,
            "features": ROOT / VALIDATED_CFG.model.features_path,
            "group_means": ROOT / VALIDATED_CFG.model.group_means_path,
            "baseline_stats": ROOT / "models" / "baseline_stats.json",
        }
        if state.classifier is not None and VALIDATED_CFG.model.classifier_path:
            artefact_files["classifier"] = ROOT / VALIDATED_CFG.model.classifier_path
        if VALIDATED_CFG.model.conformal_path:
            artefact_files["conformal"] = ROOT / VALIDATED_CFG.model.conformal_path
        mismatches = _artifact_mismatches(artefact_files, state.artifact_sha256)
        if mismatches:
            raise RuntimeError(
                "Artifact integrity check failed at startup — served files do not match "
                f"models/model_metrics.json: {'; '.join(mismatches)}. Refusing to serve a "
                "mislabeled model; re-deploy the audited artefacts or retrain."
            )

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
    def _too_large(self) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"},
        )

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                declared = int(cl)
            except ValueError:
                declared = None
            if declared is not None:
                if declared > MAX_BODY_BYTES:
                    return self._too_large()
                # A valid Content-Length within the cap is enforced by the ASGI
                # server (it reads exactly that many bytes), so trust it and skip
                # re-measuring.
                return await call_next(request)
        # No Content-Length (e.g. Transfer-Encoding: chunked) or a malformed one
        # bypasses the header check entirely — the original bug. Measure the real
        # body with a running counter and abort the moment it exceeds the cap, so
        # an unbounded chunked upload can't consume memory before validation.
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return self._too_large()
            chunks.append(chunk)
        # Cache the (<= cap) body so downstream handlers/Pydantic read it normally
        # instead of trying to re-consume the already-drained stream.
        request._body = b"".join(chunks)
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
    # Prefer the id the middleware minted (stored on request.state); fall back
    # to an inbound header, then "unknown". Reading state first means errors
    # are correlatable even when the caller sent no X-Request-ID.
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled exception", extra={"request_id": request_id, "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# Starlette types the handler's exc param as the base Exception; our handler
# narrows it to RateLimitExceeded (and reads .detail), which is correct at
# runtime but trips the invariant arg-type check. Scoped ignore, not blanket.
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
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
    # Store on request.state so downstream handlers (notably the 500 handler)
    # can correlate even when the client did not send an X-Request-ID.
    request.state.request_id = request_id
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

# Counts predictions whose raw quantiles crossed (p10>p50 or p50>p90) and were
# clamped by build_response. A rising rate is a model-health signal, so it is
# surfaced on /metrics instead of being silently corrected.
QUANTILE_CROSSINGS = Counter(
    "salary_quantile_crossings_total",
    "Predictions where the model's raw quantiles crossed before clamping.",
)


# ── Validation helper ────────────────────────────────────────────────────────


def _validate_domain(req: PredictRequest) -> None:
    """Domain validation against loaded data. Raises 422 on unknown values."""
    if req.state not in VALID_STATES_SET:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown state '{req.state}'. Use /meta to see valid values.",
        )
    if req.occupation not in state.occupation_set:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown occupation '{req.occupation}'. Use /meta to see valid values.",
        )
    if req.education_level not in VALID_EDUCATION_SET:
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
        artifact_sha256=state.artifact_sha256,
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

    # ── Feature encoding + drift observation (BEFORE the cache) ──────────────
    # Encode and observe drift for every request, including cache hits — the
    # drift monitor must see all production traffic. If observation happened
    # after the cache short-circuit, repeated (cached) queries would be
    # invisible to drift detection. Only the expensive model inference below
    # is cached.
    values = encode_feature_values(
        req,
        edu_order=EDU_ORDER,
        region_map=REGION_MAP,
        region_codes=state.region_codes,
        occ_means=state.occ_means,
        state_means=state.state_means,
        occ_fallback=state.occ_fallback,
        state_fallback=state.state_fallback,
        bls_defaults_lookup=state.bls_defaults_lookup,
    )
    row = build_feature_frame([values])

    if state.drift_monitor is not None:
        state.drift_monitor.observe(values)

    # ── Cache lookup (keyed on validated request payload) ────────────────────
    cache_key = req.model_dump()
    cached = cache.get(cache_key)
    if cached is not None:
        return PredictResponse(**cached)

    p10, p50, p90 = run_model(state.model, row, conformal_delta=state.conformal_delta)
    if quantiles_crossed(p10, p50, p90):
        QUANTILE_CROSSINGS.inc()
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

    # 2. Encode every item once and observe drift for ALL of them (cache hits
    #    included) so the drift monitor sees all batch traffic — consistent
    #    with /predict, where only inference is cached, never observation.
    encoded_all = [
        encode_feature_values(
            item,
            edu_order=EDU_ORDER,
            region_map=REGION_MAP,
            region_codes=state.region_codes,
            occ_means=state.occ_means,
            state_means=state.state_means,
            occ_fallback=state.occ_fallback,
            state_fallback=state.state_fallback,
            bls_defaults_lookup=state.bls_defaults_lookup,
        )
        for item in req.items
    ]
    if state.drift_monitor is not None:
        for features in encoded_all:
            state.drift_monitor.observe(features)

    responses: list[PredictResponse | None] = [None] * len(req.items)
    rows_to_score: list[tuple[int, PredictRequest]] = []

    # 3. Cache pass — return hits without touching the model.
    for idx, item in enumerate(req.items):
        cached = cache.get(item.model_dump())
        if cached is not None:
            responses[idx] = PredictResponse(**cached)
        else:
            rows_to_score.append((idx, item))

    # 4. Single vectorised model call for the un-cached items.
    if rows_to_score:
        batch_df = build_feature_frame([encoded_all[idx] for idx, _ in rows_to_score])
        preds_dollar = predict_quantiles_batch(state.model, batch_df, conformal_delta=state.conformal_delta)

        # Batched classifier call — one predict_proba for the whole batch
        # keeps overhead amortised. ``None`` when the classifier isn't
        # loaded, same graceful-degradation contract as /predict.
        if state.classifier is not None:
            clf_proba = state.classifier.predict_proba(batch_df)[:, 1]
        else:
            clf_proba = None

        for local_idx, (global_idx, item) in enumerate(rows_to_score):
            p10, p50, p90 = (float(x) for x in preds_dollar[local_idx])
            if quantiles_crossed(p10, p50, p90):
                QUANTILE_CROSSINGS.inc()
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
