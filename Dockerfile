# ── Stage 1a: API dependency builder (lean, pinned) ─────────────────────────
# Installs from requirements-api.txt with exact == pins for reproducible
# builds. Includes ONLY what api/ needs — do NOT add jupyter / pytest /
# streamlit / shap / lightgbm / statsmodels / geopandas here. Keeping the
# API image lean also makes pip-audit scans faster and reduces the CVE
# surface.
FROM python:3.12-slim-bookworm AS api-builder

WORKDIR /build

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# Exact version pinning (==) so a rebuild in 6 months pulls the same wheels.
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 1b: Dashboard dependency builder ──────────────────────────────────
# Separate builder for Streamlit + viz stack so the api image does not pull
# shap/plotly/matplotlib it never uses.
FROM python:3.12-slim-bookworm AS dashboard-builder

WORKDIR /build

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Dashboard uses loose lower bounds for now (notebook/viz stack).
# TODO: split into requirements-dashboard.txt with pinned versions when
# the notebook stack is stabilised.
RUN pip install --no-cache-dir --prefix=/install \
        "pandas>=1.5.0" \
        "numpy>=1.21.0" \
        "scikit-learn>=1.1.0" \
        "xgboost>=1.7.0" \
        "lightgbm>=3.3.0" \
        "shap>=0.41.0" \
        "plotly>=5.10.0" \
        "streamlit>=1.20.0" \
        "pyyaml>=6.0" \
        "matplotlib>=3.5.0" \
        "httpx>=0.27.0"


# ── Stage 2: Streamlit dashboard ──────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS dashboard

WORKDIR /app

# Install curl for HEALTHCHECK — not present in slim by default
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=dashboard-builder /install /usr/local

COPY config.yaml      ./config.yaml
COPY streamlit_app.py ./streamlit_app.py
COPY pipeline.py      ./pipeline.py
# Data/ is NOT baked into the image — mount ./Data as a read-only volume
# in docker-compose.yml so the image stays lean and dataset changes don't
# require a rebuild.

RUN mkdir -p models Images Data

# Run as non-root for security (required by many enterprise registries)
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]


# ── Stage 3: FastAPI salary predictor ────────────────────────────────────────
FROM python:3.12-slim-bookworm AS api

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=api-builder /install /usr/local

COPY config.yaml       ./config.yaml
COPY config_schema.py  ./config_schema.py
COPY pipeline.py       ./pipeline.py
COPY api/              ./api/
# Data/ is NOT baked into the image — mounted as a read-only volume.

RUN mkdir -p models Data

# Run as non-root
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
