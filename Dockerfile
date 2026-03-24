# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Builds all wheels into /install so runtime stages stay lean.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install only runtime packages (excludes jupyter / pytest / dev tools).
# Pin to requirements.txt so Docker is consistent with the local environment.
RUN pip install --no-cache-dir --prefix=/install \
        pandas \
        numpy \
        scikit-learn \
        xgboost \
        lightgbm \
        shap \
        plotly \
        streamlit \
        pyyaml \
        scipy \
        statsmodels \
        matplotlib \
        fastapi \
        "uvicorn[standard]" \
        pydantic


# ── Stage 2: Streamlit dashboard ──────────────────────────────────────────────
FROM python:3.11-slim AS dashboard

WORKDIR /app

# Install curl for HEALTHCHECK — not present in slim by default
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

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
FROM python:3.11-slim AS api

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY config.yaml ./config.yaml
COPY pipeline.py ./pipeline.py
COPY api/        ./api/
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
