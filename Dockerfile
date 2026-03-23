# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed by scipy / lightgbm wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install only runtime packages (no Jupyter / pytest / flake8)
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

COPY --from=builder /install /usr/local

COPY config.yaml      ./config.yaml
COPY streamlit_app.py ./streamlit_app.py
COPY Data/            ./Data/

RUN mkdir -p models Images

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

COPY --from=builder /install /usr/local

COPY config.yaml ./config.yaml
COPY Data/       ./Data/
COPY api/        ./api/

RUN mkdir -p models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
