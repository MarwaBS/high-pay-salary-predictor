# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some wheels (scipy, lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install only the packages needed to run the Streamlit dashboard
# (excludes Jupyter, flake8, pytest which are dev-only)
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
        matplotlib


# ── Stage 2: lean runtime image ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what the app needs at runtime
COPY config.yaml          ./config.yaml
COPY streamlit_app.py     ./streamlit_app.py
COPY Data/                ./Data/

# Pre-create writable directories for model artefacts and generated images
RUN mkdir -p models Images

EXPOSE 8501

# Streamlit health endpoint used by Docker and cloud platforms
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
