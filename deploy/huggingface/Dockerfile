# ── Hugging Face Spaces single-container image ──────────────────────────────
#
# Runs BOTH the FastAPI service (uvicorn on :8000, localhost-only) and
# the Streamlit dashboard (on :7860, the HF Spaces default port) inside
# a single container. The Streamlit predictor tab calls the local API
# via http://localhost:8000/predict so both services share one process
# group and there's only one URL to share with reviewers.
#
# HF Spaces constraints this image respects:
#   1. Runs as non-root user (uid 1000 = "user", home /home/user)
#   2. Listens on $PORT (default 7860)
#   3. Writes only to /home/user or /tmp
#   4. Avoids network calls at build time beyond `pip install`
#
# This Dockerfile is separate from the project-root Dockerfile so the
# docker-compose workflow stays unchanged.

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HF Spaces convention) ────────────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# ── Python deps ──────────────────────────────────────────────────────────────
# Install the API runtime from the pinned lock first (smallest surface),
# then layer the dashboard + viz extras on top. Order matters for Docker
# layer caching — heavy wheels land before the source copy.
COPY --chown=user:user requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir \
        "streamlit>=1.20.0" \
        "plotly>=5.10.0" \
        "matplotlib>=3.5.0" \
        "seaborn>=0.11.0" \
        "httpx>=0.27.0" \
        "scikit-learn>=1.1.0" \
        "scipy>=1.9.0" \
        "statsmodels>=0.13.0"

# ── Project source ──────────────────────────────────────────────────────────
# Explicit COPY list keeps the image lean and avoids pulling notebooks,
# raw Resources, test assets, or the private/ directory.
COPY --chown=user:user api/            /home/user/app/api/
COPY --chown=user:user scripts/        /home/user/app/scripts/
COPY --chown=user:user pipeline.py     /home/user/app/
COPY --chown=user:user config_schema.py /home/user/app/
COPY --chown=user:user config.yaml     /home/user/app/
COPY --chown=user:user streamlit_app.py /home/user/app/
COPY --chown=user:user pyproject.toml  /home/user/app/

# Model + dataset artefacts baked into the image (no volumes on HF Spaces)
COPY --chown=user:user models/         /home/user/app/models/
COPY --chown=user:user Data/           /home/user/app/Data/

# Dashboard tries to load the us_state shapefile for the choropleth tab.
# If the directory is missing the dashboard falls back gracefully — but
# including it is cheap and keeps the map tab working.
COPY --chown=user:user us_state/       /home/user/app/us_state/

# Startup script (launches uvicorn + streamlit via bash)
COPY --chown=user:user deploy/huggingface/start.sh /home/user/app/start.sh
RUN chmod +x /home/user/app/start.sh

# ── Install project as editable so `api.main:app` resolves cleanly ──────────
RUN pip install --no-cache-dir --user -e .

# ── Runtime user + env ──────────────────────────────────────────────────────
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The dashboard calls the colocated API via localhost — single process
    # namespace inside the container means no network latency.
    API_BASE_URL=http://localhost:8000 \
    # Permissive CORS for the demo — tighten if fronting with a real origin.
    CORS_ORIGINS="*" \
    # HF Spaces fronts the container with its own proxy. Rate limit keys
    # on the direct client since TRUSTED_PROXY_HOPS=0.
    TRUSTED_PROXY_HOPS=0 \
    # No Redis on the free tier — the cache gracefully no-ops.
    REDIS_URL="" \
    # HF Spaces injects $PORT at runtime; default matches the frontmatter
    # app_port so local `docker run` also works.
    PORT=7860

EXPOSE 7860

# Single entrypoint launches both processes and exits when Streamlit exits.
CMD ["bash", "/home/user/app/start.sh"]
