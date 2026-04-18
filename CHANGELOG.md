# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses SemVer.

## [Unreleased]

### Added
- `SECURITY.md` with disclosure policy + in-scope / out-of-scope
  boundaries.
- `CHANGELOG.md` (this file).
- `.trivyignore` at repo root (starter, empty ignore list).
- CI lint job: `bandit` static-analysis step (hardcoded secrets, unsafe
  yaml.load, weak crypto detection).
- CI test job: CycloneDX SBOM emission via `pip-audit
  --format=cyclonedx-json` on `requirements-api.txt`; 90-day artifact.
- CI deploy job: Trivy HIGH/CRITICAL scan on both API and Dashboard
  Docker images BEFORE push to GHCR. Vulnerable images are blocked
  pre-push.
- Dependabot `docker` ecosystem (was `pip` + `github-actions` only).

### Changed
- Python target bumped: `requires-python >= 3.11` (was `>= 3.10`).
  CI matrix now `3.11 + 3.12` (was `3.10 + 3.11`). `ruff.target-version`
  and `mypy.python_version` bumped to `py312` / `"3.12"`.
- Dockerfile base image pinned to `python:3.12-slim-bookworm` across
  all three stages (was `python:3.11-slim`).
- Dockerfile runtime stages now run `apt-get upgrade -y` before
  installing runtime deps, to refresh OS security patches even when the
  GHA layer cache reuses a stale base-image layer.

## [2.0.0] — 2026-04-xx

### Changed
- **Quantile reframe (breaking semantic upgrade).** The API now returns
  P10/P50/P90 from a multi-quantile XGBoost model instead of a single
  point estimate. `predicted_salary` is retained as an alias for P50 so
  v1 clients keep working, but the framing change is load-bearing:
  point-estimate R² is a weak fit-statistic under a quantile loss on a
  truncated `INCTOT ≥ $100K` cohort.

### Added
- Premium-tier classifier head surfaced on `/predict`.
- Model registry with provenance string; `/health` exposes the current
  model fingerprint.
- Scheduled training workflow (`.github/workflows/train.yml`).
- Hugging Face Spaces deployment package (`a74211d`) + live demo badge
  in README.
- `config_schema.py` with Pydantic validation of `config.yaml`.

### Fixed
- Streamlit predictor tab now routes through the FastAPI `/predict`
  endpoint instead of calling the model directly, so cache hits + rate
  limiting + drift monitoring flow through a single path.

## [1.0.0] — prior

Initial end-to-end pipeline: BLS OEWS + Census data integration → XGBoost
point estimator → FastAPI + Streamlit + Docker.
