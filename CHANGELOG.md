# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses SemVer.

## [Unreleased]

### Changed
- **`age` now accepts 19–94 instead of 18–80** (breaking both ways: `18` is
  refused, `81`–`94` are accepted). The bounds are the model's training
  support, recorded as `Age` min/max in `models/baseline_stats.json`. The old
  range both extrapolated below the support and declined answerable requests
  inside it. The dashboard derives its range from that artefact through
  `pipeline.training_age_support`; the API's bounds stay declared on the schema,
  where `PredictRequest` needs them at class-definition time, and a test holds
  them to the artefact so a retrain that shifts the support fails CI.
- **`config.yaml` gains a required `drift.window`** (breaking for any config
  file carried over: `ProjectConfig` rejects one without it). The drift
  monitor's rolling-window size was a default buried in `DriftMonitor`, where
  nothing could state why it held that value. It is an operating choice, so it
  now lives beside the bound it has to clear, and `DriftMonitor` requires the
  caller to name one. A window below the detector's effect-floor handover aborts
  API startup, since the bound depends on the detector's tuning and only both
  together determine it; there is no derivable upper bound, and the trade-off a
  larger window buys is written beside the key.

### Fixed
- **The k8s dashboard pod now stages `baseline_stats.json`.** Its initContainer
  fetched four artefacts into an `emptyDir`; the dashboard reads a fifth to
  bound its Age input, so the predictor tab would have raised on first render.
  Both pods now stage every artefact `config.yaml` declares, rather than a list
  per pod, and a test holds each manifest to that set — previously only the API
  pod's list was checked at all.
- **Prediction cache keys carry the full SHA-256 digest** rather than its first
  16 hex chars. The truncation bought nothing and left a 64-bit key space in
  which a collision serves one caller another's prediction. Cached entries
  written before the change are unreachable and expire on their existing TTL.
- **`GET /drift` returns `any_drifted: null` in two further cases** — fewer than
  30 observations, and no observed feature matching the baseline (a renamed
  feature is the realistic trigger). Both previously returned `false`, which
  reads as "no drift" when in fact nothing was testable. Callers treating the
  field as a boolean should treat `null` as "verdict withheld"; `message` says
  which case applies.
- **`GET /drift` now requires `X-API-Key` when `API_KEY` is set, and spends
  the same per-IP `RATE_LIMIT` budget as `/predict`** (breaking for any caller
  that read it anonymously). It aggregates live request traffic down to
  per-feature means, so it was the one route publishing that summary without
  a key.

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
- `requirements-dashboard.txt`: exact pins for the dashboard Docker image
  (previously hand-listed `>=` floors inside the Dockerfile, never
  audited); now consumed by the `dashboard-builder` stage and covered by
  the CI `pip-audit` gate.
- CI `schedule:` trigger (weekly, Mondays 05:00 UTC) re-running the full
  pipeline on `main` — including the Docker builds + Trivy scans — so
  newly published image CVEs are caught by time, not only by pushes.
- Annotated tag `training/2.0.0` pinning training commit `1c5e9d896ee5`,
  the code/data state the 2.0.0 model release was trained at, so
  provenance survives feature-branch cleanup. (The shipped
  `model_metrics.json` has since been regenerated — see Fixed below —
  so its `model_version` records the regeneration commit instead.)
- Drift monitor ramp-up regression tests: familywise false-alarm bounds
  on stationary windows at n=30/n=100 (150 i.i.d. trials each) plus a
  mid-window (n=150) deaf-check on a genuine 0.5 σ shift.

### Fixed
- Reported cross-validation R² was inflated by a target encoding
  computed over the full training set before folding; each CV fold now
  recomputes its own encoding means, so a validation row is never
  encoded with a mean that saw its own target. `cv_r2_mean` drops from
  0.029 to the honest 0.0224. Model artefact bytes are unchanged;
  `model_metrics.json` was regenerated, so its `model_version` now
  records commit `4aa98be6699e`.
- Drift monitor no longer chronically false-alarms while its window
  fills: per-feature significance is now Šidák-corrected across the ~10
  monitored features and the 0.2 σ effect floor ramp-scales as
  `max(0.2, 2·√(2/n))`. Uncorrected, the union of ~10 per-feature z>2
  tests false-alarms with probability 1 − (1 − 0.0455)^10 ≈ 37 % while
  n < (z/d)² = 100. The corrected monitor is gated at ≤7 % over 150
  stationary windows at both n=30 and n=100, against the ≈4.6 % design
  level — a bound, not a published rate. Detection of an Age +5 yr
  (0.5 σ) shift is unchanged: 25 of 25 trials at n=150, and full-window
  (n=500) behaviour is unchanged.

### Changed
- Python target bumped: `requires-python >= 3.11` (was `>= 3.10`).
  CI matrix now `3.11 + 3.12` (was `3.10 + 3.11`). `ruff.target-version`
  and `mypy.python_version` bumped to `py312` / `"3.12"`.
- Dockerfile base image pinned to `python:3.12-slim-bookworm` across
  all three stages (was `python:3.11-slim`).
- Dockerfile runtime stages now run `apt-get upgrade -y` before
  installing runtime deps, to refresh OS security patches even when the
  GHA layer cache reuses a stale base-image layer.

## [2.0.0] — 2026-04-11

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
- Hugging Face Spaces deployment package (`2db1825`) + live demo badge
  in README.
- `config_schema.py` with Pydantic validation of `config.yaml`.

### Fixed
- Streamlit predictor tab now routes through the FastAPI `/predict`
  endpoint instead of calling the model directly, so cache hits + rate
  limiting + drift monitoring flow through a single path.

## [1.0.0] — prior

Initial end-to-end pipeline: BLS OEWS + Census data integration → XGBoost
point estimator → FastAPI + Streamlit + Docker.
