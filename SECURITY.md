# Security Policy

## Supported versions

Only the `main` branch is actively maintained. Security patches are applied
to the latest tagged release only.

## Reporting a vulnerability

**Do not open a public GitHub issue for security bugs.**

Email: `marwabensalem30@gmail.com` with subject prefix `[SECURITY]` and:

- A description of the vulnerability
- Steps to reproduce (PoC if available)
- Affected component (API endpoint, data pipeline, Docker image, etc.)
- Your assessment of severity and potential impact

Expect acknowledgement within 72 hours.

## Scope

**In scope:**
- `POST /predict` endpoint input validation + auth (X-API-Key header)
- `GET /health`, `GET /meta`, `GET /drift`, `GET /metrics` endpoint
  information disclosure
- Prediction cache / drift-monitor state integrity (Redis-backed)
- CORS configuration (`CORS_ORIGINS` env var)
- Reverse-proxy trust boundary (`TRUSTED_PROXY_HOPS`)
- Container-image CVEs scanned by Trivy in CI (see `.trivyignore` for
  managed risks)
- Supply-chain findings from `pip-audit` + CycloneDX SBOM artifacts
- Model-inversion / membership-inference attacks on the trained
  XGBoost quantile regressor

**Out of scope:**
- Issues requiring physical access to a user's machine
- Social engineering / phishing reports
- Denial-of-service against the HuggingFace Spaces live demo (public,
  rate-limited via `slowapi`)
- Bugs in the BLS OEWS / Census datasets themselves

## Handling of known managed risks

### pip-audit ignore: `CVE-2025-69872`

The `security` CI step ignores this CVE by design (documented inline in
`.github/workflows/ci.yml`). Re-evaluate quarterly or when the upstream
patch lands.

### `.trivyignore`

Currently empty. Any entry here must have inline rationale and a
re-evaluation trigger.
