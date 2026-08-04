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

### pip-audit suppressions

None. The `security` CI step runs `pip-audit` clean against the five
requirement files that reach a build: `requirements.txt`,
`requirements-lock.txt`, `requirements-api.txt`, `requirements-dashboard.txt`
and the Space's. `requirements-notebooks.txt` is deliberately outside the gate
— it is never installed in an image and its packages have no importer in the
serving path. There is no ignore-file mechanism to inspect — `pip-audit`
reads none, so a suppression can only be a flag. The CI gate carries no
`--ignore-vuln` flags, so there is no active suppression path — a new CVE fails
the build until it is fixed, or until an explicit, rationale-carrying
`--ignore-vuln` is added to `ci.yml`.

### `.trivyignore`

Currently empty. Any entry here must have inline rationale and a
re-evaluation trigger.
