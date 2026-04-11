"""High-Pay Salary Predictor API package."""

# Single source of truth for the service version string. Imported by
# api.main (FastAPI app version + root endpoint) and api.schemas
# (HealthResponse default) so they cannot drift out of sync.
__version__ = "2.0.0"
