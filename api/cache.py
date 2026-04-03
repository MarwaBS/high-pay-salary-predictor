"""
Optional Redis caching for deterministic salary predictions.

Salary predictions are pure functions of their inputs — same features
always produce the same output. A cache keyed on the input hash avoids
redundant XGBoost inference for repeated queries.

Graceful degradation: if REDIS_URL is unset or Redis is unreachable,
the cache silently becomes a no-op. No runtime errors, no code changes.

Usage:
    export REDIS_URL=redis://localhost:6379/0   # enable caching
    # unset REDIS_URL                           # disable (default)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default


def _feature_hash(payload: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of prediction inputs (first 16 chars)."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class PredictionCache:
    """Redis-backed cache for deterministic salary predictions.

    Falls back to no-op if Redis is unavailable — zero impact on API
    correctness, only on latency under repeated queries.
    """

    def __init__(self) -> None:
        self._client = None
        if REDIS_URL:
            try:
                import redis

                self._client = redis.from_url(REDIS_URL, decode_responses=True)
                self._client.ping()
                logger.info("Redis prediction cache enabled at %s", REDIS_URL)
            except Exception:
                logger.warning("Redis at %s unreachable — caching disabled", REDIS_URL)
                self._client = None

    @property
    def enabled(self) -> bool:
        """True if Redis is connected and responding."""
        return self._client is not None

    def get(self, payload: dict[str, Any]) -> dict | None:
        """Look up cached prediction. Returns None on miss or if disabled."""
        if not self._client:
            return None
        try:
            key = f"predict:{_feature_hash(payload)}"
            cached = self._client.get(key)
            return json.loads(cached) if cached else None
        except Exception:
            return None

    def set(self, payload: dict[str, Any], result: dict[str, Any], ttl: int | None = None) -> None:
        """Cache a prediction result. No-op if disabled or on error."""
        if not self._client:
            return
        try:
            key = f"predict:{_feature_hash(payload)}"
            self._client.setex(key, ttl or CACHE_TTL, json.dumps(result, default=str))
        except Exception:
            pass  # cache write failure is non-critical
