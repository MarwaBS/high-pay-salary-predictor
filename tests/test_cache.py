"""
PredictionCache behaviour — version namespacing and graceful degradation.

The cache key is namespaced by the loaded model version so a retrain never
serves a previous model's cached predictions from a shared Redis. These tests
use a tiny in-memory fake client (no Redis required) to lock that contract.
"""

from __future__ import annotations

import hashlib
import json

from api.cache import PredictionCache, _feature_hash


class _FakeRedis:
    """Minimal dict-backed stand-in for the redis client surface used here."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def ping(self) -> bool:
        return True


def _cache_with_fake() -> PredictionCache:
    c = PredictionCache()
    c._client = _FakeRedis()  # force-enable without a live Redis
    return c


def test_roundtrip_when_enabled():
    c = _cache_with_fake()
    c.version = "v1"
    payload = {"state": "CA", "age": 30}
    c.set(payload, {"predicted_salary": 123.0})
    assert c.get(payload) == {"predicted_salary": 123.0}


def test_key_namespaced_by_model_version():
    """A version bump must invalidate the previous model's cached entries."""
    c = _cache_with_fake()
    payload = {"state": "CA", "age": 30}

    c.version = "v1"
    c.set(payload, {"predicted_salary": 100.0})
    assert c.get(payload) == {"predicted_salary": 100.0}

    # New model version → the old entry is no longer addressable.
    c.version = "v2"
    assert c.get(payload) is None

    # And the two versions coexist without collision.
    c.set(payload, {"predicted_salary": 200.0})
    assert c.get(payload) == {"predicted_salary": 200.0}
    c.version = "v1"
    assert c.get(payload) == {"predicted_salary": 100.0}


def test_disabled_cache_is_noop():
    c = PredictionCache()  # no client → disabled
    assert c.enabled is False
    assert c.get({"state": "CA"}) is None
    c.set({"state": "CA"}, {"predicted_salary": 1.0})  # must not raise


class _DeadRedis(_FakeRedis):
    """Every operation raises — Redis died after the cache was constructed."""

    def get(self, key: str):
        raise ConnectionError("simulated redis outage")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise ConnectionError("simulated redis outage")


def test_read_failure_degrades_to_a_miss():
    """A dead Redis must cost a cache hit, never a failed prediction."""
    c = _cache_with_fake()
    c._client = _DeadRedis()
    c.version = "v1"
    assert c.get({"state": "CA", "age": 30}) is None


def test_write_failure_is_swallowed():
    """A failed cache write must not propagate into the request path."""
    c = _cache_with_fake()
    c._client = _DeadRedis()
    c.version = "v1"
    c.set({"state": "CA", "age": 30}, {"predicted_salary": 1.0})  # must not raise


def test_corrupt_cached_value_degrades_to_a_miss():
    """Undecodable bytes in Redis are a miss, not a 500."""
    c = _cache_with_fake()
    c.version = "v1"
    payload = {"state": "CA", "age": 30}
    c.set(payload, {"predicted_salary": 1.0})
    key = next(iter(c._client.store))
    c._client.store[key] = "{not-json"
    assert c.get(payload) is None


def test_the_key_carries_the_whole_digest():
    """A truncated digest shrinks the key space below SHA-256's collision
    resistance, and a cache collision serves one caller another's prediction."""
    payload = {"state": "CA", "age": 30}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    assert _feature_hash(payload) == hashlib.sha256(canonical.encode()).hexdigest()
