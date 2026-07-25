"""
API security tests — auth, CORS, rate limiting, and proxy-IP handling.

These cover the controls the README advertises: ``API_KEY`` enforcement (401),
the CORS allow-list, the per-IP rate limiter (429), and the ``X-Forwarded-For``
parsing that decides *which* IP a caller is bucketed under.

``API_KEY``, ``CORS_ORIGINS``, ``RATE_LIMIT`` and ``TRUSTED_PROXY_HOPS`` are all
read from the environment at module-import time (the limiter and the CORS
middleware are constructed once when ``api.main`` is imported). To exercise a
given configuration we therefore set the environment and ``importlib.reload``
the module, then reload it again on teardown so no configuration leaks into
other test files.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers


def _clear_prometheus_registry() -> None:
    """Unregister all collectors from the default Prometheus registry.

    ``api.main`` registers a module-level ``Counter`` and the FastAPI
    instrumentator against the global default registry at import time.
    Reloading the module re-runs that registration, which raises
    ``Duplicated timeseries`` unless the registry is cleared first. This is a
    reload-only concern; nothing in production re-imports the module.
    """
    from prometheus_client import REGISTRY

    for collector in list(getattr(REGISTRY, "_collector_to_names", {}).keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@contextmanager
def reloaded_module(**env: str | None):
    """Reload ``api.main`` with the given environment applied.

    A value of ``None`` deletes the variable for the duration of the block.
    Construction-time globals (``API_KEY``, ``CORS_ORIGINS``, ``RATE_LIMIT``,
    ``TRUSTED_PROXY_HOPS``, the ``Limiter`` and the CORS middleware) are
    rebuilt by the reload. Importing does **not** load the model — that only
    happens inside the lifespan, when a ``TestClient`` is used as a context
    manager.
    """
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import api.main as m

        _clear_prometheus_registry()
        importlib.reload(m)
        yield m
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        import api.main as m2

        _clear_prometheus_registry()
        importlib.reload(m2)


def _fake_request(xff: str | None = None, client_host: str = "203.0.113.250"):
    """A minimal stand-in for ``starlette.Request`` for ``_client_ip``."""
    raw = {"X-Forwarded-For": xff} if xff is not None else {}

    class _Req:
        headers = Headers(raw)
        client = type("Client", (), {"host": client_host})()

    return _Req()


# ── X-Forwarded-For / rate-limit bucketing (spoofing defense) ───────────────


class TestClientIpProxySecurity:
    """Lock the security property that a forged X-Forwarded-For cannot move a
    caller to a fresh rate-limit bucket: with N trusted proxy hops the client IP
    is read from the Nth-from-right XFF entry, so an attacker-controlled
    *left-most* entry cannot mint a new bucket."""

    def test_no_proxy_uses_direct_peer(self):
        with reloaded_module(TRUSTED_PROXY_HOPS="0") as m:
            req = _fake_request(xff="1.2.3.4", client_host="198.51.100.9")
            # With 0 trusted hops, XFF is ignored entirely.
            assert m._client_ip(req) == "198.51.100.9"

    def test_single_proxy_trusts_rightmost_entry(self):
        with reloaded_module(TRUSTED_PROXY_HOPS="1") as m:
            # The trusted proxy appended the real peer (203.0.113.7); the
            # leading entry is whatever the client sent and is untrusted.
            req = _fake_request(xff="1.1.1.1, 203.0.113.7")
            assert m._client_ip(req) == "203.0.113.7"

    def test_forged_xff_cannot_mint_new_bucket(self):
        """Two requests from the same real peer but different forged leading
        entries MUST resolve to the same IP — otherwise the rate limiter is
        trivially bypassable by rotating the spoofed value."""
        with reloaded_module(TRUSTED_PROXY_HOPS="1") as m:
            a = m._client_ip(_fake_request(xff="1.1.1.1, 203.0.113.7"))
            b = m._client_ip(_fake_request(xff="2.2.2.2, 203.0.113.7"))
            c = m._client_ip(_fake_request(xff="9.9.9.9, 203.0.113.7"))
            assert a == b == c == "203.0.113.7"
            # And specifically NOT the attacker-controlled leading entries.
            assert a not in {"1.1.1.1", "2.2.2.2", "9.9.9.9"}

    def test_two_trusted_proxies_peel_two_entries(self):
        with reloaded_module(TRUSTED_PROXY_HOPS="2") as m:
            # client(spoof), real-client, proxy1 — 2 trusted hops peel the
            # right two; real client is index -2.
            req = _fake_request(xff="evil, 203.0.113.7, 10.0.0.1")
            assert m._client_ip(req) == "203.0.113.7"

    def test_too_few_hops_falls_back_to_peer(self):
        with reloaded_module(TRUSTED_PROXY_HOPS="2") as m:
            # Only one entry but two proxies configured → header can't have
            # come through the chain, so it is ignored.
            req = _fake_request(xff="1.1.1.1", client_host="198.51.100.9")
            assert m._client_ip(req) == "198.51.100.9"


# ── API key authentication ───────────────────────────────────────────────────


class TestUnhandledErrorsAreScrubbed:
    """A 500 body must carry a correlation id and nothing about the internals.

    Exception text routinely embeds file paths, feature values and library
    internals, so returning it verbatim to an unauthenticated caller leaks the
    server's shape.
    """

    def test_500_body_hides_the_exception_and_keeps_the_request_id(self):
        with reloaded_module() as m:
            secret = "s3cr3t-internal-detail-/srv/models/xgb.ubj"

            @m.app.get("/_boom")
            async def _boom():  # pragma: no cover - body raises by design
                raise RuntimeError(secret)

            client = TestClient(m.app, raise_server_exceptions=False)
            r = client.get("/_boom", headers={"X-Request-ID": "corr-123"})

            assert r.status_code == 500
            body = r.json()
            assert body["detail"] == "Internal server error"
            assert body["request_id"] == "corr-123"
            assert secret not in r.text
            assert "RuntimeError" not in r.text
            assert "Traceback" not in r.text


class TestApiKeyAuth:
    def test_missing_key_rejected_401(self):
        with reloaded_module(API_KEY="s3cret") as m:
            client = TestClient(m.app)  # no lifespan needed: auth runs first
            r = client.post("/predict", json={"state": "CA"})
            assert r.status_code == 401

    def test_wrong_key_rejected_401(self):
        with reloaded_module(API_KEY="s3cret") as m:
            client = TestClient(m.app)
            r = client.post("/predict", json={"state": "CA"}, headers={"X-API-Key": "nope"})
            assert r.status_code == 401

    def test_non_ascii_key_rejected_401(self):
        """A key carrying bytes outside ASCII is a wrong key, not a server fault.

        Header bytes are decoded as latin-1, so any byte >= 0x80 yields a
        non-ASCII ``str``; the comparison must still resolve to a clean 401.
        """
        with reloaded_module(API_KEY="s3cret") as m:
            client = TestClient(m.app)
            r = client.post("/predict", json={"state": "CA"}, headers={"X-API-Key": b"caf\xe9"})
            assert r.status_code == 401, r.text

    def test_non_ascii_key_failures_count_against_the_throttle(self):
        """Every rejected key counts toward the per-IP failure budget.

        The throttle is the brute-force control, so no encoding of the
        submitted key may provide an unmetered guessing channel.
        """
        with reloaded_module(API_KEY="s3cret", AUTH_FAILURE_LIMIT="3") as m:
            client = TestClient(m.app)
            codes = [
                client.post("/predict", json={"state": "CA"}, headers={"X-API-Key": b"caf\xe9"}).status_code
                for _ in range(6)
            ]
            assert codes[:3] == [401, 401, 401], codes
            assert codes[3:] == [429, 429, 429], codes

    @pytest.mark.parametrize(
        ("label", "bad_key"),
        [
            ("non-ascii", "café-secret"),
            ("trailing newline", "s3cret\n"),  # `kubectl create secret --from-file`
            ("leading space", " s3cret"),
            ("trailing space", "s3cret "),
            ("embedded tab", "s3\tcret"),
            ("embedded space", "s3 cret"),
            ("control character", "s3cret\x01"),
        ],
    )
    def test_key_that_cannot_survive_a_header_is_refused_at_startup(self, label, bad_key):
        """Configurations that would 401 the correct key must fail loudly.

        None of these can round-trip an HTTP header: non-ASCII has no agreed
        encoding (httpx refuses to send it, http.client sends latin-1), leading
        and trailing whitespace is stripped by the server's parser, and control
        characters are rejected by the client. Each would otherwise reject the
        operator's own key on every request with no diagnostic.
        """
        with pytest.raises(RuntimeError, match="API_KEY must consist only of printable ASCII"):
            with reloaded_module(API_KEY=bad_key):
                pass

    def test_ordinary_key_is_accepted(self):
        """The guard must not reject keys operators actually use."""
        for good in ("s3cret", "YWJjZA==", "a1b2c3d4e5f6", "key-with_symbols.~+/"):
            with reloaded_module(API_KEY=good) as m:
                assert m.API_KEY == good

    def test_correct_key_accepted(self):
        with reloaded_module(API_KEY="s3cret") as m:
            with TestClient(m.app) as client:  # lifespan → model loaded
                occ = m.state.occupations[0]
                payload = {
                    "state": "CA",
                    "occupation": occ,
                    "education_level": "Bachelor's degree",
                    "gender": "Female",
                    "age": 32,
                }
                r = client.post("/predict", json=payload, headers={"X-API-Key": "s3cret"})
                assert r.status_code == 200, r.text

    def test_dev_mode_no_key_required(self):
        with reloaded_module(API_KEY=None) as m:
            with TestClient(m.app) as client:
                occ = m.state.occupations[0]
                payload = {
                    "state": "CA",
                    "occupation": occ,
                    "education_level": "Bachelor's degree",
                    "gender": "Female",
                    "age": 32,
                }
                r = client.post("/predict", json=payload)
                assert r.status_code == 200, r.text


# ── CORS allow-list ───────────────────────────────────────────────────────────


class TestCors:
    def test_allowed_origin_is_reflected(self):
        with reloaded_module(CORS_ORIGINS="https://allowed.example") as m:
            client = TestClient(m.app)
            r = client.options(
                "/predict",
                headers={
                    "Origin": "https://allowed.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert r.headers.get("access-control-allow-origin") == "https://allowed.example"

    def test_disallowed_origin_not_granted(self):
        with reloaded_module(CORS_ORIGINS="https://allowed.example") as m:
            client = TestClient(m.app)
            r = client.options(
                "/predict",
                headers={
                    "Origin": "https://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert r.headers.get("access-control-allow-origin") != "https://evil.example"

    def test_default_is_closed(self):
        # CORS_ORIGINS unset → no cross-origin access granted to anyone.
        with reloaded_module(CORS_ORIGINS=None) as m:
            client = TestClient(m.app)
            r = client.options(
                "/predict",
                headers={
                    "Origin": "https://anything.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert r.headers.get("access-control-allow-origin") is None


# ── Rate limiting (429) ───────────────────────────────────────────────────────


class TestRateLimiting:
    """The limiter is wired to ``/predict`` via ``@limiter.limit(RATE_LIMIT)``;
    undecorated routes (``/health``) are intentionally unlimited, so these tests
    drive ``/predict`` with a loaded model."""

    @staticmethod
    def _payload(m):
        return {
            "state": "CA",
            "occupation": m.state.occupations[0],
            "education_level": "Bachelor's degree",
            "gender": "Female",
            "age": 32,
        }

    def test_limit_enforced_returns_429(self):
        """Under a tight limit, repeated calls eventually return 429, and every
        success precedes the first rejection (monotonic enforcement)."""
        with reloaded_module(RATE_LIMIT="5/minute") as m:
            with TestClient(m.app) as client:
                payload = self._payload(m)
                statuses = [client.post("/predict", json=payload).status_code for _ in range(8)]

            assert statuses[0] == 200, f"first request must be under the limit: {statuses}"
            assert 429 in statuses, f"rate limit was never enforced: {statuses}"
            first_reject = statuses.index(429)
            assert all(s == 200 for s in statuses[:first_reject]), (
                f"a request was rejected before the limit: {statuses}"
            )

    def test_429_body_has_detail(self):
        with reloaded_module(RATE_LIMIT="2/minute") as m:
            with TestClient(m.app) as client:
                payload = self._payload(m)
                last = None
                for _ in range(6):
                    last = client.post("/predict", json=payload)
                    if last.status_code == 429:
                        break
            assert last is not None and last.status_code == 429
            assert "detail" in last.json()


# ── Request body size limit (413) ─────────────────────────────────────────────


class TestBodySizeLimit:
    """The middleware caps the actual streamed byte count, so a chunked upload
    (Transfer-Encoding: chunked, no Content-Length) is bounded exactly like a
    declared-length body — no Content-Length header is required to enforce the
    cap."""

    def test_declared_oversize_content_length_rejected_413(self):
        with reloaded_module(MAX_BODY_BYTES="1024") as m:
            client = TestClient(m.app)
            # A normal (Content-Length-bearing) oversize body is still rejected.
            r = client.post("/predict", content=b"x" * 4096)
            assert r.status_code == 413

    def test_chunked_oversize_body_rejected_413(self):
        with reloaded_module(MAX_BODY_BYTES="1024") as m:
            client = TestClient(m.app)

            def big_chunks():
                # An iterable body makes httpx use Transfer-Encoding: chunked with
                # NO Content-Length — the case a Content-Length-only size cap misses.
                for _ in range(8):
                    yield b"x" * 512  # 4096 bytes total, over the 1024 cap

            r = client.post("/predict", content=big_chunks())
            assert r.status_code == 413, f"chunked oversize body bypassed the limit: {r.status_code}"

    def test_chunked_under_cap_passes_through(self):
        # A small chunked body must still reach the handler (here it fails
        # validation with 422, proving it got past the size gate, not 413).
        with reloaded_module(MAX_BODY_BYTES="65536") as m:
            with TestClient(m.app) as client:

                def small_chunks():
                    yield b'{"state":'
                    yield b'"CA"}'

                r = client.post("/predict", content=small_chunks())
                assert r.status_code != 413
                assert r.status_code == 422  # reached Pydantic; missing fields


# ── Failed-auth brute-force throttle (429) ────────────────────────────────────


class TestAuthFailureThrottle:
    """A 401 from verify_api_key short-circuits before the route-level limiter,
    so the route limiter alone never throttles repeated wrong keys. The per-IP
    failure throttle caps them: the first N return 401, then further attempts
    return 429."""

    def test_repeated_failed_auth_eventually_429(self):
        with reloaded_module(API_KEY="s3cret", AUTH_FAILURE_LIMIT="3") as m:
            client = TestClient(m.app)
            statuses = [
                client.post("/predict", json={"state": "CA"}, headers={"X-API-Key": "wrong"}).status_code
                for _ in range(6)
            ]
            assert statuses[:3] == [401, 401, 401], statuses
            assert 429 in statuses[3:], f"brute-force was never throttled: {statuses}"

    def test_successful_auth_not_throttled(self):
        # A valid key must never be throttled, even after the failure budget would
        # have been exhausted by OTHER (failed) attempts — success bypasses it.
        with reloaded_module(API_KEY="s3cret", AUTH_FAILURE_LIMIT="2") as m:
            with TestClient(m.app) as client:
                payload = {
                    "state": "CA",
                    "occupation": m.state.occupations[0],
                    "education_level": "Bachelor's degree",
                    "gender": "Female",
                    "age": 32,
                }
                # Many valid requests in a row stay 200 (failure throttle untouched).
                for _ in range(5):
                    r = client.post("/predict", json=payload, headers={"X-API-Key": "s3cret"})
                    assert r.status_code == 200, r.text

    def test_throttle_evicts_stale_ips_and_does_not_grow_unbounded(self):
        """Regression: the per-IP failure map must not grow without bound as
        attackers rotate source IPs. A per-IP deque is never emptied in place (the
        just-recorded hit keeps it non-empty), so eviction happens via a windowed
        sweep of keys whose newest failure has aged out. Drives time explicitly via
        the injected ``now`` so it is deterministic."""
        from api.main import _AuthFailureThrottle

        thr = _AuthFailureThrottle(limit=3, window_s=60.0)
        # 1000 distinct IPs all fail once at t=0.
        for i in range(1000):
            thr.record_failure(f"10.0.{i // 256}.{i % 256}", now=0.0)
        assert len(thr._hits) == 1000
        # A single new IP fails two windows later → triggers the stale sweep, which
        # drops every aged-out IP. The map collapses to just the active one.
        still_ok = thr.record_failure("203.0.113.7", now=130.0)
        assert still_ok is True
        assert len(thr._hits) == 1, f"stale IPs not evicted: {len(thr._hits)} remain"
        assert "203.0.113.7" in thr._hits
