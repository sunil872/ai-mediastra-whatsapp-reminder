"""API protection for service-to-service use (the Mediastra backend calls this API).

- **Every** ``/api/*`` route needs header ``X-API-Key`` = ``REFILLCARE_API_KEY`` (reads too: the
  responses contain patient names and phones). The store comes from ``X-Store-Id``.
- Development without a key: only requests from the same machine are accepted.
- Production (``REFILLCARE_ENV=production``): fails closed - no key configured -> 503.
- The WhatsApp webhook is the one exception (the provider cannot send our key): it must carry a
  valid HMAC signature (``X-Hub-Signature-256``) or token (``X-Webhook-Token``) for
  ``WHATSAPP_WEBHOOK_SECRET``; in production the secret is mandatory.
- A per-client token bucket limits request rates (``REFILLCARE_RATE_LIMIT_PER_MIN``).

User login (tenant / store / org) is handled by Mediastra; v2 can swap the shared key for its token.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from typing import Dict, Tuple

from fastapi import HTTPException, Request, status

logger = logging.getLogger("refillcare.api.security")
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_warned = False


def _is_production() -> bool:
    from refillcare.tenancy import is_production

    return is_production()


def check_api_key(request: Request) -> str:
    """Actor label for audit fields; raises 401 / 403 / 503."""
    global _warned
    expected = os.getenv("REFILLCARE_API_KEY", "").strip()
    if not expected:
        if _is_production():
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Server misconfigured: REFILLCARE_API_KEY is required in production.")
        client = request.client.host if request.client else ""
        if client in _LOCAL_HOSTS:
            if not _warned:
                logger.warning("REFILLCARE_API_KEY is not set: the API only accepts requests from this machine.")
                _warned = True
            return "local"
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Only requests from the server machine are allowed (set REFILLCARE_API_KEY to enable remote use).")
    provided = request.headers.get("X-API-Key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key.")
    caller = request.headers.get("X-Actor", "").strip()[:64]  # optional: Mediastra user id for the audit trail
    return f"api-key:{caller}" if caller else "api-key"


def require_api_key(request: Request) -> str:
    """FastAPI dependency (write endpoints; reads are checked by the middleware)."""
    return check_api_key(request)


# ----------------------------------------------------------------------------- webhook signature
def verify_webhook(request: Request, body: bytes) -> None:
    secret = os.getenv("WHATSAPP_WEBHOOK_SECRET", "").strip()
    if not secret:
        if _is_production():
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Server misconfigured: WHATSAPP_WEBHOOK_SECRET is required in production.")
        return  # development: unsigned callbacks accepted
    sig = request.headers.get("X-Hub-Signature-256", "")
    if sig:
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig.strip(), expected):
            return
    token = request.headers.get("X-Webhook-Token", "") or request.query_params.get("token", "")
    if token and hmac.compare_digest(token, secret):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature.")


# ----------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Token bucket per client key (in-process; put a gateway limit in front for multi-instance)."""

    def __init__(self, per_minute: int):
        self.rate = max(1, per_minute) / 60.0
        self.capacity = max(1, per_minute)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - cost, now)
            if len(self._buckets) > 10000:
                self._buckets.clear()
            return True


rate_limiter = RateLimiter(int(os.getenv("REFILLCARE_RATE_LIMIT_PER_MIN", "240")))
