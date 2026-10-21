"""v1 API protection: a shared API key for write actions.

- Read endpoints are open (run the API on the pharmacy LAN only).
- Write actions (uploads, rollback, approvals, WhatsApp sends, rule changes, model promotion)
  require header ``X-API-Key`` equal to ``REFILLCARE_API_KEY`` from ``.env``.
- If ``REFILLCARE_API_KEY`` is not set, write actions are accepted only from the same machine
  (127.0.0.1 / ::1) so a key-less local setup cannot be driven from the network.

v2 replaces this with the Mediastra login token (tenant / store / org scoped).
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request, status

logger = logging.getLogger("refillcare.api.security")
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_warned = False


def require_api_key(request: Request) -> str:
    """FastAPI dependency for write endpoints. Returns an actor label for audit fields."""
    global _warned
    expected = os.getenv("REFILLCARE_API_KEY", "").strip()
    if not expected:
        client = request.client.host if request.client else ""
        if client in _LOCAL_HOSTS:
            if not _warned:
                logger.warning("REFILLCARE_API_KEY is not set: write actions are allowed from this machine only.")
                _warned = True
            return "local"
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Write actions are only allowed from the server machine (set REFILLCARE_API_KEY to enable remote use).")
    provided = request.headers.get("X-API-Key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key.")
    return "api-key"
