"""API protection: X-API-Key on every /api route; local-machine only without a key; fail closed in production."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.security import require_api_key


def _request(host: str, headers=None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw, "client": (host, 50000)})


def test_without_key_only_local_machine_may_write(monkeypatch):
    monkeypatch.delenv("REFILLCARE_API_KEY", raising=False)
    assert require_api_key(_request("127.0.0.1")) == "local"
    with pytest.raises(HTTPException) as e:
        require_api_key(_request("192.168.1.50"))
    assert e.value.status_code == 403


def test_with_key_header_is_required_and_checked(monkeypatch):
    monkeypatch.setenv("REFILLCARE_API_KEY", "s3cret-key")
    with pytest.raises(HTTPException) as e:
        require_api_key(_request("127.0.0.1"))
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        require_api_key(_request("192.168.1.50", {"X-API-Key": "wrong"}))
    assert require_api_key(_request("192.168.1.50", {"X-API-Key": "s3cret-key"})) == "api-key"


def test_reads_and_writes_need_the_key_health_stays_open(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("REFILLCARE_API_KEY", "s3cret-key")
    c = TestClient(app)
    assert c.post("/api/v1/sales/batches/BATCH_X/rollback").status_code == 401
    assert c.delete("/api/v1/medications/rules/1").status_code == 401
    # Reads carry patient names and phones: they need the key too
    assert c.get("/api/v1/ml/models").status_code == 401
    assert c.get("/api/v1/reminders/export-csv").status_code == 401
    assert c.get("/api/v1/ml/models", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert c.get("/livez").status_code == 200


def test_production_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("REFILLCARE_API_KEY", raising=False)
    monkeypatch.setenv("REFILLCARE_ENV", "production")
    with pytest.raises(HTTPException) as e:
        require_api_key(_request("127.0.0.1"))
    assert e.value.status_code == 503


def test_webhook_needs_a_valid_signature_when_a_secret_is_set(monkeypatch):
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("WHATSAPP_WEBHOOK_SECRET", "hook-secret")
    c = TestClient(app)
    body = b'{"from": "919848012345", "text": "STOP"}'
    assert c.post("/api/v1/whatsapp/webhook", content=body).status_code == 401
    bad = {"X-Hub-Signature-256": "sha256=" + "0" * 64}
    assert c.post("/api/v1/whatsapp/webhook", content=body, headers=bad).status_code == 401
    good = {"X-Hub-Signature-256": "sha256=" + hmac.new(b"hook-secret", body, hashlib.sha256).hexdigest(),
            "Content-Type": "application/json"}
    assert c.post("/api/v1/whatsapp/webhook", content=body, headers=good).status_code == 200


def test_store_header_is_validated(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    c = TestClient(app)
    r = c.get("/api/v1/stores/current", headers={"X-Store-Id": "bad store; drop"})
    assert r.status_code == 400
    r = c.get("/api/v1/stores/current", headers={"X-Store-Id": "ngos-colony"})
    assert r.status_code == 200 and r.json()["store_id"] == "ngos-colony"


def test_rate_limiter_blocks_bursts():
    from api.security import RateLimiter

    rl = RateLimiter(per_minute=3)
    assert [rl.allow("k") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("other")
