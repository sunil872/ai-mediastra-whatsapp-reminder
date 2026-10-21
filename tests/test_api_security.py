"""v1 write protection: X-API-Key for write actions; local-machine only when no key is configured."""
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


def test_read_endpoints_stay_open_and_writes_need_key(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("REFILLCARE_API_KEY", "s3cret-key")
    c = TestClient(app)
    assert c.post("/api/v1/sales/batches/BATCH_X/rollback").status_code == 401
    assert c.delete("/api/v1/medications/rules/1").status_code == 401
    assert c.get("/api/v1/ml/models").status_code == 200
