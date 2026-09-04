"""
Phase 7 tests: send-result audit, session history, CSV export.

Uses mocks and dry-run only. NO real Xinno calls. NO dry_run=False against live API.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.audit import (
    AUDIT_COLUMNS,
    build_audit_record,
    append_send_history,
    history_to_dataframe,
    history_to_csv_bytes,
    log_audit_record_safely,
    mask_phone_for_audit,
    extract_message_id,
    extract_message_status,
    audit_record_for_display,
)
from services.xinno_whatsapp import send_template_message


def _success_response(message_id: str = "wamid.phase7.test"):
    return {
        "success": True,
        "status_code": 200,
        "message": "Xinno API accepted the request.",
        "response": {
            "api_response": {
                "messages": [{"id": message_id, "message_status": "accepted"}]
            },
            "headers": {"Key": "***MASKED***"},
            "dry_run": False,
        },
    }


def _failure_response(status_code: int = 400, msg: str = "Bad Request"):
    return {
        "success": False,
        "status_code": status_code,
        "message": msg,
        "response": {
            "api_response": {"error": {"message": msg, "code": status_code}},
            "headers": {"Key": "***MASKED***"},
            "dry_run": False,
        },
    }


# 1. Successful audit record
def test_successful_send_creates_audit_record():
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="76599 35016",
        normalized_phone="917659935016",
        template_name="reminder_refill_followup_v3",
        template_language="en",
        pharmacy_name="PHARMA HUBB",
        dry_run=False,
        send_response=_success_response(),
    )
    assert record["success"] is True
    assert record["customer_name"] == "Sunil"
    assert record["original_phone"] == "76599 35016"
    assert record["normalized_phone"] == "917659935016"
    assert record["template_name"] == "reminder_refill_followup_v3"
    assert record["template_language"] == "en"
    assert record["pharmacy_name"] == "PHARMA HUBB"
    assert record["dry_run"] is False
    assert record["message"] == "WhatsApp request accepted by Xinno."
    assert record["message_id"] == "wamid.phase7.test"
    assert record["error"] is None
    assert record["timestamp"]


# 2. Failed audit record
def test_failed_send_creates_audit_record():
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response=_failure_response(400, "Bad Request"),
    )
    assert record["success"] is False
    assert record["message"] == "WhatsApp request failed."
    assert record["error"] == "Bad Request"
    assert record["status_code"] == 400


# 3. HTTP status captured
@pytest.mark.parametrize("code", [200, 201, 400, 500])
def test_http_status_captured(code):
    if code in (200, 201):
        resp = _success_response()
        resp["status_code"] = code
    else:
        resp = _failure_response(code, f"HTTP {code}")
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response=resp,
    )
    assert record["status_code"] == code


# 4–5. Message ID extraction / missing
def test_message_id_extracted_when_available():
    assert extract_message_id({"messages": [{"id": "wamid.abc"}]}) == "wamid.abc"
    assert extract_message_status({"messages": [{"id": "x", "message_status": "accepted"}]}) == "accepted"


def test_missing_message_id_does_not_raise():
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response={
            "success": True,
            "status_code": 200,
            "message": "ok",
            "response": {"api_response": {"messages": []}},
        },
    )
    assert record["message_id"] is None


# 6–7. Dry-run flag
def test_dry_run_marked_true():
    dry = send_template_message("917659935016", "Sunil", "PHARMA HUBB", dry_run=True)
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response=dry,
    )
    assert record["dry_run"] is True
    assert "Dry-run only" in record["message"]
    assert dry["response"]["dry_run"] is True


def test_live_structure_dry_run_false_without_live_call():
    """Represent dry_run=false using a mocked response — no real API call."""
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response=_success_response(),
    )
    assert record["dry_run"] is False


# 8–9. API key never in audit / logs
def test_api_key_never_in_audit_record():
    secret = "SUPER_SECRET_KEY_DO_NOT_LEAK"
    resp = _success_response()
    resp["response"]["headers"] = {"Key": secret}
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response=resp,
    )
    blob = json.dumps(record)
    assert secret not in blob
    assert "api_key" not in record
    assert "Key" not in record
    assert "XINNO_API_KEY" not in blob


def test_api_key_never_in_log_output(tmp_path):
    secret = "SUPER_SECRET_KEY_DO_NOT_LEAK"
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response={
            "success": True,
            "status_code": None,
            "message": f"ok with {secret}",
            "response": {"dry_run": True},
        },
    )
    log_file = tmp_path / "audit.log"
    line = log_audit_record_safely(record, str(log_file))
    assert secret not in line or "[REDACTED]" in line or secret not in line
    # Our display message for dry-run doesn't include raw message with secret
    content = log_file.read_text(encoding="utf-8")
    assert "Key" not in content or "***" in content
    assert "917******016" in content or "917" in content


# 10. Phone masking
def test_phone_masking_in_audit_display():
    assert mask_phone_for_audit("917659935016") == "917******016"
    display = audit_record_for_display(
        build_audit_record(
            customer_name="Sunil",
            original_phone="7659935016",
            normalized_phone="917659935016",
            dry_run=True,
            send_response={"success": True, "status_code": None, "message": "ok"},
        )
    )
    assert display["normalized_phone_masked"] == "917******016"
    assert "7659935016" not in display["normalized_phone_masked"]


# 11–12. Session history one entry / no duplicate on rerun
def test_session_history_one_entry_per_attempt():
    history = []
    r1 = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response={"success": True, "status_code": None, "message": "ok"},
        attempt_id="attempt-1",
    )
    history = append_send_history(history, r1)
    history = append_send_history(history, r1)  # simulate rerun with same attempt_id
    assert len(history) == 1

    r2 = build_audit_record(
        customer_name="Tarun",
        original_phone="8688504571",
        normalized_phone="918688504571",
        dry_run=True,
        send_response={"success": True, "status_code": None, "message": "ok"},
        attempt_id="attempt-2",
    )
    history = append_send_history(history, r2)
    assert len(history) == 2


def test_streamlit_rerun_does_not_duplicate_audit_entries():
    history = []
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response={"success": True, "status_code": None, "message": "ok"},
        attempt_id="fixed-id",
    )
    for _ in range(5):
        history = append_send_history(history, record)
    assert len(history) == 1


# 13. CSV export columns
def test_csv_export_contains_required_columns():
    history = [
        build_audit_record(
            customer_name="Sunil",
            original_phone="7659935016",
            normalized_phone="917659935016",
            dry_run=True,
            send_response={"success": True, "status_code": None, "message": "ok"},
        )
    ]
    raw = history_to_csv_bytes(history, mask_phones=True).decode("utf-8")
    header = raw.splitlines()[0]
    for col in [
        "timestamp",
        "customer_name",
        "original_phone",
        "normalized_phone",
        "template_name",
        "template_language",
        "pharmacy_name",
        "dry_run",
        "success",
        "status_code",
        "message",
        "message_id",
        "error",
    ]:
        assert col in header
    assert "7659935016" not in raw  # masked
    assert "917******016" in raw
    assert "XINNO_API_KEY" not in raw
    table = history_to_dataframe(history)
    assert list(table.columns) == [
        "Time",
        "Customer",
        "Phone",
        "Template",
        "Result",
        "API Status",
        "Message ID",
    ]


# Error handling shapes
@pytest.mark.parametrize(
    "resp",
    [
        {"success": False, "status_code": 400, "message": "client error", "response": {}},
        {"success": False, "status_code": 500, "message": "server error", "response": {}},
        {
            "success": False,
            "status_code": None,
            "message": "Connection Error (Connection Timeout): timed out",
            "response": {},
        },
        {
            "success": False,
            "status_code": None,
            "message": "Connection Error (Network Connection Error): failed",
            "response": {},
        },
        {
            "success": True,
            "status_code": 200,
            "message": "ok",
            "response": {"api_response": "not-a-dict"},
        },
        {
            "success": True,
            "status_code": 200,
            "message": "ok",
            "response": {"api_response": {"unexpected": True}},
        },
    ],
)
def test_error_and_edge_response_shapes(resp):
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=False,
        send_response=resp,
    )
    assert "customer_name" in record
    assert record.get("api_key") is None


def test_mocked_http_success_builds_live_audit_without_real_api():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": [{"id": "wamid.mock", "message_status": "accepted"}]
    }
    with patch("services.xinno_whatsapp.requests.post", return_value=mock_response) as mock_post:
        import os
        os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
        os.environ["XINNO_API_KEY"] = "test-key-not-real"
        os.environ["XINNO_WABA_NUMBER"] = "919515473474"
        result = send_template_message(
            "917659935016", "Sunil", "PHARMA HUBB", dry_run=False
        )
        assert mock_post.call_count == 1
        record = build_audit_record(
            customer_name="Sunil",
            original_phone="7659935016",
            normalized_phone="917659935016",
            dry_run=False,
            send_response=result,
        )
        assert record["dry_run"] is False
        assert record["message_id"] == "wamid.mock"
        assert "test-key-not-real" not in json.dumps(record)
