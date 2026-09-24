"""Unit tests for Phase 7 RefillCare Controlled WhatsApp Reminder Dispatch."""

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import requests

from reminder.dispatch import (
    DispatchResult,
    RefillCareReminderDispatcher,
    build_refillcare_whatsapp_payload,
    select_eligible_reminders_for_dispatch,
    mask_phone_for_ui,
    sanitize_template_text,
    REFILLCARE_TEMPLATE_NAME,
)
from reminder.scheduler import ReminderRecord


def test_dispatch_imports_and_constants():
    """Verify dispatch constants and default template."""
    assert REFILLCARE_TEMPLATE_NAME == "refillcare_medicine_reminder"
    assert mask_phone_for_ui("919849012345") == "***2345"
    assert mask_phone_for_ui("12") == "***"


def test_sanitize_template_text():
    """Verify newlines, tabs, and whitespace are sanitized to single spaces."""
    raw = "Line 1\nLine 2\r\nLine 3\t\twith   spaces"
    sanitized = sanitize_template_text(raw)
    assert "\n" not in sanitized
    assert "\r" not in sanitized
    assert "\t" not in sanitized
    assert sanitized == "Line 1 Line 2 Line 3 with spaces"
    assert sanitize_template_text(None, default="fallback") == "fallback"


def test_build_valid_reminder_payload():
    """Verify construction of all 5 parameters in RefillCare WhatsApp template."""
    rem = ReminderRecord(
        reminder_id="C001___M001___2026-09-09___-7d",
        customerId="C001",
        itemId="M001",
        customerName="John Doe",
        MOBILE_NO="9849012345",
        itemName="TELMISARTAN 40MG",
        latest_purchase_date="2026-08-10",
        predicted_interval_days=30.0,
        expected_refill_date="2026-09-09",
        reminder_stage=-7,
        reminder_date="2026-09-02",
        message="Your regular medicine refill may be due in about 7 days, around 09-09-2026.",
        status="scheduled",
    )

    res = build_refillcare_whatsapp_payload(
        reminder=rem,
        store_name="PHARMA HUBB",
        store_contact="9876543210",
    )

    assert res["is_valid"] is True
    assert res["normalized_phone"] == "919849012345"
    assert res["ui_masked_phone"] == "***2345"
    assert res["template_name"] == REFILLCARE_TEMPLATE_NAME

    payload = res["payload"]
    assert payload["to"] == "919849012345"
    assert payload["template"]["name"] == REFILLCARE_TEMPLATE_NAME

    params = payload["template"]["components"][0]["parameters"]
    assert len(params) == 6
    assert params[0]["text"] == "John Doe"  # {{1}} Customer Name
    assert params[1]["text"] == "PHARMA HUBB"  # {{2}} Store Name
    assert params[2]["text"] == "Your regular medicine refill may be due in about 7 days, around 09-09-2026."  # {{3}} Message
    assert params[3]["text"] == "TELMISARTAN 40MG"  # {{4}} Medicine
    assert params[4]["text"] == "9876543210"  # {{5}} Store Contact
    assert params[5]["text"] == "PHARMA HUBB"  # {{6}} Store Name


def test_payload_validation_errors():
    """Verify validation failures for missing customerId, itemId, phone, or date."""
    base_rem = {
        "reminder_id": "REM_001",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Jane",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine X",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Reminder text",
    }

    # Missing customerId
    rem_no_cust = dict(base_rem, customerId="")
    res = build_refillcare_whatsapp_payload(rem_no_cust)
    assert res["is_valid"] is False
    assert "customerId" in res["error"]

    # Missing itemId
    rem_no_item = dict(base_rem, itemId="")
    res = build_refillcare_whatsapp_payload(rem_no_item)
    assert res["is_valid"] is False
    assert "itemId" in res["error"]

    # Missing phone
    rem_no_phone = dict(base_rem, MOBILE_NO="")
    res = build_refillcare_whatsapp_payload(rem_no_phone)
    assert res["is_valid"] is False
    assert "phone" in res["error"].lower()

    # Invalid phone (letters only)
    rem_bad_phone = dict(base_rem, MOBILE_NO="abcdef")
    res = build_refillcare_whatsapp_payload(rem_bad_phone)
    assert res["is_valid"] is False
    assert "Invalid destination phone" in res["error"]

    # Missing expected refill date
    rem_no_date = dict(base_rem, expected_refill_date="")
    res = build_refillcare_whatsapp_payload(rem_no_date)
    assert res["is_valid"] is False
    assert "expected refill date" in res["error"]


def test_dry_run_makes_no_external_api_call():
    """Verify that dry_run=True performs full validation with ZERO HTTP calls."""
    dispatcher = RefillCareReminderDispatcher()
    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "TELMISARTAN 40MG",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Your regular medicine refill may be due in about 7 days, around 09-09-2026.",
        "status": "scheduled",
    }

    with patch("requests.post") as mock_post:
        result = dispatcher.dispatch_single_reminder(rem, dry_run=True)
        assert mock_post.called is False  # Zero external HTTP calls

    assert result.success is True
    assert result.status == "accepted"
    assert result.dry_run is True
    assert "[DRY RUN]" in result.message
    assert result.phone_masked == "***2345"


def test_duplicate_reminder_protection():
    """Verify that already accepted/dispatched reminders are never sent twice."""
    dispatcher = RefillCareReminderDispatcher()
    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "TELMISARTAN 40MG",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Your regular medicine refill may be due in about 7 days.",
        "status": "scheduled",
    }

    # First dispatch (accepted)
    res1 = dispatcher.dispatch_single_reminder(rem, dry_run=True)
    assert res1.status == "accepted"

    # Second dispatch with identical reminder_id -> duplicate prevented
    res2 = dispatcher.dispatch_single_reminder(rem, dry_run=True)
    assert res2.status == "duplicate_prevented"
    assert res2.success is False
    assert "Duplicate Prevention" in res2.message


def test_in_batch_deduplication():
    """Verify in-batch duplicate deduplication."""
    dispatcher = RefillCareReminderDispatcher()
    rem1 = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Message text",
        "status": "scheduled",
    }
    rem2 = dict(rem1)  # Exact duplicate entry in batch

    results = dispatcher.dispatch_batch_reminders([rem1, rem2], dry_run=True)
    assert len(results) == 2
    assert results[0].status == "accepted"
    assert results[1].status == "duplicate_prevented"


def test_cancelled_reminder_is_not_sent():
    """Verify that reminders marked 'cancelled' by Phase 5 are not dispatched."""
    dispatcher = RefillCareReminderDispatcher()
    cancelled_rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Message text",
        "status": "cancelled",  # Repurchased early
    }

    result = dispatcher.dispatch_single_reminder(cancelled_rem, dry_run=True)
    assert result.status == "cancelled"
    assert result.success is False
    assert "cancelled" in result.message.lower()


def test_multiple_customers_sharing_phone_remain_separate():
    """Verify multiple customers sharing a single phone number are preserved as separate dispatches."""
    dispatcher = RefillCareReminderDispatcher()

    # Father on phone 9849012345
    rem_father = {
        "reminder_id": "CUST_FATHER___M001___2026-09-09___-7d",
        "customerId": "CUST_FATHER",
        "itemId": "M001",
        "customerName": "Father",
        "MOBILE_NO": "9849012345",
        "itemName": "BP Tablet",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Father reminder",
        "status": "scheduled",
    }

    # Mother on SAME phone 9849012345
    rem_mother = {
        "reminder_id": "CUST_MOTHER___M001___2026-09-09___-7d",
        "customerId": "CUST_MOTHER",
        "itemId": "M001",
        "customerName": "Mother",
        "MOBILE_NO": "9849012345",
        "itemName": "BP Tablet",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Mother reminder",
        "status": "scheduled",
    }

    results = dispatcher.dispatch_batch_reminders([rem_father, rem_mother], dry_run=True)
    assert len(results) == 2
    assert results[0].customerId == "CUST_FATHER"
    assert results[0].status == "accepted"
    assert results[1].customerId == "CUST_MOTHER"
    assert results[1].status == "accepted"


def test_mocked_live_success_dispatch(monkeypatch):
    """Verify live sending path with mocked successful 200 HTTP response."""
    monkeypatch.setenv("XINNO_API_KEY", "test_key_12345")
    monkeypatch.setenv("XINNO_WABA_NUMBER", "919999999999")

    dispatcher = RefillCareReminderDispatcher()
    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Stage message",
        "status": "scheduled",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "messages": [{"id": "wamid_HBgMOTE5ODQ5MDEyMzQ1FQIAERgSQ0ZGMzgzN0E2OTVDRDAyNzJBAA=="}]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = dispatcher.dispatch_single_reminder(rem, dry_run=False)
        assert mock_post.called is True

    assert result.success is True
    assert result.status == "accepted"
    assert result.provider_message_id == "wamid_HBgMOTE5ODQ5MDEyMzQ1FQIAERgSQ0ZGMzgzN0E2OTVDRDAyNzJBAA=="
    assert result.dry_run is False


def test_mocked_live_failure_dispatch(monkeypatch):
    """Verify live sending error path with mocked 400 response from Meta/Xinno."""
    monkeypatch.setenv("XINNO_API_KEY", "test_key_12345")
    monkeypatch.setenv("XINNO_WABA_NUMBER", "919999999999")

    dispatcher = RefillCareReminderDispatcher()
    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Stage message",
        "status": "scheduled",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error": {
            "message": "Template does not exist in the specified language.",
            "type": "OAuthException",
            "code": 100,
        }
    }

    with patch("requests.post", return_value=mock_resp):
        result = dispatcher.dispatch_single_reminder(rem, dry_run=False)

    assert result.success is False
    assert result.status == "failed"
    assert result.status_code == 400
    assert result.error_category == "API Error"
    assert "Template does not exist" in result.message


def test_mocked_live_network_error_dispatch(monkeypatch):
    """Verify safe handling and masking when requests.post raises a connection timeout."""
    monkeypatch.setenv("XINNO_API_KEY", "super_secret_api_key_xyz")
    monkeypatch.setenv("XINNO_WABA_NUMBER", "919999999999")

    dispatcher = RefillCareReminderDispatcher()
    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Stage message",
        "status": "scheduled",
    }

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        result = dispatcher.dispatch_single_reminder(rem, dry_run=False)

    assert result.success is False
    assert result.status == "failed"
    assert "Timeout" in result.error_category
    assert "super_secret_api_key_xyz" not in result.message


def test_select_eligible_reminders_for_dispatch():
    """Verify filtering of reminders due on a specific target date."""
    rems = [
        {"reminder_id": "R1", "customerId": "C1", "itemId": "M1", "MOBILE_NO": "9849012345", "reminder_date": "2026-09-02", "status": "scheduled"},
        {"reminder_id": "R2", "customerId": "C2", "itemId": "M2", "MOBILE_NO": "9849012345", "reminder_date": "2026-09-03", "status": "scheduled"},
        {"reminder_id": "R3", "customerId": "C3", "itemId": "M3", "MOBILE_NO": "9849012345", "reminder_date": "2026-09-02", "status": "cancelled"},
        {"reminder_id": "R4", "customerId": "", "itemId": "M4", "MOBILE_NO": "9849012345", "reminder_date": "2026-09-02", "status": "scheduled"},
    ]

    due_02 = select_eligible_reminders_for_dispatch(rems, target_date="2026-09-02")
    assert len(due_02) == 1
    assert due_02[0]["reminder_id"] == "R1"


def test_dispatcher_to_dataframe():
    """Verify dispatch results export to a clean pandas DataFrame."""
    dispatcher = RefillCareReminderDispatcher()
    df_empty = dispatcher.to_dataframe()
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0

    rem = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "expected_refill_date": "2026-09-09",
        "reminder_stage": -7,
        "message": "Message text",
        "status": "scheduled",
    }
    dispatcher.dispatch_single_reminder(rem, dry_run=True)

    df_populated = dispatcher.to_dataframe()
    assert len(df_populated) == 1
    assert "reminder_id" in df_populated.columns
    assert "phone_masked" in df_populated.columns
    assert "dry_run" in df_populated.columns
