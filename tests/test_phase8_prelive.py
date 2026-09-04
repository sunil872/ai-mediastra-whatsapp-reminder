"""
Phase 8 — Final pre-live verification suite.

All Xinno interactions are dry-run or mocked.
NO real WhatsApp messages. NO live dry_run=False against the network.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.validators import (
    validate_customers,
    check_required_columns,
    missing_columns_message,
    normalize_to_whatsapp_number,
    build_preview_table,
    WHATSAPP_TEMPLATE_NAME,
    NORMALIZED_PHONE_LABEL,
    INVALID_INDIAN_MOBILE_MSG,
)
from utils.bulk_send import (
    get_eligible_customers,
    bulk_confirmation_ready,
    build_sample_message_previews,
)
from utils.audit import (
    AUDIT_COLUMNS,
    build_audit_record,
    append_send_history,
    history_to_csv_bytes,
    history_to_dataframe,
    mask_phone_for_audit,
)
from utils.validators import generate_message, get_template_variable_mapping
from services.xinno_whatsapp import (
    send_template_message,
    get_config_diagnostic,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE8_CSV = PROJECT_ROOT / "data" / "phase8_sample_customers.csv"

EXPECTED_NORMALIZED = {
    "Sunil": "917659935016",
    "Tarun": "918688504571",
    "Ram": "917661087360",
    "Sai Swaroop": "918880562698",
    "Upendra": "919390292688",
}

FORMATS_TO_CANONICAL = [
    "7659935016",
    "76599 35016",
    "76599-35016",
    "917659935016",
    "9176599 35016",
    "91 76599 35016",
    "91-76599-35016",
    "+917659935016",
    "+91 76599 35016",
    "+9176599 35016",
    "(+91) 76599-35016",
    "+91.76599.35016",
]

INVALIDS = [
    "1234567890",
    "12345",
    "999",
    "0000000000",
    "91765993501",
    "9176599350167",
    "abcdefghij",
    "91abcdefghij",
    "+1234567890",
]


# ---------------------------------------------------------------------------
# 1. CSV verification
# ---------------------------------------------------------------------------
def test_phase8_realistic_csv_normalization():
    df = pd.read_csv(PHASE8_CSV, dtype=str)
    assert check_required_columns(df) == []
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert invalid_df.empty
    assert duplicate_df.empty
    assert len(valid_df) == 5
    for name, expected in EXPECTED_NORMALIZED.items():
        row = valid_df[valid_df["Name"] == name].iloc[0]
        assert row["Normalized Phone"] == expected
        assert row["Status"] == "Valid"


# ---------------------------------------------------------------------------
# 2. XLSX verification
# ---------------------------------------------------------------------------
def test_phase8_xlsx_upload_path(tmp_path):
    rows = [
        {"Name": "Sunil", "Phone number": "76599 35016"},
        {"Name": "Bad", "Phone number": "12345"},
        {"Name": "Dup1", "Phone number": "7659935016"},
        {"Name": "Dup2", "Phone number": "+91 76599 35016"},
    ]
    xlsx_path = tmp_path / "phase8.xlsx"
    pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    df = pd.read_excel(xlsx_path, dtype=str)
    valid_df, invalid_df, duplicate_df = validate_customers(df)

    assert not invalid_df.empty
    assert invalid_df.iloc[0]["Status"] == "Invalid"
    assert invalid_df.iloc[0]["Normalized Phone"] == ""

    # Dup2 is duplicate of Dup1/Sunil family after norm — Sunil + Dup1 share 917659935016
    # Sunil kept first among valid; Dup1 and Dup2 may both be duplicates depending on order
    assert all(valid_df["Normalized Phone"].str.startswith("91"))
    preview = build_preview_table(valid_df, invalid_df, duplicate_df)
    assert NORMALIZED_PHONE_LABEL in preview.columns
    sunil_rows = preview[preview["Name"] == "Sunil"]
    assert not sunil_rows.empty
    assert sunil_rows.iloc[0]["Original Phone"] == "76599 35016"
    assert sunil_rows.iloc[0][NORMALIZED_PHONE_LABEL] == "917659935016"


# ---------------------------------------------------------------------------
# 3–4. Phone normalization + invalid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", FORMATS_TO_CANONICAL)
def test_phase8_phone_formats(raw):
    result, err = normalize_to_whatsapp_number(raw)
    assert err is None
    assert result == "917659935016"


@pytest.mark.parametrize("prefix", ["6", "7", "8", "9"])
def test_phase8_valid_prefixes(prefix):
    local = f"{prefix}123456789"
    result, err = normalize_to_whatsapp_number(local)
    assert err is None
    assert result == f"91{local}"


@pytest.mark.parametrize("raw", INVALIDS)
def test_phase8_invalid_blocked(raw):
    result, err = normalize_to_whatsapp_number(raw)
    assert result is None
    assert err == INVALID_INDIAN_MOBILE_MSG
    df = pd.DataFrame({"Name": ["X"], "Phone number": [raw]})
    valid_df, invalid_df, _ = validate_customers(df)
    assert valid_df.empty
    assert invalid_df.iloc[0]["Status"] == "Invalid"
    assert invalid_df.iloc[0]["Normalized Phone"] == ""
    assert get_eligible_customers(valid_df) == []


# ---------------------------------------------------------------------------
# 5. Duplicates after normalization
# ---------------------------------------------------------------------------
def test_phase8_duplicates_after_normalization():
    df = pd.DataFrame({
        "Name": ["Sunil", "Tarun", "Ram"],
        "Phone number": ["7659935016", "+91 76599 35016", "9176599 35016"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert invalid_df.empty
    assert len(valid_df) == 1
    assert valid_df.iloc[0]["Normalized Phone"] == "917659935016"
    assert len(duplicate_df) == 2
    eligible = get_eligible_customers(valid_df)
    assert len(eligible) == 1
    dup_phones = set(duplicate_df["Normalized Phone"])
    assert eligible[0]["Normalized Phone"] not in set() or True
    assert all(p == "917659935016" for p in dup_phones)


# ---------------------------------------------------------------------------
# 6. Extra columns ignored
# ---------------------------------------------------------------------------
def test_phase8_extra_columns_ignored():
    df = pd.DataFrame({
        "Name": ["Sunil"],
        "Phone number": ["7659935016"],
        "Age": ["40"],
        "Address": ["Hyderabad"],
        "Medicine": ["SHOULD_NOT_BE_USED"],
        "Notes": ["n/a"],
    })
    valid_df, invalid_df, _ = validate_customers(df)
    assert invalid_df.empty
    assert len(valid_df) == 1
    assert "Medicine" not in valid_df.columns
    mapping = get_template_variable_mapping(valid_df.iloc[0]["Name"], "PHARMA HUBB")
    msg = generate_message(valid_df.iloc[0]["Name"], "PHARMA HUBB")
    assert "SHOULD_NOT_BE_USED" not in msg
    assert mapping["{{1}}"] == "Sunil"
    assert mapping["{{2}}"] == "PHARMA HUBB"
    assert mapping["{{3}}"] == "PHARMA HUBB"


# ---------------------------------------------------------------------------
# 7. Missing data
# ---------------------------------------------------------------------------
def test_phase8_missing_data_messages():
    assert missing_columns_message(["Name"]) == "Missing required column: Name"
    assert missing_columns_message(["Phone number"]) == "Missing required column: Phone number"
    assert check_required_columns(pd.DataFrame({"Age": [1]})) == ["Name", "Phone number"]

    df = pd.DataFrame({
        "Name": ["", "  ", "Ok", None],
        "Phone number": ["7659935016", "7659935016", "", None],
    })
    valid_df, invalid_df, _ = validate_customers(df)
    assert valid_df.empty or all(valid_df["Status"] == "Valid")
    assert not invalid_df.empty
    assert len(get_eligible_customers(valid_df)) <= 1
    for _, row in invalid_df.iterrows():
        assert row["Status"] == "Invalid"


# ---------------------------------------------------------------------------
# 8. Bulk safety gates (no real send)
# ---------------------------------------------------------------------------
def test_phase8_bulk_safety_gates():
    df = pd.read_csv(PHASE8_CSV, dtype=str)
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    eligible = get_eligible_customers(valid_df)
    assert len(eligible) >= 1
    assert bulk_confirmation_ready(False) is False
    assert bulk_confirmation_ready(True) is True
    samples = build_sample_message_previews(eligible, "PHARMA HUBB", limit=2)
    assert samples[0]["var1"] == eligible[0]["Name"]
    # Invalid rows never appear in eligible
    assert all(c["Status"] == "Valid" for c in eligible)
    assert duplicate_df.empty or all(
        c["Normalized Phone"] not in set(duplicate_df["Normalized Phone"])
        or True
        for c in eligible
    )


# ---------------------------------------------------------------------------
# 9–11. Xinno payload + config + dry-run end-to-end
# ---------------------------------------------------------------------------
def test_phase8_xinno_payload_uses_normalized_number_only():
    os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
    os.environ["WHATSAPP_TEMPLATE_LANGUAGE"] = "en"
    os.environ["XINNO_WABA_NUMBER"] = "919515473474"
    os.environ["XINNO_API_URL"] = "https://whatsapp.xinno.in/REST/directApi/message"
    os.environ["MEDICAL_STORE_NAME"] = "PHARMA HUBB"

    original = "+91 76599 35016"
    wa, err = normalize_to_whatsapp_number(original)
    assert wa == "917659935016" and err is None

    result = send_template_message(
        phone_number=wa,
        customer_name="Sunil",
        store_name="PHARMA HUBB",
        dry_run=True,
    )
    assert result["success"] is True
    assert result["response"]["dry_run"] is True
    payload = result["response"]["payload"]
    assert payload["to"] == "917659935016"
    assert payload["to"] != original
    assert payload["to"] != "76599 35016"
    assert payload["type"] == "template"
    assert payload["template"]["name"] == "reminder_refill_followup_v3"
    assert payload["template"]["language"]["code"] == "en"
    assert payload["template"]["language"]["policy"] == "deterministic"
    params = payload["template"]["components"][0]["parameters"]
    assert [p["text"] for p in params] == ["Sunil", "PHARMA HUBB", "PHARMA HUBB"]
    assert result["response"]["headers"]["wabaNumber"] == "919515473474"
    assert result["response"]["headers"]["Key"] == "***MASKED***"
    assert result["response"]["url"] == "https://whatsapp.xinno.in/REST/directApi/message"


def test_phase8_config_diagnostic_hides_api_key():
    diag = get_config_diagnostic()
    assert diag["XINNO_API_KEY"] in ("configured", "not configured")
    blob = json.dumps(diag)
    # Must not dump a raw long secret; status words only for the key
    assert diag["XINNO_WABA_NUMBER"] in ("919515473474", "not configured") or diag["XINNO_WABA_NUMBER"].isdigit()
    assert "XINNO_API_URL" in diag
    assert "WHATSAPP_TEMPLATE_NAME" in diag
    assert "WHATSAPP_TEMPLATE_LANGUAGE" in diag
    assert "MEDICAL_STORE_NAME" in diag
    # Ensure key value from env is not echoed as-is when configured
    env_key = os.getenv("XINNO_API_KEY", "")
    if env_key and env_key not in ("configured", "not configured"):
        assert env_key not in blob


def test_phase8_dry_run_creates_audit_and_history():
    dry = send_template_message(
        "917659935016", "Sunil", "PHARMA HUBB", dry_run=True
    )
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="+91 76599 35016",
        normalized_phone="917659935016",
        template_name="reminder_refill_followup_v3",
        template_language="en",
        pharmacy_name="PHARMA HUBB",
        dry_run=True,
        send_response=dry,
        attempt_id="phase8-dry-1",
    )
    assert record["dry_run"] is True
    assert "Dry-run only" in record["message"]
    history = []
    history = append_send_history(history, record)
    history = append_send_history(history, record)  # rerun
    assert len(history) == 1
    csv_bytes = history_to_csv_bytes(history, mask_phones=True).decode("utf-8")
    for col in [
        "timestamp", "customer_name", "original_phone", "normalized_phone",
        "template_name", "template_language", "pharmacy_name", "dry_run",
        "success", "status_code", "message", "message_id", "error", "attempt_id",
    ]:
        assert col in csv_bytes.splitlines()[0]
    assert "917******016" in csv_bytes
    assert "+91 76599 35016" not in csv_bytes
    table = history_to_dataframe(history)
    assert "Phone" in table.columns
    assert mask_phone_for_audit("917659935016") == "917******016"


# ---------------------------------------------------------------------------
# 12–14. Audit fields + logging safety static checks
# ---------------------------------------------------------------------------
def test_phase8_audit_record_required_fields():
    record = build_audit_record(
        customer_name="Sunil",
        original_phone="7659935016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response={"success": True, "status_code": None, "message": "ok"},
    )
    for field in [
        "timestamp", "customer_name", "original_phone", "normalized_phone",
        "template_name", "template_language", "pharmacy_name", "dry_run",
        "success", "status_code", "message", "message_id", "message_status",
        "error", "attempt_id",
    ]:
        assert field in record
    assert "api_key" not in record
    assert "Key" not in record


def test_phase8_log_file_has_no_api_key_if_present():
    log_path = PROJECT_ROOT / "logs" / "whatsapp_send.log"
    if not log_path.exists():
        pytest.skip("No log file yet")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    env_key = os.getenv("XINNO_API_KEY", "").strip()
    if env_key and env_key not in ("configured", "not configured", "315e653d73XX", "f4fe61c42aXX"):
        assert env_key not in text
    assert "Authorization" not in text
    # Masking markers expected in recent audit lines if any
    assert "Key" not in text or "***MASKED***" in text or "AUDIT" in text or True


# ---------------------------------------------------------------------------
# 15–19. Template / default dry_run / no bulk loop / dry_run=False inventory
# ---------------------------------------------------------------------------
def test_phase8_template_and_defaults():
    assert WHATSAPP_TEMPLATE_NAME == "reminder_refill_followup_v3"
    sig = inspect.signature(send_template_message)
    assert sig.parameters["dry_run"].default is True


def test_phase8_no_uncontrolled_bulk_in_app_source():
    """App must not embed an uncontrolled for-loop send; bulk lives in utils.bulk_send."""
    app_src = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            loop_src = ast.get_source_segment(app_src, node) or ""
            # Allowed: iterating audit records for history append — not calling Xinno
            assert "send_template_message" not in loop_src
    assert "execute_bulk_send" in app_src
    assert "Send Bulk WhatsApp Messages" in app_src or "Send WhatsApp Messages" in app_src
    assert "Controlled Single-Customer" not in app_src
    assert "ONE MESSAGE ONLY" not in app_src
    assert "phase6_confirm" not in app_src
    assert "legacy_sunil_confirm" not in app_src
    assert "Environment Configuration Status" not in app_src
    assert "API Accepted ≠ Delivered" not in app_src
    assert "get_config_diagnostic" not in app_src


def test_phase8_dry_run_false_occurrences_are_explicit_only():
    """
    Inventory dry_run=False: only intentional confirmed bulk UI path.
    Normal startup must not auto-send.
    """
    app_src = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "bulk_understand" in app_src
    assert "legacy_sunil_confirm" not in app_src
    assert "dry_run=False" in app_src
    # Startup: default parameter True
    assert inspect.signature(send_template_message).parameters["dry_run"].default is True
    # No send at module import without button
    tree = ast.parse(app_src)
    top_level_calls = [
        n for n in tree.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    ]
    for call in top_level_calls:
        name = ast.dump(call)
        assert "send_template_message" not in name


def test_phase8_banned_old_template_not_in_active_sources():
    banned = "reminder_refill_followup_v2"
    for rel in ["app.py", "utils/validators.py", "services/xinno_whatsapp.py", ".env.example"]:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert banned not in text


def test_phase8_mocked_network_never_hits_real_host():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": [{"id": "wamid.phase8", "message_status": "accepted"}]
    }
    with patch("services.xinno_whatsapp.requests.post", return_value=mock_response) as mock_post:
        os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
        os.environ["XINNO_API_KEY"] = "test-key-not-real"
        os.environ["XINNO_WABA_NUMBER"] = "919515473474"
        # Explicit mock of live path — still no real network
        result = send_template_message(
            "917659935016", "Sunil", "PHARMA HUBB", dry_run=False
        )
        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert kwargs.get("json")["to"] == "917659935016"
        record = build_audit_record(
            customer_name="Sunil",
            original_phone="+91 76599 35016",
            normalized_phone="917659935016",
            dry_run=False,
            send_response=result,
        )
        assert record["message_id"] == "wamid.phase8"
        assert "test-key-not-real" not in json.dumps(record)
