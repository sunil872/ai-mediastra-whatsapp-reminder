"""
Bulk WhatsApp send tests — all Xinno calls mocked / dry-run.

NO real WhatsApp messages. Production dry_run=False path is exercised only via mocks.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.validators import validate_customers, generate_message, WHATSAPP_TEMPLATE_NAME
from utils.bulk_send import (
    get_eligible_customers,
    build_bulk_summary,
    build_sample_message_previews,
    bulk_confirmation_ready,
    execute_bulk_send,
    bulk_results_table,
    new_bulk_attempt_id,
)
from utils.audit import append_send_history, history_to_csv_bytes, mask_phone_for_audit
from services.xinno_whatsapp import send_template_message


def _valid_df_multi():
    df = pd.DataFrame({
        "Name": ["Upendra", "Tarun", "Ram", "Bad", "DupA", "DupB"],
        "Phone number": [
            "9390292688",
            "86885 04571",
            "+91 76610 87360",
            "12345",
            "7659935016",
            "+91 76599 35016",
        ],
    })
    return validate_customers(df)


def test_multiple_valid_customers_eligible():
    valid_df, invalid_df, duplicate_df = _valid_df_multi()
    eligible = get_eligible_customers(valid_df)
    names = {c["Name"] for c in eligible}
    assert "Upendra" in names
    assert "Tarun" in names
    assert "Ram" in names
    assert "Bad" not in names
    assert "DupB" not in names  # duplicate excluded from valid_df


def test_invalid_customers_skipped():
    valid_df, invalid_df, _ = _valid_df_multi()
    assert not invalid_df.empty
    eligible = get_eligible_customers(valid_df)
    phones = {c["Normalized Phone"] for c in eligible}
    assert all(p.startswith("91") and len(p) == 12 for p in phones)


def test_duplicate_normalized_phones_skipped():
    valid_df, _, duplicate_df = _valid_df_multi()
    assert not duplicate_df.empty
    eligible = get_eligible_customers(valid_df)
    # Only one of DupA/DupB family kept in valid_df
    assert len([c for c in eligible if c["Normalized Phone"] == "917659935016"]) <= 1


def test_dynamic_name_mapping_not_sunil():
    samples = build_sample_message_previews(
        [
            {
                "Name": "Upendra",
                "Normalized Phone": "919390292688",
                "Original Phone": "9390292688",
                "Status": "Valid",
            }
        ],
        "PHARMA HUBB",
    )
    assert samples[0]["var1"] == "Upendra"
    assert samples[0]["var2"] == "PHARMA HUBB"
    assert samples[0]["var3"] == "PHARMA HUBB"
    assert samples[0]["message_preview"].startswith("Dear Upendra,")
    assert "Dear Sunil," not in samples[0]["message_preview"]


def test_independent_payloads_per_customer():
    """Customer A's name never gets Customer B's phone."""
    calls = []

    def fake_send(phone_number, customer_name, store_name, dry_run):
        calls.append({
            "phone": phone_number,
            "name": customer_name,
            "store": store_name,
            "dry_run": dry_run,
        })
        # Also exercise real dry-run payload builder for identity check
        real = send_template_message(
            phone_number, customer_name, store_name, dry_run=True
        )
        payload = real["response"]["payload"]
        assert payload["to"] == phone_number
        assert payload["template"]["components"][0]["parameters"][0]["text"] == customer_name
        assert payload["template"]["components"][0]["parameters"][1]["text"] == "PHARMA HUBB"
        assert payload["template"]["components"][0]["parameters"][2]["text"] == "PHARMA HUBB"
        assert payload["template"]["name"] == "reminder_refill_followup_v3"
        return real

    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "9390292688", "Status": "Valid"},
        {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "8688504571", "Status": "Valid"},
        {"Name": "Ram", "Normalized Phone": "917661087360", "Original Phone": "7661087360", "Status": "Valid"},
    ]
    result = execute_bulk_send(
        eligible,
        store_name="PHARMA HUBB",
        send_fn=fake_send,
        dry_run=True,
        bulk_attempt_id="bulk-test-1",
    )
    assert result["attempted"] == 3
    assert result["successful"] == 3
    assert calls[0]["name"] == "Upendra" and calls[0]["phone"] == "919390292688"
    assert calls[1]["name"] == "Tarun" and calls[1]["phone"] == "918688504571"
    assert calls[2]["name"] == "Ram" and calls[2]["phone"] == "917661087360"
    # No cross-wiring
    assert not (calls[0]["name"] == "Upendra" and calls[0]["phone"] == "918688504571")


def test_one_failure_does_not_stop_remaining():
    def flaky_send(phone_number, customer_name, store_name, dry_run):
        if customer_name == "Tarun":
            return {
                "success": False,
                "status_code": 400,
                "message": "failed for Tarun",
                "response": {"api_response": {"error": {"message": "fail"}}},
            }
        return send_template_message(phone_number, customer_name, store_name, dry_run=True)

    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "x", "Status": "Valid"},
        {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "y", "Status": "Valid"},
        {"Name": "Ram", "Normalized Phone": "917661087360", "Original Phone": "z", "Status": "Valid"},
    ]
    result = execute_bulk_send(
        eligible, store_name="PHARMA HUBB", send_fn=flaky_send, dry_run=True
    )
    assert result["attempted"] == 3
    assert result["successful"] == 2
    assert result["failed"] == 1
    failed = [r for r in result["records"] if not r["success"]]
    assert failed[0]["customer_name"] == "Tarun"
    assert failed[0]["error"]


def test_no_automatic_retry_on_failure():
    calls = {"n": 0}

    def fail_once(phone_number, customer_name, store_name, dry_run):
        calls["n"] += 1
        return {
            "success": False,
            "status_code": 500,
            "message": "server error",
            "response": {},
        }

    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "x", "Status": "Valid"},
    ]
    execute_bulk_send(eligible, store_name="PHARMA HUBB", send_fn=fail_once, dry_run=True)
    assert calls["n"] == 1  # no retry


def test_empty_valid_list_does_not_call_xinno():
    mock_fn = MagicMock()
    result = execute_bulk_send([], store_name="PHARMA HUBB", send_fn=mock_fn, dry_run=False)
    mock_fn.assert_not_called()
    assert result["attempted"] == 0
    assert "No valid customers" in result["message"]


def test_confirmation_required():
    assert bulk_confirmation_ready(False) is False
    assert bulk_confirmation_ready(True) is True


def test_bulk_rerun_protection_same_attempt_id():
    """Same bulk_attempt_id records are distinct per customer but batch id is stable."""
    history = []
    bulk_id = new_bulk_attempt_id()

    def ok_send(phone_number, customer_name, store_name, dry_run):
        return send_template_message(phone_number, customer_name, store_name, dry_run=True)

    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "x", "Status": "Valid"},
        {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "y", "Status": "Valid"},
    ]
    result = execute_bulk_send(
        eligible, store_name="PHARMA HUBB", send_fn=ok_send, dry_run=True, bulk_attempt_id=bulk_id
    )
    for rec in result["records"]:
        history = append_send_history(history, rec)
        history = append_send_history(history, rec)  # simulate rerun
    assert len(history) == 2
    assert all(r["bulk_attempt_id"] == bulk_id for r in history)


def test_audit_every_customer_and_no_api_key():
    def ok_send(phone_number, customer_name, store_name, dry_run):
        return send_template_message(phone_number, customer_name, store_name, dry_run=True)

    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "9390292688", "Status": "Valid"},
        {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "8688504571", "Status": "Valid"},
    ]
    result = execute_bulk_send(
        eligible, store_name="PHARMA HUBB", send_fn=ok_send, dry_run=True
    )
    assert len(result["records"]) == 2
    blob = json.dumps(result)
    assert "XINNO_API_KEY" not in blob
    assert "api_key" not in blob
    csv_bytes = history_to_csv_bytes(result["records"], mask_phones=True).decode("utf-8")
    assert "9390292688" not in csv_bytes
    assert mask_phone_for_audit("919390292688") in csv_bytes
    table = bulk_results_table(result["records"], mask_phones=True)
    assert list(table.columns)[:5] == ["Customer", "Phone", "Status", "Message ID", "Error"]


def test_send_default_still_dry_run_true():
    sig = inspect.signature(send_template_message)
    assert sig.parameters["dry_run"].default is True


def test_single_preview_uses_selected_customer_name():
    samples = build_sample_message_previews(
        [
            {
                "Name": "Upendra",
                "Original Phone": "9390292688",
                "Normalized Phone": "919390292688",
                "Status": "Valid",
            }
        ],
        "PHARMA HUBB",
    )
    assert samples[0]["customer_name"] == "Upendra"
    assert samples[0]["message_preview"].startswith("Dear Upendra,")
    assert "Dear Sunil," not in samples[0]["message_preview"]
    assert generate_message("Tarun", "PHARMA HUBB").startswith("Dear Tarun,")


def test_mocked_production_path_dry_run_false_no_network():
    """dry_run=False only through mocked requests — no real Xinno traffic."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": [{"id": "wamid.bulk", "message_status": "accepted"}]
    }
    with patch("services.xinno_whatsapp.requests.post", return_value=mock_response) as mock_post:
        import os
        os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
        os.environ["XINNO_API_KEY"] = "test-key-not-real"
        os.environ["XINNO_WABA_NUMBER"] = "919515473474"

        def live_send(phone_number, customer_name, store_name, dry_run):
            assert dry_run is False
            return send_template_message(
                phone_number, customer_name, store_name, dry_run=False
            )

        eligible = [
            {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "x", "Status": "Valid"},
            {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "y", "Status": "Valid"},
        ]
        result = execute_bulk_send(
            eligible,
            store_name="PHARMA HUBB",
            send_fn=live_send,
            dry_run=False,
            bulk_attempt_id="bulk-live-mock",
        )
        assert mock_post.call_count == 2
        assert result["successful"] == 2
        assert result["dry_run"] is False
        first_payload = mock_post.call_args_list[0].kwargs["json"]
        assert first_payload["to"] == "919390292688"
        assert first_payload["template"]["components"][0]["parameters"][0]["text"] == "Upendra"


def test_bulk_summary_counts():
    s = build_bulk_summary(10, 7, 2, 1, 7)
    assert s["eligible_for_sending"] == 7
    assert s["template_name"] == WHATSAPP_TEMPLATE_NAME


def test_no_hardcoded_sunil_in_bulk_module():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    src = (root / "utils" / "bulk_send.py").read_text(encoding="utf-8")
    assert "Sunil" not in src
    assert "7659935016" not in src


def test_phase6_single_customer_module_removed():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert not (root / "utils" / "phase6_send.py").exists()
    assert not (root / "test_phase6_send.py").exists()
    assert not (root / "tests" / "test_phase6_send.py").exists()
    assert not (root / "test_xinno_live.py").exists()
    assert not (root / "tests" / "test_xinno_live.py").exists()
    app_src = (root / "app.py").read_text(encoding="utf-8")
    assert "Bulk WhatsApp" in app_src
    assert "Send WhatsApp Messages" in app_src
    assert "phase6_send" not in app_src
    assert "assert_single_customer_send" not in app_src
    assert "Environment Configuration Status" not in app_src
