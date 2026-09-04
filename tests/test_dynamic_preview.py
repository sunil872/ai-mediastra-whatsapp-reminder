"""
Dynamic personalized preview tests for BULK-ONLY messaging.

No single-customer selection. Previews use each eligible customer's own name.
All Xinno calls are mocked / dry-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.validators import generate_message, get_template_variable_mapping
from utils.bulk_send import build_sample_message_previews, get_eligible_customers
from utils.validators import validate_customers
from services.xinno_whatsapp import send_template_message
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_customers.csv"


def test_sample_previews_are_dynamic_per_customer():
    eligible = [
        {"Name": "Upendra", "Normalized Phone": "919390292688", "Original Phone": "9390292688", "Status": "Valid"},
        {"Name": "Tarun", "Normalized Phone": "918688504571", "Original Phone": "8688504571", "Status": "Valid"},
        {"Name": "Ram", "Normalized Phone": "917661087360", "Original Phone": "7661087360", "Status": "Valid"},
    ]
    samples = build_sample_message_previews(eligible, "PHARMA HUBB", limit=3)
    assert samples[0]["message_preview"].startswith("Dear Upendra,")
    assert samples[1]["message_preview"].startswith("Dear Tarun,")
    assert samples[2]["message_preview"].startswith("Dear Ram,")
    assert samples[0]["var1"] == "Upendra"
    assert samples[1]["var1"] == "Tarun"
    assert samples[2]["var1"] == "Ram"
    for s in samples:
        assert s["var2"] == "PHARMA HUBB"
        assert s["var3"] == "PHARMA HUBB"
        assert "Dear Sunil," not in s["message_preview"] or s["customer_name"] == "Sunil"


def test_generate_message_approved_template_wording():
    msg = generate_message("Upendra", "PHARMA HUBB")
    assert msg.startswith("Dear Upendra,\n\n")
    assert "PHARMA HUBB" in msg
    assert "Dear Sunil," not in msg


def test_send_payload_uses_each_customer_independently():
    for name, phone in [
        ("Upendra", "919390292688"),
        ("Tarun", "918688504571"),
    ]:
        res = send_template_message(phone, name, "PHARMA HUBB", dry_run=True)
        payload = res["response"]["payload"]
        params = payload["template"]["components"][0]["parameters"]
        assert payload["to"] == phone
        assert params[0]["text"] == name
        assert params[1]["text"] == "PHARMA HUBB"
        assert params[2]["text"] == "PHARMA HUBB"


@pytest.fixture
def loaded_app():
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at.run()
    at.file_uploader[0].upload("sample_customers.csv", SAMPLE_CSV.read_bytes())
    at.run()
    return at


def test_app_is_bulk_only_no_customer_selectbox(loaded_app):
    at = loaded_app
    # No single-customer selectbox
    assert len(at.selectbox) == 0
    # Bulk samples present
    bulk_samples = [
        t for t in at.text_area
        if getattr(t, "key", None) and str(t.key).startswith("bulk_sample_")
    ]
    assert len(bulk_samples) >= 1
    # Button exists but disabled until confirmation
    btn = [b for b in at.button if getattr(b, "key", None) == "bulk_send_btn"]
    assert btn
    assert btn[0].disabled is True
    # No obsolete single-customer language
    page_text = " ".join(str(x) for x in at.main)
    assert "Controlled Single-Customer" not in page_text
    assert "ONE MESSAGE ONLY" not in page_text
    assert "Send ONE WhatsApp" not in page_text


def test_app_bulk_previews_use_uploaded_names(loaded_app):
    at = loaded_app
    values = [t.value for t in at.text_area if getattr(t, "key", None) and "bulk_sample_" in str(t.key)]
    joined = "\n".join(values)
    # Sample CSV includes multiple customers — at least one dynamic Dear line
    assert "Dear " in joined
    # Confirm Upendra appears if in sample file
    df = pd.read_csv(SAMPLE_CSV, dtype=str)
    valid, _, _ = validate_customers(df)
    eligible = get_eligible_customers(valid)
    if any(c["Name"] == "Upendra" for c in eligible[:3]):
        assert "Dear Upendra," in joined


def test_app_confirmation_enables_bulk_button(loaded_app):
    at = loaded_app
    cb = [c for c in at.checkbox if getattr(c, "key", None) == "bulk_understand"]
    assert cb
    cb[0].check()
    at.run()
    btn = [b for b in at.button if getattr(b, "key", None) == "bulk_send_btn"][0]
    assert btn.disabled is False


def test_mapping_helpers_dynamic():
    m = get_template_variable_mapping("Upendra", "PHARMA HUBB")
    assert m == {"{{1}}": "Upendra", "{{2}}": "PHARMA HUBB", "{{3}}": "PHARMA HUBB"}
