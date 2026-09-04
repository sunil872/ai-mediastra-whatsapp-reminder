"""
Phase 5 tests: customer validation + template configuration.

NO live Xinno API calls. Uses local validation and dry-run only.
"""

import os
import json
from pathlib import Path

import pandas as pd
import pytest

from utils.validators import (
    check_required_columns,
    missing_columns_message,
    normalize_columns,
    validate_name,
    validate_phone,
    validate_customers,
    build_preview_table,
    get_template_variable_mapping,
    generate_message,
    to_whatsapp_phone,
    WHATSAPP_TEMPLATE_NAME,
)
from services.xinno_whatsapp import send_template_message, normalize_phone_number


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_customers.csv"


# ---------------------------------------------------------------------------
# 1. Valid customer CSV
# ---------------------------------------------------------------------------
def test_valid_customer_csv():
    df = pd.read_csv(SAMPLE_CSV, dtype=str)
    assert check_required_columns(df) == []
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert len(valid_df) == 5
    assert invalid_df.empty
    assert duplicate_df.empty
    assert list(valid_df["Name"]) == ["Sunil", "Tarun", "Ram", "Sai Swaroop", "Upendra"]
    assert valid_df.iloc[0]["Normalized Phone"] == "917659935016"


# ---------------------------------------------------------------------------
# 2–3. Missing columns
# ---------------------------------------------------------------------------
def test_normalize_columns_customer_and_mobile_no():
    """Accepts spreadsheet headers: Customer, MOBILE NO."""
    df = pd.DataFrame({
        "Customer": ["Upendra"],
        "MOBILE NO.": ["9390292688"],
    })
    normalized = normalize_columns(df)
    assert check_required_columns(normalized) == []
    assert "Name" in normalized.columns
    assert "Phone number" in normalized.columns
    valid_df, invalid_df, _ = validate_customers(normalized)
    assert invalid_df.empty
    assert valid_df.iloc[0]["Name"] == "Upendra"
    assert valid_df.iloc[0]["Normalized Phone"] == "919390292688"


def test_normalize_columns_case_insensitive():
    df = pd.DataFrame({
        "Customer_Name": ["Tarun"],
        "PHONE": ["8688504571"],
    })
    normalized = normalize_columns(df)
    assert check_required_columns(normalized) == []
    valid_df, _, _ = validate_customers(normalized)
    assert valid_df.iloc[0]["Name"] == "Tarun"
    assert valid_df.iloc[0]["Normalized Phone"] == "918688504571"


def test_normalize_columns_keeps_canonical_names():
    df = pd.DataFrame({
        "Name": ["Ram"],
        "Phone number": ["7661087360"],
        "Age": ["40"],
    })
    normalized = normalize_columns(df)
    assert check_required_columns(normalized) == []
    assert "Age" in normalized.columns


def test_missing_name_column():
    df = pd.DataFrame({"Phone number": ["7659935016"]})
    missing = check_required_columns(df)
    assert missing == ["Name"]
    assert missing_columns_message(missing) == "Missing required column: Name"


def test_missing_phone_column():
    df = pd.DataFrame({"Name": ["Sunil"]})
    missing = check_required_columns(df)
    assert missing == ["Phone number"]
    assert missing_columns_message(missing) == "Missing required column: Phone number"


# ---------------------------------------------------------------------------
# 4–5. Empty name / phone
# ---------------------------------------------------------------------------
def test_empty_customer_name():
    cleaned, err = validate_name("")
    assert cleaned is None
    assert "missing" in err.lower()

    df = pd.DataFrame({"Name": [""], "Phone number": ["7659935016"]})
    valid_df, invalid_df, _ = validate_customers(df)
    assert valid_df.empty
    assert len(invalid_df) == 1
    assert "Customer name is missing" in invalid_df.iloc[0]["Reason"]


def test_empty_phone_number():
    cleaned, err = validate_phone("")
    assert cleaned is None
    assert "empty" in err.lower()

    df = pd.DataFrame({"Name": ["Sunil"], "Phone number": [""]})
    valid_df, invalid_df, _ = validate_customers(df)
    assert valid_df.empty
    assert "Phone number is empty" in invalid_df.iloc[0]["Reason"]


# ---------------------------------------------------------------------------
# 6–8. Phone normalization / invalid
# ---------------------------------------------------------------------------
def test_valid_10_digit_indian_number():
    ten, err = validate_phone("7659935016")
    assert err is None
    assert ten == "7659935016"
    assert to_whatsapp_phone(ten) == "917659935016"
    assert normalize_phone_number("7659935016") == "917659935016"


def test_already_normalized_indian_number():
    ten, err = validate_phone("917659935016")
    assert err is None
    assert ten == "7659935016"
    assert to_whatsapp_phone(ten) == "917659935016"
    assert normalize_phone_number("917659935016") == "917659935016"


def test_invalid_phone_number():
    ten, err = validate_phone("12345")
    assert ten is None
    assert err is not None

    df = pd.DataFrame({"Name": ["Bad"], "Phone number": ["12345"]})
    valid_df, invalid_df, _ = validate_customers(df)
    assert valid_df.empty
    assert "Invalid Indian mobile number" in invalid_df.iloc[0]["Reason"]
    assert invalid_df.iloc[0]["Normalized Phone"] == ""
    assert invalid_df.iloc[0]["Status"] == "Invalid"


# ---------------------------------------------------------------------------
# 9. Duplicate phone numbers
# ---------------------------------------------------------------------------
def test_duplicate_phone_number():
    df = pd.DataFrame({
        "Name": ["Sunil", "Other"],
        "Phone number": ["7659935016", "917659935016"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert len(valid_df) == 1
    assert valid_df.iloc[0]["Name"] == "Sunil"
    assert len(duplicate_df) == 1
    assert duplicate_df.iloc[0]["Status"] == "Duplicate"
    assert invalid_df.empty


# ---------------------------------------------------------------------------
# 10. Blank rows
# ---------------------------------------------------------------------------
def test_blank_rows():
    df = pd.DataFrame({
        "Name": ["Sunil", "", None],
        "Phone number": ["7659935016", "", None],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert len(valid_df) == 1
    assert len(invalid_df) == 2
    assert all("Blank row" in r for r in invalid_df["Reason"].tolist())
    assert duplicate_df.empty


# ---------------------------------------------------------------------------
# 11–12. Template parameter generation / 3-variable mapping
# ---------------------------------------------------------------------------
def test_template_parameter_generation():
    mapping = get_template_variable_mapping("Sunil", "PHARMA HUBB")
    assert mapping == {
        "{{1}}": "Sunil",
        "{{2}}": "PHARMA HUBB",
        "{{3}}": "PHARMA HUBB",
    }
    msg = generate_message("Sunil", "PHARMA HUBB")
    assert "Sunil" in msg
    assert "PHARMA HUBB" in msg


def test_three_variable_mapping_dry_run():
    os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
    os.environ["MEDICAL_STORE_NAME"] = "PHARMA HUBB"
    result = send_template_message(
        phone_number="7659935016",
        customer_name="Sunil",
        store_name="PHARMA HUBB",
        dry_run=True,
    )
    assert result["success"] is True
    assert result["response"]["dry_run"] is True
    params = result["response"]["payload"]["template"]["components"][0]["parameters"]
    assert len(params) == 3
    assert params[0]["text"] == "Sunil"
    assert params[1]["text"] == "PHARMA HUBB"
    assert params[2]["text"] == "PHARMA HUBB"


# ---------------------------------------------------------------------------
# 13–14. Template name v3 only / v2 not used
# ---------------------------------------------------------------------------
def test_correct_template_name_v3():
    assert WHATSAPP_TEMPLATE_NAME == "reminder_refill_followup_v3"
    os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
    result = send_template_message("7659935016", "Sunil", "PHARMA HUBB", dry_run=True)
    assert result["response"]["payload"]["template"]["name"] == "reminder_refill_followup_v3"
    assert result["response"]["payload"]["template"]["language"]["code"] == "en"
    assert "category" not in result["response"]["payload"]["template"]


def test_old_template_name_v2_not_in_project_sources():
    """Scan active project source files for the retired template name."""
    banned = "reminder_refill_followup_v2"
    roots = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "utils" / "validators.py",
        PROJECT_ROOT / "services" / "xinno_whatsapp.py",
        PROJECT_ROOT / "tests" / "test_xinno.py",
        PROJECT_ROOT / ".env.example",
    ]
    for path in roots:
        text = path.read_text(encoding="utf-8")
        assert banned not in text, f"Found retired template name in {path}"

    # .env if present
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        assert banned not in env_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Extra: preview table + extra columns ignored + API key masked
# ---------------------------------------------------------------------------
def test_preview_table_and_extra_columns_ignored():
    df = pd.DataFrame({
        "Name": ["Sunil"],
        "Phone number": ["7659935016"],
        "Medication": ["SHOULD_BE_IGNORED"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert len(valid_df) == 1
    preview = build_preview_table(valid_df, invalid_df, duplicate_df)
    from utils.validators import NORMALIZED_PHONE_LABEL
    assert list(preview.columns) == ["Name", "Original Phone", NORMALIZED_PHONE_LABEL, "Status"]
    assert "Medication" not in preview.columns
    assert preview.iloc[0]["Status"] == "Valid"


def test_dry_run_masks_api_key_and_uses_correct_waba_env():
    os.environ["WHATSAPP_TEMPLATE_NAME"] = "reminder_refill_followup_v3"
    os.environ["XINNO_WABA_NUMBER"] = "919515473474"
    result = send_template_message("7659935016", "Sunil", "PHARMA HUBB", dry_run=True)
    headers = result["response"]["headers"]
    assert headers["Key"] == "***MASKED***"
    assert headers["wabaNumber"] == "919515473474"
    blob = json.dumps(result)
    assert "***MASKED***" in blob
    # Ensure no obvious raw key leak pattern from placeholder env in tests
    assert "XINNO_API_KEY" not in blob or "***MASKED***" in blob


def test_send_default_is_dry_run():
    import inspect
    from services.xinno_whatsapp import send_template_message as stm
    assert inspect.signature(stm).parameters["dry_run"].default is True
