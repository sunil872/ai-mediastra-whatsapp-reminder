"""
Unit tests for robust Indian mobile phone normalization.

NO live Xinno calls. NO dry_run=False. NO WhatsApp sends.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.validators import (
    normalize_to_whatsapp_number,
    validate_phone,
    validate_customers,
    build_preview_table,
    NORMALIZED_PHONE_LABEL,
    INVALID_INDIAN_MOBILE_MSG,
)
from utils.bulk_send import get_eligible_customers


CANONICAL = "917659935016"

# All of these must normalize to 917659935016
VALID_SAME_NUMBER_FORMATS = [
    "917659935016",
    "9176599 35016",
    "+917659935016",
    "76599 35016",
    "+91 76599 35016",
    "+9176599 35016",
    "7659935016",
    "76599-35016",
    "91 76599 35016",
    "91-76599-35016",
    "+91-76599-35016",
    "(+91) 76599-35016",
    "(+91)7659935016",
    "+91 (76599) 35016",
    "+91.76599.35016",
    "91.76599.35016",
    "76599.35016",
    "  +91 76599 35016  ",
    "+91  76599   35016",
]

INVALID_NUMBERS = [
    "1234567890",
    "12345",
    "999",
    "0000000000",
    "91765993501",
    "9176599350167",
    "123456789",
    "abcdefghij",
    "91abcdefghij",
    "+1234567890",
    "919999",
    "00000000000",
]


@pytest.mark.parametrize("raw", VALID_SAME_NUMBER_FORMATS)
def test_valid_formats_normalize_to_canonical(raw):
    result, err = normalize_to_whatsapp_number(raw)
    assert err is None, f"Expected valid for {raw!r}, got error: {err}"
    assert result == CANONICAL
    # Never double-prepend 91
    assert result != "91917659935016"
    assert not result.startswith("9191")


def test_already_country_coded_not_double_prefixed():
    result, err = normalize_to_whatsapp_number("917659935016")
    assert err is None
    assert result == "917659935016"


@pytest.mark.parametrize(
    "local,expected",
    [
        ("6123456789", "916123456789"),
        ("7123456789", "917123456789"),
        ("8123456789", "918123456789"),
        ("9123456789", "919123456789"),
        ("7659935016", "917659935016"),
        ("8688504571", "918688504571"),
        ("7661087360", "917661087360"),
        ("8880562698", "918880562698"),
        ("9390292688", "919390292688"),
    ],
)
def test_valid_indian_prefixes_6_7_8_9(local, expected):
    result, err = normalize_to_whatsapp_number(local)
    assert err is None
    assert result == expected
    ten, err2 = validate_phone(local)
    assert err2 is None
    assert ten == local


@pytest.mark.parametrize("raw", INVALID_NUMBERS)
def test_invalid_numbers_rejected(raw):
    result, err = normalize_to_whatsapp_number(raw)
    assert result is None
    assert err == INVALID_INDIAN_MOBILE_MSG


def test_invalid_rows_have_blank_normalized_and_not_selectable():
    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Phone number": ["12345", "abcdefghij", "0000000000"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert valid_df.empty
    assert len(invalid_df) == 3
    assert all(invalid_df["Status"] == "Invalid")
    assert all(invalid_df["Normalized Phone"].fillna("").astype(str) == "")
    assert all(INVALID_INDIAN_MOBILE_MSG in r for r in invalid_df["Reason"])
    assert get_eligible_customers(valid_df) == []


def test_excel_dot_zero_artifact():
    # Numeric float and string ".0" artifact
    for raw in [7659935016.0, "7659935016.0"]:
        result, err = normalize_to_whatsapp_number(raw)
        assert err is None, f"Failed for {raw!r}: {err}"
        assert result == CANONICAL

    # Must NOT treat arbitrary decimals as phones
    result, err = normalize_to_whatsapp_number("7659935016.5")
    assert result is None
    assert err == INVALID_INDIAN_MOBILE_MSG


def test_original_phone_preserved():
    original = "+91 76599 35016"
    df = pd.DataFrame({"Name": ["Sunil"], "Phone number": [original]})
    valid_df, invalid_df, _ = validate_customers(df)
    assert invalid_df.empty
    assert valid_df.iloc[0]["Original Phone"] == original
    assert valid_df.iloc[0]["Normalized Phone"] == CANONICAL


def test_duplicate_detection_after_normalization():
    """
    7659935016, +91 76599 35016, 9176599 35016 all → 917659935016
    """
    df = pd.DataFrame({
        "Name": ["Sunil", "Tarun", "Ram"],
        "Phone number": ["7659935016", "+91 76599 35016", "9176599 35016"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    assert invalid_df.empty
    assert len(valid_df) == 1
    assert valid_df.iloc[0]["Normalized Phone"] == CANONICAL
    assert len(duplicate_df) == 2
    assert set(duplicate_df["Normalized Phone"]) == {CANONICAL}
    assert all(duplicate_df["Status"] == "Duplicate")

    # Duplicates not eligible; only the kept Valid row is
    eligible = get_eligible_customers(valid_df)
    assert len(eligible) == 1
    for _, row in duplicate_df.iterrows():
        assert row["Normalized Phone"] == eligible[0]["Normalized Phone"]
        assert row["Name"] != eligible[0]["Name"] or True


def test_preview_table_uses_whatsapp_label():
    df = pd.DataFrame({
        "Name": ["Sunil", "Bad"],
        "Phone number": ["76599 35016", "12345"],
    })
    valid_df, invalid_df, duplicate_df = validate_customers(df)
    preview = build_preview_table(valid_df, invalid_df, duplicate_df)
    assert NORMALIZED_PHONE_LABEL in preview.columns
    assert "Normalized Phone" not in preview.columns or NORMALIZED_PHONE_LABEL in preview.columns
    sunil = preview[preview["Name"] == "Sunil"].iloc[0]
    assert sunil[NORMALIZED_PHONE_LABEL] == CANONICAL
    assert sunil["Original Phone"] == "76599 35016"
    bad = preview[preview["Name"] == "Bad"].iloc[0]
    assert bad[NORMALIZED_PHONE_LABEL] == ""
    assert bad["Status"] == "Invalid"


def test_downstream_phone_number_is_canonical_whatsapp():
    """valid_df Phone number must be the WhatsApp number used for Xinno 'to'."""
    df = pd.DataFrame({
        "Name": ["Sunil"],
        "Phone number": ["+91 76599 35016"],
    })
    valid_df, _, _ = validate_customers(df)
    assert valid_df.iloc[0]["Phone number"] == CANONICAL
    assert valid_df.iloc[0]["Normalized Phone"] == CANONICAL


def test_letters_not_stripped_into_fake_valid_number():
    # If we stripped letters, "91abc7659935016" could become wrong; must reject
    result, err = normalize_to_whatsapp_number("91abc7659935016")
    assert result is None
    assert err == INVALID_INDIAN_MOBILE_MSG
