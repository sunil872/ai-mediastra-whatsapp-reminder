from __future__ import annotations

"""
Validation utilities for customer data.

Handles:
- Column presence checks
- Name validation and cleaning
- Robust Indian mobile phone normalization → WhatsApp format 91XXXXXXXXXX
- Duplicate detection AFTER normalization
- Local template preview helpers (no API calls)
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Required columns (canonical names after alias normalization)
REQUIRED_COLUMNS = ["Name", "Phone number"]

# Accepted upload header aliases → canonical REQUIRED_COLUMNS names
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Name": [
        "name",
        "customer",
        "customer_name",
        "customer name",
        "full_name",
        "full name",
        "fullname",
        "patient_name",
        "patient name",
    ],
    "Phone number": [
        "phone number",
        "phone",
        "phone_number",
        "mobile",
        "mobile number",
        "mobile_number",
        "mobile no",
        "mobile no.",
        "contact",
        "contact number",
        "whatsapp",
        "whatsapp number",
        "whatsapp_number",
    ],
}

# Confirmed working Xinno template (Phase 4+)
WHATSAPP_TEMPLATE_NAME = "reminder_refill_followup_v3"

# Canonical WhatsApp country code for India
INDIA_COUNTRY_CODE = "91"

# Display label for UI / preview tables
NORMALIZED_PHONE_LABEL = "Normalized Phone (WhatsApp number)"

# Harmless formatting characters to strip (NOT letters — letters remain and fail validation)
_FORMATTING_PATTERN = re.compile(r"[\s\+\-\(\)\.]")

# Message template with placeholders (local preview only)
MESSAGE_TEMPLATE = (
    "Dear {{customer_name}},\n\n"
    "You are a regular customer of {{store_name}}, and we noticed you haven't refilled your medication in recent months.\n\n"
    "If there is any issue, please contact us at 9581473474\n\n"
    "For any complaints or suggestions, contact our manager at 9885473474\n\n"
    "Thank you for being with us.\n\n"
    "Team\n"
    "{{store_name}}\n"
    "Chadargatt"
)

INVALID_INDIAN_MOBILE_MSG = "Invalid Indian mobile number"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common header aliases to canonical columns: Name, Phone number.

    Matching is case-insensitive and strips surrounding whitespace.
    Extra columns are left unchanged. First matching alias wins per canonical name.
    """
    if df is None or len(df.columns) == 0:
        return df

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    rename_map: Dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        # Prefer exact canonical header if already present
        if canonical.lower() in lower_cols:
            original = lower_cols[canonical.lower()]
            if original != canonical:
                rename_map[original] = canonical
            continue
        for alias in aliases:
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = canonical
                break

    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def check_required_columns(df: pd.DataFrame) -> List[str]:
    """
    Check if the DataFrame contains all required columns.

    Returns a list of missing column names (empty if all present).
    """
    missing = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    return missing


def missing_columns_message(missing: List[str]) -> str:
    """User-friendly message for missing required columns."""
    if not missing:
        return ""
    if len(missing) == 1:
        return f"Missing required column: {missing[0]}"
    return f"Missing required columns: {', '.join(missing)}"


def validate_name(name) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate and clean a customer name.

    Returns:
        (cleaned_name, error_reason)
    """
    if pd.isna(name) or str(name).strip() == "":
        return None, "Customer name is missing"

    cleaned = str(name).strip()
    return cleaned, None


def _coerce_phone_input_to_string(phone) -> Optional[str]:
    """
    Convert a phone cell value to a string for normalization.

    Handles Excel numeric artifacts such as 7659935016.0 / "7659935016.0"
    ONLY when the value is clearly an integer phone with a trailing .0.
    Does NOT strip decimals from arbitrary values.
    """
    if phone is None or (isinstance(phone, float) and pd.isna(phone)):
        return None
    if isinstance(phone, bool):
        # Avoid treating True/False as 1/0
        return str(phone).strip()

    if isinstance(phone, int):
        return str(phone)

    if isinstance(phone, float):
        if phone.is_integer() and phone >= 0:
            return str(int(phone))
        return str(phone).strip()

    text = str(phone).strip()
    if text == "" or text.lower() in ("nan", "none", "nat"):
        return None

    # Excel-as-text artifact: "7659935016.0"
    if re.fullmatch(r"[0-9]+\.0", text):
        return text[:-2]

    return text


def strip_harmless_phone_formatting(phone_str: str) -> str:
    """
    Remove harmless formatting: spaces, +, -, parentheses, dots.
    Does NOT remove letters — leftover letters cause validation failure.
    """
    return _FORMATTING_PATTERN.sub("", phone_str)


def normalize_phone(phone) -> str:
    """
    Legacy helper: strip formatting from a phone value after coercion.
    Prefer normalize_to_whatsapp_number() for validation.
    """
    coerced = _coerce_phone_input_to_string(phone)
    if coerced is None:
        return ""
    return strip_harmless_phone_formatting(coerced)


def _is_valid_indian_local_mobile(ten_digits: str) -> bool:
    return (
        len(ten_digits) == 10
        and ten_digits.isdigit()
        and ten_digits[0] in ("6", "7", "8", "9")
    )


def normalize_to_whatsapp_number(phone) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize any accepted Indian mobile representation to canonical WhatsApp format:

        91XXXXXXXXXX

    NEVER blindly prepends 91 onto an already-country-coded number
    (917659935016 must NOT become 91917659935016).

    Returns:
        (whatsapp_number, None) on success
        (None, error_message) on failure
    """
    coerced = _coerce_phone_input_to_string(phone)
    if coerced is None or coerced == "":
        return None, "Phone number is empty"

    cleaned = strip_harmless_phone_formatting(coerced)

    # Letters or other non-digits remaining → invalid (do not strip them away)
    if not cleaned.isdigit():
        return None, INVALID_INDIAN_MOBILE_MSG

    # 10-digit local Indian mobile
    if len(cleaned) == 10:
        if not _is_valid_indian_local_mobile(cleaned):
            return None, INVALID_INDIAN_MOBILE_MSG
        return f"{INDIA_COUNTRY_CODE}{cleaned}", None

    # 11-digit with leading trunk 0 (07659935016)
    if len(cleaned) == 11 and cleaned.startswith("0"):
        local = cleaned[1:]
        if not _is_valid_indian_local_mobile(local):
            return None, INVALID_INDIAN_MOBILE_MSG
        return f"{INDIA_COUNTRY_CODE}{local}", None

    # 12-digit already with country code 91
    if len(cleaned) == 12 and cleaned.startswith(INDIA_COUNTRY_CODE):
        local = cleaned[2:]
        if not _is_valid_indian_local_mobile(local):
            return None, INVALID_INDIAN_MOBILE_MSG
        # Keep unchanged — do NOT prepend 91 again
        return cleaned, None

    return None, INVALID_INDIAN_MOBILE_MSG


def to_whatsapp_phone(ten_digit: str, country_code: str = INDIA_COUNTRY_CODE) -> str:
    """Convert a validated 10-digit Indian mobile to WhatsApp format."""
    return f"{country_code}{ten_digit}"


def validate_phone(phone) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate an Indian mobile phone number.

    Returns:
        (local_10_digit_number, None) if valid
        (None, error_reason) if invalid

    Use normalize_to_whatsapp_number() when the canonical 91XXXXXXXXXX value is needed.
    """
    whatsapp, err = normalize_to_whatsapp_number(phone)
    if err:
        return None, err
    # Local 10-digit portion
    return whatsapp[len(INDIA_COUNTRY_CODE):], None


def _is_blank_row(raw_name, raw_phone) -> bool:
    name_blank = raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)) or str(raw_name).strip() == ""
    if raw_phone is None or (isinstance(raw_phone, float) and pd.isna(raw_phone)):
        phone_blank = True
    else:
        coerced = _coerce_phone_input_to_string(raw_phone)
        phone_blank = coerced is None or coerced == ""
    return name_blank and phone_blank


def _original_phone_display(value) -> str:
    """
    Preserve the original phone as supplied for display.
    For Excel float integers, show without scientific notation.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        # Preserve readable original without corrupting to scientific notation
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def validate_customers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate all customer records in the DataFrame.

    Extra columns are ignored for WhatsApp personalization (only Name + Phone number used).

    Duplicate detection uses Normalized Phone (WhatsApp number) AFTER normalization.

    Returns:
        (valid_df, invalid_df, duplicate_df)

        - valid_df: Name, Phone number (= WhatsApp 91…), Original Phone,
                    Normalized Phone, Status
        - invalid_df: Row, Name, Phone number, Original Phone, Reason, Status
          (Normalized Phone blank for invalid)
        - duplicate_df: rows with duplicate Normalized Phone (keep first as Valid)
    """
    valid_records = []
    invalid_records = []

    for offset, (idx, row) in enumerate(df.iterrows()):
        row_number = offset + 1
        raw_name = row.get("Name") if "Name" in df.columns else None
        raw_phone = row.get("Phone number") if "Phone number" in df.columns else None
        original_phone = _original_phone_display(raw_phone)

        if _is_blank_row(raw_name, raw_phone):
            invalid_records.append({
                "Row": row_number,
                "Name": "",
                "Phone number": "",
                "Original Phone": "",
                "Normalized Phone": "",
                "Reason": f"Blank row at row {row_number}",
                "Status": "Invalid",
            })
            continue

        cleaned_name, name_error = validate_name(raw_name)
        whatsapp_number, phone_error = normalize_to_whatsapp_number(raw_phone)

        errors = []
        if name_error:
            errors.append(f"{name_error} for row {row_number}")
        if phone_error:
            if phone_error == "Phone number is empty":
                errors.append(f"Phone number is empty for row {row_number}")
            else:
                errors.append(f"{INVALID_INDIAN_MOBILE_MSG} for row {row_number}")

        if errors:
            name_display = ""
            if raw_name is not None and not (isinstance(raw_name, float) and pd.isna(raw_name)):
                name_display = str(raw_name).strip()
            invalid_records.append({
                "Row": row_number,
                "Name": name_display,
                "Phone number": original_phone,
                "Original Phone": original_phone,
                "Normalized Phone": "",
                "Reason": "; ".join(errors),
                "Status": "Invalid",
            })
        else:
            # Phone number stored as canonical WhatsApp number for downstream send
            valid_records.append({
                "Name": cleaned_name,
                "Phone number": whatsapp_number,
                "Original Phone": original_phone,
                "Normalized Phone": whatsapp_number,
                "Status": "Valid",
            })

    valid_df = pd.DataFrame(
        valid_records,
        columns=["Name", "Phone number", "Original Phone", "Normalized Phone", "Status"],
    )
    invalid_df = pd.DataFrame(
        invalid_records,
        columns=[
            "Row", "Name", "Phone number", "Original Phone",
            "Normalized Phone", "Reason", "Status",
        ],
    )

    # Duplicate detection AFTER normalization on WhatsApp number
    if not valid_df.empty:
        duplicated_mask = valid_df["Normalized Phone"].duplicated(keep="first")
        duplicate_df = valid_df[duplicated_mask].copy().reset_index(drop=True)
        if not duplicate_df.empty:
            duplicate_df["Status"] = "Duplicate"
        valid_df = valid_df[~duplicated_mask].copy().reset_index(drop=True)
    else:
        duplicate_df = pd.DataFrame(
            columns=["Name", "Phone number", "Original Phone", "Normalized Phone", "Status"]
        )

    return valid_df, invalid_df, duplicate_df


def build_preview_table(
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a combined preview table for the UI:
    Name | Original Phone | Normalized Phone (WhatsApp number) | Status
    """
    rows: List[Dict[str, Any]] = []

    for _, row in valid_df.iterrows():
        rows.append({
            "Name": row["Name"],
            "Original Phone": row["Original Phone"],
            NORMALIZED_PHONE_LABEL: row["Normalized Phone"],
            "Status": "Valid",
        })

    for _, row in duplicate_df.iterrows():
        rows.append({
            "Name": row["Name"],
            "Original Phone": row.get("Original Phone", row.get("Phone number", "")),
            NORMALIZED_PHONE_LABEL: row.get("Normalized Phone", ""),
            "Status": "Duplicate",
        })

    for _, row in invalid_df.iterrows():
        rows.append({
            "Name": row.get("Name", ""),
            "Original Phone": row.get("Original Phone", row.get("Phone number", "")),
            NORMALIZED_PHONE_LABEL: "",
            "Status": "Invalid",
        })

    return pd.DataFrame(
        rows,
        columns=["Name", "Original Phone", NORMALIZED_PHONE_LABEL, "Status"],
    )


def get_template_variable_mapping(customer_name: str, store_name: str) -> Dict[str, str]:
    """
    Local-only mapping for the approved 3-variable template.
    Does NOT call Xinno.
    """
    return {
        "{{1}}": str(customer_name).strip(),
        "{{2}}": str(store_name).strip(),
        "{{3}}": str(store_name).strip(),
    }


def generate_message(customer_name: str, store_name: str) -> str:
    """Generate a personalized refill reminder message (local preview only)."""
    name = str(customer_name or "").strip()
    store = str(store_name or "").strip()
    msg = MESSAGE_TEMPLATE.replace("{{customer_name}}", name)
    msg = msg.replace("{{store_name}}", store)
    return msg
