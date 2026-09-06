"""
Image Campaign utilities: customer grouping, medicine list building,
column normalization for image-campaign-specific fields, and
template variable generation for the Image + Text WhatsApp campaign.

This module does NOT import Streamlit and is fully testable standalone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from utils.column_aliases import (
    ALL_CANONICAL_FIELDS,
    CANONICAL_TO_INTERNAL_MAP,
    COLUMN_ALIASES as NEW_COLUMN_ALIASES,
    REQUIRED_CANONICAL_FIELDS,
    AliasResolutionResult,
    normalize_dataframe_columns,
    normalize_header_string,
    resolve_column_aliases,
)
from utils.validators import (
    NORMALIZED_PHONE_LABEL,
    normalize_columns,
    normalize_to_whatsapp_number,
    validate_name,
)


# ---------------------------------------------------------------------------
# Required columns for the image campaign
# ---------------------------------------------------------------------------
IMAGE_CAMPAIGN_REQUIRED_COLUMNS = [
    "Name",
    "Phone number",
    "Medicine",
    "Branch",
]

# Backward-compatible column aliases dictionary
IMAGE_CAMPAIGN_COLUMN_ALIASES: Dict[str, List[str]] = {
    "Name": list(NEW_COLUMN_ALIASES.get("customer_name", [])),
    "Phone number": list(NEW_COLUMN_ALIASES.get("phone", [])),
    "Medicine": list(NEW_COLUMN_ALIASES.get("medicine", [])),
    "Branch": list(NEW_COLUMN_ALIASES.get("branch", [])),
    "Contact No.": list(NEW_COLUMN_ALIASES.get("contact_no", [])),
    "Manager Contact": list(NEW_COLUMN_ALIASES.get("manager_contact", [])),
}

# ---------------------------------------------------------------------------
# Local preview template (matches approved 'refill_reminder_image' template)
# ---------------------------------------------------------------------------
IMAGE_CAMPAIGN_PREVIEW_TEMPLATE = (
    "Dear *{{1}}* Garu,\n"
    "You are a valued customer of\n"
    "*{{2}}*,\n"
    "*{{3}}*.\n\n"
    "📋 Our records indicate that it may be time to refill your medication(s):\n"
    "{{4}}\n\n"
    "For any assistance/To place your orders please contact our store\n"
    "📞 {{5}}\n\n"
    "For complaints or suggestions, please contact our manager at\n"
    "📞 {{6}}\n\n"
    "🙏 Thank you for choosing us for your pharmacy needs.\n\n"
    "🧡 *Team*\n"
    "*{{7}}*\n"
    "📍 *{{8}}*\n"
    "We look forward to serving you."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phone_display(value) -> str:
    """Display original phone value, handling NaN/None and float artifacts."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


# ---------------------------------------------------------------------------
# Column normalization using the production alias system
# ---------------------------------------------------------------------------

def normalize_image_campaign_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column headers for the image campaign using the central alias system.
    """
    if df is None or len(df.columns) == 0:
        return df
    renamed_df, _ = normalize_dataframe_columns(df)
    return renamed_df


def check_image_campaign_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of missing required columns (empty if all present)."""
    if df is None:
        return list(IMAGE_CAMPAIGN_REQUIRED_COLUMNS)
    return [col for col in IMAGE_CAMPAIGN_REQUIRED_COLUMNS if col not in df.columns]


def missing_image_columns_message(missing: List[str]) -> str:
    """User-friendly message for missing required columns."""
    if not missing:
        return ""
    if len(missing) == 1:
        return f"Missing required column: {missing[0]}"
    return f"Missing required columns: {', '.join(missing)}"


import re


def sanitize_template_variable(text: str) -> str:
    """
    Sanitize a template variable string for Meta WhatsApp API compliance.

    Meta WhatsApp API Rules:
    - No newlines (\\n, \\r)
    - No tab characters (\\t)
    - No more than 4 consecutive spaces
    """
    if not text:
        return ""
    # Replace newlines, carriage returns, and tabs with spaces
    sanitized = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Replace multiple consecutive spaces with a single space
    sanitized = re.sub(r" {2,}", " ", sanitized)
    return sanitized.strip()


def build_medicine_list(medicines: List[str]) -> str:
    """
    Build a formatted single-line medicine list string from individual medicine entries.

    Removes duplicates (case-insensitive), preserves order of first occurrence.
    Splits comma-, semicolon-, or newline-separated values within a single entry.

    Returns a single-line string with comma-separated medicines::

        METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG

    Complies with Meta WhatsApp Cloud API rules (no newlines/tabs/consecutive spaces).
    """
    seen: set[str] = set()
    unique: List[str] = []
    for med in medicines:
        cleaned = str(med).strip()
        if not cleaned or cleaned.lower() in ("nan", "none", "nat", ""):
            continue
        parts = re.split(r"[,;\n]+", cleaned)
        for part in parts:
            p = part.strip().lstrip("-*• ").strip()
            if not p or p.lower() in ("nan", "none", "nat", ""):
                continue
            sanitized_p = sanitize_template_variable(p.upper())
            if not sanitized_p:
                continue
            key = sanitized_p.upper()
            if key not in seen:
                seen.add(key)
                unique.append(key)

    if not unique:
        return ""

    result = ", ".join(unique)
    return sanitize_template_variable(result)


# ---------------------------------------------------------------------------
# Customer validation + grouping
# ---------------------------------------------------------------------------

def validate_and_group_customers(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Validate rows and group valid customers by (normalized name + normalized phone).

    Two customers with different names sharing the same phone number remain
    separate customers.  For rows with the SAME customer identity (name + phone),
    medicines are combined into a single-line comma-separated string.

    If rows for the same customer identity have conflicting Branch, Contact No.,
    or Manager Contact values, a data conflict is flagged and the customer is
    marked invalid rather than guessing a conflicting value.

    Returns
    -------
    grouped_customers : list[dict]
        Each dict contains: Name, Phone number, Original Phone,
        Normalized Phone, Branch, Medicine List, Medicine Count,
        Contact No., Manager Contact, Status.
    invalid_df : pd.DataFrame
        Invalid rows with Row, Name, Phone number, Original Phone,
        Normalized Phone, Medicine, Reason, Status.
    """
    valid_rows: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []

    for offset, (idx, row) in enumerate(df.iterrows()):
        row_number = offset + 1
        raw_name = row.get("Name") if "Name" in df.columns else None
        raw_phone = row.get("Phone number") if "Phone number" in df.columns else None
        raw_medicine = row.get("Medicine") if "Medicine" in df.columns else None
        raw_branch = row.get("Branch") if "Branch" in df.columns else None
        raw_contact = row.get("Contact No.") if "Contact No." in df.columns else None
        raw_manager = row.get("Manager Contact") if "Manager Contact" in df.columns else None

        original_phone = _phone_display(raw_phone)

        # Re-use existing validators
        cleaned_name, name_error = validate_name(raw_name)
        whatsapp_number, phone_error = normalize_to_whatsapp_number(raw_phone)

        errors: List[str] = []
        if name_error:
            errors.append(f"{name_error} for row {row_number}")
        if phone_error:
            if phone_error == "Phone number is empty":
                errors.append(f"Phone number is empty for row {row_number}")
            else:
                errors.append(f"Invalid Indian mobile number for row {row_number}")

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
                "Medicine": str(raw_medicine or "").strip(),
                "Reason": "; ".join(errors),
                "Status": "Invalid",
            })
        else:
            valid_rows.append({
                "Row": row_number,
                "Name": cleaned_name,
                "Phone number": whatsapp_number,
                "Original Phone": original_phone,
                "Normalized Phone": whatsapp_number,
                "Medicine": str(raw_medicine or "").strip(),
                "Branch": str(raw_branch or "").strip(),
                "Contact No.": str(raw_contact or "").strip(),
                "Manager Contact": str(raw_manager or "").strip(),
                "Status": "Valid",
            })

    grouped, conflict_records = _group_by_name_and_phone(valid_rows)
    all_invalid = invalid_records + conflict_records
    all_invalid.sort(key=lambda x: x.get("Row", 0))

    invalid_df = pd.DataFrame(
        all_invalid,
        columns=[
            "Row", "Name", "Phone number", "Original Phone",
            "Normalized Phone", "Medicine", "Reason", "Status",
        ],
    )

    return grouped, invalid_df


def _group_by_name_and_phone(
    valid_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Group validated rows by (normalized_name, normalized_phone).

    Combines medicines from rows sharing the SAME normalized name + phone.
    Different names with the same phone remain separate customers.
    Detects data conflicts (different Branch, Contact No., or Manager Contact
    for the same customer identity) and returns them as conflict records.
    """
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    group_order: List[Tuple[str, str]] = []

    for row in valid_rows:
        norm_name = str(row["Name"]).strip().lower()
        norm_phone = str(row["Normalized Phone"]).strip()
        key = (norm_name, norm_phone)

        if key not in groups:
            group_order.append(key)
            groups[key] = {
                "first_row_num": row["Row"],
                "Name": row["Name"],
                "Phone number": row["Phone number"],
                "Original Phone": row["Original Phone"],
                "Normalized Phone": norm_phone,
                "Branch": row["Branch"],
                "Contact No.": row["Contact No."],
                "Manager Contact": row["Manager Contact"],
                "medicines": [],
                "rows": [row],
                "conflicts": [],
            }
        else:
            g = groups[key]
            g["rows"].append(row)

            # Check for data conflicts across rows with the same identity
            row_num = row["Row"]
            first_num = g["first_row_num"]

            # Branch conflict check
            curr_branch = str(row.get("Branch", "")).strip()
            first_branch = str(g.get("Branch", "")).strip()
            if curr_branch and first_branch and curr_branch.lower() != first_branch.lower():
                conflict_msg = f"Conflicting Branch values: '{first_branch}' (row {first_num}) vs '{curr_branch}' (row {row_num})"
                if conflict_msg not in g["conflicts"]:
                    g["conflicts"].append(conflict_msg)
            elif not first_branch and curr_branch:
                g["Branch"] = curr_branch

            # Contact No. conflict check
            curr_contact = str(row.get("Contact No.", "")).strip()
            first_contact = str(g.get("Contact No.", "")).strip()
            if curr_contact and first_contact and curr_contact != first_contact:
                conflict_msg = f"Conflicting Contact No. values: '{first_contact}' (row {first_num}) vs '{curr_contact}' (row {row_num})"
                if conflict_msg not in g["conflicts"]:
                    g["conflicts"].append(conflict_msg)
            elif not first_contact and curr_contact:
                g["Contact No."] = curr_contact

            # Manager Contact conflict check
            curr_manager = str(row.get("Manager Contact", "")).strip()
            first_manager = str(g.get("Manager Contact", "")).strip()
            if curr_manager and first_manager and curr_manager != first_manager:
                conflict_msg = f"Conflicting Manager Contact values: '{first_manager}' (row {first_num}) vs '{curr_manager}' (row {row_num})"
                if conflict_msg not in g["conflicts"]:
                    g["conflicts"].append(conflict_msg)
            elif not first_manager and curr_manager:
                g["Manager Contact"] = curr_manager

        med = row.get("Medicine", "").strip()
        if med and med.lower() not in ("nan", "none", "nat", ""):
            groups[key]["medicines"].append(med)

    grouped: List[Dict[str, Any]] = []
    conflict_records: List[Dict[str, Any]] = []

    for key in group_order:
        g = groups[key]
        if g["conflicts"]:
            conflict_reason = f"Data conflict for customer '{g['Name']}': " + "; ".join(g["conflicts"])
            for r in g["rows"]:
                conflict_records.append({
                    "Row": r["Row"],
                    "Name": r["Name"],
                    "Phone number": r["Original Phone"],
                    "Original Phone": r["Original Phone"],
                    "Normalized Phone": r["Normalized Phone"],
                    "Medicine": str(r.get("Medicine", "")).strip(),
                    "Reason": conflict_reason,
                    "Status": "Invalid",
                })
        else:
            medicine_text = build_medicine_list(g["medicines"])
            medicine_count = len(
                [m for m in medicine_text.split(",") if m.strip()]
            ) if medicine_text else 0
            grouped.append({
                "Name": g["Name"],
                "Phone number": g["Phone number"],
                "Original Phone": g["Original Phone"],
                "Normalized Phone": g["Normalized Phone"],
                "Branch": g["Branch"],
                "Contact No.": g["Contact No."],
                "Manager Contact": g["Manager Contact"],
                "Medicine List": medicine_text,
                "Medicine Count": medicine_count,
                "Status": "Valid",
            })

    return grouped, conflict_records


# Backward-compatible alias
_group_by_phone = _group_by_name_and_phone


# ---------------------------------------------------------------------------
# Template variable generation
# ---------------------------------------------------------------------------

def build_image_template_variables(
    customer: Dict[str, Any],
    store_name: str,
) -> List[Dict[str, str]]:
    """
    Build the 8 template body variables in the EXACT required order:

        1. Customer Name
        2. Store Name
        3. Branch
        4. Dynamic complete medicine list
        5. Contact No.
        6. Manager Contact
        7. Store Name
        8. Branch

    Returns a list of ``{"type": "text", "text": value}`` dicts suitable
    for the Xinno ``template.components[body].parameters`` array.
    """
    return [
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Name", "")))},
        {"type": "text", "text": sanitize_template_variable(str(store_name))},
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Branch", "")))},
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Medicine List", "")))},
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Contact No.", "")))},
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Manager Contact", "")))},
        {"type": "text", "text": sanitize_template_variable(str(store_name))},
        {"type": "text", "text": sanitize_template_variable(str(customer.get("Branch", "")))},
    ]


def get_variable_labels() -> List[str]:
    """Human-readable labels for the 8 template variables."""
    return [
        "Customer Name",
        "Store Name",
        "Branch",
        "Medicine List",
        "Contact No.",
        "Manager Contact",
        "Store Name",
        "Branch",
    ]


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------

def build_image_campaign_preview_table(
    grouped_customers: List[Dict[str, Any]],
    invalid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combined preview table for the image campaign UI."""
    rows: List[Dict[str, Any]] = []

    for c in grouped_customers:
        rows.append({
            "Name": c["Name"],
            "Original Phone": c["Original Phone"],
            NORMALIZED_PHONE_LABEL: c["Normalized Phone"],
            "Branch": c.get("Branch", ""),
            "Medicines": c.get("Medicine Count", 0),
            "Status": "Valid",
        })

    for _, row in invalid_df.iterrows():
        rows.append({
            "Name": row.get("Name", ""),
            "Original Phone": row.get("Original Phone", ""),
            NORMALIZED_PHONE_LABEL: "",
            "Branch": "",
            "Medicines": 0,
            "Status": "Invalid",
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Name", "Original Phone", NORMALIZED_PHONE_LABEL,
            "Branch", "Medicines", "Status",
        ],
    )


def generate_image_campaign_preview(
    customer: Dict[str, Any],
    store_name: str,
) -> str:
    """
    Generate an approximate rendered message preview for local display.

    NOTE: The actual rendered message depends entirely on the approved
    Xinno template.  This preview uses a reasonable approximation.
    """
    variables = build_image_template_variables(customer, store_name)
    text = IMAGE_CAMPAIGN_PREVIEW_TEMPLATE
    for i, var in enumerate(variables, start=1):
        text = text.replace(f"{{{{{i}}}}}", var["text"])
    return text


def build_image_sample_previews(
    grouped_customers: List[Dict[str, Any]],
    store_name: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build preview dicts for up to *limit* customers.

    Each dict contains the 8 variable values, a rendered message preview,
    and customer identification for the UI.
    """
    samples: List[Dict[str, Any]] = []
    for customer in grouped_customers[:max(0, limit)]:
        variables = build_image_template_variables(customer, store_name)
        labels = get_variable_labels()
        samples.append({
            "customer_name": customer["Name"],
            "normalized_phone": customer["Normalized Phone"],
            "original_phone": customer.get("Original Phone", ""),
            "branch": customer.get("Branch", ""),
            "medicine_list": customer.get("Medicine List", ""),
            "medicine_count": customer.get("Medicine Count", 0),
            "contact_no": customer.get("Contact No.", ""),
            "manager_contact": customer.get("Manager Contact", ""),
            "variables": variables,
            "variable_labels": labels,
            "message_preview": generate_image_campaign_preview(customer, store_name),
        })
    return samples


# ---------------------------------------------------------------------------
# Image URL validation
# ---------------------------------------------------------------------------

def validate_image_url(url: Optional[str]) -> Tuple[bool, str]:
    """
    Validate that a given string is a valid public HTTPS image URL.

    Rules:
    - Not empty
    - Must start with 'https://' (HTTP is rejected for security/WhatsApp requirements)
    - Must have a valid network location (hostname)
    - Rejects local filesystem paths (e.g. C:\\, /var, file://)
    - Does NOT require specific file extension (Cloudinary URLs often omit extension)

    Returns:
        (is_valid, error_message)
    """
    if not url or not str(url).strip():
        return False, "Image URL cannot be empty."

    cleaned = str(url).strip()

    # Reject local Windows or Unix file paths
    if cleaned.startswith("/") or cleaned.startswith("\\") or (len(cleaned) > 2 and cleaned[1] == ":"):
        return False, "Image URL cannot be a local filesystem path. It must be a public HTTPS URL."

    if cleaned.startswith("file://") or cleaned.startswith("ftp://"):
        return False, "Only HTTPS URLs are supported."

    if cleaned.startswith("http://"):
        return False, "Insecure HTTP URLs are not allowed. Please provide an HTTPS URL."

    if not cleaned.startswith("https://"):
        return False, "Image URL must start with 'https://'."

    try:
        parsed = urlparse(cleaned)
        if not parsed.scheme or parsed.scheme.lower() != "https":
            return False, "Image URL must use HTTPS."
        if not parsed.netloc:
            return False, "Image URL has an invalid or missing hostname."
    except Exception as exc:
        return False, f"Invalid URL format: {exc}"

    return True, ""

