"""Customer-Item Identity Normalization Module (Audit Only).

Provides conservative, deterministic normalization for:
- Store / Branch ID
- Customer Phone (reusing utils.validators.normalize_to_whatsapp_number)
- Customer Name (uppercase, whitespace collapsed, punctuation stripped, NO fuzzy matching)
- Item ID

Generates:
- Current Entity Key: f"{customerId}_{itemId}"
- Proposed Entity Key: f"{store_id}_{normalized_phone}_{normalized_customer_name}_{normalized_item_id}"
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import pandas as pd

from utils.validators import (
    INDIA_COUNTRY_CODE,
    _coerce_phone_input_to_string,
    normalize_to_whatsapp_number,
    strip_harmless_phone_formatting,
)

# Punctuation characters to strip out during name normalization
# Retains alphanumeric characters and spaces
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_store_id(store: Any, default: str = "MAIN") -> str:
    """Normalize store / branch identifier deterministically."""
    if store is None or (isinstance(store, float) and pd.isna(store)):
        return default
    text = str(store).strip().upper()
    if text == "" or text.lower() in ("nan", "none", "nat", "null"):
        return default
    return text


def normalize_customer_name(name: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Conservative, deterministic customer name normalization.

    Rules:
    - Convert to string
    - Trim leading/trailing whitespace
    - Uppercase
    - Strip punctuation (dots, commas, hyphens, brackets, etc.)
    - Collapse repeated whitespace to a single space
    - DO NOT use fuzzy matching (SUNIL, SUNIL K, SUNIL KAREN remain distinct)

    Returns:
        (normalized_name, error_reason)
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None, "Missing customer name"

    text = str(name).strip().upper()
    if text == "" or text.lower() in ("nan", "none", "nat", "null"):
        return None, "Missing customer name"

    # Strip punctuation
    text = _PUNCTUATION_RE.sub(" ", text)
    # Collapse multiple whitespaces
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return None, "Customer name became empty after punctuation stripping"

    return text, None


def normalize_phone_number(phone: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize phone number using the repository's authoritative WhatsApp format (91XXXXXXXXXX).

    Returns:
        (normalized_phone_12_digit, error_reason)
    """
    if phone is None or (isinstance(phone, float) and pd.isna(phone)):
        return None, "Missing phone number"

    coerced = _coerce_phone_input_to_string(phone)
    if coerced is None or coerced == "" or coerced.lower() in ("nan", "none", "nat", "null", "0", "0.0"):
        return None, "Missing phone number"

    return normalize_to_whatsapp_number(phone)


def normalize_item_id(item_id: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalize item identifier deterministically."""
    if item_id is None or (isinstance(item_id, float) and pd.isna(item_id)):
        return None, "Missing item ID"

    if isinstance(item_id, float) and item_id.is_integer() and item_id >= 0:
        text = str(int(item_id))
    else:
        text = str(item_id).strip().upper()

    if text == "" or text.lower() in ("nan", "none", "nat", "null"):
        return None, "Missing item ID"

    return text, None


def mask_phone(phone: Any) -> str:
    """Mask phone number to ensure privacy in reports (e.g., ******3210)."""
    if phone is None or (isinstance(phone, float) and pd.isna(phone)):
        return "MISSING_PHONE"
    text = str(phone).strip()
    if text == "" or text.lower() in ("nan", "none", "nat", "null"):
        return "MISSING_PHONE"

    digits = re.sub(r"\D", "", text)
    if len(digits) >= 4:
        return f"******{digits[-4:]}"
    return "INVALID_PHONE"


def build_current_key(customer_id: Any, item_id: Any) -> str:
    """Build current operational entity key: f'{customerId}_{itemId}'."""
    c_str = "" if customer_id is None or pd.isna(customer_id) else str(customer_id).strip()
    i_str = "" if item_id is None or pd.isna(item_id) else str(item_id).strip()
    return f"{c_str}_{i_str}"


def build_proposed_key(
    store_id: Any,
    phone: Any,
    customer_name: Any,
    item_id: Any,
    default_store: str = "MAIN",
) -> Tuple[str, str]:
    """
    Build proposed composite customer-item key:
    f"{normalized_store_id}_{normalized_phone}_{normalized_customer_name}_{normalized_item_id}"

    Returns:
        (proposed_key, safety_status)
        safety_status is one of:
            - 'SAFE'
            - 'INSUFFICIENT_IDENTITY_DATA'
    """
    norm_store = normalize_store_id(store_id, default=default_store)
    norm_phone, phone_err = normalize_phone_number(phone)
    norm_name, name_err = normalize_customer_name(customer_name)
    norm_item, item_err = normalize_item_id(item_id)

    has_issue = False
    phone_token = norm_phone if norm_phone else "NO_PHONE"
    name_token = norm_name if norm_name else "NO_NAME"
    item_token = norm_item if norm_item else "NO_ITEM"

    if phone_err or name_err or item_err:
        safety_status = "INSUFFICIENT_IDENTITY_DATA"
    else:
        safety_status = "SAFE"

    key = f"{norm_store}_{phone_token}_{name_token}_{item_token}"
    return key, safety_status
