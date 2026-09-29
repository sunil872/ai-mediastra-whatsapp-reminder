"""Centralized Transaction Channel Classification & Filtering Layer for RefillCare V1.

Identifies transaction channels to prevent B2B / Inter-Store transactions (SB/ prefix)
from contaminating individual customer refill modeling, cadence, Path A/B routing,
feature engineering, or reminder generation.

Classification Rules:
- S0/...  -> CUSTOMER_SALE (refillcare_eligible=True)
- SB/...  -> B2B_INTER_STORE (refillcare_eligible=False, reason=INTER_STORE_TRANSACTION)
- other   -> UNKNOWN (refillcare_eligible=False, reason=UNKNOWN_TRANSACTION_TYPE)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Canonical Transaction Types
TRANSACTION_TYPE_CUSTOMER_SALE = "CUSTOMER_SALE"
TRANSACTION_TYPE_B2B_INTER_STORE = "B2B_INTER_STORE"
TRANSACTION_TYPE_UNKNOWN = "UNKNOWN"

# Canonical Exclusion Reasons
EXCLUSION_REASON_NONE = None
EXCLUSION_REASON_INTER_STORE = "INTER_STORE_TRANSACTION"
EXCLUSION_REASON_UNKNOWN_TYPE = "UNKNOWN_TRANSACTION_TYPE"


class ChannelClassification:
    CUSTOMER_SALE = TRANSACTION_TYPE_CUSTOMER_SALE
    B2B_INTER_STORE = TRANSACTION_TYPE_B2B_INTER_STORE
    UNKNOWN = TRANSACTION_TYPE_UNKNOWN


class ExclusionReason:
    NONE = EXCLUSION_REASON_NONE
    INTER_STORE_TRANSACTION = EXCLUSION_REASON_INTER_STORE
    UNKNOWN_TRANSACTION_TYPE = EXCLUSION_REASON_UNKNOWN_TYPE


# Normalization Prefixes
PREFIX_CUSTOMER_SALE = "S0/"
PREFIX_B2B_INTER_STORE = "SB/"


def normalize_transaction_number(val: Any) -> str:
    """Normalize raw transaction / invoice numbers for robust matching.

    Handles leading/trailing whitespace, uppercase conversion, and None/NaN values.

    Args:
        val: Raw transaction number string, number, or NaN.

    Returns:
        str: Cleaned, upper-case transaction number (empty string if invalid/missing).
    """
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().upper()
    if s in ("NAN", "NONE", "NULL", "<NA>"):
        return ""
    return s


def classify_transaction(transaction_number: Any) -> Tuple[str, bool, Optional[str]]:
    """Classify a single transaction by its normalized prefix.

    Args:
        transaction_number: Raw transaction identifier (e.g. 'S0/102001', 'SB/021001', ' sb/021001 ').

    Returns:
        Tuple[str, bool, Optional[str]]:
            - transaction_type: CUSTOMER_SALE, B2B_INTER_STORE, or UNKNOWN
            - refillcare_eligible: True if eligible for customer refill modeling, False otherwise
            - exclusion_reason: None, INTER_STORE_TRANSACTION, or UNKNOWN_TRANSACTION_TYPE
    """
    norm = normalize_transaction_number(transaction_number)
    if not norm:
        return TRANSACTION_TYPE_UNKNOWN, False, EXCLUSION_REASON_UNKNOWN_TYPE

    if norm.startswith(PREFIX_CUSTOMER_SALE):
        return TRANSACTION_TYPE_CUSTOMER_SALE, True, EXCLUSION_REASON_NONE
    elif norm.startswith(PREFIX_B2B_INTER_STORE):
        return TRANSACTION_TYPE_B2B_INTER_STORE, False, EXCLUSION_REASON_INTER_STORE
    else:
        return TRANSACTION_TYPE_UNKNOWN, False, EXCLUSION_REASON_UNKNOWN_TYPE


def classify_transactions_df(
    df: pd.DataFrame,
    invoice_col: str = "invoice_number",
    allow_generic_customer: bool = False,
) -> pd.DataFrame:
    """Classify all rows in a DataFrame and annotate channel metadata.

    Adds three columns to the DataFrame:
    - transaction_type: CUSTOMER_SALE, B2B_INTER_STORE, or UNKNOWN
    - refillcare_eligible: boolean (True for CUSTOMER_SALE, False otherwise)
    - exclusion_reason: None, INTER_STORE_TRANSACTION, or UNKNOWN_TRANSACTION_TYPE

    Also ensures 'transaction_number' is present as an alias to the invoice column.

    Args:
        df: Input DataFrame containing sales or purchase records.
        invoice_col: Column name containing invoice / transaction numbers.
        allow_generic_customer: If True, non-SB transactions without S0/ prefix are treated as customer sales.

    Returns:
        pd.DataFrame: DataFrame annotated with transaction channel columns.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out["transaction_type"] = pd.Series(dtype=str)
        out["refillcare_eligible"] = pd.Series(dtype=bool)
        out["exclusion_reason"] = pd.Series(dtype=str)
        if "transaction_number" not in out.columns:
            out["transaction_number"] = pd.Series(dtype=str)
        return out

    out = df.copy()
    col = None
    for cand in [invoice_col, "invoice_number", "transaction_number", "TRAN_NO", "bill_no", "Bill No"]:
        if cand and cand in out.columns:
            col = cand
            break

    if col is None:
        if allow_generic_customer:
            out["transaction_type"] = TRANSACTION_TYPE_CUSTOMER_SALE
            out["refillcare_eligible"] = True
            out["exclusion_reason"] = EXCLUSION_REASON_NONE
            out["transaction_number"] = ""
        else:
            # If no invoice column is present, mark all as UNKNOWN
            out["transaction_type"] = TRANSACTION_TYPE_UNKNOWN
            out["refillcare_eligible"] = False
            out["exclusion_reason"] = EXCLUSION_REASON_UNKNOWN_TYPE
            out["transaction_number"] = ""
        return out

    norm_series = out[col].astype(str).str.strip().str.upper()
    is_cust = norm_series.str.startswith(PREFIX_CUSTOMER_SALE)
    is_b2b = norm_series.str.startswith(PREFIX_B2B_INTER_STORE)

    if allow_generic_customer:
        types = np.where(is_b2b, TRANSACTION_TYPE_B2B_INTER_STORE, TRANSACTION_TYPE_CUSTOMER_SALE)
        eligible = (~is_b2b).values
        reasons = np.where(is_b2b, EXCLUSION_REASON_INTER_STORE, EXCLUSION_REASON_NONE)
    else:
        types = np.where(is_cust, TRANSACTION_TYPE_CUSTOMER_SALE, np.where(is_b2b, TRANSACTION_TYPE_B2B_INTER_STORE, TRANSACTION_TYPE_UNKNOWN))
        eligible = is_cust.values
        reasons = np.where(
            is_cust,
            EXCLUSION_REASON_NONE,
            np.where(is_b2b, EXCLUSION_REASON_INTER_STORE, EXCLUSION_REASON_UNKNOWN_TYPE),
        )

    out["transaction_type"] = types
    out["refillcare_eligible"] = eligible
    out["exclusion_reason"] = reasons

    if "transaction_number" not in out.columns:
        out["transaction_number"] = out[col]

    return out


def filter_eligible_customer_transactions(
    df: pd.DataFrame,
    invoice_col: str = "invoice_number",
) -> pd.DataFrame:
    """Filter DataFrame to retain ONLY transactions eligible for customer refill modeling.

    Strictly excludes B2B_INTER_STORE and UNKNOWN transaction types.

    Args:
        df: Input DataFrame.
        invoice_col: Column containing invoice or transaction identifier.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid CUSTOMER_SALE transactions.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    if "refillcare_eligible" in df.columns:
        return df[df["refillcare_eligible"] == True].copy()

    # If not yet classified, classify and filter
    classified = classify_transactions_df(df, invoice_col=invoice_col)
    return classified[classified["refillcare_eligible"] == True].copy()


def get_transaction_channel_summary(
    df: pd.DataFrame,
    invoice_col: str = "invoice_number",
) -> Dict[str, int]:
    """Calculate aggregate transaction channel statistics from a DataFrame.

    Returns:
        Dict[str, int]: Dictionary with counts for:
            - total_transactions
            - customer_sales
            - b2b_inter_store
            - unknown
            - excluded_from_refillcare
            - refillcare_eligible
    """
    if df is None or df.empty:
        return {
            "total_transactions": 0,
            "customer_sales": 0,
            "b2b_inter_store": 0,
            "unknown": 0,
            "excluded_from_refillcare": 0,
            "refillcare_eligible": 0,
        }

    if "transaction_type" not in df.columns or "refillcare_eligible" not in df.columns:
        df_classified = classify_transactions_df(df, invoice_col=invoice_col)
    else:
        df_classified = df

    total = len(df_classified)
    cust = int((df_classified["transaction_type"] == TRANSACTION_TYPE_CUSTOMER_SALE).sum())
    b2b = int((df_classified["transaction_type"] == TRANSACTION_TYPE_B2B_INTER_STORE).sum())
    unknown = int((df_classified["transaction_type"] == TRANSACTION_TYPE_UNKNOWN).sum())
    eligible = int((df_classified["refillcare_eligible"] == True).sum())
    excluded = total - eligible

    return {
        "total_transactions": total,
        "customer_sales": cust,
        "b2b_inter_store": b2b,
        "unknown": unknown,
        "excluded_from_refillcare": excluded,
        "refillcare_eligible": eligible,
    }
