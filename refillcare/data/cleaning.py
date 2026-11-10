"""Transaction cleaning and invoice aggregation module for RefillCare.

Provides robust functions to load raw pharmacy transaction data,
clean missing and duplicate records, and aggregate duplicate invoice lines
into single purchase events per (customerId + invoice_number + invoice_date + itemId).
"""

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np


NUMERIC_COLUMNS = [
    "quantity",
    "freeQuantity",
    "rate",
    "saleRate",
    "mrp",
    "invoice_total",
    "discountPercent",
    "item_discount",
    "gstAmount",
    "netAmount",
    "costPrice",
]

STRING_COLUMNS = [
    "customerId",
    "itemId",
    "itemCode",
    "customerName",
    "itemName",
    "invoice_number",
    "packing",
    "batchNo",
    "expiryDate",
    "MOBILE_NO",
    "therapeuticCategory",
    "generic_name",
]

PURCHASE_EVENT_KEY = ["customerId", "invoice_number", "invoice_date", "itemId"]

# Legacy pharmacy export headers → canonical RefillCare column
_MOBILE_NO_ALIASES = ("phone1", "Phone1", "PHONE1", "phone_1", "Phone 1")


def normalize_mobile_no_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy phone columns to canonical ``MOBILE_NO``."""
    if df is None or df.empty:
        return df
    out = df
    if "MOBILE_NO" not in out.columns:
        for alias in _MOBILE_NO_ALIASES:
            if alias in out.columns:
                out = out.rename(columns={alias: "MOBILE_NO"})
                break
    # Drop leftover legacy aliases if both somehow exist
    drop_cols = [c for c in _MOBILE_NO_ALIASES if c in out.columns and c != "MOBILE_NO"]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def load_raw_transactions(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw transaction CSV data with date parsing and type conversion.

    Args:
        file_path: Path to the raw transaction CSV file.

    Returns:
        pd.DataFrame: Loaded transactions with parsed dates and typed columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found at: {path}")

    # Load CSV treating strings as object/string.
    # Include legacy phone1 so old exports still load as strings before rename.
    dtype_map = {col: str for col in STRING_COLUMNS}
    for alias in _MOBILE_NO_ALIASES:
        dtype_map[alias] = str

    df = pd.read_csv(
        path,
        dtype=dtype_map,
        low_memory=False,
        on_bad_lines="skip",
    )
    df = normalize_mobile_no_column(df)

    # Strip whitespace from string columns
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Convert 'nan' or empty strings back to NaN / empty string
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Parse invoice_date with dayfirst=True
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(
            df["invoice_date"],
            dayfirst=True,
            errors="coerce",
        )

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


import re


def clean_customer_identifier(val: str) -> str:
    """Normalize and clean customerId/customerName by removing noisy punctuation
    (e.g., leading/trailing dots, commas, slashes, quotes, semicolons) and collapsing extra whitespace,
    while preserving phone numbers and alphanumeric identity.
    """
    if pd.isna(val) or not str(val).strip():
        return ""
    s = str(val).strip()

    # Check if there is a _PHONE suffix (e.g. NAME_9849338327)
    phone_match = re.search(r"^(.*)_(\d{7,15})$", s)
    if phone_match:
        name_part = phone_match.group(1)
        phone_part = phone_match.group(2)
        name_clean = re.sub(r"[,.\/;'\"_\-+=\\~`!@#\$%\^&\*\(\)\[\]\<\>\?\|]+", " ", name_part)
        name_clean = re.sub(r"\s+", " ", name_clean).strip()
        if name_clean:
            return f"{name_clean}_{phone_part}"
        else:
            return phone_part

    # If the whole string is purely numeric (phone number as name/id)
    if re.match(r"^\d+$", s):
        return s

    # Otherwise clean the whole name string
    name_clean = re.sub(r"[,.\/;'\"_\-+=\\~`!@#\$%\^&\*\(\)\[\]\<\>\?\|]+", " ", s)
    name_clean = re.sub(r"\s+", " ", name_clean).strip()
    return name_clean


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transactions by removing invalid rows, exact duplicates, and unused fields.

    Cleaning rules:
    1. Normalize and clean `customerId` and `customerName` (strip leading/trailing special characters).
    2. Drop rows with missing or empty `customerId`.
    3. Drop rows with missing or empty `itemId`.
    4. Drop rows with invalid / unparseable `invoice_date`.
    5. Drop `mfgDate` column (Phase 1 verified 100% null).
    6. Remove exact duplicate rows across all columns.

    Args:
        df: Raw transactions DataFrame.

    Returns:
        pd.DataFrame: Cleaned transactions.
    """
    cleaned = normalize_mobile_no_column(df.copy())

    # 1. Drop mfgDate if present
    if "mfgDate" in cleaned.columns:
        cleaned = cleaned.drop(columns=["mfgDate"])

    # 2. Clean special characters in customerId and customerName
    if "customerId" in cleaned.columns:
        cleaned["customerId"] = cleaned["customerId"].apply(clean_customer_identifier)
        cleaned["customerId"] = cleaned["customerId"].replace({"": np.nan})

    if "customerName" in cleaned.columns:
        cleaned["customerName"] = cleaned["customerName"].apply(clean_customer_identifier)
        cleaned["customerName"] = cleaned["customerName"].replace({"": np.nan})

    # 3. Filter out missing customerId
    valid_customer = cleaned["customerId"].notna() & (cleaned["customerId"].astype(str).str.strip() != "")
    cleaned = cleaned[valid_customer]

    # 4. Filter out missing itemId
    valid_item = cleaned["itemId"].notna() & (cleaned["itemId"].astype(str).str.strip() != "")
    cleaned = cleaned[valid_item]

    # 5. Filter out invalid invoice_date
    if "invoice_date" in cleaned.columns:
        valid_date = cleaned["invoice_date"].notna()
        cleaned = cleaned[valid_date]

    # 6. Identical lines are NOT removed: at the counter the same batch scanned twice (or a strip added
    #    later in the bill) produces identical export lines that are real strips sold. Validated on
    #    23,641 purchases (9 Nov 2026): adding them predicts the next visit at 32 d (real 31 d), dropping
    #    them predicts 16 d. Lines are added up per bill by ``merge_bill_lines`` / ``aggregate_invoice_items``.

    return cleaned.reset_index(drop=True)


BILL_LINE_KEY = ["customerId", "itemId", "invoice_number", "_bill_day"]
BILL_SUM_COLS = ("quantity", "freeQuantity", "netAmount", "gstAmount", "item_discount")


def merge_bill_lines(df: pd.DataFrame, by: str = "bill") -> pd.DataFrame:
    """THE purchase rule (shared by uploads, re-scoring, training and evaluation):

    every line with the same customer + medicine + bill number + bill date is ONE purchase, and its
    strips are ADDED UP - billed and free quantity - including lines that are exactly identical
    (same batch scanned again). Example: S0/26135 NODOSIS lines 1 + 3 + 1 -> one purchase of 5 strips.

    Rows without a bill number are left as they are. Other columns keep the first line's value;
    ``daysSupply`` keeps the largest value. Idempotent (an already merged frame is unchanged).

    ``by="day"``: also adds up separate bills of the same medicine on the same day (two visits, or a
    second bill for a forgotten strip) - one purchase per patient + medicine + day, as training and
    evaluation count purchases. Used by the decision engine.
    """
    if by == "day":
        if df is None or df.empty or not {"customerId", "itemId", "invoice_date"}.issubset(df.columns):
            return df
        work = df.copy()
        work["_bill_day"] = pd.to_datetime(work["invoice_date"], errors="coerce").dt.normalize()
        key = ["customerId", "itemId", "_bill_day"]
        ok = work["_bill_day"].notna()
        keyed, rest = work[ok], work[~ok]
        if not keyed.duplicated(subset=key).any():
            return df
        for c in BILL_SUM_COLS:
            if c in keyed.columns:
                keyed = keyed.assign(**{c: pd.to_numeric(keyed[c], errors="coerce")})
        agg = {c: "first" for c in keyed.columns if c not in key}
        for c in BILL_SUM_COLS:
            if c in keyed.columns:
                agg[c] = lambda s: s.sum(min_count=1)
        if "daysSupply" in keyed.columns:
            agg["daysSupply"] = "max"
        merged = keyed.groupby(key, sort=False, dropna=False).agg(agg).reset_index()
        out = pd.concat([merged, rest], ignore_index=True) if not rest.empty else merged
        out = out.drop(columns=["_bill_day"]).sort_values("invoice_date", kind="stable").reset_index(drop=True)
        return out[[c for c in df.columns if c in out.columns]]
    if df is None or df.empty or not {"customerId", "itemId", "invoice_number", "invoice_date"}.issubset(df.columns):
        return df
    work = df.copy()
    work["_bill_day"] = pd.to_datetime(work["invoice_date"], errors="coerce").dt.normalize()
    inv = work["invoice_number"].astype(str).str.strip()
    has_bill = work["invoice_number"].notna() & (inv != "") & work["_bill_day"].notna()
    if not has_bill.any():
        return df
    keyed, rest = work[has_bill], work[~has_bill]
    if not keyed.duplicated(subset=BILL_LINE_KEY).any():
        return df  # nothing to merge
    for c in BILL_SUM_COLS:
        if c in keyed.columns:
            keyed = keyed.assign(**{c: pd.to_numeric(keyed[c], errors="coerce")})
    agg = {c: "first" for c in keyed.columns if c not in BILL_LINE_KEY}
    for c in BILL_SUM_COLS:
        if c in keyed.columns:
            agg[c] = lambda s: s.sum(min_count=1)
    if "daysSupply" in keyed.columns:
        agg["daysSupply"] = "max"
    merged = keyed.groupby(BILL_LINE_KEY, sort=False, dropna=False).agg(agg).reset_index()
    out = pd.concat([merged, rest], ignore_index=True) if not rest.empty else merged
    out = out.drop(columns=["_bill_day"])
    if "invoice_date" in out.columns:
        out = out.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    return out[[c for c in df.columns if c in out.columns]]


def aggregate_invoice_items(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple invoice lines into a single purchase event per
    (customerId + invoice_number + invoice_date + itemId).

    Business rationale:
    Pharmacy transactions frequently record multiple lines for the same medicine in the same invoice
    (e.g., due to multiple batches or partial quantities). These represent a single purchase visit,
    not multiple refill cycles. Summing quantities and financial amounts while retaining deterministic
    item attributes provides one canonical purchase event.

    Aggregation rules:
    - Sum: quantity, freeQuantity, netAmount, gstAmount, item_discount
    - First deterministic non-null: customerName, itemName, itemCode, packing, batchNo,
      expiryDate, rate, saleRate, mrp, discountPercent, costPrice, MOBILE_NO,
      therapeuticCategory, generic_name, invoice_total

    Args:
        df: Cleaned transactions DataFrame.

    Returns:
        pd.DataFrame: One purchase event per customer-invoice-date-item.
    """
    if df.empty:
        return df.copy()

    # Ensure required grouping keys exist
    for key in PURCHASE_EVENT_KEY:
        if key not in df.columns:
            raise KeyError(f"Required grouping column '{key}' missing from DataFrame.")

    # Define custom aggregation dictionary
    agg_dict = {}

    sum_cols = ["quantity", "freeQuantity", "netAmount", "gstAmount", "item_discount"]
    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = "sum"

    # All other columns use 'first' to deterministically keep representative details
    non_group_cols = [col for col in df.columns if col not in PURCHASE_EVENT_KEY]
    for col in non_group_cols:
        if col not in sum_cols:
            agg_dict[col] = "first"

    # Group and aggregate
    aggregated = df.groupby(PURCHASE_EVENT_KEY, as_index=False).agg(agg_dict)

    return aggregated
