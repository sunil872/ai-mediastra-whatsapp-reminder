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

    # 6. Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    return cleaned.reset_index(drop=True)


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
