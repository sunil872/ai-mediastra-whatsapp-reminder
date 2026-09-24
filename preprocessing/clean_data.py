"""Data cleaning and validation module for pharmacy sales transactions.

Enforces:
- Strict DD-MM-YYYY date parsing (preserving Excel native datetimes)
- Customer identity: customerId + itemId (NOT phone number)
- Positive quantities only (excludes returns/zero quantities)
- Same-day / same-invoice line aggregation
- Preservation of customers with missing mobile numbers
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from refillcare.data.dates import parse_pharmacy_dates, format_date_dd_mm_yyyy
from refillcare.data.monthly_ingestion import determine_mobile_status


def normalize_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible pharmacy column names to canonical schema without creating duplicate columns."""
    if df is None or len(df.columns) == 0:
        return df

    existing_cols = set(df.columns)
    col_map = {}
    assigned_targets = set()

    for c in df.columns:
        c_low = str(c).strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace(".", "")
        target = None
        if c_low in ("invoicedate", "date", "billdate", "orderdate", "transactiondate", "salestime", "trndate"):
            target = "invoice_date"
        elif c_low in ("trnno", "trnnbr", "transactionno", "invoicenumber", "invoiceno", "billnumber", "billno", "invoice"):
            target = "invoice_number"
        elif c_low in ("partycode", "patientcode", "customerid", "customer", "patientid", "custid"):
            target = "customerId"
        elif c_low in ("partyname", "customername", "patientname", "custname", "party", "patient"):
            target = "customerName"
        elif c_low in ("itemcode", "itemid", "medicineid", "drugid", "productid", "item"):
            target = "itemId"
        elif c_low in ("itemname", "medicinename", "drugname", "productname", "itemdescription", "medicine"):
            target = "itemName"
        elif c_low in ("quantity", "qty", "units", "quantitysold", "qtyordered"):
            target = "quantity"
        elif c_low in ("packing", "pack", "packsize", "packaging"):
            target = "packing"
        elif c_low in ("mobileno", "mobile", "phone", "phonenumber", "contact", "cell", "contactno"):
            target = "MOBILE_NO"
        elif c_low in ("saltcomposition", "salt", "composition", "genericname"):
            target = "salt_composition"

        if target:
            # If current column is already the exact canonical name, keep it and record target
            if c == target:
                assigned_targets.add(target)
            elif target not in existing_cols and target not in assigned_targets:
                col_map[c] = target
                assigned_targets.add(target)

    return df.rename(columns=col_map).copy()


def clean_sales_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean and validate raw sales dataframe.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and hygiene diagnostics.
    """
    if df.empty:
        return pd.DataFrame(), {"records_received": 0, "valid_records": 0}

    normalized = normalize_sales_columns(df)
    total_received = len(normalized)

    # 1. Strict Date Parsing
    if "invoice_date" in normalized.columns:
        parsed_dates, date_diag = parse_pharmacy_dates(normalized["invoice_date"])
        normalized["invoice_date"] = parsed_dates
    else:
        date_diag = {"is_ambiguous": True, "date_range_formatted": "Missing Date Column"}
        normalized["invoice_date"] = pd.NaT

    # 2. Positive quantity check (drop returns / <= 0)
    if "quantity" in normalized.columns:
        qty_num = pd.to_numeric(normalized["quantity"], errors="coerce")
        valid_qty_mask = qty_num.notna() & (qty_num > 0)
    else:
        valid_qty_mask = pd.Series(False, index=normalized.index)

    # 3. Customer & Item identity check (customerId + itemId)
    valid_id_mask = (
        normalized.get("customerId", pd.Series(dtype=object)).notna() &
        (normalized.get("customerId", pd.Series(dtype=object)).astype(str).str.strip() != "") &
        normalized.get("itemId", pd.Series(dtype=object)).notna() &
        (normalized.get("itemId", pd.Series(dtype=object)).astype(str).str.strip() != "")
    )

    valid_mask = normalized["invoice_date"].notna() & valid_qty_mask & valid_id_mask
    valid_df = normalized[valid_mask].copy().reset_index(drop=True)

    # 4. Mobile Status tagging (missing phones are NEVER deleted)
    if "MOBILE_NO" in valid_df.columns:
        valid_df["mobile_status"] = valid_df["MOBILE_NO"].apply(determine_mobile_status)
    else:
        valid_df["mobile_status"] = "Missing"

    # 5. Same-invoice line aggregation
    agg_cols = ["customerId", "itemId", "invoice_date"]
    if "invoice_number" in valid_df.columns:
        agg_cols.append("invoice_number")

    if all(c in valid_df.columns for c in agg_cols):
        valid_df["quantity"] = pd.to_numeric(valid_df["quantity"], errors="coerce").fillna(1)
        valid_df = valid_df.sort_values("invoice_date").reset_index(drop=True)

    metrics = {
        "records_received": total_received,
        "valid_records": len(valid_df),
        "invalid_records": total_received - len(valid_df),
        "date_diagnostics": date_diag,
    }

    return valid_df, metrics
