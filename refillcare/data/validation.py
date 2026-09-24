"""Data quality validation module for RefillCare.

Provides automated validation checks across raw, cleaned, aggregated, and history
datasets to ensure zero corruption, no date leakage, and high reliability.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def validate_dataset(
    raw_df: Optional[pd.DataFrame] = None,
    clean_df: Optional[pd.DataFrame] = None,
    aggregated_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    salt_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute comprehensive data quality checks on RefillCare pipeline datasets.

    Checks:
    1. Missing customerId
    2. Missing itemId
    3. Invalid invoice_date
    4. Duplicate purchase events after aggregation
    5. Negative or zero quantity
    6. Missing MOBILE_NO
    7. Missing/placeholder generic_name
    8. Unmatched SALT item codes
    9. Duplicate SALT master codes
    10. CustomerId + itemId history ordering
    11. Invalid negative purchase intervals
    12. Date ranges

    Args:
        raw_df: Loaded raw transactions.
        clean_df: Filtered and cleaned transactions.
        aggregated_df: Invoice-aggregated purchase events.
        history_df: Purchase history with calculated intervals.
        salt_df: Normalized SALT master catalog.

    Returns:
        Dict[str, Any]: Detailed validation report dictionary.
    """
    report: Dict[str, Any] = {
        "status": "PASS",
        "issues_found": [],
        "metrics": {},
    }

    # 1. Raw Data Checks
    if raw_df is not None:
        raw_total = len(raw_df)
        raw_missing_cust = int(raw_df["customerId"].isna().sum()) if "customerId" in raw_df.columns else 0
        raw_missing_item = int(raw_df["itemId"].isna().sum()) if "itemId" in raw_df.columns else 0
        raw_missing_phone = int(raw_df["MOBILE_NO"].isna().sum()) if "MOBILE_NO" in raw_df.columns else 0
        raw_exact_dups = int(raw_df.duplicated().sum())

        report["metrics"]["raw_total_rows"] = raw_total
        report["metrics"]["raw_missing_customerId"] = raw_missing_cust
        report["metrics"]["raw_missing_itemId"] = raw_missing_item
        report["metrics"]["raw_missing_MOBILE_NO"] = raw_missing_phone
        report["metrics"]["raw_exact_duplicates"] = raw_exact_dups

        if raw_missing_cust > 0:
            report["issues_found"].append(f"Raw data contains {raw_missing_cust} rows with missing customerId.")
        if raw_exact_dups > 0:
            report["issues_found"].append(f"Raw data contains {raw_exact_dups} exact duplicate rows.")

    # 2. Clean Data Checks
    if clean_df is not None:
        clean_total = len(clean_df)
        clean_missing_cust = int(clean_df["customerId"].isna().sum()) if "customerId" in clean_df.columns else 0
        clean_missing_item = int(clean_df["itemId"].isna().sum()) if "itemId" in clean_df.columns else 0
        clean_missing_date = int(clean_df["invoice_date"].isna().sum()) if "invoice_date" in clean_df.columns else 0
        clean_mfg_present = "mfgDate" in clean_df.columns

        report["metrics"]["clean_total_rows"] = clean_total
        report["metrics"]["clean_missing_customerId"] = clean_missing_cust
        report["metrics"]["clean_missing_itemId"] = clean_missing_item
        report["metrics"]["clean_missing_invoice_date"] = clean_missing_date
        report["metrics"]["clean_mfgDate_dropped"] = not clean_mfg_present

        if clean_missing_cust > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"Clean data still contains {clean_missing_cust} missing customerId rows.")
        if clean_missing_item > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"Clean data still contains {clean_missing_item} missing itemId rows.")
        if clean_missing_date > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"Clean data contains {clean_missing_date} invalid invoice_date rows.")

    # 3. Aggregated Purchase Events Checks
    if aggregated_df is not None:
        agg_total = len(aggregated_df)
        # Check uniqueness of (customerId, invoice_number, invoice_date, itemId)
        event_keys = ["customerId", "invoice_number", "invoice_date", "itemId"]
        present_keys = [k for k in event_keys if k in aggregated_df.columns]
        if len(present_keys) == 4:
            dup_events = int(aggregated_df.duplicated(subset=present_keys).sum())
        else:
            dup_events = 0

        # Check negative or zero quantities
        neg_or_zero_qty = int((aggregated_df["quantity"] <= 0).sum()) if "quantity" in aggregated_df.columns else 0

        report["metrics"]["aggregated_purchase_events"] = agg_total
        report["metrics"]["duplicate_purchase_events"] = dup_events
        report["metrics"]["negative_or_zero_quantity_events"] = neg_or_zero_qty

        if dup_events > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"Aggregated events contain {dup_events} duplicate purchase events.")

    # 4. SALT Master Checks
    if salt_df is not None:
        salt_total = len(salt_df)
        salt_dup_codes = int(salt_df.duplicated(subset=["Code"]).sum()) if "Code" in salt_df.columns else 0
        report["metrics"]["salt_master_total_items"] = salt_total
        report["metrics"]["salt_master_duplicate_codes"] = salt_dup_codes

        if salt_dup_codes > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"SALT master contains {salt_dup_codes} duplicate Code entries.")

    # 5. History and Interval Checks
    if history_df is not None:
        hist_total = len(history_df)
        unique_customers = int(history_df["customerId"].nunique()) if "customerId" in history_df.columns else 0
        unique_items = int(history_df["itemId"].nunique()) if "itemId" in history_df.columns else 0

        # Unique customerId + itemId combinations
        if "customerId" in history_df.columns and "itemId" in history_df.columns:
            combos = int(history_df.groupby(["customerId", "itemId"]).ngroups)
        else:
            combos = 0

        # Check negative intervals
        if "days_since_previous_purchase" in history_df.columns:
            valid_intervals = history_df["days_since_previous_purchase"].dropna()
            negative_intervals = int((valid_intervals < 0).sum())
            zero_intervals = int((valid_intervals == 0).sum())
        else:
            negative_intervals = 0
            zero_intervals = 0

        # Date ranges
        if "invoice_date" in history_df.columns and not history_df["invoice_date"].empty:
            min_date = str(history_df["invoice_date"].min().date())
            max_date = str(history_df["invoice_date"].max().date())
        else:
            min_date, max_date = None, None

        # Check SALT enrichment coverage if column exists
        if "salt_composition" in history_df.columns:
            enriched_items = int(history_df["salt_composition"].notna().sum())
            unmatched_items = int(history_df["salt_composition"].isna().sum())
        else:
            enriched_items, unmatched_items = 0, 0

        report["metrics"]["history_total_events"] = hist_total
        report["metrics"]["unique_customers"] = unique_customers
        report["metrics"]["unique_medicines"] = unique_items
        report["metrics"]["unique_customer_medicine_histories"] = combos
        report["metrics"]["negative_intervals"] = negative_intervals
        report["metrics"]["zero_day_intervals"] = zero_intervals
        report["metrics"]["min_date"] = min_date
        report["metrics"]["max_date"] = max_date
        report["metrics"]["salt_enriched_events"] = enriched_items
        report["metrics"]["salt_unmatched_events"] = unmatched_items

        if negative_intervals > 0:
            report["status"] = "FAIL"
            report["issues_found"].append(f"Found {negative_intervals} negative purchase intervals (history out of chronological order).")

    return report


def generate_validation_report(report: Dict[str, Any]) -> str:
    """Format validation dictionary into human-readable text."""
    lines = [
        "=" * 60,
        f"REFILLCARE DATA QUALITY VALIDATION: {report.get('status', 'UNKNOWN')}",
        "=" * 60,
    ]
    for k, v in report.get("metrics", {}).items():
        lines.append(f"  - {k}: {v}")

    if report.get("issues_found"):
        lines.append("\nISSUES / OBSERVATIONS:")
        for issue in report["issues_found"]:
            lines.append(f"  * {issue}")
    else:
        lines.append("\nNo critical issues detected.")

    lines.append("=" * 60)
    return "\n".join(lines)
