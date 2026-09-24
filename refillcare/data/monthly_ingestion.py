"""Production monthly sales ingestion and prediction pipeline for RefillCare.

Executes the standardized production lifecycle when receiving a new month's sales dataset:
1. Validate incoming sales transactions.
2. Append new transactions to existing multi-year purchase history (NEVER replacing it).
3. Deduplicate transaction events.
4. Reconstruct chronological customer + medication history timelines and intervals.
5. Match pending historical prediction snapshots against newly arrived actual purchase events.
6. Calculate refill predictions & 2-day reminder dates for active customer-medication pairs.
7. Store new prediction snapshots in append-only storage (unseen production data).
8. Prepare updated reminder schedules and export datasets.
"""

import io
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple, Optional, List, Union
import pandas as pd
import numpy as np

from refillcare.data.dates import parse_pharmacy_dates, format_date_dd_mm_yyyy, UI_DATE_FORMAT
from refillcare.data.history import create_purchase_history
from refillcare.data.packing import enrich_total_units_purchased
from refillcare.features.engineering import build_feature_dataset
from refillcare.models.prediction import generate_batch_predictions
from refillcare.models.supply_hybrid import calculate_hybrid_refill_date
from reminder.scheduler import evaluate_refill_eligibility, RefillReminderScheduler
from reminder.storage import RefillCareStorage


def normalize_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary CSV/Excel column names into RefillCare canonical schema."""
    if df.empty:
        return df.copy()

    col_map: Dict[str, str] = {}
    for c in df.columns:
        c_low = str(c).strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace(".", "")
        if c_low in ("invoicedate", "date", "billdate", "orderdate", "transactiondate", "salestime", "trndate"):
            col_map[c] = "invoice_date"
        elif c_low in ("trnno", "trnnbr", "transactionno", "invoicenumber", "invoiceno", "billnumber", "billno", "invoice"):
            col_map[c] = "invoice_number"
        elif c_low in ("partycode", "patientcode", "customerid", "customer", "patientid", "custid"):
            col_map[c] = "customerId"
        elif c_low in ("partyname", "customername", "patientname", "custname", "party", "patient"):
            col_map[c] = "customerName"
        elif c_low in ("itemcode", "itemid", "medicineid", "drugid", "productid", "item"):
            col_map[c] = "itemId"
        elif c_low in ("itemname", "medicinename", "drugname", "productname", "itemdescription", "medicine"):
            col_map[c] = "itemName"
        elif c_low in ("quantity", "qty", "units", "quantitysold", "qtyordered"):
            col_map[c] = "quantity"
        elif c_low in ("packing", "pack", "packsize", "packaging"):
            col_map[c] = "packing"
        elif c_low in ("mobileno", "mobile", "phone", "phonenumber", "contact", "cell", "contactno"):
            col_map[c] = "MOBILE_NO"
        elif c_low in ("saltcomposition", "salt", "composition", "genericname"):
            col_map[c] = "salt_composition"

    renamed = df.rename(columns=col_map).copy()

    # If invoice_number is missing, synthesize a deterministic fallback
    if "invoice_number" not in renamed.columns:
        renamed["invoice_number"] = [f"INV_{i+1:06d}" for i in range(len(renamed))]

    return renamed


def validate_monthly_sales_data(
    sales_input: Union[pd.DataFrame, Any],
    existing_history_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Validate incoming sales transactions and extract business metrics.

    Args:
        sales_input: DataFrame or file-like object containing sales data.
        existing_history_df: Optional current historical purchase DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - valid_transactions_df: Clean records meeting operational criteria.
            - review_records_df: Flagged records requiring review.
            - business_metrics: High-level operational summary metrics.
    """
    if isinstance(sales_input, pd.DataFrame):
        df = sales_input.copy()
    else:
        try:
            filename = getattr(sales_input, "name", "sales.csv").lower()
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(sales_input)
            else:
                df = pd.read_csv(sales_input)
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), {"error": f"Failed to read sales input: {str(e)}"}

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "file_date_range": "N/A (Empty)",
            "records_received": 0,
            "valid_records": 0,
            "records_requiring_review": 0,
            "new_customers": 0,
            "existing_customers": 0,
            "unique_customers": 0,
        }

    renamed = normalize_sales_dataframe(df)
    total_received = len(renamed)

    # Date range parsing using strict pharmacy date parsing
    date_diagnostics: Dict[str, Any] = {}
    if "invoice_date" in renamed.columns:
        parsed_dates, date_meta = parse_pharmacy_dates(renamed["invoice_date"])
        renamed["invoice_date"] = parsed_dates
        date_range_str = date_meta["date_range_formatted"]
        date_diagnostics = date_meta
    else:
        date_range_str = "Missing Date Column"
        date_diagnostics = {
            "source_format": "Missing",
            "date_range_formatted": "Missing Date Column",
            "min_date_formatted": "-",
            "max_date_formatted": "-",
            "sample_dates_formatted": [],
            "valid_count": 0,
            "invalid_count": len(renamed),
            "is_ambiguous": True,
            "warning": "No invoice_date column found in file.",
        }

    # Validation criteria:
    # 1. Non-null, parseable invoice_date
    # 2. Non-empty customerId
    # 3. Non-empty itemId
    # 4. Numeric quantity > 0
    valid_mask = pd.Series(True, index=renamed.index)

    if "invoice_date" in renamed.columns:
        valid_mask &= renamed["invoice_date"].notna()
    else:
        valid_mask = pd.Series(False, index=renamed.index)

    if "customerId" in renamed.columns:
        valid_mask &= renamed["customerId"].notna() & (renamed["customerId"].astype(str).str.strip() != "")
    elif "customerName" in renamed.columns:
        renamed["customerId"] = renamed["customerName"]
        valid_mask &= renamed["customerId"].notna() & (renamed["customerId"].astype(str).str.strip() != "")
    else:
        valid_mask = pd.Series(False, index=renamed.index)

    if "itemId" in renamed.columns:
        valid_mask &= renamed["itemId"].notna() & (renamed["itemId"].astype(str).str.strip() != "")
    elif "itemName" in renamed.columns:
        renamed["itemId"] = renamed["itemName"]
        valid_mask &= renamed["itemId"].notna() & (renamed["itemId"].astype(str).str.strip() != "")
    else:
        valid_mask = pd.Series(False, index=renamed.index)

    if "quantity" in renamed.columns:
        qty_num = pd.to_numeric(renamed["quantity"], errors="coerce")
        valid_mask &= qty_num.notna() & (qty_num > 0)

    valid_df = renamed[valid_mask].copy().reset_index(drop=True)
    review_df = renamed[~valid_mask].copy().reset_index(drop=True)

    # Customer breakdown
    cust_col = "customerId" if "customerId" in renamed.columns else "customerName"
    if cust_col in renamed.columns:
        uploaded_cust_set = set(renamed[cust_col].dropna().astype(str).str.strip().unique())
        uploaded_cust_set.discard("")
        unique_customers = len(uploaded_cust_set)

        if existing_history_df is not None and not existing_history_df.empty:
            exist_cust_col = "customerId" if "customerId" in existing_history_df.columns else "customerName"
            if exist_cust_col in existing_history_df.columns:
                existing_cust_set = set(existing_history_df[exist_cust_col].dropna().astype(str).str.strip().unique())
                new_customers = len(uploaded_cust_set - existing_cust_set)
                existing_customers = len(uploaded_cust_set & existing_cust_set)
            else:
                new_customers = unique_customers
                existing_customers = 0
        else:
            new_customers = unique_customers
            existing_customers = 0
    else:
        unique_customers = 0
        new_customers = 0
        existing_customers = 0

    metrics = {
        "file_date_range": date_range_str,
        "records_received": total_received,
        "valid_records": len(valid_df),
        "records_requiring_review": len(review_df),
        "new_customers": new_customers,
        "existing_customers": existing_customers,
        "unique_customers": unique_customers,
        "date_diagnostics": date_diagnostics,
    }

    return valid_df, review_df, metrics


def determine_mobile_status(phone: Any) -> str:
    """Determine client-facing mobile status without removing any customer record."""
    if phone is None or pd.isna(phone):
        return "Missing"
    s = str(phone).strip()
    if not s or s.lower() in ("nan", "none", "-", "null", "n/a", "0", ""):
        return "Missing"
    digits = "".join(filter(str.isdigit, s))
    if len(digits) == 10 and digits[0] in "6789":
        return "Valid"
    elif len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return "Valid"
    elif len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return "Valid"
    return "Invalid format"


def process_monthly_sales_data(
    sales_input: Union[pd.DataFrame, Any],
    existing_history_df: Optional[pd.DataFrame] = None,
    storage: Optional[RefillCareStorage] = None,
    source_filename: str = "uploaded_sales.xlsx",
) -> Dict[str, Any]:
    """Process incoming monthly sales data with batch traceability without mutating unselected history.

    1. Validate uploaded records.
    2. Normalize them into the existing RefillCare transaction schema.
    3. Generate a traceable import_batch_id.
    4. Detect duplicate transactions (intra-file and against existing history).
    5. Separate valid and review-required records.
    6. Append valid new transactions with import_batch_id to existing historical data.
    7. Preserve existing records and customer info (including missing mobile numbers).
    8. Update the purchase-history data used by prediction.
    9. Record the latest available sales date formatted as DD-MM-YYYY.

    Args:
        sales_input: Incoming sales DataFrame or file-like object.
        existing_history_df: Optional current canonical historical purchase DataFrame.
        storage: Optional RefillCareStorage instance for batch logging.
        source_filename: Source file name for audit traceability.

    Returns:
        Dict[str, Any] containing processing metrics, import_batch_id, and updated_history_df.
    """
    valid_new_df, review_df, val_metrics = validate_monthly_sales_data(sales_input, existing_history_df)
    records_requiring_review = len(review_df)

    if valid_new_df.empty:
        latest_date_str = "N/A"
        if existing_history_df is not None and not existing_history_df.empty and "invoice_date" in existing_history_df.columns:
            parsed_d = pd.to_datetime(existing_history_df["invoice_date"], errors="coerce").dropna()
            if not parsed_d.empty:
                latest_date_str = parsed_d.max().strftime(UI_DATE_FORMAT)

        return {
            "status": "warning",
            "message": "No valid records found in uploaded sales file.",
            "import_batch_id": None,
            "updated_history_df": existing_history_df if existing_history_df is not None else pd.DataFrame(),
            "metrics": {
                "import_batch_id": None,
                "new_records_added": 0,
                "existing_duplicates_skipped": 0,
                "customers_updated": 0,
                "new_customers": 0,
                "new_medicines": 0,
                "latest_sales_date": latest_date_str,
                "records_requiring_review": records_requiring_review,
            },
            "valid_records_df": valid_new_df,
            "review_records_df": review_df,
        }

    # Generate unique traceable import_batch_id
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import_batch_id = f"BATCH_{batch_timestamp}_{uuid.uuid4().hex[:6]}"

    # Deduplicate within incoming records
    dedup_cols = ["customerId", "itemId", "invoice_date"]
    if "quantity" in valid_new_df.columns:
        dedup_cols.append("quantity")
    if "invoice_number" in valid_new_df.columns and not valid_new_df["invoice_number"].astype(str).str.startswith("INV_").all():
        dedup_cols.append("invoice_number")

    valid_clean = valid_new_df.drop_duplicates(subset=dedup_cols, keep="first").copy()
    intra_file_dupes = len(valid_new_df) - len(valid_clean)

    # Detect duplicates against existing history
    if existing_history_df is not None and not existing_history_df.empty:
        exist_df = existing_history_df.copy()
        exist_df["_dt"] = pd.to_datetime(exist_df["invoice_date"], errors="coerce")
        valid_clean["_dt"] = pd.to_datetime(valid_clean["invoice_date"], errors="coerce")

        exist_cids = exist_df["customerId"].dropna().astype(str).str.strip().str.lower().tolist()
        exist_iids = exist_df["itemId"].dropna().astype(str).str.strip().str.lower().tolist()
        exist_dts = exist_df["_dt"].dt.strftime("%Y-%m-%d").fillna("").tolist()

        exist_keys = set(zip(exist_cids, exist_iids, exist_dts))

        is_existing_mask = valid_clean.apply(
            lambda r: (
                str(r.get("customerId", "")).strip().lower(),
                str(r.get("itemId", "")).strip().lower(),
                r["_dt"].strftime("%Y-%m-%d") if pd.notna(r["_dt"]) else "",
            ) in exist_keys,
            axis=1
        )

        existing_dupes_in_upload = int(is_existing_mask.sum())
        records_to_append = valid_clean[~is_existing_mask].drop(columns=["_dt"], errors="ignore").copy()
        existing_duplicates_skipped = intra_file_dupes + existing_dupes_in_upload
    else:
        records_to_append = valid_clean.drop(columns=["_dt"], errors="ignore").copy() if "_dt" in valid_clean.columns else valid_clean.copy()
        existing_duplicates_skipped = intra_file_dupes

    new_records_added = len(records_to_append)

    # Tag records with import_batch_id
    records_to_append["import_batch_id"] = import_batch_id

    # Compute customer and medicine changes
    if existing_history_df is not None and not existing_history_df.empty:
        exist_cust_set = set(existing_history_df["customerId"].dropna().astype(str).str.strip().str.lower().unique()) - {"", "nan", "none"}
        exist_med_set = set(existing_history_df["itemId"].dropna().astype(str).str.strip().str.lower().unique()) - {"", "nan", "none"}
    else:
        exist_cust_set = set()
        exist_med_set = set()

    if not records_to_append.empty:
        appended_cust_set = set(records_to_append["customerId"].dropna().astype(str).str.strip().str.lower().unique()) - {"", "nan", "none"}
        appended_med_set = set(records_to_append["itemId"].dropna().astype(str).str.strip().str.lower().unique()) - {"", "nan", "none"}
    else:
        appended_cust_set = set()
        appended_med_set = set()

    customers_updated = len(appended_cust_set & exist_cust_set)
    new_customers = len(appended_cust_set - exist_cust_set)
    new_medicines = len(appended_med_set - exist_med_set)

    import gc
    gc.collect()

    # Append valid new records to existing history (NEVER overwrite existing records)
    if existing_history_df is not None and not existing_history_df.empty:
        if "import_batch_id" not in existing_history_df.columns:
            existing_history_df["import_batch_id"] = "LEGACY_HISTORY"
        combined_raw = pd.concat([existing_history_df, records_to_append], ignore_index=True)
    else:
        combined_raw = records_to_append

    # Normalize invoice_date & reconstruct canonical chronological purchase history
    combined_raw["invoice_date"] = pd.to_datetime(combined_raw["invoice_date"], errors="coerce")
    valid_mask = combined_raw["invoice_date"].notna()
    if not valid_mask.all():
        combined_raw = combined_raw[valid_mask]

    gc.collect()
    updated_history_df = create_purchase_history(combined_raw)
    if "packing" in updated_history_df.columns:
        updated_history_df = enrich_total_units_purchased(updated_history_df)
    gc.collect()

    # Determine latest available sales date in DD-MM-YYYY format
    if not updated_history_df.empty and "invoice_date" in updated_history_df.columns:
        parsed_dates = pd.to_datetime(updated_history_df["invoice_date"], errors="coerce").dropna()
        latest_sales_date_str = parsed_dates.max().strftime(UI_DATE_FORMAT) if not parsed_dates.empty else "N/A"
    else:
        latest_sales_date_str = "N/A"

    batch_metrics = {
        "import_batch_id": import_batch_id,
        "new_records_added": new_records_added,
        "existing_duplicates_skipped": existing_duplicates_skipped,
        "customers_updated": customers_updated,
        "new_customers": new_customers,
        "new_medicines": new_medicines,
        "latest_sales_date": latest_sales_date_str,
        "records_requiring_review": records_requiring_review,
        "detected_date_range": val_metrics.get("file_date_range", "-"),
    }

    # Save import batch metadata in storage if available
    if storage is not None:
        storage.save_import_batch({
            "import_batch_id": import_batch_id,
            "upload_timestamp": datetime.now().isoformat(),
            "source_filename": source_filename,
            "detected_date_range": val_metrics.get("file_date_range", "-"),
            "record_count": val_metrics.get("records_received", 0),
            "records_inserted": new_records_added,
            "records_skipped": existing_duplicates_skipped,
            "records_requiring_review": records_requiring_review,
            "processing_status": "completed",
            "created_at": datetime.now().isoformat(),
            "is_active": 1,
        })

    return {
        "status": "success",
        "message": "Sales Data Updated Successfully",
        "import_batch_id": import_batch_id,
        "updated_history_df": updated_history_df,
        "metrics": batch_metrics,
        "valid_records_df": valid_new_df,
        "review_records_df": review_df,
    }


def rollback_monthly_sales_import(
    current_history_df: pd.DataFrame,
    import_batch_id: str,
    storage: Optional[RefillCareStorage] = None,
) -> Dict[str, Any]:
    """Safely undo a sales data import by removing records associated with import_batch_id.

    Workflow:
    1. Filter out all transactions matching import_batch_id.
    2. Leave older historical data (LEGACY_HISTORY and other batches) 100% untouched.
    3. Reconstruct canonical chronological purchase intervals from the remaining records.
    4. Invalidate / remove prediction snapshots linked to this import batch.
    5. Recalculate restored latest sales date in DD-MM-YYYY format.

    Args:
        current_history_df: Current working purchase history DataFrame.
        import_batch_id: The unique identifier of the batch to rollback.
        storage: Optional RefillCareStorage instance to deactivate batch in SQLite.

    Returns:
        Dict[str, Any] containing restored_history_df, records_removed, and restored_latest_sales_date.
    """
    if current_history_df is None or current_history_df.empty:
        return {
            "status": "warning",
            "message": "History DataFrame is empty; nothing to rollback.",
            "records_removed": 0,
            "restored_latest_sales_date": "N/A",
            "restored_history_df": current_history_df,
        }

    if "import_batch_id" not in current_history_df.columns:
        return {
            "status": "warning",
            "message": "Current history data does not contain batch identifiers for rollback.",
            "records_removed": 0,
            "restored_latest_sales_date": "N/A",
            "restored_history_df": current_history_df,
        }

    # Identify records belonging to the target batch
    batch_mask = current_history_df["import_batch_id"].astype(str) == str(import_batch_id)
    records_to_remove = int(batch_mask.sum())

    if records_to_remove == 0:
        return {
            "status": "warning",
            "message": f"No records found matching import batch ID '{import_batch_id}'.",
            "records_removed": 0,
            "restored_latest_sales_date": format_date_dd_mm_yyyy(current_history_df["invoice_date"].max()),
            "restored_history_df": current_history_df,
        }

    # Retain strictly the unaffected transactions
    remaining_df = current_history_df[~batch_mask].copy().reset_index(drop=True)

    # Reconstruct canonical purchase history from remaining transactions
    restored_history_df = create_purchase_history(remaining_df)
    if "packing" in restored_history_df.columns:
        restored_history_df = enrich_total_units_purchased(restored_history_df)

    # Determine restored latest sales date
    if not restored_history_df.empty and "invoice_date" in restored_history_df.columns:
        parsed_dates = pd.to_datetime(restored_history_df["invoice_date"], errors="coerce").dropna()
        restored_latest_date_str = parsed_dates.max().strftime(UI_DATE_FORMAT) if not parsed_dates.empty else "N/A"
    else:
        restored_latest_date_str = "N/A"

    # Deactivate batch and invalidate prediction snapshots in SQLite storage
    storage_res: Dict[str, Any] = {}
    if storage is not None:
        storage_res = storage.rollback_import_batch(import_batch_id)

    return {
        "status": "success",
        "message": f"Successfully rolled back import batch '{import_batch_id}'. Removed {records_to_remove:,} records.",
        "import_batch_id": import_batch_id,
        "records_removed": records_to_remove,
        "restored_latest_sales_date": restored_latest_date_str,
        "restored_history_df": restored_history_df,
        "storage_rollback": storage_res,
    }

    return {
        "status": "success",
        "message": "Sales Data Updated Successfully",
        "updated_history_df": updated_history_df,
        "metrics": {
            "new_records_added": new_records_added,
            "existing_duplicates_skipped": existing_duplicates_skipped,
            "customers_updated": customers_updated,
            "new_customers": new_customers,
            "new_medicines": new_medicines,
            "latest_sales_date": latest_sales_date_str,
            "records_requiring_review": records_requiring_review,
        },
        "valid_records_df": valid_new_df,
        "review_records_df": review_df,
    }


def generate_updated_predictions(
    history_df: pd.DataFrame,
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute client-facing refill prediction generation using existing trained engine and historical sales data.

    Flow:
    Latest historical data
            ↓
    Updated customer/medicine history
            ↓
    Existing prediction engine
            ↓
    Expected refill date
            ↓
    Reminder eligibility
            ↓
    Reminder candidate list

    Args:
        history_df: Canonical historical purchase DataFrame.
        model_bundle: Trained RefillCare prediction model bundle.

    Returns:
        Dict[str, Any] containing eligible_df, ineligible_df, and the 5 client summary metrics.
    """
    if history_df is None or history_df.empty or model_bundle is None:
        return {
            "status": "warning",
            "message": "Historical data or model bundle is unavailable.",
            "eligible_df": pd.DataFrame(),
            "ineligible_df": pd.DataFrame(),
            "metrics": {
                "customers_evaluated": 0,
                "predictions_generated": 0,
                "customers_requiring_review": 0,
                "customers_not_eligible": 0,
                "missing_mobile_predictions": 0,
            },
        }

    working_hist = history_df.copy()
    if "invoice_number" not in working_hist.columns:
        working_hist["invoice_number"] = [f"INV_{i+1:06d}" for i in range(len(working_hist))]

    # Normalize chronological timeline & enrich physical units
    working_hist = create_purchase_history(working_hist)
    if "packing" in working_hist.columns:
        working_hist = enrich_total_units_purchased(working_hist)

    # Build feature dataset from purchase history
    feature_df = build_feature_dataset(working_hist)

    feature_df["invoice_date"] = pd.to_datetime(feature_df["invoice_date"], errors="coerce")
    valid_features = feature_df[feature_df["invoice_date"].notna()].copy()

    # Isolate latest purchase event per customer + medicine
    latest_events = valid_features.sort_values("invoice_date").groupby(["customerId", "itemId"], as_index=False).last()

    p_counts = latest_events.get("purchase_count_so_far", latest_events.get("purchase_seq", 1))
    is_cold_start = p_counts.fillna(1).astype(int) < 2

    raw_candidates = latest_events[~is_cold_start].copy()
    raw_cold_start = latest_events[is_cold_start].copy()

    # Existing prediction engine execution
    if not raw_candidates.empty:
        preds_df = generate_batch_predictions(model_bundle, raw_candidates)
    else:
        preds_df = raw_candidates.copy()

    # Evaluate reminder eligibility
    eligible_rows: List[Dict[str, Any]] = []
    ineligible_rows: List[Dict[str, Any]] = []

    for _, row in preds_df.iterrows():
        rec_dict = row.to_dict()
        p_cnt = int(rec_dict.get("purchase_count_so_far", rec_dict.get("purchase_seq", 1)))
        elig = evaluate_refill_eligibility(rec_dict, min_purchase_count=2)

        if elig["is_eligible"]:
            rec_dict["history_quality"] = elig["history_quality"]
            rec_dict["prediction_source"] = rec_dict.get("prediction_source", "hybrid_supply")
            eligible_rows.append(rec_dict)
        elif p_cnt >= 2:
            # Fallback for multi-purchase histories where supply or historical interval is available
            dos_cand = rec_dict.get("estimated_days_of_supply")
            med_cand = rec_dict.get("historical_interval_median")
            cand_interval = dos_cand if (dos_cand is not None and not pd.isna(dos_cand) and dos_cand > 0) else med_cand

            if cand_interval is not None and not pd.isna(cand_interval) and cand_interval > 0:
                rec_dict["predicted_days_until_refill"] = float(cand_interval)
                rec_dict["prediction_source"] = "estimated_days_of_supply" if dos_cand else "historical_interval_median"
                rec_dict["history_quality"] = "medium_history" if p_cnt >= 3 else "low_history"
                rec_dict["Pilot Tier"] = "Tier B (Review Required)" if p_cnt >= 3 else "Tier C (Suppressed / Low History)"
                eligible_rows.append(rec_dict)
            else:
                rec_dict["Reason for Ineligibility"] = elig["reason"]
                ineligible_rows.append(rec_dict)
        else:
            rec_dict["Reason for Ineligibility"] = elig["reason"]
            ineligible_rows.append(rec_dict)

    # Format eligible DataFrame
    if eligible_rows:
        el_raw = pd.DataFrame(eligible_rows)
        eligible_df = pd.DataFrame()
        eligible_df["customerId"] = el_raw["customerId"].astype(str)
        eligible_df["itemId"] = el_raw["itemId"].astype(str)
        eligible_df["Customer Name"] = el_raw.get("customerName", el_raw["customerId"]).fillna("Unknown Customer")
        eligible_df["Medicine"] = el_raw.get("itemName", el_raw["itemId"]).fillna("Unknown Medicine")
        eligible_df["Medication"] = eligible_df["Medicine"]
        eligible_df["Delivery Phone"] = el_raw.get("MOBILE_NO", "").fillna("").astype(str).str.strip()
        eligible_df["Mobile Number"] = eligible_df["Delivery Phone"]
        eligible_df["Mobile Status"] = eligible_df["Mobile Number"].apply(determine_mobile_status)

        last_dt = pd.to_datetime(el_raw["invoice_date"], errors="coerce")
        eligible_df["Last Purchase Date"] = last_dt.dt.strftime("%Y-%m-%d").fillna("-")
        eligible_df["Purchase Count"] = el_raw.get("purchase_count_so_far", 1).astype(int)

        if "estimated_days_of_supply" in el_raw.columns and el_raw["estimated_days_of_supply"].notna().any():
            dos_series = pd.to_numeric(el_raw["estimated_days_of_supply"], errors="coerce")
            pred_days = pd.to_numeric(el_raw["predicted_days_until_refill"], errors="coerce")
            final_dos = dos_series.fillna(pred_days).round(1)
        else:
            final_dos = pd.to_numeric(el_raw["predicted_days_until_refill"], errors="coerce").round(1)

        eligible_df["Estimated Days of Supply"] = final_dos
        eligible_df["Predicted Interval (Days)"] = final_dos
        eligible_df["Estimated Daily Consumption"] = el_raw.get("estimated_daily_consumption", el_raw.get("historical_consumption_rate", np.nan))
        eligible_df["estimated_daily_consumption"] = eligible_df["Estimated Daily Consumption"]

        exp_dates = []
        rem_dates = []
        for l_dt, dos in zip(last_dt, final_dos):
            if pd.notna(l_dt) and pd.notna(dos) and dos > 0:
                e_d = l_dt + timedelta(days=int(round(dos)))
                buf_days = max(1, int(round(dos - 2.0)))
                r_d = l_dt + timedelta(days=buf_days)
                exp_dates.append(e_d.strftime("%Y-%m-%d"))
                rem_dates.append(r_d.strftime("%Y-%m-%d"))
            else:
                exp_dates.append("-")
                rem_dates.append("-")

        eligible_df["Expected Refill Date"] = exp_dates
        eligible_df["Reminder Date"] = rem_dates
        eligible_df["History Quality"] = el_raw.get("history_quality", "medium_history")

        # Scheduling / export aliases
        eligible_df["MOBILE_NO"] = eligible_df["Delivery Phone"]
        eligible_df["invoice_date"] = eligible_df["Last Purchase Date"]
        eligible_df["latest_purchase_date"] = eligible_df["Last Purchase Date"]
        eligible_df["predicted_days_until_refill"] = eligible_df["Estimated Days of Supply"]
        eligible_df["expected_refill_date"] = eligible_df["Expected Refill Date"]
        eligible_df["reminder_date"] = eligible_df["Reminder Date"]
        eligible_df["raw_reminder_date"] = eligible_df["Reminder Date"]
        eligible_df["customerName"] = eligible_df["Customer Name"]
        eligible_df["itemName"] = eligible_df["Medicine"]
        eligible_df["purchase_count_so_far"] = eligible_df["Purchase Count"]
        eligible_df["estimated_days_of_supply"] = eligible_df["Estimated Days of Supply"]
        eligible_df["mobile_status"] = eligible_df["Mobile Status"]
    else:
        eligible_df = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Delivery Phone",
            "Mobile Number", "Mobile Status", "Last Purchase Date", "Purchase Count",
            "Estimated Days of Supply", "Predicted Interval (Days)", "Expected Refill Date",
            "Reminder Date", "History Quality", "MOBILE_NO", "invoice_date",
            "latest_purchase_date", "predicted_days_until_refill", "expected_refill_date",
            "reminder_date", "raw_reminder_date", "customerName", "itemName",
            "purchase_count_so_far", "estimated_days_of_supply", "mobile_status"
        ])

    # Format ineligible DataFrame
    ineligible_list = []
    for _, r in raw_cold_start.iterrows():
        p_val = str(r.get("MOBILE_NO", "")).strip()
        m_val = str(r.get("itemName", r.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r.get("customerId", "")),
            "itemId": str(r.get("itemId", "")),
            "Customer Name": str(r.get("customerName", r.get("customerId", "Unknown"))),
            "Medicine": m_val,
            "Medication": m_val,
            "Delivery Phone": p_val,
            "Mobile Number": p_val,
            "Mobile Status": determine_mobile_status(p_val),
            "Last Purchase Date": pd.to_datetime(r.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r.get("invoice_date")) else "-",
            "Purchase Count": int(r.get("purchase_count_so_far", r.get("purchase_seq", 1))),
            "Reason for Ineligibility": "Cold-start history: purchase count is 1 (< 2).",
        })

    for r_dict in ineligible_rows:
        p_val = str(r_dict.get("MOBILE_NO", "")).strip()
        m_val = str(r_dict.get("itemName", r_dict.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r_dict.get("customerId", "")),
            "itemId": str(r_dict.get("itemId", "")),
            "Customer Name": str(r_dict.get("customerName", r_dict.get("customerId", "Unknown"))),
            "Medicine": m_val,
            "Medication": m_val,
            "Delivery Phone": p_val,
            "Mobile Number": p_val,
            "Mobile Status": determine_mobile_status(p_val),
            "Last Purchase Date": pd.to_datetime(r_dict.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r_dict.get("invoice_date")) else "-",
            "Purchase Count": int(r_dict.get("purchase_count_so_far", r_dict.get("purchase_seq", 1))),
            "Reason for Ineligibility": str(r_dict.get("Reason for Ineligibility", "Invalid prediction")),
        })

    if ineligible_list:
        ineligible_df = pd.DataFrame(ineligible_list)
    else:
        ineligible_df = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Delivery Phone",
            "Mobile Number", "Mobile Status", "Last Purchase Date", "Purchase Count",
            "Reason for Ineligibility"
        ])

    # Extract the 5 client summary metrics
    customers_evaluated = int(working_hist["customerId"].dropna().astype(str).str.strip().nunique())
    predictions_generated = len(eligible_df)
    customers_not_eligible = len(raw_cold_start)
    customers_requiring_review = len(ineligible_rows)
    missing_mobile_predictions = int((eligible_df["Mobile Status"] != "Valid").sum()) if not eligible_df.empty and "Mobile Status" in eligible_df.columns else 0

    metrics = {
        "customers_evaluated": customers_evaluated,
        "predictions_generated": predictions_generated,
        "customers_requiring_review": customers_requiring_review,
        "customers_not_eligible": customers_not_eligible,
        "missing_mobile_predictions": missing_mobile_predictions,
    }

    return {
        "status": "success",
        "eligible_df": eligible_df,
        "ineligible_df": ineligible_df,
        "metrics": metrics,
    }


def ingest_monthly_sales_pipeline(
    new_sales: Union[pd.DataFrame, Any],
    existing_history_df: pd.DataFrame,
    model_bundle: Dict[str, Any],
    storage: Optional[RefillCareStorage] = None,
    prediction_date: Optional[Union[str, date]] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end production workflow for receiving a new month's sales data.

    Workflow:
    1. Validate incoming sales file.
    2. Add new transactions to existing multi-year purchase history (never replacing it).
    3. Deduplicate transaction events.
    4. Reconstruct customer + medicine history timelines and calculate refill intervals.
    5. Match pending historical prediction snapshots against newly arrived actual purchase dates.
    6. Calculate refill predictions & 2-day reminder dates for active customer-medication pairs.
    7. Store new prediction snapshots in append-only storage (unseen production data).
    8. Return updated datasets and reminder lists ready for operations and CSV download.

    Args:
        new_sales: Incoming sales DataFrame or file object.
        existing_history_df: Canonical historical purchase DataFrame (5.8-year history).
        model_bundle: Trained RefillCare prediction model bundle.
        storage: Optional RefillCareStorage instance for snapshot persistence.
        prediction_date: Date to record on new prediction snapshots (defaults to today).

    Returns:
        Dict[str, Any] containing:
            - combined_history_df
            - eligible_df
            - ineligible_df
            - reminder_schedule_df
            - business_metrics
            - outcome_evaluation_metrics
            - new_snapshots_saved
    """
    if prediction_date is None:
        pred_date_str = date.today().isoformat()
    elif isinstance(prediction_date, date):
        pred_date_str = prediction_date.isoformat()
    else:
        pred_date_str = str(prediction_date)

    # 1. Validate incoming sales data
    valid_new_df, review_df, val_metrics = validate_monthly_sales_data(new_sales, existing_history_df)

    if valid_new_df.empty:
        return {
            "status": "warning",
            "message": "No valid transactions found in uploaded file.",
            "business_metrics": val_metrics,
            "combined_history_df": existing_history_df,
            "eligible_df": pd.DataFrame(),
            "ineligible_df": review_df,
            "reminder_schedule_df": pd.DataFrame(),
            "outcome_evaluation_metrics": {"evaluated_count": 0, "pending_count": 0},
            "new_snapshots_saved": 0,
        }

    # 2. Add new transactions to existing purchase history (NEVER REPLACE)
    if existing_history_df is not None and not existing_history_df.empty:
        exist_copy = existing_history_df.copy()
        exist_copy["invoice_date"] = pd.to_datetime(exist_copy["invoice_date"], errors="coerce")
        valid_copy = valid_new_df.copy()
        valid_copy["invoice_date"] = pd.to_datetime(valid_copy["invoice_date"], errors="coerce")
        combined_raw = pd.concat([exist_copy, valid_copy], ignore_index=True)
    else:
        combined_raw = valid_new_df.copy()
        combined_raw["invoice_date"] = pd.to_datetime(combined_raw["invoice_date"], errors="coerce")

    # 3. Deduplicate across primary transaction identifiers
    dedup_cols = ["customerId", "itemId", "invoice_date"]
    if "invoice_number" in combined_raw.columns:
        dedup_cols.append("invoice_number")
    elif "quantity" in combined_raw.columns:
        dedup_cols.append("quantity")

    combined_dedup = combined_raw.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)

    # 4. Reconstruct customer + medicine history timelines
    if not pd.api.types.is_datetime64_any_dtype(combined_dedup["invoice_date"]):
        combined_dedup["invoice_date"] = pd.to_datetime(combined_dedup["invoice_date"], errors="coerce")

    # Drop any null dates
    combined_dedup = combined_dedup[combined_dedup["invoice_date"].notna()].copy()

    # Create canonical chronological purchase history
    combined_history_df = create_purchase_history(combined_dedup)

    # Enrich physical units if packing is available
    if "packing" in combined_history_df.columns:
        combined_history_df = enrich_total_units_purchased(combined_history_df)

    # 5. Match and evaluate previous prediction snapshots against incoming actual purchases
    outcome_metrics = {"evaluated_count": 0, "pending_count": 0}
    if storage is not None:
        outcome_metrics = storage.match_and_update_prediction_outcomes(valid_new_df)

    # 6. Build feature dataset for active latest purchase events
    feature_df = build_feature_dataset(combined_history_df)

    # Isolate the latest purchase event per customer + medicine
    latest_events = feature_df.sort_values("invoice_date").groupby(["customerId", "itemId"], as_index=False).last()

    # Separate multi-purchase candidates (purchase_count >= 2) from cold-start
    p_counts = latest_events.get("purchase_count_so_far", latest_events.get("purchase_seq", 1))
    is_cold_start = p_counts.fillna(1).astype(int) < 2

    raw_candidates = latest_events[~is_cold_start].copy()
    raw_cold_start = latest_events[is_cold_start].copy()

    # Generate predictions for multi-purchase candidates
    if not raw_candidates.empty:
        preds_df = generate_batch_predictions(model_bundle, raw_candidates)
    else:
        preds_df = raw_candidates

    eligible_rows: List[Dict[str, Any]] = []
    ineligible_rows: List[Dict[str, Any]] = []

    for _, row in preds_df.iterrows():
        rec_dict = row.to_dict()
        p_cnt = int(rec_dict.get("purchase_count_so_far", rec_dict.get("purchase_seq", 1)))
        elig = evaluate_refill_eligibility(rec_dict, min_purchase_count=2)

        if elig["is_eligible"]:
            rec_dict["history_quality"] = elig["history_quality"]
            rec_dict["prediction_source"] = rec_dict.get("prediction_source", "hybrid_supply")
            eligible_rows.append(rec_dict)
        elif p_cnt >= 2:
            # Fallback for multi-purchase histories where supply or historical interval is available
            dos_cand = rec_dict.get("estimated_days_of_supply")
            med_cand = rec_dict.get("historical_interval_median")
            cand_interval = dos_cand if (dos_cand is not None and not pd.isna(dos_cand) and dos_cand > 0) else med_cand

            if cand_interval is not None and not pd.isna(cand_interval) and cand_interval > 0:
                rec_dict["predicted_days_until_refill"] = float(cand_interval)
                rec_dict["prediction_source"] = "estimated_days_of_supply" if dos_cand else "historical_interval_median"
                rec_dict["history_quality"] = "medium_history" if p_cnt >= 3 else "low_history"
                rec_dict["Pilot Tier"] = "Tier B (Review Required)" if p_cnt >= 3 else "Tier C (Suppressed / Low History)"
                eligible_rows.append(rec_dict)
            else:
                rec_dict["Reason for Ineligibility"] = elig["reason"]
                ineligible_rows.append(rec_dict)
        else:
            rec_dict["Reason for Ineligibility"] = elig["reason"]
            ineligible_rows.append(rec_dict)

    # Format eligible DataFrame
    if eligible_rows:
        el_raw = pd.DataFrame(eligible_rows)
        eligible_df = pd.DataFrame()
        eligible_df["customerId"] = el_raw["customerId"].astype(str)
        eligible_df["itemId"] = el_raw["itemId"].astype(str)
        eligible_df["Customer Name"] = el_raw.get("customerName", el_raw["customerId"]).fillna("Unknown Customer")
        eligible_df["Medicine"] = el_raw.get("itemName", el_raw["itemId"]).fillna("Unknown Medicine")
        eligible_df["Medication"] = eligible_df["Medicine"]

        phone_val = el_raw.get("MOBILE_NO", "").fillna("").astype(str).str.strip()
        eligible_df["Delivery Phone"] = phone_val
        eligible_df["Mobile Number"] = phone_val
        eligible_df["Mobile Status"] = phone_val.apply(determine_mobile_status)

        last_dt = pd.to_datetime(el_raw["invoice_date"], errors="coerce")
        eligible_df["Last Purchase Date"] = last_dt.dt.strftime("%Y-%m-%d").fillna("-")
        eligible_df["Purchase Count"] = el_raw.get("purchase_count_so_far", 1).astype(int)

        # Estimated days of supply
        if "estimated_days_of_supply" in el_raw.columns and el_raw["estimated_days_of_supply"].notna().any():
            dos_series = pd.to_numeric(el_raw["estimated_days_of_supply"], errors="coerce")
            pred_days = pd.to_numeric(el_raw["predicted_days_until_refill"], errors="coerce")
            final_dos = dos_series.fillna(pred_days).round(1)
        else:
            final_dos = pd.to_numeric(el_raw["predicted_days_until_refill"], errors="coerce").round(1)

        eligible_df["Estimated Days of Supply"] = final_dos
        eligible_df["Predicted Interval (Days)"] = final_dos
        eligible_df["Estimated Daily Consumption"] = el_raw.get("estimated_daily_consumption", el_raw.get("historical_consumption_rate", np.nan))
        eligible_df["estimated_daily_consumption"] = eligible_df["Estimated Daily Consumption"]

        # Expected Refill Date & 2-day adherence reminder date
        exp_dates = []
        rem_dates = []
        for l_dt, dos in zip(last_dt, final_dos):
            if pd.notna(l_dt) and pd.notna(dos) and dos > 0:
                e_d = l_dt + timedelta(days=int(round(dos)))
                buf_days = max(1, int(round(dos - 2.0)))
                r_d = l_dt + timedelta(days=buf_days)
                exp_dates.append(e_d.strftime("%Y-%m-%d"))
                rem_dates.append(r_d.strftime("%Y-%m-%d"))
            else:
                exp_dates.append("-")
                rem_dates.append("-")

        eligible_df["Expected Refill Date"] = exp_dates
        eligible_df["Reminder Date"] = rem_dates
        eligible_df["History Quality"] = el_raw.get("history_quality", "medium_history")
        eligible_df["Pilot Tier"] = el_raw.get("Pilot Tier", "Tier B (Review Required)")

        eligible_df["MOBILE_NO"] = eligible_df["Delivery Phone"]
        eligible_df["invoice_date"] = eligible_df["Last Purchase Date"]
        eligible_df["latest_purchase_date"] = eligible_df["Last Purchase Date"]
        eligible_df["predicted_days_until_refill"] = eligible_df["Estimated Days of Supply"]
        eligible_df["expected_refill_date"] = eligible_df["Expected Refill Date"]
        eligible_df["reminder_date"] = eligible_df["Reminder Date"]
        eligible_df["raw_reminder_date"] = eligible_df["Reminder Date"]
        eligible_df["customerName"] = eligible_df["Customer Name"]
        eligible_df["itemName"] = eligible_df["Medicine"]
        eligible_df["purchase_count_so_far"] = eligible_df["Purchase Count"]
        eligible_df["estimated_days_of_supply"] = eligible_df["Estimated Days of Supply"]
        eligible_df["mobile_status"] = eligible_df["Mobile Status"]
        eligible_df["prediction_source"] = el_raw.get("prediction_source", "hybrid_supply")
    else:
        eligible_df = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Medication",
            "Delivery Phone", "Mobile Number", "Mobile Status", "Last Purchase Date",
            "Purchase Count", "Estimated Days of Supply", "Predicted Interval (Days)",
            "Expected Refill Date", "Reminder Date", "History Quality", "Pilot Tier",
            "MOBILE_NO", "invoice_date", "latest_purchase_date", "predicted_days_until_refill",
            "expected_refill_date", "reminder_date", "raw_reminder_date", "customerName",
            "itemName", "purchase_count_so_far", "estimated_days_of_supply", "mobile_status",
            "prediction_source"
        ])

    # Format ineligible DataFrame
    ineligible_list: List[Dict[str, Any]] = []
    for _, r in raw_cold_start.iterrows():
        p_val = str(r.get("MOBILE_NO", "")).strip()
        m_val = str(r.get("itemName", r.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r.get("customerId", "")),
            "itemId": str(r.get("itemId", "")),
            "Customer Name": str(r.get("customerName", r.get("customerId", "Unknown"))),
            "Medicine": m_val,
            "Medication": m_val,
            "Delivery Phone": p_val,
            "Mobile Number": p_val,
            "Mobile Status": determine_mobile_status(p_val),
            "Last Purchase Date": pd.to_datetime(r.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r.get("invoice_date")) else "-",
            "Purchase Count": int(r.get("purchase_count_so_far", r.get("purchase_seq", 1))),
            "Reason for Ineligibility": "Cold-start history: purchase count is 1 (< 2).",
        })

    for r_dict in ineligible_rows:
        p_val = str(r_dict.get("MOBILE_NO", "")).strip()
        m_val = str(r_dict.get("itemName", r_dict.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r_dict.get("customerId", "")),
            "itemId": str(r_dict.get("itemId", "")),
            "Customer Name": str(r_dict.get("customerName", r_dict.get("customerId", "Unknown"))),
            "Medicine": m_val,
            "Medication": m_val,
            "Delivery Phone": p_val,
            "Mobile Number": p_val,
            "Mobile Status": determine_mobile_status(p_val),
            "Last Purchase Date": pd.to_datetime(r_dict.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r_dict.get("invoice_date")) else "-",
            "Purchase Count": int(r_dict.get("purchase_count_so_far", r_dict.get("purchase_seq", 1))),
            "Reason for Ineligibility": str(r_dict.get("Reason for Ineligibility", "Invalid prediction")),
        })

    # Add any rejected rows from upload validation
    for _, r in review_df.iterrows():
        p_val = str(r.get("MOBILE_NO", "")).strip()
        m_val = str(r.get("itemName", r.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r.get("customerId", "")),
            "itemId": str(r.get("itemId", "")),
            "Customer Name": str(r.get("customerName", r.get("customerId", "Unknown"))),
            "Medicine": m_val,
            "Medication": m_val,
            "Delivery Phone": p_val,
            "Mobile Number": p_val,
            "Mobile Status": determine_mobile_status(p_val),
            "Last Purchase Date": pd.to_datetime(r.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r.get("invoice_date")) else "-",
            "Purchase Count": int(r.get("quantity", 0)),
            "Reason for Ineligibility": "Invalid sales record (missing required fields or non-positive quantity).",
        })

    if ineligible_list:
        ineligible_df = pd.DataFrame(ineligible_list)
    else:
        ineligible_df = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Medication",
            "Delivery Phone", "Mobile Number", "Mobile Status", "Last Purchase Date",
            "Purchase Count", "Reason for Ineligibility"
        ])

    # 7. Store new prediction snapshots in append-only storage (unseen production data)
    new_snapshots_saved = 0
    if storage is not None and not eligible_df.empty:
        snapshot_records: List[Dict[str, Any]] = []
        for _, row in eligible_df.iterrows():
            cid = str(row["customerId"])
            iid = str(row["itemId"])
            last_p_dt = str(row["Last Purchase Date"])
            snapshot_records.append({
                "snapshot_id": f"snap_{cid}_{iid}_{pred_date_str}_{last_p_dt}",
                "customer_id": cid,
                "item_id": iid,
                "customer_name": str(row["Customer Name"]),
                "item_name": str(row["Medication"]),
                "last_purchase_date": last_p_dt,
                "prediction_date": pred_date_str,
                "estimated_days_of_supply": float(row["Estimated Days of Supply"]) if pd.notna(row["Estimated Days of Supply"]) else None,
                "expected_refill_date": str(row["Expected Refill Date"]),
                "reminder_date": str(row["Reminder Date"]),
                "prediction_source": str(row.get("prediction_source", "hybrid_supply")),
                "status": "pending",
            })

        new_snapshots_saved = storage.save_prediction_snapshots(snapshot_records)

    # 8. Generate updated reminder schedule
    reminder_schedule_df = eligible_df.copy()

    return {
        "status": "success",
        "business_metrics": val_metrics,
        "combined_history_df": combined_history_df,
        "eligible_df": eligible_df,
        "ineligible_df": ineligible_df,
        "reminder_schedule_df": reminder_schedule_df,
        "outcome_evaluation_metrics": outcome_metrics,
        "new_snapshots_saved": new_snapshots_saved,
    }


def evaluate_prediction_outcomes(
    predictions_input: Union[pd.DataFrame, List[Dict[str, Any]]],
    actual_sales_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate previous predictions against newly arrived actual sales transactions.

    Identifies the actual subsequent purchase for each customer + medication and compares
    the predicted refill date against the real-world purchase date.

    Calculates:
    - Predictions evaluated
    - Absolute error in days
    - Within ±1 day (count & percentage)
    - Within ±3 days (count & percentage)
    - Within ±7 days (count & percentage)
    - Mean Absolute Error (Average difference between expected and actual refill)

    Rules:
    - Only calculates metrics where a valid actual subsequent purchase exists.
    - Does NOT treat customers who have not purchased yet as prediction failures (marked pending).
    - Excludes technical ML/train/test diagnostics from client results.

    Args:
        predictions_input: DataFrame or list of prediction snapshots/records.
        actual_sales_df: DataFrame containing incoming verified sales transactions.

    Returns:
        Dict[str, Any] containing outcome metrics and clean evaluated DataFrame.
    """
    empty_res = {
        "status": "no_evaluated_records",
        "predictions_evaluated": 0,
        "within_1_day_count": 0,
        "within_1_day_pct": 0.0,
        "within_3_days_count": 0,
        "within_3_days_pct": 0.0,
        "within_7_days_count": 0,
        "within_7_days_pct": 0.0,
        "mean_absolute_error": 0.0,
        "pending_count": 0,
        "evaluated_df": pd.DataFrame(columns=[
            "Customer Name", "Medication", "Last Purchase Date",
            "Expected Refill Date", "Actual Purchase Date",
            "Difference (Days)", "Within ±3 Days", "Within ±7 Days"
        ]),
    }

    if predictions_input is None or actual_sales_df is None:
        return empty_res

    # Normalize predictions input
    if isinstance(predictions_input, pd.DataFrame):
        if predictions_input.empty:
            return empty_res
        preds_df = predictions_input.copy()
    elif isinstance(predictions_input, list):
        if not predictions_input:
            return empty_res
        preds_df = pd.DataFrame(predictions_input)
    else:
        return empty_res

    # Normalize actual sales
    sales_df = normalize_sales_dataframe(actual_sales_df)
    if sales_df.empty or "invoice_date" not in sales_df.columns:
        return empty_res

    sales_df["_dt"] = pd.to_datetime(sales_df["invoice_date"], errors="coerce")
    valid_sales = sales_df[sales_df["_dt"].notna()].copy()
    if valid_sales.empty:
        return empty_res

    # Identify column mappings in predictions DataFrame
    cid_col = "customerId" if "customerId" in preds_df.columns else ("customer_id" if "customer_id" in preds_df.columns else None)
    iid_col = "itemId" if "itemId" in preds_df.columns else ("item_id" if "item_id" in preds_df.columns else None)
    cname_col = "Customer Name" if "Customer Name" in preds_df.columns else ("customerName" if "customerName" in preds_df.columns else "customer_name")
    iname_col = "Medication" if "Medication" in preds_df.columns else ("Medicine" if "Medicine" in preds_df.columns else ("itemName" if "itemName" in preds_df.columns else "item_name"))
    last_dt_col = "Last Purchase Date" if "Last Purchase Date" in preds_df.columns else ("last_purchase_date" if "last_purchase_date" in preds_df.columns else ("invoice_date" if "invoice_date" in preds_df.columns else "prediction_date"))
    exp_dt_col = "Expected Refill Date" if "Expected Refill Date" in preds_df.columns else ("expected_refill_date" if "expected_refill_date" in preds_df.columns else None)

    if not cid_col or not iid_col or not exp_dt_col:
        return empty_res

    evaluated_records: List[Dict[str, Any]] = []
    pending_count = 0

    sales_cid_col = "customerId" if "customerId" in valid_sales.columns else "customer_id"
    sales_iid_col = "itemId" if "itemId" in valid_sales.columns else "item_id"

    # Pre-index sales by (customer, item) for fast matching
    valid_sales["_key"] = (
        valid_sales[sales_cid_col].astype(str).str.strip().str.lower() + "___" +
        valid_sales[sales_iid_col].astype(str).str.strip().str.lower()
    )
    sales_by_key = {k: v for k, v in valid_sales.groupby("_key")}

    for _, row in preds_df.iterrows():
        cid_val = str(row[cid_col]).strip()
        iid_val = str(row[iid_col]).strip()
        exp_dt_raw = row.get(exp_dt_col)
        last_dt_raw = row.get(last_dt_col) if last_dt_col in row else None

        if not cid_val or not iid_val or pd.isna(exp_dt_raw) or str(exp_dt_raw).strip() in ("", "-"):
            continue

        exp_dt = pd.to_datetime(exp_dt_raw, errors="coerce")
        last_dt = pd.to_datetime(last_dt_raw, errors="coerce") if last_dt_raw else None

        if pd.isna(exp_dt):
            continue

        key = cid_val.lower() + "___" + iid_val.lower()
        subsequent_sales = sales_by_key.get(key)

        if subsequent_sales is None or subsequent_sales.empty:
            pending_count += 1
            continue

        # Look for purchases strictly after the last purchase date (or on/after if baseline is last purchase)
        if pd.notna(last_dt):
            after_mask = subsequent_sales["_dt"] > last_dt
            candidate_matches = subsequent_sales[after_mask]
        else:
            candidate_matches = subsequent_sales

        if candidate_matches.empty:
            pending_count += 1
            continue

        # First qualifying subsequent purchase
        first_next_sale = candidate_matches.sort_values("_dt").iloc[0]
        actual_dt = first_next_sale["_dt"]
        diff_days = float((actual_dt - exp_dt).days)
        abs_err = abs(diff_days)

        cname_val = str(row.get(cname_col, cid_val)) if cname_col in row and pd.notna(row.get(cname_col)) else cid_val
        iname_val = str(row.get(iname_col, iid_val)) if iname_col in row and pd.notna(row.get(iname_col)) else iid_val
        last_dt_str = last_dt.strftime("%Y-%m-%d") if (last_dt and pd.notna(last_dt)) else "-"

        def _fmt_diff(d: float) -> str:
            val = int(round(d))
            return f"+{val} days" if val > 0 else (f"{val} days" if val < 0 else "0 days (exact)")

        evaluated_records.append({
            "Customer Name": cname_val,
            "Medication": iname_val,
            "Last Purchase Date": last_dt_str,
            "Expected Refill Date": exp_dt.strftime("%Y-%m-%d"),
            "Actual Purchase Date": actual_dt.strftime("%Y-%m-%d"),
            "Difference (Days)": _fmt_diff(diff_days),
            "diff_days_raw": diff_days,
            "abs_error_days": abs_err,
            "Within ±3 Days": "Yes" if abs_err <= 3.0 else "No",
            "Within ±7 Days": "Yes" if abs_err <= 7.0 else "No",
            "Within ±1 Day": "Yes" if abs_err <= 1.0 else "No",
        })

    if not evaluated_records:
        return {
            "status": "no_evaluated_records",
            "predictions_evaluated": 0,
            "within_1_day_count": 0,
            "within_1_day_pct": 0.0,
            "within_3_days_count": 0,
            "within_3_days_pct": 0.0,
            "within_7_days_count": 0,
            "within_7_days_pct": 0.0,
            "mean_absolute_error": 0.0,
            "pending_count": pending_count,
            "evaluated_df": pd.DataFrame(columns=[
                "Customer Name", "Medication", "Last Purchase Date",
                "Expected Refill Date", "Actual Purchase Date",
                "Difference (Days)", "Within ±3 Days", "Within ±7 Days"
            ]),
        }

    eval_df = pd.DataFrame(evaluated_records)
    n_eval = len(eval_df)
    w1_cnt = int((eval_df["abs_error_days"] <= 1.0).sum())
    w3_cnt = int((eval_df["abs_error_days"] <= 3.0).sum())
    w7_cnt = int((eval_df["abs_error_days"] <= 7.0).sum())
    mae = float(eval_df["abs_error_days"].mean())

    display_cols = [
        "Customer Name",
        "Medication",
        "Last Purchase Date",
        "Expected Refill Date",
        "Actual Purchase Date",
        "Difference (Days)",
        "Within ±3 Days",
        "Within ±7 Days",
    ]

    return {
        "status": "success",
        "predictions_evaluated": n_eval,
        "within_1_day_count": w1_cnt,
        "within_1_day_pct": round(w1_cnt / n_eval * 100, 1),
        "within_3_days_count": w3_cnt,
        "within_3_days_pct": round(w3_cnt / n_eval * 100, 1),
        "within_7_days_count": w7_cnt,
        "within_7_days_pct": round(w7_cnt / n_eval * 100, 1),
        "mean_absolute_error": round(mae, 1),
        "pending_count": pending_count,
        "evaluated_df": eval_df[display_cols],
    }

