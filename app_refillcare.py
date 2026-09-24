"""RefillCare — Medication Refill Reminder System (Streamlit UI).

Pharmacy-centric operational interface for reviewing validated refill predictions,
managing date-filtered reminder lists, uploading monthly sales updates,
and exporting customer CSVs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import warnings

# Suppress serialization and version mismatch warnings for cross-version compatibility
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*unpickle estimator.*")

import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refillcare.models.prediction import generate_batch_predictions
from refillcare.data.packing import compute_total_units_purchased, parse_pack_units
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)
from reminder.scheduler import (
    RefillReminderScheduler,
    REMINDER_STAGES,
    evaluate_refill_eligibility,
    calculate_expected_refill_date,
    format_reminder_message,
)
from reminder.storage import (
    RefillCareStorage,
    DEFAULT_DB_PATH,
)
from refillcare.data.dates import (
    parse_pharmacy_dates,
    format_date_dd_mm_yyyy,
    UI_DATE_FORMAT,
)
from refillcare.data.monthly_ingestion import (
    generate_updated_predictions,
    ingest_monthly_sales_pipeline,
    process_monthly_sales_data,
    validate_monthly_sales_data,
    rollback_monthly_sales_import,
    evaluate_prediction_outcomes,
)

# Constants & Default Paths
DEFAULT_MODEL_PATH = "data/refillcare/processed/models/refill_model.joblib"
DEFAULT_TEST_DATA_PATH = "data/refillcare/processed/test.parquet"
DEFAULT_HISTORY_DATA_PATH = "data/refillcare/processed/purchase_history.parquet"
DEFAULT_TRAIN_DATA_PATH = "data/refillcare/processed/training_dataset.parquet"


def find_artifact_path(relative_path: str) -> Path:
    """Safely resolve an artifact path across workspace roots."""
    candidates = [
        PROJECT_ROOT / relative_path,
        Path.cwd() / relative_path,
        Path("..") / relative_path,
        Path("../..") / relative_path,
        Path(r"C:\Users\sunil\ai-mediastra-whatsapp-reminder\ai-mediastra-whatsapp-reminder") / relative_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


@st.cache_resource(show_spinner=False)
def load_refill_model(model_path: Optional[Union[Path, str]] = None) -> Optional[Dict[str, Any]]:
    """Load the trained refill model bundle."""
    target_path = Path(model_path) if model_path else find_artifact_path(DEFAULT_MODEL_PATH)
    if not target_path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(target_path)
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_refill_data(data_path: Optional[Union[Path, str]] = None) -> Optional[pd.DataFrame]:
    """Load the processed evaluation dataset."""
    if data_path is not None:
        target_path = Path(data_path)
        if not target_path.exists():
            return None
    else:
        target_path = find_artifact_path(DEFAULT_TEST_DATA_PATH)
        if not target_path.exists():
            fallback = find_artifact_path(DEFAULT_TRAIN_DATA_PATH)
            if fallback.exists():
                target_path = fallback
            else:
                return None
    try:
        return pd.read_parquet(target_path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_purchase_history(history_path: Optional[Union[Path, str]] = None) -> Optional[pd.DataFrame]:
    """Load the canonical purchase history dataset."""
    if history_path is not None:
        target_path = Path(history_path)
        if not target_path.exists():
            target_path = find_artifact_path(str(history_path))
        if not target_path.exists():
            return None
    else:
        target_path = find_artifact_path(DEFAULT_HISTORY_DATA_PATH)
        if not target_path.exists():
            return None
    try:
        return pd.read_parquet(target_path)
    except Exception:
        return None


def determine_mobile_status(phone: Any) -> str:
    """Determine client-facing mobile status without removing any customer.

    Status values:
    - 'Valid': 10-digit number (or 12-digit starting with 91, or 11-digit with leading 0)
    - 'Missing': Empty, None, or NaN phone
    - 'Invalid format': Non-empty phone with invalid length or characters
    """
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


def _determine_pilot_tier(purchase_count: int, is_recurring: int, quality: str) -> Tuple[str, str]:
    """Assign an explainable pilot operational tier and human-readable reason."""
    if purchase_count >= 5 or (purchase_count >= 3 and is_recurring == 1) or quality == "high_history":
        return "Tier A (Strong Pilot)", f"Consistent recurring history ({purchase_count} previous purchases)."
    elif purchase_count >= 3 or quality == "medium_history":
        return "Tier B (Review Required)", f"Moderate history ({purchase_count} purchases) — pharmacist verification recommended."
    elif purchase_count == 2:
        return "Tier C (Suppressed / Low History)", "Limited history (2 purchases) — low statistical basis; suppressed from default pilot queue."
    else:
        return "Tier C (Excluded / Cold-Start)", "Insufficient history (< 2 purchases)."


def filter_predictions(
    df: pd.DataFrame,
    customer_query: str = "",
    medicine_query: str = "",
    quality_filter: str = "All",
    tier_filter: str = "All",
    date_range: Optional[Tuple[Any, Any]] = None,
) -> pd.DataFrame:
    """Filter prediction overview table."""
    if df.empty:
        return df

    filtered = df.copy()

    if customer_query.strip():
        q = customer_query.strip().lower()
        col = "Customer Name" if "Customer Name" in filtered.columns else "Customer"
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).str.lower().str.contains(q)]

    if medicine_query.strip():
        q = medicine_query.strip().lower()
        col = "Medicine" if "Medicine" in filtered.columns else "Medication"
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).str.lower().str.contains(q)]

    if quality_filter != "All" and "History Quality" in filtered.columns:
        filtered = filtered[filtered["History Quality"] == quality_filter]

    if tier_filter != "All" and "Pilot Tier" in filtered.columns:
        filtered = filtered[filtered["Pilot Tier"] == tier_filter]

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_d, end_d = str(date_range[0]), str(date_range[1])
        if "Expected Refill Date" in filtered.columns:
            filtered = filtered[
                (filtered["Expected Refill Date"] >= start_d) &
                (filtered["Expected Refill Date"] <= end_d)
            ]

    return filtered


def filter_schedules(
    df: pd.DataFrame,
    customer_query: str = "",
    medicine_query: str = "",
    stage_filter: str = "All",
    status_filter: str = "All",
    tier_filter: str = "All",
    target_date: Optional[date] = None,
) -> pd.DataFrame:
    """Filter reminder schedule table."""
    if df.empty:
        return df

    filtered = df.copy()

    if customer_query.strip():
        q = customer_query.strip().lower()
        col = "Customer" if "Customer" in filtered.columns else "Customer Name"
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).str.lower().str.contains(q)]

    if medicine_query.strip():
        q = medicine_query.strip().lower()
        col = "Medicine" if "Medicine" in filtered.columns else "Medication"
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).str.lower().str.contains(q)]

    if stage_filter != "All" and "Reminder Stage" in filtered.columns:
        filtered = filtered[filtered["Reminder Stage"] == stage_filter]

    if status_filter != "All" and "Status" in filtered.columns:
        filtered = filtered[filtered["Status"] == status_filter]

    if tier_filter != "All" and "Pilot Tier" in filtered.columns:
        filtered = filtered[filtered["Pilot Tier"] == tier_filter]

    if target_date is not None:
        target_str = str(target_date)
        if "raw_reminder_date" in filtered.columns:
            filtered = filtered[filtered["raw_reminder_date"] == target_str]
        elif "Reminder Date" in filtered.columns:
            filtered = filtered[filtered["Reminder Date"] == target_str]

    return filtered


def map_reason_client_friendly(reason: str) -> str:
    """Map technical evaluation reasons into clean client-facing terminology."""
    r = str(reason).lower()
    if "cold-start" in r or "purchase count is 1" in r or "< 2" in r:
        return "Insufficient purchase history"
    elif "invalid prediction" in r or "low quality" in r:
        return "Irregular purchase pattern"
    else:
        return "Unable to determine expected refill date"


@st.cache_data(show_spinner=False)
def prepare_prediction_overview(
    df: pd.DataFrame,
    _model_bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate clean predictions and separate eligible vs ineligible customer records."""
    model_bundle = _model_bundle
    if df.empty:
        empty_el = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Delivery Phone",
            "Mobile Number", "Mobile Status", "Last Purchase Date", "Purchase Count",
            "Estimated Days of Supply", "Predicted Interval (Days)", "Expected Refill Date",
            "Reminder Date", "MOBILE_NO", "invoice_date", "predicted_days_until_refill",
            "expected_refill_date", "customerName", "itemName", "purchase_count_so_far",
            "latest_purchase_date", "estimated_days_of_supply", "reminder_date",
            "mobile_status", "raw_reminder_date"
        ])
        empty_inel = pd.DataFrame(columns=[
            "customerId", "itemId", "Customer Name", "Medicine", "Delivery Phone",
            "Mobile Number", "Mobile Status", "Last Purchase Date", "Purchase Count",
            "Reason for Ineligibility"
        ])
        metrics = {
            "total_histories": 0,
            "eligible_count": 0,
            "ineligible_count": 0,
        }
        return empty_el, empty_inel, metrics

    # Isolate the latest purchase event per customer + medicine history
    working_df = df.sort_values("invoice_date").groupby(["customerId", "itemId"], as_index=False).last()

    # Identify single purchase histories
    p_counts = working_df.get("purchase_count_so_far", working_df.get("purchase_seq", 1))
    is_cold_start = p_counts.fillna(1).astype(int) < 2

    raw_candidates = working_df[~is_cold_start].copy()
    raw_cold_start = working_df[is_cold_start].copy()

    # Generate refill predictions for multi-purchase candidates
    if not raw_candidates.empty:
        preds_df = generate_batch_predictions(model_bundle, raw_candidates)
    else:
        preds_df = raw_candidates

    # Evaluate eligibility
    eligible_rows = []
    ineligible_rows = []

    for _, row in preds_df.iterrows():
        rec_dict = row.to_dict()
        elig = evaluate_refill_eligibility(rec_dict, min_purchase_count=2)
        if elig["is_eligible"]:
            rec_dict["history_quality"] = elig["history_quality"]
            p_cnt = int(rec_dict.get("purchase_count_so_far", 1))
            is_rec = int(rec_dict.get("is_recurring_history", 0) or 0)
            tier, note = _determine_pilot_tier(p_cnt, is_rec, elig["history_quality"])
            rec_dict["Pilot Tier"] = tier
            rec_dict["Operator Notes"] = note
            eligible_rows.append(rec_dict)
        else:
            rec_dict["Reason for Ineligibility"] = elig["reason"]
            rec_dict["Pilot Tier"] = "Tier C (Excluded / Ineligible)"
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

        # Dates & Intervals
        last_dt = pd.to_datetime(el_raw["invoice_date"], errors="coerce")
        eligible_df["Last Purchase Date"] = last_dt.dt.strftime("%Y-%m-%d").fillna("-")
        eligible_df["Purchase Count"] = el_raw.get("purchase_count_so_far", 1).astype(int)

        # Estimated Days of Supply
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

        # Expected Refill Date & Reminder Date (with 2-day buffer)
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
        eligible_df["Operator Notes"] = el_raw.get("Operator Notes", "Standard history.")

        # Keys required by scheduler engine and CSV exports
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
        phone_val = str(r.get("MOBILE_NO", "")).strip()
        med_val = str(r.get("itemName", r.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r.get("customerId", "")),
            "itemId": str(r.get("itemId", "")),
            "Customer Name": str(r.get("customerName", r.get("customerId", "Unknown"))),
            "Medicine": med_val,
            "Medication": med_val,
            "Delivery Phone": phone_val,
            "Mobile Number": phone_val,
            "Mobile Status": determine_mobile_status(phone_val),
            "Last Purchase Date": pd.to_datetime(r.get("invoice_date")).strftime("%Y-%m-%d") if pd.notna(r.get("invoice_date")) else "-",
            "Purchase Count": int(r.get("purchase_count_so_far", r.get("purchase_seq", 1))),
            "Reason for Ineligibility": "Cold-start history: purchase count is 1 (< 2).",
        })

    for r_dict in ineligible_rows:
        phone_val = str(r_dict.get("MOBILE_NO", "")).strip()
        med_val = str(r_dict.get("itemName", r_dict.get("itemId", "Unknown")))
        ineligible_list.append({
            "customerId": str(r_dict.get("customerId", "")),
            "itemId": str(r_dict.get("itemId", "")),
            "Customer Name": str(r_dict.get("customerName", r_dict.get("customerId", "Unknown"))),
            "Medicine": med_val,
            "Medication": med_val,
            "Delivery Phone": phone_val,
            "Mobile Number": phone_val,
            "Mobile Status": determine_mobile_status(phone_val),
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

    metrics = {
        "total_histories": len(working_df),
        "eligible_count": len(eligible_df),
        "ineligible_count": len(ineligible_df),
    }

    return eligible_df, ineligible_df, metrics


def _format_stage_label(s: int) -> str:
    """Format reminder stage integer into clean label."""
    if s == -1:
        return "-1 day"
    elif s == 0:
        return "0 days"
    elif s > 0:
        return f"+{s} days"
    else:
        return f"{s} days"


def generate_reminder_schedule_table(
    eligible_df: pd.DataFrame,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Run scheduler across eligible records and return structured schedule dataframe."""
    if eligible_df.empty:
        return pd.DataFrame(columns=[
            "Customer Name", "Mobile Number", "Medication", "Last Purchase Date",
            "Estimated Days of Supply", "Expected Refill Date", "Reminder Date",
            "Mobile Status", "Reminder Stage", "Status", "customerId", "itemId",
            "raw_reminder_date", "latest_purchase_date_raw", "expected_refill_date_raw",
            "estimated_days_of_supply"
        ])

    scheduler = RefillReminderScheduler()
    subset = eligible_df.head(max_records) if max_records else eligible_df

    for _, row in subset.iterrows():
        scheduler.schedule_refill_cycle(row.to_dict())

    sched_df = scheduler.to_dataframe()
    if sched_df.empty:
        return pd.DataFrame()

    output = pd.DataFrame()
    output["Customer Name"] = sched_df["customerName"]
    output["Mobile Number"] = sched_df["MOBILE_NO"]
    output["Medication"] = sched_df["itemName"]
    output["Last Purchase Date"] = pd.to_datetime(sched_df["latest_purchase_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Estimated Days of Supply
    if "predicted_interval" in sched_df.columns:
        output["Estimated Days of Supply"] = sched_df["predicted_interval"].round(1)
    elif "estimated_days_of_supply" in sched_df.columns:
        output["Estimated Days of Supply"] = sched_df["estimated_days_of_supply"].round(1)
    else:
        output["Estimated Days of Supply"] = 30.0

    output["Expected Refill Date"] = pd.to_datetime(sched_df["expected_refill_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    output["Reminder Date"] = pd.to_datetime(sched_df["reminder_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    output["Mobile Status"] = output["Mobile Number"].apply(determine_mobile_status)
    output["Reminder Stage"] = sched_df["reminder_stage"].apply(_format_stage_label)
    output["Status"] = sched_df["status"]

    # Aliases for backward compatibility with existing tests
    output["Customer"] = sched_df["customerName"]
    output["Medicine"] = sched_df["itemName"]
    output["Delivery Phone"] = sched_df["MOBILE_NO"]
    output["Reminder Message"] = sched_df["message"]
    output["History Quality"] = sched_df["history_quality"]
    output["Pilot Tier"] = sched_df.get("Pilot Tier", "Tier B (Review Required)")
    output["reminder_id"] = sched_df["reminder_id"]

    # Identifiers for internal logic and CSV export
    output["customerId"] = sched_df["customerId"]
    output["itemId"] = sched_df["itemId"]
    output["raw_reminder_date"] = sched_df["reminder_date"]
    output["latest_purchase_date_raw"] = sched_df["latest_purchase_date"]
    output["expected_refill_date_raw"] = sched_df["expected_refill_date"]
    output["estimated_days_of_supply"] = output["Estimated Days of Supply"]
    output["mobile_status"] = output["Mobile Status"]

    return output


def get_customer_history_summary(
    history_df: pd.DataFrame,
    customer_id: str,
    item_id: str,
) -> Dict[str, Any]:
    """Retrieve detailed purchasing timeline for a customer and medication."""
    if history_df is None or history_df.empty:
        return {"total_purchases": 0, "visits": pd.DataFrame()}

    cid = str(customer_id)
    iid = str(item_id)

    subset = history_df[
        (history_df["customerId"].astype(str) == cid) &
        (history_df["itemId"].astype(str) == iid)
    ].sort_values("invoice_date")

    if subset.empty:
        return {"total_purchases": 0, "visits": pd.DataFrame()}

    intervals = subset["days_since_previous_purchase"].dropna().tolist() if "days_since_previous_purchase" in subset.columns else []

    visits_display = pd.DataFrame({
        "Visit": range(1, len(subset) + 1),
        "Invoice Date": pd.to_datetime(subset["invoice_date"]).dt.strftime("%d-%m-%Y"),
        "Quantity": subset["quantity"].astype(int) if "quantity" in subset.columns else 1,
        "Days Since Prior Purchase": subset["days_since_previous_purchase"].fillna("-") if "days_since_previous_purchase" in subset.columns else "-",
        "Salt / Composition": subset.get("salt_composition", "-").fillna("-") if "salt_composition" in subset.columns else "-",
    })

    return {
        "customerName": str(subset["customerName"].iloc[-1]) if "customerName" in subset.columns else str(cid),
        "itemName": str(subset["itemName"].iloc[-1]) if "itemName" in subset.columns else str(iid),
        "MOBILE_NO": str(subset["MOBILE_NO"].iloc[-1]) if "MOBILE_NO" in subset.columns else "",
        "total_purchases": len(subset),
        "first_purchase_date": str(pd.to_datetime(subset["invoice_date"].iloc[0]).date()),
        "latest_purchase_date": str(pd.to_datetime(subset["invoice_date"].iloc[-1]).date()),
        "median_interval": float(np.median(intervals)) if intervals else None,
        "mean_interval": float(np.mean(intervals)) if intervals else None,
        "intervals": intervals,
        "visits": visits_display,
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for Streamlit download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_export_csv(df: pd.DataFrame) -> bytes:
    """Build CSV bytes with the exact 10 required client columns:
    customer_id, customer_name, phone_number, mobile_status, item_id,
    medication_name, last_purchase_date, estimated_days_of_supply,
    expected_refill_date, reminder_date.
    """
    exact_columns = [
        "customer_id",
        "customer_name",
        "phone_number",
        "mobile_status",
        "item_id",
        "medication_name",
        "last_purchase_date",
        "estimated_days_of_supply",
        "expected_refill_date",
        "reminder_date",
    ]

    if df.empty:
        export_df = pd.DataFrame(columns=exact_columns)
    else:
        export_df = pd.DataFrame()
        # customer_id
        if "customerId" in df.columns:
            export_df["customer_id"] = df["customerId"].astype(str)
        elif "customer_id" in df.columns:
            export_df["customer_id"] = df["customer_id"].astype(str)
        else:
            export_df["customer_id"] = "-"

        # customer_name
        if "Customer Name" in df.columns:
            export_df["customer_name"] = df["Customer Name"].astype(str)
        elif "customerName" in df.columns:
            export_df["customer_name"] = df["customerName"].astype(str)
        elif "Customer" in df.columns:
            export_df["customer_name"] = df["Customer"].astype(str)
        elif "customer_name" in df.columns:
            export_df["customer_name"] = df["customer_name"].astype(str)
        else:
            export_df["customer_name"] = "Unknown Customer"

        # phone_number
        if "Mobile Number" in df.columns:
            export_df["phone_number"] = df["Mobile Number"].astype(str)
        elif "MOBILE_NO" in df.columns:
            export_df["phone_number"] = df["MOBILE_NO"].astype(str)
        elif "Delivery Phone" in df.columns:
            export_df["phone_number"] = df["Delivery Phone"].astype(str)
        elif "phone_number" in df.columns:
            export_df["phone_number"] = df["phone_number"].astype(str)
        else:
            export_df["phone_number"] = ""

        # mobile_status
        if "Mobile Status" in df.columns:
            export_df["mobile_status"] = df["Mobile Status"].astype(str)
        elif "mobile_status" in df.columns:
            export_df["mobile_status"] = df["mobile_status"].astype(str)
        else:
            export_df["mobile_status"] = export_df["phone_number"].apply(determine_mobile_status)

        # item_id
        if "itemId" in df.columns:
            export_df["item_id"] = df["itemId"].astype(str)
        elif "item_id" in df.columns:
            export_df["item_id"] = df["item_id"].astype(str)
        else:
            export_df["item_id"] = "-"

        # medication_name
        if "Medication" in df.columns:
            export_df["medication_name"] = df["Medication"].astype(str)
        elif "itemName" in df.columns:
            export_df["medication_name"] = df["itemName"].astype(str)
        elif "Medicine" in df.columns:
            export_df["medication_name"] = df["Medicine"].astype(str)
        elif "medication_name" in df.columns:
            export_df["medication_name"] = df["medication_name"].astype(str)
        else:
            export_df["medication_name"] = "Unknown Medicine"

        # last_purchase_date
        if "Last Purchase Date" in df.columns:
            export_df["last_purchase_date"] = df["Last Purchase Date"].astype(str)
        elif "latest_purchase_date_raw" in df.columns:
            export_df["last_purchase_date"] = pd.to_datetime(df["latest_purchase_date_raw"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("-")
        elif "last_purchase_date" in df.columns:
            export_df["last_purchase_date"] = df["last_purchase_date"].astype(str)
        else:
            export_df["last_purchase_date"] = "-"

        # estimated_days_of_supply
        if "Estimated Days of Supply" in df.columns:
            export_df["estimated_days_of_supply"] = df["Estimated Days of Supply"]
        elif "estimated_days_of_supply" in df.columns:
            export_df["estimated_days_of_supply"] = df["estimated_days_of_supply"]
        elif "predicted_days_until_refill" in df.columns:
            export_df["estimated_days_of_supply"] = df["predicted_days_until_refill"].round(1)
        else:
            export_df["estimated_days_of_supply"] = 30.0

        # expected_refill_date
        if "Expected Refill Date" in df.columns:
            export_df["expected_refill_date"] = df["Expected Refill Date"].astype(str)
        elif "expected_refill_date_raw" in df.columns:
            export_df["expected_refill_date"] = pd.to_datetime(df["expected_refill_date_raw"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("-")
        elif "expected_refill_date" in df.columns:
            export_df["expected_refill_date"] = df["expected_refill_date"].astype(str)
        else:
            export_df["expected_refill_date"] = "-"

        # reminder_date
        if "Reminder Date" in df.columns:
            export_df["reminder_date"] = df["Reminder Date"].astype(str)
        elif "raw_reminder_date" in df.columns:
            export_df["reminder_date"] = pd.to_datetime(df["raw_reminder_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("-")
        elif "reminder_date" in df.columns:
            export_df["reminder_date"] = df["reminder_date"].astype(str)
        else:
            export_df["reminder_date"] = "-"

        # Clean NaN values
        export_df = export_df[exact_columns].fillna("-")

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_reminder_list_csv(df: pd.DataFrame) -> bytes:
    """Build client-facing downloadable Reminder List CSV with only valid mobile numbers.

    Exact operational columns:
    - customerId
    - customerName
    - MOBILE_NO
    - itemId
    - itemName
    - last_purchase_date
    - expected_refill_date
    - reminder_date
    - predicted_days_until_refill
    """
    exact_columns = [
        "customerId",
        "customerName",
        "MOBILE_NO",
        "itemId",
        "itemName",
        "last_purchase_date",
        "expected_refill_date",
        "reminder_date",
        "predicted_days_until_refill",
    ]

    if df.empty:
        empty_df = pd.DataFrame(columns=exact_columns)
        buf = io.StringIO()
        empty_df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    working = df.copy()

    # Filter to only records with valid mobile numbers (missing mobile numbers excluded from delivery CSV)
    if "Mobile Status" in working.columns:
        valid_mobiles = working[working["Mobile Status"] == "Valid"].copy()
    elif "mobile_status" in working.columns:
        valid_mobiles = working[working["mobile_status"] == "Valid"].copy()
    elif "MOBILE_NO" in working.columns:
        valid_mobiles = working[working["MOBILE_NO"].apply(determine_mobile_status) == "Valid"].copy()
    elif "Delivery Phone" in working.columns:
        valid_mobiles = working[working["Delivery Phone"].apply(determine_mobile_status) == "Valid"].copy()
    elif "phone_number" in working.columns:
        valid_mobiles = working[working["phone_number"].apply(determine_mobile_status) == "Valid"].copy()
    else:
        valid_mobiles = working.copy()

    if valid_mobiles.empty:
        empty_df = pd.DataFrame(columns=exact_columns)
        buf = io.StringIO()
        empty_df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    export_df = pd.DataFrame()
    export_df["customerId"] = valid_mobiles.get("customerId", "").astype(str)
    export_df["customerName"] = valid_mobiles.get("Customer Name", valid_mobiles.get("customerName", export_df["customerId"])).fillna("Unknown").astype(str)

    # Phone number
    phone_col = "Mobile Number" if "Mobile Number" in valid_mobiles.columns else ("MOBILE_NO" if "MOBILE_NO" in valid_mobiles.columns else "Delivery Phone")
    export_df["MOBILE_NO"] = valid_mobiles.get(phone_col, valid_mobiles.get("phone_number", "")).fillna("").astype(str).str.strip()

    # Medicine
    export_df["itemId"] = valid_mobiles.get("itemId", "").astype(str)
    export_df["itemName"] = valid_mobiles.get("Medication", valid_mobiles.get("Medicine", valid_mobiles.get("itemName", valid_mobiles.get("medication_name", export_df["itemId"])))).fillna("Unknown Medicine").astype(str)

    # Dates
    last_dt = valid_mobiles.get("Last Purchase Date", valid_mobiles.get("last_purchase_date", valid_mobiles.get("invoice_date", "-")))
    export_df["last_purchase_date"] = last_dt.astype(str)

    exp_dt = valid_mobiles.get("Expected Refill Date", valid_mobiles.get("expected_refill_date", "-"))
    export_df["expected_refill_date"] = exp_dt.astype(str)

    rem_dt = valid_mobiles.get("Reminder Date", valid_mobiles.get("reminder_date", valid_mobiles.get("raw_reminder_date", "-")))
    export_df["reminder_date"] = rem_dt.astype(str)

    # predicted_days_until_refill
    dos_vals = valid_mobiles.get("Estimated Days of Supply", valid_mobiles.get("estimated_days_of_supply", valid_mobiles.get("predicted_days_until_refill", valid_mobiles.get("Predicted Interval (Days)", 30.0))))
    export_df["predicted_days_until_refill"] = pd.to_numeric(dos_vals, errors="coerce").fillna(30.0).round(1)

    export_df = export_df[exact_columns].fillna("-")
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def parse_and_validate_uploaded_sales_file(
    uploaded_file: Any,
    existing_history_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Parse an uploaded monthly sales Excel/CSV file and extract business metrics."""
    try:
        filename = getattr(uploaded_file, "name", "sales.csv").lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, {"error": "Unsupported file format. Please upload a .csv, .xlsx, or .xls file."}
    except Exception as e:
        return None, {"error": f"Failed to read file: {str(e)}"}

    if df.empty:
        return df, {
            "file_date_range": "N/A (Empty File)",
            "records_received": 0,
            "valid_records": 0,
            "records_requiring_review": 0,
            "new_customers": 0,
            "existing_customers": 0,
            "unique_customers": 0,
            "medicines_count": 0,
            "missing_customer_info": 0,
            "missing_mobile_numbers": 0,
            "duplicate_records": 0,
            "invalid_quantities": 0,
        }

    # Normalize column names for flexible parsing using project mappings
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

    renamed_df = df.rename(columns=col_map).copy()

    # Total records received
    records_received = len(renamed_df)

    # Date range analysis using strict pharmacy date parser
    date_col = "invoice_date" if "invoice_date" in renamed_df.columns else None
    if date_col is not None:
        parsed_dates, date_diagnostics = parse_pharmacy_dates(renamed_df[date_col])
        renamed_df["invoice_date"] = parsed_dates
        file_date_range = date_diagnostics["date_range_formatted"]
    else:
        file_date_range = "No invoice date column found"
        date_diagnostics = {
            "source_format": "Missing",
            "date_range_formatted": "No invoice date column found",
            "min_date_formatted": "-",
            "max_date_formatted": "-",
            "sample_dates_formatted": [],
            "valid_count": 0,
            "invalid_count": records_received,
            "is_ambiguous": True,
            "warning": "No invoice_date column found in file.",
        }

    # Number of customers
    cust_col = "customerId" if "customerId" in renamed_df.columns else ("customerName" if "customerName" in renamed_df.columns else None)
    if cust_col is not None:
        cust_series = renamed_df[cust_col].dropna().astype(str).str.strip()
        unique_cust_set = set(cust_series[~cust_series.str.lower().isin(["", "nan", "none", "null", "0", "-"] )])
        unique_customers = len(unique_cust_set)
    else:
        unique_cust_set = set()
        unique_customers = 0

    # Number of medicines
    item_col = "itemId" if "itemId" in renamed_df.columns else ("itemName" if "itemName" in renamed_df.columns else None)
    if item_col is not None:
        item_series = renamed_df[item_col].dropna().astype(str).str.strip()
        unique_item_set = set(item_series[~item_series.str.lower().isin(["", "nan", "none", "null", "0", "-"] )])
        medicines_count = len(unique_item_set)
    else:
        medicines_count = 0

    # Missing customer information
    if cust_col is not None:
        cust_str = renamed_df[cust_col].fillna("").astype(str).str.strip().str.lower()
        missing_customer_info = int((cust_str.isin(["", "nan", "none", "null", "-", "0"])).sum())
    else:
        missing_customer_info = records_received

    # Missing mobile numbers (customers with missing mobile numbers are NOT deleted; they remain in sales & history)
    if "MOBILE_NO" in renamed_df.columns:
        mob_series = renamed_df["MOBILE_NO"].fillna("").astype(str).str.strip()
        missing_mobile_numbers = int((mob_series.apply(lambda x: determine_mobile_status(x) == "Missing")).sum())
    else:
        missing_mobile_numbers = records_received

    # Duplicate records
    duplicate_records = int(renamed_df.duplicated().sum())

    # Invalid / negative quantities
    if "quantity" in renamed_df.columns:
        qty_num = pd.to_numeric(renamed_df["quantity"], errors="coerce")
        invalid_quantities = int((qty_num.isna() | (qty_num <= 0)).sum())
    else:
        invalid_quantities = records_received

    # Validation criteria: valid date, valid customer, valid item, quantity > 0
    valid_mask = pd.Series(True, index=renamed_df.index)

    if date_col in renamed_df.columns:
        valid_mask = valid_mask & renamed_df["invoice_date"].notna()
    else:
        valid_mask = pd.Series(False, index=renamed_df.index)

    if "customerId" in renamed_df.columns:
        valid_mask = valid_mask & renamed_df["customerId"].notna() & (~renamed_df["customerId"].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "-", "0"]))
    elif "customerName" in renamed_df.columns:
        renamed_df["customerId"] = renamed_df["customerName"]
        valid_mask = valid_mask & renamed_df["customerId"].notna() & (~renamed_df["customerId"].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "-", "0"]))
    else:
        valid_mask = pd.Series(False, index=renamed_df.index)

    if "itemId" in renamed_df.columns:
        valid_mask = valid_mask & renamed_df["itemId"].notna() & (~renamed_df["itemId"].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "-", "0"]))
    elif "itemName" in renamed_df.columns:
        renamed_df["itemId"] = renamed_df["itemName"]
        valid_mask = valid_mask & renamed_df["itemId"].notna() & (~renamed_df["itemId"].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "-", "0"]))
    else:
        valid_mask = pd.Series(False, index=renamed_df.index)

    if "quantity" in renamed_df.columns:
        qty_num = pd.to_numeric(renamed_df["quantity"], errors="coerce")
        valid_mask = valid_mask & qty_num.notna() & (qty_num > 0)

    valid_records = int(valid_mask.sum())
    records_requiring_review = records_received - valid_records

    # Customer base analysis (New vs Existing)
    if cust_col in renamed_df.columns and existing_history_df is not None and not existing_history_df.empty:
        exist_cust_col = "customerId" if "customerId" in existing_history_df.columns else "customerName"
        if exist_cust_col in existing_history_df.columns:
            exist_s = existing_history_df[exist_cust_col].dropna().astype(str).str.strip()
            existing_cust_set = set(exist_s[~exist_s.str.lower().isin(["", "nan", "none", "null", "0", "-"] )])
            new_customers = len(unique_cust_set - existing_cust_set)
            existing_customers = len(unique_cust_set & existing_cust_set)
        else:
            new_customers = unique_customers
            existing_customers = 0
    else:
        new_customers = unique_customers
        existing_customers = 0

    metrics = {
        "file_date_range": file_date_range,
        "records_received": records_received,
        "valid_records": valid_records,
        "records_requiring_review": records_requiring_review,
        "new_customers": new_customers,
        "existing_customers": existing_customers,
        "unique_customers": unique_customers,
        "medicines_count": medicines_count,
        "missing_customer_info": missing_customer_info,
        "missing_mobile_numbers": missing_mobile_numbers,
        "duplicate_records": duplicate_records,
        "invalid_quantities": invalid_quantities,
        "date_diagnostics": date_diagnostics,
    }

    return renamed_df, metrics


# ==============================================================================
# STREAMLIT UI RENDERER
# ==============================================================================
def render_app():
    """Main Streamlit application layout."""
    st.set_page_config(
        page_title="Pharmacy Refill Management System",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean Pharmacy Branding & Styles
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0c2461;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4b6584;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #e9ecef;
            text-align: center;
        }
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0c2461;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">💊 Pharmacy Refill Management System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Client Refill Prediction & Patient Reminder Management System</div>',
        unsafe_allow_html=True
    )

    # Load Data & Model
    with st.spinner("Loading RefillCare system data..."):
        model_bundle = load_refill_model()
        if "active_history_data" in st.session_state and st.session_state.active_history_data is not None:
            history_data = st.session_state.active_history_data
        else:
            history_data = load_purchase_history()

        if "active_sales_data" in st.session_state and st.session_state.active_sales_data is not None:
            recent_data = st.session_state.active_sales_data
        else:
            recent_data = load_refill_data()

    if model_bundle is None or recent_data is None:
        st.error("⚠️ RefillCare System Not Ready")
        st.warning("Required system data artifacts could not be loaded. Please ensure data files exist.")
        return

    # Storage & Predictions Overview
    storage = RefillCareStorage()
    if "active_eligible_df" in st.session_state and st.session_state.active_eligible_df is not None:
        eligible_df = st.session_state.active_eligible_df
        ineligible_df = st.session_state.get("active_ineligible_df", pd.DataFrame())
        metrics = {
            "total_histories": len(eligible_df) + len(ineligible_df),
            "eligible_count": len(eligible_df),
            "ineligible_count": len(ineligible_df),
        }
    else:
        eligible_df, ineligible_df, metrics = prepare_prediction_overview(recent_data, model_bundle)

    # Reminder List DataFrame
    reminder_list_df = eligible_df.copy()

    # Calculate Reminders Due Today
    today_str = date.today().strftime("%Y-%m-%d")
    if not reminder_list_df.empty and "Reminder Date" in reminder_list_df.columns:
        due_today_count = len(reminder_list_df[reminder_list_df["Reminder Date"] == today_str])
    else:
        due_today_count = 0

    # Calculate Sales Data Available Coverage and Latest Sales Date
    sales_coverage_str = "Dec 2020 – Aug 2026"
    latest_sales_date_str = "N/A"
    eval_data_for_dates = history_data if (history_data is not None and not history_data.empty) else recent_data
    if eval_data_for_dates is not None and not eval_data_for_dates.empty and "invoice_date" in eval_data_for_dates.columns:
        parsed_dts = pd.to_datetime(eval_data_for_dates["invoice_date"], errors="coerce").dropna()
        if not parsed_dts.empty:
            min_d = parsed_dts.min()
            max_d = parsed_dts.max()
            sales_coverage_str = f"{min_d.strftime('%b %Y')} – {max_d.strftime('%b %Y')}"
            latest_sales_date_str = max_d.strftime("%d-%m-%Y")

    # Top Status Bar
    col_status1, col_status2, col_status3 = st.columns([2, 2, 1])
    with col_status1:
        st.success("🟢 **System Ready:** Refill Prediction Engine Active")
    with col_status2:
        st.info(f"📅 **Sales Data Available:** {sales_coverage_str}")
    with col_status3:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            for k in ("active_sales_data", "active_history_data", "active_eligible_df", "active_ineligible_df"):
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Main Client Tabs
    tab_dash, tab_upload, tab_results, tab_reminders, tab_review, tab_whatsapp = st.tabs([
        "📊 Operations Dashboard",
        "📁 Data Update",
        "🎯 Prediction Results",
        "📅 Reminder List",
        "⚠️ Customers Needing Review",
        "💬 WhatsApp",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: OPERATIONS DASHBOARD
    # --------------------------------------------------------------------------
    with tab_dash:
        st.markdown("### 📈 Operations Summary")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Customers Monitored", f"{metrics['total_histories']:,}")
        with m2:
            st.metric("Upcoming Reminders", f"{metrics['eligible_count']:,}")
        with m3:
            st.metric("Reminders Due", f"{due_today_count:,}")
        with m4:
            st.metric("Customers Needing Review", f"{metrics['ineligible_count']:,}")
        with m5:
            st.metric("Latest Sales Date", latest_sales_date_str)

        st.markdown("---")

        # Key Business Notes
        st.markdown("#### 💡 Adherence & Refill Operations")
        st.write(
            "- **Estimated Days of Supply:** Calculated from verified quantity and historical consumption patterns.\n"
            "- **2-Day Reminder Target:** Proactive reminders are timed 2 days before the expected refill date to ensure continuous adherence.\n"
            "- **Phone Verification:** Customers without valid mobile numbers remain visible in the queue for manual outreach."
        )

    # --------------------------------------------------------------------------
    # TAB 2: DATA UPDATE (UPLOAD LATEST SALES DATA)
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown("### 📁 Upload Latest Sales Data")
        st.write("Upload a newly received monthly sales file (**CSV** or **XLSX/XLS**) to inspect transaction hygiene, validate records, and integrate updates into the customer refill queue.")

        uploaded_file = st.file_uploader(
            "Select Sales Data File (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            key="monthly_sales_uploader",
            help="Supported file types: CSV, Excel (.xlsx, .xls). Mappings support standard pharmacy transaction headers.",
        )

        if uploaded_file is not None:
            with st.spinner("Analyzing uploaded sales data..."):
                parsed_sales_df, upload_metrics = parse_and_validate_uploaded_sales_file(
                    uploaded_file,
                    existing_history_df=history_data,
                )

            if "error" in upload_metrics:
                st.error(f"❌ {upload_metrics['error']}")
            else:
                st.success(f"📁 **File Uploaded:** `{uploaded_file.name}`")

                # Section 1: Overview & Coverage
                st.markdown("#### 📊 Uploaded Sales Data Summary")
                
                date_diag = upload_metrics.get("date_diagnostics", {})
                st.info(f"📅 **Uploaded Date Range (DD-MM-YYYY):** `{upload_metrics['file_date_range']}`")

                # Detailed Date Diagnostics Panel
                with st.expander("🔍 Date Parsing & Validation Details (DD-MM-YYYY Standard)", expanded=True):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.markdown(f"**Detected Source Format:** `{date_diag.get('source_format', 'Auto')}`")
                        st.markdown(f"**Min Date (Earliest):** `{date_diag.get('min_date_formatted', '-')}`")
                    with dc2:
                        st.markdown(f"**Max Date (Latest):** `{date_diag.get('max_date_formatted', '-')}`")
                        st.markdown(f"**Valid Date Records:** `{date_diag.get('valid_count', upload_metrics['valid_records']):,}`")
                    with dc3:
                        inv_dates = date_diag.get("invalid_count", 0)
                        if inv_dates > 0:
                            st.markdown(f"**Invalid / Unparseable Dates:** ⚠️ `{inv_dates:,}`")
                        else:
                            st.markdown("**Invalid / Unparseable Dates:** `0` (Clean)")
                        samples_str = ", ".join(date_diag.get("sample_dates_formatted", [])) if date_diag.get("sample_dates_formatted") else "None"
                        st.markdown(f"**Sample Parsed Dates:** `{samples_str}`")

                    if date_diag.get("warning"):
                        st.warning(f"⚠️ **Date Notice:** {date_diag['warning']}")
                    if date_diag.get("is_ambiguous"):
                        st.error("❌ **Ambiguous Date Format:** The uploaded dates cannot be deterministically resolved to DD-MM-YYYY. Processing has been blocked.")

                # 8 Required Business & Data Hygiene Metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Number of Sales Records", f"{upload_metrics['records_received']:,}")
                with c2:
                    st.metric("Number of Customers", f"{upload_metrics['unique_customers']:,}")
                with c3:
                    st.metric("Number of Medicines", f"{upload_metrics['medicines_count']:,}")
                with c4:
                    st.metric("Valid Records", f"{upload_metrics['valid_records']:,}")

                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    st.metric("Missing Customer Info", f"{upload_metrics['missing_customer_info']:,}")
                with c6:
                    st.metric("Missing Mobile Numbers", f"{upload_metrics['missing_mobile_numbers']:,}")
                with c7:
                    st.metric("Duplicate Records", f"{upload_metrics['duplicate_records']:,}")
                with c8:
                    st.metric("Invalid/Negative Qty", f"{upload_metrics['invalid_quantities']:,}")

                st.markdown("---")
                st.markdown("#### 🔍 Data Preview (First 10 Records)")
                if parsed_sales_df is not None and not parsed_sales_df.empty:
                    preview_df = parsed_sales_df.copy()
                    if "invoice_date" in preview_df.columns:
                        preview_df["invoice_date"] = preview_df["invoice_date"].apply(format_date_dd_mm_yyyy)
                    preview_cols = [c for c in ["invoice_date", "customerId", "customerName", "itemId", "itemName", "quantity", "packing", "MOBILE_NO"] if c in preview_df.columns]
                    if not preview_cols:
                        preview_cols = list(preview_df.columns[:8])
                    st.dataframe(preview_df[preview_cols].head(10), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("#### ⚙️ Data Actions")

                st.caption(
                    "📌 **Data Policy:** Customers with missing or invalid mobile numbers are **never deleted**. "
                    "They remain in historical sales records for refill interval tracking and are only filtered when exporting delivery-ready reminder files."
                )

                is_date_blocked = date_diag.get("is_ambiguous", False) or upload_metrics["valid_records"] == 0

                btn_col1, btn_col2, btn_col3 = st.columns([1.3, 1.6, 2.1])
                with btn_col1:
                    btn_validate = st.button("🔍 Validate Data", key="btn_validate_sales_data")
                with btn_col2:
                    btn_process = st.button(
                        "🚀 Process Sales Data",
                        key="btn_process_sales_data",
                        disabled=is_date_blocked,
                        help="Disabled if date interpretation is ambiguous or no valid records exist." if is_date_blocked else "Ingest verified sales records with traceable batch ID.",
                    )
                with btn_col3:
                    btn_gen_pred = st.button("🔮 Generate Updated Predictions", key="btn_gen_predictions_upload")

                if is_date_blocked:
                    st.error("🛑 **Processing Disabled:** Date verification failed or contains unresolved ambiguity. Please check the file formatting.")

                # Handle [Validate Data]
                if btn_validate:
                    st.session_state["sales_validation_run"] = True
                    st.session_state["sales_validation_file"] = uploaded_file.name

                if st.session_state.get("sales_validation_run") and st.session_state.get("sales_validation_file") == uploaded_file.name:
                    st.markdown("##### 📋 Validation Report")
                    if upload_metrics["valid_records"] > 0:
                        st.success(
                            f"✅ **Validation Complete:** {upload_metrics['valid_records']:,} out of {upload_metrics['records_received']:,} "
                            f"records are fully validated and meet operational ingestion criteria."
                        )
                    if upload_metrics["records_requiring_review"] > 0:
                        st.warning(
                            f"⚠️ **Review Notice:** {upload_metrics['records_requiring_review']:,} records require review "
                            f"({upload_metrics['invalid_quantities']:,} non-positive/invalid quantity, "
                            f"{upload_metrics['missing_customer_info']:,} missing customer identity)."
                        )
                    st.info(
                        f"ℹ️ **Customer Mobile Breakdown:** {upload_metrics['unique_customers']:,} total customers in file. "
                        f"{upload_metrics['missing_mobile_numbers']:,} transactions have missing phone numbers (preserved in history for continuity)."
                    )

                # Handle [Process Sales Data]
                if btn_process:
                    with st.spinner("Processing sales data into RefillCare purchase history with batch ID..."):
                        process_result = process_monthly_sales_data(
                            sales_input=parsed_sales_df,
                            existing_history_df=history_data if history_data is not None else pd.DataFrame(),
                            storage=storage,
                            source_filename=uploaded_file.name,
                        )
                    if process_result.get("status") == "success":
                        st.session_state.active_history_data = process_result["updated_history_df"]
                        st.session_state.active_sales_data = process_result["updated_history_df"]
                        st.session_state["sales_processed_metrics"] = process_result["metrics"]
                        st.session_state["sales_processed_file"] = uploaded_file.name
                        st.session_state["last_import_batch_id"] = process_result.get("import_batch_id")
                        # Clear old predictions so updated ones must be generated explicitly
                        st.session_state.pop("active_eligible_df", None)
                        st.session_state.pop("active_ineligible_df", None)
                        st.session_state.pop("prediction_gen_metrics", None)
                        st.session_state.pop("sales_validation_run", None)
                        st.session_state.pop("sales_validation_file", None)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning(process_result.get("message", "Processing completed with warnings."))

                # Handle [Generate Updated Predictions] from upload
                if btn_gen_pred:
                    with st.spinner("Generating updated refill predictions using existing prediction engine..."):
                        pred_result = generate_updated_predictions(
                            history_df=history_data if history_data is not None else pd.DataFrame(),
                            model_bundle=model_bundle,
                        )
                    if pred_result.get("status") == "success":
                        st.session_state.active_eligible_df = pred_result["eligible_df"]
                        st.session_state.active_ineligible_df = pred_result["ineligible_df"]
                        st.session_state["prediction_gen_metrics"] = pred_result["metrics"]
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning(pred_result.get("message", "Prediction generation completed with warnings."))

                # Display Processing Summary if available for current file
                if st.session_state.get("sales_processed_metrics") and st.session_state.get("sales_processed_file") == uploaded_file.name:
                    p_metrics = st.session_state["sales_processed_metrics"]
                    st.markdown("---")
                    st.success("✅ **Sales Data Updated Successfully**")
                    st.markdown("#### 📋 Sales Processing Summary")
                    st.info(f"🏷️ **Import Batch ID:** `{p_metrics.get('import_batch_id', 'N/A')}`")

                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.metric("New Records Added", f"{p_metrics['new_records_added']:,}")
                    with p2:
                        st.metric("Existing Duplicates Skipped", f"{p_metrics['existing_duplicates_skipped']:,}")
                    with p3:
                        st.metric("Customers Updated", f"{p_metrics['customers_updated']:,}")
                    with p4:
                        st.metric("New Customers", f"{p_metrics['new_customers']:,}")

                    p5, p6, p7, _ = st.columns(4)
                    with p5:
                        st.metric("New Medicines", f"{p_metrics['new_medicines']:,}")
                    with p6:
                        st.metric("Latest Sales Date (DD-MM-YYYY)", str(p_metrics['latest_sales_date']))
                    with p7:
                        st.metric("Records Requiring Review", f"{p_metrics['records_requiring_review']:,}")

                # Section: Undo Last Import (Rollback)
                active_batch_id = st.session_state.get("last_import_batch_id")
                if active_batch_id:
                    st.markdown("---")
                    st.markdown("#### ↺ Undo Last Import (Rollback)")
                    st.write(
                        "Safely rollback the most recent sales data import. This operation removes only "
                        "the transaction records inserted by this batch and invalidates any predictions generated from it."
                    )
                    st.warning(f"⚠️ **Target Batch for Rollback:** `{active_batch_id}`")

                    col_chk, col_rb = st.columns([3, 1])
                    with col_chk:
                        confirm_rollback = st.checkbox(
                            f"I confirm that I want to rollback batch `{active_batch_id}` and restore the previous database state.",
                            key="chk_confirm_rollback_upload",
                        )
                    with col_rb:
                        if st.button("⚠️ Rollback Import", key="btn_rollback_upload", disabled=not confirm_rollback):
                            with st.spinner("Rolling back import batch..."):
                                rb_result = rollback_monthly_sales_import(
                                    current_history_df=history_data,
                                    import_batch_id=active_batch_id,
                                    storage=storage,
                                )
                            if rb_result["status"] == "success":
                                st.session_state.active_history_data = rb_result["restored_history_df"]
                                st.session_state.active_sales_data = rb_result["restored_history_df"]
                                st.session_state.pop("last_import_batch_id", None)
                                st.session_state.pop("sales_processed_metrics", None)
                                st.session_state.pop("sales_processed_file", None)
                                st.session_state.pop("active_eligible_df", None)
                                st.session_state.pop("active_ineligible_df", None)
                                st.session_state.pop("prediction_gen_metrics", None)
                                st.cache_data.clear()
                                st.success(f"✅ {rb_result['message']} Latest Sales Date restored to `{rb_result['restored_latest_sales_date']}`.")
                                st.rerun()
                            else:
                                st.error(rb_result["message"])
        else:
            st.info("👆 Upload an Excel or CSV sales file above to view the business summary and update predictions.")

            # Display current active file summary if available
            st.markdown("#### 📋 Current Active File Status")
            if eval_data_for_dates is not None and not eval_data_for_dates.empty:
                d_col = "invoice_date" if "invoice_date" in eval_data_for_dates.columns else None
                if d_col:
                    d_parsed = pd.to_datetime(eval_data_for_dates[d_col], errors="coerce").dropna()
                    curr_range = f"{d_parsed.min().strftime(UI_DATE_FORMAT)} to {d_parsed.max().strftime(UI_DATE_FORMAT)}" if not d_parsed.empty else "N/A"
                else:
                    curr_range = "N/A"

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sales Date Range (DD-MM-YYYY)", curr_range)
                with c2:
                    st.metric("Total Transactions", f"{len(eval_data_for_dates):,}")
                with c3:
                    st.metric("Monitored Customers", f"{metrics['total_histories']:,}")

            # Refill Prediction Updates for active history
            st.markdown("---")
            st.markdown("#### 🔮 Refill Prediction Updates")
            st.write("Generate updated expected refill dates and reminder schedules based on the latest historical purchase records.")
            if st.button("🔮 Generate Updated Predictions", key="btn_generate_updated_predictions_main"):
                with st.spinner("Generating updated refill predictions using existing prediction engine..."):
                    pred_result = generate_updated_predictions(
                        history_df=history_data if history_data is not None else pd.DataFrame(),
                        model_bundle=model_bundle,
                    )
                if pred_result.get("status") == "success":
                    st.session_state.active_eligible_df = pred_result["eligible_df"]
                    st.session_state.active_ineligible_df = pred_result["ineligible_df"]
                    st.session_state["prediction_gen_metrics"] = pred_result["metrics"]
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(pred_result.get("message", "Prediction generation completed with warnings."))

            # Rollback active batch from database if one exists
            latest_db_batch = storage.get_latest_active_import_batch()
            if latest_db_batch and history_data is not None and "import_batch_id" in history_data.columns:
                batch_id = latest_db_batch["import_batch_id"]
                if (history_data["import_batch_id"].astype(str) == batch_id).any():
                    st.markdown("---")
                    st.markdown("#### ↺ Undo Last Import (Rollback)")
                    st.write(
                        f"Active import batch found in database: **`{batch_id}`** "
                        f"(Source: `{latest_db_batch['source_filename']}`, Records: {latest_db_batch['records_inserted']:,}, "
                        f"Date Range: {latest_db_batch['detected_date_range']})."
                    )
                    confirm_main_rb = st.checkbox(
                        f"I confirm that I want to rollback batch `{batch_id}`.",
                        key="chk_confirm_main_rb",
                    )
                    if st.button("⚠️ Rollback Import Batch", key="btn_main_rollback", disabled=not confirm_main_rb):
                        with st.spinner("Rolling back import batch..."):
                            rb_result = rollback_monthly_sales_import(
                                current_history_df=history_data,
                                import_batch_id=batch_id,
                                storage=storage,
                            )
                        if rb_result["status"] == "success":
                            st.session_state.active_history_data = rb_result["restored_history_df"]
                            st.session_state.active_sales_data = rb_result["restored_history_df"]
                            st.session_state.pop("active_eligible_df", None)
                            st.session_state.pop("active_ineligible_df", None)
                            st.session_state.pop("prediction_gen_metrics", None)
                            st.cache_data.clear()
                            st.success(f"✅ {rb_result['message']}")
                            st.rerun()
                        else:
                            st.error(rb_result["message"])

        # Display Prediction Generation Summary if available
        if st.session_state.get("prediction_gen_metrics"):
            gen_m = st.session_state["prediction_gen_metrics"]
            st.markdown("---")
            st.success("✅ **Refill Predictions Generated Successfully**")
            st.markdown("#### 🔮 Prediction Generation Summary")

            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric("Customers Evaluated", f"{gen_m['customers_evaluated']:,}")
            with g2:
                st.metric("Predictions Generated", f"{gen_m['predictions_generated']:,}")
            with g3:
                st.metric("Customers Requiring Review", f"{gen_m['customers_requiring_review']:,}")

            g4, g5 = st.columns(2)
            with g4:
                st.metric("Customers Not Eligible for Automatic Prediction", f"{gen_m['customers_not_eligible']:,}")
            with g5:
                st.metric("Predictions Requiring a Valid Mobile Number for Reminder Delivery", f"{gen_m['missing_mobile_predictions']:,}")

    # --------------------------------------------------------------------------
    # TAB: PREDICTION RESULTS (INTERNAL EVALUATION & OUTCOME ACCURACY)
    # --------------------------------------------------------------------------
    with tab_results:
        st.markdown("### 🎯 Prediction Results")
        st.write("Comparison between calculated expected refill dates and actual customer purchase dates from subsequent sales.")

        # Load evaluated prediction results from storage
        eval_results = storage.get_evaluated_prediction_results()

        # If session state has cached evaluated outcomes from newly uploaded sales, blend or use them
        if st.session_state.get("latest_evaluation_results"):
            sess_eval = st.session_state["latest_evaluation_results"]
            if sess_eval.get("predictions_evaluated", 0) > 0:
                eval_results = sess_eval

        # Display the 4 client-facing summary metrics:
        # 1. Predictions evaluated
        # 2. Predictions within 3 days
        # 3. Predictions within 7 days
        # 4. Average difference between expected and actual refill
        st.markdown("#### 📊 Accuracy & Outcomes")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Predictions Evaluated", f"{eval_results['predictions_evaluated']:,}")
        with r2:
            st.metric(
                "Predictions within 3 Days",
                f"{eval_results['within_3_days_count']:,}",
                f"{eval_results['within_3_days_pct']:.1f}%",
            )
        with r3:
            st.metric(
                "Predictions within 7 Days",
                f"{eval_results['within_7_days_count']:,}",
                f"{eval_results['within_7_days_pct']:.1f}%",
            )
        with r4:
            st.metric(
                "Average Difference",
                f"{eval_results['mean_absolute_error']:.1f} days",
                help="Average difference between expected and actual refill (Mean Absolute Error)",
            )

        st.markdown("---")

        if eval_results.get("predictions_evaluated", 0) > 0 and not eval_results["evaluated_df"].empty:
            st.markdown("#### 📋 Evaluated Refill Outcomes")

            f_c1, f_c2 = st.columns(2)
            with f_c1:
                search_eval_cust = st.text_input("Search Customer Name", key="eval_cust_search")
            with f_c2:
                search_eval_med = st.text_input("Search Medication", key="eval_med_search")

            df_display = eval_results["evaluated_df"].copy()
            if search_eval_cust.strip():
                df_display = df_display[
                    df_display["Customer Name"].astype(str).str.lower().str.contains(search_eval_cust.strip().lower())
                ]
            if search_eval_med.strip():
                df_display = df_display[
                    df_display["Medication"].astype(str).str.lower().str.contains(search_eval_med.strip().lower())
                ]

            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info(
                "ℹ️ **No completed refill cycles to evaluate yet.**\n\n"
                "- Expected refill dates are compared against actual purchases as new monthly sales files arrive.\n"
                "- Customers who have not purchased yet remain pending and are not treated as prediction failures."
            )

    # --------------------------------------------------------------------------
    # TAB 4: REMINDER LIST (DATE SELECTOR & DOWNLOAD REMINDER LIST CSV)
    # --------------------------------------------------------------------------
    with tab_reminders:
        st.markdown("### 📅 Customer Reminder List")
        st.write("Select a **Reminder Date** to view all customers scheduled for refill outreach on that day.")

        # Summary before download: Eligible for Reminder, Missing Mobile Number, Needs Review, Not Eligible
        eligible_for_rem_count = int((reminder_list_df["Mobile Status"] == "Valid").sum()) if not reminder_list_df.empty and "Mobile Status" in reminder_list_df.columns else 0
        missing_mobile_rem_count = int((reminder_list_df["Mobile Status"] != "Valid").sum()) if not reminder_list_df.empty and "Mobile Status" in reminder_list_df.columns else 0
        needs_review_count = int((ineligible_df["Purchase Count"] >= 2).sum()) if not ineligible_df.empty and "Purchase Count" in ineligible_df.columns else len(ineligible_df)
        not_eligible_count = int((ineligible_df["Purchase Count"] < 2).sum()) if not ineligible_df.empty and "Purchase Count" in ineligible_df.columns else 0

        st.markdown("#### 📊 Reminder Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Eligible for Reminder", f"{eligible_for_rem_count:,}")
        with s2:
            st.metric("Missing Mobile Number", f"{missing_mobile_rem_count:,}")
        with s3:
            st.metric("Needs Review", f"{needs_review_count:,}")
        with s4:
            st.metric("Not Eligible", f"{not_eligible_count:,}")

        st.markdown("---")

        # Multi-stage schedule vs Single Primary lead-time schedule
        full_schedules_df = generate_reminder_schedule_table(eligible_df)

        # Filters Row 1: Mode & Date Selector
        c_mode, c_date = st.columns([3, 2])
        with c_mode:
            schedule_mode = st.selectbox(
                "Schedule Mode / Stage View",
                [
                    "All Active Multi-Stage Triggers (-7d, -3d, -1d, 0d, +2d, +5d)",
                    "Primary Lead Time (-2 Days Buffer)",
                    "Stage: -7 days (Due in 7 Days)",
                    "Stage: -3 days (Due in 3 Days)",
                    "Stage: -1 day (Due Tomorrow)",
                    "Stage: 0 days (Due Today)",
                    "Stage: +2 days (Follow-up)",
                    "Stage: +5 days (Follow-up)",
                ],
                key="reminder_schedule_mode_selector",
                help="Switch between all 6 active reminder stage triggers or the single primary 2-day lead-time buffer.",
            )

        # Determine target base dataframe and available dates based on mode
        if schedule_mode == "Primary Lead Time (-2 Days Buffer)":
            base_df = reminder_list_df.copy()
            show_stage_col = False
        elif schedule_mode.startswith("Stage:"):
            stage_name = schedule_mode.split(":")[1].split("(")[0].strip()
            base_df = full_schedules_df[full_schedules_df["Reminder Stage"] == stage_name].copy() if not full_schedules_df.empty else pd.DataFrame()
            show_stage_col = True
        else:
            base_df = full_schedules_df.copy()
            show_stage_col = True

        all_rem_dates: List[date] = []
        if not base_df.empty and "Reminder Date" in base_df.columns:
            all_rem_dates = (
                pd.to_datetime(base_df["Reminder Date"], errors="coerce")
                .dropna()
                .dt.date
                .tolist()
            )

        today_val = date.today()
        if today_val in all_rem_dates:
            default_date = today_val
        elif date(2026, 9, 21) in all_rem_dates:
            default_date = date(2026, 9, 21)
        elif all_rem_dates:
            default_date = min(all_rem_dates)
        else:
            default_date = today_val

        with c_date:
            selected_date = st.date_input("Reminder Date", value=default_date, key="reminder_date_selector")

        # Filters Row 2: Customer, Medication, Mobile Status
        c_cust, c_med, c_stat = st.columns([2, 2, 1.5])
        with c_cust:
            search_cust = st.text_input("Search Customer Name", key="rem_search_cust")
        with c_med:
            search_med = st.text_input("Search Medication", key="rem_search_med")
        with c_stat:
            status_filter = st.selectbox("Mobile Status", ["All", "Valid", "Missing", "Invalid format"], key="rem_status_filter")

        selected_date_str = selected_date.strftime("%Y-%m-%d")

        # Filter by selected reminder date
        if not base_df.empty and "Reminder Date" in base_df.columns:
            filtered_reminders = base_df[base_df["Reminder Date"] == selected_date_str].copy()
        else:
            filtered_reminders = pd.DataFrame()

        # Search Filters
        if search_cust.strip() and not filtered_reminders.empty:
            col_c = "Customer Name" if "Customer Name" in filtered_reminders.columns else "Customer"
            filtered_reminders = filtered_reminders[
                filtered_reminders[col_c].astype(str).str.lower().str.contains(search_cust.strip().lower())
            ]

        if search_med.strip() and not filtered_reminders.empty:
            col_m = "Medication" if "Medication" in filtered_reminders.columns else "Medicine"
            filtered_reminders = filtered_reminders[
                filtered_reminders[col_m].astype(str).str.lower().str.contains(search_med.strip().lower())
            ]

        if status_filter != "All" and not filtered_reminders.empty and "Mobile Status" in filtered_reminders.columns:
            filtered_reminders = filtered_reminders[
                filtered_reminders["Mobile Status"] == status_filter
            ]

        # Display Columns
        if show_stage_col:
            display_columns = [
                "Customer Name",
                "Mobile Number",
                "Medication",
                "Last Purchase Date",
                "Estimated Days of Supply",
                "Expected Refill Date",
                "Reminder Date",
                "Reminder Stage",
                "Mobile Status",
            ]
        else:
            display_columns = [
                "Customer Name",
                "Mobile Number",
                "Medication",
                "Last Purchase Date",
                "Estimated Days of Supply",
                "Expected Refill Date",
                "Reminder Date",
                "Mobile Status",
            ]

        # Ensure all display columns exist in dataframe
        if not filtered_reminders.empty:
            if "Medication" not in filtered_reminders.columns and "Medicine" in filtered_reminders.columns:
                filtered_reminders["Medication"] = filtered_reminders["Medicine"]
            if "Customer Name" not in filtered_reminders.columns and "Customer" in filtered_reminders.columns:
                filtered_reminders["Customer Name"] = filtered_reminders["Customer"]
            if "Mobile Number" not in filtered_reminders.columns and "Delivery Phone" in filtered_reminders.columns:
                filtered_reminders["Mobile Number"] = filtered_reminders["Delivery Phone"]

            avail_cols = [c for c in display_columns if c in filtered_reminders.columns]
            st.dataframe(
                filtered_reminders[avail_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No customer reminders scheduled for **{selected_date.strftime('%d-%m-%Y')}** under the selected mode.")

        # Download Reminder List Button
        col_csv, col_info = st.columns([1.8, 3.2])
        with col_csv:
            reminder_csv_bytes = build_reminder_list_csv(filtered_reminders)
            st.download_button(
                label="📥 Download Reminder List",
                data=reminder_csv_bytes,
                file_name=f"reminder_list_{selected_date_str}.csv",
                mime="text/csv",
                help="Download operational reminder delivery CSV containing only valid mobile numbers.",
            )
        with col_info:
            valid_count_on_date = len(filtered_reminders[filtered_reminders["Mobile Status"] == "Valid"]) if not filtered_reminders.empty and "Mobile Status" in filtered_reminders.columns else 0
            st.write(
                f"**{valid_count_on_date:,}** delivery-ready reminder records scheduled for **{selected_date.strftime('%d-%m-%Y')}** (records without valid mobile numbers are excluded and available in Review tab)."
            )

    # --------------------------------------------------------------------------
    # TAB 4: CUSTOMERS NEEDING REVIEW
    # --------------------------------------------------------------------------
    with tab_review:
        st.markdown("### ⚠️ Customers Needing Review")
        st.write("Patients requiring purchase history review, manual verification, or missing phone number update.")

        review_rows = []
        if not ineligible_df.empty:
            for _, r in ineligible_df.iterrows():
                review_rows.append({
                    "customerId": str(r.get("customerId", "")),
                    "Customer Name": str(r.get("Customer Name", r.get("customerName", "Unknown"))),
                    "Mobile Number": str(r.get("Mobile Number", r.get("MOBILE_NO", ""))).strip(),
                    "Mobile Status": str(r.get("Mobile Status", determine_mobile_status(r.get("Mobile Number", "")))),
                    "Medication": str(r.get("Medication", r.get("Medicine", r.get("itemName", "Unknown")))),
                    "Last Purchase Date": str(r.get("Last Purchase Date", "-")),
                    "Review Reason": map_reason_client_friendly(str(r.get("Reason for Ineligibility", "Review Required"))),
                })

        # Append eligible predictions with missing mobile numbers
        if not eligible_df.empty and "Mobile Status" in eligible_df.columns:
            missing_mob_eligible = eligible_df[eligible_df["Mobile Status"] != "Valid"]
            for _, r in missing_mob_eligible.iterrows():
                review_rows.append({
                    "customerId": str(r.get("customerId", "")),
                    "Customer Name": str(r.get("Customer Name", r.get("customerName", "Unknown"))),
                    "Mobile Number": str(r.get("Mobile Number", r.get("MOBILE_NO", ""))).strip(),
                    "Mobile Status": str(r.get("Mobile Status", "Missing")),
                    "Medication": str(r.get("Medication", r.get("Medicine", r.get("itemName", "Unknown")))),
                    "Last Purchase Date": str(r.get("Last Purchase Date", "-")),
                    "Review Reason": "Missing / invalid mobile number (excluded from automated delivery list)",
                })

        if review_rows:
            combined_review_df = pd.DataFrame(review_rows)
            # Filter options for review queue
            rev_filter = st.selectbox(
                "Filter Review Category",
                ["All Records", "Missing Mobile Numbers", "History Review Required", "Cold-Start / Single Purchase"],
                key="review_cat_filter"
            )

            if rev_filter == "Missing Mobile Numbers":
                filtered_review_df = combined_review_df[combined_review_df["Mobile Status"] != "Valid"]
            elif rev_filter == "History Review Required":
                filtered_review_df = combined_review_df[combined_review_df["Review Reason"].str.contains("Review|verification|variance|Refill interval", case=False)]
            elif rev_filter == "Cold-Start / Single Purchase":
                filtered_review_df = combined_review_df[combined_review_df["Review Reason"].str.contains("single|cold-start|< 2", case=False)]
            else:
                filtered_review_df = combined_review_df

            st.dataframe(
                filtered_review_df[["Customer Name", "Mobile Number", "Mobile Status", "Medication", "Last Purchase Date", "Review Reason"]],
                use_container_width=True,
                hide_index=True,
            )

            buf = io.StringIO()
            filtered_review_df.to_csv(buf, index=False)
            st.download_button(
                label="📥 Export Review Records (CSV)",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"customers_needing_review_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No customer records currently requiring review.")

    # --------------------------------------------------------------------------
    # TAB 5: WHATSAPP (ON HOLD)
    # --------------------------------------------------------------------------
    with tab_whatsapp:
        st.markdown("### 💬 WhatsApp Refill Reminders")
        st.warning(
            "⚠️ **WhatsApp automated dispatch is currently ON HOLD.**\n\n"
            "Please use the **Reminder List** tab to select reminder dates and download the customer CSV for phone calls or manual outreach."
        )


if __name__ == "__main__":
    render_app()
