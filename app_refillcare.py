"""RefillCare — Medication Refill Reminder System (Streamlit UI).

Pharmacy-centric operational interface for reviewing validated refill predictions,
managing date-filtered reminder lists, uploading monthly sales updates,
and exporting customer CSVs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple, Optional, List, Union
import io
import math
import warnings

# Suppress serialization and version mismatch warnings for cross-version compatibility
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*unpickle estimator.*")

# Silence noisy network polling logs from Supabase HTTP client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

# Pharmacist-facing tabs (7 Nov 2026): technical tabs hidden from the store app. True = shown.
TAB_VISIBILITY = {
    "results": False,  # "Upcoming Refills" (forecast explorer + accuracy panel)
    "ml": False,       # "AI Accuracy & Settings" (retrain / promote / rollback - admin only, via API / scripts)
}


def _visible_tab(tab):
    """Run a tab's block only when the tab is shown: ``for _ in _visible_tab(tab): ...``."""
    if tab is not None:
        with tab:
            yield


def store_cache_data(**cache_kwargs):
    """``st.cache_data`` whose cache key also includes the current store.

    Each store has its own database; without the store in the key, switching between two stores of
    the same tenant would show the first store's cached numbers for up to the TTL.
    """
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def _keyed(store_key: str, *args, **kwargs):
            return fn(*args, **kwargs)

        cached = st.cache_data(**cache_kwargs)(_keyed)

        @functools.wraps(fn)
        def call(*args, **kwargs):
            from refillcare.tenancy import current_store_id

            return cached(current_store_id(), *args, **kwargs)

        call.clear = cached.clear
        return call

    return deco


def _is_default_store() -> bool:
    """Legacy parquet / CSV files on disk belong to the original store only."""
    from refillcare.tenancy import current_store_id, default_store_id

    return current_store_id() == default_store_id()


STORE_SCOPED_SESSION_KEYS = ("active_sales_data", "active_history_data", "active_eligible_df", "active_ineligible_df",
                             "wa_last_send", "bulk_add_msg")


def apply_store_from_request() -> str:
    """Store of this browser session: ``?store_id=`` in the link Mediastra opens (else REFILLCARE_STORE_ID).

    In production only ids listed in REFILLCARE_ALLOWED_STORES are accepted (Mediastra handles the
    login; this stops a typed URL from opening / creating another store). When the store changes,
    the previous store's in-session data is dropped.
    """
    from refillcare.tenancy import InvalidStoreId, default_store_id, is_production, normalise_store_id, set_current_store

    requested = st.query_params.get("store_id") or st.session_state.get("_store_id") or default_store_id()
    tenant = st.query_params.get("tenant_id") or st.session_state.get("_tenant_id")
    try:
        sid = normalise_store_id(requested)
    except InvalidStoreId:
        st.error("Invalid store link. Please open RefillCare again from Mediastra.")
        st.stop()
    allowed = {s.strip().lower() for s in os.getenv("REFILLCARE_ALLOWED_STORES", "").split(",") if s.strip()}
    if (allowed and sid not in allowed) or (not allowed and is_production() and sid != default_store_id()):
        st.error("This store is not enabled for RefillCare. Please contact Mediastra support.")
        st.stop()
    if st.session_state.get("_store_id") not in (None, sid):
        for k in STORE_SCOPED_SESSION_KEYS:
            st.session_state.pop(k, None)
    st.session_state["_store_id"], st.session_state["_tenant_id"] = sid, tenant
    return set_current_store(sid, tenant)


DEFAULT_HUMAN_MODEL_PATH = "data/refillcare/processed/models/human_ml_refill_model.joblib"
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
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


MEDIASTRA_LOGO_PATH = find_artifact_path("assets/images/MediAstra_Logo.png")
MEDIASTRA_ICON_PATH = find_artifact_path("assets/images/MediAstra_Icon.png")


@st.cache_data
def get_mediastra_logo_base64() -> str:
    """Load MediAstra logo as base64 string for embedded HTML rendering."""
    target = find_artifact_path("assets/images/MediAstra_Logo.png")
    if target.exists():
        import base64
        return base64.b64encode(target.read_bytes()).decode("utf-8")
    return ""


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
def load_human_refill_model(model_path: Optional[Union[Path, str]] = None) -> Optional[Dict[str, Any]]:
    """Load the human consensus model bundle."""
    target_path = Path(model_path) if model_path else find_artifact_path(DEFAULT_HUMAN_MODEL_PATH)
    if not target_path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(target_path)
    except Exception as e:
        st.error(f"Error loading human model bundle: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_refill_data(data_path: Optional[Union[Path, str]] = None) -> Optional[pd.DataFrame]:
    """Load the processed evaluation dataset or canonical purchase history."""
    if data_path is not None:
        target_path = Path(data_path)
        if not target_path.exists():
            return None
    else:
        target_path = find_artifact_path(DEFAULT_TEST_DATA_PATH)
        if not target_path.exists():
            target_path = find_artifact_path(DEFAULT_HISTORY_DATA_PATH)
            if not target_path.exists():
                fallback_train = find_artifact_path(DEFAULT_TRAIN_DATA_PATH)
                if fallback_train.exists():
                    target_path = fallback_train
                else:
                    return None
    try:
        return pd.read_parquet(target_path)
    except Exception:
        return None


def _store_history_version(store_id: str) -> tuple:
    """(rows, max id) of the store's sales history: changes whenever an upload or rollback lands."""
    from database.connection import SessionLocal
    from refillcare.data import history_store as hs

    with SessionLocal(store_id) as db:
        return hs.history_version(db)


@st.cache_data(show_spinner=False, max_entries=4)
def load_store_history(store_id: str, version: tuple) -> Optional[pd.DataFrame]:
    """The store's purchase history from its database (cache key: store + data version)."""
    from database.connection import SessionLocal
    from refillcare.data import history_store as hs

    if not version or version[0] == 0:
        return None
    with SessionLocal(store_id) as db:
        return hs.load_all(db, store_id=store_id, with_history_columns=True)


def current_history() -> Optional[pd.DataFrame]:
    """History of the selected store: database when it has rows, else the legacy parquet."""
    from refillcare.tenancy import current_store_id, default_store_id

    sid = current_store_id()
    df = load_store_history(sid, _store_history_version(sid))
    if df is not None:
        return df
    return load_purchase_history() if sid == default_store_id() else None


def latest_active_batch(storage: Any) -> Optional[Dict[str, Any]]:
    """Latest active upload of the selected store (database), else the legacy storage record."""
    from database.connection import SessionLocal
    from database.models import ImportBatchModel

    with SessionLocal() as db:
        b = (db.query(ImportBatchModel).filter(ImportBatchModel.is_active.is_(True), ImportBatchModel.import_batch_id != "LEGACY_HISTORY")
             .order_by(ImportBatchModel.upload_timestamp.desc()).first())
        if b is not None:
            return {"import_batch_id": b.import_batch_id, "source_filename": b.source_filename,
                    "records_inserted": b.records_inserted or 0, "detected_date_range": b.detected_date_range or "-"}
    return storage.get_latest_active_import_batch()


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


@st.cache_data(show_spinner=False)
def _count_lines_in_csv_file(file_path: str, mtime: float, size: int) -> int:
    """Fast buffer line counter cached by exact file mtime and byte size."""
    try:
        with open(file_path, "rb") as f:
            cnt = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))
            if cnt > 0:
                return max(0, cnt - 1)
    except Exception:
        pass
    return 0


def get_master_raw_transaction_count(_active_data_override: Optional[int] = None) -> int:
    """Dynamically count master raw transactions from customer_data_fields.csv.
    Updates immediately when new records are added on disk or uploaded in-session."""
    if _active_data_override and _active_data_override > 0:
        return _active_data_override

    csv_file = find_artifact_path("data/refillcare/customer_data_fields.csv")
    if csv_file.exists():
        try:
            stat = csv_file.stat()
            return _count_lines_in_csv_file(str(csv_file), stat.st_mtime, stat.st_size)
        except Exception:
            pass
    return 0  # no master file yet - never show a stale hard-coded count


@st.cache_data(show_spinner=False)
def _read_clean_channel_metrics_cached(file_path: str, mtime: float, size: int) -> Dict[str, int]:
    """Dynamically parse channel distribution from clean_transactions.parquet, cached by file stat."""
    tot_clean = cust_sales = b2b_sales = refill_el = 0
    try:
        df_c_tx = pd.read_parquet(file_path, columns=["transaction_type", "refillcare_eligible"])
        chan_col = "transaction_type" if "transaction_type" in df_c_tx.columns else None
        if chan_col:
            cust_sales = int((df_c_tx[chan_col] == "CUSTOMER_SALE").sum())
            b2b_sales = int((df_c_tx[chan_col] == "B2B_INTER_STORE").sum())
            refill_el = int(df_c_tx["refillcare_eligible"].sum()) if "refillcare_eligible" in df_c_tx.columns else cust_sales
            tot_clean = len(df_c_tx)
    except Exception:
        try:
            df_c_tx = pd.read_parquet(file_path)
            chan_col = "transaction_type" if "transaction_type" in df_c_tx.columns else ("transaction_channel" if "transaction_channel" in df_c_tx.columns else None)
            if chan_col:
                cust_sales = int((df_c_tx[chan_col] == "CUSTOMER_SALE").sum())
                b2b_sales = int((df_c_tx[chan_col] == "B2B_INTER_STORE").sum())
                refill_el = int(df_c_tx["refillcare_eligible"].sum()) if "refillcare_eligible" in df_c_tx.columns else cust_sales
                tot_clean = len(df_c_tx)
        except Exception:
            pass
    return {
        "tot_clean": tot_clean,
        "cust_sales": cust_sales,
        "b2b_sales": b2b_sales,
        "refill_el": refill_el,
    }


@store_cache_data(show_spinner=False, ttl=15)
def get_enterprise_channel_distribution_cached() -> Dict[str, int]:
    """Sales-line counts of the CURRENT store, from its own database (updates on every upload / rollback).

    raw_tx_cnt = lines received in the store's active uploads; tot_clean = lines stored after removing
    duplicates; the store's legacy parquet is used only for the original store before any DB import.
    """
    try:
        from database.connection import SessionLocal
        from database.models import SalesTransactionModel as Line, ImportBatchModel
        from sqlalchemy import func, select
        from refillcare.data import history_store as hs

        with SessionLocal() as db:
            if hs.history_count(db) > 0:
                c = hs.channel_counts(db)  # same function as the API
                return {
                    "raw_tx_cnt": c["received"],
                    "tot_clean": c["sales_lines"],
                    "cust_sales": c["customer_sales"],
                    "b2b_sales": c["b2b_inter_store"],
                    "refill_el": c["refillcare_eligible"],
                    "purged_cnt": c["not_stored"],
                    "returns": {**c, "text": hs.return_breakdown_text(c)},
                }
    except Exception:
        pass

    if _is_default_store():
        clean_tx_file = find_artifact_path("data/refillcare/processed/clean_transactions.parquet")
        if clean_tx_file.exists():
            try:
                stat = clean_tx_file.stat()
                m = _read_clean_channel_metrics_cached(str(clean_tx_file), stat.st_mtime, stat.st_size)
                raw = max(get_master_raw_transaction_count() or 0, m["tot_clean"])
                return {
                    "raw_tx_cnt": raw,
                    "tot_clean": m["tot_clean"],
                    "cust_sales": m["cust_sales"],
                    "b2b_sales": m["b2b_sales"],
                    "refill_el": m["refill_el"],
                    "purged_cnt": raw - m["tot_clean"],
                }
            except Exception:
                pass
    return {"raw_tx_cnt": 0, "tot_clean": 0, "cust_sales": 0, "b2b_sales": 0, "refill_el": 0, "purged_cnt": 0}


@store_cache_data(show_spinner=False, ttl=60)
def load_store_b2b_lines(limit: int = 200) -> Tuple[int, pd.DataFrame]:
    """(total, latest rows) of the CURRENT store's B2B / inter-store lines (excluded from reminders)."""
    cols = ["invoice_number", "invoice_date", "customerId", "customerName", "MOBILE_NO", "itemId", "itemName",
            "quantity", "transaction_type", "refillcare_eligible", "exclusion_reason"]
    try:
        from database.connection import SessionLocal
        from database.models import SalesTransactionModel as Line
        from refillcare.data import history_store as hs

        with SessionLocal() as db:
            if hs.history_count(db) > 0:
                q = db.query(Line).filter(Line.transaction_type == "B2B_INTER_STORE")
                total = q.count()
                rows = q.order_by(Line.invoice_date.desc()).limit(limit).all()
                return total, pd.DataFrame([[r.invoice_number, r.invoice_date, r.customer_id, r.customer_name, r.mobile_no,
                                             r.item_id, r.item_name, r.quantity, r.transaction_type, r.refillcare_eligible,
                                             r.exclusion_reason] for r in rows], columns=cols)
    except Exception:
        pass
    if _is_default_store():
        clean_tx_file = find_artifact_path("data/refillcare/processed/clean_transactions.parquet")
        if clean_tx_file.exists():
            all_tx = pd.read_parquet(clean_tx_file)
            b2b = all_tx[all_tx["transaction_type"] == "B2B_INTER_STORE"]
            return len(b2b), pd.DataFrame(b2b.head(limit)).reindex(columns=cols)
    return 0, pd.DataFrame(columns=cols)


def get_clean_transaction_channel_metrics() -> Dict[str, int]:
    """Dynamically compute channel metrics from enterprise database or fallback."""
    dist = get_enterprise_channel_distribution_cached()
    return {
        "tot_clean": dist["tot_clean"],
        "cust_sales": dist["cust_sales"],
        "b2b_sales": dist["b2b_sales"],
        "refill_el": dist["refill_el"],
    }


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


def format_display_phone_10digits(phone: Any) -> str:
    """Format phone number to standard 10-digit display (strips leading 91 or 0 if valid 10-digit base)."""
    if phone is None or pd.isna(phone):
        return "-"
    s = str(phone).strip()
    if not s or s.lower() in ("nan", "none", "-", "null", "n/a", "0", ""):
        return "-"
    digits = "".join(filter(str.isdigit, s))
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return digits[2:]
    elif len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return digits[1:]
    elif len(digits) == 10 and digits[0] in "6789":
        return digits
    elif len(digits) >= 10:
        return digits[-10:]
    return s if s else "-"



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
            filtered = pd.DataFrame(filtered[filtered[col].astype(str).str.lower().str.contains(q, regex=False, na=False)])

    if medicine_query.strip():
        q = medicine_query.strip().lower()
        col = "Medicine" if "Medicine" in filtered.columns else "Medication"
        if col in filtered.columns:
            filtered = pd.DataFrame(filtered[filtered[col].astype(str).str.lower().str.contains(q, regex=False, na=False)])

    if quality_filter != "All" and "History Quality" in filtered.columns:
        filtered = pd.DataFrame(filtered[filtered["History Quality"] == quality_filter])

    if tier_filter != "All" and "Pilot Tier" in filtered.columns:
        filtered = pd.DataFrame(filtered[filtered["Pilot Tier"] == tier_filter])

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_d, end_d = str(date_range[0]), str(date_range[1])
        if "Expected Refill Date" in filtered.columns:
            filtered = pd.DataFrame(filtered[
                (filtered["Expected Refill Date"] >= start_d) &
                (filtered["Expected Refill Date"] <= end_d)
            ])

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
            filtered = pd.DataFrame(filtered[filtered[col].astype(str).str.lower().str.contains(q, regex=False, na=False)])

    if medicine_query.strip():
        q = medicine_query.strip().lower()
        col = "Medicine" if "Medicine" in filtered.columns else "Medication"
        if col in filtered.columns:
            filtered = pd.DataFrame(filtered[filtered[col].astype(str).str.lower().str.contains(q, regex=False, na=False)])

    if stage_filter != "All" and "Reminder Stage" in filtered.columns:
        filtered = pd.DataFrame(filtered[filtered["Reminder Stage"] == stage_filter])

    if status_filter != "All" and "Status" in filtered.columns:
        filtered = pd.DataFrame(filtered[filtered["Status"] == status_filter])

    if tier_filter != "All" and "Pilot Tier" in filtered.columns:
        filtered = pd.DataFrame(filtered[filtered["Pilot Tier"] == tier_filter])

    if target_date is not None:
        target_str = str(target_date)
        if "raw_reminder_date" in filtered.columns:
            filtered = pd.DataFrame(filtered[filtered["raw_reminder_date"] == target_str])
        elif "Reminder Date" in filtered.columns:
            filtered = pd.DataFrame(filtered[filtered["Reminder Date"] == target_str])

    return filtered


def map_reason_client_friendly(reason: str) -> str:
    """Map technical evaluation reasons into clean client-facing terminology."""
    r = str(reason).lower()
    if "returned all strips" in r:
        return str(reason)  # late full return: date + "therapy stopped or changed?" are what the pharmacist needs
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
    if df is None or df.empty:
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
    working_df = pd.DataFrame(df.sort_values("invoice_date").groupby(["customerId", "itemId"], as_index=False).last())

    # Identify single purchase histories
    p_counts = pd.Series(working_df.get("purchase_count_so_far", working_df.get("purchase_seq", 1)))
    is_cold_start = p_counts.fillna(1).astype(int) < 2

    raw_candidates = pd.DataFrame(working_df[~is_cold_start].copy())
    raw_cold_start = pd.DataFrame(working_df[is_cold_start].copy())

    # Generate refill predictions for multi-purchase candidates
    if not raw_candidates.empty:
        preds_df = generate_batch_predictions(model_bundle, raw_candidates)
    else:
        preds_df = raw_candidates

    # Evaluate eligibility
    eligible_rows = []
    ineligible_rows = []

    from refillcare.data.medication_classifier import classify_medication

    for _, row in preds_df.iterrows():
        rec_dict = row.to_dict()
        med_val = str(rec_dict.get("itemName", rec_dict.get("itemId", "")))
        pack_val = str(rec_dict.get("packing", ""))
        med_class = classify_medication(med_val, packing=pack_val)
        if not med_class["is_chronic_eligible"]:
            rec_dict["Reason for Ineligibility"] = f"Excluded: {med_class['exclusion_reason']} ({med_class['category']})"
            rec_dict["Pilot Tier"] = "Tier C (Excluded / Non-Chronic)"
            ineligible_rows.append(rec_dict)
            continue

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
        cname_s = pd.Series(el_raw.get("customerName", el_raw["customerId"])).fillna("Unknown Customer")
        eligible_df["Customer Name"] = cname_s
        med_s = pd.Series(el_raw.get("itemName", el_raw["itemId"])).fillna("Unknown Medicine")
        eligible_df["Medicine"] = med_s
        eligible_df["Medication"] = eligible_df["Medicine"]
        phone_s = pd.Series(el_raw.get("MOBILE_NO", "")).fillna("").astype(str).str.strip()
        eligible_df["Delivery Phone"] = phone_s
        eligible_df["Mobile Number"] = eligible_df["Delivery Phone"]
        eligible_df["Mobile Status"] = eligible_df["Mobile Number"].apply(determine_mobile_status)

        # Dates & Intervals
        last_dt = pd.to_datetime(el_raw["invoice_date"], errors="coerce")
        eligible_df["Last Purchase Date"] = last_dt.dt.strftime("%Y-%m-%d").fillna("-")
        pc_s = pd.Series(el_raw.get("purchase_count_so_far", 1)).astype(int)
        eligible_df["Purchase Count"] = pc_s

        # Estimated Days of Supply
        final_dos_list: List[float] = []
        dos_raw_list = el_raw["estimated_days_of_supply"].tolist() if "estimated_days_of_supply" in el_raw.columns else []
        pred_raw_list = el_raw["predicted_days_until_refill"].tolist() if "predicted_days_until_refill" in el_raw.columns else []
        for i in range(len(el_raw)):
            val = None
            if dos_raw_list and i < len(dos_raw_list) and pd.notna(dos_raw_list[i]):
                try:
                    val = float(dos_raw_list[i])
                except (ValueError, TypeError):
                    val = None
            if val is None or math.isnan(val):
                if pred_raw_list and i < len(pred_raw_list) and pd.notna(pred_raw_list[i]):
                    try:
                        val = float(pred_raw_list[i])
                    except (ValueError, TypeError):
                        val = 30.0
                else:
                    val = 30.0
            final_dos_list.append(round(float(val), 1))

        eligible_df["Estimated Days of Supply"] = final_dos_list
        eligible_df["Predicted Interval (Days)"] = final_dos_list
        eligible_df["Estimated Daily Consumption"] = el_raw.get("estimated_daily_consumption", el_raw.get("historical_consumption_rate", np.nan))
        eligible_df["estimated_daily_consumption"] = eligible_df["Estimated Daily Consumption"]

        # Expected Refill Date & Reminder Date (with 2-day buffer)
        exp_dates = []
        rem_dates = []
        last_dt_list = last_dt.tolist()
        for l_dt, dos in zip(last_dt_list, final_dos_list):
            if pd.notna(l_dt) and dos > 0:
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

    # Format ineligible DataFrame (sample up to 2000 for UI presentation and memory efficiency)
    ineligible_list = []
    cold_sample = raw_cold_start.head(2000) if len(raw_cold_start) > 2000 else raw_cold_start
    if not cold_sample.empty:
        for r in cold_sample.to_dict(orient="records"):
            phone_val = str(r.get("MOBILE_NO", "")).strip()
            med_val = str(r.get("itemName", r.get("itemId", "Unknown")))
            inv_d = r.get("invoice_date")
            pc_val = r.get("purchase_count_so_far", r.get("purchase_seq", 1))
            ineligible_list.append({
                "customerId": str(r.get("customerId", "")),
                "itemId": str(r.get("itemId", "")),
                "Customer Name": str(r.get("customerName", r.get("customerId", "Unknown"))),
                "Medicine": med_val,
                "Medication": med_val,
                "Delivery Phone": phone_val,
                "Mobile Number": phone_val,
                "Mobile Status": determine_mobile_status(phone_val),
                "Last Purchase Date": pd.to_datetime(str(inv_d)).strftime("%Y-%m-%d") if (inv_d is not None and str(inv_d).strip() != "") else "-",
                "Purchase Count": int(pc_val) if pc_val is not None else 1,
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


@st.cache_data(show_spinner=False, ttl=120)
def _decisions_version(store_id: str) -> tuple:
    """Changes whenever the engine writes decisions (cache key for the overview).

    Itself cached briefly (a full-table MAX costs ~2-3 s): every Streamlit mutation clears
    st.cache_data, so changes made here show at once; changes made through the API within 2 minutes.
    """
    from sqlalchemy import func

    from database.connection import SessionLocal
    from database.models import RefillDecisionModel

    with SessionLocal(store_id) as db:
        n, last = db.query(func.count(RefillDecisionModel.decision_id), func.max(RefillDecisionModel.updated_at)).one()
    return int(n or 0), str(last)


@st.cache_data(show_spinner=False, max_entries=4)
def prepare_overview_from_decisions(store_id: str, version: tuple, ineligible_limit: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Eligible / ineligible overview from the decisions the engine already stored (same as FastAPI).

    One query instead of re-predicting every patient on each page load: the latest decision per
    patient-medicine. Same columns as :func:`prepare_prediction_overview`, so every tab works unchanged.
    """
    from database.connection import SessionLocal
    from database.models import RefillDecisionModel as D

    cols = [D.customer_item_key, D.customer_id, D.customer_name, D.item_id, D.item_name, D.mobile_no, D.path,
            D.purchase_count, D.is_eligible, D.stability_tier, D.dos_days, D.predicted_interval_days,
            D.last_purchase_date, D.expected_refill_date, D.decision_reason, D.updated_at]
    with SessionLocal(store_id) as db:
        rows = db.query(*cols).all()
    df = pd.DataFrame.from_records(rows, columns=[c.key for c in cols])
    if df.empty:
        return prepare_prediction_overview(pd.DataFrame(), None)
    # Latest decision per patient-medicine (the table keeps one row per purchase cycle)
    df = df.sort_values(["last_purchase_date", "updated_at"]).drop_duplicates("customer_item_key", keep="last")
    phones = df["mobile_no"].fillna("").astype(str).str.strip()
    status_of = {p_: determine_mobile_status(p_) for p_ in phones.unique()}  # once per distinct phone
    last = pd.to_datetime(df["last_purchase_date"], errors="coerce")
    exp = pd.to_datetime(df["expected_refill_date"], errors="coerce")
    interval = pd.to_numeric(df["predicted_interval_days"], errors="coerce")
    dos = pd.to_numeric(df["dos_days"], errors="coerce").fillna(interval).round(1)
    base = pd.DataFrame({
        "customerId": df["customer_id"].astype(str).values,
        "itemId": df["item_id"].astype(str).values,
        "Customer Name": df["customer_name"].fillna("Unknown Customer").values,
        "Medicine": df["item_name"].fillna("Unknown Medicine").values,
        "Delivery Phone": phones.values,
        "Mobile Status": phones.map(status_of).values,
        "Last Purchase Date": last.dt.strftime("%Y-%m-%d").fillna("-").values,
        "Purchase Count": pd.to_numeric(df["purchase_count"], errors="coerce").fillna(1).astype(int).values,
    })
    base["Medication"] = base["Medicine"]
    base["Mobile Number"] = base["Delivery Phone"]
    eligible_mask = (df["is_eligible"].fillna(False).astype(bool) & exp.notna()).values

    el = base[eligible_mask].copy()
    e_exp, e_int, e_dos = exp[eligible_mask], interval[eligible_mask], dos[eligible_mask]
    el["Estimated Days of Supply"] = e_dos.fillna(e_int).values
    el["Predicted Interval (Days)"] = e_int.values
    el["Estimated Daily Consumption"] = np.nan
    el["estimated_daily_consumption"] = np.nan
    el["Expected Refill Date"] = e_exp.dt.strftime("%Y-%m-%d").values
    el["Reminder Date"] = (e_exp - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d").values  # 2-day buffer, as before
    tiers = df.loc[eligible_mask, "stability_tier"].fillna("").astype(str).values
    el["History Quality"] = tiers
    el["Pilot Tier"] = np.where(df.loc[eligible_mask, "path"].astype(str).values == "PATH_A", "Tier A (Path A - stable history)", "Tier B (Path B - short history)")
    el["Operator Notes"] = df.loc[eligible_mask, "decision_reason"].fillna("").astype(str).str.slice(0, 300).values
    el["MOBILE_NO"] = el["Delivery Phone"]
    el["invoice_date"] = el["Last Purchase Date"]
    el["latest_purchase_date"] = el["Last Purchase Date"]
    el["predicted_days_until_refill"] = el["Estimated Days of Supply"]
    el["expected_refill_date"] = el["Expected Refill Date"]
    el["reminder_date"] = el["Reminder Date"]
    el["raw_reminder_date"] = el["Reminder Date"]
    el["customerName"] = el["Customer Name"]
    el["itemName"] = el["Medicine"]
    el["purchase_count_so_far"] = el["Purchase Count"]
    el["estimated_days_of_supply"] = el["Estimated Days of Supply"]
    el["mobile_status"] = el["Mobile Status"]

    inel = base[~eligible_mask].copy()
    inel["Reason for Ineligibility"] = df.loc[~eligible_mask, "decision_reason"].fillna("Not eligible").astype(str).str.slice(0, 300).values
    inel_total = len(inel)
    inel = inel.head(ineligible_limit)  # UI sample (memory); the count below is the full number
    metrics = {"total_histories": int(len(base)), "eligible_count": int(len(el)), "ineligible_count": int(inel_total),
               "source": "stored engine decisions"}
    return el.reset_index(drop=True), inel.reset_index(drop=True), metrics


@store_cache_data(show_spinner=False, ttl=300)
def serving_model_status() -> Dict[str, Any]:
    """Which model serves predictions (same as /health). Cached 5 min; cleared by every mutation."""
    try:
        from database.connection import SessionLocal
        from refillcare.ml.model_provider import ModelProvider

        with SessionLocal() as db:
            return ModelProvider.from_registry(db).describe()
    except Exception as e:  # never block the page on a status probe
        return {"error": str(e)}


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

    # Clinical Chronic Filter
    from refillcare.data.medication_classifier import is_chronic_medication
    if "itemName" in subset.columns:
        subset = subset[subset["itemName"].apply(lambda x: is_chronic_medication(str(x)))].copy()

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

    subset = pd.DataFrame(history_df[
        (history_df["customerId"].astype(str) == cid) &
        (history_df["itemId"].astype(str) == iid)
    ]).sort_values("invoice_date")

    if subset.empty:
        return {"total_purchases": 0, "visits": pd.DataFrame()}

    intervals = subset["days_since_previous_purchase"].dropna().tolist() if "days_since_previous_purchase" in subset.columns else []

    sc_series = pd.Series(subset.get("salt_composition", "-")).fillna("-") if "salt_composition" in subset.columns else "-"
    visits_display = pd.DataFrame({
        "Visit": range(1, len(subset) + 1),
        "Invoice Date": pd.to_datetime(subset["invoice_date"]).dt.strftime("%d-%m-%Y"),
        "Quantity": subset["quantity"].astype(int) if "quantity" in subset.columns else 1,
        "Days Since Prior Purchase": subset["days_since_previous_purchase"].fillna("-") if "days_since_previous_purchase" in subset.columns else "-",
        "Salt / Composition": sc_series,
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
    export_df["customerId"] = pd.Series(valid_mobiles.get("customerId", "")).astype(str)
    cname_val = valid_mobiles.get("Customer Name", valid_mobiles.get("customerName", export_df["customerId"]))
    export_df["customerName"] = pd.Series(cname_val).fillna("Unknown").astype(str)

    # Phone number
    phone_col = "Mobile Number" if "Mobile Number" in valid_mobiles.columns else ("MOBILE_NO" if "MOBILE_NO" in valid_mobiles.columns else "Delivery Phone")
    export_df["MOBILE_NO"] = pd.Series(valid_mobiles.get(phone_col, valid_mobiles.get("phone_number", ""))).fillna("").astype(str).str.strip()

    # Medicine
    export_df["itemId"] = pd.Series(valid_mobiles.get("itemId", "")).astype(str)
    item_val = valid_mobiles.get("Medication", valid_mobiles.get("Medicine", valid_mobiles.get("itemName", valid_mobiles.get("medication_name", export_df["itemId"]))))
    export_df["itemName"] = pd.Series(item_val).fillna("Unknown Medicine").astype(str)

    # Dates
    last_dt = valid_mobiles.get("Last Purchase Date", valid_mobiles.get("last_purchase_date", valid_mobiles.get("invoice_date", "-")))
    export_df["last_purchase_date"] = pd.Series(last_dt).astype(str)

    exp_dt = valid_mobiles.get("Expected Refill Date", valid_mobiles.get("expected_refill_date", "-"))
    export_df["expected_refill_date"] = pd.Series(exp_dt).astype(str)

    rem_dt = valid_mobiles.get("Reminder Date", valid_mobiles.get("reminder_date", valid_mobiles.get("raw_reminder_date", "-")))
    export_df["reminder_date"] = pd.Series(rem_dt).astype(str)

    # predicted_days_until_refill
    dos_vals = valid_mobiles.get("Estimated Days of Supply", valid_mobiles.get("estimated_days_of_supply", valid_mobiles.get("predicted_days_until_refill", valid_mobiles.get("Predicted Interval (Days)", 30.0))))
    dos_export_list = []
    for x in pd.Series(dos_vals).tolist():
        try:
            v = float(x)
            dos_export_list.append(round(v, 1) if not math.isnan(v) else 30.0)
        except (ValueError, TypeError):
            dos_export_list.append(30.0)
    export_df["predicted_days_until_refill"] = dos_export_list

    export_df = pd.DataFrame(export_df[exact_columns].fillna("-"))
    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def rollback_import_batch_shared(batch_id: str, history_data: Optional[pd.DataFrame], storage: Any) -> Dict[str, Any]:
    """One rollback for both Streamlit buttons (same services as FastAPI).

    Database batches: delete the batch's lines, revert the reminders it created (current patient /
    therapy keys), re-score its patients. Legacy file batches: the old file rollback + revert.
    """
    from database.connection import SessionLocal
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import rollback_batch

    with SessionLocal() as db:
        if not hs.batch_lines(db, batch_id).empty:
            try:
                res = rollback_batch(db, batch_id)
            except (LookupError, ValueError) as e:
                return {"status": "error", "message": str(e)}
            res["message"] = f"Batch {batch_id} rolled back: {res['records_removed']} sales lines removed and its patients re-scored."
            return res
        rb = rollback_monthly_sales_import(
            current_history_df=history_data if history_data is not None else pd.DataFrame(),
            import_batch_id=batch_id, storage=storage,
        )
        if rb["status"] == "success":
            from refillcare.engine.post_ingest import revert_after_rollback

            revert_after_rollback(db, rb, batch_id)
        return rb


def render_data_status_panel() -> None:
    """How current the store's sales are, and uploads whose re-scoring failed (with a retry)."""
    from database.connection import SessionLocal
    from database.models import ImportBatchModel
    from refillcare.data.incremental_ingest import data_status, sync_batch

    with SessionLocal() as db:
        stt = data_status(db)
        failed = (db.query(ImportBatchModel).filter(ImportBatchModel.sync_status == "SYNC_FAILED", ImportBatchModel.is_active.is_(True))
                  .order_by(ImportBatchModel.upload_timestamp.desc()).all())
        failed = [(b.import_batch_id, b.source_filename, (b.sync_error or "")[:200]) for b in failed]
    if stt["ok_to_send"]:
        st.success(f"Sales data verified current: latest sale {stt['latest_sales_date']} ({stt['days_behind']} day(s) ago).")
    else:
        why = (f"{stt['batches_failed_to_sync']} upload(s) failed to re-score" if stt["batches_failed_to_sync"]
               else f"latest sale {stt['latest_sales_date'] or 'none'} is {stt['days_behind']} day(s) behind (max {stt['max_allowed_lag_days']})")
        st.warning(f"Live WhatsApp dispatch is blocked: {why}. Upload missing days first.")
    for bid, fname, err in failed:
        c1, c2 = st.columns([5, 1])
        c1.caption(f"Re-scoring failed for `{bid}` ({fname}): {err}")
        if c2.button("Retry", key=f"retry_sync_{bid}"):
            with SessionLocal() as db:
                res = sync_batch(db, bid)
            st.cache_data.clear()
            st.cache_resource.clear()
            if res.get("error"):
                st.error(f"Retry failed: {res['error']}")
            else:
                st.rerun()


def render_waiting_returns() -> None:
    """Return lines still waiting for their sale (same data as GET /api/v1/sales/returns/waiting), with a
    re-check (POST /api/v1/sales/returns/recheck) after the return window changed or a missed day was uploaded."""
    from database.connection import SessionLocal
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import recheck_returns

    try:
        with SessionLocal() as db:
            info = hs.waiting_returns(db, limit=200)
    except Exception:
        return
    if not info["total"]:
        return
    with st.expander(f"↩️ {info['total']:,} return lines waiting for their sale", expanded=False):
        st.caption(f"These returns match no sale of the same patient + medicine in the {info['return_window_days']} days before. "
                   "They are saved and applied automatically when that sale is uploaded.")
        st.dataframe(pd.DataFrame(info["rows"]), width="stretch", hide_index=True)
        if st.button("Re-check waiting returns now", key="btn_recheck_returns"):
            with st.spinner("Applying returns that now match a sale..."):
                with SessionLocal() as db:
                    res = recheck_returns(db)
            st.cache_data.clear()
            st.success(f"{res['returns_applied']:,} returns applied, {res['returns_still_waiting']:,} still waiting.")


def render_saved_column_mappings() -> None:
    """Saved per-store column mappings (delete one to have its layout detected again)."""
    from database.connection import SessionLocal
    from refillcare.data.column_mapping import delete_profile, list_profiles
    from refillcare.store import StoreConfig

    with SessionLocal() as mdb:
        profiles = list_profiles(mdb, StoreConfig.from_env().store_id)
    with st.expander(f"Saved Column Mappings ({len(profiles)})", expanded=False):
        if not profiles:
            st.caption("No saved mappings yet - the first successful upload saves one.")
            return
        for prof in profiles:
            mapped = ", ".join(f"{f} ← {c}" for f, c in (prof["mapping"] or {}).items() if c)
            c1, c2 = st.columns([5, 1])
            c1.caption(f"{mapped}  ·  {prof['source']} ({prof['confirmed_by']}) · used {prof['use_count']}x · last {prof['last_used_at'] or '-'}")
            if c2.button("Delete", key=f"del_mapping_{prof['profile_id']}"):
                with SessionLocal() as mdb:
                    delete_profile(mdb, prof["profile_id"])
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


_MAPPING_SOURCE_LABEL = {
    "saved": "saved for this store", "user": "chosen by you", "auto": "auto-matched",
    "saved_hint": "auto-matched (known header)", "suggestion": "suggested - please confirm", "none": "not in file",
}


def render_column_mapping_panel(report: Dict[str, Any], map_key: str, force_open: bool = False) -> None:
    """Column mapping of the uploaded file: summary, and an editor to confirm / change it.

    The confirmed mapping is saved per store and header layout (enterprise.db), so the next upload
    with the same columns is mapped automatically (same code path as the FastAPI confirm screen).
    """
    fields = report.get("fields", []) or []
    headers = list(report.get("headers", []) or [])
    needs = bool(report.get("needs_confirmation"))
    if needs:
        status = f"Requires mapping review ({len(report.get('missing_required', []))} required field(s) not found)"
    elif any(f.get("source") == "saved" for f in fields):
        status = "Active store profile applied"
    else:
        status = "All required columns matched"
    with st.expander(f"Column Mapping - {status}", expanded=needs or force_open):
        for w in report.get("warnings", []) or []:
            st.warning(w)
        ignored = report.get("ignored_columns", []) or []
        if ignored:
            st.caption("Not stored (not needed for reminders): " + ", ".join(map(str, ignored)))
        none_opt = "- not in file -"
        options = [none_opt] + headers
        h1, h2, h3, h4 = st.columns([2.2, 1, 2.2, 3])
        h1.markdown("**RefillCare field**")
        h2.markdown("**Needed**")
        h3.markdown("**Column in your file**")
        h4.markdown("**Sample values / match**")
        chosen: Dict[str, Optional[str]] = {}
        for f in fields:
            c1, c2, c3, c4 = st.columns([2.2, 1, 2.2, 3])
            c1.markdown(f.get("label", f["field"]))
            c2.caption(f.get("role", ""))
            col = f.get("column")
            sel = c3.selectbox(
                f.get("label", f["field"]), options, index=options.index(col) if col in headers else 0,
                key=f"{map_key}::{f['field']}", label_visibility="collapsed",
            )
            samples = " | ".join(map(str, f.get("samples") or [])) or "-"
            match = _MAPPING_SOURCE_LABEL.get(f.get("source", "none"), f.get("source", "")) if col else "not in file"
            c4.caption(f"{samples}  ({match})")
            chosen[f["field"]] = None if sel == none_opt else sel
        if st.button("Save Mapping & Validate", key=f"{map_key}::save"):
            from database.connection import SessionLocal
            from refillcare.data.column_mapping import save_profile

            try:
                with SessionLocal() as mdb:
                    save_profile(mdb, headers, chosen, source="user", confirmed_by="streamlit")
                st.session_state[map_key] = chosen
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except ValueError as e:
                st.error(f"{e}")


def parse_and_validate_uploaded_sales_file(
    uploaded_file: Any,
    existing_history_df: Optional[pd.DataFrame] = None,
    column_mapping: Optional[Dict[str, Optional[str]]] = None,
    db: Any = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Parse an uploaded monthly sales Excel/CSV file and extract business metrics.

    Columns are mapped with the shared RefillCare mapping (the given one, the store's saved profile
    when ``db`` is passed, else auto-detected) and only the RefillCare columns are kept.
    """
    from refillcare.data.schema import read_sales_file

    try:
        filename = getattr(uploaded_file, "name", "sales.csv").lower()
        if not filename.endswith((".csv", ".xlsx", ".xls")):
            return None, {"error": "Unsupported file format. Please upload a .csv, .xlsx, or .xls file."}
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        df = read_sales_file(uploaded_file, filename)
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

    # Shared column mapping (same code as FastAPI): user mapping > saved store profile > auto-detect
    from refillcare.data.column_mapping import resolve_mapping
    from refillcare.data.schema import apply_column_mapping

    mres = resolve_mapping(df, db=db, user_mapping=column_mapping)
    mapping_report = mres.to_dict()
    if mres.needs_confirmation:
        return df, {"needs_column_mapping": True, "column_mapping": mapping_report, "records_received": len(df)}
    renamed_df = apply_column_mapping(df, mres.mapping)

    # Total records received
    records_received = len(renamed_df)

    # Date range analysis using strict pharmacy date parser
    date_col = "invoice_date" if "invoice_date" in renamed_df.columns else None
    if date_col is not None:
        parsed_dates, date_diagnostics = parse_pharmacy_dates(pd.Series(renamed_df[date_col]))
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
        invalid_quantities = 0
        for q in renamed_df["quantity"].tolist():
            try:
                v = float(q)
                if math.isnan(v) or v <= 0:
                    invalid_quantities += 1
            except (ValueError, TypeError):
                invalid_quantities += 1
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
        q_valid_list = []
        for q in renamed_df["quantity"].tolist():
            try:
                v = float(q)
                q_valid_list.append(not math.isnan(v) and v > 0)
            except (ValueError, TypeError):
                q_valid_list.append(False)
        valid_mask = valid_mask & pd.Series(q_valid_list, index=renamed_df.index)

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
        "column_mapping": mapping_report,
        "column_mapping_result": mres,
    }

    return renamed_df, metrics


@store_cache_data(ttl=60, show_spinner=False)
def get_available_reminder_dates() -> List[date]:
    """Get list of distinct scheduled reminder dates from persistent enterprise.db."""
    try:
        from database.connection import SessionLocal
        from database.models import ReminderStageModel
        with SessionLocal() as db:
            dts = (
                db.query(ReminderStageModel.target_send_date)
                .distinct()
                .order_by(ReminderStageModel.target_send_date)
                .all()
            )
            return [d[0] for d in dts if d[0] is not None]
    except Exception:
        return []


@store_cache_data(ttl=60, show_spinner=False)
def get_available_reminder_months() -> List[str]:
    """Get list of distinct scheduled reminder months (YYYY-MM) from enterprise.db."""
    try:
        from database.connection import SessionLocal
        from database.models import ReminderStageModel
        with SessionLocal() as db:
            dts = (
                db.query(ReminderStageModel.target_send_date)
                .filter(ReminderStageModel.status.in_(["PENDING", "APPROVED", "DUE"]))
                .distinct()
                .order_by(ReminderStageModel.target_send_date.asc())
                .all()
            )
            months = set()
            for d in dts:
                if d[0]:
                    months.add(d[0].strftime("%Y-%m"))
            if months:
                return sorted(list(months))
    except Exception:
        pass
    cur_m = date.today().strftime("%Y-%m")
    return [cur_m]


@store_cache_data(ttl=60, show_spinner=False)
def load_v1_reminder_queue_df(
    target_date: Optional[date] = None,
    target_month: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Load persistent reminder queue for target date, full target month, or rolling date range from enterprise.db with rich clinical metadata."""
    try:
        from database.connection import SessionLocal
        from refillcare.engine.persistence import RefillPersistenceManager
        with SessionLocal() as db:
            pm = RefillPersistenceManager()
            if start_date and end_date:
                queue = pm.get_date_range_review_queue(db, start_date=start_date, end_date=end_date)
            elif target_month:
                queue = pm.get_monthly_review_queue(db, target_month=target_month)
            else:
                dt = target_date or date.today()
                queue = pm.get_today_review_queue(db, target_date=dt)
            if queue:
                df = pd.DataFrame(queue)
                # Map to standard display column names
                df["Customer Name"] = df["customer_name"].fillna("Valued Customer")
                df["raw_phone_number"] = df["phone_number"].fillna("")
                df["Mobile Number"] = df["phone_number"].apply(format_display_phone_10digits)
                df["Medication"] = df["item_name"].fillna(df["item_id"])
                df["Last Purchase Date"] = df["last_purchase_date"].fillna("-")
                dos_list = []
                for x in df["estimated_days_of_supply"].tolist():
                    try:
                        v = float(x)
                        dos_list.append(round(v, 1) if not math.isnan(v) else 30.0)
                    except (ValueError, TypeError):
                        dos_list.append(30.0)
                df["Estimated Days of Supply"] = dos_list
                df["Expected Refill Date"] = df["expected_refill_date"].fillna("-")
                df["Reminder Date"] = df["target_send_date"].fillna("-")
                df["Mobile Status"] = df["phone_number"].apply(determine_mobile_status)
                df["Status"] = df["status"].fillna("PENDING")

                def _fmt_stage(offset):
                    try:
                        o = int(offset)
                        if o == -7: return "-7 days (Due in 7 Days)"
                        elif o == -3: return "-3 days (Due in 3 Days)"
                        elif o == -1: return "-1 day (Due Tomorrow)"
                        elif o == 0: return "0 days (Due Today)"
                        elif o == 2: return "+2 days (Follow-up)"
                        elif o == 5: return "+5 days (Follow-up)"
                        elif o in (40, 45): return "+45 days (Re-engagement)"
                        elif o > 0: return f"+{o} days"
                        else: return f"{o} days"
                    except Exception:
                        return str(offset)

                df["Reminder Stage"] = df["stage_offset"].apply(_fmt_stage)

                def _fmt_path(p):
                    if p == "PATH_A": return "Path A (Adherence)"
                    elif p == "PATH_B": return "Path B (Developing)"
                    elif p == "PHARMACIST_ADDED": return "Added by pharmacist"
                    return str(p)

                df["Clinical Path"] = df["path"].apply(_fmt_path)
                df["Stability Tier"] = df["stability_tier"].fillna("UNKNOWN")
                df["Clinical Regimen"] = df["dosage_regimen"].fillna("1.0 tab/d (OD)") if "dosage_regimen" in df.columns else "1.0 tab/d (OD)"
                df["Consensus Archetype"] = df["archetype"].fillna("Standard Consensus") if "archetype" in df.columns else "Standard Consensus"
                df["Decision Provenance"] = df["decision_reason"].fillna("")
                df["Decision Reason"] = df["decision_reason"].fillna("")
                df["Prediction Method"] = df["prediction_method"].fillna("")

                # Columns for CSV export
                df["customerId"] = df["customer_id"]
                df["itemId"] = df["item_id"]
                df["customerName"] = df["Customer Name"]
                df["itemName"] = df["Medication"]
                df["MOBILE_NO"] = df["Mobile Number"]
                df["last_purchase_date"] = df["Last Purchase Date"]
                df["expected_refill_date"] = df["Expected Refill Date"]
                df["reminder_date"] = df["Reminder Date"]
                df["raw_reminder_date"] = df["Reminder Date"]
                df["predicted_days_until_refill"] = df["Estimated Days of Supply"]
                df["estimated_days_of_supply"] = df["Estimated Days of Supply"]
                df["mobile_status"] = df["Mobile Status"]

                # The send rule (rules-chronic or pharmacist-added) is applied once, inside the shared
                # queue (refillcare.engine.persistence.medicine_can_be_sent) - same rows as the API.
                return df
    except Exception:
        pass
    return pd.DataFrame()


@store_cache_data(ttl=5, show_spinner=False)
def get_forecast_anchor_month() -> str:
    """Active forecast month = current calendar month or latest purchase + 1 day, whichever is newer."""
    try:
        from database.connection import SessionLocal
        from database.models import RefillDecisionModel
        from sqlalchemy import func

        with SessionLocal() as db:
            latest = db.query(func.max(RefillDecisionModel.last_purchase_date)).scalar()
        if latest:
            data_m = (latest + timedelta(days=1)).strftime("%Y-%m")
            cur_m = date.today().strftime("%Y-%m")
            return max(data_m, cur_m)
    except Exception:
        pass
    return date.today().strftime("%Y-%m")


def shift_month(month_key: str, delta: int) -> str:
    y, m = map(int, month_key.split("-"))
    idx = y * 12 + (m - 1) + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def month_label(month_key: str) -> str:
    y, m = map(int, month_key.split("-"))
    return f"{date(y, m, 1).strftime('%B %Y')} ({month_key})"


@store_cache_data(ttl=60, show_spinner=False)
def load_overview_kpis(today: date) -> Dict[str, Any]:
    """Pharmacy Overview numbers of the current store for this month (same function as the API)."""
    from database.connection import SessionLocal
    from refillcare.engine.overview_kpis import overview_kpis

    with SessionLocal() as db:
        return overview_kpis(db, today)


def get_enterprise_dashboard_kpis(history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Backwards-compatible wrapper around :func:`load_overview_kpis` (this store, this month)."""
    k = dict(load_overview_kpis(date.today()))
    k.update({"upcoming_reminders": k["reminder_messages_month"], "customers_needing_review": k["patients_needing_review_today"],
              "target_month": k["month"], "target_month_reminders": k["reminder_messages_month"]})
    return k


@store_cache_data(ttl=15, show_spinner=False)
def load_v1_customers_needing_review_df(target_month: Optional[str] = None) -> pd.DataFrame:
    """Load live review candidates from enterprise.db."""
    try:
        from database.connection import SessionLocal
        from database.models import RefillDecisionModel
        with SessionLocal() as db:
            m_key = target_month[:7] if (target_month and target_month != "ALL") else None

            # 1. Missing mobile on eligible refills
            q1 = db.query(RefillDecisionModel).filter(
                RefillDecisionModel.is_eligible == True,
                (RefillDecisionModel.mobile_no == None) | (RefillDecisionModel.mobile_no == "") | (RefillDecisionModel.mobile_no == "nan")
            )
            if m_key:
                q1 = q1.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
            r_missing = q1.all()

            # 2. Cold-start / single purchase from latest sales
            q2 = db.query(RefillDecisionModel).filter(
                RefillDecisionModel.is_eligible == False
            )
            if m_key:
                try:
                    y, m = map(int, m_key.split("-"))
                    prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
                    prev_str = f"{prev_y:04d}-{prev_m:02d}"
                except Exception:
                    prev_str = shift_month(get_forecast_anchor_month(), -1)
                q2 = q2.filter(RefillDecisionModel.last_purchase_date.like(f"{prev_str}%"))
            r_cold = q2.limit(2000).all()

            # 3. Stability risk / variance
            q3 = db.query(RefillDecisionModel).filter(
                RefillDecisionModel.is_eligible == True,
                RefillDecisionModel.stability_tier.in_(["MEDIUM-RISK", "UNSTABLE"])
            )
            if m_key:
                q3 = q3.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
            r_stab = q3.limit(500).all()

            records = []
            for r in r_missing:
                records.append({
                    "customerId": str(r.customer_id),
                    "Customer Name": r.customer_name or "Unknown",
                    "Mobile Number": "-",
                    "Mobile Status": "Missing",
                    "Medication": r.item_name or "Medication",
                    "Last Purchase Date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                    "Expected Refill Date": str(r.expected_refill_date) if r.expected_refill_date else "-",
                    "Review Category": "Missing Mobile Numbers",
                    "Review Reason": "Missing mobile number for automated reminder delivery",
                })
            for r in r_cold:
                mob_st = determine_mobile_status(r.mobile_no or "")
                records.append({
                    "customerId": str(r.customer_id),
                    "Customer Name": r.customer_name or "Unknown",
                    "Mobile Number": str(r.mobile_no or "-"),
                    "Mobile Status": mob_st,
                    "Medication": r.item_name or "Medication",
                    "Last Purchase Date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                    "Expected Refill Date": "-",
                    "Review Category": "Cold-Start / Single Purchase",
                    "Review Reason": r.decision_reason or "Cold-start: single purchase requiring recurrence validation",
                })
            for r in r_stab:
                mob_st = determine_mobile_status(r.mobile_no or "")
                records.append({
                    "customerId": str(r.customer_id),
                    "Customer Name": r.customer_name or "Unknown",
                    "Mobile Number": str(r.mobile_no or "-"),
                    "Mobile Status": mob_st,
                    "Medication": r.item_name or "Medication",
                    "Last Purchase Date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                    "Expected Refill Date": str(r.expected_refill_date) if r.expected_refill_date else "-",
                    "Review Category": "History Review Required",
                    "Review Reason": f"Variance risk ({r.stability_tier}): pharmacist interval review recommended",
                })
            return pd.DataFrame(records)
    except Exception:
        pass
    return pd.DataFrame()


@store_cache_data(ttl=15, show_spinner=False)
def load_v1_refill_decisions_df(target_month: Optional[str] = None) -> pd.DataFrame:
    """Load persistent refill decisions from enterprise.db for interactive forecast exploration."""
    try:
        from database.connection import SessionLocal
        from database.models import RefillDecisionModel
        with SessionLocal() as db:
            q = db.query(RefillDecisionModel).filter(RefillDecisionModel.is_eligible == True)
            if target_month and target_month != "All Forecast Months":
                clean_m = target_month[:7]
                q = q.filter(RefillDecisionModel.expected_refill_date.like(f"{clean_m}%"))
            rows = q.order_by(RefillDecisionModel.expected_refill_date.asc()).all()
            if rows:
                data = []
                for r in rows:
                    data.append({
                        "Customer Name": r.customer_name or "Unknown Patient",
                        "Medication": r.item_name or "Medication",
                        "Mobile Number": r.mobile_no or "-",
                        "Mobile Status": determine_mobile_status(r.mobile_no),
                        "Clinical Path": r.path or "Path A",
                        "Purchase Count": int(r.purchase_count or 1),
                        "Last Purchase Date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                        "Estimated Supply (Days)": round(float(r.dos_days or 30.0), 1) if r.dos_days else "-",
                        "Predicted Interval (Days)": round(float(r.predicted_interval_days), 1) if r.predicted_interval_days else "-",
                        "Expected Refill Date": str(r.expected_refill_date) if r.expected_refill_date else "-",
                        "Stability Tier": r.stability_tier or "STABLE",
                        "Prediction Method": r.prediction_method or "PATH_A_PERSONAL_HISTORICAL_MEDIAN",
                        "Decision Reason": r.decision_reason or "",
                    })
                return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()


def load_active_refill_decisions_for_medsync(fallback_df: Optional[pd.DataFrame] = None) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """Load active, persistent refill decisions from enterprise.db for complete Med-Sync clustering."""
    try:
        from database.connection import SessionLocal
        from database.models import RefillDecisionModel
        with SessionLocal() as db:
            db_decisions = db.query(RefillDecisionModel).filter(RefillDecisionModel.is_eligible == True).all()
            if db_decisions:
                records = []
                for d in db_decisions:
                    records.append({
                        "customer_id": d.customer_id,
                        "customer_name": d.customer_name,
                        "mobile_no": d.mobile_no,
                        "item_id": d.item_id,
                        "item_name": d.item_name,
                        "last_purchase_date": d.last_purchase_date,
                        "expected_refill_date": d.expected_refill_date,
                        "is_eligible": d.is_eligible,
                        "stability_tier": d.stability_tier,
                        "predicted_interval_days": d.predicted_interval_days,
                        "dos_days": d.dos_days,
                    })
                return records
    except Exception:
        pass
    return fallback_df if fallback_df is not None else pd.DataFrame()


# ==============================================================================
# STREAMLIT UI RENDERER
# ==============================================================================
def render_app():
    """Main Streamlit application layout."""
    page_icon = str(MEDIASTRA_ICON_PATH) if MEDIASTRA_ICON_PATH.exists() else None
    st.set_page_config(
        page_title="Mediastra Refillcare — Clinical Operations",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if hasattr(st, "logo") and MEDIASTRA_LOGO_PATH.exists():
        st.logo(
            image=str(MEDIASTRA_LOGO_PATH),
            icon_image=str(MEDIASTRA_ICON_PATH) if MEDIASTRA_ICON_PATH.exists() else None,
            size="medium",
        )

    from refillcare.logging_setup import configure_logging

    # No login / store picker here: the Mediastra application handles tenant > store logins and
    # opens this app with ?store_id=<store> (default: REFILLCARE_STORE_ID). Every database session,
    # cache and file below follows this store.
    configure_logging()
    apply_store_from_request()

    # Clean Pharmacy Branding & Enterprise Hospital Styles
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0c2461;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4b6584;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            text-align: left;
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 800;
            color: #1e3a8a;
            line-height: 1.1;
        }
        .metric-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.25rem;
        }

        /* Hospital Progress Bars */
        .med-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
            padding: 8px 12px;
            border-radius: 8px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .med-rank {
            width: 30px;
            height: 30px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.9rem;
            color: #ffffff;
            flex-shrink: 0;
        }
        .bar-wrap {
            flex: 1;
            background: #e2e8f0;
            height: 8px;
            border-radius: 9999px;
            overflow: hidden;
        }
        .bar-inner {
            height: 100%;
            border-radius: 9999px;
        }

        /* Search Buttons Red Background with White Text */
        div[data-testid="stButton"] button:has(p:contains("Search")),
        div[data-testid="stButton"] button:has(p:contains("Search")),
        button[key="btn_apply_rem_search"],
        button[key="btn_apply_medsync_search"] {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #b91c1c !important;
            font-weight: 600 !important;
        }
        div[data-testid="stButton"] button:hover:has(p:contains("Search")),
        div[data-testid="stButton"] button:hover:has(p:contains("Search")) {
            background-color: #b91c1c !important;
            color: #ffffff !important;
            border-color: #991b1b !important;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.35) !important;
        }
        div[data-testid="stButton"] button:has(p:contains("Search")) p,
                /* Bold Tab Labels & Professional Sizing */
        div[data-testid="stTabs"] button,
        button[data-testid="stTab"],
        div[data-testid="stTabs"] button p,
        button[data-testid="stTab"] p {
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: -0.01em !important;
        }

        div[data-testid="stButton"] button:has(p:contains("Search")) p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Enterprise Hospital Brand Header Banner
    logo_b64 = get_mediastra_logo_base64()
    if logo_b64:
        brand_icon_html = f'<div style="height: 48px; min-width: 54px; background: rgba(37, 99, 235, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; display: flex; align-items: center; justify-content: center; padding: 4px 8px; box-shadow: 0 0 16px rgba(37,99,235,0.35);"><img src="data:image/png;base64,{logo_b64}" alt="Mediastra Logo" style="height: 38px; width: auto; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(37, 99, 235, 0.5));" /></div>'
    else:
        brand_icon_html = '<div style="width: 46px; height: 46px; background: linear-gradient(135deg, #2563eb, #0ea5e9); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 800; color: #fff; box-shadow: 0 0 16px rgba(37,99,235,0.4);">MA</div>'

    st.markdown(f"""<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; padding: 1.2rem 1.6rem; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 24px rgba(0,0,0,0.12);">
<div style="display: flex; align-items: center; gap: 1rem;">
{brand_icon_html}
<div>
<div style="font-size: 1.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
Mediastra <span style="color: #38bdf8;">Refillcare</span>
</div>
<div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">
Enterprise Medication Refill Forecasting & Clinical WhatsApp Adherence Platform
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # Never let the store run on the baseline silently (e.g. a model pickled by another scikit-learn)
    _serving = serving_model_status()
    if sys.version_info < (3, 10):
        # The project runs on .venv (Python 3.13, scikit-learn 1.7.2); an older Python cannot load its models
        st.error(f"This app was started with Python {sys.version.split()[0]} ({sys.executable}), not the project's "
                 "environment, so the ML model cannot load and predictions fall back to the baseline engine. "
                 r"Stop it and start it with **run_streamlit.bat** (or `.venv\Scripts\streamlit run app_refillcare.py`).")
    elif _serving.get("degraded"):
        st.error(f"ML model warning: {_serving.get('warning')}")

    from refillcare.data.identity import check_identity_setting_warning

    _ident_warn = check_identity_setting_warning()
    if _ident_warn:
        st.warning(f"⚠️ **Configuration Notice:** {_ident_warn}")

    # Load Data & Model
    with st.spinner("Loading RefillCare system data..."):
        model_bundle = load_refill_model()
        if "active_history_data" in st.session_state and st.session_state.active_history_data is not None:
            history_data = st.session_state.active_history_data
        else:
            history_data = current_history()

        if "active_sales_data" in st.session_state and st.session_state.active_sales_data is not None:
            recent_data = st.session_state.active_sales_data
        elif history_data is not None and not history_data.empty:
            recent_data = history_data
        elif _is_default_store():
            recent_data = load_refill_data()
        else:
            # New store with no uploads yet: never fall back to another store's files
            recent_data = history_data = pd.DataFrame(columns=["invoice_date", "customerId", "customerName", "itemId",
                                                               "itemName", "quantity", "MOBILE_NO"])
            st.info("No sales data has been uploaded for this store yet. Open **Upload Sales Data** to add the first file.")

    if model_bundle is None or recent_data is None:
        st.error("RefillCare System Telemetry Offline")
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
        # From the decisions the engine stored (fast, same as FastAPI); the old on-the-fly
        # re-prediction only runs while a store has no stored decisions yet
        from refillcare.tenancy import current_store_id as _sid

        _store = _sid()
        _dver = _decisions_version(_store)
        if _dver[0] > 0:
            eligible_df, ineligible_df, metrics = prepare_overview_from_decisions(_store, _dver)
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
    sales_coverage_str = "N/A"
    latest_sales_date_str = "N/A"
    eval_data_for_dates = history_data if (history_data is not None and not history_data.empty) else recent_data
    # Store summary from the database (same function as the API and every tab)
    store_sum = load_overview_kpis(date.today())
    if store_sum.get("latest_sales_date"):
        _lo, _hi = pd.Timestamp(store_sum["first_sales_date"]), pd.Timestamp(store_sum["latest_sales_date"])
        sales_coverage_str = f"{_lo.strftime('%b %Y')} – {_hi.strftime('%b %Y')}"
        latest_sales_date_str = _hi.strftime("%d-%m-%Y")

    # Top Status Bar
    col_status1, col_status2, col_status3 = st.columns([2, 2, 1])
    with col_status1:
        st.success("**Engine Status:** Operational · Continuous Clinical Forecasting Active")
    with col_status2:
        st.info(f"**Verified Ledger Coverage:** {sales_coverage_str}")
    with col_status3:
        if st.button("Refresh Telemetry"):
            st.cache_data.clear()
            st.cache_resource.clear()
            for k in ("active_sales_data", "active_history_data", "active_eligible_df", "active_ineligible_df"):
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Main Client Tabs. Pharmacist-facing app: tabs switched off in TAB_VISIBILITY are not shown and
    # their code does not run (set True to bring one back). ML lifecycle stays available through the
    # API (/api/v1/ml/*), scripts/retrain_production.py and the worker's monthly retrain.
    tab_specs = [
        ("dash", "**Pharmacy Overview**"),
        ("upload", "**Upload Sales Data**"),
        ("results", "**Upcoming Refills**"),
        ("reminders", "**Patient Reminders**"),
        ("medsync", "**Combined Prescriptions (Med-Sync)**"),
        ("review", "**Patients Needing Review**"),
        ("whatsapp", "**WhatsApp Messages**"),
        ("ml", "**AI Accuracy & Settings**"),
    ]
    shown = [(key, label) for key, label in tab_specs if TAB_VISIBILITY.get(key, True)]
    tab_objs = dict(zip([k for k, _ in shown], st.tabs([label for _, label in shown])))
    tab_dash, tab_upload, tab_results, tab_reminders, tab_medsync, tab_review, tab_whatsapp, tab_ml = (
        tab_objs.get(k) for k, _ in tab_specs
    )

    # --------------------------------------------------------------------------
    # TAB 1: OPERATIONS DASHBOARD
    # --------------------------------------------------------------------------
    with tab_dash:
        st.markdown("### Pharmacy Overview Summary")

        dash_kpis = load_overview_kpis(date.today())
        st.caption(f"This store · this month: **{dash_kpis['month_label']}** · numbers update after every sales upload.")
        m1, m3, m4, m5, m6 = st.columns(5)
        with m1:
            active_lbl = dash_kpis.get("active_patients_label", "Active patients (last 12 months)")
            active_cnt = dash_kpis.get("active_patients", dash_kpis.get("customers_monitored", 0))
            all_time_cnt = dash_kpis.get("all_time_customers", dash_kpis.get("customers_monitored", 0))
            has_12m = dash_kpis.get("has_12m_history", True)
            span_mos = dash_kpis.get("history_span_months", 0)

            help_text = (
                f"Active patients with at least one customer sale in the last 12 months. All-time total on file: {all_time_cnt:,}."
                if has_12m else
                f"Store has {span_mos} months of sales data on file. All-time total: {all_time_cnt:,}."
            )
            st.metric(
                active_lbl,
                f"{active_cnt:,}",
                delta=f"All-time: {all_time_cnt:,}",
                delta_color="off",
                help=help_text,
            )
        with m3:
            st.metric(f"Reminder Messages ({dash_kpis['month_label'].split()[0]})", f"{dash_kpis['reminder_messages_month']:,}",
                      help="WhatsApp messages scheduled this month. Each patient-medicine gets up to 7 messages "
                           "(7, 3 and 1 day before, due day, 2 and 5 days after, 45 days after).")
        with m4:
            st.metric("Reminders Due Today", f"{dash_kpis['reminders_due_today']:,}",
                      help="Reminder messages scheduled for today.")
        with m5:
            st.metric("Patients Needing Review Today", f"{dash_kpis['patients_needing_review_today']:,}",
                      help="Patients not getting reminders (chronic / not classified medicine) whose refill is due today. "
                           "See the Patients Needing Review tab.")
        with m6:
            st.metric("Latest Sales Date", latest_sales_date_str)

        try:
            _latest_dt = datetime.strptime(latest_sales_date_str, "%d-%m-%Y").date()
            _lag = (date.today() - _latest_dt).days
        except ValueError:
            _lag = None
        if _lag is not None and _lag > 2:
            st.warning(f"Latest sales uploaded are from **{latest_sales_date_str}** ({_lag} days ago). "
                       "Upload the newer sales so reminders and review lists are up to date.")

        st.markdown("---")
        st.markdown("### Transaction Channel Distribution")
        st.caption("Channel classification and isolation: Individual customer sales (S0/) participate in RefillCare; B2B / Inter-Store wholesale transfers (SB/) are strictly excluded from patient models.")

        # Dynamically compute channel metrics from master dataset and clean_transactions.parquet
        override_cnt = None
        if "active_sales_data" in st.session_state and st.session_state.active_sales_data is not None and not st.session_state.active_sales_data.empty:
            override_cnt = len(st.session_state.active_sales_data)

        chan_dist = get_enterprise_channel_distribution_cached()
        raw_tx_cnt = override_cnt if (override_cnt and override_cnt > chan_dist["raw_tx_cnt"]) else chan_dist["raw_tx_cnt"]
        clean_tx_cnt = chan_dist["tot_clean"]
        cust_sales = chan_dist["cust_sales"]
        b2b_sales = chan_dist["b2b_sales"]
        refill_el = chan_dist["refill_el"]
        purged_cnt = max(0, raw_tx_cnt - clean_tx_cnt)

        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        with tc1:
            st.metric(
                "Sales Lines Received",
                f"{raw_tx_cnt:,}",
                f"{clean_tx_cnt:,} Stored",
                delta_color="normal",
                help=f"Bill lines received in this store's uploads ({raw_tx_cnt:,}). Every line is kept, including the same "
                     "medicine billed on several lines of one bill (batches / repeat scans) - their strips are added up per bill."
            )
        rt = chan_dist.get("returns") or {}
        with tc2:
            st.metric(
                "Sales Lines Stored",
                f"{clean_tx_cnt:,}",
                (rt["text"]["summary"] if rt.get("text") else f"-{purged_cnt:,} not stored"),
                delta_color="off",
                help=rt["text"]["detail"] if rt.get("text") else None,
            )
        with tc3:
            st.metric(
                "Customer Sales (S0/)",
                f"{cust_sales:,}",
                help="Individual patient sales participating in RefillCare cadence and prediction algorithms"
            )
        with tc4:
            st.metric(
                "B2B / Inter-Store (SB/)",
                f"{b2b_sales:,}",
                help="Wholesale inter-store transfers strictly isolated and excluded from patient refill outreach"
            )
        with tc5:
            st.metric(
                "RefillCare Eligible",
                f"{refill_el:,}",
                help="Active patient records participating in cadence & reminder algorithms"
            )

        st.caption(
            f"**Transaction Reconciliation Audit** (through **{latest_sales_date_str}**): "
            + (rt["text"]["audit"] if rt.get("text") else f"{raw_tx_cnt:,} lines received, {clean_tx_cnt:,} stored.")
        )

        st.markdown("---")

        # Key Business Notes & Architecture
        st.markdown("#### Clinical Operations Governance & Rules Summary")
        st.markdown(
            "- **Dual-Path Routing:** High-frequency regular patients ($\\ge 6$ buys) receive precision ML forecasting; developing patients (< 6 buys) receive authoritative Days-of-Supply scheduling.\n"
            "- **Multi-Pack Awareness:** Supply intervals dynamically expand for multi-pack purchases, preventing unwanted early messages.\n"
            "- **7-Stage Lifecycle Outreach:** Gentle, structured messages timed at Day -7, -3, -1, Day 0 (Due Date), follow-ups (+2d, +5d), and Day +45 (Re-engagement).\n"
            "- **Live Repurchase Reset:** Pending reminder alerts are immediately superseded and cancelled the moment a patient repurchases.\n"
            "- **Clean Data Isolation:** Wholesale/B2B transfers (`SB/...`) and non-chronic OTC items (soaps, shampoos, balms, creams) are automatically filtered out."
        )

    # --------------------------------------------------------------------------
    # TAB 2: DATA UPDATE (UPLOAD LATEST SALES DATA)
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown("### Upload Sales Data")
        st.write("Upload newly received pharmacy sales transactions (**CSV** or **XLSX/XLS**) to inspect transaction hygiene, validate records, and dynamically update the patient refill queue.")

        render_data_status_panel()

        uploaded_file = st.file_uploader(
            "Select Sales Data File (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            key="sales_data_uploader",
            help="Supported file types: CSV, Excel (.xlsx, .xls). Any column names: they are mapped to the RefillCare fields and only those are stored.",
        )
        render_saved_column_mappings()

        if uploaded_file is not None:
            map_key = f"column_mapping::{uploaded_file.name}::{getattr(uploaded_file, 'size', 0)}"
            with st.spinner("Analyzing uploaded sales data..."):
                from database.connection import SessionLocal

                with SessionLocal() as map_db:
                    parsed_sales_df, upload_metrics = parse_and_validate_uploaded_sales_file(
                        uploaded_file,
                        existing_history_df=history_data,
                        column_mapping=st.session_state.get(map_key),
                        db=map_db,
                    )

            if upload_metrics.get("column_mapping"):
                render_column_mapping_panel(upload_metrics["column_mapping"], map_key)

            if upload_metrics.get("needs_column_mapping"):
                st.info("Confirm the column mapping above (pick the matching column for each required field) to validate this file. Only the RefillCare columns are stored.")
            elif "error" in upload_metrics:
                st.error(f"{upload_metrics['error']}")
            else:
                st.success(f"**File Uploaded:** `{uploaded_file.name}`")

                # Section 1: Overview & Coverage
                st.markdown("#### Uploaded File Summary")
                
                date_diag = upload_metrics.get("date_diagnostics", {})
                st.info(f"**Uploaded Date Range (DD-MM-YYYY):** `{upload_metrics['file_date_range']}`")

                # Detailed Date Diagnostics Panel
                with st.expander("Date Parsing & Validation Details (DD-MM-YYYY Standard)", expanded=True):
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
                            st.markdown(f"**Invalid / Unparseable Dates:** `{inv_dates:,}` (Flagged)")
                        else:
                            st.markdown("**Invalid / Unparseable Dates:** `0` (Clean)")
                        samples_str = ", ".join(date_diag.get("sample_dates_formatted", [])) if date_diag.get("sample_dates_formatted") else "None"
                        st.markdown(f"**Sample Parsed Dates:** `{samples_str}`")

                    if date_diag.get("warning"):
                        st.warning(f"**Date Notice:** {date_diag['warning']}")
                    if date_diag.get("is_ambiguous"):
                        st.error("**Ambiguous Date Format:** The uploaded dates cannot be deterministically resolved to DD-MM-YYYY. Processing has been blocked.")

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
                    st.metric("Repeated Bill Lines", f"{upload_metrics['duplicate_records']:,}",
                              help="Identical lines (same bill, medicine, quantity) - e.g. the same batch scanned twice. "
                                   "They are KEPT: all strips of one bill are added up into one purchase.")
                with c8:
                    st.metric("Invalid/Negative Qty", f"{upload_metrics['invalid_quantities']:,}")

                st.markdown("---")
                st.markdown("#### 👀 Data Preview (first 10 lines)")
                if parsed_sales_df is not None and not parsed_sales_df.empty:
                    preview_df = parsed_sales_df.copy()
                    if "invoice_date" in preview_df.columns:
                        preview_df["invoice_date"] = preview_df["invoice_date"].apply(format_date_dd_mm_yyyy)
                    preview_cols = [c for c in ["invoice_date", "customerId", "customerName", "itemId", "itemName", "quantity", "packing", "MOBILE_NO"] if c in preview_df.columns]
                    if not preview_cols:
                        preview_cols = list(preview_df.columns[:8])
                    st.dataframe(preview_df[preview_cols].head(10), width="stretch", hide_index=True)

                st.markdown("---")
                st.markdown("#### ✅ Check & Save")

                is_date_blocked = date_diag.get("is_ambiguous", False) or upload_metrics["valid_records"] == 0
                # File check, shown automatically (no separate "validate" step)
                if upload_metrics["valid_records"] > 0:
                    st.success(
                        f"**Ready to save:** {upload_metrics['valid_records']:,} of {upload_metrics['records_received']:,} lines are valid. "
                        "Lines already uploaded before are skipped automatically."
                    )
                if upload_metrics["records_requiring_review"] > 0:
                    st.warning(
                        f"**Not sales lines:** {upload_metrics['records_requiring_review']:,} lines "
                        f"({upload_metrics['invalid_quantities']:,} returns / zero quantity - return lines are saved and subtracted "
                        f"from the patient's sale, {upload_metrics['missing_customer_info']:,} without a customer). "
                        "Rows that cannot be saved are listed with the reason after saving."
                    )
                if upload_metrics["missing_mobile_numbers"] > 0:
                    st.info(
                        f"{upload_metrics['missing_mobile_numbers']:,} lines have no mobile number: they are kept for refill tracking, "
                        "but those patients cannot get WhatsApp reminders until the number is added in billing."
                    )
                if is_date_blocked:
                    st.error("**Cannot save:** the dates in this file could not be read reliably, or no line is valid. Please check the file.")

                # ONE click: validate + store the lines + re-score the patients + update their reminders
                btn_process = st.button(
                    f"✅  Save & Update Reminders ({upload_metrics['valid_records']:,} lines)",
                    key="btn_process_sales_data",
                    type="primary",
                    width="stretch",
                    disabled=is_date_blocked,
                    help="Saves the new sales lines and updates every affected patient's refill dates and WhatsApp reminders in one step.",
                )
                st.caption(
                    "Patients with missing or invalid mobile numbers are **never deleted**: their purchases still count for refill tracking."
                )

                # Handle [Process Sales Data]
                if btn_process:
                    with st.spinner("Saving the sales lines and updating refill dates and reminders... (this can take a minute)"):
                        try:
                            from database.connection import SessionLocal
                            from refillcare.data.incremental_ingest import (
                                ColumnMappingRequired,
                                DuplicateUpload,
                                file_checksum,
                                ingest_sales_batch,
                            )

                            with SessionLocal() as db:
                                process_result = ingest_sales_batch(
                                    db, parsed_sales_df, uploaded_file.name,
                                    checksum=file_checksum(uploaded_file.getvalue()), actor="streamlit",
                                    force=bool(st.session_state.get("force_reupload")),
                                )
                                if upload_metrics.get("column_mapping_result") is not None:
                                    from refillcare.data.column_mapping import mark_profile_used

                                    # Remember this layout for the store: next upload is mapped instantly
                                    mark_profile_used(db, upload_metrics["column_mapping_result"], confirmed_by="streamlit")
                        except DuplicateUpload as e:
                            process_result = {"status": "duplicate", "message": str(e)}
                        except (ColumnMappingRequired, ValueError) as e:
                            process_result = {"status": "error", "message": str(e)}
                    if process_result.get("status") == "success":
                        st.session_state.pop("active_history_data", None)
                        st.session_state.pop("active_sales_data", None)
                        st.session_state["sales_processed_metrics"] = process_result["metrics"]
                        st.session_state["sales_processed_file"] = uploaded_file.name
                        st.session_state["last_import_batch_id"] = process_result.get("import_batch_id")
                        st.session_state["online_sync_metrics"] = process_result.get("online_sync") or {}
                        st.session_state["sales_data_warnings"] = process_result.get("data_warnings") or {}
                        st.session_state["sales_skipped_rows"] = process_result.get("skipped_rows_df")
                        # Reminders were updated by the save itself; drop in-session copies so every tab reloads
                        st.session_state.pop("active_eligible_df", None)
                        st.session_state.pop("active_ineligible_df", None)
                        st.session_state.pop("prediction_gen_metrics", None)
                        st.session_state.pop("sales_validation_run", None)
                        st.session_state.pop("sales_validation_file", None)
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.rerun()
                    elif process_result.get("status") == "duplicate":
                        st.warning(f"{process_result['message']}")
                        st.checkbox("Upload it again anyway (only lines not stored yet are added)", key="force_reupload")
                    else:
                        st.error(f"{process_result.get('message', 'Processing failed.')}")

                # Display Processing Summary if available for current file
                if st.session_state.get("sales_processed_metrics") and st.session_state.get("sales_processed_file") == uploaded_file.name:
                    p_metrics = st.session_state["sales_processed_metrics"]
                    st.markdown("---")
                    st.success("**Saved. Refill dates and WhatsApp reminders are updated** for every patient in this file.")
                    st.markdown("#### What was saved")
                    st.info(f"**Import Batch ID:** `{p_metrics.get('import_batch_id', 'N/A')}`")

                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.metric("New Records Added", f"{p_metrics['new_records_added']:,}")
                    with p2:
                        st.metric("Existing Duplicates Skipped", f"{p_metrics['existing_duplicates_skipped']:,}")
                    with p3:
                        st.metric("Customers Updated", f"{p_metrics['customers_updated']:,}")
                    with p4:
                        st.metric("New Customers", f"{p_metrics['new_customers']:,}")

                    p5, p6, p7, p8 = st.columns(4)
                    with p5:
                        st.metric("New Medicines", f"{p_metrics['new_medicines']:,}")
                    with p6:
                        st.metric("Latest Sales Date (DD-MM-YYYY)", str(p_metrics['latest_sales_date']))
                    with p7:
                        st.metric("Return Lines Saved", f"{p_metrics.get('return_lines_saved', 0):,}",
                                  f"{p_metrics.get('returns_applied', 0):,} applied · {p_metrics.get('returns_waiting', 0):,} waiting",
                                  delta_color="off",
                                  help="Return lines are saved as their own rows (never a purchase) and subtracted from the sale "
                                       "they follow. A return whose sale is not uploaded yet waits and is applied automatically "
                                       "when that sale arrives.")
                    _skipped = st.session_state.get("sales_skipped_rows")
                    _n_skipped = 0 if _skipped is None else len(_skipped)
                    with p8:
                        st.metric("Rows Not Saved", f"{_n_skipped:,}",
                                  help="Rows without customer / medicine / date / quantity. Download the list below to fix them in billing.")
                    if p_metrics.get("earlier_returns_applied_now"):
                        st.info(f"{p_metrics['earlier_returns_applied_now']:,} returns from earlier uploads found their sale in this file and were applied.")
                    if _n_skipped:
                        st.download_button(
                            f"⬇️ Download the {_n_skipped:,} rows not saved (with reason)",
                            data=_skipped.to_csv(index=False).encode("utf-8"),
                            file_name=f"rows_not_saved_{p_metrics.get('import_batch_id', 'upload')}.csv",
                            mime="text/csv", key="dl_skipped_rows",
                        )
                        st.dataframe(_skipped.head(20), width="stretch", hide_index=True)

                    sync_m = st.session_state.get("online_sync_metrics") or {}
                    if sync_m:
                        s1, s2, s3, s4 = st.columns(4)
                        with s1:
                            st.metric("Customer-Items Re-scored", f"{sync_m.get('customer_items_rescored', 0):,}")
                        with s2:
                            st.metric("Reminder Cycles Created", f"{sync_m.get('cycles_created', 0):,}")
                        with s3:
                            st.metric("Cycles Superseded (Repurchase)", f"{sync_m.get('cycles_superseded', 0):,}")
                        with s4:
                            st.metric("Past-Due Stages Expired", f"{sync_m.get('stages_expired', 0):,}")
                    fresh = st.session_state.get("ml_data_freshness") or {}
                    if fresh.get("retrain_recommended"):
                        st.info(
                            "**New sales data moved the ML evaluation windows.** A model retrain is recommended - "
                            "see the AI Accuracy & Settings tab."
                        )

                # Section: Undo Last Import (Rollback)
                active_batch_id = st.session_state.get("last_import_batch_id")
                if active_batch_id:
                    st.markdown("---")
                    st.markdown("#### Undo Last Ingestion (Rollback)")
                    st.write(
                        "Safely rollback the most recent sales data import. This operation removes only "
                        "the transaction records inserted by this batch and invalidates any predictions generated from it."
                    )
                    st.warning(f"**Target Batch for Rollback:** `{active_batch_id}`")

                    col_chk, col_rb = st.columns([3, 1])
                    with col_chk:
                        confirm_rollback = st.checkbox(
                            f"I confirm that I want to rollback batch `{active_batch_id}` and restore the previous database state.",
                            key="chk_confirm_rollback_upload",
                        )
                    with col_rb:
                        if st.button("Rollback Import", key="btn_rollback_upload", disabled=not confirm_rollback):
                            with st.spinner("Rolling back import batch..."):
                                rb_result = rollback_import_batch_shared(active_batch_id, history_data, storage)
                            if rb_result["status"] == "success":
                                st.session_state.pop("active_history_data", None)
                                st.session_state.pop("active_sales_data", None)
                                st.session_state.pop("last_import_batch_id", None)
                                st.session_state.pop("sales_processed_metrics", None)
                                st.session_state.pop("sales_processed_file", None)
                                st.session_state.pop("active_eligible_df", None)
                                st.session_state.pop("active_ineligible_df", None)
                                st.session_state.pop("prediction_gen_metrics", None)
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                st.success(f"{rb_result['message']} Latest Sales Date restored to `{rb_result['restored_latest_sales_date']}`.")
                                st.rerun()
                            else:
                                st.error(rb_result["message"])
        else:
            st.info("Upload an Excel or CSV sales file above to view the business summary and update predictions.")

            # Display current active file summary if available
            st.markdown("#### Current Active File Status")
            if store_sum.get("latest_sales_date"):
                curr_range = (f"{pd.Timestamp(store_sum['first_sales_date']).strftime(UI_DATE_FORMAT)} to "
                              f"{pd.Timestamp(store_sum['latest_sales_date']).strftime(UI_DATE_FORMAT)}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sales Date Range (DD-MM-YYYY)", curr_range)
                with c2:
                    _rt = (get_enterprise_channel_distribution_cached().get("returns") or {})
                    st.metric("Sales Lines Stored", f"{store_sum['sales_lines_stored']:,}",
                              (_rt["text"]["summary"] if _rt.get("text") else None), delta_color="off",
                              help=(_rt["text"]["detail"] if _rt.get("text") else None))
                with c3:
                    active_lbl = store_sum.get("active_patients_label", "Active patients (last 12 months)")
                    active_cnt = store_sum.get("active_patients", store_sum.get("customers_monitored", 0))
                    all_time_cnt = store_sum.get("all_time_customers", store_sum.get("customers_monitored", 0))
                    st.metric(
                        active_lbl,
                        f"{active_cnt:,}",
                        delta=f"All-time: {all_time_cnt:,}",
                        delta_color="off",
                        help=f"Same number as the Pharmacy Overview. All-time: {all_time_cnt:,}.",
                    )
            else:
                st.info("No sales data stored for this store yet.")

            render_waiting_returns()

            # Rollback active batch from database if one exists
            latest_db_batch = latest_active_batch(storage)
            if latest_db_batch and history_data is not None and "import_batch_id" in history_data.columns:
                batch_id = latest_db_batch["import_batch_id"]
                if (history_data["import_batch_id"].astype(str) == batch_id).any():
                    st.markdown("---")
                    st.markdown("#### Undo Last Ingestion (Rollback)")
                    st.write(
                        f"Active import batch found in database: **`{batch_id}`** "
                        f"(Source: `{latest_db_batch['source_filename']}`, Records: {latest_db_batch['records_inserted']:,}, "
                        f"Date Range: {latest_db_batch['detected_date_range']})."
                    )
                    confirm_main_rb = st.checkbox(
                        f"I confirm that I want to rollback batch `{batch_id}`.",
                        key="chk_confirm_main_rb",
                    )
                    if st.button("Rollback Ingestion Batch", key="btn_main_rollback", disabled=not confirm_main_rb):
                        with st.spinner("Rolling back import batch..."):
                            rb_result = rollback_import_batch_shared(batch_id, history_data, storage)
                        if rb_result["status"] == "success":
                            st.session_state.pop("active_history_data", None)
                            st.session_state.pop("active_sales_data", None)
                            st.session_state.pop("active_eligible_df", None)
                            st.session_state.pop("active_ineligible_df", None)
                            st.session_state.pop("prediction_gen_metrics", None)
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.success(f"{rb_result['message']}")
                            st.rerun()
                        else:
                            st.error(rb_result["message"])

            # ------------------------------------------------------------------
            # PHARMACIST MEDICATION RULES MANAGER (SUPABASE CLOUD SYNC)
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### Pharmacist Medication Rules Manager (Live Supabase Cloud)")
            st.caption(
                "Easily add or exclude medications from automated refill reminder modeling without touching code. "
                "Rules are securely synced across your store via Supabase PostgreSQL."
            )

            with st.expander("Add / Search / Manage Medication Classification Rules", expanded=False):
                from refillcare.data.medication_classifier import (
                    add_classification_rule,
                    list_classification_rules,
                    delete_classification_rule,
                    get_dynamic_classification_rules,
                )

                # Quick Add Form
                st.markdown("##### Add Medication Classification Rule")
                with st.form("form_add_pharmacist_med_rule", clear_on_submit=True):
                    col_m1, col_m2 = st.columns([2.5, 1.5])
                    with col_m1:
                        rule_med_name = st.text_input(
                            "Medication Brand / Keyword / Item Name",
                            placeholder="e.g. TELMA, DAPAVEL, COOL-GEL, VASELINE",
                            help="Enter exact brand name or keyword. The rule will match all dosage variations."
                        )
                    with col_m2:
                        rule_type_choice = st.selectbox(
                            "Classification Decision",
                            options=["CHRONIC (Participates in Refill Modeling)", "EXCLUDE (Suppress - OTC/FMCG/Acute/Topical)"],
                            index=0,
                        )
                    rule_notes = st.text_input("Optional Pharmacist Notes", placeholder="e.g. New anti-diabetic launch / Personal care exclusion")

                    btn_save_rule = st.form_submit_button("Save Rule to Database")
                    if btn_save_rule and rule_med_name.strip():
                        r_type_val = "CHRONIC" if "CHRONIC" in rule_type_choice else "EXCLUDE"
                        save_res = add_classification_rule(
                            medication_name_or_keyword=rule_med_name.strip(),
                            rule_type=r_type_val,
                            notes=rule_notes.strip(),
                        )
                        if save_res["status"] == "success":
                            sync_info = save_res.get("sync") or {}
                            dec_cnt = sync_info.get("decisions_saved", 0)
                            cyc_cnt = sync_info.get("cycles_created", 0)
                            sync_msg = f" ({dec_cnt} customer decisions updated, {cyc_cnt} reminder cycles synced in real time)" if dec_cnt else ""
                            st.success(f"{save_res['message']}{sync_msg}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"{save_res['message']}")

                st.markdown("---")
                st.markdown("##### Active Supabase Classification Rules")
                col_s1, col_s2, col_s3 = st.columns([2, 1.5, 1])
                with col_s1:
                    search_rules_txt = st.text_input("Search Saved Rules", placeholder="e.g. TELMA, DOLO, SOAP", key="med_rule_search_txt")
                with col_s2:
                    filter_rules_type = st.selectbox("Filter Rule Type", ["All", "CHRONIC", "EXCLUDE"], key="med_rule_filter_type")
                with col_s3:
                    st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("Sync Rules", key="btn_refresh_rules"):
                        get_dynamic_classification_rules(force_refresh=True)
                        st.cache_data.clear()
                        st.success("Rules refreshed and cache synchronized from Supabase Cloud!")
                        st.rerun()

                existing_rules = list_classification_rules(search_query=search_rules_txt, rule_type_filter=filter_rules_type, limit=None)
                if existing_rules:
                    rules_df = pd.DataFrame(existing_rules)
                    disp_cols = [c for c in ["medication_name_or_keyword", "rule_type", "notes", "created_at"] if c in rules_df.columns]
                    rules_df_disp = rules_df[disp_cols].rename(columns={
                        "medication_name_or_keyword": "Medication / Keyword",
                        "rule_type": "Classification",
                        "notes": "Notes",
                        "created_at": "Added On",
                    })

                    # Metrics & Full Download Buttons
                    cnt_total = len(existing_rules)
                    cnt_excl = sum(1 for r in existing_rules if str(r.get("rule_type", "")).upper() == "EXCLUDE")
                    cnt_chron = sum(1 for r in existing_rules if str(r.get("rule_type", "")).upper() == "CHRONIC")
                    
                    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns([1.1, 1.1, 1.1, 1.5, 1.7])
                    with m_c1:
                        st.metric("Total Rules", f"{cnt_total:,}")
                    with m_c2:
                        st.metric("Exclusions", f"{cnt_excl:,}")
                    with m_c3:
                        st.metric("Chronic Allowed", f"{cnt_chron:,}")
                    with m_c4:
                        st.markdown("<div style='padding-top: 0.6rem;'></div>", unsafe_allow_html=True)
                        csv_data_disp = rules_df_disp.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Export Table ({cnt_total:,} Rows)",
                            data=csv_data_disp,
                            file_name="mediastra_active_rules_display.csv",
                            mime="text/csv",
                            key="download_all_rules_csv"
                        )
                    with m_c5:
                        st.markdown("<div style='padding-top: 0.6rem;'></div>", unsafe_allow_html=True)
                        # Complete un-truncated Supabase export with all database fields
                        all_live_rules = list_classification_rules(limit=None)
                        df_complete = pd.DataFrame(all_live_rules)
                        std_order = [c for c in ["id", "medication_name_or_keyword", "rule_type", "notes", "is_active", "created_at", "updated_at"] if c in df_complete.columns]
                        if std_order:
                            df_complete = df_complete[std_order]
                        csv_complete = df_complete.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Export All {len(all_live_rules):,} (Full CSV)",
                            data=csv_complete,
                            file_name="medication_classification_rules_complete.csv",
                            mime="text/csv",
                            key="download_all_supabase_complete_csv"
                        )

                    st.dataframe(rules_df_disp, width="stretch", hide_index=True)

                    # Delete rule selector
                    del_options = {r["id"]: f"{r['medication_name_or_keyword']} ({r['rule_type']})" for r in existing_rules if "id" in r}
                    if del_options:
                        c_del1, c_del2 = st.columns([3, 1])
                        with c_del1:
                            target_del_id = st.selectbox("Select Rule to Delete:", options=list(del_options.keys()), format_func=lambda x: del_options.get(x, x), key="del_rule_selector")
                        with c_del2:
                            st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
                            if st.button("Delete Rule", key="btn_delete_rule"):
                                del_res = delete_classification_rule(target_del_id)
                                if del_res["status"] == "success":
                                    st.success("Rule deleted successfully.")
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error(f"{del_res['message']}")
                else:
                    st.info("No matching rules found in Supabase.")

        # Display Prediction Generation Summary if available
        if st.session_state.get("prediction_gen_metrics"):
            gen_m = st.session_state["prediction_gen_metrics"]
            st.markdown("---")
            st.success("**Refill Predictions Generated Successfully**")
            st.markdown("#### Prediction Generation Summary")

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
    for _ in _visible_tab(tab_results):  # skipped when the tab is hidden (TAB_VISIBILITY)
        st.markdown("### Upcoming Refills & Predictions")
        st.write("Comparison between calculated expected refill dates and actual customer purchase dates from subsequent sales.")

        # Load evaluated prediction results from storage
        eval_results = storage.get_evaluated_prediction_results()

        # If session state has cached evaluated outcomes from newly uploaded sales, blend or use them
        if st.session_state.get("latest_evaluation_results"):
            sess_eval = st.session_state["latest_evaluation_results"]
            if sess_eval.get("predictions_evaluated", 0) > 0:
                eval_results = sess_eval

        # Section 1: Active Refill Forecast Explorer
        st.markdown("#### Active Refill Forecast Explorer")
        st.caption("Review calculated refill predictions, expected runout dates, and lifecycle timelines.")

        # Target Month Filter for Predictions
        _anchor_m = get_forecast_anchor_month()
        _pred_months = {
            f"{month_label(_anchor_m)} - Active Forecast": _anchor_m,
            f"{month_label(shift_month(_anchor_m, 1))} - Next Cycle": shift_month(_anchor_m, 1),
            f"{month_label(shift_month(_anchor_m, -1))} - Historical": shift_month(_anchor_m, -1),
            "All Forecast Months": None,
        }
        pred_month_opts = list(_pred_months.keys())
        p_c1, p_c2, p_c3, p_c4 = st.columns([1.8, 1.8, 1.4, 1.2])
        with p_c1:
            sel_pred_month_label = st.selectbox(
                "Prediction Target Month",
                pred_month_opts,
                index=0,
                key="pred_month_filter_select",
                help="Filter refill forecast by patient expected refill month.",
            )
        with p_c2:
            search_pred_cust = st.text_input("Search Customer / Mobile", key="pred_cust_search_input")
        with p_c3:
            search_pred_med = st.text_input("Search Medication", key="pred_med_search_input")
        with p_c4:
            pred_path_filter = st.selectbox("Clinical Path", ["All Paths", "PATH_A", "PATH_B"], key="pred_path_filter_select")

        sel_m_key = _pred_months.get(sel_pred_month_label)

        forecast_df = load_v1_refill_decisions_df(target_month=sel_m_key)

        if not forecast_df.empty:
            df_filtered = forecast_df.copy()
            if search_pred_cust.strip():
                c_query = search_pred_cust.strip().lower()
                c_clean = "".join(filter(str.isdigit, search_pred_cust))
                m1 = df_filtered["Customer Name"].astype(str).str.lower().str.contains(c_query, regex=False, na=False)
                m2 = df_filtered["Mobile Number"].astype(str).str.contains(search_pred_cust.strip(), regex=False, na=False)
                if c_clean:
                    m2 = m2 | df_filtered["Mobile Number"].astype(str).str.replace(r"\D", "", regex=True).str.contains(c_clean, regex=False, na=False)
                df_filtered = df_filtered[m1 | m2]
            if search_pred_med.strip():
                df_filtered = df_filtered[
                    df_filtered["Medication"].astype(str).str.lower().str.contains(search_pred_med.strip().lower(), regex=False, na=False)
                ]
            if pred_path_filter != "All Paths":
                df_filtered = df_filtered[df_filtered["Clinical Path"] == pred_path_filter]

            # Summary Cards for Predictions
            pf1, pf2, pf3, pf4 = st.columns(4)
            with pf1:
                st.metric("Forecasted Patients", f"{df_filtered['Customer Name'].nunique():,}")
            with pf2:
                st.metric("Total Prescriptions", f"{len(df_filtered):,}")
            with pf3:
                valid_phones = int((df_filtered["Mobile Status"] == "Valid").sum())
                st.metric("Delivery-Ready (Valid Phone)", f"{valid_phones:,}")
            with pf4:
                path_a_cnt = int((df_filtered["Clinical Path"] == "PATH_A").sum())
                st.metric("Path A Chronic", f"{path_a_cnt:,}")

            st.dataframe(df_filtered, width="stretch", hide_index=True)

            # CSV Export
            csv_forecast = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export Refill Forecasts (CSV)",
                data=csv_forecast,
                file_name=f"refill_forecasts_{sel_m_key or 'all'}.csv",
                mime="text/csv",
                key="btn_download_forecast_csv",
            )
        else:
            st.info(f"No eligible refill forecasts found for `{sel_pred_month_label}`.")

        st.markdown("---")
        # Section 2: Measured accuracy of the production model (latest out-of-time evaluation).
        # Every number comes from the evaluation report - no typed-in fallbacks.
        st.markdown("#### Prediction Accuracy (measured on real refills)")
        from database.connection import SessionLocal
        from refillcare.ml.service import production_accuracy

        with SessionLocal() as acc_db:
            acc = production_accuracy(acc_db)
        acc_ev = acc.get("evaluation") or {}
        prod_m = acc_ev.get("production") or {}
        if not prod_m:
            st.info("No measured evaluation yet for the production model. Run a retrain or evaluation in the ML tab.")
        else:
            win = acc_ev.get("window") or {}

            def _pct(d, k):
                v = (d or {}).get(k)
                return f"{float(v):.1f}%" if v is not None else "-"

            st.caption(
                f"Model `{acc.get('production_model_version')}` · test month {win.get('start', '?')} to {win.get('end', '?')} · "
                f"{acc_ev.get('n_paired_events') or 0:,} real refills the model had never seen."
            )
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Reminder reached patient before buying", _pct(prod_m, "reminder_before_purchase_pct"),
                      help="At least one reminder went out on or before the day the patient bought.")
            a2.metric("Reminder 0-7 days before buying", _pct(prod_m, "reminder_0_7_days_before_pct"),
                      help="A reminder went out in the week before the purchase (well-timed).")
            a3.metric("Within ±14 days", _pct(prod_m, "within_14_days_pct"))
            a4.metric("Within ±7 days", _pct(prod_m, "within_7_days_pct"),
                      help="Strict point-date accuracy: predicted date within 7 days of the actual purchase.")

            comp_rows = []
            for label, key in (("RefillCare ML model (production)", "production"), ("Baseline (personal median engine)", "baseline_engine"),
                               ("Fixed interval / days of supply", "fixed_interval_days_of_supply")):
                m = acc_ev.get(key) or {}
                if m:
                    comp_rows.append({"Method": label, "Within ±7 days": _pct(m, "within_7_days_pct"),
                                      "Within ±14 days": _pct(m, "within_14_days_pct"),
                                      "Reminder 0-7 days before buying": _pct(m, "reminder_0_7_days_before_pct"),
                                      "Avg error (days)": f"{float(m['mae_days']):.1f}" if m.get("mae_days") is not None else "-"})
            if comp_rows:
                st.markdown("##### Compared with simpler methods (same refills)")
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

            bv = acc_ev.get("business_view") or {}
            if bv:
                st.markdown("##### By patient group")
                grp_labels = {
                    "all_refills": "All refills",
                    "regular_refills": "Regular refills (lapses and early top-ups counted separately)",
                    "patients_12_plus_purchases": "Regular patients (12+ purchases)",
                    "ml_served": "Predicted by the ML model",
                    "short_history_path_b": "New patients (3-5 purchases, rule-based)",
                }
                st.dataframe(pd.DataFrame([{
                    "Group": lbl, "Refills": f"{int(bv[k].get('n', 0)):,}",
                    "Within ±7 days": _pct(bv[k], "within_7_days_pct"), "Within ±14 days": _pct(bv[k], "within_14_days_pct"),
                    "Reminder 0-7 days before buying": _pct(bv[k], "reminder_0_7_days_before_pct"),
                } for k, lbl in grp_labels.items() if isinstance(bv.get(k), dict) and bv[k].get("n")]), width="stretch", hide_index=True)
                kinds = bv.get("purchase_kinds_pct") or {}
                if kinds:
                    st.caption(
                        f"Of all refills, {kinds.get('lapse_or_gap', 0):.1f}% were long gaps (more than twice the usual interval: "
                        f"bought elsewhere, hospital stay or stopped - handled by the +45-day win-back message) and "
                        f"{kinds.get('early_top_up', 0):.1f}% were early top-ups (less than half the usual interval). "
                        "They are shown, not hidden; the promotion gate always uses all refills."
                    )
            else:
                st.caption("Group breakdown appears after the next evaluation (retrain or 'Run evaluation' in the ML tab).")

        if eval_results.get("predictions_evaluated", 0) > 0 and not eval_results["evaluated_df"].empty:
            with st.expander("Completed refill cycles (tracked outcomes)"):
                st.dataframe(eval_results["evaluated_df"], width="stretch", hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: REMINDER LIST (DATE / MONTH SELECTOR & CSV/JSON EXPORTS)
    # --------------------------------------------------------------------------
    with tab_reminders:
        st.markdown("### Patient Reminders")
        st.write("View patients scheduled for refill outreach by **Specific Date** or for the **Complete Month** (e.g. the active forecast month) from the persistent decision engine.")

        # Determine available dates and months from enterprise.db
        avail_dates = get_available_reminder_dates()
        avail_months = get_available_reminder_months()
        today_val = date.today()

        # Helper for month format labels
        month_labels = {}
        for m_str in avail_months:
            try:
                dt_m = datetime.strptime(m_str, "%Y-%m")
                month_labels[m_str] = dt_m.strftime("%B %Y") + f" ({m_str})"
            except Exception:
                month_labels[m_str] = m_str

        # Dynamic default date resolution
        if "reminder_selected_date" in st.session_state and st.session_state["reminder_selected_date"] is not None:
            default_date = st.session_state["reminder_selected_date"]
        elif today_val in avail_dates:
            default_date = today_val
        elif avail_dates:
            default_date = avail_dates[-1]
        else:
            default_date = today_val

        # Filters Row 1: Filter Mode Toggle (Date vs Month vs Rolling 30d), Target Selector, Function Filter, Stage Filter, Clinical Path Filter
        c_mode_toggle, c_target, c_func, c_stage, c_path = st.columns([1.3, 1.6, 1.4, 1.6, 1.4])
        with c_mode_toggle:
            view_mode = st.radio(
                "Filter Mode",
                ["Specific Date", "Rolling 30 Days", "Complete Month"],
                index=0,
                key="reminder_view_mode",
                help="Switch between viewing a single operational day, a dynamic 30-day continuous outreach horizon, or a full calendar month.",
            )

        with c_target:
            if view_mode == "Specific Date":
                selected_date = st.date_input(
                    "Reminder Date",
                    value=default_date,
                    key="reminder_date_selector",
                    help="Select any active date to view scheduled patient refill reminders.",
                )
                selected_month = None
                horizon_start, horizon_end = None, None
                filter_label = selected_date.strftime("%d-%m-%Y")
                file_stem = f"reminder_list_{selected_date.strftime('%Y-%m-%d')}"
                st.session_state["reminder_selected_date"] = selected_date
            elif view_mode == "Rolling 30 Days":
                selected_date = None
                selected_month = None
                horizon_start = default_date
                horizon_end = horizon_start + timedelta(days=29)
                filter_label = f"Rolling 30d ({horizon_start.strftime('%d-%b')} - {horizon_end.strftime('%d-%b-%Y')})"
                file_stem = f"reminder_list_rolling_30d_{horizon_start.strftime('%Y%m%d')}_{horizon_end.strftime('%Y%m%d')}"
                st.caption(f"**Active Horizon:** {horizon_start.strftime('%d-%b-%Y')} → {horizon_end.strftime('%d-%b-%Y')} (30 Days Outreach)")
            else:
                # Default to the active forecast month (latest sale + 1 day), else the latest month
                default_m_idx = 0
                _anchor_m = get_forecast_anchor_month()
                if _anchor_m in avail_months:
                    default_m_idx = avail_months.index(_anchor_m)
                elif avail_months:
                    default_m_idx = len(avail_months) - 1

                selected_month = st.selectbox(
                    "Select Month",
                    options=avail_months,
                    index=default_m_idx,
                    format_func=lambda m: month_labels.get(m, m),
                    key="reminder_month_selector",
                    help="Select target prediction month to view all scheduled outreach.",
                )
                selected_date = None
                horizon_start, horizon_end = None, None
                filter_label = month_labels.get(selected_month, selected_month)
                file_stem = f"reminder_list_{selected_month}"

        with c_func:
            func_filter_mode = st.selectbox(
                "Refill Function",
                [
                    "All Functions (3-Tier)",
                    "DUE_REFILL (Due Soon)",
                    "REFILL_FOLLOW_UP (Follow-up)",
                    "LAPSED_REENGAGEMENT (+45d)",
                ],
                key="reminder_func_selector",
                help="Segregate reminders by communication purpose: Advance due reminder, Follow-up, or +45d Re-engagement.",
            )

        with c_stage:
            schedule_mode = st.selectbox(
                "Lifecycle Stage Filter",
                [
                    "All Active Stages (-7d, -3d, -1d, 0d, +2d, +5d, +45d)",
                    "Stage: -7 days (Due in 7 Days)",
                    "Stage: -3 days (Due in 3 Days)",
                    "Stage: -1 day (Due Tomorrow)",
                    "Stage: 0 days (Due Today)",
                    "Stage: +2 days (Follow-up)",
                    "Stage: +5 days (Follow-up)",
                    "Stage: +45 days (Re-engagement)",
                ],
                key="reminder_schedule_mode_selector",
                help="Filter by specific patient lifecycle communication stages.",
            )

        with c_path:
            path_mode = st.selectbox(
                "Clinical Path Filter",
                [
                    "All Paths (Path A + Path B)",
                    "Path A (Chronic Adherence, ≥ 6 Buys)",
                    "Path B (Developing Adherence, < 6 Buys)",
                ],
                key="reminder_path_mode_selector",
                help="Filter between Path A historical cadence patients and Path B developing DOS patients.",
            )

        # Filters Row 2: Customer, Medication, Mobile Status, Transaction Channel, Search Button
        c_cust, c_med, c_stat, c_chan, c_btn = st.columns([1.8, 1.8, 1.1, 1.3, 0.9])
        with c_cust:
            search_cust = st.text_input("Search Customer / Phone", key="rem_search_cust")
        with c_med:
            search_med = st.text_input("Search Medication", key="rem_search_med")
        with c_stat:
            status_filter = st.selectbox("Mobile Status", ["All", "Valid", "Missing", "Invalid format"], key="rem_status_filter")
        with c_chan:
            channel_filter = st.selectbox(
                "Transaction Channel",
                ["Customer Sales (Default)", "All Channels", "B2B / Inter-Store (Audit)", "Unknown"],
                key="rem_channel_filter",
                help="Only individual customer sales (S0/...) enter the reminder queue. Select B2B / Inter-Store to audit excluded transactions."
            )
        with c_btn:
            st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
            st.button("Search", key="btn_apply_rem_search", width="stretch")

        # Load queue from persistent enterprise.db for selected date, month, or rolling 30-day range
        db_queue_df = load_v1_reminder_queue_df(
            target_date=selected_date,
            target_month=selected_month,
            start_date=horizon_start if view_mode == "Rolling 30 Days" else None,
            end_date=horizon_end if view_mode == "Rolling 30 Days" else None,
        )

        if not db_queue_df.empty:
            working_rem_df = db_queue_df.copy()
        else:
            # Fallback to in-memory scheduler if DB has no records
            full_schedules_df = generate_reminder_schedule_table(eligible_df)
            if not full_schedules_df.empty and "Reminder Date" in full_schedules_df.columns:
                if view_mode == "Specific Date" and selected_date:
                    selected_date_str = selected_date.strftime("%Y-%m-%d")
                    working_rem_df = full_schedules_df[full_schedules_df["Reminder Date"] == selected_date_str].copy()
                elif view_mode == "Rolling 30 Days" and horizon_start and horizon_end:
                    s_str, e_str = horizon_start.strftime("%Y-%m-%d"), horizon_end.strftime("%Y-%m-%d")
                    working_rem_df = full_schedules_df[(full_schedules_df["Reminder Date"] >= s_str) & (full_schedules_df["Reminder Date"] <= e_str)].copy()
                elif selected_month:
                    working_rem_df = full_schedules_df[full_schedules_df["Reminder Date"].astype(str).str.startswith(selected_month)].copy()
                else:
                    working_rem_df = pd.DataFrame()
            else:
                working_rem_df = pd.DataFrame()

        # Ensure Refill Function is populated
        if not working_rem_df.empty:
            if "Refill Function" not in working_rem_df.columns:
                if "refill_function" in working_rem_df.columns:
                    working_rem_df["Refill Function"] = working_rem_df["refill_function"]
                elif "stage_offset" in working_rem_df.columns:
                    from refillcare.whatsapp.template_formatter import get_refill_function
                    working_rem_df["Refill Function"] = working_rem_df["stage_offset"].apply(get_refill_function)
                elif "Reminder Stage" in working_rem_df.columns:
                    def _derive_rf(stg):
                        s = str(stg).lower()
                        if "+2" in s or "+5" in s:
                            return "REFILL_FOLLOW_UP"
                        elif "+45" in s or "+40" in s or "re-engagement" in s or "lapsed" in s:
                            return "LAPSED_REENGAGEMENT"
                        return "DUE_REFILL"
                    working_rem_df["Refill Function"] = working_rem_df["Reminder Stage"].apply(_derive_rf)
                else:
                    working_rem_df["Refill Function"] = "DUE_REFILL"

        # Tag channel provenance for transparency
        if not working_rem_df.empty:
            working_rem_df["Transaction Channel"] = "CUSTOMER_SALE"
            working_rem_df["RefillCare Eligible"] = "YES"

        # Summary KPIs for the selected date/month
        tot_on_date = len(working_rem_df)
        valid_mob_on_date = int((working_rem_df["Mobile Status"] == "Valid").sum()) if tot_on_date > 0 and "Mobile Status" in working_rem_df.columns else 0
        missing_mob_on_date = tot_on_date - valid_mob_on_date
        path_a_on_date = int((working_rem_df["path"] == "PATH_A").sum()) if tot_on_date > 0 and "path" in working_rem_df.columns else 0

        overview_title = f"#### {'Monthly' if view_mode == 'Complete Month' else 'Daily'} Queue Overview ({filter_label})"
        st.markdown(overview_title)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Scheduled Patients", f"{tot_on_date:,}")
        with s2:
            st.metric("Delivery-Ready (Valid Phone)", f"{valid_mob_on_date:,}")
        with s3:
            st.metric("Missing / Manual Review Phone", f"{missing_mob_on_date:,}")
        with s4:
            st.metric("Path A Chronic Patients", f"{path_a_on_date:,}")

        st.markdown("---")

        # Handle B2B Audit View
        if channel_filter == "B2B / Inter-Store (Audit)":
            st.warning("**B2B / Inter-Store Audit View:** Displaying excluded inter-store transactions (SB/...). These records do not participate in patient reminder delivery.")
            b2b_total, b2b_df = load_store_b2b_lines(100)
            if b2b_total:
                st.info(f"Total B2B Inter-Store Transactions in this store: **{b2b_total:,}** (RefillCare Eligible: **NO**, Exclusion Reason: **INTER_STORE_TRANSACTION**)")
                st.dataframe(
                    b2b_df[["invoice_number", "invoice_date", "customerName", "itemName", "quantity", "transaction_type", "refillcare_eligible", "exclusion_reason"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No B2B / inter-store transactions in this store's sales data.")
            filtered_reminders = pd.DataFrame()
        elif channel_filter == "Unknown":
            st.warning("**Unknown Transactions Audit View:** No unknown transaction types detected in source dataset.")
            filtered_reminders = pd.DataFrame()
        else:
            # Apply Regular Filters
            filtered_reminders = pd.DataFrame(working_rem_df.copy())

            # Refill function filter
            if func_filter_mode.startswith("DUE_REFILL") and not filtered_reminders.empty and "Refill Function" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[filtered_reminders["Refill Function"] == "DUE_REFILL"])
            elif func_filter_mode.startswith("REFILL_FOLLOW_UP") and not filtered_reminders.empty and "Refill Function" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[filtered_reminders["Refill Function"] == "REFILL_FOLLOW_UP"])
            elif func_filter_mode.startswith("LAPSED_REENGAGEMENT") and not filtered_reminders.empty and "Refill Function" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[filtered_reminders["Refill Function"] == "LAPSED_REENGAGEMENT"])

            # Lifecycle stage filter
            if schedule_mode.startswith("Stage:") and not filtered_reminders.empty:
                import re
                stg_target = schedule_mode.replace("Stage:", "").strip()
                m_offset = re.search(r"Stage:\s*([+-]?\d+)", schedule_mode)
                target_offset = int(m_offset.group(1)) if m_offset else None

                cond_stage = pd.Series(False, index=filtered_reminders.index)
                if "Reminder Stage" in filtered_reminders.columns:
                    cond_stage = cond_stage | filtered_reminders["Reminder Stage"].astype(str).str.contains(stg_target[:6], case=False, na=False, regex=False)
                if target_offset is not None and "stage_offset" in filtered_reminders.columns:
                    cond_stage = cond_stage | (filtered_reminders["stage_offset"] == target_offset)

                filtered_reminders = pd.DataFrame(filtered_reminders[cond_stage])

            # Path filter
            if path_mode.startswith("Path A") and not filtered_reminders.empty and "path" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[filtered_reminders["path"] == "PATH_A"])
            elif path_mode.startswith("Path B") and not filtered_reminders.empty and "path" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[filtered_reminders["path"] == "PATH_B"])

            # Search Filters (Customer Name and Phone Number)
            if search_cust.strip() and not filtered_reminders.empty:
                col_c = "Customer Name" if "Customer Name" in filtered_reminders.columns else "Customer"
                term = search_cust.strip().lower()
                clean_digits = "".join(filter(str.isdigit, term))

                cond_name = filtered_reminders[col_c].astype(str).str.lower().str.contains(term, na=False, regex=False)
                cond_phone = pd.Series(False, index=filtered_reminders.index)
                for p_col in ["Mobile Number", "raw_phone_number", "phone_number", "MOBILE_NO"]:
                    if p_col in filtered_reminders.columns:
                        p_str = filtered_reminders[p_col].astype(str)
                        cond_phone = cond_phone | p_str.str.lower().str.contains(term, na=False, regex=False)
                        if clean_digits:
                            p_clean = p_str.str.replace(r"\D", "", regex=True)
                            cond_phone = cond_phone | p_clean.str.contains(clean_digits, na=False, regex=False)

                filtered_reminders = pd.DataFrame(filtered_reminders[cond_name | cond_phone])

            if search_med.strip() and not filtered_reminders.empty:
                col_m = "Medication" if "Medication" in filtered_reminders.columns else "Medicine"
                filtered_reminders = pd.DataFrame(filtered_reminders[
                    filtered_reminders[col_m].astype(str).str.lower().str.contains(search_med.strip().lower(), na=False, regex=False)
                ])

            if status_filter != "All" and not filtered_reminders.empty and "Mobile Status" in filtered_reminders.columns:
                filtered_reminders = pd.DataFrame(filtered_reminders[
                    filtered_reminders["Mobile Status"] == status_filter
                ])

            # Header and Technical View Toggle
            t_col1, t_col2 = st.columns([3, 1])
            with t_col1:
                st.markdown(f"##### Scheduled Patients Queue ({len(filtered_reminders):,} records)")
            with t_col2:
                show_technical_details = st.checkbox(
                    "Technical Diagnostics",
                    value=False,
                    key="chk_show_tech_details_reminders",
                    help="Show internal clinical regimen, archetype, stability tier, and decision provenance columns.",
                )

            # Essential Enterprise Columns (Clean View for Store Staff)
            core_display_columns = [
                "Customer Name",
                "Mobile Number",
                "Refill Function",
                "Medication",
                "Last Purchase Date",
                "Estimated Days of Supply",
                "Expected Refill Date",
                "Reminder Date",
                "Reminder Stage",
                "Status",
            ]

            # Advanced Provenance Columns (Shown when toggle is enabled)
            technical_display_columns = [
                "Clinical Regimen",
                "Consensus Archetype",
                "Clinical Path",
                "Stability Tier",
                "Decision Reason",
                "Transaction Channel",
                "Mobile Status",
            ]

            display_columns = core_display_columns + (technical_display_columns if show_technical_details else [])

            if not filtered_reminders.empty:
                avail_cols = [c for c in display_columns if c in filtered_reminders.columns]
                st.dataframe(
                    filtered_reminders[avail_cols],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(f"No customer reminders match the selected criteria for **{filter_label}**.")

        # Download Reminder List Buttons (CSV and JSON)
        col_csv, col_json, col_info = st.columns([1.8, 1.8, 2.8])
        with col_csv:
            from reminder.reminder_engine import RefillReminderEngine
            reminder_csv_bytes = RefillReminderEngine.build_10_column_export_csv(pd.DataFrame(filtered_reminders))
            st.download_button(
                label=f"Export CSV ({'Month' if view_mode == 'Complete Month' else 'Date'})",
                data=reminder_csv_bytes,
                file_name=f"{file_stem}_export.csv",
                mime="text/csv",
                help="Download operational reminder delivery CSV containing only valid mobile numbers (10 standard columns).",
            )
        with col_json:
            import json
            json_export_data = {
                "filter_mode": view_mode,
                "target_filter": filter_label,
                "total_records": len(filtered_reminders),
                "generated_at": datetime.utcnow().isoformat(),
                "records": filtered_reminders.to_dict(orient="records") if not filtered_reminders.empty else [],
            }
            json_str = json.dumps(json_export_data, indent=2, default=str)
            st.download_button(
                label=f"Export JSON ({'Month' if view_mode == 'Complete Month' else 'Date'})",
                data=json_str.encode("utf-8"),
                file_name=f"{file_stem}_export.json",
                mime="application/json",
                help="Download full structured JSON representation with clinical metadata and delivery payloads.",
            )
        with col_info:
            valid_count_on_date = len(pd.DataFrame(filtered_reminders[filtered_reminders["Mobile Status"] == "Valid"])) if not filtered_reminders.empty and "Mobile Status" in filtered_reminders.columns else 0
            st.write(
                f"**{valid_count_on_date:,}** delivery-ready reminder records scheduled for **{filter_label}** "
                f"(out of **{len(filtered_reminders):,}** total customer records; records without valid mobile numbers are excluded from delivery CSV and available in Review tab)."
            )

        # Informational Explainer Box
        with st.expander("Clinical Operations & Decision Engine Architecture", expanded=False):
            st.markdown("""
            **V1 Unified Refill Decision Engine Architecture:**
            - **Dual-Path Clinical Routing:**
              - **Path A (Chronic Adherence, ≥ 6 purchases):** Evaluated with MAD stability filtering. Highly stable cohorts use personal historical median intervals; medium/unstable cohorts are corroborated with Days-of-Supply (DOS).
              - **Path B (Developing Adherence, < 6 purchases):** Governed by Days-of-Supply (DOS) calculated from verified quantity sold and daily consumption rates with safety bounds.
            - **Multi-Pack & Quantity Scaling:** When a customer purchases multiple packs (e.g. 4 packs instead of typical 2), refill dates scale dynamically with partial-purchase safety caps, preventing premature notifications.
            - **6-Stage Lifecycle Triggers:** Scheduled at **Day -7**, **Day -3**, **Day -1**, **Day 0 (Due)**, **Day +2**, and **Day +5** with stage-specific patient messaging.
            - **Repurchase Auto-Reset:** When a patient buys medication early or on time, previous pending stages are automatically marked `SUPERSEDED_BY_PURCHASE` to prevent duplicate alerts.
            """)

    # --------------------------------------------------------------------------
    # TAB 5: MED-SYNC (MULTI-PRESCRIPTION SYNCHRONIZATION & APPOINTMENT BUNDLING)
    # --------------------------------------------------------------------------
    with tab_medsync:
        st.markdown("### Combined Prescriptions (Med-Sync)")
        st.caption(
            "Med-Sync automatically clusters multiple chronic prescriptions for each patient due within a configurable "
            "synchronization window into a single unified appointment reminder. This eliminates patient notification fatigue, "
            "reduces delivery friction by 50%+, and maximizes long-term adherence."
        )

        from refillcare.engine.med_sync import MedSyncEngine

        # Determine latest sales date to establish next-month prediction horizon
        latest_sales_dt = None
        eval_ds = history_data if (history_data is not None and not history_data.empty) else recent_data
        if eval_ds is not None and not eval_ds.empty and "invoice_date" in eval_ds.columns:
            dts = pd.to_datetime(eval_ds["invoice_date"], errors="coerce").dropna()
            if not dts.empty:
                latest_sales_dt = dts.max().date()

        if latest_sales_dt is None:
            try:
                from database.connection import SessionLocal
                from sqlalchemy import text
                with SessionLocal() as db_temp:
                    latest_lpd = db_temp.execute(text("SELECT MAX(last_purchase_date) FROM refill_decisions")).scalar()
                    if latest_lpd:
                        latest_sales_dt = datetime.strptime(str(latest_lpd)[:10], "%Y-%m-%d").date()
            except Exception:
                pass

        if latest_sales_dt:
            if latest_sales_dt.month == 12:
                target_next_year = latest_sales_dt.year + 1
                target_next_month = 1
            else:
                target_next_year = latest_sales_dt.year
                target_next_month = latest_sales_dt.month + 1
            last_sales_month_str = latest_sales_dt.strftime("%B %Y")
        else:
            now_dt = date.today()
            target_next_year = now_dt.year
            target_next_month = now_dt.month
            last_month_dt = (now_dt.replace(day=1) - timedelta(days=1))
            last_sales_month_str = last_month_dt.strftime("%B %Y")

        target_next_month_str = date(target_next_year, target_next_month, 1).strftime("%B %Y")

        # Run Med-Sync Clustering on full active persistent decisions
        med_sync_engine = MedSyncEngine(sync_window_days=0)
        medsync_candidates = load_active_refill_decisions_for_medsync(eligible_df)

        # Controls Row 1: View Mode, Target Date/Month, Sync Window, Bundle Type
        c_vmode, c_target, c_sync_win, c_sync_filt = st.columns([1.5, 2.2, 1.3, 1.8])

        with c_vmode:
            medsync_view_mode = st.selectbox(
                "Filter Mode",
                ["Specific Date", "Complete Month"],
                index=0,
                key="medsync_view_mode_selector",
                help="Switch between pinpointing a specific prediction target date (default) or viewing all bundles for a complete calendar month.",
            )

        with c_sync_win:
            sync_window = st.slider(
                "Sync Window (Days)",
                min_value=0,
                max_value=14,
                value=0,
                step=1,
                key="medsync_window_slider",
                help="0 = Exact Same-Date Group-By (default in QA mode). 1-14 = Multi-Day Synchronized Window.",
            )

        # Cluster with chosen window
        all_bundles = med_sync_engine.cluster_patient_decisions(medsync_candidates, sync_window_days=sync_window)

        # Build dynamic month options with target next month as default first option
        available_tuples = sorted(list(set((b.anchor_refill_date.year, b.anchor_refill_date.month) for b in all_bundles)))
        target_tuple = (target_next_year, target_next_month)
        
        future_tuples = [t for t in available_tuples if t >= target_tuple]
        past_tuples = [t for t in available_tuples if t < target_tuple]
        ordered_tuples = ([target_tuple] if target_tuple in available_tuples else []) + [t for t in future_tuples if t != target_tuple] + past_tuples

        month_display_map = {}
        for y, m in ordered_tuples:
            m_name = date(y, m, 1).strftime("%B %Y")
            if (y, m) == target_tuple:
                month_display_map[f"{y}-{m:02d}"] = f"{m_name} (Active Target — Default)"
            else:
                month_display_map[f"{y}-{m:02d}"] = f"{m_name}"

        month_display_map["ALL"] = "All Future Months (Full Pipeline)"
        month_keys = list(month_display_map.keys())

        with c_target:
            if medsync_view_mode == "Specific Date":
                if "reminder_selected_date" in st.session_state and st.session_state["reminder_selected_date"] is not None:
                    default_target_date = st.session_state["reminder_selected_date"]
                elif today_val in avail_dates:
                    default_target_date = today_val
                elif avail_dates:
                    default_target_date = avail_dates[-1]
                else:
                    default_target_date = date.today()

                medsync_selected_date = st.date_input(
                    "Prediction Target Date",
                    value=default_target_date,
                    key="medsync_date_selector",
                    help="Select specific date to view all patient bundles anchored on that day.",
                )
                selected_month_key = None
            else:
                medsync_selected_date = None
                selected_month_key = st.selectbox(
                    "Prediction Target Month",
                    options=month_keys,
                    index=0,
                    format_func=lambda k: month_display_map.get(k, k),
                    key="medsync_month_selector",
                    help=f"Select prediction target month. By default, RefillCare focuses on the active target month ({target_next_month_str}) following the latest uploaded sales data ({last_sales_month_str}).",
                )

        with c_sync_filt:
            bundle_filter_type = st.selectbox(
                "Bundle Type",
                [
                    "All Bundles",
                    "Multi-Prescription Bundles (≥2 Meds)",
                    "Single-Prescription Bundles",
                ],
                key="medsync_bundle_filter",
            )

        # Controls Row 2: Customer Name / Phone, Medication, Mobile Status, Lifecycle Tier, Search Button
        c_sync_cust, c_sync_med, c_sync_status, c_sync_tier, c_sync_btn = st.columns([1.8, 1.8, 1.1, 1.4, 0.9])
        with c_sync_cust:
            search_sync_cust = st.text_input("Search Customer / Phone", key="medsync_search_cust")
        with c_sync_med:
            search_sync_med = st.text_input("Search Medication", key="medsync_search_med")
        with c_sync_status:
            search_sync_status = st.selectbox("Mobile Status", ["All", "Valid", "Missing"], key="medsync_status_filter")
        with c_sync_tier:
            search_sync_tier = st.selectbox("Refill Function / Tier", ["All Functions", "DUE_REFILL (Due Soon)", "REFILL_FOLLOW_UP (Follow-up)", "LAPSED_REENGAGEMENT (+45d)"], key="medsync_tier_filter")
        with c_sync_btn:
            st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
            st.button("Search", key="btn_apply_medsync_search", width="stretch")

        # Load candidates from scheduled review queue for selected target date/month
        if medsync_view_mode == "Specific Date" and medsync_selected_date:
            queue_candidates = load_v1_reminder_queue_df(target_date=medsync_selected_date)
            active_target_label = medsync_selected_date.strftime("%d %B %Y")
        elif selected_month_key and selected_month_key != "ALL":
            queue_candidates = load_v1_reminder_queue_df(target_month=selected_month_key)
            sel_y, sel_m = map(int, selected_month_key.split("-"))
            active_target_label = date(sel_y, sel_m, 1).strftime("%B %Y")
        else:
            queue_candidates = pd.DataFrame()
            active_target_label = "All Future Months"

        if not queue_candidates.empty:
            medsync_input = queue_candidates
        else:
            medsync_input = load_active_refill_decisions_for_medsync(eligible_df)

        # Cluster candidate records with selected sync window
        all_bundles = med_sync_engine.cluster_patient_decisions(medsync_input, sync_window_days=sync_window)

        # Filter by refill function
        if search_sync_tier.startswith("DUE_REFILL"):
            month_filtered_bundles = [b for b in all_bundles if getattr(b, "refill_function", "DUE_REFILL") == "DUE_REFILL"]
        elif search_sync_tier.startswith("REFILL_FOLLOW_UP"):
            month_filtered_bundles = [b for b in all_bundles if getattr(b, "refill_function", "DUE_REFILL") == "REFILL_FOLLOW_UP"]
        elif search_sync_tier.startswith("LAPSED_REENGAGEMENT"):
            month_filtered_bundles = [b for b in all_bundles if getattr(b, "refill_function", "DUE_REFILL") == "LAPSED_REENGAGEMENT"]
        else:
            month_filtered_bundles = all_bundles

        # Calculate target-specific impact summary
        sync_impact = med_sync_engine.summarize_sync_impact(month_filtered_bundles)

        st.info(
            f"**Active Prediction Target: {active_target_label}** — Clustered refill schedules "
            f"(predicted from uploaded sales up to **{last_sales_month_str}** with **{sync_window}d** sync window). "
            f"**{sync_impact['total_prescriptions_synced']:,}** prescriptions grouped into **{sync_impact['total_dispatches_generated']:,}** bundles."
        )

        # Enterprise Impact KPI Row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("Prescriptions Synced", f"{sync_impact['total_prescriptions_synced']:,}")
        with kpi2:
            st.metric("Total Refill Bundles", f"{sync_impact['total_dispatches_generated']:,}")
        with kpi3:
            st.metric("Multi-Med Bundles (≥2)", f"{sync_impact['multi_item_bundles_count']:,}", f"{sync_impact['multi_item_bundle_rate_pct']:.1f}% of total")
        with kpi4:
            st.metric("Messages Saved", f"{sync_impact['individual_messages_saved']:,}", f"-{sync_impact['message_reduction_rate_pct']:.1f}% noise", delta_color="normal")
        with kpi5:
            st.metric("Max Meds in Bundle", f"{sync_impact['max_items_in_single_bundle']} Meds", f"Avg {sync_impact['avg_items_per_bundle']} / bundle")

        st.markdown("---")

        # Convert target-filtered bundles into structured dataframe for display and filtering
        bundle_rows = []
        for b in month_filtered_bundles:
            med_names = [item.item_name for item in b.synced_items]
            meds_str = ", ".join(med_names)
            p10_str = b.earliest_p10_date.strftime("%d-%m-%Y") if b.earliest_p10_date else "-"
            p90_str = b.latest_p90_date.strftime("%d-%m-%Y") if b.latest_p90_date else "-"
            disp_phone = format_display_phone_10digits(b.mobile_no or "")

            tier_label_map = {"DUE": "Due / Advance", "FOLLOWUP": "Follow-up", "LAPSED": "Lapsed Re-engagement"}
            tier_disp = tier_label_map.get(getattr(b, "lifecycle_tier", "DUE"), "Due / Advance")

            bundle_rows.append({
                "Bundle ID": b.bundle_id,
                "Customer ID": b.customer_id,
                "Customer Name": b.customer_name,
                "Mobile Number": disp_phone,
                "raw_mobile_no": b.mobile_no or "",
                "Mobile Status": determine_mobile_status(b.mobile_no or ""),
                "Refill Function": getattr(b, "refill_function", "DUE_REFILL"),
                "Lifecycle Stage": tier_disp,
                "Anchor Due Date": b.anchor_refill_date.strftime("%d-%m-%Y"),
                "Anchor Medication": b.anchor_item_name,
                "Synced Prescriptions": meds_str,
                "P10 Early Window": p10_str,
                "P90 Late Alert": p90_str,
                "Total Meds": b.total_items_count,
                "Messages Saved": b.message_reduction_count,
                "raw_bundle": b,
            })

        bundles_df = pd.DataFrame(bundle_rows) if bundle_rows else pd.DataFrame()

        # Apply user filters
        filtered_bundles_df = bundles_df.copy()
        if not filtered_bundles_df.empty:
            if bundle_filter_type == "Multi-Prescription Bundles (≥2 Meds)":
                filtered_bundles_df = filtered_bundles_df[filtered_bundles_df["Total Meds"] >= 2]
            elif bundle_filter_type == "Single-Prescription Bundles":
                filtered_bundles_df = filtered_bundles_df[filtered_bundles_df["Total Meds"] == 1]

            if search_sync_status != "All":
                filtered_bundles_df = filtered_bundles_df[filtered_bundles_df["Mobile Status"] == search_sync_status]

            if search_sync_cust.strip():
                term_c = search_sync_cust.strip().lower()
                clean_dig = "".join(filter(str.isdigit, term_c))
                cond_name = filtered_bundles_df["Customer Name"].astype(str).str.lower().str.contains(term_c, na=False, regex=False)
                cond_phone = filtered_bundles_df["Mobile Number"].astype(str).str.lower().str.contains(term_c, na=False, regex=False)
                if clean_dig and "raw_mobile_no" in filtered_bundles_df.columns:
                    cond_phone = cond_phone | filtered_bundles_df["raw_mobile_no"].astype(str).str.replace(r"\D", "", regex=True).str.contains(clean_dig, na=False, regex=False)
                filtered_bundles_df = filtered_bundles_df[cond_name | cond_phone]

            if search_sync_med.strip() and "Synced Prescriptions" in filtered_bundles_df.columns:
                filtered_bundles_df = filtered_bundles_df[
                    filtered_bundles_df["Synced Prescriptions"].astype(str).str.lower().str.contains(search_sync_med.strip().lower(), na=False, regex=False)
                ]

        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.markdown(f"#### Synchronized Refill Bundles Queue ({len(filtered_bundles_df):,} matching)")
        with t_col2:
            show_tech_medsync = st.checkbox(
                "Technical Diagnostics",
                value=False,
                key="chk_show_tech_details_medsync",
                help="Show internal clinical P10 Early Window and P90 Late Alert quantile bounds.",
            )

        core_display_cols = [
            "Customer Name",
            "Mobile Number",
            "Refill Function",
            "Anchor Due Date",
            "Anchor Medication",
            "Synced Prescriptions",
            "Total Meds",
            "Messages Saved",
        ]
        if show_tech_medsync:
            display_cols = core_display_cols + ["P10 Early Window", "P90 Late Alert"]
        else:
            display_cols = core_display_cols

        if not filtered_bundles_df.empty:
            st.dataframe(
                filtered_bundles_df[display_cols],
                width="stretch",
                hide_index=True,
            )

            # Export Button
            st.markdown("---")
            export_medsync_df = filtered_bundles_df[[
                "Customer ID",
                "Customer Name",
                "Mobile Number",
                "Mobile Status",
                "Anchor Due Date",
                "Anchor Medication",
                "Synced Prescriptions",
                "Total Meds",
                "Messages Saved",
            ]].copy()
            csv_buf = io.StringIO()
            export_medsync_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Export Med-Sync Schedule (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"medsync_delivery_schedule_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Export synchronized multi-prescription delivery schedule with consolidated medications.",
            )
        else:
            st.info("No synchronized bundles matched your search criteria.")

    # --------------------------------------------------------------------------
    # TAB 6: CUSTOMERS NEEDING REVIEW
    # --------------------------------------------------------------------------
    with tab_review:
        from database.connection import SessionLocal
        from refillcare.engine.pharmacist_overrides import (
            MAX_INTERVAL,
            MIN_INTERVAL,
            add_to_reminders,
            add_many_to_reminders,
            list_pharmacist_added,
            remove_from_reminders,
            review_candidates,
        )

        st.markdown("### Patients Needing Review")
        st.caption(
            "Check patients who are not getting automatic reminders. If a patient on a regular (chronic) medicine "
            "should be reminded, click the row and add them. Medicines the medication rules mark as non-chronic "
            "(acute, OTC, FMCG) are never listed."
        )

        today_rev = date.today()
        # Controls row 1: one review date (default today) + search by customer name / mobile number
        rc1, rc2 = st.columns([1.0, 2.0])
        with rc1:
            rev_date = st.date_input("Prediction / Review Target Date", value=today_rev, key="review_target_date", format="DD/MM/YYYY")
        with rc2:
            search_rev_cust = st.text_input("Search customer name / mobile number", key="review_search_cust",
                                            placeholder="e.g. Ramesh or 98765 43210")
        # Only the selected day — no ± window
        rev_start = rev_end = rev_date
        rev_label = f"refill due on {rev_date.strftime('%d %b %Y')}"

        # Controls row 2: category, mobile status, medicine search
        rk1, rk2, rk4 = st.columns([2.2, 1.0, 1.3])
        with rk1:
            rev_cat = st.selectbox(
                "Review Category",
                [
                    "Not getting reminders (chronic / not classified)",
                    "Missing mobile number (in reminders)",
                    "Unstable refill pattern (in reminders)",
                    "B2B / Inter-Store excluded (audit)",
                ],
                key="review_cat_filter",
            )
        with rk2:
            rev_mobile_filter = st.selectbox("Mobile Status", ["All", "Valid", "Missing", "Invalid"], key="review_mobile_status_filter")
        with rk4:
            search_rev_med = st.text_input("Search medicine", key="review_search_med")

        def _search_filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            if rev_mobile_filter != "All" and "Mobile Status" in df.columns:
                df = df[df["Mobile Status"] == rev_mobile_filter]
            if search_rev_cust.strip():
                q_txt = search_rev_cust.strip().lower()
                q_dig = "".join(filter(str.isdigit, search_rev_cust))
                m_name = df["Customer Name"].astype(str).str.lower().str.contains(q_txt, regex=False, na=False)
                m_mob = (df["Mobile Number"].astype(str).str.replace(r"\D", "", regex=True).str.contains(q_dig, regex=False, na=False)
                         if q_dig else pd.Series(False, index=df.index))
                df = df[m_name | m_mob]
            if search_rev_med.strip():
                df = df[df["Medication"].astype(str).str.lower().str.contains(search_rev_med.strip().lower(), regex=False, na=False)]
            return df

        if rev_cat.startswith("Not getting reminders"):
            with SessionLocal() as rev_db:
                cand_rows = review_candidates(rev_db, rev_start, rev_end)
            cand_df = pd.DataFrame([{
                "customer_item_key": r["customer_item_key"],
                "Customer Name": r["customer_name"],
                "Mobile Number": format_display_phone_10digits(r["mobile_number"]) if r["mobile_number"] else "-",
                "Mobile Status": r["mobile_status"],
                "Medication": r["medicine"],
                "Medicine Type": r["medicine_class"],
                "Purchases": r["purchase_count"],
                "Last Purchase": datetime.strptime(r["last_purchase_date"], "%Y-%m-%d").strftime("%d-%m-%Y"),
                "Usual Gap (days)": r["usual_gap_days"],
                "Estimated Due": datetime.strptime(r["estimated_due_date"], "%Y-%m-%d").strftime("%d-%m-%Y"),
                "Why Not In Reminders": map_reason_client_friendly(r["reason_not_in_reminders"]),
                "_raw": r,
            } for r in cand_rows])
            cand_df = _search_filter(cand_df)

            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Not getting reminders", f"{len(cand_df):,}")
            kc2.metric("Chronic medicine", f"{int((cand_df['Medicine Type'] == 'Chronic').sum()) if not cand_df.empty else 0:,}")
            kc3.metric("Not classified yet", f"{int((cand_df['Medicine Type'] == 'Not classified').sum()) if not cand_df.empty else 0:,}")
            kc4.metric("No valid mobile", f"{int((cand_df['Mobile Status'] != 'Valid').sum()) if not cand_df.empty else 0:,}")

            if st.session_state.get("bulk_add_msg"):
                st.success(st.session_state.pop("bulk_add_msg"))
            if cand_df.empty:
                st.info(f"No patients missing from reminders for {rev_label}.")
            else:
                st.markdown(f"#### Patients not getting reminders ({len(cand_df):,}) · {rev_label}")
                st.caption("👆 Tick one row to review and edit before adding, or tick several rows "
                           "(or the header box to select all) to add them all with one click.")
                show_cols = ["Customer Name", "Mobile Number", "Mobile Status", "Medication", "Medicine Type", "Purchases",
                             "Last Purchase", "Usual Gap (days)", "Estimated Due", "Why Not In Reminders"]
                sel = st.dataframe(cand_df[show_cols], width="stretch", hide_index=True, height=360,
                                   on_select="rerun", selection_mode="multi-row", key="review_candidates_table")
                sel_rows = list(getattr(getattr(sel, "selection", None), "rows", []) or [])
                if len(sel_rows) > 1:
                    picks = cand_df.iloc[sel_rows]
                    n_pick = len(picks)
                    with st.container(border=True):
                        st.markdown(f"##### ➕ Add {n_pick} selected patients to reminders")
                        st.caption("Each patient keeps their own usual refill gap and estimated due date "
                                   "(dates in the past move to today).")
                        n_uncl = int((picks["Medicine Type"] == "Not classified").sum())
                        n_nomob = int((picks["Mobile Status"] != "Valid").sum())
                        if n_uncl:
                            st.info(f"{n_uncl} of these medicines are not classified yet — add them only if they are regular (chronic) medicines.")
                        if n_nomob:
                            st.warning(f"{n_nomob} have no valid mobile number: scheduled, but not sent until the number is updated in billing.")
                        st.dataframe(picks[["Customer Name", "Mobile Number", "Medication", "Usual Gap (days)", "Estimated Due"]],
                                     width="stretch", hide_index=True, height=min(35 * n_pick + 38, 250))
                        bulk_note = st.text_input("Note for all (optional)", key="bulk_add_note",
                                                  placeholder="e.g. reviewed with store pharmacist")
                        if st.button(f"➕ Add {n_pick} patients to reminders", type="primary", key="btn_bulk_add"):
                            with st.spinner(f"Adding {n_pick} patients to reminders..."):
                                with SessionLocal() as add_db:
                                    res_bulk = add_many_to_reminders(add_db, [r["customer_item_key"] for r in picks["_raw"]],
                                                                     added_by="pharmacist", note=bulk_note)
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            msg = f"✅ {res_bulk['added']} patients added to reminders."
                            if res_bulk["already_in_reminders"]:
                                msg += f" {res_bulk['already_in_reminders']} were already getting reminders."
                            for f in res_bulk["failed"]:
                                st.error(f"Not added: {f['reason']}")
                            if res_bulk["added"] and not res_bulk["failed"]:
                                st.session_state["bulk_add_msg"] = msg
                                st.rerun()
                            st.success(msg)
                elif sel_rows:
                    pick = cand_df.iloc[sel_rows[0]]
                    raw = pick["_raw"]
                    with st.container(border=True):
                        st.markdown(f"##### ➕ Add to reminders: **{pick['Customer Name']}** — {pick['Medication']}")
                        st.caption(
                            f"📱 {pick['Mobile Number']} ({pick['Mobile Status']}) · {pick['Medicine Type']} · "
                            f"{pick['Purchases']} purchases · last bought {pick['Last Purchase']}"
                        )
                        if pick["Medicine Type"] == "Not classified":
                            st.info("This medicine is not in the medication rules yet. Add it only if it is a regular (chronic) medicine for this patient.")
                        if pick["Mobile Status"] != "Valid":
                            st.warning("No valid mobile number: the reminder is scheduled but cannot be sent until the number is updated in billing.")
                        est_due = datetime.strptime(raw["estimated_due_date"], "%Y-%m-%d").date()
                        ac1, ac2 = st.columns(2)
                        with ac1:
                            gap_days = st.number_input("Refill every (days)", min_value=MIN_INTERVAL, max_value=MAX_INTERVAL,
                                                       value=int(raw["usual_gap_days"]), step=1, key=f"add_gap_{raw['customer_item_key']}")
                        with ac2:
                            next_due = st.date_input("Next refill due date", value=max(today_rev, est_due), min_value=today_rev,
                                                     format="DD/MM/YYYY", key=f"add_due_{raw['customer_item_key']}")
                        add_note = st.text_input("Note (optional)", key=f"add_note_{raw['customer_item_key']}",
                                                 placeholder="e.g. patient confirmed monthly BP medicine")
                        if st.button("➕ Add to reminders", type="primary", key=f"btn_add_{raw['customer_item_key']}"):
                            with SessionLocal() as add_db:
                                try:
                                    res_add = add_to_reminders(add_db, raw["customer_item_key"], interval_days=int(gap_days),
                                                               next_refill_date=next_due, added_by="pharmacist", note=add_note)
                                except ValueError as e:
                                    res_add = None
                                    st.error(str(e))
                            if res_add:
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                if res_add.get("already_in_reminders"):
                                    st.info("This patient is already getting reminders.")
                                else:
                                    st.success(
                                        f"✅ {pick['Customer Name']} added to reminders for {pick['Medication']} "
                                        f"(next refill {next_due.strftime('%d-%m-%Y')}, every {int(gap_days)} days)."
                                    )
                                    st.rerun()

        else:
            # Existing review categories (patients already in reminders, or B2B audit)
            from refillcare.engine.review_lists import review_other

            cat_key = "b2b" if rev_cat.startswith("B2B") else "missing_mobile" if rev_cat.startswith("Missing") else "unstable"
            with SessionLocal() as oth_db:
                oth_rows = review_other(oth_db, cat_key, rev_date)
            other_df = pd.DataFrame([{
                "Customer Name": r["customer_name"],
                "Mobile Number": format_display_phone_10digits(r["mobile_number"]) if r["mobile_number"] else "-",
                "Mobile Status": r["mobile_status"],
                "Medication": r["medicine"],
                "Last Purchase Date": datetime.strptime(r["last_purchase_date"], "%Y-%m-%d").strftime("%d-%m-%Y") if r["last_purchase_date"] else "-",
                "Expected Refill Date": datetime.strptime(r["expected_refill_date"], "%Y-%m-%d").strftime("%d-%m-%Y") if r["expected_refill_date"] else "-",
                "Review Reason": r["review_reason"],
            } for r in oth_rows])
            other_df = _search_filter(other_df)
            if other_df.empty:
                st.info(f"No records for this category ({rev_label}).")
            else:
                st.markdown(f"#### {rev_cat} ({len(other_df):,}) · {rev_label}")
                other_cols = [c for c in ["Customer Name", "Mobile Number", "Mobile Status", "Medication", "Last Purchase Date",
                                          "Expected Refill Date", "Review Reason"] if c in other_df.columns]
                st.dataframe(other_df[other_cols], width="stretch", hide_index=True)
                buf = io.StringIO()
                other_df[other_cols].to_csv(buf, index=False)
                st.download_button("Export Review Records (CSV)", data=buf.getvalue().encode("utf-8"),
                                   file_name=f"patients_needing_review_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

        # Patients added by a pharmacist (with undo)
        with SessionLocal() as added_db:
            added_rows = list_pharmacist_added(added_db)
        with st.expander(f"Patients added to reminders by a pharmacist ({len(added_rows):,})"):
            if added_rows:
                st.dataframe(pd.DataFrame([{
                    "Patient": r["customer_name"] or "-", "Medicine": r["medicine"] or "-",
                    "Refill every (days)": r["interval_days"],
                    "Next refill": datetime.strptime(r["next_refill_date"], "%Y-%m-%d").strftime("%d-%m-%Y"),
                    "Added on": r["added_at"][:10], "Note": r["note"] or "-",
                } for r in added_rows]), width="stretch", hide_index=True)
                rm_key = st.selectbox("Remove a patient from reminders", [r["customer_item_key"] for r in added_rows],
                                      format_func=lambda k: next(f"{r['customer_name']} — {r['medicine']}" for r in added_rows if r["customer_item_key"] == k),
                                      key="review_remove_pick")
                if st.button("Remove from reminders", key="btn_review_remove"):
                    with SessionLocal() as rm_db:
                        remove_from_reminders(rm_db, rm_key, removed_by="pharmacist")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Removed. Their pending reminders were cancelled.")
                    st.rerun()
            else:
                st.caption("No patients added manually yet.")

    # --------------------------------------------------------------------------
    # TAB 5: WHATSAPP 3-TIER REFILL OUTREACH GATEWAY
    # --------------------------------------------------------------------------
    with tab_whatsapp:
        from refillcare.whatsapp import (
            RefillWhatsAppClient,
            RefillWhatsAppDispatcher,
            build_dynamic_tier_text,
            format_full_message_preview,
            get_tier_for_stage_offset,
        )
        from database.connection import SessionLocal
        from database.models import (
            ReminderCycleModel,
            ReminderStageModel,
            WhatsAppDeliveryLogModel,
            RefillDecisionModel,
        )

        st.markdown("### WhatsApp Messages & Outreach")
        st.write(
            "Execute and preview automated patient refill communications using a **single unified WhatsApp template** "
            "with dynamic content tailored to **Tier 1 (Due / Advance)**, **Tier 2 (Follow-up)**, and **Tier 3 (+45d Re-engagement)**."
        )

        # Configuration and customization
        active_store_name = st.session_state.get("wa_custom_store_name", "PHARMA HUBB")
        active_store_contact = st.session_state.get("wa_custom_store_contact", "+91 9966473474")

        with st.expander("Gateway Configuration & Pharmacy Profile (Click to change Store / Contact)", expanded=False):
            st.markdown("##### Pharmacy Profile & Dynamic Template Defaults")
            cfg_c1, cfg_c2 = st.columns(2)
            with cfg_c1:
                ui_store_name = st.text_input(
                    "Medical Store Name ({{2}}, {{6}})",
                    value=active_store_name,
                    key="wa_ui_store_name",
                    help="Store name filled into {{2}} and {{6}} in WhatsApp template.",
                )
                if ui_store_name != active_store_name:
                    st.session_state["wa_custom_store_name"] = ui_store_name
                    active_store_name = ui_store_name
            with cfg_c2:
                ui_store_contact = st.text_input(
                    "Pharmacy Contact Number ({{5}})",
                    value=active_store_contact,
                    key="wa_ui_store_contact",
                    help="Contact phone number filled into {{5}} in WhatsApp template.",
                )
                if ui_store_contact != active_store_contact:
                    st.session_state["wa_custom_store_contact"] = ui_store_contact
                    active_store_contact = ui_store_contact

            temp_client = RefillWhatsAppClient(store_name=active_store_name, store_contact=active_store_contact)
            wa_diag = temp_client.get_diagnostic_info()

            st.markdown("---")
            st.markdown("##### Xinno CPaaS Connection Diagnostics")
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                st.markdown(f"**WABA Number:** `{wa_diag['waba_number']}`")
                st.markdown(f"**Language:** `{wa_diag['template_language']}`")
            with wc2:
                st.markdown(f"**Active Template:** `{wa_diag['template_name']}`")
                st.markdown(f"**Active Store:** `{active_store_name}`")
            with wc3:
                api_stat_str = "Configured (Active)" if wa_diag["api_key_configured"] else "Missing Key"
                st.markdown(f"**API Key Status:** {api_stat_str}")
                st.markdown(f"**Active Contact:** `{active_store_contact}`")
            with wc4:
                st.markdown(f"**API Endpoint:** `{wa_diag['api_url']}`")
                st.markdown(f"**Parameter Count:** `{wa_diag['param_count']} Variables`")

        wa_client = RefillWhatsAppClient(
            store_name=active_store_name,
            store_contact=active_store_contact,
        )
        wa_diag = wa_client.get_diagnostic_info()

        # Row 1: Target Controls & Filters
        st.markdown("#### Outreach Target & Lifecycle Stage Selection")
        avail_months = get_available_reminder_months()
        today_val = date.today()
        today_m = today_val.strftime("%Y-%m")
        default_m_idx = avail_months.index(today_m) if today_m in avail_months else (len(avail_months) - 1 if avail_months else 0)

        wcol1, wcol2, wcol3, wcol4 = st.columns([1.5, 1.8, 1.4, 1.3])

        with wcol1:
            wa_filter_mode = st.selectbox(
                "Target Selection Mode",
                ["Specific Date", "Complete Month"],
                key="wa_target_filter_mode",
            )

        with wcol2:
            if wa_filter_mode == "Specific Date":
                wa_sel_date = st.date_input(
                    "Select Target Send Date",
                    value=today_val,
                    key="wa_sel_date",
                )
                wa_sel_month = None
                wa_target_label = wa_sel_date.strftime("%d %B %Y")
            else:
                wa_sel_month = st.selectbox(
                    "Select Target Prediction Month",
                    options=avail_months,
                    index=default_m_idx,
                    key="wa_sel_month",
                )
                wa_sel_date = None
                wa_target_label = wa_sel_month

        with wcol3:
            wa_tier_filter = st.selectbox(
                "Lifecycle Tier",
                ["ALL", "DUE (Due / Advance)", "FOLLOWUP (Follow-up)", "LAPSED (+45d Re-engagement)"],
                key="wa_tier_filter",
            )
            raw_tier = wa_tier_filter.split(" ")[0]

        with wcol4:
            wa_sync_win = st.slider(
                "Sync Window (Days)",
                min_value=0,
                max_value=14,
                value=0,
                step=1,
                key="wa_medsync_window_slider",
                help="0 = Exact Same-Date Group-By. 1-14 = Multi-Day Window Sync.",
            )

        # Load candidates from database using Med-Sync Clustering
        candidate_records = []
        with SessionLocal() as db:
            from refillcare.engine.med_sync import MedSyncEngine
            engine = MedSyncEngine(sync_window_days=wa_sync_win)

            # Load candidates for bundling from persistent queue / decisions
            if wa_filter_mode == "Specific Date" and wa_sel_date:
                medsync_input = load_v1_reminder_queue_df(target_date=wa_sel_date)
            elif wa_sel_month and wa_sel_month != "ALL":
                medsync_input = load_v1_reminder_queue_df(target_month=wa_sel_month)
            else:
                medsync_input = pd.DataFrame()

            is_from_queue = not medsync_input.empty
            if medsync_input.empty:
                medsync_input = load_active_refill_decisions_for_medsync(eligible_df)

            all_bundles = engine.cluster_patient_decisions(medsync_input, sync_window_days=wa_sync_win)

            # Filter bundles by target date / month if not already pre-filtered by queue
            if is_from_queue:
                filtered_b = all_bundles
            else:
                if wa_filter_mode == "Specific Date" and wa_sel_date:
                    filtered_b = [b for b in all_bundles if b.anchor_refill_date == wa_sel_date]
                elif wa_sel_month and wa_sel_month != "ALL":
                    try:
                        y_val, m_val = map(int, wa_sel_month.split("-"))
                        filtered_b = [b for b in all_bundles if b.anchor_refill_date.year == y_val and b.anchor_refill_date.month == m_val]
                    except Exception:
                        filtered_b = all_bundles
                else:
                    filtered_b = all_bundles

            # Filter by tier / refill function
            if raw_tier == "DUE":
                filtered_b = [b for b in filtered_b if getattr(b, "lifecycle_tier", "DUE") == "DUE" or getattr(b, "refill_function", "DUE_REFILL") == "DUE_REFILL"]
            elif raw_tier == "FOLLOWUP":
                filtered_b = [b for b in filtered_b if getattr(b, "lifecycle_tier", "DUE") == "FOLLOWUP" or getattr(b, "refill_function", "DUE_REFILL") == "REFILL_FOLLOW_UP"]
            elif raw_tier == "LAPSED":
                filtered_b = [b for b in filtered_b if getattr(b, "lifecycle_tier", "DUE") == "LAPSED" or getattr(b, "refill_function", "DUE_REFILL") == "LAPSED_REENGAGEMENT"]

            from refillcare.whatsapp.consent import load_opt_out_index

            with SessionLocal() as consent_db:
                wa_opt_outs = load_opt_out_index(consent_db)
            for b in filtered_b:
                phone_val = b.mobile_no or ""
                if wa_opt_outs.blocks(getattr(b, "customer_id", None), phone_val):
                    continue  # patient opted out of WhatsApp (same rule as API / dispatcher)
                meds_list = [item.item_name for item in b.synced_items] if b.synced_items else [b.anchor_item_name]
                mob_stat = determine_mobile_status(phone_val)
                tier_val = getattr(b, "lifecycle_tier", "DUE")

                candidate_records.append({
                    "reminder_id": b.bundle_id,
                    "customer_name": b.customer_name or "Valued Customer",
                    "phone_number": phone_val,
                    "mobile_display": format_display_phone_10digits(phone_val),
                    "mobile_status": mob_stat,
                    "medication": ", ".join(meds_list),
                    "expected_refill_date": b.anchor_refill_date,
                    "target_send_date": wa_sel_date if wa_sel_date else b.anchor_refill_date,
                    "stage_offset": 0 if tier_val == "DUE" else (5 if tier_val == "FOLLOWUP" else 45),
                    "tier": tier_val,
                    "refill_function": getattr(b, "refill_function", "DUE_REFILL"),
                    "status": "PENDING",
                    "bundle_obj": b,
                })

        valid_count = sum(1 for r in candidate_records if r["mobile_status"] == "Valid")
        missing_count = len(candidate_records) - valid_count

        # KPI Summary
        wk1, wk2, wk3, wk4 = st.columns(4)
        with wk1:
            st.metric("Total Scheduled", f"{len(candidate_records):,}")
        with wk2:
            st.metric("Delivery Ready (Valid)", f"{valid_count:,}")
        with wk3:
            st.metric("Missing Mobile", f"{missing_count:,}")
        with wk4:
            st.metric("Active Target", wa_target_label)

        st.markdown("---")

        # Two-Column Layout: Left = Candidate Queue, Right = Live Dynamic Preview & Testing
        col_queue, col_preview = st.columns([1.4, 1.6])

        with col_queue:
            st.markdown(f"#### Candidate Outreach Queue ({len(candidate_records):,} records)")
            if candidate_records:
                df_candidates = pd.DataFrame([
                    {
                        "Customer": r["customer_name"],
                        "Mobile": r["mobile_display"],
                        "Status": r["mobile_status"],
                        "Medication": r["medication"],
                        "Stage": "DUE" if r["tier"] == "DUE" else ("FOLLOW" if r["tier"] == "FOLLOWUP" else "LAPSED"),
                        "Refill Due": r["expected_refill_date"].strftime("%d-%m-%Y") if isinstance(r["expected_refill_date"], date) else str(r["expected_refill_date"]),
                    }
                    for r in candidate_records
                ])
                st.dataframe(df_candidates, width="stretch", hide_index=True, height=360)
            else:
                st.info("No reminder stages match the selected date/month and tier filter.")

        with col_preview:
            st.markdown("#### Clinical Template Simulation & Variable Preview")
            
            # Select customer for preview
            sample_cust = candidate_records[0] if candidate_records else {
                "customer_name": "ANIL KUMAR K",
                "phone_number": "919640568227",
                "medication": "ATCHOL F",
                "expected_refill_date": date.today(),
                "stage_offset": 0,
                "tier": "DUE",
            }

            cust_names = [f"{r['customer_name']} — {r['medication']} ({r['tier']})" for r in candidate_records] if candidate_records else [f"{sample_cust['customer_name']} — {sample_cust['medication']} (Sample)"]
            selected_sample_idx = st.selectbox("Select Patient to Preview WhatsApp Text", range(len(cust_names)), format_func=lambda i: cust_names[i], key="wa_preview_sample_idx")
            
            active_preview_record = candidate_records[selected_sample_idx] if candidate_records else sample_cust

            # Generate dynamic preview (exact approved template text, rendered as WhatsApp shows it)
            preview_res = wa_client.preview_message(
                customer_name=active_preview_record["customer_name"],
                medications=active_preview_record["medication"],
                expected_refill_date=active_preview_record["expected_refill_date"],
                tier=active_preview_record["tier"],
                stage_offset=active_preview_record["stage_offset"],
            )
            from refillcare.whatsapp.template_formatter import whatsapp_text_to_html

            bubble_html = whatsapp_text_to_html(preview_res["visual_preview"]).replace("\n", "<br>")
            badge_txt = f"{active_preview_record['tier']} • {active_preview_record.get('refill_function', 'DUE_REFILL')}"
            # One line of HTML (no blank lines) so Markdown never re-parses the message text
            st.markdown(
                "<div style=\"background:#efeae2;border:1px solid #d1d5db;border-radius:14px;padding:0.9rem 1rem 1rem;margin-bottom:1rem;\">"
                "<div style=\"display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:0.7rem;\">"
                "<span style=\"font-weight:700;color:#075e54;font-size:0.95rem;\">💬 WhatsApp message preview</span>"
                f"<span style=\"background:#16a34a;color:#ffffff;font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:9999px;\">{badge_txt}</span>"
                "</div>"
                "<div style=\"background:#d9fdd3;border-radius:0 12px 12px 12px;padding:0.75rem 0.95rem 0.45rem;box-shadow:0 1px 1px rgba(0,0,0,0.13);"
                "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:0.95rem;line-height:1.5;color:#111b21;\">"
                f"{bubble_html}"
                f"<div style=\"text-align:right;font-size:0.7rem;color:#667781;margin-top:0.35rem;\">Template: {wa_client.template_name}</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Section: Send today's WhatsApp reminders (live). The pharmacist confirms in a dialog;
        # "Yes" approves the reminders and sends them all. Opt-outs, invalid mobiles, stale sales
        # data and double sends are still blocked by the dispatcher.
        from refillcare.whatsapp.dispatcher import StaleSalesData, check_live_send_allowed

        send_bundles = [r["bundle_obj"] for r in candidate_records if "bundle_obj" in r]
        is_today_target = wa_filter_mode == "Specific Date" and wa_sel_date == today_val
        with SessionLocal() as db_sum:
            send_summary = RefillWhatsAppDispatcher(client=wa_client).summarize_medsync_send(db_sum, send_bundles)
        n_to_send = send_summary["to_send"]
        n_skipped = send_summary["no_valid_mobile"] + send_summary["opted_out"] + send_summary["already_sent"]
        n_holdout = send_summary.get("holdout", 0)

        st.markdown(
            """
<style>
.st-key-wa_send_cta button {
    background: linear-gradient(135deg, #22c55e 0%, #15803d 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 14px !important;
    min-height: 3.6rem; font-size: 1.15rem !important; font-weight: 800 !important;
    box-shadow: 0 6px 18px rgba(22, 163, 74, 0.35) !important; transition: transform .12s ease, box-shadow .12s ease;
}
.st-key-wa_send_cta button p { font-size: 1.15rem !important; font-weight: 800 !important; }
.st-key-wa_send_cta button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(22, 163, 74, 0.45) !important; }
.st-key-wa_send_cta button:disabled { background: #334155 !important; box-shadow: none !important; color: #94a3b8 !important; }
</style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style=\"background:linear-gradient(135deg,#052e16 0%,#14532d 100%);border:1px solid #16a34a;border-radius:16px;padding:1.1rem 1.4rem;margin-bottom:0.8rem;\">"
            "<div style=\"color:#bbf7d0;font-size:0.85rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;\">📤 Send WhatsApp reminders</div>"
            f"<div style=\"color:#ffffff;font-size:2.1rem;font-weight:800;line-height:1.2;margin-top:0.2rem;\">{n_to_send:,} <span style=\"font-size:1rem;font-weight:600;color:#dcfce7;\">messages ready for {wa_target_label}</span></div>"
            f"<div style=\"color:#bbf7d0;font-size:0.85rem;margin-top:0.3rem;\">{send_summary['medicines']:,} medicines covered · {n_skipped:,} skipped (no valid mobile, opted out or already sent) · {n_holdout:,} in the no-reminder control group</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        @st.dialog("Send WhatsApp reminders", width="large")
        def _confirm_send_dialog(bundles, summary, target_label):
            apply_store_from_request()  # dialog reruns skip the top of the script: keep the same store
            last = st.session_state.get("wa_last_send")
            if last:
                if last["failed"] == 0:
                    st.success(f"✅ **{last['sent']:,} WhatsApp messages sent** for {target_label}.")
                else:
                    st.warning(f"**{last['sent']:,} sent, {last['failed']:,} failed** for {target_label}. Failed messages are listed below and can be sent again later.")
                if last["skipped"]:
                    st.caption(f"{last['skipped']:,} skipped (already sent or not ready).")
                if last["failed_rows"]:
                    st.dataframe(pd.DataFrame(last["failed_rows"]), width="stretch", hide_index=True)
                if st.button("Done", type="primary", width="stretch", key="wa_dlg_done"):
                    st.session_state.pop("wa_last_send", None)
                    st.rerun()
                return

            st.markdown(f"You are about to send **{summary['to_send']:,} WhatsApp messages** to patients for **{target_label}**.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Messages to send", f"{summary['to_send']:,}")
            m2.metric("Medicines covered", f"{summary['medicines']:,}")
            m3.metric("Skipped", f"{summary['no_valid_mobile'] + summary['opted_out'] + summary['already_sent'] + summary.get('holdout', 0):,}")
            st.caption(
                f"Skipped: {summary['no_valid_mobile']:,} without a valid mobile number · "
                f"{summary['opted_out']:,} opted out (STOP) · {summary['already_sent']:,} already sent · "
                f"{summary.get('holdout', 0):,} in the control group (not messaged, so we can measure how many extra refills reminders bring)."
            )

            if not (wa_client.api_key and wa_client.waba_number):
                st.error("WhatsApp sending is not set up (Xinno API key or WABA number missing). Please contact your administrator.")
                return

            stale = False
            override_ok = False
            override_reason = ""
            with SessionLocal() as db_chk:
                try:
                    check_live_send_allowed(db_chk)
                except StaleSalesData as e:
                    stale = True
                    st.warning(f"⚠️ {e}")
                    override_cb = st.checkbox("Send anyway - today's sales are already uploaded", key="wa_dlg_override")
                    override_reason = st.text_input("Reason (saved in the log)", key="wa_dlg_reason")
                    override_ok = bool(override_cb and override_reason.strip())

            b_yes, b_no = st.columns(2)
            with b_no:
                if st.button("No, cancel", width="stretch", key="wa_dlg_cancel"):
                    st.rerun()
            with b_yes:
                clicked_yes = st.button(
                    f"Yes, send {summary['to_send']:,} messages",
                    type="primary",
                    width="stretch",
                    key="wa_dlg_yes",
                    disabled=summary["to_send"] == 0 or (stale and not override_ok),
                )
            if clicked_yes:
                if override_ok:
                    logging.getLogger("refillcare.streamlit").warning("Stale-data send override: %s", override_reason)
                progress = st.progress(0.0, text="Sending WhatsApp messages...")

                def _on_progress(done, total):
                    progress.progress(done / max(total, 1), text=f"Sending... {done:,} of {total:,}")

                dispatcher = RefillWhatsAppDispatcher(client=wa_client, allow_stale_data=override_ok)
                with SessionLocal() as db_send:
                    try:
                        outcome = dispatcher.dispatch_medsync_batch(
                            db=db_send,
                            bundles=bundles,
                            dry_run=False,
                            batch_limit=None,
                            approve_before_send=True,
                            approved_by="streamlit-pharmacist",
                            on_progress=_on_progress,
                        )
                    except StaleSalesData as e:
                        st.error(str(e))
                        return
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state["wa_last_send"] = {
                    "sent": outcome["success_count"],
                    "failed": outcome["failed_count"],
                    "skipped": outcome.get("skipped_not_approved_or_already_sent", 0),
                    "failed_rows": [
                        {"Customer": r["customer_name"], "Mobile": r["phone_number"], "Medicines": r["medication_name"], "Reason": r["message"]}
                        for r in outcome.get("delivery_logs", []) if r.get("status") == "FAILED"
                    ],
                }
                st.rerun(scope="fragment")

        with st.container(key="wa_send_cta"):
            btn_open_send = st.button(
                f"📤  Send WhatsApp Messages ({n_to_send:,})",
                key="btn_open_wa_send",
                type="primary",
                width="stretch",
                disabled=(not is_today_target) or n_to_send == 0,
            )
        if not is_today_target:
            st.info("To send, choose **Specific Date** with **today's date** above. Messages are sent only for today's reminders.")
        elif n_to_send == 0:
            st.info("No messages to send for today - every reminder is already sent, opted out or missing a mobile number.")
        if btn_open_send:
            st.session_state.pop("wa_last_send", None)
            _confirm_send_dialog(send_bundles, send_summary, wa_target_label)

        # Section: Patient WhatsApp messages on / off (pharmacist searches by name or mobile)
        from refillcare.whatsapp.consent import list_opted_out
        from refillcare.whatsapp.pharmacist_views import delivery_log_rows, search_patients, set_messages_allowed

        st.markdown("---")
        st.markdown("#### 🔕 Stop or Resume WhatsApp Messages for a Patient")
        st.caption(
            "Use this when a patient asks at the counter to stop (or restart) reminders. "
            "Patients who reply STOP on WhatsApp are stopped automatically. "
            "Stopping works on the mobile number, so family members sharing that number are also stopped."
        )
        consent_q = st.text_input(
            "Search patient by name or mobile number",
            key="wa_consent_search",
            placeholder="e.g. KRISHNA RAO or 9866078851",
        )
        if consent_q.strip():
            with SessionLocal() as consent_db:
                found = search_patients(consent_db, consent_q)
            if not found:
                st.info("No patient found. Check the spelling or type at least 4 digits of the mobile number.")
            for i, pt in enumerate(found):
                pc1, pc2, pc3 = st.columns([2.6, 1.4, 1.2])
                with pc1:
                    st.markdown(
                        f"**{pt['customer_name']}**  \n"
                        f"📱 {format_display_phone_10digits(pt['mobile_number']) if pt['mobile_number'] else 'No mobile number'}"
                    )
                with pc2:
                    if pt["messages_allowed"]:
                        st.markdown("<span style='color:#16a34a;font-weight:700;'>✅ Receiving messages</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#dc2626;font-weight:700;'>🔕 Messages stopped</span>", unsafe_allow_html=True)
                with pc3:
                    if pt["messages_allowed"]:
                        do_change = st.button("Stop messages", key=f"wa_stop_{i}_{pt['customer_id']}", width="stretch")
                    else:
                        do_change = st.button("Resume messages", key=f"wa_resume_{i}_{pt['customer_id']}", type="primary", width="stretch")
                if do_change:
                    with SessionLocal() as consent_db:
                        set_messages_allowed(
                            consent_db,
                            allowed=not pt["messages_allowed"],
                            mobile_number=pt["mobile_number"] or None,
                            customer_id=pt["customer_id"],
                        )
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success(
                        f"WhatsApp messages {'resumed' if not pt['messages_allowed'] else 'stopped'} for **{pt['customer_name']}**."
                    )
                    st.rerun()

        with SessionLocal() as consent_db:
            opted_rows = list_opted_out(consent_db)
        with st.expander(f"Patients who stopped messages ({len(opted_rows):,})"):
            if opted_rows:
                st.dataframe(
                    pd.DataFrame([{
                        "Patient": r["customer_name"] or "-",
                        "Mobile": format_display_phone_10digits(r["mobile_number"]) if r["mobile_number"] else "-",
                        "Stopped on": (r["updated_at"] or "-").replace("T", " ")[:16],
                    } for r in opted_rows]),
                    width="stretch", hide_index=True,
                )
            else:
                st.info("No patients have stopped messages.")

        # Section: Messages sent (live only, India time, plain columns)
        st.markdown("---")
        st.markdown("#### 📋 Messages Sent")
        lc1, lc2 = st.columns([1.2, 1.8])
        with lc1:
            log_day = st.date_input("Day", value=today_val, key="wa_log_day", format="DD/MM/YYYY")
        with lc2:
            log_status = st.radio("Show", ["All", "Sent", "Failed"], horizontal=True, key="wa_log_status")
        with SessionLocal() as log_db:
            log_view = delivery_log_rows(log_db, day=log_day, status=log_status)
        lk1, lk2 = st.columns(2)
        lk1.metric("✅ Sent", f"{log_view['sent']:,}")
        lk2.metric("❌ Failed", f"{log_view['failed']:,}")
        if log_view["rows"]:
            st.dataframe(
                pd.DataFrame([{
                    "Time": r["time"],
                    "Patient": r["patient"],
                    "Mobile": format_display_phone_10digits(r["mobile"]),
                    "Status": ("✅ " if r["status"] == "Sent" else "❌ ") + r["status"],
                    "Reason": r["reason"] or "-",
                } for r in log_view["rows"]]),
                width="stretch", hide_index=True, height=320,
            )
            if log_view["failed"]:
                st.caption("Failed messages are not marked as sent - they can be sent again with the Send button.")
        else:
            st.info(f"No WhatsApp messages were sent on {log_day.strftime('%d %B %Y')}.")

    # --------------------------------------------------------------------------
    # TAB 8: ML MODELS (registry, dynamic windows, evaluations, promotion gate)
    # Same domain layer as FastAPI /api/v1/ml/* - Streamlit never calls the API.
    # --------------------------------------------------------------------------
    for _ in _visible_tab(tab_ml):  # skipped when the tab is hidden (TAB_VISIBILITY)
        from database.connection import SessionLocal
        from database.models import MLEvaluationRunModel
        from refillcare.ml import registry as ml_reg
        from refillcare.ml.evaluation import BASELINE_METHOD, engine_method_name
        from refillcare.ml.service import data_freshness, gate_for_version, promote, retrain_cycle

        st.markdown("### AI Accuracy & Prediction Settings")
        st.caption(
            "Train / validation / test windows are re-derived from the latest sale every time this tab "
            "loads, so they move forward automatically when a new month of sales is uploaded."
        )

        with SessionLocal() as ml_db:
            fresh = data_freshness(ml_db, history_data if history_data is not None and not history_data.empty else None)
            plan = fresh.get("plan")

            # --- Data freshness & windows
            if not plan:
                st.warning(fresh.get("reason", "Not enough sales history to plan evaluation windows."))
            else:
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    st.metric("Serving Model", fresh.get("production_version") or "none")
                with f2:
                    st.metric("Trained Up To", str(fresh.get("production_train_cutoff") or "unknown"))
                with f3:
                    st.metric("Sales Data Up To", str(plan["data_end"]))
                with f4:
                    lag = fresh.get("days_of_new_training_data")
                    st.metric("Newer Training Data", f"{lag} days" if lag is not None else "-")
                if fresh.get("retrain_recommended"):
                    st.warning("**Retrain recommended:** newer sales data is available or the serving model is not validated.")
                else:
                    st.success("Serving model is current.")

                st.dataframe(pd.DataFrame([
                    {"Window": "Train", "From": str(plan["train"]["start"]), "To": str(plan["train"]["end"]),
                     "Purchases": f"{plan['train']['purchases']:,}", "Used For": "Fit the candidate model"},
                    {"Window": "Validation", "From": str(plan["validation"]["start"]), "To": str(plan["validation"]["end"]),
                     "Purchases": f"{plan['validation']['purchases']:,}", "Used For": "Pick the best model"},
                    {"Window": "Test", "From": str(plan["test"]["start"]), "To": str(plan["test"]["end"]),
                     "Purchases": f"{plan['test']['purchases']:,}", "Used For": "Promotion gate only (never used to choose)"},
                    {"Window": "Not scored", "From": f"last {plan['censor_buffer_days']} days", "To": str(plan["data_end"]),
                     "Purchases": "-", "Used For": "Refills not yet observable"},
                ]), width="stretch", hide_index=True)

            # --- Registry
            st.markdown("#### Model Version Registry")
            versions = ml_reg.list_versions(ml_db)
            reg_rows = []
            for v in versions:
                summ = ml_reg.to_dict(v, ml_db).get("evaluation_summary") or {}
                met = (summ.get("methods") or {}).get(engine_method_name(v.version_id)) or {}
                reg_rows.append({
                    "Version": v.version_id,
                    "Type": v.model_type,
                    "Status": v.status + (" - Validated" if v.is_validated else ""),
                    "Train Cutoff": str(v.train_window_end or "-"),
                    "Train Samples": f"{v.training_sample_count:,}" if v.training_sample_count else "-",
                    "Within ±7d (measured)": f"{met['within_7_days_pct']:.1f}%" if met.get("within_7_days_pct") is not None else "-",
                    "MAE (days)": f"{float(met['mae_days']):.2f}" if (met.get("mae_days") is not None and met.get("mae_days") != "-") else "-",
                })
            st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)

            # --- Promotion gate & promote
            version_ids = [v.version_id for v in versions]
            g1, g2, g3 = st.columns([2.5, 1, 1])
            with g1:
                sel_version = st.selectbox("Model version", version_ids, key="ml_gate_version") if version_ids else None
            with g2:
                st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
                btn_gate = st.button("Check Gate", key="btn_ml_gate", width="stretch")
            with g3:
                st.markdown("<div style='padding-top: 1.75rem;'></div>", unsafe_allow_html=True)
                btn_promote = st.button("Promote", key="btn_ml_promote", width="stretch")

            if sel_version and (btn_gate or btn_promote):
                gate = gate_for_version(ml_db, sel_version)
                if btn_promote:
                    res = promote(ml_db, sel_version, promoted_by="streamlit")
                    if res.get("promoted"):
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success(f"`{sel_version}` is now the production model.")
                    else:
                        st.error(f"Promotion refused by the evidence gate: {', '.join(res['decision'].get('failed_checks', []))}")
                    gate = res.get("decision", gate)
                st.markdown(f"**Gate for `{sel_version}`:** {'APPROVED' if gate.get('approved') else 'REJECTED'}")
                st.dataframe(pd.DataFrame([
                    {"Passed": "Passed" if c.get("passed") else "Failed", "Check": c.get("check"), "Detail": c.get("detail", "")}
                    for c in gate.get("checks", [])
                ]), width="stretch", hide_index=True)

            if st.button("Roll back to previous validated model", key="btn_ml_rollback"):
                try:
                    v = ml_reg.rollback_production(ml_db, rolled_back_by="streamlit")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success(f"Production model is now `{v.version_id}`.")
                except ValueError as e:
                    st.error(str(e))

            # --- Retrain cycle
            st.markdown("#### Retrain on the Current Windows")
            st.caption(
                "Trains on the train window, selects the best model on validation, then applies the promotion "
                "gate on the untouched test window. Needs ~2-3 GB free RAM and can take several minutes."
            )
            r1, r2 = st.columns(2)
            with r1:
                btn_retrain = st.button("Retrain & Evaluate", key="btn_ml_retrain", width="stretch")
            with r2:
                btn_retrain_promote = st.button("Retrain & Promote if Gate Approves", key="btn_ml_retrain_promote", width="stretch")
            if btn_retrain or btn_retrain_promote:
                with st.spinner("Running retrain cycle (train → validation → test gate)..."):
                    try:
                        cycle = retrain_cycle(ml_db, promote_best=bool(btn_retrain_promote), created_by="streamlit")
                        st.session_state["ml_last_cycle"] = cycle
                        st.cache_data.clear()
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Retrain cycle failed: {e}")
            cycle = st.session_state.get("ml_last_cycle")
            if cycle:
                st.markdown(
                    f"**Last cycle:** candidate `{cycle['candidate_version']}` · selected on validation "
                    f"`{cycle['selected_on_validation']}` · gate {'approved' if cycle['gate'].get('approved') else 'rejected'}"
                    f" · {'promoted' if cycle.get('promoted') else 'not promoted'}"
                )
                st.dataframe(pd.DataFrame([
                    {"Version": vid,
                     "Validation ±7d": cycle["validation"]["versions"][vid].get("within_7_days_pct"),
                     "Test ±7d": cycle["test"]["versions"][vid].get("within_7_days_pct"),
                     "Test MAE (days)": f"{float(cycle['test']['versions'][vid].get('mae_days')):.2f}" if cycle["test"]["versions"][vid].get("mae_days") is not None else "-"}
                    for vid in cycle["test"]["versions"]
                ]), width="stretch", hide_index=True)

            # --- Recent evaluations
            st.markdown("#### Recent Out-of-Time Evaluations")
            runs = ml_db.query(MLEvaluationRunModel).order_by(MLEvaluationRunModel.started_at.desc()).limit(10).all()
            if runs:
                ev_rows = []
                for r in runs:
                    summ = ml_reg.summarize_report_for_version(r.report, r.model_version or "") if r.report else {}
                    methods = (summ or {}).get("methods") or {}
                    cand = methods.get(engine_method_name(r.model_version or "")) or {}
                    base = methods.get(BASELINE_METHOD) or {}
                    win = (summ or {}).get("evaluated_window") or {}
                    ev_rows.append({
                        "Run": r.run_id,
                        "Status": r.status,
                        "Window": f"{win.get('start')} → {win.get('end')}" if win else "-",
                        "Paired Events": (summ or {}).get("n_paired_events") or "-",
                        "Candidate": r.model_version or "-",
                        "Candidate ±7d": cand.get("within_7_days_pct", "-"),
                        "Baseline ±7d": base.get("within_7_days_pct", "-"),
                    })
                st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
            else:
                st.info("No evaluations yet. Use **Retrain & Evaluate** above.")

        render_ml_settings_and_tests()


def render_ml_settings_and_tests() -> None:
    """Prediction settings, rule A/B, multi-month test and identity cut-over (same services as FastAPI)."""
    from database.connection import SessionLocal
    from refillcare.ml.service import compare_rules, latest_report, prediction_settings, rolling_evaluation, save_report

    st.markdown("#### Prediction Settings")
    st.caption("Set in `.env` and restart the app. Measure a rule with the A/B test below before switching it on.")
    cfg = prediction_settings()
    sw = cfg["env_switches"]
    r = cfg["rules"]
    st.dataframe(pd.DataFrame([
        {"Setting": "Short-history mode (Path B)", "Value": r["path_b_mode"], ".env switch": sw["path_b_mode"]},
        {"Setting": "Device cap (insulin / inhaler)", "Value": r["device_cap_mode"], ".env switch": sw["device_cap_mode"]},
        {"Setting": "High-stability -> personal median", "Value": "on" if r["high_stability_median"] else "off", ".env switch": sw["high_stability_median"]},
        {"Setting": "Insulin -> personal median", "Value": "on" if r["insulin_median"] else "off", ".env switch": sw["insulin_median"]},
        {"Setting": "Day-of-month anchor", "Value": "on" if r["dom_anchor"] else "off", ".env switch": sw["dom_anchor"]},
        {"Setting": "Patient identity", "Value": cfg["patient_identity"], ".env switch": sw["patient_identity"]},
        {"Setting": "Therapy grouping", "Value": cfg["item_grouping"], ".env switch": sw["item_grouping"]},
        {"Setting": "Item / salt catalogue", "Value": f"{cfg['salt_catalog']['file']} ({'found' if cfg['salt_catalog']['exists'] else 'MISSING'})", ".env switch": "REFILLCARE_SALT_CATALOG"},
        {"Setting": "Store ID (mapping profiles)", "Value": cfg["store_id"], ".env switch": "REFILLCARE_STORE_ID"},
        {"Setting": "Sales columns stored", "Value": ", ".join(cfg["stored_sales_columns"]), ".env switch": "-"},
    ]), width="stretch", hide_index=True)

    st.markdown("#### Rule A/B and Multi-Month Test")
    st.caption(
        "Rule A/B scores every rule on the same unseen test month (~3 min). The multi-month test trains one "
        "model per month and checks the gain holds every month (~20 min, needs free RAM). Nothing is promoted."
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Run Rule A/B Evaluation", key="btn_ml_rules_ab", width="stretch"):
            with st.spinner("Scoring every rule on the test month..."):
                try:
                    with SessionLocal() as db:
                        save_report("rules_ab", compare_rules(db))
                    st.cache_data.clear()
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Rule A/B failed: {e}")
    with b2:
        if st.button("Run 3-Month Test", key="btn_ml_rolling", width="stretch"):
            with st.spinner("Training one model per test month and scoring it (several minutes)..."):
                try:
                    with SessionLocal() as db:
                        save_report("rolling", rolling_evaluation(db, created_by="streamlit"))
                    st.cache_data.clear()
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Multi-month test failed: {e}")

    ab = latest_report("rules_ab")
    if ab:
        rep_ab = ab["report"]
        st.markdown(
            f"**Rule A/B** of `{rep_ab.get('model_version')}` on {rep_ab.get('window', {}).get('start')} → "
            f"{rep_ab.get('window', {}).get('end')} ({rep_ab.get('n_paired_events')} refills) · run {ab['created_at'][:16]}"
        )
        st.dataframe(pd.DataFrame([
            {"Rule variant": k.replace("rules:", ""),
             "Within ±7d %": f"{float(v.get('within_7_days_pct')):.1f}%" if v.get("within_7_days_pct") is not None else "-",
             "Gain vs no rules (pp)": f"{float(v.get('within_7_gain_pp_vs_none')):+.2f}" if v.get("within_7_gain_pp_vs_none") is not None else "-",
             "MAE (days)": f"{float(v.get('mae_days')):.2f}" if v.get("mae_days") is not None else "-",
             "Early %": f"{float(v.get('early_pct')):.1f}%" if v.get("early_pct") is not None else "-",
             "Late %": f"{float(v.get('late_pct')):.1f}%" if v.get("late_pct") is not None else "-"}
            for k, v in (rep_ab.get("variants") or {}).items()
        ]), width="stretch", hide_index=True)
    ro = latest_report("rolling")
    if ro:
        rep_ro = ro["report"]
        st.markdown(
            f"**Multi-month test:** {rep_ro.get('n_folds')} months, {rep_ro.get('n_paired_events_total')} refills · "
            f"Model beats baseline every month: {'Yes' if rep_ro.get('model_beats_baseline_every_fold') else 'No'} · "
            f"Rules help every month: {'Yes' if rep_ro.get('rules_help_every_fold') else 'No'} · run {ro['created_at'][:16]}"
        )
        rows = []
        for f in rep_ro.get("folds", []):
            m = f.get("methods", {})
            rows.append({
                "Test month": f"{f['window']['start']} → {f['window']['end']}",
                "Refills": f"{f.get('n_paired_events', 0):,}",
                "Raw median ±7d": f"{float(m.get('raw_median', {}).get('within_7_days_pct')):.1f}%" if m.get("raw_median", {}).get("within_7_days_pct") is not None else "-",
                "Baseline engine ±7d": f"{float(m.get('baseline_engine', {}).get('within_7_days_pct')):.1f}%" if m.get("baseline_engine", {}).get("within_7_days_pct") is not None else "-",
                "Model ±7d": f"{float(m.get('model_no_rules', {}).get('within_7_days_pct')):.1f}%" if m.get("model_no_rules", {}).get("within_7_days_pct") is not None else "-",
                "Model + rules ±7d": f"{float(m.get('model_current_rules', {}).get('within_7_days_pct')):.1f}%" if m.get("model_current_rules", {}).get("within_7_days_pct") is not None else "-",
                "Model + rules MAE": f"{float(m.get('model_current_rules', {}).get('mae_days')):.2f}" if m.get("model_current_rules", {}).get("mae_days") is not None else "-",
            })
        pooled = rep_ro.get("pooled", {})
        rows.append({
            "Test month": "Pooled",
            "Refills": f"{rep_ro.get('n_paired_events_total', 0):,}",
            "Raw median ±7d": f"{float(pooled.get('raw_median', {}).get('within_7_days_pct')):.1f}%" if pooled.get("raw_median", {}).get("within_7_days_pct") is not None else "-",
            "Baseline engine ±7d": f"{float(pooled.get('baseline_engine', {}).get('within_7_days_pct')):.1f}%" if pooled.get("baseline_engine", {}).get("within_7_days_pct") is not None else "-",
            "Model ±7d": f"{float(pooled.get('model_no_rules', {}).get('within_7_days_pct')):.1f}%" if pooled.get("model_no_rules", {}).get("within_7_days_pct") is not None else "-",
            "Model + rules ±7d": f"{float(pooled.get('model_current_rules', {}).get('within_7_days_pct')):.1f}%" if pooled.get("model_current_rules", {}).get("within_7_days_pct") is not None else "-",
            "Model + rules MAE": f"{float(pooled.get('model_current_rules', {}).get('mae_days')):.2f}" if pooled.get("model_current_rules", {}).get("mae_days") is not None else "-",
        })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Patient Identity Cut-over")
    st.caption(
        "Run once after switching `REFILLCARE_PATIENT_IDENTITY=name_mobile` or `REFILLCARE_ITEM_GROUPING=salt`. "
        "Old-key reminders are retired, every patient is re-scored, stages already sent are never re-sent, "
        "and the database is backed up first."
    )
    k1, k2 = st.columns(2)
    with k1:
        btn_dry = st.button("Execute Dry Run", key="btn_rekey_dry", width="stretch")
    with k2:
        confirm_rekey = st.checkbox("I want to apply the re-key now", key="chk_rekey_confirm")
        btn_apply = st.button("Apply Re-key", key="btn_rekey_apply", width="stretch", disabled=not confirm_rekey)
    if btn_dry or btn_apply:
        from refillcare.engine.rekey import backup_sqlite_db, rekey_reminder_cycles
        from refillcare.ml.service import load_history

        with st.spinner("Re-keying..." if btn_apply else "Running dry run..."):
            try:
                backup = backup_sqlite_db() if btn_apply else None
                with SessionLocal() as db:
                    res = rekey_reminder_cycles(db, load_history(), apply=bool(btn_apply))
                if btn_apply:
                    res["database_backup"] = backup
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Re-key successfully applied.")
                st.json(res)
            except Exception as e:
                st.error(f"Re-key failed: {e}")


if __name__ == "__main__":
    render_app()
