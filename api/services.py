"""Enterprise Service Layer for RefillCare & Mediastra API.

Encapsulates all domain logic, model loading/training, parquet & database synchronization,
batch rollback, and reminder scheduling.
"""

from __future__ import annotations

import sys
import os
import io
import hashlib
from pathlib import Path
from datetime import datetime, date, timedelta
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure project root is on sys.path before local module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("refillcare.api.services")

import pandas as pd
import numpy as np
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from database.models import (
    ImportBatchModel,
    SalesTransactionModel,
    CustomerModel,
    MedicineModel,
    ModelRegistryModel,
    TrainingRunModel,
    PredictionSnapshotModel,
    PredictionOutcomeModel,
    ReminderScheduleModel,
    WhatsAppDeliveryLogModel,
    RefillDecisionModel,
    ReminderCycleModel,
    ReminderStageModel,
)
from refillcare.data.dates import (
    parse_pharmacy_dates,
    format_date_dd_mm_yyyy,
    UI_DATE_FORMAT,
)
from refillcare.data.monthly_ingestion import (
    validate_monthly_sales_data,
    process_monthly_sales_data,
    rollback_monthly_sales_import,
    generate_updated_predictions,
    evaluate_prediction_outcomes,
    determine_mobile_status,
    format_display_phone_10digits,
)
from refillcare.models.prediction import generate_batch_predictions
from reminder.scheduler import (
    RefillReminderScheduler,
    evaluate_refill_eligibility,
)
from reminder.storage import RefillCareStorage
from refillcare.store import StoreConfig



_MONTH_RE = __import__("re").compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def validate_month_key(target_month: Optional[str], allow_all: bool = False) -> Optional[str]:
    """Validate a YYYY-MM month key (raises ValueError -> HTTP 400)."""
    if target_month is None:
        return None
    tm = str(target_month).strip()
    if allow_all and tm.upper() == "ALL":
        return "ALL"
    if not _MONTH_RE.match(tm[:7]) or len(tm) not in (7, 10):
        raise ValueError(f"Invalid month '{target_month}': expected YYYY-MM.")
    return tm[:7]



def _records(df: Optional[pd.DataFrame], limit: int = 5000) -> List[Dict[str, Any]]:
    """DataFrame -> JSON-safe rows (NaN -> None)."""
    if df is None or len(df) == 0:
        return []
    out = df.head(limit).astype(object).where(pd.notna(df.head(limit)), None)
    return out.to_dict(orient="records")

class EnterpriseServices:
    """Centralized enterprise service orchestrator (v1: single store)."""

    def __init__(self, db: Session, store: Optional[StoreConfig] = None):
        self.db = db
        self.store = (store or StoreConfig.from_env()).ensure()
        self.paths = self.store
        self.storage = RefillCareStorage(db_path=str(self.paths.legacy_sqlite))

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------
    def _pm(self):
        from refillcare.engine.persistence import RefillPersistenceManager

        return RefillPersistenceManager(store_name=self.store.store_name, store_contact=self.store.store_contact or None)

    def latest_sales_date(self) -> Optional[date]:
        """Latest purchase date known to the decision store (cheap; no parquet read)."""
        row = self.db.query(func.max(RefillDecisionModel.last_purchase_date)).scalar()
        if row:
            return row
        hist = self.get_active_purchase_history()
        if not hist.empty and "invoice_date" in hist.columns:
            dts = pd.to_datetime(hist["invoice_date"], errors="coerce").dropna()
            if not dts.empty:
                return dts.max().date()
        return None

    def forecast_as_of(self) -> date:
        """T_eval = max(latest sales date + 1 day, today) (falls back to today when no data)."""
        latest = self.latest_sales_date()
        if latest:
            return max(latest + timedelta(days=1), date.today())
        return date.today()

    def current_forecast_month(self) -> str:
        return self.forecast_as_of().strftime("%Y-%m")

    def _decision_is_chronic(self, item_name: Optional[str], purchase_count: Optional[int], cadence_median: Optional[float]) -> bool:
        """Same send rule as the reminder queue (rules-chronic medicine; see ``medicine_can_be_sent``)."""
        from refillcare.engine.persistence import medicine_can_be_sent

        return medicine_can_be_sent(item_name)

    # --------------------------------------------------------------------------
    # 1. SALES INGESTION & BATCH SERVICES
    # --------------------------------------------------------------------------
    # Column mapping (any store's headers -> RefillCare fields; shared with Streamlit)
    def _read_sales_upload(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        from refillcare.data.schema import read_sales_file

        try:
            return read_sales_file(file_bytes, filename)
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

    def detect_column_mapping(self, file_bytes: bytes, filename: str,
                              column_mapping: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """User mapping > saved store profile for this header layout > auto-detect."""
        from refillcare.data.column_mapping import resolve_mapping

        df = self._read_sales_upload(file_bytes, filename)
        return resolve_mapping(df, db=self.db, store_id=self.paths.store_id, user_mapping=column_mapping).to_dict()

    def save_column_mapping(self, headers: List[str], mapping: Dict[str, Optional[str]], actor: str = "api") -> Dict[str, Any]:
        from refillcare.data.column_mapping import save_profile

        p = save_profile(self.db, headers, mapping, store_id=self.paths.store_id, source="user", confirmed_by=actor)
        return {"profile_id": p.profile_id, "store_id": p.store_id, "header_fingerprint": p.header_fingerprint, "mapping": p.mapping}

    def list_column_mappings(self) -> List[Dict[str, Any]]:
        from refillcare.data.column_mapping import list_profiles

        return list_profiles(self.db, self.paths.store_id)

    def delete_column_mapping(self, profile_id: int) -> bool:
        from refillcare.data.column_mapping import delete_profile

        return delete_profile(self.db, profile_id)

    def preview_sales_file(self, file_bytes: bytes, filename: str,
                           column_mapping: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """Validate and preview an uploaded sales Excel/CSV file with date diagnostics."""
        from refillcare.data.column_mapping import resolve_mapping

        df = self._read_sales_upload(file_bytes, filename)
        mres = resolve_mapping(df, db=self.db, store_id=self.paths.store_id, user_mapping=column_mapping)
        mapping_report = mres.to_dict()
        if mres.needs_confirmation:
            return {
                "source_filename": filename, "file_date_range": "-", "records_received": len(df), "valid_records": 0,
                "records_requiring_review": 0, "new_customers": 0, "existing_customers": 0, "unique_customers": 0,
                "medicines_count": 0, "missing_customer_info": 0, "missing_mobile_numbers": 0, "duplicate_records": 0,
                "invalid_quantities": 0,
                "date_diagnostics": {"source_format": "-", "date_range_formatted": "-", "min_date_formatted": "-",
                                     "max_date_formatted": "-", "sample_dates_formatted": [], "valid_count": 0,
                                     "invalid_count": 0, "is_ambiguous": False,
                                     "warning": "Confirm the column mapping to validate this file."},
                "preview_records": [], "can_process": False, "column_mapping": mapping_report,
            }

        from refillcare.data import history_store as hs

        history_df = None if hs.history_count(self.db) > 0 else self.get_active_purchase_history()
        validated_df, review_df, metrics = validate_monthly_sales_data(df, existing_history_df=history_df, column_mapping=mres.mapping)
        if history_df is None and not validated_df.empty:
            ids = set(validated_df["customerId"].dropna().astype(str).str.strip())
            known = hs.known_customers(self.db, ids)
            metrics["new_customers"], metrics["existing_customers"] = len(ids - known), len(ids & known)

        date_diag = metrics["date_diagnostics"]
        can_process = not date_diag.get("is_ambiguous", False) and metrics["valid_records"] > 0

        # Create preview rows
        preview_df = validated_df.copy()
        if "invoice_date" in preview_df.columns:
            preview_df["invoice_date"] = preview_df["invoice_date"].apply(format_date_dd_mm_yyyy)
        preview_cols = [c for c in ["invoice_date", "customerId", "customerName", "itemId", "itemName", "quantity", "packing", "MOBILE_NO"] if c in preview_df.columns]
        if not preview_cols:
            preview_cols = list(preview_df.columns[:8])
        preview_records: List[Dict[str, Any]] = pd.DataFrame(preview_df[preview_cols].head(15).fillna("-")).to_dict(orient="records")

        # Calculate hygiene stats safely
        missing_cust = int(pd.Series(validated_df["customerId"].isna()).sum()) if "customerId" in validated_df.columns else 0
        missing_mob = int(pd.Series(validated_df["MOBILE_NO"].fillna("").apply(determine_mobile_status) != "Valid").sum()) if "MOBILE_NO" in validated_df.columns else 0
        dup_recs = int(pd.Series(validated_df.duplicated()).sum())
        invalid_q = 0
        if "quantity" in validated_df.columns:
            for q in validated_df["quantity"].tolist():
                try:
                    val = float(q)
                    if math.isnan(val) or val <= 0:
                        invalid_q += 1
                except (ValueError, TypeError):
                    invalid_q += 1
        med_cnt = len(set(validated_df["itemId"].dropna().tolist())) if "itemId" in validated_df.columns else 0

        return {
            "source_filename": filename,
            "file_date_range": metrics.get("file_date_range", "N/A"),
            "records_received": metrics.get("records_received", len(df)),
            "valid_records": metrics.get("valid_records", len(validated_df)),
            "records_requiring_review": metrics.get("records_requiring_review", 0),
            "new_customers": metrics.get("new_customers", 0),
            "existing_customers": metrics.get("existing_customers", 0),
            "unique_customers": metrics.get("unique_customers", 0),
            "medicines_count": med_cnt,
            "missing_customer_info": missing_cust,
            "missing_mobile_numbers": missing_mob,
            "duplicate_records": dup_recs,
            "invalid_quantities": invalid_q,
            "date_diagnostics": date_diag,
            "preview_records": preview_records,
            "can_process": can_process,
            "column_mapping": mapping_report,
        }

    def ingest_sales_file(self, file_bytes: bytes, filename: str,
                          column_mapping: Optional[Dict[str, Optional[str]]] = None, actor: str = "api",
                          force: bool = False) -> Dict[str, Any]:
        """Ingest a sales file incrementally (append-only; only its patients are re-scored).

        Only the RefillCare columns are stored (refillcare/data/schema.py). The same file twice is
        refused (checksum) unless ``force``.
        """
        from refillcare.data.column_mapping import mark_profile_used, resolve_mapping
        from refillcare.data.incremental_ingest import file_checksum, ingest_sales_batch

        df = self._read_sales_upload(file_bytes, filename)
        max_rows = int(os.getenv("REFILLCARE_MAX_UPLOAD_ROWS", "2000000"))
        if len(df) > max_rows:
            raise ValueError(f"File has {len(df):,} rows; the limit is {max_rows:,}. Split it by month.")
        mres = resolve_mapping(df, db=self.db, store_id=self.paths.store_id, user_mapping=column_mapping)
        if mres.needs_confirmation:
            raise ValueError("Column mapping needs confirmation - required fields not found: " + ", ".join(mres.missing_required))

        checksum = file_checksum(file_bytes)
        res = ingest_sales_batch(self.db, df, filename, column_mapping=mres.mapping, checksum=checksum, actor=actor, force=force)
        mark_profile_used(self.db, mres, confirmed_by=actor)
        m = res["metrics"]
        return {
            "status": "success",
            "import_batch_id": res["import_batch_id"],
            "source_filename": filename,
            "records_inserted": m["new_records_added"],
            "records_skipped": m["existing_duplicates_skipped"],
            "customers_updated": m["customers_updated"],
            "new_customers": m["new_customers"],
            "new_medicines": m["new_medicines"],
            "latest_sales_date": str(m["latest_sales_date"]),
            "date_range": m.get("file_date_range", ""),
            "records_requiring_review": m["records_requiring_review"],
            "return_lines": m.get("return_lines", 0),
            "return_lines_saved": m.get("return_lines_saved", 0),
            "returns_applied": m.get("returns_applied", 0),
            "returns_waiting": m.get("returns_waiting", 0),
            "earlier_returns_applied_now": m.get("earlier_returns_applied_now", 0),
            "skipped_rows": _records(res.get("skipped_rows_df")),
            "online_sync": res.get("online_sync") or {},
            "ml_data_freshness": self._ml_freshness(None),
            "column_mapping": mres.to_dict(),
            "ignored_columns": mres.ignored_columns,
            "data_warnings": res.get("data_warnings") or {},
            "message": "Sales data ingested; only the new lines were added and their patients re-scored.",
        }

    def waiting_returns(self, limit: int = 200) -> Dict[str, Any]:
        """Return lines saved but still waiting for their sale (same list as the Streamlit Upload tab)."""
        from refillcare.data import history_store as hs

        return hs.waiting_returns(self.db, limit=limit)

    def recheck_returns(self) -> Dict[str, Any]:
        """Apply waiting returns that now match a sale (after the return window changed / a missed day)."""
        from refillcare.data.incremental_ingest import recheck_returns

        return recheck_returns(self.db)

    def _ml_freshness(self, history_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Re-plan the ML windows from the sales dates only (no full-history load)."""
        from refillcare.data import history_store as hs
        from refillcare.engine.post_ingest import ml_freshness_after_ingest

        if history_df is None:
            dates = hs.sales_dates(self.db)
            history_df = pd.DataFrame({"invoice_date": dates}) if not dates.empty else self.get_active_purchase_history()
        return ml_freshness_after_ingest(self.db, history_df)

    def list_import_batches(self) -> List[Dict[str, Any]]:
        """All import batches of the store: its database (authoritative) plus legacy-storage batches."""
        from database.models import ImportBatchModel

        rows = self.db.query(ImportBatchModel).order_by(ImportBatchModel.upload_timestamp.desc()).all()
        out = [{
            "import_batch_id": b.import_batch_id, "source_filename": b.source_filename, "upload_timestamp": b.upload_timestamp,
            "detected_date_range": b.detected_date_range or "N/A", "source_format": b.source_format or "Auto",
            "record_count": b.record_count or 0, "records_inserted": b.records_inserted or 0, "records_skipped": b.records_skipped or 0,
            "records_requiring_review": b.records_requiring_review or 0, "processing_status": b.processing_status or "ACTIVE",
            "is_active": bool(b.is_active), "sync_status": b.sync_status, "sync_error": b.sync_error, "data_warnings": b.data_warnings,
        } for b in rows]
        seen = {b["import_batch_id"] for b in out}
        try:
            out += [b for b in self.storage.get_import_batches() if b.get("import_batch_id") not in seen]
        except Exception:  # legacy storage absent
            pass
        return out

    def rollback_batch(self, batch_id: str) -> Dict[str, Any]:
        """Safely undo an import batch, remove inserted records, and invalidate predictions."""
        from refillcare.data import history_store as hs
        from refillcare.data.incremental_ingest import rollback_batch as rollback_db_batch

        if not hs.batch_lines(self.db, batch_id).empty:
            res = rollback_db_batch(self.db, batch_id)
            return {
                "status": "success",
                "rolled_back_batch_id": batch_id,
                "records_removed": res["records_removed"],
                "restored_latest_sales_date": res["restored_latest_sales_date"],
                "invalidated_snapshots": 0,
                "master_csv_rows_removed": 0,
                "reminder_state_reverted": res.get("reminder_state_reverted", {}),
                "message": f"Batch {batch_id} rolled back: {res['records_removed']} sales lines removed and its patients re-scored.",
            }
        history_df = self.get_active_purchase_history()
        res = rollback_monthly_sales_import(
            current_history_df=history_df,
            import_batch_id=batch_id,
            storage=self.storage,
            data_dir=self.paths.data_dir,
        )

        if res["status"] != "success":
            raise ValueError(res.get("message", "Rollback failed"))

        # Revert the decision / cycle / stage effects of the removed purchases (same path as Streamlit)
        from refillcare.engine.post_ingest import revert_after_rollback

        db_revert = revert_after_rollback(self.db, res, batch_id)

        return {
            "status": "success",
            "rolled_back_batch_id": batch_id,
            "records_removed": res["records_removed"],
            "restored_latest_sales_date": str(res["restored_latest_sales_date"]),
            "invalidated_snapshots": res.get("invalidated_prediction_snapshots", 0),
            "master_csv_rows_removed": res.get("master_csv_rows_removed", 0),
            "reminder_state_reverted": db_revert,
            "message": res["message"],
        }

    # --------------------------------------------------------------------------
    # 2. PREDICTION & OUTCOME SERVICES
    # --------------------------------------------------------------------------
    def generate_predictions(self, prediction_date: Optional[date] = None) -> Dict[str, Any]:
        """Generate updated refill predictions and persist snapshots."""
        history_df = self.get_active_purchase_history()
        model_bundle = self.get_active_model_bundle()
        if model_bundle is None:
            raise ValueError("No active model bundle found. Please train or register a model first.")

        res = generate_updated_predictions(
            history_df=history_df,
            model_bundle=model_bundle,
        )

        if res["status"] != "success":
            raise ValueError(res.get("message", "Prediction generation failed"))

        m = res["metrics"]
        return {
            "status": "success",
            "prediction_date": m.get("prediction_date", str(date.today())),
            "customers_evaluated": m["customers_evaluated"],
            "predictions_generated": m["predictions_generated"],
            "customers_requiring_review": m["customers_requiring_review"],
            "customers_not_eligible": m["customers_not_eligible"],
            "missing_mobile_predictions": m["missing_mobile_predictions"],
            "message": f"Successfully generated {m['predictions_generated']:,} refill predictions.",
        }

    def get_prediction_snapshots(
        self,
        prediction_date: Optional[date] = None,
        tier: Optional[str] = None,
        mobile_status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Query active prediction snapshots from database."""
        snapshots_list = self.storage.get_prediction_snapshots(status="PENDING_EVALUATION")
        if not snapshots_list:
            snapshots_list = self.storage.get_prediction_snapshots()
        if not snapshots_list:
            return []

        df = pd.DataFrame(snapshots_list)
        if tier and tier != "All" and "pilot_tier" in df.columns:
            df = pd.DataFrame(df[df["pilot_tier"].astype(str).str.contains(tier, case=False, na=False, regex=False)])
        if mobile_status and mobile_status != "All" and "mobile_status" in df.columns:
            df = pd.DataFrame(df[df["mobile_status"] == mobile_status])

        records: List[Dict[str, Any]] = df.head(limit).to_dict(orient="records")
        for r in records:
            if "expected_refill_date" in r and pd.notna(r["expected_refill_date"]):
                r["expected_refill_date"] = str(r["expected_refill_date"])
            if "reminder_date" in r and pd.notna(r["reminder_date"]):
                r["reminder_date"] = str(r["reminder_date"])
            if "last_purchase_date" in r and pd.notna(r["last_purchase_date"]):
                r["last_purchase_date"] = str(r["last_purchase_date"])
            if "prediction_date" in r and pd.notna(r["prediction_date"]):
                r["prediction_date"] = str(r["prediction_date"])
            if "estimated_days_of_supply" in r and pd.isna(r["estimated_days_of_supply"]):
                r["estimated_days_of_supply"] = 30.0

        return records

    def evaluate_outcomes(self) -> Dict[str, Any]:
        """Evaluate accuracy of pending prediction snapshots against actual sales."""
        history_df = self.get_active_purchase_history()
        snapshots_list = self.storage.get_prediction_snapshots(status="PENDING_EVALUATION")
        if not snapshots_list:
            snapshots_list = self.storage.get_prediction_snapshots()
        predictions_df = pd.DataFrame(snapshots_list) if snapshots_list else pd.DataFrame()
        res = evaluate_prediction_outcomes(
            predictions_input=predictions_df,
            actual_sales_df=history_df,
        )
        return {
            "predictions_evaluated": res.get("predictions_evaluated", 0),
            "within_3_days_count": res.get("within_3_days_count", 0),
            "within_3_days_pct": float(res.get("within_3_days_pct", 0.0)),
            "within_7_days_count": res.get("within_7_days_count", 0),
            "within_7_days_pct": float(res.get("within_7_days_pct", 0.0)),
            "mean_absolute_error": float(res.get("mean_absolute_error", 0.0)),
            "median_absolute_error": float(res.get("median_absolute_error", 0.0)),
            "early_refills_count": res.get("early_refills_count", 0),
            "late_refills_count": res.get("late_refills_count", 0),
            "on_time_count": res.get("on_time_count", 0),
        }

    # --------------------------------------------------------------------------
    # 3. REMINDER SCHEDULE & EXPORT SERVICES
    # --------------------------------------------------------------------------
    def get_daily_reminders(
        self,
        target_date: Optional[date] = None,
        target_month: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        mobile_filter: str = "All",
        stage: Optional[str] = None,
        refill_function: Optional[str] = None,
        path: Optional[str] = None,
        customer_query: Optional[str] = None,
        medication_query: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get reminder queue scheduled for a target date, full target month (YYYY-MM), or rolling date range with comprehensive clinical filters."""
        import re
        from refillcare.whatsapp.template_formatter import get_refill_function

        target_month = validate_month_key(target_month)
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together.")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be on or before end_date.")
        if start_date and end_date and (end_date - start_date).days > 366:
            raise ValueError("Date range may not exceed 366 days.")

        # Query the persistent review queue
        pm = self._pm()
        if start_date and end_date:
            queue = pm.get_date_range_review_queue(self.db, start_date=start_date, end_date=end_date)
        elif target_month:
            queue = pm.get_monthly_review_queue(self.db, target_month=target_month)
        elif target_date:
            queue = pm.get_today_review_queue(self.db, target_date=target_date)
        else:
            # Default: the active forecast month derived from the latest sales date (T_max + 1)
            queue = pm.get_monthly_review_queue(self.db, target_month=self.current_forecast_month())

        if queue:
            from refillcare.data.medication_classifier import is_chronic_medication
            items = []
            
            # Prepare search clean digits
            c_term = (customer_query or "").strip().lower()
            c_digits = "".join(filter(str.isdigit, c_term))
            m_term = (medication_query or "").strip().lower()

            # Target stage offset parser
            target_stage_offset = None
            if stage and stage != "All" and not stage.startswith("All Active"):
                m_off = re.search(r"([+-]?\d+)", stage)
                if m_off:
                    target_stage_offset = int(m_off.group(1))

            for r in queue:
                i_name = str(r.get("item_name") or "")
                phone = str(r.get("phone_number") or "").strip()
                mob_stat = determine_mobile_status(phone)
                if mobile_filter == "Valid" and mob_stat != "Valid":
                    continue
                elif mobile_filter == "Missing" and mob_stat == "Valid":
                    continue

                # Clinical Path filter
                r_path = str(r.get("path") or "")
                if path and path != "All" and not path.startswith("All"):
                    clean_p = "PATH_A" if "A" in path.upper() else ("PATH_B" if "B" in path.upper() else path)
                    if r_path != clean_p:
                        continue

                # Stage offset & refill function
                stage_val = r.get("stage_offset", 0)
                rf = r.get("refill_function") or get_refill_function(stage_val)

                # Refill function filter
                if refill_function and refill_function != "All" and not refill_function.startswith("All"):
                    if "DUE_REFILL" in refill_function and rf != "DUE_REFILL":
                        continue
                    elif "REFILL_FOLLOW_UP" in refill_function and rf != "REFILL_FOLLOW_UP":
                        continue
                    elif "LAPSED" in refill_function and rf != "LAPSED_REENGAGEMENT":
                        continue

                # Stage offset filter
                if target_stage_offset is not None and stage_val != target_stage_offset:
                    continue

                c_name = str(r.get("customer_name") or "Unknown")

                # Customer search
                if c_term:
                    match_name = c_term in c_name.lower()
                    match_phone = c_term in phone.lower()
                    if c_digits:
                        p_digits = "".join(filter(str.isdigit, phone))
                        match_phone = match_phone or (c_digits in p_digits)
                    if not (match_name or match_phone):
                        continue

                # Medication search
                if m_term and m_term not in i_name.lower():
                    continue

                if stage_val == -7:
                    stage_label = "-7 days (due in 7 days)"
                elif stage_val == -3:
                    stage_label = "-3 days (due in 3 days)"
                elif stage_val == -1:
                    stage_label = "-1 day (due tomorrow)"
                elif stage_val == 0:
                    stage_label = "0 days (due today)"
                elif stage_val in (40, 45):
                    stage_label = f"+{stage_val} days (lapsed re-engagement)"
                elif stage_val > 0:
                    stage_label = f"+{stage_val} days (follow-up)"
                else:
                    stage_label = f"{stage_val} days"

                items.append({
                    "reminder_id": r["reminder_id"],
                    "customer_id": str(r.get("customer_id", "")),
                    "customer_name": c_name,
                    "phone_number": format_display_phone_10digits(phone),
                    "raw_phone_number": phone,
                    "mobile_status": mob_stat,
                    "item_id": str(r.get("item_id", "")),
                    "item_name": i_name,
                    "last_purchase_date": str(r.get("last_purchase_date", "-")),
                    "estimated_days_of_supply": float(r.get("estimated_days_of_supply") or 30.0),
                    "expected_refill_date": str(r.get("expected_refill_date", "-")),
                    "reminder_date": str(r.get("target_send_date", "-")),
                    "reminder_stage": stage_label,
                    "refill_function": rf,
                    "delivery_status": str(r.get("status", "SCHEDULED")),
                })
            if limit is not None:
                return items[offset : offset + limit]
            elif offset > 0:
                return items[offset:]
            return items

        return []

    def export_reminder_csv(
        self,
        target_date: Optional[date] = None,
        target_month: Optional[str] = None,
    ) -> bytes:
        """Export standard 10-column delivery-ready CSV with valid mobile numbers for date or month."""
        from reminder.reminder_engine import RefillReminderEngine

        reminders = self.get_daily_reminders(target_date=target_date, target_month=target_month, mobile_filter="Valid")
        if not reminders:
            return RefillReminderEngine.build_10_column_export_csv(pd.DataFrame())

        df = pd.DataFrame(reminders)
        return RefillReminderEngine.build_10_column_export_csv(df)

    def export_reminder_json(
        self,
        target_date: Optional[date] = None,
        target_month: Optional[str] = None,
        mobile_filter: str = "All",
    ) -> Dict[str, Any]:
        """Export reminder records as full structured JSON format for date or month."""
        reminders = self.get_daily_reminders(target_date=target_date, target_month=target_month, mobile_filter=mobile_filter)
        filter_label = f"Month: {target_month}" if target_month else (f"Date: {target_date.strftime('%Y-%m-%d')}" if target_date else f"Month: {self.current_forecast_month()}")
        return {
            "query_filter": filter_label,
            "target_date": target_date.strftime("%Y-%m-%d") if target_date else None,
            "target_month": target_month,
            "total_records": len(reminders),
            "generated_at": datetime.utcnow().isoformat(),
            "records": reminders,
        }

    def list_available_reminder_months(self) -> List[str]:
        """Retrieve distinct scheduled reminder months (YYYY-MM) from enterprise database."""
        try:
            from database.models import ReminderStageModel
            results = (
                self.db.query(ReminderStageModel.target_send_date)
                .filter(ReminderStageModel.status.in_(["PENDING", "DUE", "APPROVED", "SENT"]))
                .distinct()
                .order_by(ReminderStageModel.target_send_date.asc())
                .all()
            )
            months = set()
            for row in results:
                if row[0]:
                    months.add(row[0].strftime("%Y-%m"))
            return sorted(list(months))
        except Exception:
            logger.exception("Failed to list reminder months")
            return []

    # --------------------------------------------------------------------------
    # 4. MODEL REGISTRY SERVICES
    # --------------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """Registered model versions with measured evaluation evidence."""
        from refillcare.ml import registry as reg

        reg.ensure_bootstrap_versions(self.db)
        out = []
        for v in reg.list_versions(self.db):
            d = reg.to_dict(v, self.db)
            out.append({
                "version_id": v.version_id,
                "model_name": v.model_family,
                "model_type": v.model_type,
                "artifact_path": v.artifact_uri or "",
                "dataset_cutoff_date": v.train_window_end,
                "training_sample_count": v.training_sample_count or 0,
                "hyperparameters": v.hyperparameters or {},
                "metrics_train": v.train_metrics or {},
                "metrics_val": d.get("evaluation_summary") or {},
                "is_active_production": v.status == reg.STATUS_PRODUCTION,
                "created_at": v.created_at,
                "notes": v.notes or "",
                "status": v.status,
                "is_validated": bool(v.is_validated),
                "latest_evaluation_run_id": v.latest_evaluation_run_id,
            })
        return out

    def activate_model(self, version_id: str, promoted_by: str = "api", force: bool = False, force_reason: Optional[str] = None) -> Dict[str, Any]:
        """Promote a version to production - only through the evidence gate."""
        from refillcare.ml.service import promote

        res = promote(self.db, version_id, promoted_by=promoted_by, force=force, force_reason=force_reason)
        if not res.get("promoted"):
            failed = ", ".join(res["decision"].get("failed_checks", []))
            raise ValueError(f"Promotion gate rejected '{version_id}': {failed}")
        return {
            "status": "success",
            "active_version": res["version"]["version_id"],
            "message": f"Model '{res['version']['version_id']}' is now the production model.",
            "decision": res["decision"],
        }

    # --------------------------------------------------------------------------
    # 5. WHATSAPP GATEWAY DISPATCH
    # --------------------------------------------------------------------------
    def dispatch_whatsapp_batch(
        self,
        reminder_date: Optional[date] = None,
        target_month: Optional[str] = None,
        tier_filter: str = "ALL",
        dry_run: bool = True,
        batch_size: int = 50,
        override_stale_data: bool = False,
        override_reason: Optional[str] = None,
        actor: str = "api",
        approve_before_send: bool = False,
    ) -> Dict[str, Any]:
        """Dispatch templated WhatsApp messages for scheduled reminders using 3-tier engine.

        ``approve_before_send`` mirrors the Streamlit "Yes, send" confirmation: the caller's
        pharmacist confirmed, so selected PENDING/DUE stages are approved and sent in one step.

        Live sends are refused while the store's sales data is stale (a patient may already have
        bought); ``override_stale_data`` needs a reason and is logged.
        """
        from refillcare.whatsapp import RefillWhatsAppDispatcher
        from refillcare.whatsapp.client import RefillWhatsAppClient

        target_month = validate_month_key(target_month, allow_all=True)
        if override_stale_data:
            if not (override_reason or "").strip():
                raise ValueError("override_stale_data needs an override_reason.")
            logger.warning("Stale-data send override by %s: %s", actor, override_reason)
        dispatcher = RefillWhatsAppDispatcher(
            client=RefillWhatsAppClient(store_name=self.store.store_name, store_contact=self.store.store_contact or None),
            allow_stale_data=bool(override_stale_data),
        )
        return dispatcher.dispatch_batch(
            db=self.db,
            target_date=reminder_date,
            target_month=target_month,
            tier_filter=tier_filter,
            dry_run=dry_run,
            batch_limit=batch_size,
            approve_before_send=approve_before_send,
            approved_by=actor,
        )

    def _wa_client(self):
        from refillcare.whatsapp.client import RefillWhatsAppClient

        return RefillWhatsAppClient(store_name=self.store.store_name, store_contact=self.store.store_contact or None)

    def get_whatsapp_send_summary(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Counts shown in the "Send WhatsApp reminders" confirmation (same as the Streamlit dialog)."""
        from refillcare.whatsapp import RefillWhatsAppDispatcher
        from refillcare.whatsapp.dispatcher import StaleSalesData, check_live_send_allowed

        today = date.today()
        target_date = target_date or today
        bundles, target_label, _, _ = self._outreach_bundles(target_date=target_date)
        client = self._wa_client()
        summary = RefillWhatsAppDispatcher(client=client).summarize_medsync_send(self.db, bundles)
        stale_message = None
        try:
            check_live_send_allowed(self.db)
        except StaleSalesData as e:
            stale_message = str(e)
        return {
            **summary,
            "target_date": target_date.isoformat(),
            "target_label": target_label,
            "is_today": target_date == today,
            "can_send": target_date == today and summary["to_send"] > 0,
            "whatsapp_configured": bool(client.api_key and client.waba_number),
            "sales_data_stale": stale_message is not None,
            "stale_message": stale_message,
        }

    def send_whatsapp_today(self, override_reason: Optional[str] = None, actor: str = "api") -> Dict[str, Any]:
        """The pharmacist pressed "Yes, send": approve and send ALL of today's reminders (Streamlit parity).

        Only today's reminders can be sent. Opt-outs, invalid mobiles, the control group and double sends are
        still skipped by the dispatcher; stale sales data blocks the send unless a reason is given (logged).
        """
        from refillcare.whatsapp import RefillWhatsAppDispatcher

        client = self._wa_client()
        if not (client.api_key and client.waba_number):
            raise ValueError("WhatsApp sending is not set up (Xinno API key or WABA number missing).")
        override = bool((override_reason or "").strip())
        if override:
            logger.warning("Stale-data send override by %s: %s", actor, override_reason)
        bundles, target_label, _, _ = self._outreach_bundles(target_date=date.today())
        outcome = RefillWhatsAppDispatcher(client=client, allow_stale_data=override).dispatch_medsync_batch(
            db=self.db, bundles=bundles, dry_run=False, batch_limit=None, approve_before_send=True, approved_by=actor,
        )
        return {
            "target_label": target_label,
            "sent": outcome.get("success_count", 0),
            "failed": outcome.get("failed_count", 0),
            "skipped": outcome.get("skipped_not_approved_or_already_sent", 0),
            "failed_rows": [
                {"customer": r.get("customer_name"), "mobile": r.get("phone_number"), "medicines": r.get("medication_name"), "reason": r.get("message")}
                for r in outcome.get("delivery_logs", []) if r.get("status") == "FAILED"
            ],
        }

    # --------------------------------------------------------------------------
    # 6. OPERATIONS KPI ANALYTICS
    # --------------------------------------------------------------------------
    def _chronic_filtered_stage_distribution(self) -> Dict[str, int]:
        """Active eligible stages per month, filtered by the SAME chronic gate as the reminder lists."""
        from database.models import ReminderCycleModel, ReminderStageModel, RefillDecisionModel

        rows = (
            self.db.query(
                func.substr(ReminderStageModel.target_send_date, 1, 7).label("m"),
                RefillDecisionModel.item_name,
                RefillDecisionModel.purchase_count,
                RefillDecisionModel.cadence_median,
                func.count().label("cnt"),
            )
            .join(ReminderCycleModel, ReminderStageModel.cycle_id == ReminderCycleModel.cycle_id)
            .join(RefillDecisionModel, ReminderCycleModel.decision_id == RefillDecisionModel.decision_id)
            .filter(
                ReminderStageModel.status.in_(["PENDING", "DUE", "APPROVED"]),
                ReminderCycleModel.is_active == True,
                RefillDecisionModel.is_eligible == True,
            )
            .group_by("m", RefillDecisionModel.item_name, RefillDecisionModel.purchase_count, RefillDecisionModel.cadence_median)
            .all()
        )
        dist: Dict[str, int] = {}
        for m_key, item, pc, cad, cnt in rows:
            if m_key and self._decision_is_chronic(item, pc, cad):
                dist[m_key] = dist.get(m_key, 0) + int(cnt)
        return dict(sorted(dist.items()))

    def get_ml_evidence(self) -> Dict[str, Any]:
        """Production model + its latest measured evaluation (never typed-in numbers)."""
        from refillcare.ml.service import production_accuracy

        return production_accuracy(self.db)

    def get_operations_kpi(self) -> Dict[str, Any]:
        """Aggregate executive operations KPIs (all values measured from store data)."""
        from database.models import ReminderCycleModel, ReminderStageModel, RefillDecisionModel
        from refillcare.ml.outcomes import outcome_summary

        from refillcare.data import history_store as hs

        latest_date_str = "N/A"
        date_range_str = "N/A"
        customers = 0
        if hs.history_count(self.db) > 0:
            # SQL aggregates instead of loading the whole history (~4-5 s per cold worker); same values
            from sqlalchemy import func
            from database.models import SalesTransactionModel as Line

            total_tx, customers, d_min, d_max = self.db.query(
                func.count(Line.id), func.count(func.distinct(Line.customer_id)),
                func.min(Line.invoice_date), func.max(Line.invoice_date)).one()
            total_tx, customers = int(total_tx or 0), int(customers or 0)
            dts = pd.to_datetime(pd.Series([d_min, d_max]), errors="coerce").dropna()
        else:  # stores still on the legacy parquet
            history_df = self.get_active_purchase_history()
            total_tx = len(history_df) if history_df is not None else 0
            if history_df is not None and "customerId" in history_df.columns:
                customers = len(set(history_df["customerId"].dropna().tolist()))
            dts = (pd.to_datetime(history_df["invoice_date"], errors="coerce").dropna()
                   if history_df is not None and not history_df.empty and "invoice_date" in history_df.columns else pd.Series(dtype="datetime64[ns]"))
        if not dts.empty:
            latest_date_str = dts.max().strftime(UI_DATE_FORMAT)
            date_range_str = f"{dts.min().strftime('%b %Y')} – {dts.max().strftime('%b %Y')}"

        as_of = self.forecast_as_of()
        target_prediction_month = as_of.strftime("%Y-%m")
        monthly_stage_dist = self._chronic_filtered_stage_distribution()
        target_month_reminders_count = monthly_stage_dist.get(target_prediction_month, 0)

        due_today = len(self._pm().get_today_review_queue(self.db, target_date=as_of))
        rev_summary = self.get_clinical_review_summary(target_prediction_month)
        review_count = rev_summary.get("total_in_review_queue", 0)

        active_batch = self.storage.get_latest_active_import_batch()
        evidence = self.get_ml_evidence()
        prod_eval = ((evidence.get("evaluation") or {}).get("production") or {})
        live = outcome_summary(self.db, window_days=90)
        overall = live.get("overall") or {}

        chr_count = None
        try:
            from refillcare.data.medication_classifier import list_classification_rules

            all_rules = list_classification_rules()
            chr_count = len([r for r in all_rules if r.get("rule_type") == "CHRONIC"]) or None
        except Exception:
            chr_count = None

        from refillcare.engine.overview_kpis import overview_kpis

        ov = overview_kpis(self.db)
        return {
            "overview_month": ov["month"],
            "reminder_messages_this_month": ov["reminder_messages_month"],
            "patients_needing_review_today": ov["patients_needing_review_today"],
            "total_customers_monitored": ov["customers_monitored"],  # history_store.store_summary (same as every tab)
            "all_time_customers": ov.get("all_time_customers"),
            "active_patients": ov.get("active_patients"),
            "active_patients_label": ov.get("active_patients_label"),
            "has_12m_history": ov.get("has_12m_history"),
            "history_span_months": ov.get("history_span_months"),
            "total_sales_transactions": ov["sales_lines_stored"],
            "latest_sales_date": latest_date_str,
            "sales_coverage_date_range": date_range_str,
            "upcoming_reminders_count": target_month_reminders_count,
            "reminders_due_today": due_today,
            "customers_requiring_review": review_count,
            "active_model_version": evidence.get("production_model_version") or "none",
            "active_batch_id": active_batch.get("import_batch_id") if active_batch else None,
            "model_is_validated": evidence.get("is_validated", False),
            "measured_backtest_within_7_days_pct": prod_eval.get("within_7_days_pct"),
            "measured_backtest_mae_days": prod_eval.get("mae_days"),
            "measured_backtest_run_id": (evidence.get("evaluation") or {}).get("run_id"),
            "live_within_7_days_pct": overall.get("within_7_days_pct"),
            "live_resolved_predictions": overall.get("resolved"),
            "channel_summary": self.get_transaction_channel_kpis(),
            "target_prediction_month": target_prediction_month,
            "target_month_reminders_count": target_month_reminders_count,
            "monthly_stage_distribution": monthly_stage_dist,
            "adherence_rate_pct": overall.get("refill_rate_pct"),
            "chronic_medicines_count": chr_count,
            "out_of_stock_or_critical": due_today,
        }

    # --------------------------------------------------------------------------
    # 7. CLINICAL REVIEW QUEUE SERVICES
    # --------------------------------------------------------------------------
    def get_clinical_review_summary(self, target_month: Optional[str] = None) -> Dict[str, Any]:
        """Compute live Clinical Review Queue summary metrics matching Streamlit."""
        from database.models import RefillDecisionModel
        from refillcare.data.monthly_ingestion import determine_mobile_status

        m_key = validate_month_key(target_month, allow_all=True) if target_month else None
        if not m_key or m_key == "ALL":
            m_key = self.current_forecast_month()

        # 1. Missing mobile on eligible refills
        q_missing = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == True,
            (RefillDecisionModel.mobile_no == None) | (RefillDecisionModel.mobile_no == "") | (RefillDecisionModel.mobile_no == "nan")
        )
        if m_key:
            q_missing = q_missing.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
        r_missing = q_missing.all()

        # 2. Cold-start / single purchase from previous sales month
        try:
            y, m = map(int, m_key.split("-"))
            prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
            prev_str = f"{prev_y:04d}-{prev_m:02d}"
        except Exception:
            prev_str = (self.forecast_as_of().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")

        q_cold = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == False,
            RefillDecisionModel.last_purchase_date.like(f"{prev_str}%")
        )
        r_cold = q_cold.limit(2000).all()

        # 3. Stability risk / variance
        q_stab = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == True,
            RefillDecisionModel.stability_tier.in_(["MEDIUM-RISK", "UNSTABLE"])
        )
        if m_key:
            q_stab = q_stab.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
        r_stab = q_stab.limit(500).all()

        total_in_queue = len(r_missing) + len(r_cold) + len(r_stab)

        # Missing mobile count across all review records
        missing_mob_cnt = len(r_missing)
        for r in r_cold:
            if determine_mobile_status(r.mobile_no or "") != "Valid":
                missing_mob_cnt += 1
        for r in r_stab:
            if determine_mobile_status(r.mobile_no or "") != "Valid":
                missing_mob_cnt += 1

        # Stability risk & Cold start matching Streamlit keywords
        stab_risk_cnt = len(r_stab)
        cold_start_cnt = 0
        for r in r_cold:
            reas = str(r.decision_reason or "").lower()
            if any(k in reas for k in ["single", "cold-start", "< 2", "insufficient", "recurrence"]):
                cold_start_cnt += 1
            if any(k in reas for k in ["review", "verification", "variance", "refill interval", "stability", "risk"]):
                stab_risk_cnt += 1

        avail_months = self.list_available_reminder_months()
        if m_key not in avail_months:
            avail_months.append(m_key)
        avail_months = sorted(list(set(avail_months)), reverse=True)

        return {
            "target_month": m_key,
            "total_in_review_queue": total_in_queue,
            "missing_mobile_numbers": missing_mob_cnt,
            "stability_variance_risk": stab_risk_cnt,
            "cold_start_single_purchase": cold_start_cnt,
            "available_months": avail_months,
        }

    def get_clinical_review_queue(
        self,
        target_month: Optional[str] = None,
        category: Optional[str] = None,
        missing_mobile_only: bool = False,
        mobile_status: Optional[str] = None,
        search: Optional[str] = None,
        customer_search: Optional[str] = None,
        medication_search: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve live Clinical Review Queue candidate records matching Streamlit with regex safety."""
        from database.models import RefillDecisionModel
        from refillcare.data.monthly_ingestion import determine_mobile_status, format_display_phone_10digits

        if missing_mobile_only:
            mobile_status = "Missing"
        if search and not customer_search:
            customer_search = search

        m_key = validate_month_key(target_month, allow_all=True) if target_month else None
        if not m_key or m_key == "ALL":
            m_key = self.current_forecast_month()

        # 1. Missing mobile on eligible refills
        q1 = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == True,
            (RefillDecisionModel.mobile_no == None) | (RefillDecisionModel.mobile_no == "") | (RefillDecisionModel.mobile_no == "nan")
        )
        if m_key:
            q1 = q1.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
        r_missing = q1.all()

        # 2. Cold-start / single purchase from previous sales month
        try:
            y, m = map(int, m_key.split("-"))
            prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
            prev_str = f"{prev_y:04d}-{prev_m:02d}"
        except Exception:
            prev_str = (self.forecast_as_of().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")

        q2 = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == False,
            RefillDecisionModel.last_purchase_date.like(f"{prev_str}%")
        )
        r_cold = q2.limit(2000).all()

        # 3. Stability risk / variance
        q3 = self.db.query(RefillDecisionModel).filter(
            RefillDecisionModel.is_eligible == True,
            RefillDecisionModel.stability_tier.in_(["MEDIUM-RISK", "UNSTABLE"])
        )
        if m_key:
            q3 = q3.filter(RefillDecisionModel.expected_refill_date.like(f"{m_key}%"))
        r_stab = q3.limit(500).all()

        raw_items = []
        for r in r_missing:
            raw_items.append({
                "customer_id": str(r.customer_id or ""),
                "customer_name": r.customer_name or "Unknown",
                "mobile_number": "-",
                "raw_mobile": "",
                "mobile_status": "Missing",
                "medication": r.item_name or "Medication",
                "last_purchase_date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                "expected_refill_date": str(r.expected_refill_date) if r.expected_refill_date else "-",
                "review_category": "Missing Mobile Numbers",
                "review_reason": "Missing mobile number for automated reminder delivery",
            })
        for r in r_cold:
            mob_st = determine_mobile_status(r.mobile_no or "")
            raw_items.append({
                "customer_id": str(r.customer_id or ""),
                "customer_name": r.customer_name or "Unknown",
                "mobile_number": format_display_phone_10digits(r.mobile_no or ""),
                "raw_mobile": str(r.mobile_no or ""),
                "mobile_status": mob_st,
                "medication": r.item_name or "Medication",
                "last_purchase_date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                "expected_refill_date": "-",
                "review_category": "Cold-Start / Single Purchase",
                "review_reason": r.decision_reason or "Cold-start: single purchase requiring recurrence validation",
            })
        for r in r_stab:
            mob_st = determine_mobile_status(r.mobile_no or "")
            raw_items.append({
                "customer_id": str(r.customer_id or ""),
                "customer_name": r.customer_name or "Unknown",
                "mobile_number": format_display_phone_10digits(r.mobile_no or ""),
                "raw_mobile": str(r.mobile_no or ""),
                "mobile_status": mob_st,
                "medication": r.item_name or "Medication",
                "last_purchase_date": str(r.last_purchase_date) if r.last_purchase_date else "-",
                "expected_refill_date": str(r.expected_refill_date) if r.expected_refill_date else "-",
                "review_category": "History Review Required",
                "review_reason": f"Variance risk ({r.stability_tier}): pharmacist interval review recommended",
            })

        # Apply filters with regex safety
        c_term = (customer_search or "").strip().lower()
        c_digits = "".join(filter(str.isdigit, c_term))
        m_term = (medication_search or "").strip().lower()

        filtered = []
        for it in raw_items:
            # Category filter (flexible matching across variant category labels)
            if category and category != "All" and not category.startswith("All"):
                cat_l = category.lower().replace("-", " ").strip()
                it_cat_l = it["review_category"].lower().replace("-", " ").strip()
                match_cat = (cat_l in it_cat_l) or (it_cat_l in cat_l)
                if cat_l.startswith("missing") and "missing" in it_cat_l:
                    match_cat = True
                elif ("cold" in cat_l or "single" in cat_l) and ("cold" in it_cat_l or "single" in it_cat_l):
                    match_cat = True
                elif ("stability" in cat_l or "variance" in cat_l) and ("stability" in it_cat_l or "review" in it_cat_l or "variance" in it_cat_l):
                    match_cat = True
                if not match_cat:
                    continue

            # Mobile status filter
            if mobile_status and mobile_status != "All" and not mobile_status.startswith("All"):
                if it["mobile_status"] != mobile_status:
                    continue

            # Customer name or phone search
            if c_term:
                match_c = c_term in it["customer_name"].lower() or c_term in it["customer_id"].lower()
                if c_digits:
                    raw_digits = "".join(filter(str.isdigit, it["raw_mobile"]))
                    match_c = match_c or (c_digits in raw_digits)
                if not match_c:
                    continue

            # Medication search
            if m_term and m_term not in it["medication"].lower():
                continue

            # Clean output
            clean_item = dict(it)
            clean_item.pop("raw_mobile", None)
            filtered.append(clean_item)

        return filtered[offset : offset + limit]

    def get_consensus_regimen_analytics(self) -> Dict[str, Any]:
        """Regimen / archetype prevalence computed from eligible decisions, and
        benchmarks taken from the latest completed evaluation run (no typed-in numbers)."""
        from collections import Counter
        from database.models import MLEvaluationRunModel
        from refillcare.engine.persistence import parse_decision_provenance

        reasons = [
            r[0]
            for r in self.db.query(RefillDecisionModel.decision_reason)
            .filter(RefillDecisionModel.is_eligible == True, RefillDecisionModel.path == "PATH_A")
            .all()
        ]
        reg_counter: Counter = Counter()
        arch_counter: Counter = Counter()
        for reason in reasons:
            arch, regimen = parse_decision_provenance(reason)
            reg_counter[regimen] += 1
            arch_counter[arch] += 1
        total = sum(reg_counter.values()) or 1
        regimen_meta = {
            "1.0 tab/d (OD)": ("Once Daily (OD)", 1.0, "Once-a-day maintenance therapy."),
            "0.5 tab/d (QOD)": ("Alternate Day / Intermittent (QOD)", 0.5, "Alternate-day or intermittent dosing."),
            "2.0 tabs/d (BD)": ("Twice Daily (BD)", 2.0, "Morning and evening dosing."),
            "3.0 tabs/d (TID)": ("Thrice Daily (TID) / Multiple", 3.0, "Three or more doses per day."),
        }
        regimens = []
        for key, (label, rate, desc) in regimen_meta.items():
            cnt = reg_counter.get(key, 0)
            regimens.append({
                "regimen": label,
                "daily_rate": rate,
                "prevalence_pct": round(100.0 * cnt / total, 1),
                "sample_count": cnt,
                "description": f"{desc} (inferred from decision reasons of eligible Path A decisions)",
            })
        archetypes = [
            {"archetype": a, "description": f"{c} eligible Path A decisions", "example": "Computed from current decisions"}
            for a, c in arch_counter.most_common()
        ]

        benchmarks = []
        run = (
            self.db.query(MLEvaluationRunModel)
            .filter(MLEvaluationRunModel.status == "COMPLETED")
            .order_by(MLEvaluationRunModel.completed_at.desc())
            .first()
        )
        evidence = self.get_ml_evidence()
        if run is not None and run.report:
            win = run.report.get("evaluation_window") or {}
            for name, mtr in ((run.report.get("paired_comparison") or {}).get("methods") or {}).items():
                is_prod = evidence.get("production_model_version") and name == f"engine[{evidence['production_model_version']}]"
                benchmarks.append({
                    "method": name,
                    "within_7_days_pct": float(mtr.get("within_7_days_pct") or 0.0),
                    "mae_days": float(mtr.get("mae_days") or 0.0),
                    "status": f"{'Production - ' if is_prod else ''}measured {win.get('start')}..{win.get('end')} (n={mtr.get('n')}, run {run.run_id})",
                })

        prod_version = evidence.get("production_model_version") or "none"
        from refillcare.ml import registry as reg

        pv = reg.get_version(self.db, prod_version) if prod_version != "none" else None
        return {
            "engine_version": prod_version,
            "total_transitions_trained": int(pv.training_sample_count or 0) if pv else 0,
            "regimens": regimens,
            "archetypes": archetypes,
            "benchmarks": benchmarks,
        }

    def get_transaction_channel_kpis(self) -> Dict[str, int]:
        """Dynamically compute transaction channel distribution from enterprise database (live across batches)."""
        from refillcare.data import history_store as hs
        if self.db is not None and hs.history_count(self.db) > 0:
            try:
                c = hs.channel_counts(self.db)
                text = hs.return_breakdown_text(c)
                return {
                    "total_transactions": c["stored"],
                    "customer_sales": c["customer_sales"],
                    "b2b_inter_store": c["b2b_inter_store"],
                    "unknown": c["unknown"],
                    "excluded_from_refillcare": c["stored"] - c["refillcare_eligible"],
                    "refillcare_eligible": c["refillcare_eligible"],
                    "raw_total_transactions": c["received"],
                    "cleaned_transactions": c["stored"],
                    "purged_duplicates": c["not_stored"],  # kept for compatibility: returns / invalid / already uploaded
                    **{k: c[k] for k in ("sales_lines", "return_lines", "return_lines_stored", "return_lines_not_stored",
                                         "returns_netted", "returns_unmatched", "returns_waiting", "returns_unmatched_old", "invalid_rows",
                                         "sales_fully_returned", "returned_corrections", "returned_late",
                                         "return_window_days", "return_correction_days")},
                    "breakdown_summary": text["summary"],
                    "breakdown_detail": text["detail"],
                    "breakdown_audit": text["audit"],
                }
            except Exception:
                pass

        # Static fallback if database is unavailable
        from refillcare.data.transaction_classifier import get_transaction_channel_summary
        clean_tx_path = self.paths.clean_transactions_parquet
        raw_csv_path = self.paths.master_csv

        raw_total = 0
        if raw_csv_path.exists():
            try:
                with open(raw_csv_path, "rb") as f:
                    cnt = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))
                    if cnt > 0:
                        raw_total = max(0, cnt - 1)
            except Exception:
                raw_total = 0

        if clean_tx_path.exists():
            try:
                df = pd.read_parquet(clean_tx_path, columns=["transaction_type", "refillcare_eligible"])
            except Exception:
                df = pd.read_parquet(clean_tx_path)
            res = get_transaction_channel_summary(df)
        elif self.paths.history_parquet.exists():
            try:
                df = pd.read_parquet(self.paths.history_parquet, columns=["transaction_type", "refillcare_eligible"])
            except Exception:
                df = pd.read_parquet(self.paths.history_parquet)
            res = get_transaction_channel_summary(df)
        else:
            res = {
                "total_transactions": 0,
                "customer_sales": 0,
                "b2b_inter_store": 0,
                "unknown": 0,
                "excluded_from_refillcare": 0,
                "refillcare_eligible": 0,
            }

        res["raw_total_transactions"] = raw_total or res.get("total_transactions", 0)
        res["cleaned_transactions"] = res.get("total_transactions", 0)
        res["purged_duplicates"] = max(0, res["raw_total_transactions"] - res["cleaned_transactions"])
        return res

    # --------------------------------------------------------------------------
    # 9. MED-SYNC & QUANTILE UNCERTAINTY SERVICES
    # --------------------------------------------------------------------------
    def get_med_sync_bundles(
        self,
        sync_window_days: int = 0,
        pharmacy_name: str = "Mediastra Pharmacy",
        customer_id: Optional[str] = None,
        target_month: Optional[str] = None,
        target_date: Optional[date] = None,
        lifecycle_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate patient-grouped Med-Sync bundles and operational impact metrics for a target date or month."""
        from refillcare.engine.med_sync import MedSyncEngine
        from refillcare.engine.persistence import RefillPersistenceManager

        target_month = validate_month_key(target_month, allow_all=True)
        pm = self._pm()
        if target_date:
            queue_list = pm.get_today_review_queue(self.db, target_date=target_date)
        elif target_month and target_month != "ALL":
            queue_list = pm.get_monthly_review_queue(self.db, target_month=target_month)
        else:
            # Default: the active forecast date (latest sales date + 1 day)
            target_date = self.forecast_as_of()
            queue_list = pm.get_today_review_queue(self.db, target_date=target_date)

        records: List[Dict[str, Any]] = []

        from refillcare.data.medication_classifier import is_chronic_medication
        if queue_list:
            for q in queue_list:
                if customer_id and q.get("customer_id") != customer_id:
                    continue
                # queue rows already passed the engine-consistent chronic gate
                records.append({
                    "reminder_id": q.get("reminder_id"),
                    "customer_id": q.get("customer_id", ""),
                    "customer_name": q.get("customer_name", "Valued Patient"),
                    "mobile_no": q.get("phone_number"),
                    "item_id": q.get("item_id", ""),
                    "item_name": q.get("item_name", ""),
                    "is_eligible": True,
                    "stability_tier": q.get("stability_tier", "MEDIUM-SAFE"),
                    "predicted_interval_days": q.get("predicted_interval_days") or q.get("estimated_days_of_supply"),
                    "last_purchase_date": q.get("last_purchase_date"),
                    "expected_refill_date": q.get("expected_refill_date") or q.get("target_send_date"),
                    "target_send_date": q.get("target_send_date"),
                    "stage_offset": q.get("stage_offset", 0),
                    "dos_days": q.get("estimated_days_of_supply"),
                })
        else:
            query = self.db.query(RefillDecisionModel).filter(RefillDecisionModel.is_eligible == True)
            if customer_id:
                query = query.filter(RefillDecisionModel.customer_id == customer_id)
            decisions_db = query.all()

            if decisions_db:
                for d in decisions_db:
                    if d.path != "PHARMACIST_ADDED" and not is_chronic_medication(str(d.item_name or "")):
                        continue
                    records.append({
                        "customer_id": d.customer_id,
                        "customer_name": d.customer_name,
                        "mobile_no": d.mobile_no,
                        "item_id": d.item_id,
                        "item_name": d.item_name,
                        "is_eligible": d.is_eligible,
                        "stability_tier": d.stability_tier,
                        "predicted_interval_days": d.predicted_interval_days,
                        "last_purchase_date": d.last_purchase_date,
                        "expected_refill_date": d.expected_refill_date,
                        "dos_days": d.dos_days,
                    })
            else:
                snapshots = self.storage.get_prediction_snapshots()
                if snapshots:
                    for s in snapshots:
                        records.append({
                            "customer_id": s.get("customer_id", ""),
                            "customer_name": s.get("customer_name", "Valued Patient"),
                            "mobile_no": s.get("phone_number"),
                            "item_id": s.get("item_id", ""),
                            "item_name": s.get("item_name", ""),
                            "is_eligible": True,
                            "stability_tier": s.get("stability_tier", "MEDIUM-SAFE"),
                            "predicted_interval_days": s.get("predicted_interval_days") or s.get("estimated_days_of_supply"),
                            "last_purchase_date": s.get("last_purchase_date"),
                            "expected_refill_date": s.get("expected_refill_date"),
                            "dos_days": s.get("estimated_days_of_supply"),
                        })

        # Determine latest sales date to establish next-month prediction horizon
        latest_sales_dt = None
        for r in records:
            lpd = r.get("last_purchase_date")
            if lpd:
                try:
                    if isinstance(lpd, str):
                        dt = datetime.strptime(lpd[:10], "%Y-%m-%d").date()
                    elif isinstance(lpd, (datetime, date)):
                        dt = lpd if isinstance(lpd, date) else lpd.date()
                    else:
                        dt = None
                    if dt and (latest_sales_dt is None or dt > latest_sales_dt):
                        latest_sales_dt = dt
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
            last_sales_month_str = (now_dt.replace(day=1) - timedelta(days=1)).strftime("%B %Y")

        default_target_month_key = f"{target_next_year}-{target_next_month:02d}"

        engine = MedSyncEngine(sync_window_days=sync_window_days, default_pharmacy_name=pharmacy_name)
        all_bundles = engine.cluster_patient_decisions(records, sync_window_days=sync_window_days, pharmacy_name=pharmacy_name)

        # Build available months list
        available_tuples = sorted(list(set((b.anchor_refill_date.year, b.anchor_refill_date.month) for b in all_bundles)))
        target_tuple = (target_next_year, target_next_month)
        future_tuples = [t for t in available_tuples if t >= target_tuple]
        past_tuples = [t for t in available_tuples if t < target_tuple]
        ordered_tuples = ([target_tuple] if target_tuple in available_tuples else []) + [t for t in future_tuples if t != target_tuple] + past_tuples

        available_months = []
        for y, m in ordered_tuples:
            m_key = f"{y}-{m:02d}"
            m_name = date(y, m, 1).strftime("%B %Y")
            available_months.append({
                "key": m_key,
                "label": f"{m_name} (Active Target)" if (y, m) == target_tuple else m_name,
                "year": y,
                "month": m,
                "is_default": (y, m) == target_tuple,
            })
        available_months.append({
            "key": "ALL",
            "label": "All Future Months",
            "year": 0,
            "month": 0,
            "is_default": False,
        })

        # Apply target date or target month filter (mirrors Streamlit medsync logic)
        if queue_list:
            filtered_bundles = all_bundles
            selected_target = target_date.strftime("%Y-%m-%d") if target_date else (target_month or default_target_month_key)
            filter_mode = "DATE" if target_date else "MONTH"
        elif target_date:
            filtered_bundles = [
                b for b in all_bundles
                if b.anchor_refill_date == target_date
            ]
            selected_target = target_date.strftime("%Y-%m-%d")
            filter_mode = "DATE"
        elif target_month is not None and target_month != "ALL":
            try:
                sel_y, sel_m = map(int, target_month.split("-"))
                filtered_bundles = [
                    b for b in all_bundles
                    if b.anchor_refill_date.year == sel_y and b.anchor_refill_date.month == sel_m
                ]
            except Exception:
                filtered_bundles = all_bundles
            selected_target = target_month
            filter_mode = "MONTH"
        elif target_month == "ALL":
            filtered_bundles = all_bundles
            selected_target = "ALL"
            filter_mode = "MONTH"
        else:
            # Default to target next month
            try:
                sel_y, sel_m = map(int, default_target_month_key.split("-"))
                filtered_bundles = [
                    b for b in all_bundles
                    if b.anchor_refill_date.year == sel_y and b.anchor_refill_date.month == sel_m
                ]
            except Exception:
                filtered_bundles = all_bundles
            selected_target = default_target_month_key
            filter_mode = "MONTH"

        # Apply optional lifecycle tier filter (DUE, FOLLOWUP, LAPSED)
        if lifecycle_tier and lifecycle_tier.upper() != "ALL":
            filtered_bundles = [
                b for b in filtered_bundles
                if getattr(b, "lifecycle_tier", "DUE").upper() == lifecycle_tier.upper()
            ]

        impact = engine.summarize_sync_impact(filtered_bundles)

        # Format bundles with 10-digit mobile number
        formatted_bundles = []
        for b in filtered_bundles:
            bd = b.to_dict()
            bd["raw_mobile_no"] = b.mobile_no or ""
            bd["mobile_no"] = format_display_phone_10digits(b.mobile_no or "")
            formatted_bundles.append(bd)

        return {
            "sync_window_days": sync_window_days,
            "filter_mode": filter_mode,
            "selected_target": selected_target,
            "selected_target_month": selected_target if filter_mode == "MONTH" else default_target_month_key,
            "default_target_month": default_target_month_key,
            "last_sales_month": last_sales_month_str,
            "available_months": available_months,
            "impact_summary": impact,
            "bundles": formatted_bundles,
        }

    def get_quantile_uncertainty_metrics(self) -> Dict[str, Any]:
        """Retrieve 3-head Quantile uncertainty envelope benchmark metrics."""
        import json
        report_path = self.paths.processed_dir / "chronic_model_benchmark_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["used_in_production"] = False
            data["status"] = "offline_experiment"
            data["note"] = (
                "Offline experiment from scripts/train_chronic_specialized_model.py. The quantile model is not "
                "registered and does not serve predictions; production accuracy comes from /api/v1/ml/evaluations."
            )
            return data
        return {
            "used_in_production": False,
            "status": "not_trained",
            "message": "Benchmark report not found. Train via scripts/train_chronic_specialized_model.py.",
        }

    # --------------------------------------------------------------------------
    # 9. PHARMACIST MEDICATION RULES & LIVE CLASSIFIER SERVICES
    # --------------------------------------------------------------------------
    def list_medication_rules(self, rule_type: Optional[str] = None, search: Optional[str] = None) -> Dict[str, Any]:
        """Fetch active medication classification rules from Supabase (or local fallback)."""
        from refillcare.data.medication_classifier import (
            list_classification_rules,
            EXCLUDED_CATEGORIES,
            CHRONIC_STRONG_MARKERS,
        )
        filter_val = "All"
        if rule_type:
            rt_u = rule_type.strip().upper()
            if rt_u in ("CHRONIC", "EXCLUDE"):
                filter_val = rt_u
            elif rt_u == "EXCLUSION":
                filter_val = "EXCLUDE"

        raw_rules = list_classification_rules(search_query=search or "", rule_type_filter=filter_val, limit=None)
        formatted = []
        for r in raw_rules:
            formatted.append({
                "id": str(r.get("id", "")),
                "pattern": r.get("medication_name_or_keyword", ""),
                "rule_type": "EXCLUSION" if r.get("rule_type") == "EXCLUDE" else r.get("rule_type", "CHRONIC"),
                "category": r.get("category", "CLINICAL_GATE"),
                "brand_name": r.get("brand_name", r.get("medication_name_or_keyword")),
                "notes": r.get("notes", "Pharmacist Rule"),
                "created_by": r.get("created_by", "Pharmacist"),
                "is_active": bool(r.get("is_active", True)),
                "created_at": str(r.get("created_at", "")),
            })

        if not formatted:
            if filter_val in ("All", "CHRONIC"):
                for m in CHRONIC_STRONG_MARKERS[:25]:
                    if not search or search.upper() in m:
                        formatted.append({
                            "id": f"code-chr-{m}",
                            "pattern": m,
                            "rule_type": "CHRONIC",
                            "category": "CHRONIC_MAINTENANCE",
                            "brand_name": m,
                            "notes": "Built-in Clinical Marker",
                            "created_by": "System",
                            "is_active": True,
                            "created_at": None,
                        })
            if filter_val in ("All", "EXCLUDE"):
                for cat_name, kw_list in EXCLUDED_CATEGORIES.items():
                    for kw in kw_list[:5]:
                        if not search or search.upper() in kw:
                            formatted.append({
                                "id": f"code-excl-{kw}",
                                "pattern": kw,
                                "rule_type": "EXCLUSION",
                                "category": cat_name,
                                "brand_name": kw,
                                "notes": "Built-in Exclusion Pattern",
                                "created_by": "System",
                                "is_active": True,
                                "created_at": None,
                            })

        return {
            "total_rules": len(formatted),
            "rules": formatted,
            "source": "Supabase Cloud Database",
        }

    def add_medication_rule(
        self,
        pattern: str,
        rule_type: str,
        category: str,
        brand_name: Optional[str] = None,
        notes: Optional[str] = None,
        created_by: Optional[str] = "Pharmacist",
    ) -> Dict[str, Any]:
        """Add or update a medication classification rule in Supabase."""
        from refillcare.data.medication_classifier import add_classification_rule
        rt = "EXCLUDE" if rule_type.upper() in ("EXCLUDE", "EXCLUSION") else "CHRONIC"
        combined_notes = f"{category} | {notes or 'Pharmacist entry'}"
        return add_classification_rule(
            medication_name_or_keyword=pattern,
            rule_type=rt,
            notes=combined_notes,
        )

    def delete_medication_rule(self, rule_id: Union[int, str]) -> Dict[str, Any]:
        """Delete a medication classification rule by ID."""
        from refillcare.data.medication_classifier import delete_classification_rule
        return delete_classification_rule(rule_id=str(rule_id))

    def classify_medication_item(self, item_name: str, packing: Optional[str] = None) -> Dict[str, Any]:
        """Real-time clinical test classification of a medication item."""
        from refillcare.data.medication_classifier import classify_medication
        res = classify_medication(item_name, packing=packing)
        return {
            "item_name": item_name,
            "is_chronic_eligible": res.get("is_chronic_eligible", False),
            "category": res.get("category", "UNKNOWN"),
            "matched_rule": res.get("matched_rule"),
            "exclusion_reason": res.get("exclusion_reason"),
            "source": "RefillCare Clinical Engine & Supabase Rules",
        }


    # --------------------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------------------
    def get_active_purchase_history(self) -> pd.DataFrame:
        """The store's purchase history: its database (cached until rows change), else the legacy parquet."""
        from refillcare.data import history_store as hs

        if hs.history_count(self.db) > 0:
            return hs.load_all(self.db, with_history_columns=True)
        if self.paths.history_parquet.exists():
            return pd.read_parquet(self.paths.history_parquet)
        return pd.DataFrame()

    def _is_store_patient_phone(self, phone: Any) -> bool:
        from refillcare.data.identity import mobile10
        from database.models import SalesTransactionModel as Line

        m = mobile10(phone)
        if not m:
            return False
        if self.db.query(Line.id).filter(Line.mobile_no.like(f"%{m}")).first() is not None:
            return True
        hist = self.get_active_purchase_history()
        return bool(not hist.empty and "MOBILE_NO" in hist.columns and hist["MOBILE_NO"].map(mobile10).eq(m).any())

    def get_active_model_bundle(self) -> Optional[Dict[str, Any]]:
        """Production model bundle (None for the baseline policy)."""
        from refillcare.ml.model_provider import ModelProvider

        provider = ModelProvider.from_registry(self.db)
        return provider.production.bundle if provider.production else None


# --------------------------------------------------------------------------
    # 10. WHATSAPP & XINNO MESSAGING GATEWAY SERVICES
    # --------------------------------------------------------------------------
    def _outreach_bundles(self, target_date: Optional[date] = None, target_month: Optional[str] = None,
                          tier: str = "ALL", sync_window: int = 0):
        """Med-Sync bundles for a day / month (ONE builder for the candidate list, send summary and send)."""
        from refillcare.engine.med_sync import MedSyncEngine
        from refillcare.engine.persistence import RefillPersistenceManager

        target_month = validate_month_key(target_month, allow_all=True)
        pm = self._pm()
        if target_date:
            queue = pm.get_today_review_queue(self.db, target_date=target_date)
            target_label = target_date.strftime("%d %B %Y")
            target_date_str = target_date.strftime("%Y-%m-%d")
            target_month_str = None
        elif target_month and target_month != "ALL":
            queue = pm.get_monthly_review_queue(self.db, target_month=target_month)
            try:
                sel_y, sel_m = map(int, target_month.split("-"))
                target_label = date(sel_y, sel_m, 1).strftime("%B %Y")
            except Exception:
                target_label = target_month
            target_date_str = None
            target_month_str = target_month
        else:
            dt = self.forecast_as_of()
            queue = pm.get_today_review_queue(self.db, target_date=dt)
            target_label = dt.strftime("%d %B %Y")
            target_date_str = dt.strftime("%Y-%m-%d")
            target_month_str = None

        raw_records = []
        for q in queue:
            i_name = str(q.get("item_name") or "")
            # queue rows already passed the engine-consistent chronic gate
            raw_records.append({
                "reminder_id": q.get("reminder_id"),
                "customer_id": q.get("customer_id", ""),
                "customer_name": q.get("customer_name", "Valued Patient"),
                "mobile_no": q.get("phone_number"),
                "item_id": q.get("item_id", ""),
                "item_name": i_name,
                "is_eligible": True,
                "stability_tier": q.get("stability_tier", "MEDIUM-SAFE"),
                "predicted_interval_days": q.get("predicted_interval_days") or q.get("estimated_days_of_supply"),
                "last_purchase_date": q.get("last_purchase_date"),
                "expected_refill_date": q.get("expected_refill_date") or q.get("target_send_date"),
                "target_send_date": q.get("target_send_date"),
                "stage_offset": q.get("stage_offset", 0),
                "dos_days": q.get("estimated_days_of_supply"),
            })

        engine = MedSyncEngine(sync_window_days=sync_window, default_pharmacy_name=self.store.store_name)
        all_bundles = engine.cluster_patient_decisions(raw_records, sync_window_days=sync_window, pharmacy_name=self.store.store_name)

        # Filter by tier / refill function
        raw_tier = tier.split(" ")[0].upper()
        if raw_tier == "DUE":
            filtered_b = [b for b in all_bundles if getattr(b, "lifecycle_tier", "DUE") == "DUE" or getattr(b, "refill_function", "DUE_REFILL") == "DUE_REFILL"]
        elif raw_tier == "FOLLOWUP":
            filtered_b = [b for b in all_bundles if getattr(b, "lifecycle_tier", "DUE") == "FOLLOWUP" or getattr(b, "refill_function", "DUE_REFILL") == "REFILL_FOLLOW_UP"]
        elif raw_tier == "LAPSED":
            filtered_b = [b for b in all_bundles if getattr(b, "lifecycle_tier", "DUE") == "LAPSED" or getattr(b, "refill_function", "DUE_REFILL") == "LAPSED_REENGAGEMENT"]
        else:
            filtered_b = all_bundles

        return filtered_b, target_label, target_date_str, target_month_str

    def get_whatsapp_outreach_candidates(
        self,
        target_date: Optional[date] = None,
        target_month: Optional[str] = None,
        tier: str = "ALL",
        sync_window: int = 0,
        mobile_status: str = "All",
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load and cluster candidates for WhatsApp outreach matching Streamlit live gateway."""
        from refillcare.data.monthly_ingestion import determine_mobile_status

        filtered_b, target_label, target_date_str, target_month_str = self._outreach_bundles(
            target_date, target_month, tier, sync_window)

        from refillcare.whatsapp.consent import load_opt_out_index

        opt_outs = load_opt_out_index(self.db)
        candidate_records = []
        for b in filtered_b:
            phone_val = str(b.mobile_no or "").strip()
            if opt_outs.blocks(getattr(b, "customer_id", None), phone_val):
                continue

            meds_list = [item.item_name for item in b.synced_items] if b.synced_items else [b.anchor_item_name]
            mob_stat = determine_mobile_status(phone_val)
            tier_val = getattr(b, "lifecycle_tier", "DUE")

            # Mobile status filter
            if mobile_status == "Valid" and mob_stat != "Valid":
                continue
            elif mobile_status == "Missing" and mob_stat == "Valid":
                continue

            # Search filter
            if search:
                s_term = search.strip().lower()
                s_digits = "".join(filter(str.isdigit, s_term))
                c_name = str(b.customer_name or "").lower()
                m_str = ", ".join(meds_list).lower()
                p_digits = "".join(filter(str.isdigit, phone_val))

                matches = (s_term in c_name) or (s_term in m_str) or (s_digits and s_digits in p_digits)
                if not matches:
                    continue

            cand_target_send = target_date_str or (b.anchor_refill_date.strftime("%Y-%m-%d") if b.anchor_refill_date else "")
            cand_refill_date = b.anchor_refill_date.strftime("%Y-%m-%d") if b.anchor_refill_date else ""

            candidate_records.append({
                "reminder_id": b.bundle_id,
                "customer_name": b.customer_name or "Valued Customer",
                "phone_number": phone_val,
                "mobile_display": format_display_phone_10digits(phone_val),
                "mobile_status": mob_stat,
                "medication": ", ".join(meds_list),
                "expected_refill_date": cand_refill_date,
                "target_send_date": cand_target_send,
                "stage_offset": 0 if tier_val == "DUE" else (5 if tier_val == "FOLLOWUP" else 45),
                "tier": tier_val,
                "stage_display": "DUE" if tier_val == "DUE" else ("FOLLOW" if tier_val == "FOLLOWUP" else "LAPSED"),
                "refill_function": getattr(b, "refill_function", "DUE_REFILL"),
                "status": "PENDING",
                "items_count": len(b.synced_items) if b.synced_items else 1,
            })

        valid_count = sum(1 for r in candidate_records if r["mobile_status"] == "Valid")
        missing_count = len(candidate_records) - valid_count

        return {
            "target_date": target_date_str,
            "target_month": target_month_str,
            "target_label": target_label,
            "total_scheduled": len(candidate_records),
            "delivery_ready": valid_count,
            "missing_mobile": missing_count,
            "candidates": candidate_records,
        }

    def preview_whatsapp_message(
        self,
        customer_name: str,
        medications: Any,
        expected_refill_date: Any,
        tier: Optional[str] = "DUE",
        stage_offset: Optional[int] = 0,
    ) -> Dict[str, Any]:
        """Render high-contrast formatted WhatsApp template preview."""
        from refillcare.whatsapp.client import RefillWhatsAppClient
        client = RefillWhatsAppClient(store_name=self.store.store_name, store_contact=self.store.store_contact or None)
        return client.preview_message(
            customer_name=customer_name,
            medications=medications,
            expected_refill_date=expected_refill_date,
            tier=tier,
            stage_offset=stage_offset,
        )

    def send_single_whatsapp_reminder(
        self,
        phone_number: str,
        customer_name: str,
        medications: Any,
        expected_refill_date: Any,
        tier: Optional[str] = "DUE",
        stage_offset: Optional[int] = 0,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Dispatch a single test WhatsApp reminder in dry-run or live mode.

        A live send only goes to a phone that belongs to a patient of this store (no arbitrary numbers).
        """
        from refillcare.whatsapp.client import RefillWhatsAppClient
        client = RefillWhatsAppClient(store_name=self.store.store_name, store_contact=self.store.store_contact or None)
        from refillcare.whatsapp.consent import is_opted_out

        if not dry_run and not self._is_store_patient_phone(phone_number):
            return {
                "success": False,
                "message": "Cannot send live: this phone number is not a patient of this store.",
                "status_code": 403,
                "response_payload": {"skipped": True, "reason": "UNKNOWN_RECIPIENT"},
            }

        if is_opted_out(self.db, phone=phone_number):
            return {
                "success": False,
                "message": f"Cannot send: Customer phone {phone_number} has opted out of WhatsApp reminders.",
                "status_code": 400,
                "response_payload": {"skipped": True, "reason": "OPTED_OUT"},
            }

        res = client.send_refill_reminder(
            phone_number=phone_number,
            customer_name=customer_name,
            medications=medications,
            expected_refill_date=expected_refill_date,
            tier=tier,
            stage_offset=stage_offset,
            dry_run=dry_run,
        )
        return {
            "success": bool(res.get("success", False)),
            "message": res.get("message", "Dispatch executed"),
            "status_code": res.get("status_code", 200),
            "response_payload": res,
        }

    def set_customer_opt_in(self, customer_id: str, opt_in: bool) -> Dict[str, Any]:
        """Manually toggle a patient's WhatsApp opt-in consent status."""
        from refillcare.whatsapp.consent import set_opt_in

        customer = set_opt_in(self.db, customer_id, opt_in)
        return {
            "customer_id": customer_id,
            "opt_in_whatsapp": customer.opt_in_whatsapp,
            "status": "OPTED_IN" if opt_in else "OPTED_OUT",
            "message": f"Customer '{customer_id}' WhatsApp status set to {'OPTED_IN' if opt_in else 'OPTED_OUT'}."
        }

    def handle_inbound_whatsapp_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming WhatsApp webhook payload from Xinno or Meta cloud API.
        Automatically sets opt_in_whatsapp = False on STOP/UNSUBSCRIBE keywords.
        """
        sender_phone = None
        message_body = None

        # Format A: Meta Cloud / Xinno standard webhook format
        if "entry" in payload and isinstance(payload["entry"], list):
            for entry in payload["entry"]:
                for change in entry.get("changes", []):
                    val = change.get("value", {})
                    for msg in val.get("messages", []):
                        sender_phone = msg.get("from")
                        if "text" in msg and isinstance(msg["text"], dict):
                            message_body = msg["text"].get("body")
                        elif "body" in msg:
                            message_body = msg.get("body")

        # Format B: Direct CPaaS flat webhook format
        if not sender_phone:
            sender_phone = payload.get("from") or payload.get("phone") or payload.get("sender") or payload.get("mobile")
        if not message_body:
            message_body = payload.get("message") or payload.get("text") or payload.get("body")
            if isinstance(message_body, dict):
                message_body = message_body.get("text") or message_body.get("body")

        if not sender_phone or not message_body:
            return {
                "status": "ignored",
                "action": "none",
                "message": "Webhook payload does not contain an identifiable sender phone and text message.",
            }

        from refillcare.whatsapp.consent import CONSENT_STOP, parse_consent_keyword, set_opt_in_by_phone

        keyword = parse_consent_keyword(message_body)
        if keyword is None:
            return {
                "status": "received",
                "action": "message_logged",
                "message": f"Inbound message received from {sender_phone}, no opt-out/opt-in keyword detected.",
                "phone_number": str(sender_phone),
            }

        is_stop = keyword == CONSENT_STOP
        matched_cids = set_opt_in_by_phone(self.db, str(sender_phone), opt_in=not is_stop)

        action_name = "opted_out" if is_stop else "opted_in"
        desc = "opted out of" if is_stop else "opted into"
        return {
            "status": "success",
            "action": action_name,
            "message": f"Phone {sender_phone} ({', '.join(matched_cids)}) successfully {desc} WhatsApp reminders.",
            "customer_id": matched_cids[0] if matched_cids else None,
            "phone_number": str(sender_phone),
        }


if __name__ == "__main__":
    from database.connection import SessionLocal
    with SessionLocal() as db_session:
        service = EnterpriseServices(db_session)
        print("[SUCCESS] EnterpriseServices initialized successfully.")
        kpis = service.get_operations_kpi()
        print("[INFO] Operations KPIs:", kpis)
        as_of = service.forecast_as_of()
        reminders_today = service.get_daily_reminders(target_date=as_of)
        print(f"[INFO] Daily reminders on {as_of}: {len(reminders_today)} records.")
