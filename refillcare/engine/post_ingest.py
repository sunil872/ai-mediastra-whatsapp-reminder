"""What happens after a sales file is ingested - shared by FastAPI and Streamlit.

1. mirror the import batch into ``enterprise.db`` (so batch history / rollback see every upload)
2. re-score only the customer-items touched by the upload with the registry's serving model
3. sync reminder cycles, expire past-due stages, refresh live outcomes
4. re-plan the ML train / validation / test windows on the new data (is a retrain due?)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.models import ImportBatchModel

logger = logging.getLogger("refillcare.engine.post_ingest")


def latest_sales_date_from_metrics(metrics: Dict[str, Any]) -> date:
    v = metrics.get("latest_sales_date")
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
    return date.today()


def mirror_import_batch(db: Session, result: Dict[str, Any], filename: str, checksum: Optional[str] = None) -> None:
    batch_id = result["import_batch_id"]
    if db.query(ImportBatchModel).filter_by(import_batch_id=batch_id).first() is not None:
        return
    m = result["metrics"]
    db.add(ImportBatchModel(
        import_batch_id=batch_id,
        source_filename=filename,
        detected_date_range=m.get("file_date_range", ""),
        source_format=m.get("source_format", "Auto"),
        record_count=m.get("records_received", 0),
        records_inserted=m.get("new_records_added", 0),
        records_skipped=m.get("existing_duplicates_skipped", 0),
        records_requiring_review=m.get("records_requiring_review", 0),
        processing_status="ACTIVE",
        checksum_sha256=checksum,
        is_active=True,
    ))
    db.commit()


def ml_freshness_after_ingest(db: Session, history_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Re-plan the ML windows on the updated data; informational only, never fails an ingest."""
    if history_df is None or history_df.empty:
        return None
    try:
        from refillcare.ml.service import data_freshness

        return data_freshness(db, history_df)
    except Exception as e:
        logger.warning("ML freshness check after ingest failed: %s", e)
        return None


def revert_after_rollback(db: Session, rollback_result: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    """Undo the reminder-state effects of a rolled-back batch (shared by FastAPI and Streamlit).

    Reverts decisions / cycles / stages created from the removed purchases, expires stages that
    are now past due and marks the SQL batch row ROLLED_BACK.
    """
    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.engine.rekey import retire_stale_decisions

    db_revert: Dict[str, Any] = {}
    removed_df = rollback_result.get("removed_records_df")
    if removed_df is not None and not removed_df.empty:
        removed: Dict[str, set] = {}
        r_dates = pd.to_datetime(removed_df["invoice_date"], errors="coerce").dt.date
        for c, i, d in zip(removed_df["customerId"].astype(str).str.strip(), removed_df["itemId"].astype(str).str.strip(), r_dates):
            if d is not None and not pd.isna(d):
                removed.setdefault(f"{c}_{i}", set()).add(d)
        pm = RefillPersistenceManager()
        db_revert = pm.revert_rolled_back_purchases(db, removed)
        restored = rollback_result.get("restored_latest_sales_date")
        today = latest_sales_date_from_metrics({"latest_sales_date": restored}) + timedelta(days=1) if restored else date.today()
        db_revert["stages_expired"] = pm.expire_past_due_stages(db, today=today)
        # Clean up any stale decisions left by the removed purchases (fallback consistency)
        retire_stale_decisions(db, only_keys=set(removed.keys()))

    batch = db.query(ImportBatchModel).filter_by(import_batch_id=batch_id).first()
    if batch is not None:
        batch.processing_status = "ROLLED_BACK"
        batch.is_active = False
        db.commit()
    return db_revert
