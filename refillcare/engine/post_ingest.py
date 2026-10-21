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


def _affected_rows(result: Dict[str, Any], updated_df: pd.DataFrame) -> pd.DataFrame:
    added = result.get("records_appended_df")
    if added is None or added.empty:
        added = result.get("valid_records_df")
    if added is None or added.empty:
        return updated_df.iloc[0:0]
    keys = set(zip(added["customerId"].astype(str).str.strip(), added["itemId"].astype(str).str.strip()))
    cid = updated_df["customerId"].astype(str).str.strip()
    iid = updated_df["itemId"].astype(str).str.strip()
    return updated_df[[k in keys for k in zip(cid, iid)]]


def sync_after_ingest(db: Session, result: Dict[str, Any]) -> Dict[str, Any]:
    """Incremental re-scoring + cycle sync for one ingested batch. Returns the online-sync metrics."""
    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
    from refillcare.ml.model_provider import ModelProvider
    from refillcare.ml.outcomes import refresh_outcomes

    m = result["metrics"]
    updated_df = result.get("updated_history_df")
    latest = latest_sales_date_from_metrics(m)
    eval_as_of = latest + timedelta(days=1)  # T_max + 1
    if updated_df is None or updated_df.empty or m.get("new_records_added", 0) <= 0:
        return {}

    provider = ModelProvider.from_registry(db)
    engine = UnifiedRefillDecisionEngine(model_provider=provider)
    _, decisions = engine.evaluate_all_customer_items(_affected_rows(result, updated_df), as_of_date=eval_as_of)

    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, decisions)
    sync_res = pm.sync_reminder_cycles(db, decisions)
    expired = pm.expire_past_due_stages(db, today=eval_as_of)
    outcome_counts = refresh_outcomes(db, updated_df, as_of=latest)
    logger.info(
        "Batch %s: re-scored %d customer-items, %d cycles created, %d superseded.",
        result.get("import_batch_id"), len(decisions), sync_res.get("cycles_created", 0), sync_res.get("cycles_superseded", 0),
    )
    return {
        "evaluation_as_of_date": str(eval_as_of),
        "target_prediction_month": eval_as_of.strftime("%Y-%m"),
        "serving_model": provider.describe(),
        "customer_items_rescored": len(decisions),
        "decisions_evaluated": len(decisions),
        "cycles_created": sync_res.get("cycles_created", 0),
        "cycles_superseded": sync_res.get("cycles_superseded", 0),
        "stages_created": sync_res.get("stages_created", 0),
        "stages_superseded": sync_res.get("stages_superseded", 0),
        "stages_expired": expired,
        "outcomes_updated": outcome_counts.get("evaluated", 0),
    }


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

    batch = db.query(ImportBatchModel).filter_by(import_batch_id=batch_id).first()
    if batch is not None:
        batch.processing_status = "ROLLED_BACK"
        batch.is_active = False
        db.commit()
    return db_revert
