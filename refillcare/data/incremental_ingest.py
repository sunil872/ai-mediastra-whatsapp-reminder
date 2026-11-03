"""Incremental sales upload: append only the new lines, re-score only the patients in the file.

Shared by FastAPI and Streamlit. One upload (1 day, a week, a month or 3 years of onboarding):

1. Reject a file already ingested (same SHA-256) unless ``force``.
2. Map columns (saved / auto / user mapping) -> keep the 11 RefillCare fields -> validate
   (dates, channel, quantities, implausible dates) -> net returns against the original sale.
3. In ONE transaction, under a per-store lock: record the batch, append the lines that are not
   stored yet (line-level keys: a partial day followed by the full day adds only the missing lines).
4. Re-score only the patients in the file (all their medicines, keyed by the store's patient /
   therapy settings), sync reminder cycles, refresh their outcomes.
5. Mark the batch SYNCED - or SYNC_FAILED with the error, so the worker retries it.

Nothing rewrites the whole history, and no file backups are made: the database is the backup
target (RDS automated snapshots / SQLite online backup).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.models import ImportBatchModel, StoreCacheModel
from refillcare.data import history_store as hs

logger = logging.getLogger("refillcare.ingest")
SALT_PRIORS_KEY = "salt_priors"
SALT_PRIORS_MAX_AGE_DAYS = 30
_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


class DuplicateUpload(ValueError):
    """The same file (checksum) was already ingested for this store."""


class ColumnMappingRequired(ValueError):
    def __init__(self, message: str, column_mapping: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.column_mapping = column_mapping


@contextmanager
def store_write_lock(db: Session) -> Iterator[None]:
    """One upload at a time per store (thread lock here; PostgreSQL advisory lock across containers)."""
    from refillcare.tenancy import current_store_id

    sid = current_store_id()
    with _locks[sid]:
        if db.bind is not None and db.bind.dialect.name == "postgresql":
            db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": f"refillcare-ingest:{sid}"})
        yield


def _new_batch_id() -> str:
    return f"BATCH_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def file_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gap_warnings(db: Session, new_min: Optional[date], new_max: Optional[date], prev_max: Optional[date]) -> Dict[str, Any]:
    """Missing sales days between the stored history and this file, and inside the file."""
    out: Dict[str, Any] = {}
    if new_min is None or new_max is None:
        return out
    if prev_max is not None and new_min > prev_max + timedelta(days=1):
        missing = (new_min - prev_max).days - 1
        out["gap_before_file_days"] = missing
        out["gap_before_file"] = f"{(prev_max + timedelta(days=1)).isoformat()} to {(new_min - timedelta(days=1)).isoformat()}"
    span = pd.date_range(new_min, new_max, freq="D").date
    have = hs.distinct_days(db, new_min, new_max)
    holes = [d.isoformat() for d in span if d not in have]
    if holes:
        out["days_without_sales_in_file_range"] = holes[:31]
    if prev_max is not None and new_max < prev_max:
        out["backdated_file"] = True
    return out


def get_population_stats(db: Session, force_refresh: bool = False):
    """Salt priors for the store, cached (refreshed monthly; not rebuilt on every upload)."""
    from refillcare.features.context import PopulationStats
    from refillcare.ml.evaluation import prepare_purchase_events

    row = db.get(StoreCacheModel, SALT_PRIORS_KEY)
    if row is not None and not force_refresh and (datetime.utcnow() - row.updated_at).days < SALT_PRIORS_MAX_AGE_DAYS:
        return PopulationStats.from_dict(row.value)
    hist = hs.load_all(db)
    _, latest = hs.date_bounds(db)
    stats = PopulationStats.from_events(prepare_purchase_events(hist), as_of=latest) if not hist.empty else PopulationStats.from_dict(None)
    value = stats.to_dict()
    if row is None:
        db.add(StoreCacheModel(cache_key=SALT_PRIORS_KEY, value=value, updated_at=datetime.utcnow()))
    else:
        row.value, row.updated_at = value, datetime.utcnow()
    db.commit()
    return stats


def ingest_sales_batch(
    db: Session,
    sales_df: pd.DataFrame,
    source_filename: str,
    column_mapping: Optional[Dict[str, Optional[str]]] = None,
    checksum: Optional[str] = None,
    actor: str = "system",
    force: bool = False,
    sync: bool = True,
) -> Dict[str, Any]:
    from refillcare.data.sales_ingestion import _enrich_with_catalog_salt, validate_monthly_sales_data
    from refillcare.stores import store_setting

    hs.ensure_history_in_db(db)
    if checksum and not force:
        dup = (db.query(ImportBatchModel)
               .filter(ImportBatchModel.checksum_sha256 == checksum, ImportBatchModel.is_active.is_(True)).first())
        if dup is not None:
            raise DuplicateUpload(f"This file was already uploaded as batch {dup.import_batch_id} on {dup.upload_timestamp:%d-%m-%Y %H:%M}.")

    valid_df, review_df, metrics = validate_monthly_sales_data(sales_df, None, column_mapping)
    if metrics.get("needs_column_mapping"):
        raise ColumnMappingRequired(metrics.get("error", "Column mapping needs confirmation."), metrics.get("column_mapping"))
    if metrics.get("error"):
        raise ValueError(metrics["error"])
    diag = metrics.get("date_diagnostics") or {}
    if diag.get("is_ambiguous") and int(metrics.get("valid_records", 0)) == 0:
        raise ValueError(f"Dates could not be read: {diag.get('warning') or 'unknown format'}")
    if diag.get("is_ambiguous"):
        raise ValueError(f"Upload blocked - {diag.get('warning')}")

    mapping = ((metrics.get("column_mapping") or {}).get("mapping")) or {}
    warnings: Dict[str, Any] = {}
    if not mapping.get("invoice_number"):
        if (store_setting("REFILLCARE_REQUIRE_INVOICE_NUMBER", "false") or "").lower() in ("1", "true", "yes", "on"):
            raise ValueError("This store requires an invoice-number column (B2B transfers cannot be separated without it).")
        warnings["placeholder_invoice_numbers"] = "No invoice-number column: every row is treated as a customer sale."

    batch_id = _new_batch_id()
    if not valid_df.empty and valid_df["invoice_number"].astype(str).str.startswith("INV_").all():
        valid_df["invoice_number"] = [f"INV-{batch_id[-6:]}-{i + 1}" for i in range(len(valid_df))]

    returns = pd.DataFrame()
    if not review_df.empty and "quantity" in review_df.columns:
        q = pd.to_numeric(review_df["quantity"], errors="coerce")
        returns = review_df[(q < 0) & review_df.get("customerId", pd.Series(index=review_df.index)).notna()]

    valid_df = _enrich_with_catalog_salt(valid_df) if not valid_df.empty else valid_df
    dts = pd.to_datetime(valid_df["invoice_date"], errors="coerce") if not valid_df.empty else pd.Series(dtype="datetime64[ns]")
    new_min = dts.min().date() if not dts.dropna().empty else None
    new_max = dts.max().date() if not dts.dropna().empty else None

    with store_write_lock(db):
        _, prev_max = hs.date_bounds(db)
        cust_ids = valid_df["customerId"].dropna().astype(str).str.strip().unique().tolist() if not valid_df.empty else []
        item_ids = valid_df["itemId"].dropna().astype(str).str.strip().unique().tolist() if not valid_df.empty else []
        known_c, known_i = hs.known_customers(db, cust_ids), hs.known_items(db, item_ids)
        batch = ImportBatchModel(
            import_batch_id=batch_id, source_filename=source_filename, record_count=int(metrics.get("records_received", 0)),
            records_requiring_review=int(len(review_df)), processing_status="ACTIVE", is_active=True,
            checksum_sha256=checksum, detected_date_range=metrics.get("file_date_range", "-"),
            source_format="Auto", sync_status="PENDING", min_sales_date=new_min, max_sales_date=new_max,
        )
        db.add(batch)
        db.flush()
        counts = hs.append_lines(db, valid_df, batch_id)
        netting = hs.net_returns(db, returns) if not returns.empty else {}
        warnings.update(_gap_warnings(db, new_min, new_max, prev_max))
        if netting.get("returns_netted"):
            warnings["returns_netted"] = netting
        batch.records_inserted = counts["inserted"]
        batch.records_skipped = counts["skipped"]
        batch.data_warnings = warnings or None
        db.commit()
    hs.clear_cache()

    _, latest = hs.date_bounds(db)
    result: Dict[str, Any] = {
        "status": "success",
        "import_batch_id": batch_id,
        "metrics": {
            "import_batch_id": batch_id,
            "new_records_added": counts["inserted"],
            "existing_duplicates_skipped": counts["skipped"],
            "customers_updated": len(set(cust_ids) & known_c),
            "new_customers": len(set(cust_ids) - known_c),
            "new_medicines": len(set(item_ids) - known_i),
            "latest_sales_date": latest.strftime("%d-%m-%Y") if latest else "N/A",
            "records_requiring_review": int(len(review_df)),
            "detected_date_range": metrics.get("file_date_range", "-"),
            "file_date_range": metrics.get("file_date_range", "-"),
            "ignored_columns": metrics.get("ignored_columns", []),
        },
        "data_warnings": warnings,
        "column_mapping": metrics.get("column_mapping"),
        "review_records_df": review_df,
    }
    if sync and counts["inserted"] > 0:
        result["online_sync"] = sync_batch(db, batch_id)
    elif counts["inserted"] == 0:
        batch = db.get(ImportBatchModel, batch_id)
        batch.sync_status = "SYNCED"
        db.commit()
        result["online_sync"] = {"customer_items_rescored": 0, "note": "no new sales lines in this file"}
    return result


def sync_batch(db: Session, batch_id: str) -> Dict[str, Any]:
    """Re-score the patients of one batch and sync their reminder cycles; records SYNCED / SYNC_FAILED."""
    batch = db.get(ImportBatchModel, batch_id)
    if batch is None:
        raise LookupError(f"Batch {batch_id} not found")
    try:
        lines = hs.batch_lines(db, batch_id)
        res = rescore_customers(db, lines["customerId"].dropna().astype(str).unique().tolist(), raw_lines=lines)
        batch = db.get(ImportBatchModel, batch_id)
        batch.sync_status, batch.sync_error = "SYNCED", None
        db.commit()
        return res
    except Exception as e:
        db.rollback()
        batch = db.get(ImportBatchModel, batch_id)
        batch.sync_status, batch.sync_error = "SYNC_FAILED", repr(e)[:2000]
        db.commit()
        logger.exception("Post-upload sync failed for %s", batch_id)
        try:
            from refillcare.jobs.queue import enqueue

            enqueue("sync_batch", {"batch_id": batch_id}, dedupe_key=f"sync:{batch_id}")
        except Exception:  # queue unavailable: the batch stays SYNC_FAILED and is visible in the UI
            logger.warning("Could not queue a sync retry for %s", batch_id)
        return {"error": repr(e), "sync_status": "SYNC_FAILED"}


def rescore_customers(db: Session, customer_ids: List[str], raw_lines: Optional[pd.DataFrame] = None,
                      as_of: Optional[date] = None) -> Dict[str, Any]:
    """Re-score every medicine of the given (raw) customers from the stored history."""
    from refillcare.data.identity import BRAND_ITEM_COL, SOURCE_CUSTOMER_COL, normalise_trajectory_keys
    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
    from refillcare.ml.model_provider import ModelProvider
    from refillcare.ml.outcomes import refresh_outcomes

    patients = hs.load_customers(db, customer_ids)
    if patients.empty:
        return {"customer_items_rescored": 0}
    keyed = normalise_trajectory_keys(patients)
    if raw_lines is not None and not raw_lines.empty:
        raw = set(zip(raw_lines["customerId"].astype(str).str.strip(), raw_lines["itemId"].astype(str).str.strip()))
        src = zip(keyed[SOURCE_CUSTOMER_COL].astype(str).str.strip(), keyed[BRAND_ITEM_COL].astype(str).str.strip())
        mask = [k in raw for k in src]
        keys = set(zip(keyed.loc[mask, "customerId"].astype(str).str.strip(), keyed.loc[mask, "itemId"].astype(str).str.strip()))
    else:
        keys = set(zip(keyed["customerId"].astype(str).str.strip(), keyed["itemId"].astype(str).str.strip()))
    _, latest = hs.date_bounds(db)
    eval_as_of = as_of or ((latest + timedelta(days=1)) if latest else date.today())
    provider = ModelProvider.from_registry(db)
    engine = UnifiedRefillDecisionEngine(model_provider=provider, population_stats=get_population_stats(db))
    _, decisions = engine.evaluate_all_customer_items(keyed, as_of_date=eval_as_of, only_keys=keys)
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, decisions)
    sync_res = pm.sync_reminder_cycles(db, decisions)
    expired = pm.expire_past_due_stages(db, today=eval_as_of)
    outcomes = refresh_outcomes(db, keyed, as_of=latest, only_keys={f"{c}_{i}" for c, i in keys})
    return {
        "evaluation_as_of_date": str(eval_as_of),
        "serving_model": provider.describe(),
        "customer_items_rescored": len(decisions),
        "decisions_evaluated": len(decisions),
        "cycles_created": sync_res.get("cycles_created", 0),
        "cycles_superseded": sync_res.get("cycles_superseded", 0),
        "stages_created": sync_res.get("stages_created", 0),
        "stages_superseded": sync_res.get("stages_superseded", 0),
        "stages_expired": expired,
        "outcomes_updated": outcomes.get("evaluated", 0),
    }


def rollback_batch(db: Session, batch_id: str) -> Dict[str, Any]:
    """Undo an upload: delete its lines, revert its reminder effects, re-score its patients.

    Uses the store's current patient / therapy keys (name+mobile, salt groups) so the reminders that
    the batch created are the ones reverted.
    """
    from refillcare.data.identity import normalise_trajectory_keys
    from refillcare.engine.persistence import RefillPersistenceManager

    batch = db.get(ImportBatchModel, batch_id)
    if batch is None:
        raise LookupError(f"Batch {batch_id} not found")
    if not batch.is_active:
        raise ValueError(f"Batch {batch_id} is already rolled back.")
    with store_write_lock(db):
        lines = hs.batch_lines(db, batch_id)
        customers = lines["customerId"].dropna().astype(str).unique().tolist() if not lines.empty else []
        removed: Dict[str, set] = {}
        if not lines.empty:
            # Keys of the removed lines as the engine sees them (identity / salt grouping applied on the patients' history)
            patients = normalise_trajectory_keys(hs.load_customers(db, customers))
            ids = set(lines["invoice_date"].dt.date.astype(str) + "|" + lines["customerId"].astype(str) + "|" + lines["itemId"].astype(str))
            from refillcare.data.identity import BRAND_ITEM_COL, SOURCE_CUSTOMER_COL

            pkey = patients["invoice_date"].dt.date.astype(str) + "|" + patients[SOURCE_CUSTOMER_COL].astype(str) + "|" + patients[BRAND_ITEM_COL].astype(str)
            for c, i, d, k in zip(patients["customerId"].astype(str), patients["itemId"].astype(str), patients["invoice_date"].dt.date, pkey):
                if k in ids:
                    removed.setdefault(f"{c}_{i}", set()).add(d)
        n = hs.delete_batch(db, batch_id)
        batch.processing_status, batch.is_active = "ROLLED_BACK", False
        db.commit()
    hs.clear_cache()
    pm = RefillPersistenceManager()
    reverted = pm.revert_rolled_back_purchases(db, removed) if removed else {}
    _, latest = hs.date_bounds(db)
    reverted["stages_expired"] = pm.expire_past_due_stages(db, today=(latest + timedelta(days=1)) if latest else date.today())
    # Only the medicines the batch touched (not every medicine of these patients - a common name
    # can cover thousands of trajectories under the name-only identity)
    rescored = rescore_customers(db, customers, raw_lines=lines) if customers else {}
    return {"status": "success", "rolled_back_batch_id": batch_id, "records_removed": n,
            "restored_latest_sales_date": latest.strftime("%d-%m-%Y") if latest else "N/A",
            "reminder_state_reverted": reverted, "rescored": rescored}


def data_status(db: Session, today: Optional[date] = None) -> Dict[str, Any]:
    """How current the store's sales data is (used to block sending on stale data)."""
    from refillcare.stores import store_setting

    _, latest = hs.date_bounds(db)
    if latest is None:  # history still only in the legacy parquet (not imported yet)
        from refillcare.ml.service import load_history

        legacy = load_history()
        if not legacy.empty and "invoice_date" in legacy.columns:
            mx = pd.to_datetime(legacy["invoice_date"], errors="coerce").max()
            latest = None if pd.isna(mx) else mx.date()
    today = today or date.today()
    max_lag = int(store_setting("REFILLCARE_MAX_DATA_LAG_DAYS", "2") or 2)
    lag = (today - latest).days if latest else None
    failed = db.query(ImportBatchModel).filter(ImportBatchModel.sync_status == "SYNC_FAILED", ImportBatchModel.is_active.is_(True)).count()
    return {"latest_sales_date": latest.isoformat() if latest else None, "days_behind": lag, "max_allowed_lag_days": max_lag,
            "stale": lag is None or lag > max_lag, "batches_failed_to_sync": failed,
            "ok_to_send": bool(lag is not None and lag <= max_lag and failed == 0)}
