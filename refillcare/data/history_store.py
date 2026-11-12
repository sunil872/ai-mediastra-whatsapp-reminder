"""The store's sales history in its own database (``sales_transactions``), append-only.

Why: rewriting a master CSV + parquet on every upload costs time and storage that grows with the
whole history, and two writers lose each other's rows. Here an upload inserts only its new lines
(one row per sales line, deduplicated by ``line_key``), a rollback deletes its batch's rows, and
scoring reads only the patients in the upload. Backups are the database's own (RDS snapshots /
SQLite online backup) instead of full file copies.

The DataFrame shape returned by the loaders is the one the engine and ML code already use
(``customerId``, ``itemId``, ``invoice_date`` ...).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import SalesTransactionModel as Line

logger = logging.getLogger("refillcare.history")

# canonical DataFrame column -> table column
COLUMN_MAP: Dict[str, str] = {
    "invoice_date": "invoice_date", "invoice_number": "invoice_number", "customerName": "customer_name",
    "itemCode": "item_code", "itemName": "item_name", "packing": "packing", "quantity": "quantity",
    "freeQuantity": "free_quantity", "MOBILE_NO": "mobile_no", "customerId": "customer_id", "itemId": "item_id",
    "salt_composition": "salt_composition", "salt_category": "salt_category", "transaction_type": "transaction_type",
    "refillcare_eligible": "refillcare_eligible", "exclusion_reason": "exclusion_reason", "import_batch_id": "import_batch_id",
    "daysSupply": "days_supply",
}
_CHUNK = 900  # stays under SQLite's bound-parameter limit
_cache_lock = threading.Lock()
# (store, with_history_columns) -> (version, frame): the ML tab and the dashboard views keep separate
# entries instead of evicting each other on every switch
_full_cache: Dict[Tuple[str, bool], Tuple[Tuple, pd.DataFrame]] = {}
_patients_cache: Dict[Tuple, Any] = {}  # (store, history version, identity mode) -> patients monitored dict


def _norm_text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip().upper()


def compute_line_keys(df: pd.DataFrame) -> List[str]:
    """Stable id of a sales line: the same line in a re-uploaded or fuller file gets the same key.

    invoice + item + date + customer + quantity + free quantity + occurrence (the n-th identical line
    of that invoice). A partial day followed by the full day therefore adds only the missing lines,
    and two different invoices on the same day are both kept.
    """
    if df.empty:
        return []
    dates = pd.to_datetime(df["invoice_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    inv = df["invoice_number"].map(_norm_text) if "invoice_number" in df.columns else pd.Series("", index=df.index)
    # Placeholder invoices (no invoice column) carry no identity of their own
    inv = inv.where(~inv.str.startswith(("INV_", "INV-")), "")
    item = (df["itemCode"] if "itemCode" in df.columns else df["itemId"]).map(_norm_text)
    cust = df["customerName"].map(_norm_text) if "customerName" in df.columns else df["customerId"].map(_norm_text)
    qty = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(0).round(3).astype(str)
    free = pd.to_numeric(df["freeQuantity"], errors="coerce").fillna(0).round(3).astype(str) if "freeQuantity" in df.columns else pd.Series("0.0", index=df.index)
    base = inv + "|" + item + "|" + dates + "|" + cust + "|" + qty + "|" + free
    occ = base.groupby(base).cumcount().astype(str)
    return [hashlib.sha1(f"{b}|{o}".encode("utf-8")).hexdigest()[:24] for b, o in zip(base, occ)]


def _existing_keys(db: Session, keys: List[str]) -> set:
    found: set = set()
    for i in range(0, len(keys), _CHUNK):
        found.update(k for (k,) in db.execute(select(Line.line_key).where(Line.line_key.in_(keys[i:i + _CHUNK]))))
    return found


def append_lines(db: Session, df: pd.DataFrame, batch_id: str) -> Dict[str, int]:
    """Insert the lines not stored yet (caller commits). Returns inserted / skipped counts."""
    if df.empty:
        return {"inserted": 0, "skipped": 0}
    work = df.copy()
    work["line_key"] = compute_line_keys(work)
    work = work.drop_duplicates(subset=["line_key"])
    existing = _existing_keys(db, work["line_key"].tolist())
    new = work[~work["line_key"].isin(existing)]
    rows = []
    for rec in new.to_dict(orient="records"):
        row = {col: rec.get(canon) for canon, col in COLUMN_MAP.items() if canon in rec}
        row["import_batch_id"] = batch_id
        row["line_key"] = rec["line_key"]
        d = pd.to_datetime(row.get("invoice_date"), errors="coerce")
        row["invoice_date"] = None if pd.isna(d) else d.to_pydatetime()
        for k in ("quantity", "free_quantity"):
            v = row.get(k)
            row[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        row["quantity"] = row["quantity"] if row.get("quantity") is not None else 0.0
        row["transaction_type"] = row.get("transaction_type") or "CUSTOMER_SALE"
        row["refillcare_eligible"] = bool(row.get("refillcare_eligible", True)) if row.get("refillcare_eligible") is not None else True
        for k in ("customer_id", "item_id", "customer_name", "item_name", "item_code", "packing", "mobile_no", "invoice_number",
                  "salt_composition", "salt_category", "exclusion_reason"):
            v = row.get(k)
            row[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        rows.append(row)
    for i in range(0, len(rows), 2000):
        db.bulk_insert_mappings(Line, rows[i:i + 2000])
    return {"inserted": len(rows), "skipped": int(len(df) - len(rows))}


_SELECT_COLS = [getattr(Line, col) for col in COLUMN_MAP.values()]


def _to_frame(rows: Iterable) -> pd.DataFrame:
    """Rows are plain column tuples (``select(*_SELECT_COLS)``) - no ORM objects to build per line."""
    df = pd.DataFrame.from_records(list(rows), columns=list(COLUMN_MAP.keys()))
    if not df.empty:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df["refillcare_eligible"] = df["refillcare_eligible"].fillna(True).astype(bool)
    return df


def channel_counts(db: Session) -> Dict[str, int]:
    """Sales-line counts of the current store from its database (Streamlit overview and API share this).

    received = bill lines in the store's active uploads; stored = lines kept (every line of a bill, incl.
    repeat scans); not_stored = return lines (never stored as rows) + invalid rows + lines uploaded before.
    Return breakdown: return_lines received, returns_netted (subtracted from their sale within the return
    window), returns_unmatched, sales_fully_returned (= returned_corrections + returned_late, still stored
    as "not counted"), invalid_rows.
    """
    from database.models import ImportBatchModel

    by_type = dict(db.execute(select(Line.transaction_type, func.count(Line.id)).group_by(Line.transaction_type)).all())
    stored = int(sum(by_type.values()))
    eligible = int(db.execute(select(func.count(Line.id)).where(Line.refillcare_eligible.is_(True))).scalar() or 0)
    received = int(db.execute(select(func.sum(func.coalesce(ImportBatchModel.record_count, 0))).where(
        ImportBatchModel.is_active.is_(True))).scalar() or 0)
    received = max(received, stored)

    # Returns: never stored as their own rows; each is subtracted from the sale it follows (within the
    # return window). Counts per upload come from its import record; fully returned sales from the lines.
    import json

    return_lines = netted = review_rows = 0
    for warn, review in db.execute(select(ImportBatchModel.data_warnings, ImportBatchModel.records_requiring_review)
                                   .where(ImportBatchModel.is_active.is_(True))).all():
        w = json.loads(warn) if isinstance(warn, str) else (warn or {})
        rn = (w or {}).get("returns_netted") or {}
        return_lines += int(rn.get("returns_seen") or 0)
        netted += int(rn.get("returns_netted") or 0)
        review_rows += int(review or 0)
    corrections = int(db.execute(select(func.count(Line.id)).where(Line.exclusion_reason == RETURN_FULL_CORRECTION)).scalar() or 0)
    late = int(db.execute(select(func.count(Line.id)).where(Line.exclusion_reason.like(f"{RETURN_FULL_LATE}%"))).scalar() or 0)
    return {
        "received": received,
        "stored": stored,
        "not_stored": received - stored,
        "return_lines": return_lines,
        "returns_netted": netted,
        "returns_unmatched": max(0, return_lines - netted),
        "invalid_rows": max(0, review_rows - return_lines),
        "sales_fully_returned": corrections + late,
        "returned_corrections": corrections,
        "returned_late": late,
        "return_window_days": return_window_days(),
        "return_correction_days": return_correction_days(),
        "customer_sales": int(by_type.get("CUSTOMER_SALE", 0)),
        "b2b_inter_store": int(by_type.get("B2B_INTER_STORE", 0)),
        "unknown": int(stored - by_type.get("CUSTOMER_SALE", 0) - by_type.get("B2B_INTER_STORE", 0)),
        "refillcare_eligible": eligible,
    }


def store_summary(db: Session) -> Dict[str, Any]:
    """THE store summary every tab and the API show (one definition, counted from the database):

    first / latest sales date, sales lines stored, and active patients (last 12 months or available span)
    with all-time customers monitored as secondary information.
    """
    from datetime import timedelta
    from refillcare.data.identity import apply_patient_identity
    from refillcare.data.identity import identity_mode
    from refillcare.tenancy import current_store_id

    lo, hi = date_bounds(db)
    # Patients exactly as the engine keys them (name + mobile by default). The distinct (name, mobile)
    # scan takes ~20 s on ~1M lines, so it is cached until the store's sales lines change.
    key = (current_store_id(), history_version(db), identity_mode())
    with _cache_lock:
        cached_info = _patients_cache.get(key)
    if cached_info is None or not isinstance(cached_info, dict):
        span_days = (hi - lo).days if (lo and hi) else 0
        has_12m = span_days >= 365
        months_span = max(1, round(span_days / 30.4375)) if span_days > 0 else 0

        if has_12m:
            active_label = "Active patients (last 12 months)"
            cutoff = hi - timedelta(days=365)
        elif span_days < 30:
            active_label = f"Active patients ({max(1, span_days)} days history)"
            cutoff = lo
        else:
            active_label = f"Active patients (last {months_span} months)"
            cutoff = lo

        pairs = pd.DataFrame(db.execute(select(Line.customer_id, Line.mobile_no).where(
            Line.transaction_type == "CUSTOMER_SALE", Line.refillcare_eligible.is_(True)).distinct()).all(),
            columns=["customerId", "MOBILE_NO"])
        all_time_customers = int(apply_patient_identity(pairs)["customerId"].nunique()) if not pairs.empty else 0

        if cutoff and lo and cutoff > lo:
            pairs_act = pd.DataFrame(db.execute(select(Line.customer_id, Line.mobile_no).where(
                Line.transaction_type == "CUSTOMER_SALE", Line.refillcare_eligible.is_(True),
                Line.invoice_date >= cutoff).distinct()).all(),
                columns=["customerId", "MOBILE_NO"])
            active_customers = int(apply_patient_identity(pairs_act)["customerId"].nunique()) if not pairs_act.empty else 0
        else:
            active_customers = all_time_customers

        cached_info = {
            "all_time_customers": all_time_customers,
            "active_patients": active_customers,
            "active_patients_label": active_label,
            "has_12m_history": has_12m,
            "history_span_months": months_span,
            "history_span_days": span_days,
        }
        with _cache_lock:
            for k in [k for k in _patients_cache if k[0] == key[0]]:
                _patients_cache.pop(k, None)
            _patients_cache[key] = cached_info

    subtext = (
        f"All-time: {cached_info['all_time_customers']:,}"
        if cached_info.get("has_12m_history", True) else
        f"All-time: {cached_info['all_time_customers']:,} ({cached_info.get('history_span_months', 0)} mos history)"
    )

    return {
        "first_sales_date": lo.isoformat() if lo else None,
        "latest_sales_date": hi.isoformat() if hi else None,
        "sales_lines_stored": history_count(db),
        "customers_monitored": cached_info["active_patients"],
        "all_time_customers": cached_info["all_time_customers"],
        "active_patients": cached_info["active_patients"],
        "active_patients_label": cached_info["active_patients_label"],
        "active_patients_subtext": subtext,
        "has_12m_history": cached_info["has_12m_history"],
        "history_span_months": cached_info["history_span_months"],
        "history_span_days": cached_info["history_span_days"],
    }


def history_count(db: Session) -> int:
    return int(db.execute(select(func.count(Line.id))).scalar() or 0)


def history_version(db: Session) -> Tuple[int, int]:
    """Changes whenever rows are added or removed (cache key)."""
    n, mx = db.execute(select(func.count(Line.id), func.max(Line.id))).one()
    return int(n or 0), int(mx or 0)


def load_customers(db: Session, customer_ids: Iterable[str]) -> pd.DataFrame:
    """Every stored line of the given (raw) customer ids - all their medicines, for re-scoring."""
    ids = sorted({str(c).strip() for c in customer_ids if c is not None and str(c).strip()})
    parts = []
    for i in range(0, len(ids), _CHUNK):
        parts.append(_to_frame(db.execute(select(*_SELECT_COLS).where(Line.customer_id.in_(ids[i:i + _CHUNK])))))
    return pd.concat(parts, ignore_index=True) if parts else _to_frame([])


def load_all(db: Session, store_id: Optional[str] = None, with_history_columns: bool = False) -> pd.DataFrame:
    """The full history (cached per store until rows change). For dashboards, training and backtests."""
    from refillcare.tenancy import current_store_id

    sid = store_id or current_store_id()
    ver = history_version(db)
    key = (sid, bool(with_history_columns))
    with _cache_lock:
        hit = _full_cache.get(key)
        if hit and hit[0] == ver:
            return hit[1]
    # Fast path: the imported legacy history is also on disk as parquet (~0.3 s instead of a full table
    # scan); only the rows of later upload batches are read from the database. Used only while the
    # parquet still matches the database's legacy rows exactly (same count), so a rolled-back legacy
    # batch, another store or a replaced file can never serve the wrong history.
    from refillcare.store import StoreConfig
    from database.models import ImportBatchModel
    base_parquet = StoreConfig.from_env().history_parquet
    if base_parquet is not None and base_parquet.exists():
        try:
            legacy_active = db.execute(select(func.count()).select_from(ImportBatchModel).where(
                ImportBatchModel.import_batch_id == "LEGACY_HISTORY",
                ImportBatchModel.processing_status == "ACTIVE")).scalar()
            legacy_lines = int(db.execute(select(func.count(Line.id)).where(
                Line.import_batch_id == "LEGACY_HISTORY")).scalar() or 0)
            df_base = pd.read_parquet(base_parquet) if legacy_active and legacy_lines else None
            if df_base is not None and len(df_base) != legacy_lines:
                logger.info("History parquet (%d rows) differs from the database legacy rows (%d); reading the database",
                            len(df_base), legacy_lines)
                df_base = None
            if df_base is None:
                raise LookupError("parquet fast path not applicable")
            batch_ids = [b for (b,) in db.execute(
                select(ImportBatchModel.import_batch_id).where(
                    ImportBatchModel.import_batch_id != "LEGACY_HISTORY",
                    ImportBatchModel.processing_status == "ACTIVE"
                )
            ).all()]
            if batch_ids:
                delta_rows = db.execute(
                    select(*_SELECT_COLS).where(Line.import_batch_id.in_(batch_ids)).order_by(Line.invoice_date, Line.id)
                )
                df_delta = _to_frame(delta_rows)
                df = pd.concat([df_base, df_delta], ignore_index=True) if not df_delta.empty else df_base
            else:
                df = df_base

            if with_history_columns and not df.empty:
                from refillcare.data.history import create_purchase_history
                if not batch_ids and "days_since_previous_purchase" in df.columns:
                    pass
                else:
                    df = create_purchase_history(df)
            with _cache_lock:
                _full_cache[key] = (ver, df)
            return df
        except LookupError:
            pass
        except Exception as e:
            logger.warning("Fast baseline parquet load failed, falling back to full table scan: %s", e)

    rows = db.execute(select(*_SELECT_COLS).order_by(Line.invoice_date, Line.id))
    df = _to_frame(rows)
    if with_history_columns and not df.empty:
        from refillcare.data.history import create_purchase_history

        df = create_purchase_history(df)
    with _cache_lock:
        _full_cache[key] = (ver, df)
    return df


def sales_dates(db: Session) -> pd.Series:
    """Invoice dates only (cheap): windows planning, freshness and gap checks."""
    vals = [d for (d,) in db.execute(select(Line.invoice_date).where(Line.refillcare_eligible.is_(True)))]
    return pd.to_datetime(pd.Series(vals, dtype="datetime64[ns]"), errors="coerce")


def date_bounds(db: Session) -> Tuple[Optional[date], Optional[date]]:
    # Two queries on purpose: SQLite answers a lone MIN or MAX from the index, but MIN+MAX together scan the table
    lo = db.execute(select(func.min(Line.invoice_date))).scalar()
    hi = db.execute(select(func.max(Line.invoice_date))).scalar()
    as_date = lambda v: (v.date() if isinstance(v, datetime) else v) if v is not None else None  # noqa: E731
    return as_date(lo), as_date(hi)


def distinct_days(db: Session, start: date, end: date) -> set:
    q = select(func.date(Line.invoice_date)).where(Line.invoice_date >= datetime.combine(start, datetime.min.time()),
                                                   Line.invoice_date <= datetime.combine(end, datetime.max.time())).distinct()
    out = set()
    for (d,) in db.execute(q):
        out.add(d if isinstance(d, date) else pd.Timestamp(d).date())
    return out


def known_customers(db: Session, customer_ids: Iterable[str]) -> set:
    ids = sorted({str(c).strip() for c in customer_ids if c is not None and str(c).strip()})
    found: set = set()
    for i in range(0, len(ids), _CHUNK):
        found.update(c for (c,) in db.execute(select(Line.customer_id).where(Line.customer_id.in_(ids[i:i + _CHUNK])).distinct()))
    return found


def known_items(db: Session, item_ids: Iterable[str]) -> set:
    ids = sorted({str(c).strip() for c in item_ids if c is not None and str(c).strip()})
    found: set = set()
    for i in range(0, len(ids), _CHUNK):
        found.update(c for (c,) in db.execute(select(Line.item_id).where(Line.item_id.in_(ids[i:i + _CHUNK])).distinct()))
    return found


def batch_lines(db: Session, batch_id: str) -> pd.DataFrame:
    return _to_frame(db.execute(select(*_SELECT_COLS).where(Line.import_batch_id == batch_id)))


def delete_batch(db: Session, batch_id: str) -> int:
    """Delete a batch's lines (caller commits)."""
    n = db.query(Line).filter(Line.import_batch_id == batch_id).delete(synchronize_session=False)
    return int(n or 0)


RETURN_FULL_CORRECTION = "RETURNED"          # fully returned within the correction window (billing correction)
RETURN_FULL_LATE = "RETURNED_LATE"            # fully returned later: therapy stopped or changed? -> pharmacist review


def return_window_days() -> int:
    """How long after a sale a return is still subtracted from it (store setting, default 90 days).

    Measured on store-1 (10 Nov 2026): reminder-medicine returns come back a median 10 days after the sale,
    75% within 29 and 90% within 56 days. A longer window (the client accepts returns until expiry) would
    subtract old strips from the patient's latest purchase instead of the one they came from.
    """
    from refillcare.stores import store_setting

    try:
        return max(1, int(store_setting("REFILLCARE_RETURN_WINDOW_DAYS", "90") or 90))
    except ValueError:
        return 90


def return_correction_days() -> int:
    """A FULL return up to this many days after the sale is a billing correction (default 7)."""
    from refillcare.stores import store_setting

    try:
        return max(0, int(store_setting("REFILLCARE_RETURN_CORRECTION_DAYS", "7") or 7))
    except ValueError:
        return 7


def net_returns(db: Session, returns: pd.DataFrame, window_days: Optional[int] = None) -> Dict[str, int]:
    """Net returned quantity (negative lines) against the patient's latest sale of that item.

    A return within ``window_days`` (default :func:`return_window_days`) after a sale reduces that sale's
    quantity; a fully returned sale stops counting as a purchase (``refillcare_eligible`` False) with reason
    RETURNED (within :func:`return_correction_days`: billing correction, the earlier schedule resumes) or
    RETURNED_LATE (later: reminders for that medicine stop and the patient goes to pharmacist review).
    """
    window_days = window_days or return_window_days()
    correction_days = return_correction_days()
    out = {"returns_seen": int(len(returns)), "returns_netted": 0, "sales_fully_returned": 0,
           "late_full_returns": 0, "returned_sales": []}
    if returns.empty:
        return out
    for rec in returns.to_dict(orient="records"):
        d = pd.to_datetime(rec.get("invoice_date"), errors="coerce")
        qty = abs(float(pd.to_numeric(rec.get("quantity"), errors="coerce") or 0))
        if pd.isna(d) or qty <= 0:
            continue
        sale = (db.query(Line)
                .filter(Line.customer_id == str(rec.get("customerId")), Line.item_id == str(rec.get("itemId")),
                        Line.invoice_date <= d.to_pydatetime(), Line.invoice_date >= (d - pd.Timedelta(days=window_days)).to_pydatetime(),
                        Line.quantity > 0, Line.refillcare_eligible.is_(True))
                .order_by(Line.invoice_date.desc(), Line.id.desc()).first())
        if sale is None:
            continue
        sale.quantity = max(0.0, float(sale.quantity) - qty)
        out["returns_netted"] += 1
        if sale.quantity <= 0:
            lag = int((d.normalize() - pd.Timestamp(sale.invoice_date).normalize()).days)
            late = lag > correction_days
            sale.refillcare_eligible = False
            # late: the return date is kept with the reason ("RETURNED_LATE:2024-05-08") for review / due date
            sale.exclusion_reason = f"{RETURN_FULL_LATE}:{d.date().isoformat()}" if late else RETURN_FULL_CORRECTION
            out["sales_fully_returned"] += 1
            out["late_full_returns"] += int(late)
            # the purchase no longer counts: the caller reverts (correction) or stops (late) its reminders
            out["returned_sales"].append({"customerId": sale.customer_id, "itemId": sale.item_id,
                                          "invoice_date": pd.Timestamp(sale.invoice_date), "MOBILE_NO": sale.mobile_no,
                                          "returned_on": d.normalize(), "days_after_sale": lag, "late": late})
    return out


def import_frame(db: Session, df: pd.DataFrame, batch_id: str = "LEGACY_HISTORY", source: str = "legacy history") -> Dict[str, int]:
    """One-time load of an existing history (parquet / master CSV) into the store database."""
    from database.models import ImportBatchModel

    if df is None or df.empty:
        return {"inserted": 0, "skipped": 0}
    if db.get(ImportBatchModel, batch_id) is None:
        db.add(ImportBatchModel(import_batch_id=batch_id, source_filename=source, record_count=len(df),
                                processing_status="ACTIVE", is_active=True, sync_status="SYNCED"))
        db.flush()
    keep = [c for c in COLUMN_MAP if c in df.columns]
    res = append_lines(db, df[keep], batch_id)
    batch = db.get(ImportBatchModel, batch_id)
    batch.records_inserted = int(batch.records_inserted or 0) + res["inserted"]
    db.commit()
    clear_cache()
    return res


def ensure_history_in_db(db: Session, legacy_parquet=None) -> Dict[str, int]:
    """If the store database has no sales yet but a legacy parquet exists, import it once."""
    from refillcare.store import StoreConfig

    if history_count(db) > 0:
        return {"inserted": 0, "skipped": 0, "already_in_db": True}
    path = legacy_parquet or StoreConfig.from_env().history_parquet
    if path is None or not path.exists():
        return {"inserted": 0, "skipped": 0, "no_legacy_history": True}
    df = pd.read_parquet(path)
    if "itemCode" not in df.columns and "itemId" in df.columns:
        df["itemCode"] = df["itemId"]
    logger.info("Importing legacy history %s (%d rows) into the store database", path, len(df))
    return import_frame(db, df, source=str(path.name))


def clear_cache() -> None:
    with _cache_lock:
        _full_cache.clear()
        _patients_cache.clear()
