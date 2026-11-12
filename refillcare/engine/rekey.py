"""One-off cut-over when patient identity / therapy grouping is switched on.

Switching ``REFILLCARE_PATIENT_IDENTITY=name_mobile`` or ``REFILLCARE_ITEM_GROUPING=salt`` changes
every affected ``customer_item_key`` and ``cycle_id``. Without a cut-over the old cycles stay
active next to the new ones and a patient could get the same reminder twice (sent-ness is tracked
per reminder id). This module:

1. finds active cycles whose (customer, item) key no longer exists under the new keys,
2. supersedes them and their unsent stages (reason IDENTITY_REKEY),
3. re-scores every patient under the new keys and creates the new cycles,
4. cancels the new stages whose stage was already SENT under the old key for the same purchase,
5. retires the decisions that no longer stand (``retire_stale_decisions``): decisions on keys that no
   longer exist, or newer than their key's latest real purchase. Left in place they appear as
   duplicates wherever "the latest decision per patient-medicine" is read (review list, overview).

Dry run by default: nothing is written until ``apply=True``.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from database.models import ReminderCycleModel, ReminderStageModel
from refillcare.engine.decision_types import STATUS_APPROVED, STATUS_CANCELLED, STATUS_DUE, STATUS_PENDING, STATUS_SENT, STATUS_SUPERSEDED

logger = logging.getLogger("refillcare.engine.rekey")
SUPERSEDED_OUTCOME = "SUPERSEDED"  # outcome of a retired decision: never counted in live accuracy
REKEY_REASON = "IDENTITY_REKEY: patient / therapy key changed (name+mobile, salt group)"


def backup_sqlite_db() -> Optional[str]:
    """Consistent copy of the current store's SQLite database before a re-key (online backup API).

    PostgreSQL / RDS: returns None - rely on the automated snapshots (point-in-time restore).
    """
    import sqlite3

    from database.connection import IS_SQLITE, _sqlite_path_for
    from refillcare.tenancy import current_store_id

    if not IS_SQLITE:
        return None
    src_path = _sqlite_path_for(current_store_id())
    if not src_path.is_file():
        return None
    backup = src_path.with_name(f"{src_path.stem}_backup_rekey_{datetime.now():%Y%m%d_%H%M%S}.db")
    src, dst = sqlite3.connect(str(src_path)), sqlite3.connect(str(backup))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return str(backup)


def _cycle_id(customer_id: str, item_id: str, d: date) -> str:
    return f"RC-{customer_id}-{item_id}-{d.strftime('%Y%m%d')}"


def rekey_reminder_cycles(db: Session, history_df: pd.DataFrame, apply: bool = False, as_of: Optional[date] = None) -> Dict[str, Any]:
    from refillcare.data.identity import BRAND_ITEM_COL, SOURCE_CUSTOMER_COL, grouping_mode, identity_mode, normalise_trajectory_keys

    modes = {"patient_identity": identity_mode(), "item_grouping": grouping_mode()}
    keyed = normalise_trajectory_keys(history_df)
    dts = pd.to_datetime(keyed["invoice_date"], errors="coerce").dt.date
    new_keys: Set[Tuple[str, str]] = set(zip(keyed["customerId"].astype(str), keyed["itemId"].astype(str)))
    # (old customer, old item, purchase date) -> (new customer, new item)
    remap: Dict[Tuple[str, str, date], Tuple[str, str]] = {}
    for oc, oi, d, nc, ni in zip(keyed[SOURCE_CUSTOMER_COL].astype(str), keyed[BRAND_ITEM_COL].astype(str), dts,
                                 keyed["customerId"].astype(str), keyed["itemId"].astype(str)):
        if d is not None and (oc, oi) != (nc, ni):
            remap.setdefault((oc, oi, d), (nc, ni))

    stale = [c for c in db.query(ReminderCycleModel).filter(ReminderCycleModel.is_active.is_(True)).all()
             if (str(c.customer_id), str(c.item_id)) not in new_keys]
    sent_offsets: Dict[str, Set[int]] = {}  # new cycle id -> stage offsets already sent under the old key
    for c in stale:
        target = remap.get((str(c.customer_id), str(c.item_id), c.last_purchase_date))
        if target is None:
            continue
        offs = {int(s.stage_offset) for s in c.stages if s.status == STATUS_SENT}
        if offs:
            sent_offsets.setdefault(_cycle_id(target[0], target[1], c.last_purchase_date), set()).update(offs)

    report: Dict[str, Any] = {
        "modes": modes,
        "applied": False,
        "active_cycles_with_old_keys": len(stale),
        "old_cycles_mapped_to_new_keys": sum(1 for c in stale if (str(c.customer_id), str(c.item_id), c.last_purchase_date) in remap),
        "new_cycles_with_already_sent_stages": len(sent_offsets),
        "patients_before": int(keyed[SOURCE_CUSTOMER_COL].nunique()),
        "patients_after": int(keyed["customerId"].nunique()),
        "trajectories_after": len(new_keys),
    }
    if not apply:
        return report
    if not stale:
        report["applied"] = True
        return report

    now = datetime.utcnow()
    open_states = (STATUS_PENDING, STATUS_DUE, STATUS_APPROVED)
    stages_superseded = 0
    for c in stale:
        c.is_active = False
        c.superseded_at = now
        for s in c.stages:
            if s.status in open_states:
                s.status = STATUS_SUPERSEDED
                s.failure_reason = REKEY_REASON
                s.updated_at = now
                stages_superseded += 1
    db.commit()

    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
    from refillcare.features.context import PopulationStats
    from refillcare.ml.evaluation import prepare_purchase_events
    from refillcare.ml.model_provider import ModelProvider

    latest = max(d for d in dts if d is not None)
    eval_as_of = as_of or (latest + timedelta(days=1))
    engine = UnifiedRefillDecisionEngine(model_provider=ModelProvider.from_registry(db),
                                         population_stats=PopulationStats.from_events(prepare_purchase_events(keyed), as_of=latest))
    _, decisions = engine.evaluate_all_customer_items(keyed, as_of_date=eval_as_of)
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, decisions)
    sync = pm.sync_reminder_cycles(db, decisions)

    cancelled = 0
    for cid, offs in sent_offsets.items():
        for s in db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == cid,
                                                      ReminderStageModel.stage_offset.in_(list(offs)),
                                                      ReminderStageModel.status.in_(open_states)).all():
            s.status = STATUS_CANCELLED
            s.failure_reason = "Already sent under the previous patient key (identity re-key)"
            s.updated_at = now
            cancelled += 1
    db.commit()
    report["stale_decisions"] = retire_stale_decisions(db, keyed)
    report.update({"applied": True, "old_stages_superseded": stages_superseded, "decisions_rescored": len(decisions),
                   "cycles_created": sync.get("cycles_created", 0), "stages_created": sync.get("stages_created", 0),
                   "new_stages_cancelled_already_sent": cancelled})
    logger.info("Identity re-key applied: %s", report)
    return report


def _latest_purchase_per_key(keyed: pd.DataFrame) -> Dict[str, date]:
    """{customer_item_key: latest purchase that counts} from a history already keyed by the engine's identity."""
    df = keyed
    if "refillcare_eligible" in df.columns:
        df = df[df["refillcare_eligible"].fillna(True).astype(bool)]
    if "transaction_type" in df.columns:
        df = df[df["transaction_type"].fillna("CUSTOMER_SALE") == "CUSTOMER_SALE"]
    if df.empty:
        return {}
    keys = df["customerId"].astype(str).str.strip() + "_" + df["itemId"].astype(str).str.strip()
    dts = pd.to_datetime(df["invoice_date"], errors="coerce").dt.date
    return pd.Series(dts.values, index=keys.values).groupby(level=0).max().to_dict()


def retire_stale_decisions(db: Session, keyed_history: Optional[pd.DataFrame] = None,
                           only_keys: Optional[Set[str]] = None) -> Dict[str, int]:
    """Delete decisions that no longer stand, so every patient-medicine has ONE current decision.

    Stale = its key no longer exists under the current identity (e.g. ``NAME_ITEM`` after the switch to
    ``NAME_MOBILE_ITEM``), or it is newer than the key's latest purchase that counts (a purchase that was
    rolled back or fully returned). Old reminder cycles keep their history (their link is cleared);
    measured outcomes are kept, open ones become ``SUPERSEDED`` (never counted). Active cycles are never
    touched: a decision an active cycle still uses is kept.

    ``only_keys``: limit to these keys (after a re-score of some patients); default: the whole store.
    """
    from database.models import RefillDecisionModel as R, RefillDecisionOutcomeModel as O

    if keyed_history is None:
        from refillcare.data import history_store as hs
        from refillcare.data.identity import normalise_trajectory_keys

        keyed_history = normalise_trajectory_keys(hs.load_all(db))
    latest = _latest_purchase_per_key(keyed_history)
    q = db.query(R.decision_id, R.customer_item_key, R.last_purchase_date)
    if only_keys is not None:
        keys = sorted(only_keys)
        rows = []
        for i in range(0, len(keys), 500):
            rows.extend(q.filter(R.customer_item_key.in_(keys[i:i + 500])).all())
    else:
        rows = q.all()
    stale = [did for did, key, lpd in rows if key not in latest or (lpd is not None and lpd > latest[key])]
    out = {"decisions_checked": len(rows), "stale": len(stale), "deleted": 0, "kept_active_cycle": 0,
           "cycles_unlinked": 0, "open_outcomes_superseded": 0}
    if not stale:
        return out
    active = set()
    for i in range(0, len(stale), 500):
        chunk = stale[i:i + 500]
        active.update(d for (d,) in db.query(ReminderCycleModel.decision_id).filter(
            ReminderCycleModel.decision_id.in_(chunk), ReminderCycleModel.is_active.is_(True)).all())
    gone = [d for d in stale if d not in active]
    out["kept_active_cycle"] = len(stale) - len(gone)
    for i in range(0, len(gone), 500):
        chunk = gone[i:i + 500]
        out["cycles_unlinked"] += db.query(ReminderCycleModel).filter(ReminderCycleModel.decision_id.in_(chunk)).update(
            {ReminderCycleModel.decision_id: None}, synchronize_session=False)
        out["open_outcomes_superseded"] += db.query(O).filter(O.decision_id.in_(chunk), O.outcome_category == "OPEN").update(
            {O.outcome_category: SUPERSEDED_OUTCOME}, synchronize_session=False)
        out["deleted"] += db.query(R).filter(R.decision_id.in_(chunk)).delete(synchronize_session=False)
        db.commit()
    logger.info("Stale decisions retired: %s", out)
    return out
