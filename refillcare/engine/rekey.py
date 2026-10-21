"""One-off cut-over when patient identity / therapy grouping is switched on.

Switching ``REFILLCARE_PATIENT_IDENTITY=name_mobile`` or ``REFILLCARE_ITEM_GROUPING=salt`` changes
every affected ``customer_item_key`` and ``cycle_id``. Without a cut-over the old cycles stay
active next to the new ones and a patient could get the same reminder twice (sent-ness is tracked
per reminder id). This module:

1. finds active cycles whose (customer, item) key no longer exists under the new keys,
2. supersedes them and their unsent stages (reason IDENTITY_REKEY),
3. re-scores every patient under the new keys and creates the new cycles,
4. cancels the new stages whose stage was already SENT under the old key for the same purchase.

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
REKEY_REASON = "IDENTITY_REKEY: patient / therapy key changed (name+mobile, salt group)"


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
    report.update({"applied": True, "old_stages_superseded": stages_superseded, "decisions_rescored": len(decisions),
                   "cycles_created": sync.get("cycles_created", 0), "stages_created": sync.get("stages_created", 0),
                   "new_stages_cancelled_already_sent": cancelled})
    logger.info("Identity re-key applied: %s", report)
    return report
