"""Pharmacist "Add to reminders" for patients the engine left out (shared by Streamlit and FastAPI).

- ``review_candidates``: patients NOT getting reminders whose medicine is chronic or not yet
  classified by the medication rules (Supabase rules + built-in list). Medicines the rules mark as
  non-chronic (acute, OTC, FMCG, ...) are never offered.
- ``add_to_reminders`` / ``remove_from_reminders``: pharmacist decision, stored in
  ``pharmacist_reminder_overrides`` and turned into a normal reminder cycle (same 7 stages).
- ``apply_pharmacist_overrides``: called by the persistence layer on every save / sync, so the
  re-scoring after each upload keeps the patient in reminders (and moves the date after a repurchase).
- ``apply_return_stops``: a medicine returned in FULL later than the correction window (default 7 days
  after the sale) stops its reminders until the patient buys it again, and the patient is listed for review
  ("therapy stopped or changed?"). Adding the patient back to reminders overrides the stop.
"""
from __future__ import annotations

import dataclasses
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import PharmacistReminderOverrideModel, ReminderCycleModel, ReminderStageModel, RefillDecisionModel

logger = logging.getLogger("refillcare.pharmacist_overrides")

PATH_PHARMACIST = "PHARMACIST_ADDED"
SOURCE_PHARMACIST = "PHARMACIST"
MIN_INTERVAL, MAX_INTERVAL, DEFAULT_INTERVAL = 7, 180, 30
LOOKBACK_DAYS = MAX_INTERVAL + 30  # a refill due in the window was bought at most this long before it

CLASS_CHRONIC = "Chronic"
CLASS_UNCLASSIFIED = "Not classified"


def medicine_class(item_name: str) -> Optional[str]:
    """``Chronic`` / ``Not classified`` from the medication rules, or None for known non-chronic."""
    from refillcare.data.medication_classifier import classify_medication_cached

    res = classify_medication_cached(str(item_name or ""))
    if res.get("is_chronic_eligible"):
        return CLASS_CHRONIC
    if res.get("category") == "UNCLASSIFIED_NON_CHRONIC":
        return CLASS_UNCLASSIFIED
    return None  # matched a non-chronic rule (acute, OTC, FMCG, Supabase exclusion, ...)


def estimate_interval_days(dec: Any) -> int:
    """Patient's usual refill gap: personal median, else days of supply, else engine estimate, else 30."""
    for v in (getattr(dec, "cadence_median", None), getattr(dec, "dos_days", None), getattr(dec, "predicted_interval_days", None)):
        try:
            if v is not None and float(v) > 0:
                return int(max(MIN_INTERVAL, min(MAX_INTERVAL, round(float(v)))))
        except (TypeError, ValueError):
            continue
    return DEFAULT_INTERVAL


RETURN_STOP_REASON = ("Returned all strips on {ret} ({days} days after buying on {sale}). Therapy stopped or changed? "
                      "Reminders paused - add back to reminders if the patient still takes it.")


def _key_for_line(customer_id: Any, mobile: Any, item_id: Any) -> str:
    """customer_item_key of a sales line under the store's patient identity (same keys as the engine)."""
    from refillcare.data.identity import identity_mode, mobile10, patient_id

    m10 = mobile10(mobile) if identity_mode() == "name_mobile" else None
    return f"{patient_id(customer_id, m10)}_{str(item_id).strip()}"


def return_stops(db: Session) -> Dict[str, Dict[str, date]]:
    """{patient-medicine key: {"sale": date, "returned_on": date}} of the latest LATE full return per key."""
    from database.models import SalesTransactionModel as Line
    from refillcare.data.history_store import RETURN_FULL_LATE

    out: Dict[str, Dict[str, date]] = {}
    rows = db.query(Line.customer_id, Line.mobile_no, Line.item_id, Line.invoice_date, Line.exclusion_reason).filter(
        Line.exclusion_reason.like(f"{RETURN_FULL_LATE}%")).all()
    for cid, mob, iid, sale_dt, reason in rows:
        sale_d = sale_dt.date() if isinstance(sale_dt, datetime) else sale_dt
        try:
            ret_d = date.fromisoformat(str(reason).split(":", 1)[1])
        except (IndexError, ValueError):
            ret_d = sale_d
        key = _key_for_line(cid, mob, iid)
        cur = out.get(key)
        if cur is None or sale_d > cur["sale"]:
            out[key] = {"sale": sale_d, "returned_on": ret_d}
    return out


def apply_return_stops(db: Session, decisions: List[Any]) -> List[Any]:
    """Pause reminders (in place) for medicines whose latest purchase was returned in full, late.

    Applies only while no purchase is newer than the returned sale; a pharmacist addition (applied after
    this) brings the patient back. Idempotent.
    """
    if not decisions:
        return decisions
    try:
        stops = return_stops(db)
    except Exception:  # sales table missing on a fresh store
        logger.debug("Return stops unavailable", exc_info=True)
        return decisions
    if not stops:
        return decisions
    for i, dec in enumerate(decisions):
        st = stops.get(getattr(dec, "customer_item_key", None))
        if st is None or getattr(dec, "path", None) == PATH_PHARMACIST:
            continue
        if dec.last_purchase_date is not None and dec.last_purchase_date > st["sale"]:
            continue  # bought again after the returned sale -> normal reminders
        days = (st["returned_on"] - st["sale"]).days
        decisions[i] = dataclasses.replace(
            dec, is_eligible=False, expected_refill_date=None, predicted_interval_days=None,
            decision_reason=RETURN_STOP_REASON.format(ret=st["returned_on"].strftime("%d-%m-%Y"), days=days,
                                                     sale=st["sale"].strftime("%d-%m-%Y")))
    return decisions


def _active_override_map(db: Session) -> Dict[str, PharmacistReminderOverrideModel]:
    return {o.customer_item_key: o for o in db.query(PharmacistReminderOverrideModel).filter(
        PharmacistReminderOverrideModel.is_active == True).all()}  # noqa: E712


def pharmacist_added_customer_ids(db: Session) -> Set[str]:
    """Customers with an active pharmacist addition (never put in the no-reminder control group)."""
    return {str(c) for (c,) in db.query(PharmacistReminderOverrideModel.customer_id).filter(
        PharmacistReminderOverrideModel.is_active == True).all()}  # noqa: E712


def review_candidates(db: Session, start: Optional[date] = None, end: Optional[date] = None,
                      limit: int = 3000) -> List[Dict[str, Any]]:
    """Patients not getting reminders whose (estimated) refill falls in [start, end] (None = no date filter)."""
    from refillcare.data.monthly_ingestion import determine_mobile_status

    R = RefillDecisionModel
    # Light column scan of the window (no GROUP BY over every decision): the latest decision of a key
    # is always inside the window when any of its decisions is, so "latest per key" is done here.
    scan = db.query(R.decision_id, R.customer_item_key, R.last_purchase_date, R.is_eligible, R.path,
                    R.expected_refill_date, R.cadence_median, R.dos_days, R.predicted_interval_days)
    if start is not None:
        scan = scan.filter(R.last_purchase_date >= start - timedelta(days=LOOKBACK_DAYS))
    latest: Dict[str, Any] = {}
    for row in scan.all():
        cur = latest.get(row.customer_item_key)
        if cur is None or row.last_purchase_date > cur.last_purchase_date:
            latest[row.customer_item_key] = row

    active_keys = {k for (k,) in db.query(ReminderCycleModel.customer_item_key).filter(ReminderCycleModel.is_active == True).all()}  # noqa: E712
    stops = return_stops(db)
    due_by_id: Dict[str, Any] = {}
    for key, row in latest.items():
        if row.path == PATH_PHARMACIST or (end is not None and row.last_purchase_date > end):
            continue
        engine_scheduled = bool(row.is_eligible and key in active_keys)
        if key in active_keys and not engine_scheduled:
            continue
        if row.is_eligible and not engine_scheduled:
            continue
        interval = estimate_interval_days(row)
        due = (row.expected_refill_date if engine_scheduled and row.expected_refill_date
               else row.last_purchase_date + timedelta(days=interval))
        st = stops.get(key)
        if st is not None and not row.is_eligible and row.last_purchase_date <= st["sale"]:
            due = st["returned_on"]  # reminders paused by a late full return: review on the return day
        if (start is not None and due < start) or (end is not None and due > end):
            continue  # date check before the (slower) medicine classification
        due_by_id[row.decision_id] = (interval, due)

    rows = []
    ids = list(due_by_id)
    for i in range(0, len(ids), 500):
        rows.extend(db.query(R).filter(R.decision_id.in_(ids[i:i + 500])).all())

    out: List[Dict[str, Any]] = []
    for d in rows:
        cls = medicine_class(d.item_name or "")
        if cls is None:
            continue
        if d.is_eligible and cls == CLASS_CHRONIC:
            continue  # engine reminds this patient and the rules allow sending - nothing to review
        interval, due = due_by_id[d.decision_id]
        out.append({
            "customer_item_key": d.customer_item_key,
            "customer_id": d.customer_id,
            "customer_name": d.customer_name or "-",
            "mobile_number": d.mobile_no or "",
            "mobile_status": determine_mobile_status(d.mobile_no or ""),
            "medicine": d.item_name or d.item_id,
            "medicine_class": cls,
            "purchase_count": int(d.purchase_count or 0),
            "last_purchase_date": d.last_purchase_date.isoformat(),
            "usual_gap_days": interval,
            "estimated_due_date": due.isoformat(),
            "reason_not_in_reminders": (NOT_CLASSIFIED_SCHEDULED_REASON if d.is_eligible
                                        else d.decision_reason or "Not eligible for automatic reminders"),
        })
    out.sort(key=lambda r: (r["estimated_due_date"], r["customer_name"]))
    return out[:limit]


NOT_CLASSIFIED_SCHEDULED_REASON = ("Bought regularly, but the medicine is not classified in the Medication Rules, "
                                   "so no reminders are sent. Add it if it is a regular medicine (or add a chronic rule).")


def _already_sent_by_engine(m) -> bool:
    """Engine-eligible AND allowed by the send rule (rules-chronic) -> the patient already gets reminders."""
    return bool(m.is_eligible and m.path != PATH_PHARMACIST and medicine_class(m.item_name or "") == CLASS_CHRONIC) or         bool(m.is_eligible and m.path == PATH_PHARMACIST)


def _override_decision(dec, ov: PharmacistReminderOverrideModel):
    """Engine decision -> eligible pharmacist decision (same patient, same medicine)."""
    if ov.last_purchase_at_add is None or dec.last_purchase_date <= ov.last_purchase_at_add:
        expected = ov.next_refill_date
    else:  # patient bought again after the addition -> next reminder moves forward
        expected = dec.last_purchase_date + timedelta(days=int(ov.interval_days))
    return dataclasses.replace(
        dec,
        path=PATH_PHARMACIST,
        is_eligible=True,
        prediction_method="PHARMACIST_ADDED",
        prediction_source=SOURCE_PHARMACIST,
        predicted_interval_days=int(ov.interval_days),
        expected_refill_date=expected,
        decision_reason=(f"Added to reminders by {ov.added_by} on {ov.added_at:%d-%m-%Y} "
                         f"(refill every {int(ov.interval_days)} days)" + (f" - {ov.note}" if ov.note else "")),
        cycle_id=f"{dec.cycle_id}-PH" if not str(dec.cycle_id).endswith("-PH") else dec.cycle_id,
    )


def apply_pharmacist_overrides(db: Session, decisions: List[Any]) -> List[Any]:
    """Replace (in place) not-eligible decisions of pharmacist-added keys. Idempotent."""
    if not decisions:
        return decisions
    try:
        overrides = _active_override_map(db)
    except Exception:  # table not created yet (fresh store before migration)
        logger.debug("Pharmacist overrides unavailable", exc_info=True)
        return decisions
    if not overrides:
        return decisions
    for i, dec in enumerate(decisions):
        ov = overrides.get(getattr(dec, "customer_item_key", None))
        if ov is None:
            continue
        if not dec.is_eligible or (dec.path != PATH_PHARMACIST and medicine_class(dec.item_name or "") != CLASS_CHRONIC):
            # not eligible, or eligible but "not classified" (not sent under the send rule) -> pharmacist schedule
            if dec.is_eligible:  # retire the engine's unsent schedule so only one cycle stays active
                _deactivate_pharmacist_cycles(db, ov.customer_item_key, "Replaced by the pharmacist's schedule", engine_only=True)
            decisions[i] = _override_decision(dec, ov)
        elif dec.path != PATH_PHARMACIST:
            # The engine now reminds this patient by itself -> retire the pharmacist schedule so the
            # patient never gets two parallel reminder cycles for the same medicine.
            _deactivate_pharmacist_cycles(db, ov.customer_item_key, "Engine now schedules this refill automatically")
            ov.is_active, ov.removed_by, ov.removed_at = False, "engine", datetime.utcnow()
    return decisions


def _deactivate_pharmacist_cycles(db: Session, customer_item_key: str, reason: str, engine_only: bool = False) -> int:
    """Cancel the key's active pharmacist (-PH) cycles, or with ``engine_only`` its active engine cycles."""
    now = datetime.utcnow()
    cancelled = 0
    q = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == customer_item_key,
                                            ReminderCycleModel.is_active == True)  # noqa: E712
    q = q.filter(~ReminderCycleModel.cycle_id.like("%-PH")) if engine_only else q.filter(ReminderCycleModel.cycle_id.like("%-PH"))
    for cyc in q.all():
        cyc.is_active = False
        cyc.superseded_at = now
        for st in db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == cyc.cycle_id,
                                                      ReminderStageModel.status.in_(("PENDING", "DUE", "APPROVED"))).all():
            st.status = "CANCELLED"
            st.failure_reason = reason
            st.updated_at = now
            cancelled += 1
    return cancelled


def _decision_from_model(m: RefillDecisionModel):
    from refillcare.engine.decision_types import RefillDecision

    return RefillDecision(
        customer_id=m.customer_id, customer_name=m.customer_name or "", mobile_no=m.mobile_no,
        item_id=m.item_id, item_name=m.item_name or m.item_id, customer_item_key=m.customer_item_key,
        path=m.path, purchase_count=int(m.purchase_count or 0), is_eligible=bool(m.is_eligible),
        stability_tier=m.stability_tier or "UNSTABLE", cadence_median=m.cadence_median,
        cadence_norm_mad=m.cadence_norm_mad, cadence_drift=m.cadence_drift, dos_days=m.dos_days,
        units_purchased=m.units_purchased, prediction_method=m.prediction_method or "NONE",
        predicted_interval_days=m.predicted_interval_days, last_purchase_date=m.last_purchase_date,
        expected_refill_date=m.expected_refill_date, decision_reason=m.decision_reason or "",
        cycle_id=m.cycle_id, model_version=m.model_version, prediction_source=m.prediction_source,
    )


def add_to_reminders(db: Session, customer_item_key: str, interval_days: Optional[int] = None,
                     next_refill_date: Optional[date] = None, added_by: str = "pharmacist",
                     note: Optional[str] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Pharmacist adds a patient-medicine to reminders; creates the reminder stages now."""
    from refillcare.engine.persistence import RefillPersistenceManager

    today = today or date.today()
    m = (db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_item_key == customer_item_key)
         .order_by(RefillDecisionModel.last_purchase_date.desc()).first())
    if m is None:
        raise ValueError(f"No purchase history found for '{customer_item_key}'.")
    if _already_sent_by_engine(m):  # the engine already reminds this patient - nothing to add
        return {"customer_item_key": customer_item_key, "added": False, "already_in_reminders": True, "stages_created": 0}
    if medicine_class(m.item_name or "") is None:
        raise ValueError(f"'{m.item_name}' is marked non-chronic by the medication rules; it cannot be added to refill reminders.")
    interval = int(interval_days or estimate_interval_days(m))
    if not MIN_INTERVAL <= interval <= MAX_INTERVAL:
        raise ValueError(f"Refill gap must be between {MIN_INTERVAL} and {MAX_INTERVAL} days.")
    nxt = next_refill_date or max(today, m.last_purchase_date + timedelta(days=interval))
    if nxt < today:
        raise ValueError("The next refill date cannot be in the past.")

    ov = db.get(PharmacistReminderOverrideModel, customer_item_key)
    if ov is None:
        ov = PharmacistReminderOverrideModel(customer_item_key=customer_item_key, customer_id=m.customer_id, item_id=m.item_id)
        db.add(ov)
    ov.customer_name, ov.item_name = m.customer_name, m.item_name
    ov.interval_days, ov.next_refill_date, ov.last_purchase_at_add = interval, nxt, m.last_purchase_date
    ov.note, ov.added_by, ov.added_at = (note or None), added_by, datetime.utcnow()
    ov.is_active, ov.removed_by, ov.removed_at = True, None, None
    db.commit()

    decisions = [_decision_from_model(m)]
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, decisions)  # overrides are applied inside (same path as every upload)
    sync = pm.sync_reminder_cycles(db, decisions)
    logger.info("Pharmacist %s added %s to reminders (every %s days, next %s)", added_by, customer_item_key, interval, nxt)
    return {"customer_item_key": customer_item_key, "added": True, "already_in_reminders": False,
            "next_refill_date": nxt.isoformat(), "interval_days": interval,
            "stages_created": int(sync.get("stages_created", 0))}


def add_many_to_reminders(db: Session, customer_item_keys: List[str], added_by: str = "pharmacist",
                          note: Optional[str] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Bulk add: each patient-medicine uses its own usual refill gap and estimated due date (never in the past).

    Same rules as ``add_to_reminders`` (known non-chronic rejected, already-eligible skipped), but the
    reminder stages are created in ONE save + sync pass so adding hundreds of rows stays fast.
    """
    from refillcare.engine.persistence import RefillPersistenceManager

    today = today or date.today()
    added, already, failed, decisions = [], [], [], []
    for key in dict.fromkeys(k for k in customer_item_keys if k):  # de-duplicate, keep order
        m = (db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_item_key == key)
             .order_by(RefillDecisionModel.last_purchase_date.desc()).first())
        if m is None:
            failed.append({"customer_item_key": key, "reason": "No purchase history found."})
            continue
        if _already_sent_by_engine(m):
            already.append(key)
            continue
        if medicine_class(m.item_name or "") is None:
            failed.append({"customer_item_key": key, "reason": f"'{m.item_name}' is non-chronic in the medication rules."})
            continue
        interval = min(MAX_INTERVAL, max(MIN_INTERVAL, int(estimate_interval_days(m))))
        nxt = max(today, m.last_purchase_date + timedelta(days=interval))
        ov = db.get(PharmacistReminderOverrideModel, key)
        if ov is None:
            ov = PharmacistReminderOverrideModel(customer_item_key=key, customer_id=m.customer_id, item_id=m.item_id)
            db.add(ov)
        ov.customer_name, ov.item_name = m.customer_name, m.item_name
        ov.interval_days, ov.next_refill_date, ov.last_purchase_at_add = interval, nxt, m.last_purchase_date
        ov.note, ov.added_by, ov.added_at = (note or None), added_by, datetime.utcnow()
        ov.is_active, ov.removed_by, ov.removed_at = True, None, None
        decisions.append(_decision_from_model(m))
        added.append(key)
    db.commit()

    stages = 0
    if decisions:
        pm = RefillPersistenceManager()
        pm.save_refill_decisions(db, decisions)
        stages = int(pm.sync_reminder_cycles(db, decisions).get("stages_created", 0))
    logger.info("Pharmacist %s bulk-added %d to reminders (%d already, %d failed)", added_by, len(added), len(already), len(failed))
    return {"added": len(added), "already_in_reminders": len(already), "failed": failed,
            "added_keys": added, "stages_created": stages}


def remove_from_reminders(db: Session, customer_item_key: str, removed_by: str = "pharmacist") -> Dict[str, Any]:
    """Undo a pharmacist addition: stop its pending reminders (the engine's own decisions are untouched)."""
    ov = db.get(PharmacistReminderOverrideModel, customer_item_key)
    if ov is None or not ov.is_active:
        raise ValueError(f"'{customer_item_key}' was not added by a pharmacist.")
    ov.is_active, ov.removed_by, ov.removed_at = False, removed_by, datetime.utcnow()
    cancelled = _deactivate_pharmacist_cycles(db, customer_item_key, f"Removed from reminders by {removed_by}")
    db.commit()
    return {"customer_item_key": customer_item_key, "removed": True, "stages_cancelled": cancelled}


def list_pharmacist_added(db: Session) -> List[Dict[str, Any]]:
    rows = (db.query(PharmacistReminderOverrideModel).filter(PharmacistReminderOverrideModel.is_active == True)  # noqa: E712
            .order_by(PharmacistReminderOverrideModel.added_at.desc()).all())
    return [{"customer_item_key": o.customer_item_key, "customer_id": o.customer_id, "customer_name": o.customer_name,
             "medicine": o.item_name, "interval_days": o.interval_days, "next_refill_date": o.next_refill_date.isoformat(),
             "added_by": o.added_by, "added_at": o.added_at.isoformat(), "note": o.note} for o in rows]
