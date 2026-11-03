"""Live outcome tracking: compare every eligible prediction with the patient's actual next purchase.

``refresh_outcomes`` is idempotent and safe to run after every sales upload (or daily).
``outcome_summary`` produces the only accuracy figures that may be shown as "live accuracy".
"""
from __future__ import annotations

import bisect
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import (
    RefillDecisionModel,
    RefillDecisionOutcomeModel,
    ReminderStageModel,
    WhatsAppDeliveryLogModel,
)

ON_TIME, EARLY, LATE, NO_REFILL, OPEN = "ON_TIME", "EARLY", "LATE", "NO_REFILL", "OPEN"
NO_REFILL_AFTER_DAYS = 60  # no purchase within 60 days after the predicted date -> NO_REFILL


def _purchase_index(history_df: pd.DataFrame) -> Dict[str, List[date]]:
    df = history_df
    if "refillcare_eligible" in df.columns:
        df = df[df["refillcare_eligible"].fillna(True).astype(bool)]
    from refillcare.data.identity import normalise_trajectory_keys

    df = normalise_trajectory_keys(df)  # same patient / therapy keys as the decisions
    keys = df["customerId"].astype(str).str.strip() + "_" + df["itemId"].astype(str).str.strip()
    dts = pd.to_datetime(df["invoice_date"], errors="coerce").dt.date
    idx: Dict[str, List[date]] = {}
    for k, d in zip(keys.values, dts.values):
        if d is None or (isinstance(d, float) and np.isnan(d)):
            continue
        idx.setdefault(k, []).append(d)
    for k in idx:
        idx[k] = sorted(set(idx[k]))
    return idx


def classify_error(error_days: float) -> str:
    if error_days < -7:
        return EARLY
    if error_days > 7:
        return LATE
    return ON_TIME


def refresh_outcomes(db: Session, history_df: pd.DataFrame, as_of: Optional[date] = None,
                     only_keys: Optional[set] = None) -> Dict[str, int]:
    """Create / update outcome rows for every eligible decision with an expected refill date.

    ``only_keys`` ({customer_item_key}) limits the refresh to decisions whose purchases are in
    ``history_df`` (incremental uploads pass only the touched patients' history).
    """
    if history_df is None or history_df.empty:
        return {"evaluated": 0}
    idx = _purchase_index(history_df)
    if as_of is None:
        all_dates = [d for v in idx.values() for d in v[-1:]]
        as_of = max(all_dates) if all_dates else date.today()

    live_sent_ids = {
        r[0]
        for r in db.query(WhatsAppDeliveryLogModel.reminder_id)
        .filter(WhatsAppDeliveryLogModel.is_dry_run == False, WhatsAppDeliveryLogModel.status == "SUCCESS")  # noqa: E712
        .all()
        if r[0]
    }
    live_cycles = set()
    if live_sent_ids:
        ids = list(live_sent_ids)
        for i in range(0, len(ids), 500):
            for (cid,) in db.query(ReminderStageModel.cycle_id).filter(ReminderStageModel.reminder_id.in_(ids[i:i + 500])).all():
                live_cycles.add(cid)

    q = db.query(RefillDecisionModel).filter(RefillDecisionModel.is_eligible == True, RefillDecisionModel.expected_refill_date != None)  # noqa: E711,E712
    if only_keys is not None:
        keys = sorted(only_keys)
        decisions = []
        for i in range(0, len(keys), 500):
            decisions.extend(q.filter(RefillDecisionModel.customer_item_key.in_(keys[i:i + 500])).all())
    else:
        decisions = q.all()
    if only_keys is not None:
        ids = [d.decision_id for d in decisions]
        existing = {}
        for i in range(0, len(ids), 500):
            existing.update({o.decision_id: o for o in db.query(RefillDecisionOutcomeModel).filter(RefillDecisionOutcomeModel.decision_id.in_(ids[i:i + 500]))})
    else:
        existing = {o.decision_id: o for o in db.query(RefillDecisionOutcomeModel).all()}
    counts = {"evaluated": 0, ON_TIME: 0, EARLY: 0, LATE: 0, NO_REFILL: 0, OPEN: 0}
    new_rows = []
    now = datetime.utcnow()
    for d in decisions:
        purchases = idx.get(d.customer_item_key, [])
        pos = bisect.bisect_right(purchases, d.last_purchase_date)
        actual = purchases[pos] if pos < len(purchases) else None
        if actual is not None:
            err = float((d.expected_refill_date - actual).days)
            cat = classify_error(err)
        else:
            err = None
            cat = NO_REFILL if (as_of - d.expected_refill_date).days > NO_REFILL_AFTER_DAYS else OPEN
        counts[cat] += 1
        counts["evaluated"] += 1
        vals = dict(
            customer_item_key=d.customer_item_key,
            model_version=d.model_version,
            prediction_source=d.prediction_source,
            path=d.path,
            last_purchase_date=d.last_purchase_date,
            predicted_refill_date=d.expected_refill_date,
            predicted_interval_days=float(d.predicted_interval_days) if d.predicted_interval_days is not None else None,
            shadow_model_version=d.shadow_model_version,
            shadow_predicted_interval_days=d.shadow_predicted_interval_days,
            actual_purchase_date=actual,
            error_days=err,
            outcome_category=cat,
            reminder_sent_live=d.cycle_id in live_cycles,
            evaluated_at=now,
        )
        row = existing.get(d.decision_id)
        if row is None:
            new_rows.append(RefillDecisionOutcomeModel(decision_id=d.decision_id, **vals))
        else:
            for k, v in vals.items():
                setattr(row, k, v)
    if new_rows:
        db.add_all(new_rows)
    db.commit()
    return counts


def _summ(rows: List[RefillDecisionOutcomeModel]) -> Dict[str, Any]:
    resolved = [r for r in rows if r.outcome_category in (ON_TIME, EARLY, LATE)]
    errs = np.array([r.error_days for r in resolved], dtype=float)
    out: Dict[str, Any] = {
        "decisions": len(rows),
        "resolved": len(resolved),
        "open": sum(1 for r in rows if r.outcome_category == OPEN),
        "no_refill": sum(1 for r in rows if r.outcome_category == NO_REFILL),
    }
    if len(errs):
        out.update({
            "within_7_days_pct": round(float((np.abs(errs) <= 7).mean() * 100), 2),
            "mae_days": round(float(np.abs(errs).mean()), 2),
            "early_pct": round(float((errs < -7).mean() * 100), 2),
            "late_pct": round(float((errs > 7).mean() * 100), 2),
        })
    closed = out["resolved"] + out["no_refill"]
    out["refill_rate_pct"] = round(100.0 * out["resolved"] / closed, 2) if closed else None
    return out


def outcome_summary(db: Session, window_days: int = 90, as_of: Optional[date] = None) -> Dict[str, Any]:
    """Live accuracy over decisions whose predicted refill date fell in the last ``window_days``."""
    q = db.query(RefillDecisionOutcomeModel)
    if as_of is None:
        latest = db.query(RefillDecisionOutcomeModel.actual_purchase_date).order_by(RefillDecisionOutcomeModel.actual_purchase_date.desc()).first()
        as_of = latest[0] if latest and latest[0] else date.today()
    start = as_of - timedelta(days=window_days)
    rows = q.filter(RefillDecisionOutcomeModel.predicted_refill_date >= start, RefillDecisionOutcomeModel.predicted_refill_date <= as_of).all()

    by_version: Dict[str, List[RefillDecisionOutcomeModel]] = {}
    by_source: Dict[str, List[RefillDecisionOutcomeModel]] = {}
    for r in rows:
        by_version.setdefault(r.model_version or "UNRECORDED", []).append(r)
        by_source.setdefault(r.prediction_source or "UNRECORDED", []).append(r)

    # Shadow vs production on the same resolved decisions
    shadow_rows = [r for r in rows if r.shadow_predicted_interval_days is not None and r.actual_purchase_date is not None]
    shadow = None
    if shadow_rows:
        actual_iv = np.array([(r.actual_purchase_date - r.last_purchase_date).days for r in shadow_rows], dtype=float)
        prod_iv = np.array([r.predicted_interval_days for r in shadow_rows], dtype=float)
        sh_iv = np.array([r.shadow_predicted_interval_days for r in shadow_rows], dtype=float)
        shadow = {
            "shadow_model_version": shadow_rows[0].shadow_model_version,
            "n": len(shadow_rows),
            "production_within_7_days_pct": round(float((np.abs(prod_iv - actual_iv) <= 7).mean() * 100), 2),
            "shadow_within_7_days_pct": round(float((np.abs(sh_iv - actual_iv) <= 7).mean() * 100), 2),
        }

    live = [r for r in rows if r.reminder_sent_live]
    return {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "overall": _summ(rows),
        "by_model_version": {k: _summ(v) for k, v in by_version.items()},
        "by_prediction_source": {k: _summ(v) for k, v in by_source.items()},
        "shadow_comparison": shadow,
        "live_reminders": {"decisions_with_live_reminder": len(live), **({"conversion": _summ(live)} if live else {})},
        "definition": "within_7_days = actual next purchase within +/-7 days of the predicted refill date; "
        "refill_rate = resolved / (resolved + no refill within 60 days of the predicted date).",
    }
