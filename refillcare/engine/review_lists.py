"""The other "Patients Needing Review" lists (patients already in reminders, or B2B audit) for ONE day.

Shared by the Streamlit Review tab and ``GET /api/v1/reminders/review-other`` so both show the same rows.
The main list ("Not getting reminders") is :func:`refillcare.engine.pharmacist_overrides.review_candidates`.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from sqlalchemy import or_
from sqlalchemy.orm import Session

CATEGORIES = {
    "missing_mobile": "Missing mobile number (in reminders)",
    "unstable": "Unstable refill pattern (in reminders)",
    "b2b": "B2B / Inter-Store excluded (audit)",
}
UNSTABLE_TIERS = ("MEDIUM-RISK", "UNSTABLE")


def review_other(db: Session, category: str, day: date, limit: int = 2000) -> List[Dict[str, Any]]:
    from database.models import RefillDecisionModel as R, SalesTransactionModel as Line
    from refillcare.data.monthly_ingestion import determine_mobile_status

    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {sorted(CATEGORIES)}")

    if category == "b2b":  # audit of excluded inter-store lines (latest first, not date based)
        rows = (db.query(Line).filter(Line.transaction_type == "B2B_INTER_STORE")
                .order_by(Line.invoice_date.desc()).limit(min(limit, 200)).all())
        return [{
            "customer_name": r.customer_name or r.customer_id or "-",
            "mobile_number": r.mobile_no or "",
            "mobile_status": determine_mobile_status(r.mobile_no or ""),
            "medicine": r.item_name or r.item_id,
            "last_purchase_date": r.invoice_date.date().isoformat() if r.invoice_date else None,
            "expected_refill_date": None,
            "review_reason": "Excluded from RefillCare (B2B Inter-Store Transaction SB/...)",
        } for r in rows]

    from database.models import ReminderCycleModel as C

    active = db.query(C.decision_id).filter(C.is_active.is_(True))  # current cycles only (not superseded decisions)
    q = db.query(R).filter(R.is_eligible.is_(True), R.expected_refill_date == day, R.decision_id.in_(active))
    if category == "missing_mobile":
        q = q.filter(or_(R.mobile_no.is_(None), R.mobile_no == "", R.mobile_no == "nan"))
        reason = lambda r: "Missing mobile number for automated reminder delivery"  # noqa: E731
    else:
        q = q.filter(R.stability_tier.in_(UNSTABLE_TIERS))
        reason = lambda r: f"Variance risk ({r.stability_tier}): pharmacist interval review recommended"  # noqa: E731
    out = []
    for r in q.order_by(R.customer_name).limit(limit).all():
        out.append({
            "customer_name": r.customer_name or "Unknown",
            "mobile_number": r.mobile_no or "",
            "mobile_status": determine_mobile_status(r.mobile_no or ""),
            "medicine": r.item_name or r.item_id,
            "last_purchase_date": r.last_purchase_date.isoformat() if r.last_purchase_date else None,
            "expected_refill_date": r.expected_refill_date.isoformat() if r.expected_refill_date else None,
            "review_reason": reason(r),
        })
    return out
