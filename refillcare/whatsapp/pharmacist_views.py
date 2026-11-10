"""Pharmacist-friendly WhatsApp views shared by Streamlit and FastAPI (parity rule).

- ``search_patients``: find a patient by name or mobile number (pharmacists do not know customer IDs)
  and show whether they receive WhatsApp reminders.
- ``set_messages_allowed``: stop / resume reminders for a mobile number (opt-out is matched on the
  phone, so every record sharing that number is covered - the safe direction).
- ``delivery_log_rows``: live messages for one day (India time), Sent / Failed, without technical
  columns or dry-run test rows.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from database.models import RefillDecisionModel, WhatsAppDeliveryLogModel
from refillcare.whatsapp.consent import load_opt_out_index, phone_key, set_opt_in, set_opt_in_by_phone

# Send times are stored in UTC; pharmacists read India time (no daylight saving, fixed offset).
IST_OFFSET = timedelta(hours=5, minutes=30)


def search_patients(db: Session, query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Patients whose name contains ``query`` or whose mobile contains its digits."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    digits = "".join(ch for ch in q if ch.isdigit())
    conds = [func.upper(RefillDecisionModel.customer_name).contains(q.upper())]
    if len(digits) >= 4:
        conds.append(RefillDecisionModel.mobile_no.contains(digits[-10:]))
    rows = (
        db.query(RefillDecisionModel.customer_id, RefillDecisionModel.customer_name, RefillDecisionModel.mobile_no)
        .filter(or_(*conds))
        .distinct()
        .limit(limit * 4)
        .all()
    )
    opt_outs = load_opt_out_index(db)
    seen, out = set(), []
    for cid, name, mobile in rows:
        key = (str(cid), phone_key(mobile))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "customer_id": str(cid),
            "customer_name": name or "-",
            "mobile_number": mobile or "",
            "messages_allowed": not opt_outs.blocks(cid, mobile),
        })
        if len(out) >= limit:
            break
    return out


def set_messages_allowed(db: Session, allowed: bool, mobile_number: Optional[str] = None,
                         customer_id: Optional[str] = None) -> List[str]:
    """Stop (``allowed=False``) or resume reminders. Uses the mobile number when there is one."""
    if phone_key(mobile_number):
        return set_opt_in_by_phone(db, str(mobile_number), opt_in=allowed)
    if customer_id:
        return [set_opt_in(db, str(customer_id), opt_in=allowed).customer_id]
    raise ValueError("A mobile number or customer ID is needed.")


def _friendly_reason(lg: WhatsAppDeliveryLogModel) -> str:
    detail = (lg.error_detail or "").strip()
    low = detail.lower()
    if "validation error" in low or ("invalid" in low and "phone" in low):
        return "Invalid mobile number"
    if "configuration error" in low:
        return "WhatsApp sending is not set up"
    if lg.http_status_code and int(lg.http_status_code) >= 400:
        return f"WhatsApp service rejected the message (code {lg.http_status_code})"
    return detail[:120] or "Not delivered"


def delivery_log_rows(db: Session, day: Optional[date] = None, status: str = "ALL",
                      limit: int = 5000) -> Dict[str, Any]:
    """Live WhatsApp messages for one India-time day. ``status``: ALL, SENT or FAILED."""
    day = day or (datetime.utcnow() + IST_OFFSET).date()
    start_utc = datetime.combine(day, datetime.min.time()) - IST_OFFSET
    end_utc = start_utc + timedelta(days=1)
    q = db.query(WhatsAppDeliveryLogModel).filter(
        WhatsAppDeliveryLogModel.is_dry_run == False,  # noqa: E712 - test (dry-run) rows are never shown
        WhatsAppDeliveryLogModel.dispatched_at >= start_utc,
        WhatsAppDeliveryLogModel.dispatched_at < end_utc,
    )
    logs = q.order_by(WhatsAppDeliveryLogModel.dispatched_at.desc()).limit(limit).all()
    sent = sum(1 for lg in logs if lg.status == "SUCCESS")
    failed = sum(1 for lg in logs if lg.status != "SUCCESS")
    want = (status or "ALL").strip().upper()
    rows = []
    for lg in logs:
        ok = lg.status == "SUCCESS"
        if (want == "SENT" and not ok) or (want == "FAILED" and ok):
            continue
        rows.append({
            "time": (lg.dispatched_at + IST_OFFSET).strftime("%d-%m-%Y %I:%M %p") if lg.dispatched_at else "-",
            "patient": lg.customer_name or "-",
            "mobile": lg.phone_number or "",
            "status": "Sent" if ok else "Failed",
            "reason": "" if ok else _friendly_reason(lg),
        })
    return {"day": day.isoformat(), "sent": sent, "failed": failed, "rows": rows}
