"""WhatsApp consent (opt-out / opt-in) - the single implementation used by every send path.

Shared by FastAPI (``api/services.py``), Streamlit (``app_refillcare.py``), the batch dispatcher and
the single-stage dispatch in ``refillcare/engine/persistence.py`` so all surfaces agree on who may
be messaged.

Phones are matched on their last 10 digits, so one opt-out covers every customer record sharing
that number (the safe direction).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from database.models import CustomerModel

# Opt-out is matched generously (any standalone stop word); opt-in only on an exact command, so a
# conversational "yes" or "restart" never re-subscribes a patient who said STOP.
STOP_WORDS = {"STOP", "UNSUBSCRIBE", "OPTOUT"}
STOP_PHRASES = {"OPT OUT", "STOP ALL", "STOP REMINDERS", "PLEASE STOP"}
START_COMMANDS = {"START", "UNSTOP", "OPT IN", "OPTIN"}

CONSENT_STOP = "STOP"
CONSENT_START = "START"


def phone_key(phone: Any) -> str:
    """Last 10 digits of a phone number ('' when it has none)."""
    return "".join(ch for ch in str(phone or "") if ch.isdigit())[-10:]


def parse_consent_keyword(message: Any) -> Optional[str]:
    """``CONSENT_STOP``, ``CONSENT_START`` or ``None`` for an inbound WhatsApp message."""
    text = str(message or "").strip().upper()
    if not text:
        return None
    words = set(re.findall(r"\b[A-Z]+\b", text))
    if words & STOP_WORDS or text in STOP_PHRASES:
        return CONSENT_STOP
    if text in START_COMMANDS:
        return CONSENT_START
    return None


@dataclass
class OptOutIndex:
    """Opted-out customer ids and phone keys, loaded once per dispatch run."""

    customer_ids: Set[str] = field(default_factory=set)
    phones: Set[str] = field(default_factory=set)

    def blocks(self, customer_id: Any = None, phone: Any = None) -> bool:
        if customer_id is not None and str(customer_id) in self.customer_ids:
            return True
        key = phone_key(phone)
        return bool(key) and key in self.phones


def load_opt_out_index(db: Session) -> OptOutIndex:
    rows = db.query(CustomerModel.customer_id, CustomerModel.mobile_number).filter(
        CustomerModel.opt_in_whatsapp == False  # noqa: E712
    ).all()
    return OptOutIndex(
        customer_ids={str(cid) for cid, _ in rows if cid},
        phones={phone_key(p) for _, p in rows if phone_key(p)},
    )


def is_opted_out(db: Session, customer_id: Any = None, phone: Any = None) -> bool:
    return load_opt_out_index(db).blocks(customer_id, phone)


def set_opt_in(
    db: Session,
    customer_id: str,
    opt_in: bool,
    phone: Optional[str] = None,
) -> CustomerModel:
    """Set consent for one customer (created if the customer table has no row for it yet)."""
    customer = db.query(CustomerModel).filter(CustomerModel.customer_id == customer_id).first()
    if customer is None:
        customer = CustomerModel(customer_id=customer_id, mobile_number=phone, opt_in_whatsapp=opt_in)
        db.add(customer)
    else:
        customer.opt_in_whatsapp = opt_in
        if phone and not customer.mobile_number:
            customer.mobile_number = phone
        customer.updated_at = datetime.utcnow()
    db.commit()
    return customer


def set_opt_in_by_phone(db: Session, phone: str, opt_in: bool) -> List[str]:
    """Set consent for every customer with this phone; returns the affected customer ids."""
    key = phone_key(phone)
    if not key:
        return []
    # Normalise in Python: stored numbers may contain spaces, '+91', dashes (SQL LIKE would miss them).
    customers = [
        c for c in db.query(CustomerModel).filter(CustomerModel.mobile_number.isnot(None)).all()
        if phone_key(c.mobile_number) == key
    ]
    if not customers:
        return [set_opt_in(db, f"CUST_{key}", opt_in, phone=str(phone)).customer_id]
    now = datetime.utcnow()
    for c in customers:
        c.opt_in_whatsapp = opt_in
        c.updated_at = now
    db.commit()
    return [c.customer_id for c in customers]


def list_opted_out(db: Session) -> List[Dict[str, Any]]:
    rows = (
        db.query(CustomerModel)
        .filter(CustomerModel.opt_in_whatsapp == False)  # noqa: E712
        .order_by(CustomerModel.updated_at.desc())
        .all()
    )
    return [
        {
            "customer_id": c.customer_id,
            "customer_name": getattr(c, "customer_name", None),
            "mobile_number": c.mobile_number,
            "updated_at": c.updated_at.isoformat() if getattr(c, "updated_at", None) else None,
        }
        for c in rows
    ]
