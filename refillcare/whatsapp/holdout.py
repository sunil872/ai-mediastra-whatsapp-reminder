"""No-reminder control group (holdout) - measures the real effect of reminders (uplift).

A fixed share of patients (``REFILLCARE_HOLDOUT_PCT``, default 0 = off since 11 Nov 2026: the client wants every patient reminded; set e.g. 10 to measure uplift) is never messaged. Comparing
their refill rate with reminded patients shows how many extra refills the reminders cause - the
number the client cares about, independent of point-date accuracy.

Assignment is deterministic (hash of the mobile number, or the customer ID when there is no
number), so a patient stays in the same group every day and every send path (Streamlit, API,
batch, Med-Sync) agrees without storing anything. Household members sharing a phone share a group.
``REFILLCARE_HOLDOUT_SALT`` re-draws the groups (e.g. for a new pilot); ``REFILLCARE_HOLDOUT_PCT=0``
turns the control group off.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

from refillcare.whatsapp.consent import phone_key

DEFAULT_HOLDOUT_PCT = 0  # off: every patient with a valid mobile is reminded (11 Nov 2026)
MAX_HOLDOUT_PCT = 50


def holdout_pct() -> int:
    try:
        pct = int(os.getenv("REFILLCARE_HOLDOUT_PCT", str(DEFAULT_HOLDOUT_PCT)))
    except ValueError:
        pct = DEFAULT_HOLDOUT_PCT
    return max(0, min(MAX_HOLDOUT_PCT, pct))


def is_holdout(customer_id: Any = None, phone: Optional[Any] = None) -> bool:
    """True if this patient is in the no-reminder control group."""
    pct = holdout_pct()
    if pct <= 0:
        return False
    key = phone_key(phone) or (str(customer_id).strip() if customer_id is not None else "")
    if not key:
        return False
    salt = os.getenv("REFILLCARE_HOLDOUT_SALT", "refillcare-holdout-v1")
    bucket = int(hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < pct
