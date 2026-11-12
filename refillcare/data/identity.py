"""Who is the patient, and which therapy is the medicine? (trajectory keys for scoring and ML)

Patient identity (``REFILLCARE_PATIENT_IDENTITY``, default ``name_mobile`` since 10 Nov 2026)
    ``name``         legacy: the billing name is the patient ("LAXMI" = every Laxmi in the store).
                     Unsafe: another person with the same name could cancel a patient's reminders and
                     move them to his own phone (scenario test, CHANDRA MOHAN with 5 phones).
    ``name_mobile``  DEFAULT. name + last 10 mobile digits ("LAXMI_9876543210"). Two Laxmis with different
                     phones are two patients; a family sharing one phone stays split by name.
                     A purchase without a valid mobile borrows the phone only when that name has
                     exactly one phone in the whole history; otherwise it stays name-only.

Therapy grouping (``REFILLCARE_ITEM_GROUPING``)
    ``item``  legacy: every brand / strength is its own trajectory.
    ``salt``  brands and strengths of the same molecule(s) from the SALT ITEMS LIST sheet form one
              trajectory ("TELMA 40" -> "TELMIKIND 80" continues the same refill cycle). Merged only
              when the patient bought them one after the other; brands bought side by side (two
              insulins, two metformin strengths taken together) stay separate. Combinations are
              their own therapy (telmisartan + amlodipine != telmisartan).

Both are idempotent: rows that carry ``source_customer_id`` / ``brand_item_id`` are left alone.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

IDENTITY_MODES = ("name", "name_mobile")
GROUPING_MODES = ("item", "salt")
SOURCE_CUSTOMER_COL = "source_customer_id"
BRAND_ITEM_COL = "brand_item_id"
IDENTITY_SOURCE_COL = "identity_source"
SWITCH_OVERLAP_DAYS = 7      # a new brand may start up to a week before the old one's last purchase
MAX_NAME_CHARS = 30          # keeps cycle ids within the 64-char columns

_NON_DIGIT = re.compile(r"\D")
_STRENGTH = re.compile(
    r"\(?\s*\d+(?:\.\d+)?\s*(?:MG|MCG|IU|GM|G|ML|%|U|UNITS?)?\s*(?:W/W|W/V|/\s*\d*\s*ML)?\s*\)?",
    re.IGNORECASE,
)
_TAGS = re.compile(r"\[\d+\]|\b(?:SR|ER|XR|CR|MR|OD|DS|FORTE)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^A-Z+ ]+")
_SPACES = re.compile(r"\s+")


def identity_mode() -> str:
    from refillcare.env import ensure_dotenv

    ensure_dotenv()
    from refillcare.stores import store_setting

    m = (store_setting("REFILLCARE_PATIENT_IDENTITY", "name_mobile") or "name_mobile").strip().lower()
    return m if m in IDENTITY_MODES else "name_mobile"


def grouping_mode() -> str:
    from refillcare.env import ensure_dotenv

    ensure_dotenv()
    from refillcare.stores import store_setting

    m = (store_setting("REFILLCARE_ITEM_GROUPING", "item") or "item").strip().lower()
    return m if m in GROUPING_MODES else "item"


def mobile10(val: Any) -> Optional[str]:
    """Last 10 digits of an Indian mobile (starts 6-9), else None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    d = _NON_DIGIT.sub("", str(val))
    if len(d) < 10:
        return None
    d = d[-10:]
    return d if d[0] in "6789" else None


def patient_id(name: Any, mobile: Optional[str]) -> str:
    n = _SPACES.sub(" ", str(name or "").strip().upper())[:MAX_NAME_CHARS].strip()
    if not mobile:
        return n
    return f"{n}_{mobile}" if n else mobile


def salt_molecules(salt: Any) -> Optional[str]:
    """'AMLODIPINE-5Mg+TELMISARTAN-40Mg [00]' -> 'AMLODIPINE+TELMISARTAN' (strength / brand free)."""
    if salt is None or (isinstance(salt, float) and np.isnan(salt)):
        return None
    s = _TAGS.sub(" ", str(salt).upper())
    parts = []
    for p in s.split("+"):
        p = _STRENGTH.sub(" ", p.replace("-", " "))
        p = _SPACES.sub(" ", _PUNCT.sub(" ", p)).strip()
        if len(p) >= 3:
            parts.append(p)
    return "+".join(sorted(set(parts))) or None


def therapy_item_id(molecules: str) -> str:
    return "SALT-" + hashlib.sha1(molecules.encode("utf-8")).hexdigest()[:10].upper()


def apply_patient_identity(df: pd.DataFrame, mode: Optional[str] = None) -> pd.DataFrame:
    mode = mode or identity_mode()
    if df.empty or "customerId" not in df.columns or SOURCE_CUSTOMER_COL in df.columns:
        return df
    out = df.copy()
    out[SOURCE_CUSTOMER_COL] = out["customerId"].astype(str).str.strip()
    if mode != "name_mobile" or "MOBILE_NO" not in out.columns:
        out[IDENTITY_SOURCE_COL] = "name"
        return out
    m10 = out["MOBILE_NO"].map(mobile10)
    phones_per_name = m10.groupby(out[SOURCE_CUSTOMER_COL]).agg(lambda s: set(s.dropna()))
    only_phone = {n: next(iter(p)) for n, p in phones_per_name.items() if len(p) == 1}
    borrowed = m10.isna() & out[SOURCE_CUSTOMER_COL].map(only_phone).notna()
    m10 = m10.where(~borrowed, out[SOURCE_CUSTOMER_COL].map(only_phone))
    out["customerId"] = [patient_id(n, p if isinstance(p, str) else None) for n, p in zip(out[SOURCE_CUSTOMER_COL], m10)]
    out[IDENTITY_SOURCE_COL] = np.where(m10.isna(), "name_only", np.where(borrowed, "name_mobile_borrowed", "name_mobile"))
    return out


def apply_therapy_grouping(df: pd.DataFrame, mode: Optional[str] = None) -> pd.DataFrame:
    mode = mode or grouping_mode()
    if df.empty or "itemId" not in df.columns or BRAND_ITEM_COL in df.columns:
        return df
    out = df.copy()
    out[BRAND_ITEM_COL] = out["itemId"].astype(str).str.strip()
    salt_col = next((c for c in ("salt_composition", "SALT") if c in out.columns), None)
    if mode != "salt" or salt_col is None:
        return out
    salts = out[salt_col]
    if "itemName" in out.columns:
        # Same lookup as the engine's salt priors: the SALT ITEMS LIST sheet when the row has no salt
        from refillcare.features.context import salt_key

        missing = salts.isna() | (salts.astype(str).str.strip() == "")
        if missing.any():
            names = out.loc[missing, "itemName"]
            lookup = {n: salt_key(None, n) for n in names.dropna().unique()}
            salts = salts.copy()
            salts.loc[missing] = names.map(lookup)
    mol = salts.map(salt_molecules)
    dates = pd.to_datetime(out["invoice_date"], errors="coerce")
    frame = pd.DataFrame({"c": out["customerId"].astype(str), "i": out[BRAND_ITEM_COL], "m": mol, "d": dates}).dropna()
    if frame.empty:
        return out
    spans = frame.groupby(["c", "m", "i"])["d"].agg(["min", "max"]).reset_index().sort_values(["c", "m", "min"])
    gap = pd.Timedelta(days=SWITCH_OVERLAP_DAYS)
    merge_keys = set()
    for (c, m), g in spans.groupby(["c", "m"], sort=False):
        if len(g) < 2:
            continue
        starts, ends = g["min"].tolist(), g["max"].tolist()
        if all(starts[k] >= ends[k - 1] - gap for k in range(1, len(starts))):
            merge_keys.add((c, m))
    if not merge_keys:
        return out
    key = list(zip(out["customerId"].astype(str), mol))
    hit = np.array([k in merge_keys for k in key])
    out.loc[hit, "itemId"] = [therapy_item_id(m) for m in mol[hit]]
    return out


def normalise_trajectory_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Patient identity first (switches are per patient), then therapy grouping."""
    return apply_therapy_grouping(apply_patient_identity(df))


def check_identity_setting_warning(db: Optional[Any] = None) -> Optional[str]:
    """Verify that stored active reminder cycle keys match the configured REFILLCARE_PATIENT_IDENTITY.

    If REFILLCARE_PATIENT_IDENTITY is changed by hand in .env without running the identity re-key,
    keys in the database will diverge from keys generated during new scoring.
    Returns a warning string if mismatch detected, else None.
    """
    from database.connection import SessionLocal
    from database.models import ReminderCycleModel, RefillDecisionModel

    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        mode = identity_mode()
        cycles = db.query(ReminderCycleModel.customer_id).filter(ReminderCycleModel.is_active == True).limit(100).all()
        if not cycles:
            # Fallback to recent decisions if no active cycles exist
            cycles = db.query(RefillDecisionModel.customer_id).limit(100).all()
            if not cycles:
                return None

        mobile_suffixed = sum(1 for (cid,) in cycles if re.search(r"_\d{10}$|^\d{10}$", str(cid or "")))
        ratio = mobile_suffixed / len(cycles)

        if mode == "name" and ratio > 0.1:
            return (
                f"REFILLCARE_PATIENT_IDENTITY is 'name', but active reminder cycles use 'name_mobile' keys. "
                "Run the identity re-key (/api/v1/ml/identity/rekey or scripts/rebuild_legacy_history.py) to synchronize."
            )
        elif mode == "name_mobile" and len(cycles) >= 20 and ratio == 0.0:
            return (
                f"REFILLCARE_PATIENT_IDENTITY is 'name_mobile', but active reminder cycles use legacy 'name' keys. "
                "Run the identity re-key (/api/v1/ml/identity/rekey or scripts/rebuild_legacy_history.py) to synchronize."
            )
        return None
    except Exception:
        return None
    finally:
        if close_db:
            db.close()

