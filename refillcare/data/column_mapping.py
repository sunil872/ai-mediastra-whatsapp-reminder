"""Map any store's sales-file headers onto the RefillCare schema (shared by FastAPI and Streamlit).

Order of resolution for every upload:

1. **user mapping** sent with the upload (from the confirm screen)            -> used as given
2. **saved profile** for this store + this exact header layout (fingerprint)  -> used, no questions
3. **auto-detect**: header aliases (``config/column_aliases.json``) scored together with
   what the values look like (dates parse, 10-digit mobiles, S0/ / SB/ invoices, 1X15 packs,
   numeric quantities). A saved profile of the same store with a different layout is used as a
   hint for headers it already knows.

Confirmation is only needed when a required field is missing or matched with low confidence;
in the usual case (known header names) the file is mapped automatically and the mapping is
saved for the store, so the next upload with the same layout is instant.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from refillcare.data.schema import (
    EXPECTED_FIELDS,
    FIELD_LABELS,
    MAPPABLE_FIELDS,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    REQUIRED_ONE_OF,
    missing_required,
)

ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "column_aliases.json"
AUTO_ACCEPT = 0.75        # confidence needed to map a field without asking
SUGGEST_MIN = 0.30        # weaker candidates are only offered as a pre-selected suggestion
NAME_WEIGHT, VALUE_WEIGHT = 0.65, 0.35
SAMPLE_ROWS = 300
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_PACK_RE = re.compile(r"\d+\s*[X*]\s*\d+|\d+(\.\d+)?\s*(ML|GM|G|MG|TAB|TABS|CAP|CAPS|S|N|KIT|PEN|VIAL)\b|^\d+$", re.IGNORECASE)
_LETTERS = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_aliases_cache: Optional[Dict[str, Dict[str, List[str]]]] = None


def normalise_header(h: Any) -> str:
    return _NON_ALNUM.sub("", str(h).lower())


def load_aliases(path: Optional[Path] = None) -> Dict[str, Dict[str, List[str]]]:
    global _aliases_cache
    if path is None and _aliases_cache is not None:
        return _aliases_cache
    raw = json.loads(Path(path or ALIASES_PATH).read_text(encoding="utf-8"))
    out = {f: {"strong": [normalise_header(a) for a in v.get("strong", [])], "weak": [normalise_header(a) for a in v.get("weak", [])]}
           for f, v in raw.items() if not f.startswith("_")}
    if path is None:
        _aliases_cache = out
    return out


def header_fingerprint(headers: List[str]) -> str:
    return hashlib.sha1("|".join(sorted(normalise_header(h) for h in headers)).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- scoring
def _name_score(header: str, aliases: Dict[str, List[str]]) -> float:
    h = normalise_header(header)
    if not h:
        return 0.0
    if h in aliases["strong"]:
        return 1.0
    if h in aliases["weak"]:
        return 0.8
    # 'customermobileno' contains 'mobileno'; never the reverse ('units' must not match 'freeunits')
    if any(len(a) >= 5 and a in h for a in aliases["strong"]):
        return 0.85
    best = max((difflib.SequenceMatcher(None, h, a).ratio() for a in aliases["strong"]), default=0.0)
    return round(0.75 * best, 3) if best >= 0.85 else 0.0


def _sample(series: pd.Series) -> List[str]:
    vals = [str(v).strip() for v in series.head(SAMPLE_ROWS * 3).tolist() if v is not None and str(v).strip() not in ("", "nan", "None", "NaT")]
    return vals[:SAMPLE_ROWS]


def _share(vals: List[str], pred) -> float:
    return sum(1 for v in vals if pred(v)) / len(vals) if vals else 0.0


def _is_number(v: str) -> bool:
    try:
        float(v.replace(",", ""))
        return True
    except ValueError:
        return False


def _is_date(v: str) -> bool:
    if _is_number(v) and len(v) < 6:
        return False
    return not pd.isna(pd.to_datetime(v, errors="coerce", dayfirst=True))


def _is_mobile(v: str) -> bool:
    from refillcare.data.identity import mobile10

    s = v[: v.index(".")] if re.match(r"^\d+\.0+$", v) else v
    return mobile10(s) is not None


def value_score(fld: str, series: pd.Series) -> Optional[float]:
    """0..1: how much the column's values look like ``fld``; None when the column is empty."""
    vals = _sample(series)
    if not vals:
        return None
    if fld == "invoice_date":
        return _share(vals[:60], _is_date)
    if fld == "invoice_number":
        return _share(vals[:60], lambda v: bool(_DIGIT.search(v)) and not _is_date(v))
    if fld == "customerName":
        return _share(vals, lambda v: bool(_LETTERS.search(v)) and not _is_number(v))
    if fld == "customerId":
        return _share(vals, lambda v: bool(_DIGIT.search(v)) and len(v) <= 20)
    if fld == "itemCode":
        return _share(vals, lambda v: bool(_DIGIT.search(v)) and len(v) <= 20 and " " not in v)
    if fld == "itemName":
        return _share(vals, lambda v: bool(_LETTERS.search(v)) and len(v) >= 3 and not _is_number(v))
    if fld == "packing":
        return _share(vals, lambda v: bool(_PACK_RE.search(v)))
    if fld in ("quantity", "freeQuantity"):
        return _share(vals, _is_number)
    if fld == "MOBILE_NO":
        return _share(vals, _is_mobile)
    return 0.0


def _combined(name: float, value: Optional[float]) -> float:
    if value is None:
        return round(name * 0.9, 3)
    return round(NAME_WEIGHT * name + VALUE_WEIGHT * value, 3)


# ----------------------------------------------------------------------------- result
@dataclass
class FieldMatch:
    field: str
    column: Optional[str]
    confidence: float
    source: str  # user | saved | saved_hint | auto | suggestion | none
    samples: List[str] = field(default_factory=list)


@dataclass
class MappingResult:
    store_id: str
    fingerprint: str
    headers: List[str]
    matches: Dict[str, FieldMatch]
    profile_id: Optional[int] = None

    @property
    def mapping(self) -> Dict[str, Optional[str]]:
        """Only accepted matches (suggestions below the auto threshold are not applied)."""
        return {f: m.column for f, m in self.matches.items() if m.column and m.source != "suggestion"}

    @property
    def missing_required(self) -> List[str]:
        return missing_required(self.mapping)

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.missing_required)

    @property
    def ignored_columns(self) -> List[str]:
        used = set(self.mapping.values())
        return [h for h in self.headers if h not in used]

    def warnings(self) -> List[str]:
        m = self.mapping
        out = []
        if not m.get("invoice_number"):
            out.append("No invoice-number column: B2B transfers (SB/) cannot be told apart, so every row is treated as a customer sale.")
        if not m.get("itemCode"):
            out.append("No item-code column: the item name is used as the item key and for the salt lookup.")
        if not m.get("packing"):
            out.append("No packing column: pack size is taken from the item catalogue (by item code) where available.")
        if not m.get("freeQuantity"):
            out.append("No free-quantity column: free strips are not added to days of supply (fine if the store gives none).")
        return out

    def to_dict(self) -> Dict[str, Any]:
        def role(f: str) -> str:
            if f in REQUIRED_FIELDS or any(f in pair for pair in REQUIRED_ONE_OF):
                return "required"
            return "expected" if f in EXPECTED_FIELDS else "optional"

        return {
            "store_id": self.store_id,
            "header_fingerprint": self.fingerprint,
            "headers": self.headers,
            "profile_id": self.profile_id,
            "mapping": self.mapping,
            "fields": [
                {"field": f, "label": FIELD_LABELS.get(f, f), "role": role(f), "column": m.column,
                 "confidence": m.confidence, "source": m.source, "samples": m.samples}
                for f, m in self.matches.items()
            ],
            "missing_required": self.missing_required,
            "needs_confirmation": self.needs_confirmation,
            "ignored_columns": self.ignored_columns,
            "warnings": self.warnings(),
        }


def _samples_for(df: pd.DataFrame, col: Optional[str]) -> List[str]:
    return _sample(df[col])[:3] if col and col in df.columns else []


def auto_detect(df: pd.DataFrame, hint: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, FieldMatch]:
    """Greedy one-to-one assignment of headers to fields by combined name + value confidence."""
    aliases = load_aliases()
    headers = [str(c) for c in df.columns]
    vscores: Dict[Tuple[str, str], Optional[float]] = {}
    cands: List[Tuple[float, str, str, float]] = []
    for f in MAPPABLE_FIELDS:
        al = aliases.get(f, {"strong": [], "weak": []})
        for h in headers:
            ns = _name_score(h, al)
            if hint and hint.get(f) == h:
                ns = max(ns, 1.0)
            vs = value_score(f, df[h])
            vscores[(f, h)] = vs
            score = _combined(ns, vs) if ns > 0 else round(VALUE_WEIGHT * (vs or 0.0), 3)
            if score > 0:
                cands.append((score, f, h, ns))
    cands.sort(key=lambda t: (-t[0], -t[3]))
    taken_f, taken_h = set(), set()
    out: Dict[str, FieldMatch] = {}
    for score, f, h, ns in cands:
        if f in taken_f or h in taken_h:
            continue
        if score >= AUTO_ACCEPT:
            src = "saved_hint" if hint and hint.get(f) == h else "auto"
        elif score >= SUGGEST_MIN and ns > 0:
            src = "suggestion"
        else:
            continue
        out[f] = FieldMatch(f, h, score, src, _samples_for(df, h))
        taken_f.add(f)
        if src != "suggestion":
            taken_h.add(h)
    # Unnamed required fields: pre-select the free column whose values fit best (never auto-applied)
    needed = [f for f in REQUIRED_FIELDS if f not in out] + [a for a, b in REQUIRED_ONE_OF if a not in out and b not in out]
    for f in needed:
        best = max(((vscores.get((f, h)) or 0.0, h) for h in headers if h not in taken_h), default=(0.0, None))
        if best[1] is not None and best[0] >= 0.8:
            out[f] = FieldMatch(f, best[1], round(VALUE_WEIGHT * best[0], 3), "suggestion", _samples_for(df, best[1]))
    for f in MAPPABLE_FIELDS:
        out.setdefault(f, FieldMatch(f, None, 0.0, "none"))
    return {f: out[f] for f in MAPPABLE_FIELDS}


def resolve_mapping(
    df: pd.DataFrame,
    db: Any = None,
    store_id: Optional[str] = None,
    user_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> MappingResult:
    from refillcare.store import StoreConfig

    store_id = store_id or StoreConfig.from_env().store_id
    headers = [str(c) for c in df.columns]
    fp = header_fingerprint(headers)

    if user_mapping:
        unknown = {f: c for f, c in user_mapping.items() if c and c not in headers}
        if unknown:
            raise ValueError(f"Mapped columns not in the file: {unknown}")
        matches = {f: FieldMatch(f, user_mapping.get(f) or None, 1.0 if user_mapping.get(f) else 0.0,
                                 "user" if user_mapping.get(f) else "none", _samples_for(df, user_mapping.get(f)))
                   for f in MAPPABLE_FIELDS}
        return MappingResult(store_id, fp, headers, matches)

    profile = get_profile(db, store_id, fp) if db is not None else None
    if profile is not None:
        saved = dict(profile.mapping or {})
        matches = {f: FieldMatch(f, saved.get(f) if saved.get(f) in headers else None,
                                 1.0 if saved.get(f) in headers else 0.0,
                                 "saved" if saved.get(f) in headers else "none", _samples_for(df, saved.get(f)))
                   for f in MAPPABLE_FIELDS}
        return MappingResult(store_id, fp, headers, matches, profile_id=profile.profile_id)

    hint = latest_profile_mapping(db, store_id) if db is not None else None
    return MappingResult(store_id, fp, headers, auto_detect(df, hint))


# ----------------------------------------------------------------------------- saved profiles
def get_profile(db: Any, store_id: str, fingerprint: str):
    from database.models import ColumnMappingProfileModel

    return db.query(ColumnMappingProfileModel).filter_by(store_id=store_id, header_fingerprint=fingerprint).first()


def latest_profile_mapping(db: Any, store_id: str) -> Optional[Dict[str, Optional[str]]]:
    from database.models import ColumnMappingProfileModel

    p = (db.query(ColumnMappingProfileModel).filter_by(store_id=store_id)
         .order_by(ColumnMappingProfileModel.updated_at.desc()).first())
    return dict(p.mapping) if p is not None else None


def save_profile(db: Any, result_or_headers: Any, mapping: Optional[Dict[str, Optional[str]]] = None,
                 store_id: Optional[str] = None, source: str = "user", confirmed_by: str = "system"):
    """Create / update the store's profile for this header layout. Validates required fields."""
    from database.models import ColumnMappingProfileModel
    from refillcare.store import StoreConfig

    if isinstance(result_or_headers, MappingResult):
        headers, store_id, mapping = result_or_headers.headers, result_or_headers.store_id, mapping or result_or_headers.mapping
    else:
        headers = [str(h) for h in result_or_headers]
        store_id = store_id or StoreConfig.from_env().store_id
    mapping = {f: (mapping or {}).get(f) or None for f in MAPPABLE_FIELDS}
    bad = {f: c for f, c in mapping.items() if c and c not in headers}
    if bad:
        raise ValueError(f"Mapped columns not in the file: {bad}")
    used = [c for c in mapping.values() if c]
    if len(used) != len(set(used)):
        raise ValueError("One file column cannot be mapped to two fields.")
    miss = missing_required(mapping)
    if miss:
        raise ValueError(f"Required fields not mapped: {', '.join(FIELD_LABELS.get(m, m) for m in miss)}")
    fp = header_fingerprint(headers)
    p = get_profile(db, store_id, fp)
    now = datetime.utcnow()
    if p is None:
        p = ColumnMappingProfileModel(store_id=store_id, header_fingerprint=fp, headers=headers, mapping=mapping,
                                      source=source, confirmed_by=confirmed_by, created_at=now, updated_at=now)
        db.add(p)
    else:
        p.headers, p.mapping, p.source, p.confirmed_by, p.updated_at = headers, mapping, source, confirmed_by, now
    db.commit()
    return p


def mark_profile_used(db: Any, result: MappingResult, confirmed_by: str = "auto") -> None:
    """After a successful import: remember the mapping (auto-detected ones too) and count the use."""
    p = get_profile(db, result.store_id, result.fingerprint)
    if p is None:
        p = save_profile(db, result, source="auto" if all(m.source in ("auto", "saved_hint", "none") for m in result.matches.values()) else "user",
                         confirmed_by=confirmed_by)
    p.use_count = int(p.use_count or 0) + 1
    p.last_used_at = datetime.utcnow()
    db.commit()


def list_profiles(db: Any, store_id: Optional[str] = None) -> List[Dict[str, Any]]:
    from database.models import ColumnMappingProfileModel

    q = db.query(ColumnMappingProfileModel)
    if store_id:
        q = q.filter_by(store_id=store_id)
    return [{"profile_id": p.profile_id, "store_id": p.store_id, "header_fingerprint": p.header_fingerprint,
             "headers": p.headers, "mapping": p.mapping, "source": p.source, "confirmed_by": p.confirmed_by,
             "use_count": p.use_count, "created_at": p.created_at, "updated_at": p.updated_at, "last_used_at": p.last_used_at}
            for p in q.order_by(ColumnMappingProfileModel.updated_at.desc()).all()]


def delete_profile(db: Any, profile_id: int) -> bool:
    from database.models import ColumnMappingProfileModel

    p = db.get(ColumnMappingProfileModel, profile_id)
    if p is None:
        return False
    db.delete(p)
    db.commit()
    return True
