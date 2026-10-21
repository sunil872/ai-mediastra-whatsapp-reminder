"""Cross-item context for point-in-time refill features.

Two lookups the per-item cadence features cannot see on their own:

* ``VisitIndex`` - the patient's purchase timeline across **all** medicines (visit-level signal:
  patients buy several chronic medicines in one visit, so another medicine coming due pulls the
  next visit forward).
* ``PopulationStats`` - salt-level priors learned from the store's own history (the SALT ITEMS LIST
  ITEMS catalogue is already merged into ``salt_composition``): typical refill interval and a
  **default dosage per day** (units consumed per day) for each salt. Used for short histories,
  where a patient's own cadence is too noisy.

Both are point-in-time safe: a ``VisitIndex`` query only sees purchases dated on or before the
as-of date, and ``PopulationStats`` is built from purchases up to an explicit cut-off.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from refillcare.data.packing import parse_pack_units

_SUFFIX_RE = re.compile(r"\s*\[\d+\]\s*$")
_SPACE_RE = re.compile(r"\s+")

VISIT_LOOKBACK_DAYS = 180
STATS_MIN_CUSTOMER_ITEMS = 10
STATS_MIN_INTERVALS = 2
INTERVAL_MIN_DAYS, INTERVAL_MAX_DAYS = 5, 180


def salt_key(salt: Any = None, item_name: Any = None) -> Optional[str]:
    """Normalised salt + strength key ('TELMISARTAN-40MG'); catalogue lookup when salt is missing."""
    s = "" if salt is None or (isinstance(salt, float) and np.isnan(salt)) else str(salt)
    if not s.strip() and item_name:
        try:
            from refillcare.data.medication_classifier import get_catalog_item_salt

            hit = get_catalog_item_salt(str(item_name)) or {}
            s = hit.get("salt", "") or ""
        except Exception:
            s = ""
    s = _SPACE_RE.sub(" ", _SUFFIX_RE.sub("", s.strip().upper()))
    return s or None


# ----------------------------------------------------------------------------- visits
class VisitIndex:
    """Per-patient purchase timeline across all items (point-in-time queries)."""

    def __init__(self) -> None:
        self._visits: Dict[str, List[int]] = {}
        self._items: Dict[str, Dict[str, List[int]]] = {}

    @classmethod
    def from_events(cls, events: pd.DataFrame, customer_col: str = "customerId", item_col: str = "itemId", date_col: str = "date") -> "VisitIndex":
        idx = cls()
        if events is None or events.empty:
            return idx
        df = events[[customer_col, item_col, date_col]].dropna()
        ords = pd.to_datetime(df[date_col]).dt.date.map(date.toordinal).to_numpy()
        cids = df[customer_col].astype(str).str.strip().to_numpy()
        iids = df[item_col].astype(str).str.strip().to_numpy()
        visits: Dict[str, set] = {}
        items: Dict[str, Dict[str, set]] = {}
        for c, i, o in zip(cids, iids, ords):
            visits.setdefault(c, set()).add(int(o))
            items.setdefault(c, {}).setdefault(i, set()).add(int(o))
        idx._visits = {c: sorted(v) for c, v in visits.items()}
        idx._items = {c: {i: sorted(d) for i, d in its.items()} for c, its in items.items()}
        return idx

    @classmethod
    def from_trajectories(cls, trajectories: Dict[Any, List[date]]) -> "VisitIndex":
        """From ``{(customer_id, item_id): [purchase dates]}`` (the engine's grouped histories)."""
        idx = cls()
        visits: Dict[str, set] = {}
        for (cid, iid), ds in trajectories.items():
            c, i = str(cid).strip(), str(iid).strip()
            ords = {d.toordinal() for d in ds}
            if not ords:
                continue
            visits.setdefault(c, set()).update(ords)
            idx._items.setdefault(c, {})[i] = sorted(ords)
        idx._visits = {c: sorted(v) for c, v in visits.items()}
        return idx

    def context(self, customer_id: Any, item_id: Any, as_of: date) -> Dict[str, Optional[float]]:
        """Visit features for ``customer_id`` at ``as_of`` (only purchases <= as_of)."""
        cid, iid, t = str(customer_id).strip(), str(item_id).strip(), as_of.toordinal()
        out: Dict[str, Optional[float]] = {
            "days_since_prev_visit_any_item": None,
            "visits_last_180d": None,
            "active_items_180d": None,
            "other_item_next_due_days": None,
            "other_items_due_within_30d": None,
        }
        visits = self._visits.get(cid)
        if not visits:
            return out
        k = bisect.bisect_right(visits, t)
        past = visits[:k]
        prev = [v for v in past if v < t]
        out["days_since_prev_visit_any_item"] = float(t - prev[-1]) if prev else None
        lo = bisect.bisect_left(past, t - VISIT_LOOKBACK_DAYS)
        out["visits_last_180d"] = float(len(past) - lo)

        active, due_in, n_due_30 = 0, [], 0
        for other, ds in self._items.get(cid, {}).items():
            j = bisect.bisect_right(ds, t)
            if j == 0:
                continue
            seen = ds[:j]
            if seen[-1] >= t - VISIT_LOOKBACK_DAYS:
                active += 1
            if other == iid or len(seen) < 2:
                continue
            gaps = [b - a for a, b in zip(seen[:-1], seen[1:]) if INTERVAL_MIN_DAYS <= b - a <= INTERVAL_MAX_DAYS]
            if not gaps:
                continue
            due = seen[-1] + float(np.median(gaps)) - t
            if -30 <= due <= 120:
                due_in.append(due)
                if 0 <= due <= 30:
                    n_due_30 += 1
        out["active_items_180d"] = float(active)
        out["other_item_next_due_days"] = float(min(due_in, key=abs)) if due_in else None
        out["other_items_due_within_30d"] = float(n_due_30)
        return out


# ----------------------------------------------------------------------------- salt priors
@dataclass
class PopulationStats:
    """Salt-level refill priors: median interval and default dosage (units per day)."""

    as_of: Optional[str] = None
    by_salt: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def lookup(self, key: Optional[str]) -> Optional[Dict[str, Optional[float]]]:
        return self.by_salt.get(key) if key else None

    def to_dict(self) -> Dict[str, Any]:
        return {"as_of": self.as_of, "by_salt": self.by_salt}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PopulationStats":
        d = d or {}
        return cls(as_of=d.get("as_of"), by_salt=dict(d.get("by_salt") or {}))

    @classmethod
    def from_events(cls, events: pd.DataFrame, as_of: Optional[date] = None) -> "PopulationStats":
        """Build from purchase events (customerId, itemId, date, quantity, packing, salt, itemName).

        Only purchases dated <= ``as_of`` are used. Each customer-item contributes one vote
        (its own median interval / units-per-day), so frequent buyers don't dominate.
        """
        if events is None or events.empty:
            return cls(as_of=as_of.isoformat() if as_of else None)
        df = events.copy()
        df["date"] = pd.to_datetime(df["date"])
        if as_of is not None:
            df = df[df["date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return cls(as_of=as_of.isoformat() if as_of else None)

        salt_src = df["salt"] if "salt" in df.columns else pd.Series([None] * len(df), index=df.index)
        names = df["itemName"] if "itemName" in df.columns else pd.Series([None] * len(df), index=df.index)
        cache: Dict[Any, Optional[str]] = {}

        def _key(s: Any, n: Any) -> Optional[str]:
            k = (s if isinstance(s, str) else None, n if isinstance(n, str) else None)
            if k not in cache:
                cache[k] = salt_key(*k)
            return cache[k]

        df["_salt"] = [_key(s, n) for s, n in zip(salt_src, names)]
        df = df[df["_salt"].notna()]
        packs = df["packing"].astype(str) if "packing" in df.columns else pd.Series([""] * len(df), index=df.index)
        pack_units = packs.map({p: (parse_pack_units(p) or np.nan) for p in packs.unique()})
        qty = pd.to_numeric(df.get("quantity"), errors="coerce") if "quantity" in df.columns else pd.Series(np.nan, index=df.index)
        df["_units"] = qty.where(qty > 0) * pack_units

        df = df.sort_values(["customerId", "itemId", "date"])
        grp = df.groupby(["customerId", "itemId"], sort=False)
        df["_gap"] = grp["date"].diff().dt.days
        df["_prev_units"] = grp["_units"].shift(1)
        ok = df["_gap"].between(INTERVAL_MIN_DAYS, INTERVAL_MAX_DAYS)
        iv = df[ok].copy()
        iv["_upd"] = iv["_prev_units"] / iv["_gap"]
        per_ci = iv.groupby(["customerId", "itemId", "_salt"]).agg(
            n_int=("_gap", "size"), med_int=("_gap", "median"), upd=("_upd", "median")
        ).reset_index()
        per_ci = per_ci[per_ci["n_int"] >= STATS_MIN_INTERVALS]
        agg = per_ci.groupby("_salt").agg(
            n=("med_int", "size"), median_interval=("med_int", "median"), units_per_day=("upd", "median")
        )
        agg = agg[agg["n"] >= STATS_MIN_CUSTOMER_ITEMS]
        by_salt = {
            str(k): {
                "n": float(r["n"]),
                "median_interval": round(float(r["median_interval"]), 2),
                "units_per_day": round(float(r["units_per_day"]), 4) if pd.notna(r["units_per_day"]) else None,
            }
            for k, r in agg.iterrows()
        }
        return cls(as_of=as_of.isoformat() if as_of else None, by_salt=by_salt)


def build_context(
    visit_index: Optional[VisitIndex],
    stats: Optional[PopulationStats],
    customer_id: Any,
    item_id: Any,
    as_of: date,
    salt: Any = None,
    item_name: Any = None,
) -> Dict[str, Any]:
    """Everything the feature function needs beyond the item's own history."""
    ctx: Dict[str, Any] = {}
    if visit_index is not None:
        ctx["visit"] = visit_index.context(customer_id, item_id, as_of)
    if stats is not None:
        ctx["salt_stats"] = stats.lookup(salt_key(salt, item_name))
    return ctx
