"""Store registry: which stores exist, their name / contact and per-store settings.

Per-store ``settings`` override the ``.env`` defaults for that store only, e.g.
``{"REFILLCARE_PATIENT_IDENTITY": "name_mobile", "REFILLCARE_RULE_PATH_B_MODE": "blend"}`` - so one
store can switch a rule on without affecting the others.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from refillcare.tenancy import current_store_id, default_store_id, is_production, normalise_store_id

_CACHE_TTL = 60.0
_cache: Dict[str, tuple] = {}
_lock = threading.Lock()
# Settings a store may override (anything else in `settings` is ignored)
OVERRIDABLE_SETTINGS = (
    "REFILLCARE_PATIENT_IDENTITY", "REFILLCARE_ITEM_GROUPING", "REFILLCARE_RULE_PATH_B_MODE",
    "REFILLCARE_RULE_DEVICE_CAP_MODE", "REFILLCARE_RULE_HIGH_STABILITY_MEDIAN", "REFILLCARE_RULE_INSULIN_MEDIAN",
    "REFILLCARE_RULE_DOM_ANCHOR", "REFILLCARE_MAX_DATA_LAG_DAYS", "REFILLCARE_REQUIRE_INVOICE_NUMBER",
)


class UnknownStore(LookupError):
    pass


def _row_to_dict(s) -> Dict[str, Any]:
    return {"store_id": s.store_id, "tenant_id": s.tenant_id, "store_name": s.store_name, "store_contact": s.store_contact,
            "settings": dict(s.settings or {}), "status": s.status, "created_at": s.created_at, "updated_at": s.updated_at}


def get_store(store_id: Optional[str] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    from database.connection import ControlSession
    from database.control_models import StoreModel

    sid = normalise_store_id(store_id) if store_id else current_store_id()
    now = time.monotonic()
    if use_cache:
        with _lock:
            hit = _cache.get(sid)
            if hit and now - hit[0] < _CACHE_TTL:
                return hit[1]
    with ControlSession() as c:
        row = c.get(StoreModel, sid)
        data = _row_to_dict(row) if row is not None else None
    with _lock:
        _cache[sid] = (now, data)
    return data


def register_store(store_id: str, tenant_id: Optional[str] = None, store_name: Optional[str] = None,
                   store_contact: Optional[str] = None, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create or update a store and create its database schema / tables."""
    from database.connection import ControlSession, init_store
    from database.control_models import StoreModel

    sid = normalise_store_id(store_id)
    clean = {k: v for k, v in (settings or {}).items() if k in OVERRIDABLE_SETTINGS}
    with ControlSession() as c:
        row = c.get(StoreModel, sid)
        if row is None:
            row = StoreModel(store_id=sid, settings={})
            c.add(row)
        if tenant_id is not None:
            row.tenant_id = tenant_id
        if store_name is not None:
            row.store_name = store_name
        if store_contact is not None:
            row.store_contact = store_contact
        if settings is not None:
            row.settings = {**(row.settings or {}), **clean}
        c.commit()
        data = _row_to_dict(row)
    with _lock:
        _cache.pop(sid, None)
    init_store(sid)
    return data


def list_stores() -> List[Dict[str, Any]]:
    from database.connection import ControlSession
    from database.control_models import StoreModel

    with ControlSession() as c:
        return [_row_to_dict(s) for s in c.query(StoreModel).order_by(StoreModel.store_id).all()]


def require_known_store(store_id: str) -> str:
    """In production every store must be registered first; in development it is registered on first use."""
    sid = normalise_store_id(store_id)
    row = get_store(sid)
    if row is None:
        if is_production() and sid != default_store_id():
            raise UnknownStore(f"Store '{sid}' is not registered.")
        register_store(sid)
        row = get_store(sid, use_cache=False)
    if row and row.get("status") == "disabled":
        raise UnknownStore(f"Store '{sid}' is disabled.")
    return sid


def store_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """A setting for the current store: its registry override, else the environment, else ``default``."""
    from refillcare.env import ensure_dotenv

    ensure_dotenv()
    if name in OVERRIDABLE_SETTINGS:
        try:
            row = get_store()
        except Exception:  # registry unavailable (e.g. during start-up): fall back to the environment
            row = None
        if row and name in (row.get("settings") or {}):
            return str(row["settings"][name])
    return os.getenv(name, default if default is not None else "")


def clear_cache() -> None:
    with _lock:
        _cache.clear()
