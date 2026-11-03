"""Which store a request / job / Streamlit session works on (one deployment serves every store).

The Mediastra backend sends ``X-Store-Id`` (and ``X-Tenant-Id``) on every API call; the worker sets
the store of each job; the Streamlit app sets the store picked in its sidebar. Everything below the
entry point (database session, data folder, model files, settings) follows :func:`current_store_id`.

Store ids are validated strictly because they become PostgreSQL schema names.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_STORE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_current_store: ContextVar[Optional[str]] = ContextVar("refillcare_store_id", default=None)
_current_tenant: ContextVar[Optional[str]] = ContextVar("refillcare_tenant_id", default=None)


class InvalidStoreId(ValueError):
    pass


def default_store_id() -> str:
    from refillcare.env import ensure_dotenv

    ensure_dotenv()
    return normalise_store_id(os.getenv("REFILLCARE_STORE_ID", "") or "store-1")


def normalise_store_id(store_id: str) -> str:
    s = str(store_id or "").strip().lower()
    if not _STORE_ID_RE.match(s):
        raise InvalidStoreId("store id must be 1-40 chars: lowercase letters, digits, '-' or '_' (starting with a letter or digit)")
    return s


def schema_for_store(store_id: str) -> str:
    """PostgreSQL schema of a store ('ngos-colony' -> 'store_ngos_colony')."""
    return "store_" + normalise_store_id(store_id).replace("-", "_")


def current_store_id() -> str:
    return _current_store.get() or default_store_id()


def current_tenant_id() -> Optional[str]:
    return _current_tenant.get()


def set_current_store(store_id: Optional[str], tenant_id: Optional[str] = None) -> str:
    """Set the store for the current context (request thread, Streamlit run, job)."""
    sid = normalise_store_id(store_id) if store_id else default_store_id()
    _current_store.set(sid)
    _current_tenant.set(tenant_id)
    return sid


@contextmanager
def use_store(store_id: Optional[str], tenant_id: Optional[str] = None) -> Iterator[str]:
    sid = normalise_store_id(store_id) if store_id else default_store_id()
    t1, t2 = _current_store.set(sid), _current_tenant.set(tenant_id)
    try:
        yield sid
    finally:
        _current_store.reset(t1)
        _current_tenant.reset(t2)


def is_production() -> bool:
    from refillcare.env import ensure_dotenv

    ensure_dotenv()
    return os.getenv("REFILLCARE_ENV", "development").strip().lower() in ("production", "prod")
