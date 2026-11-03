"""Database engines and sessions, one isolated database namespace per store.

- **PostgreSQL** (``DATABASE_URL=postgresql+psycopg://...``, e.g. one AWS RDS instance): every store
  gets its own schema (``store_<id>``); the store tables are created there on first use. The
  control-plane tables (store registry, job queue) live in the default schema.
- **SQLite** (local / QA, no ``DATABASE_URL``): the default store keeps ``enterprise.db``; any other
  store gets ``enterprise_<id>.db`` next to it. Control-plane tables live in ``enterprise.db``.

The store comes from :func:`refillcare.tenancy.current_store_id` (request header, job, Streamlit
sidebar) unless passed explicitly. Schema changes are applied by versioned migrations
(``database/migrations.py``), once per store and process.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "refillcare" / "processed" / "enterprise.db"
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
IS_SQLITE = DATABASE_URL.startswith("sqlite")
# SQLite: the default store's file; other stores' files sit next to it
SQLITE_PATH = Path(DATABASE_URL.split("sqlite:///", 1)[1]) if IS_SQLITE and "sqlite:///" in DATABASE_URL else DEFAULT_SQLITE_PATH


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        eng = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30}, echo=False)

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - connection setup
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")      # readers never block the single writer
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        return eng
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
        pool_recycle=1800,
        echo=False,
    )


engine = _make_engine(DATABASE_URL)  # control plane (and the default store on SQLite)


class Base(DeclarativeBase):
    """Store tables (one copy per store)."""


class ControlBase(DeclarativeBase):
    """Control-plane tables shared by all stores (store registry, job queue)."""


_store_engines: Dict[str, Engine] = {}
_ready: set = set()
_lock = threading.Lock()


def _sqlite_path_for(store_id: str) -> Path:
    from refillcare.tenancy import default_store_id

    if store_id == default_store_id():
        return SQLITE_PATH
    return SQLITE_PATH.with_name(f"{SQLITE_PATH.stem}_{store_id.replace('-', '_')}.db")


def engine_for_store(store_id: str) -> Engine:
    """Engine whose unqualified tables resolve to the store's own namespace."""
    from refillcare.tenancy import normalise_store_id, schema_for_store

    sid = normalise_store_id(store_id)
    with _lock:
        eng = _store_engines.get(sid)
        if eng is None:
            if IS_SQLITE:
                path = _sqlite_path_for(sid)
                eng = engine if path == SQLITE_PATH else _make_engine(f"sqlite:///{path.as_posix()}")
            else:
                eng = engine.execution_options(schema_translate_map={None: schema_for_store(sid)})
            _store_engines[sid] = eng
    return eng


def init_control_plane() -> None:
    with _lock:
        if "__control__" in _ready:
            return
        import database.control_models  # noqa: F401  (register control tables)

        ControlBase.metadata.create_all(bind=engine)
        _ready.add("__control__")


def init_store(store_id: Optional[str] = None) -> str:
    """Create the store's schema / tables and apply pending migrations (idempotent, once per process)."""
    from refillcare.tenancy import current_store_id, normalise_store_id, schema_for_store

    sid = normalise_store_id(store_id) if store_id else current_store_id()
    init_control_plane()
    if sid in _ready:
        return sid
    eng = engine_for_store(sid)
    with _lock:
        if sid in _ready:
            return sid
        import database.models  # noqa: F401  (register store tables)
        from database.migrations import run_store_migrations

        if not IS_SQLITE:
            with engine.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_for_store(sid)}"'))
        Base.metadata.create_all(bind=eng)
        run_store_migrations(eng, Base.metadata, sid)
        _ready.add(sid)
    return sid


def init_db() -> None:
    """Backwards compatible: prepare the current store."""
    init_store()


def SessionLocal(store_id: Optional[str] = None, *args, **kwargs) -> Session:  # noqa: N802  (kept for backwards compatibility)
    """Open a session on a store's database (default: the current store)."""
    sid = init_store(store_id)
    return sessionmaker(bind=engine_for_store(sid), autocommit=False, autoflush=False)()


def ControlSession() -> Session:  # noqa: N802
    init_control_plane()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: a session on the request's store (set by the store middleware)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_engines_for_tests() -> None:
    """Forget cached store engines / readiness (tests that switch DATABASE_URL or stores)."""
    with _lock:
        for sid, eng in list(_store_engines.items()):
            if eng is not engine and IS_SQLITE:
                eng.dispose()
        _store_engines.clear()
        _ready.clear()
