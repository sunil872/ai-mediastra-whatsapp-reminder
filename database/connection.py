"""Enterprise Database Engine and Session Management.

Supports PostgreSQL via DATABASE_URL or fallback to local SQLite for offline/development mode.
v1 is single-store: one database. The schema is created / additively migrated on first use.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "refillcare" / "processed" / "enterprise.db"

# Ensure data directory exists
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Use SQLite fallback
    DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

# Setup engine with appropriate connection parameters
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


class Base(DeclarativeBase):
    pass


_SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
_schema_ready = False
_schema_lock = threading.Lock()


def init_db() -> None:
    """Create missing tables and add missing nullable columns (idempotent)."""
    global _schema_ready
    with _schema_lock:
        if _schema_ready:
            return
        import database.models  # noqa: F401  (register all models on Base.metadata)
        from database.migrations import apply_additive_migrations, drop_prototype_tenant_ml_tables

        drop_prototype_tenant_ml_tables(engine)
        Base.metadata.create_all(bind=engine)
        apply_additive_migrations(engine, Base.metadata)
        _schema_ready = True


def SessionLocal() -> Session:  # noqa: N802  (kept for backwards compatibility)
    """Open a session on the store database (schema ensured on first use)."""
    init_db()
    return _SessionFactory()


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session injection with auto-close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
