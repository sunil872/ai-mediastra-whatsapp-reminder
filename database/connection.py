"""Enterprise Database Engine and Session Management.

Supports PostgreSQL via DATABASE_URL or fallback to local SQLite for offline/development mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

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

from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session injection with auto-close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables and ensure schema migrations if they do not exist."""
    Base.metadata.create_all(bind=engine)

    # SQLite lightweight migration guard for added columns
    if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                cursor = conn.execute(text("PRAGMA table_info(sales_transactions)"))
                existing_cols = [row[1] for row in cursor.fetchall()]
                if "transaction_type" not in existing_cols and len(existing_cols) > 0:
                    conn.execute(text("ALTER TABLE sales_transactions ADD COLUMN transaction_type VARCHAR(32) DEFAULT 'CUSTOMER_SALE'"))
                if "refillcare_eligible" not in existing_cols and len(existing_cols) > 0:
                    conn.execute(text("ALTER TABLE sales_transactions ADD COLUMN refillcare_eligible BOOLEAN DEFAULT 1"))
                if "exclusion_reason" not in existing_cols and len(existing_cols) > 0:
                    conn.execute(text("ALTER TABLE sales_transactions ADD COLUMN exclusion_reason VARCHAR(64)"))
                conn.commit()
        except Exception:
            pass
