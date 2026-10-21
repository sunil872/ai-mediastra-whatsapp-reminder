"""Lightweight additive schema migrations (SQLite + PostgreSQL).

``create_all`` creates missing tables but never alters existing ones. This adds any column that
exists on a model but not in the database, so the existing database (e.g. the current
``enterprise.db``) picks up new nullable / defaulted columns without data loss.

Destructive changes (renames, type changes, drops) are out of scope - use Alembic for those once
the platform moves to managed PostgreSQL.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("refillcare.db.migrations")


def _default_sql(col) -> str:
    default = getattr(col, "default", None)
    arg = getattr(default, "arg", None) if default is not None else None
    if arg is None or callable(arg):
        return ""
    if isinstance(arg, bool):
        return f" DEFAULT {1 if arg else 0}"
    if isinstance(arg, (int, float)):
        return f" DEFAULT {arg}"
    return " DEFAULT '" + str(arg).replace("'", "''") + "'"


# ML tables created by the short-lived multi-tenant prototype (tenant_id NOT NULL). They only hold
# re-computable data (registry bootstrap, evaluation reports, outcomes), so they are dropped and
# recreated in the v1 single-store shape.
_PROTOTYPE_ML_TABLES = ("ml_model_versions", "ml_evaluation_runs", "refill_decision_outcomes")


def drop_prototype_tenant_ml_tables(engine: Engine) -> List[str]:
    insp = inspect(engine)
    dropped: List[str] = []
    existing = set(insp.get_table_names())
    with engine.begin() as conn:
        for t in _PROTOTYPE_ML_TABLES:
            if t in existing and "tenant_id" in {c["name"] for c in insp.get_columns(t)}:
                conn.execute(text(f'DROP TABLE "{t}"'))
                dropped.append(t)
                logger.warning("Dropped prototype multi-tenant table %s (re-created in v1 shape).", t)
    return dropped


def apply_additive_migrations(engine: Engine, metadata: MetaData) -> List[str]:
    """Add missing columns to existing tables. Returns the list of applied statements."""
    applied: List[str] = []
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_cols = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing_cols:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                default_sql = _default_sql(col)
                # New columns must be nullable or defaulted to be addable to populated tables.
                null_sql = "" if (col.nullable or default_sql) else ""
                stmt = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}{default_sql}{null_sql}'
                conn.execute(text(stmt))
                applied.append(stmt)
                logger.info("Applied migration: %s", stmt)
    return applied
