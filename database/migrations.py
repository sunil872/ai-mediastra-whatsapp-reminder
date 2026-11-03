"""Schema migrations, applied per store (PostgreSQL schema or SQLite file).

Two layers, both idempotent and run once per store and process (``database.connection.init_store``):

1. **Additive sync** - ``create_all`` creates missing tables; columns that exist on a model but not
   in the database are added (nullable; a NOT NULL column without a default is added as nullable
   and logged, because it cannot be added to a populated table).
2. **Versioned steps** - ordered, recorded in ``refillcare_schema_version``: indexes, back-fills and
   anything else that must run exactly once per store.

Run them as a deploy step with ``python -m database.migrations --all-stores`` so the first request
after a deploy is not the one migrating.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from sqlalchemy import MetaData, inspect, text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger("refillcare.db.migrations")


def _default_sql(col) -> str:
    default = getattr(col, "default", None)
    arg = getattr(default, "arg", None) if default is not None else None
    if arg is None or callable(arg):
        return ""
    if isinstance(arg, bool):
        return f" DEFAULT {'TRUE' if arg else 'FALSE'}"
    if isinstance(arg, (int, float)):
        return f" DEFAULT {arg}"
    return " DEFAULT '" + str(arg).replace("'", "''") + "'"


# ML tables created by the short-lived multi-tenant prototype (tenant_id NOT NULL). They only hold
# re-computable data, so they are dropped and recreated in the current shape.
_PROTOTYPE_ML_TABLES = ("ml_model_versions", "ml_evaluation_runs", "refill_decision_outcomes")


def _schema_of(engine: Engine) -> Optional[str]:
    tmap = (engine.get_execution_options() or {}).get("schema_translate_map") or {}
    return tmap.get(None)


def _qual(schema: Optional[str], table: str) -> str:
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def drop_prototype_tenant_ml_tables(engine: Engine) -> List[str]:
    schema = _schema_of(engine)
    insp = inspect(engine)
    dropped: List[str] = []
    existing = set(insp.get_table_names(schema=schema))
    with engine.begin() as conn:
        for t in _PROTOTYPE_ML_TABLES:
            if t in existing and "tenant_id" in {c["name"] for c in insp.get_columns(t, schema=schema)}:
                conn.execute(text(f"DROP TABLE {_qual(schema, t)}"))
                dropped.append(t)
                logger.warning("Dropped prototype multi-tenant table %s (re-created in the current shape).", t)
    return dropped


def apply_additive_migrations(engine: Engine, metadata: MetaData) -> List[str]:
    """Add missing columns to existing tables. Returns the applied statements."""
    schema = _schema_of(engine)
    applied: List[str] = []
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names(schema=schema))
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_cols = {c["name"] for c in insp.get_columns(table.name, schema=schema)}
            for col in table.columns:
                if col.name in existing_cols:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                default_sql = _default_sql(col)
                if not col.nullable and not default_sql:
                    logger.warning("Column %s.%s is NOT NULL without a default: added as nullable.", table.name, col.name)
                stmt = f'ALTER TABLE {_qual(schema, table.name)} ADD COLUMN "{col.name}" {col_type}{default_sql}'
                conn.execute(text(stmt))
                applied.append(stmt)
                logger.info("Applied migration: %s", stmt)
    return applied


# ----------------------------------------------------------------------------- versioned steps
def _v1_sales_line_key_index(conn: Connection, schema: Optional[str]) -> None:
    """One row per sales line: the incremental append skips lines already stored."""
    conn.execute(text(f'CREATE UNIQUE INDEX IF NOT EXISTS uq_sales_line_key ON {_qual(schema, "sales_transactions")} (line_key)'))


def _v2_batch_status_backfill(conn: Connection, schema: Optional[str]) -> None:
    conn.execute(text(f"UPDATE {_qual(schema, 'import_batches')} SET sync_status = 'SYNCED' WHERE sync_status IS NULL"))


def _v3_drop_redundant_sales_indexes(conn: Connection, schema: Optional[str]) -> None:
    """Drop redundant/low-cardinality indexes on sales_transactions to slim database storage."""
    indexes = [
        "ix_sales_transactions_customer_id",
        "ix_sales_transactions_invoice_date",
        "ix_sales_transactions_refillcare_eligible",
        "ix_sales_transactions_transaction_type",
    ]
    for idx in indexes:
        try:
            if schema:
                conn.execute(text(f'DROP INDEX IF EXISTS "{schema}"."{idx}"'))
            else:
                conn.execute(text(f'DROP INDEX IF EXISTS "{idx}"'))
        except Exception as e:
            logger.warning("Could not drop index %s: %s", idx, e)


MIGRATIONS: List[Tuple[int, str, Callable[[Connection, Optional[str]], None]]] = [
    (1, "unique sales line key", _v1_sales_line_key_index),
    (2, "batch sync status back-fill", _v2_batch_status_backfill),
    (3, "drop redundant sales indexes for storage slimming", _v3_drop_redundant_sales_indexes),
]


def run_versioned_migrations(engine: Engine) -> List[int]:
    schema = _schema_of(engine)
    applied: List[int] = []
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {_qual(schema, 'refillcare_schema_version')} "
                          "(version INTEGER PRIMARY KEY, description VARCHAR(255), applied_at VARCHAR(32))"))
        done = {r[0] for r in conn.execute(text(f"SELECT version FROM {_qual(schema, 'refillcare_schema_version')}"))}
        for version, desc, fn in MIGRATIONS:
            if version in done:
                continue
            fn(conn, schema)
            conn.execute(text(f"INSERT INTO {_qual(schema, 'refillcare_schema_version')} (version, description, applied_at) "
                              "VALUES (:v, :d, :t)"), {"v": version, "d": desc, "t": datetime.utcnow().isoformat()})
            applied.append(version)
            logger.info("Applied versioned migration %s (%s) on %s", version, desc, schema or "default")
    return applied


def run_store_migrations(engine: Engine, metadata: MetaData, store_id: str) -> dict:
    dropped = drop_prototype_tenant_ml_tables(engine)
    if dropped:
        metadata.create_all(bind=engine)
    return {"store_id": store_id, "dropped": dropped, "columns_added": apply_additive_migrations(engine, metadata),
            "versions_applied": run_versioned_migrations(engine)}


def main() -> int:  # pragma: no cover - deploy step
    import argparse

    from database.connection import ControlSession, init_store
    from database.control_models import StoreModel

    ap = argparse.ArgumentParser(description="Apply schema migrations (deploy step).")
    ap.add_argument("--store", help="one store id (default: the default store)")
    ap.add_argument("--all-stores", action="store_true", help="every registered store")
    args = ap.parse_args()
    if args.all_stores:
        with ControlSession() as c:
            stores = [s.store_id for s in c.query(StoreModel).all()]
    else:
        stores = [args.store] if args.store else [None]
    for s in stores or [None]:
        print("migrated", init_store(s))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
