"""Schema-per-store on a real PostgreSQL (CI service / local docker). Skipped without REFILLCARE_PG_TEST_URL.

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw postgres:16-alpine
    REFILLCARE_PG_TEST_URL=postgresql+psycopg://postgres:pw@localhost:5432/postgres pytest tests/test_postgres_tenancy.py

Runs in a subprocess because ``database.connection`` binds its engine to ``DATABASE_URL`` at import.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PG_URL = os.getenv("REFILLCARE_PG_TEST_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="REFILLCARE_PG_TEST_URL not set (needs a PostgreSQL server)")

SCRIPT = textwrap.dedent('''
    import uuid
    from datetime import date, timedelta
    import pandas as pd
    from sqlalchemy import text
    from database.connection import SessionLocal, engine
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import ingest_sales_batch, rollback_batch
    from refillcare.stores import register_store
    from refillcare.tenancy import schema_for_store, use_store

    a, b = f"pg-{uuid.uuid4().hex[:6]}", f"pg-{uuid.uuid4().hex[:6]}"
    register_store(a, tenant_id="t1"); register_store(b, tenant_id="t1")
    rows = [{"Bill No.": f"S0/{k}", "Bill Date": (date(2026, 1, 5) + timedelta(days=30 * k)).strftime("%d/%m/%Y"),
             "Party Name": "RAVI", "Mobile": "9848012345", "Product": "TELMA 40 TAB", "Prod Code": "4060",
             "Pack": "1X15", "Qty": "2"} for k in range(4)]
    with use_store(a), SessionLocal() as db:
        r = ingest_sales_batch(db, pd.DataFrame(rows), "a.csv")
        assert r["metrics"]["new_records_added"] == 4, r
        assert r["online_sync"].get("customer_items_rescored", 0) >= 1
    with use_store(b), SessionLocal() as db:
        assert hs.history_count(db) == 0, "store B sees store A rows"
    with engine.connect() as c:
        n = c.execute(text(f'SELECT count(*) FROM "{schema_for_store(a)}".sales_transactions')).scalar()
        v = c.execute(text(f'SELECT max(version) FROM "{schema_for_store(a)}".refillcare_schema_version')).scalar()
        assert n == 4 and v >= 1
    with use_store(a), SessionLocal() as db:
        assert rollback_batch(db, r["import_batch_id"])["records_removed"] == 4
        assert hs.history_count(db) == 0
    with engine.begin() as c:
        for s in (a, b):
            c.execute(text(f'DROP SCHEMA IF EXISTS "{schema_for_store(s)}" CASCADE'))
    print("PG-TENANCY-OK")
''')


def test_schema_per_store_isolation_on_postgres(tmp_path):
    env = {**os.environ, "DATABASE_URL": PG_URL, "REFILLCARE_ENV": "test", "REFILLCARE_DATA_DIR": str(tmp_path),
           "REFILLCARE_STALE_DATA_GUARD": "false", "OPENBLAS_NUM_THREADS": "1"}
    root = Path(__file__).resolve().parent.parent
    res = subprocess.run([sys.executable, "-c", SCRIPT], cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    assert "PG-TENANCY-OK" in res.stdout, res.stdout[-2000:] + res.stderr[-4000:]
