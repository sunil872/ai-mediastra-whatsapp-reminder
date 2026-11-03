"""Onboard a store: load an existing sales history (master CSV / parquet / Excel) into its database.

    python scripts/import_history_to_db.py --store ngos-colony --file data/refillcare/customer_data_fields.csv
    python scripts/import_history_to_db.py --store ngos-colony --file history.parquet --tenant t1 --name "NGOs Colony"

Any column layout works (same mapping as uploads; only the 11 RefillCare fields are stored). Lines
already in the store database are skipped, so it is safe to re-run. Afterwards new sales are uploaded
incrementally through the API / Streamlit; the worker's first scheduled run trains the store's model.
Set DATABASE_URL to the RDS instance to load straight into AWS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402

from database.connection import SessionLocal  # noqa: E402
from refillcare.data.incremental_ingest import ingest_sales_batch  # noqa: E402
from refillcare.data.schema import read_sales_file  # noqa: E402
from refillcare.stores import register_store  # noqa: E402
from refillcare.tenancy import use_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="store id (becomes the database schema store_<id>)")
    ap.add_argument("--file", required=True, help="CSV / Excel / parquet sales history")
    ap.add_argument("--tenant")
    ap.add_argument("--name")
    ap.add_argument("--chunk-rows", type=int, default=200_000, help="rows per batch (bounded memory)")
    args = ap.parse_args()

    register_store(args.store, tenant_id=args.tenant, store_name=args.name)
    path = Path(args.file)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else read_sales_file(path, path.name)
    print(f"Read {len(df):,} rows from {path.name}")
    totals = {"added": 0, "skipped": 0, "review": 0}
    t0 = time.time()
    with use_store(args.store), SessionLocal() as db:
        from database.models import ImportBatchModel
        from refillcare.data import history_store as hs
        from refillcare.data.incremental_ingest import rescore_customers

        batch_ids = []
        for i in range(0, len(df), args.chunk_rows):
            part = df.iloc[i:i + args.chunk_rows]
            res = ingest_sales_batch(db, part, f"{path.name}[{i}:{i + len(part)}]", actor="onboarding", sync=False)
            batch_ids.append(res["import_batch_id"])
            m = res["metrics"]
            totals["added"] += m["new_records_added"]
            totals["skipped"] += m["existing_duplicates_skipped"]
            totals["review"] += m["records_requiring_review"]
            print(f"  rows {i:,}-{i + len(part):,}: +{m['new_records_added']:,} new, {m['existing_duplicates_skipped']:,} already stored")
        # Score every onboarded patient once (not per chunk): all medicines, current model and rules
        customers = hs.load_all(db)["customerId"].dropna().astype(str).unique().tolist()
        res = rescore_customers(db, customers)
        for bid in batch_ids:
            b = db.get(ImportBatchModel, bid)
            if b is not None:
                b.sync_status = "SYNCED"
        db.commit()
        print("  re-scored:", json.dumps({k: res.get(k) for k in ("customer_items_rescored", "cycles_created")}))
    print(json.dumps(totals | {"seconds": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
