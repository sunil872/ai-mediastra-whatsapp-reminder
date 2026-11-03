"""Reproducible benchmark of the incremental (append-only) sales upload.

Uses a TEMPORARY database (never the live one). From a real history it holds back the last day and
one earlier day, onboards the rest once, then measures:

1. a 1-day upload (store-style headers + extra columns that are dropped),
2. the same day again (must add nothing),
3. the back-dated day arriving late,
4. rolling the 1-day batch back, then uploading it again.

    python scripts/benchmark_incremental_upload.py                                    # default history parquet
    python scripts/benchmark_incremental_upload.py --history data/refillcare/customer_data_fields.csv --out bench.json

Reports rows, lines added / skipped, medicines re-scored, seconds, peak Python memory and DB size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="refillcare_bench_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'bench.db').as_posix()}"  # before any database import
os.environ.setdefault("REFILLCARE_STALE_DATA_GUARD", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def _store_file(rows: pd.DataFrame) -> pd.DataFrame:
    """The day as a store would export it: other header names, plus columns RefillCare drops."""
    code = rows["itemCode"] if "itemCode" in rows.columns else rows["itemId"]
    return pd.DataFrame({
        "Bill No.": rows["invoice_number"].values, "Bill Date": rows["invoice_date"].dt.strftime("%d/%m/%Y").values,
        "Party Name": rows["customerName"].values, "Mobile": rows["MOBILE_NO"].values, "Product": rows["itemName"].values,
        "Prod Code": code.values, "Pack": rows["packing"].values, "Qty": rows["quantity"].values, "MRP": "100", "GST%": "12",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(ROOT / "data/refillcare/processed/purchase_history.parquet"))
    ap.add_argument("--backdated-offset", type=int, default=7, help="the late day = N sales days before the last")
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--measure-memory", action="store_true", help="also report peak Python memory (timings become slower)")
    args = ap.parse_args()

    from database.connection import SessionLocal
    from refillcare.data import history_store as hs
    from refillcare.data.incremental_ingest import get_population_stats, ingest_sales_batch
    from refillcare.data.schema import read_sales_file

    path = Path(args.history)
    full = pd.read_parquet(path) if path.suffix == ".parquet" else read_sales_file(path, path.name)
    full["invoice_date"] = pd.to_datetime(full["invoice_date"], errors="coerce", dayfirst=True)
    full = full[full["invoice_date"].notna()]
    if "refillcare_eligible" in full.columns:
        full = full[full["refillcare_eligible"].fillna(True).astype(bool)]
    if "itemCode" not in full.columns:
        full["itemCode"] = full["itemId"]
    days = sorted(full["invoice_date"].dt.normalize().unique())
    last, late = days[-1], days[-1 - args.backdated_offset]
    norm = full["invoice_date"].dt.normalize()
    onboard = full[(norm != last) & (norm != late)]
    result: dict = {"history_file": path.name, "history_rows": int(len(full)), "temp_db": str(TMP / "bench.db")}

    with SessionLocal() as db:
        t = time.time()
        hs.import_frame(db, onboard)
        result["onboarding"] = {"lines": hs.history_count(db), "seconds": round(time.time() - t, 1)}
        t = time.time()
        get_population_stats(db)
        result["salt_priors_once_seconds"] = round(time.time() - t, 1)

        def run(label: str, df: pd.DataFrame) -> None:
            # tracemalloc slows Python code several-fold: timings are clean unless --measure-memory
            if args.measure_memory:
                tracemalloc.start()
            t0 = time.time()
            r = ingest_sales_batch(db, df, f"{label}.csv")
            peak = tracemalloc.get_traced_memory()[1] / 1e6 if args.measure_memory else None
            if args.measure_memory:
                tracemalloc.stop()
            m, s = r["metrics"], r.get("online_sync") or {}
            result[label] = {
                "rows_in_file": int(len(df)), "lines_added": m["new_records_added"], "lines_already_stored": m["existing_duplicates_skipped"],
                "rows_to_review": m["records_requiring_review"], "medicines_rescored": s.get("customer_items_rescored"),
                "reminder_cycles_created": s.get("cycles_created"), "warnings": sorted(r.get("data_warnings") or {}),
                "ignored_columns": m.get("ignored_columns"), "seconds": round(time.time() - t0, 2), "peak_python_mb": round(peak) if peak is not None else None,
            }
            print(label, json.dumps(result[label]), flush=True)

        run("one_day_upload", _store_file(full[norm == last]))
        run("same_day_again", _store_file(full[norm == last]))
        run("backdated_day", _store_file(full[norm == late]))

        from database.models import ImportBatchModel
        from refillcare.data.incremental_ingest import rollback_batch

        batch = (db.query(ImportBatchModel).filter(ImportBatchModel.source_filename == "one_day_upload.csv")
                 .order_by(ImportBatchModel.upload_timestamp).first())
        t0 = time.time()
        rb = rollback_batch(db, batch.import_batch_id)
        result["rollback_one_day"] = {"lines_removed": rb["records_removed"], "medicines_rescored": (rb.get("rescored") or {}).get("customer_items_rescored"),
                                      "seconds": round(time.time() - t0, 2)}
        print("rollback_one_day", json.dumps(result["rollback_one_day"]), flush=True)
        run("upload_after_rollback", _store_file(full[norm == last]))
    result["db_size_mb"] = round((TMP / "bench.db").stat().st_size / 1e6)
    print(json.dumps({k: v for k, v in result.items() if k not in ("one_day_upload", "same_day_again", "backdated_day", "rollback_one_day", "upload_after_rollback")}, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
