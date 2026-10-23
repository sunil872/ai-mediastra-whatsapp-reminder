"""One-time: keep only the RefillCare columns in the stored sales data (raw file kept as a backup).

1. Copies the full master CSV to ``<data dir>/raw_backup/customer_data_fields_raw_<date>.csv``
   (once; an existing raw backup is never overwritten) - the untouched source for re-training.
2. Rewrites ``customer_data_fields.csv`` with only the 11 RefillCare columns
   (invoice_date, invoice_number, customerName, itemCode, itemName, packing, quantity,
   freeQuantity, MOBILE_NO, customerId, itemId). Values are copied as text, unchanged.
3. Drops the unused price / tax / batch columns from the processed history parquets.

    python scripts/slim_sales_data.py            # dry run: what would change and the size saved
    python scripts/slim_sales_data.py --apply

Row counts are verified before anything is replaced; files are swapped atomically.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from refillcare.data.schema import CANONICAL_COLUMNS, drop_legacy_columns  # noqa: E402
from refillcare.store import StoreConfig  # noqa: E402


def _mb(p: Path) -> float:
    return round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0


def slim_master(store: StoreConfig, apply: bool) -> dict:
    master = store.master_csv
    if not master.exists():
        return {"master": "not found"}
    header = list(pd.read_csv(master, nrows=0).columns)
    extra = [c for c in header if c not in CANONICAL_COLUMNS]
    out = {"master": str(master), "size_mb": _mb(master), "columns_now": len(header), "columns_dropped": extra}
    if not extra and header == list(CANONICAL_COLUMNS):
        out["status"] = "already slim"
        return out

    backups = sorted(store.raw_backup_dir.glob("customer_data_fields_raw_*.csv"))
    out["raw_backup"] = str(backups[0]) if backups else "will be created"
    if not apply:
        out["status"] = "dry run"
        return out

    if not backups:
        store.raw_backup_dir.mkdir(parents=True, exist_ok=True)
        raw = store.raw_backup_dir / f"customer_data_fields_raw_{datetime.now():%Y%m%d}.csv"
        shutil.copy2(master, raw)
        out["raw_backup"] = str(raw)

    df = pd.read_csv(master, dtype=str, keep_default_na=False)
    n = len(df)
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    tmp = master.with_suffix(".slim.tmp")
    df[list(CANONICAL_COLUMNS)].to_csv(tmp, index=False)
    check = pd.read_csv(tmp, dtype=str, keep_default_na=False, usecols=["invoice_date"])
    if len(check) != n:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Row count changed while slimming ({n} -> {len(check)}); master left untouched.")
    os.replace(tmp, master)
    out.update({"status": "slimmed", "rows": n, "size_mb_after": _mb(master)})
    return out


def slim_parquet(path: Path, apply: bool) -> dict:
    if not path.exists():
        return {"file": str(path), "status": "not found"}
    df = pd.read_parquet(path)
    slim = drop_legacy_columns(df)
    dropped = [c for c in df.columns if c not in slim.columns]
    out = {"file": str(path), "size_mb": _mb(path), "columns_dropped": dropped}
    if not dropped:
        out["status"] = "already slim"
        return out
    if not apply:
        out["status"] = "dry run"
        return out
    tmp = path.with_suffix(".slim.tmp")
    slim.to_parquet(tmp, index=False)
    if len(pd.read_parquet(tmp, columns=[slim.columns[0]])) != len(df):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Row count changed while slimming {path.name}; left untouched.")
    os.replace(tmp, path)
    out.update({"status": "slimmed", "rows": len(df), "size_mb_after": _mb(path)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()
    store = StoreConfig.from_env()
    report = {
        "master_csv": slim_master(store, args.apply),
        "purchase_history": slim_parquet(store.history_parquet, args.apply),
        "clean_transactions": slim_parquet(store.clean_transactions_parquet, args.apply),
    }
    for k, v in report.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
