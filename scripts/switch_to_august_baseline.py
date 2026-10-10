import shutil
import sys
import time
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from database.connection import SessionLocal
from refillcare.data.pipeline import run_refillcare_data_pipeline
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.engine.persistence import RefillPersistenceManager

def main():
    print("=" * 70)
    print("ROLLBACK: SWITCHING TO AUGUST 2026 BASELINE (PREDICTING SEPTEMBER)")
    print("=" * 70)

    backup_csv = Path("data/refillcare/backups/customer_data_fields_backup_pre_sep.csv")
    target_csv = Path("data/refillcare/customer_data_fields.csv")

    if not backup_csv.exists():
        print(f"[ERROR] Backup CSV not found at {backup_csv}")
        return

    # 1. Restore customer_data_fields.csv
    print(f"[1/3] Restoring {target_csv} from pre-September backup...")
    shutil.copy(backup_csv, target_csv)
    with open(target_csv, "r", encoding="utf-8", errors="ignore") as f:
        row_count = sum(1 for _ in f) - 1
    print(f"      Restored {target_csv} to {row_count:,} rows (up to 31-08-2026).")

    # 2. Run pipeline
    print(f"\n[2/3] Re-running baseline pipeline on August data...")
    t0 = time.time()
    res = run_refillcare_data_pipeline(
        transactions_path="data/refillcare/customer_data_fields.csv",
        salt_master_path="data/refillcare/SALT WISE ITEMS.xlsx",
        output_dir="data/refillcare/processed",
        save_artifacts=True
    )
    t1 = time.time()
    purchase_history_df = res.get("purchase_history_df")
    if purchase_history_df is None or purchase_history_df.empty:
        purchase_history_df = pd.read_parquet("data/refillcare/processed/purchase_history.parquet")
    print(f"      Pipeline finished in {t1-t0:.2f}s (Max Date: {purchase_history_df['invoice_date'].max()}).")

    # 3. Predict September baseline as of 2026-09-01
    print(f"\n[3/3] Evaluating Unified Decisions as of 2026-09-01...")
    engine = UnifiedRefillDecisionEngine()
    as_of_date = date(2026, 9, 1)
    decisions_df, decisions_list = engine.evaluate_all_customer_items(purchase_history_df, as_of_date=as_of_date)
    
    db = SessionLocal()
    try:
        pm = RefillPersistenceManager()
        pm.save_refill_decisions(db, decisions_list)
        pm.sync_reminder_cycles(db, decisions_list)
        sep_queue = pm.get_monthly_review_queue(db, target_month="2026-09")
        print(f"      Total September 2026 Scheduled Reminders: {len(sep_queue):,}")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print("SUCCESS: System Rolled Back to Clean August Baseline!")
    print("=" * 70)

if __name__ == "__main__":
    main()
