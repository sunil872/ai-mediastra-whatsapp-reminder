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
    print("SWITCH: INGESTING SEPTEMBER DATA (PREDICTING OCTOBER 2026)")
    print("=" * 70)

    target_csv = Path("data/refillcare/customer_data_fields.csv")
    sep_csv = Path(r"C:\Users\sunil\ai-mediastra-whatsapp-reminder\sample folder\sep data.xlsx - Sale Book With Item Details.csv")

    # 1. Check if September data already present
    with open(target_csv, "r", encoding="utf-8", errors="ignore") as f:
        curr_count = sum(1 for _ in f) - 1

    if curr_count < 900000 and sep_csv.exists():
        print(f"[1/3] Appending September data to {target_csv}...")
        backup_path = Path("data/refillcare/backups/customer_data_fields_backup_pre_sep.csv")
        if not backup_path.exists():
            shutil.copy(target_csv, backup_path)
            print(f"      Backup created at {backup_path}.")

        df_sep = pd.read_csv(sep_csv, dtype=str, low_memory=False)
        df_sep = df_sep[df_sep["DATE"].notna() & ~df_sep["DATE"].astype(str).str.contains("Total", case=False, na=False)].copy()
        df_sep = df_sep[df_sep["PARTY NAME"].notna() & (df_sep["PARTY NAME"].astype(str).str.strip() != "")].copy()

        def format_date(d_str):
            s = str(d_str).strip()
            parts = s.replace("/", "-").split("-")
            if len(parts) == 3:
                day, month, year = parts[0], parts[1], parts[2]
                if len(year) == 4:
                    year = year[2:]
                return f"{day.zfill(2)}-{month.zfill(2)}-{year}"
            return s

        def clean_code(val):
            s = str(val).strip()
            return s[:-2] if s.endswith(".0") else s

        df_mapped = pd.DataFrame()
        df_mapped["invoice_date"] = df_sep["DATE"].apply(format_date)
        df_mapped["invoice_number"] = df_sep["TRN NO."].fillna("").astype(str).str.strip()
        df_mapped["customerName"] = df_sep["PARTY NAME"].fillna("").astype(str).str.strip()
        df_mapped["itemCode"] = df_sep["ITEM CODE"].apply(clean_code)
        df_mapped["itemName"] = df_sep["ITEM NAME"].fillna("").astype(str).str.strip()
        df_mapped["packing"] = df_sep["PACK"].fillna("").astype(str).str.strip()
        df_mapped["batchNo"] = df_sep["BATCH"].fillna("").astype(str).str.strip()
        df_mapped["mfgDate"] = df_sep.get("MFG.DT.", pd.Series([""] * len(df_sep))).fillna("").astype(str).str.strip()
        df_mapped["expiryDate"] = df_sep["EXPIRY"].fillna("").astype(str).str.strip()
        df_mapped["quantity"] = pd.to_numeric(df_sep["QTY."], errors="coerce").fillna(0)
        df_mapped["freeQuantity"] = pd.to_numeric(df_sep["FREE"], errors="coerce").fillna(0)
        df_mapped["rate"] = pd.to_numeric(df_sep["BILLING RATE"], errors="coerce").fillna(0.0)
        df_mapped["saleRate"] = pd.to_numeric(df_sep["SALE RATE"], errors="coerce").fillna(0.0)
        df_mapped["mrp"] = pd.to_numeric(df_sep["MRP"], errors="coerce").fillna(0.0)
        df_mapped["invoice_total"] = pd.to_numeric(df_sep["INV AMT"], errors="coerce").fillna(0.0)
        df_mapped["discountPercent"] = pd.to_numeric(df_sep["DIS.%"], errors="coerce").fillna(0.0)
        df_mapped["item_discount"] = pd.to_numeric(df_sep["DIS.AMT."], errors="coerce").fillna(0.0)
        df_mapped["gstAmount"] = pd.to_numeric(df_sep["GST AMT."], errors="coerce").fillna(0.0)
        df_mapped["netAmount"] = pd.to_numeric(df_sep["NET AMT"], errors="coerce").fillna(0.0)
        df_mapped["costPrice"] = pd.to_numeric(df_sep["COST"], errors="coerce").fillna(0.0)
        df_mapped["MOBILE_NO"] = df_sep["MOBILE NO."].apply(clean_code)
        df_mapped["customerId"] = df_mapped["customerName"]
        df_mapped["itemId"] = df_mapped["itemCode"]

        df_existing_head = pd.read_csv(target_csv, nrows=1)
        df_mapped = df_mapped[list(df_existing_head.columns)]
        df_mapped.to_csv(target_csv, mode="a", header=False, index=False)
        print(f"      Appended {len(df_mapped):,} records to {target_csv}.")
    else:
        print(f"[1/3] September data already present in {target_csv} ({curr_count:,} rows).")

    # 2. Run pipeline
    print(f"\n[2/3] Re-running pipeline with September data...")
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

    # 3. Predict October as of 2026-10-01
    print(f"\n[3/3] Evaluating Unified Decisions as of 2026-10-01...")
    engine = UnifiedRefillDecisionEngine()
    as_of_date = date(2026, 10, 1)
    decisions_df, decisions_list = engine.evaluate_all_customer_items(purchase_history_df, as_of_date=as_of_date)
    
    db = SessionLocal()
    try:
        pm = RefillPersistenceManager()
        pm.save_refill_decisions(db, decisions_list)
        pm.sync_reminder_cycles(db, decisions_list)
        oct_queue = pm.get_monthly_review_queue(db, target_month="2026-10")
        print(f"      Total October 2026 Scheduled Reminders: {len(oct_queue):,}")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print("SUCCESS: System Updated with September Data for October 2026 Predictions!")
    print("=" * 70)

if __name__ == "__main__":
    main()
