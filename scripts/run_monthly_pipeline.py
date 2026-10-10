"""Dynamic Production Monthly Ingestion & RefillCare Prediction Pipeline.

Dynamically processes the canonical sales transaction dataset
(data/refillcare/customer_data_fields.csv), detects the latest transaction date,
and evaluates UnifiedRefillDecisionEngine predictions for the upcoming target month.

Supports:
1. Dynamic baseline execution: Runs directly on current master data (e.g. Sept 30, 2026 data predicting Oct 2026).
2. Incremental monthly sales ingestion: Pass `--new-sales <path>` to validate, normalize, backup,
   and append a new month's sales dataset before generating predictions.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal
from refillcare.data.pipeline import run_refillcare_data_pipeline
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.engine.persistence import RefillPersistenceManager
from refillcare.data.monthly_ingestion import normalize_sales_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("refillcare.monthly_pipeline")


def clean_code(val: any) -> str:
    """Normalize item or mobile numbers, stripping float artifacts."""
    s = str(val).strip()
    return s[:-2] if s.endswith(".0") else s


def format_incoming_date(d_str: any) -> str:
    """Format various input date strings into canonical DD-MM-YY or DD-MM-YYYY."""
    s = str(d_str).strip()
    parts = s.replace("/", "-").split("-")
    if len(parts) == 3:
        day, month, year = parts[0], parts[1], parts[2]
        if len(year) == 4:
            year = year[2:]
        return f"{day.zfill(2)}-{month.zfill(2)}-{year}"
    return s


def append_new_sales_data(new_sales_path: Path, master_csv_path: Path) -> int:
    """Validate, normalize, and append new monthly sales transactions to master CSV."""
    if not new_sales_path.exists():
        raise FileNotFoundError(f"Incoming sales file not found: {new_sales_path}")

    logger.info(f"Reading incoming monthly sales from {new_sales_path}...")
    if str(new_sales_path).lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(new_sales_path)
    else:
        df_raw = pd.read_csv(new_sales_path, dtype=str, low_memory=False)

    # Filter out empty rows or total lines
    date_col = next((c for c in df_raw.columns if "date" in str(c).lower()), None)
    if date_col:
        df_raw = df_raw[df_raw[date_col].notna() & ~df_raw[date_col].astype(str).str.contains("Total", case=False, na=False)].copy()

    party_col = next((c for c in df_raw.columns if any(k in str(c).lower() for k in ("party", "patient", "customer"))), None)
    if party_col:
        df_raw = df_raw[df_raw[party_col].notna() & (df_raw[party_col].astype(str).str.strip() != "")].copy()

    logger.info(f"Extracted {len(df_raw):,} raw sales records.")

    # Create safety backup of master CSV
    backup_dir = master_csv_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"customer_data_fields_pre_{ts}.csv"
    if master_csv_path.exists():
        shutil.copy(master_csv_path, backup_path)
        logger.info(f"Safety backup created at {backup_path}")

    # Standardize columns
    df_norm = normalize_sales_dataframe(df_raw)

    # Ensure required canonical fields exist
    field_mappings = {
        "invoice_date": lambda df: df["invoice_date"].apply(format_incoming_date) if "invoice_date" in df.columns else "",
        "invoice_number": lambda df: df["invoice_number"].fillna("").astype(str).str.strip() if "invoice_number" in df.columns else "",
        "customerName": lambda df: df["customerName"].fillna("").astype(str).str.strip() if "customerName" in df.columns else "",
        "customerId": lambda df: df.get("customerId", df.get("customerName", "")).fillna("").astype(str).str.strip(),
        "itemCode": lambda df: df.get("itemCode", df.get("itemId", "")).apply(clean_code),
        "itemId": lambda df: df.get("itemId", df.get("itemCode", "")).apply(clean_code),
        "itemName": lambda df: df["itemName"].fillna("").astype(str).str.strip() if "itemName" in df.columns else "",
        "packing": lambda df: df["packing"].fillna("").astype(str).str.strip() if "packing" in df.columns else "",
        "quantity": lambda df: pd.to_numeric(df["quantity"], errors="coerce").fillna(0) if "quantity" in df.columns else 0,
        "MOBILE_NO": lambda df: df["MOBILE_NO"].apply(clean_code) if "MOBILE_NO" in df.columns else "",
    }

    mapped_df = pd.DataFrame()
    for col, fn in field_mappings.items():
        try:
            mapped_df[col] = fn(df_norm)
        except Exception as e:
            logger.warning(f"Error mapping column {col}: {e}")

    # Align with master CSV header
    df_existing_head = pd.read_csv(master_csv_path, nrows=1)
    target_cols = list(df_existing_head.columns)

    for c in target_cols:
        if c not in mapped_df.columns:
            mapped_df[c] = ""

    mapped_df = mapped_df[target_cols]

    # Append to master CSV
    mapped_df.to_csv(master_csv_path, mode="a", header=False, index=False)
    logger.info(f"Successfully appended {len(mapped_df):,} validated transactions to {master_csv_path}.")
    return len(mapped_df)


def determine_dynamic_as_of_date(purchase_history_df: pd.DataFrame) -> Tuple[date, str]:
    """Calculate the prediction evaluation date and target month from latest sales data."""
    if "invoice_date" in purchase_history_df.columns:
        valid_dates = pd.to_datetime(purchase_history_df["invoice_date"], errors="coerce").dropna()
        if not valid_dates.empty:
            max_dt = valid_dates.max().date()
            # As-of date is the 1st of the next month
            if max_dt.month == 12:
                as_of = date(max_dt.year + 1, 1, 1)
            else:
                as_of = date(max_dt.year, max_dt.month + 1, 1)
            target_month = as_of.strftime("%Y-%m")
            return as_of, target_month

    # Fallback to current month
    today = date.today()
    as_of = today.replace(day=1)
    return as_of, as_of.strftime("%Y-%m")


def run_pipeline(
    master_csv: Path,
    salt_master: Optional[Path] = None,
    new_sales: Optional[Path] = None,
    as_of_date_override: Optional[date] = None,
    target_month_override: Optional[str] = None,
    dry_run: bool = False,
):
    """Execute dynamic data pipeline and refill prediction engine."""
    print("=" * 80)
    print("REFILLCARE DYNAMIC PRODUCTION PIPELINE & PREDICTION ENGINE")
    print("=" * 80)

    # 1. Ingest new sales if specified
    if new_sales:
        print(f"\n[1/4] Ingesting new sales file: {new_sales}")
        append_new_sales_data(new_sales, master_csv)
    else:
        print(f"\n[1/4] Using master dataset: {master_csv}")

    # Check total rows in master
    with open(master_csv, "r", encoding="utf-8", errors="ignore") as f:
        total_rows = sum(1 for _ in f) - 1
    print(f"      Master dataset contains {total_rows:,} total transactions.")

    # 2. Run data pipeline
    print(f"\n[2/4] Running data cleaning & purchase history pipeline...")
    t0 = time.time()
    res = run_refillcare_data_pipeline(
        transactions_path=master_csv,
        salt_master_path=salt_master if salt_master and salt_master.exists() else None,
        output_dir="data/refillcare/processed",
        save_artifacts=True,
    )
    t1 = time.time()

    purchase_history_df = res.get("purchase_history_df")
    if purchase_history_df is None or purchase_history_df.empty:
        parquet_path = Path("data/refillcare/processed/purchase_history.parquet")
        if parquet_path.exists():
            purchase_history_df = pd.read_parquet(parquet_path)
        else:
            raise RuntimeError("Pipeline failed to produce purchase_history_df.")

    max_invoice_dt = pd.to_datetime(purchase_history_df["invoice_date"], errors="coerce").max()
    print(f"      Pipeline completed in {t1 - t0:.2f}s.")
    print(f"      Purchase history records: {len(purchase_history_df):,}")
    print(f"      Latest transaction date in dataset: {max_invoice_dt.strftime('%d-%b-%Y')}")

    # 3. Determine dynamic as-of date
    if as_of_date_override:
        as_of_date = as_of_date_override
        target_month = target_month_override or as_of_date.strftime("%Y-%m")
    else:
        as_of_date, target_month = determine_dynamic_as_of_date(purchase_history_df)
        if target_month_override:
            target_month = target_month_override

    target_month_label = datetime.strptime(target_month, "%Y-%m").strftime("%B %Y")
    print(f"\n[3/4] Evaluating Unified Refill Decisions...")
    print(f"      As-Of Evaluation Date: {as_of_date.isoformat()}")
    print(f"      Target Prediction Month: {target_month_label} ({target_month})")

    engine = UnifiedRefillDecisionEngine()
    decisions_df, decisions_list = engine.evaluate_all_customer_items(purchase_history_df, as_of_date=as_of_date)
    print(f"      Total Evaluated Decisions: {len(decisions_list):,}")

    eligible_count = sum(1 for d in decisions_list if getattr(d, "is_eligible", False))
    print(f"      Eligible Active Decisions: {eligible_count:,}")

    # 4. Persistence into Enterprise Database
    if dry_run:
        print(f"\n[4/4] DRY RUN MODE: Skipping database persistence.")
    else:
        print(f"\n[4/4] Persisting Decisions & Syncing Reminder Cycles...")
        db = SessionLocal()
        try:
            pm = RefillPersistenceManager()
            pm.save_refill_decisions(db, decisions_list)
            pm.sync_reminder_cycles(db, decisions_list)

            target_queue = pm.get_monthly_review_queue(db, target_month=target_month)
            print(f"      Saved {len(decisions_list):,} decisions to database.")
            print(f"      Synced reminder cycles.")
            print(f"      Target Month ({target_month_label}) Scheduled Reminders: {len(target_queue):,}")
        finally:
            db.close()

    print("\n" + "=" * 80)
    print(f"SUCCESS: Pipeline complete! Ready for {target_month_label} operations.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Production RefillCare Monthly Ingestion & Prediction Pipeline"
    )
    parser.add_argument(
        "--new-sales",
        type=Path,
        default=None,
        help="Optional path to new monthly sales CSV/Excel file to ingest into master data.",
    )
    parser.add_argument(
        "--master-csv",
        type=Path,
        default=Path("data/refillcare/customer_data_fields.csv"),
        help="Path to master transaction dataset (default: data/refillcare/customer_data_fields.csv).",
    )
    parser.add_argument(
        "--salt-master",
        type=Path,
        default=Path("data/refillcare/SALT WISE ITEMS.xlsx"),
        help="Path to salt composition master Excel file (default: data/refillcare/SALT WISE ITEMS.xlsx).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Optional as-of date (YYYY-MM-DD). If omitted, automatically defaults to 1st of next month.",
    )
    parser.add_argument(
        "--target-month",
        type=str,
        default=None,
        help="Optional target review month (YYYY-MM). If omitted, derived from as-of date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline and predictions without updating database tables.",
    )

    args = parser.parse_args()

    as_of_override = None
    if args.as_of_date:
        as_of_override = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()

    run_pipeline(
        master_csv=args.master_csv,
        salt_master=args.salt_master,
        new_sales=args.new_sales,
        as_of_date_override=as_of_override,
        target_month_override=args.target_month,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
