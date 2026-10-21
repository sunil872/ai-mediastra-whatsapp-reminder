"""Dynamic Production Sales Ingestion & RefillCare Prediction Pipeline.

Dynamically processes the canonical sales transaction dataset
(data/refillcare/customer_data_fields.csv), detects the latest transaction date,
and evaluates UnifiedRefillDecisionEngine predictions for the upcoming target horizon.

Supports:
1. Dynamic baseline execution: Runs directly on current master data.
2. Incremental sales ingestion: Pass `--new-sales <path>` to validate, normalize, backup,
   and append any new sales dataset before generating predictions.
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
from refillcare.data.sales_ingestion import (
    normalize_sales_dataframe,
    append_sales_records_to_master_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("refillcare.sales_pipeline")


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
    """Validate, normalize, backup pre-image, and append new sales transactions to master CSV."""
    if not new_sales_path.exists():
        raise FileNotFoundError(f"Incoming sales file not found: {new_sales_path}")

    logger.info(f"Reading incoming sales from {new_sales_path}...")
    if str(new_sales_path).lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(new_sales_path)
    else:
        df_raw = pd.read_csv(new_sales_path, dtype=str, low_memory=False)

    # Filter out empty rows or total lines
    date_col = next((c for c in df_raw.columns if "date" in str(c).lower()), None)
    if date_col:
        df_raw = df_raw[df_raw[date_col].notna() & ~df_raw[date_col].astype(str).str.contains("Total", case=False, na=False, regex=False)].copy()

    party_col = next((c for c in df_raw.columns if any(k in str(c).lower() for k in ("party", "patient", "customer"))), None)
    if party_col:
        df_raw = df_raw[df_raw[party_col].notna() & (df_raw[party_col].astype(str).str.strip() != "")].copy()

    logger.info(f"Extracted {len(df_raw):,} raw sales records.")

    # Standardize columns
    df_norm = normalize_sales_dataframe(df_raw)

    # Safely append to master CSV with timestamped pre-image safety backup
    appended = append_sales_records_to_master_csv(
        records_df=df_norm,
        master_csv_path=master_csv_path,
    )
    logger.info(f"Successfully processed and appended {appended:,} transactions to master CSV.")
    return appended


def determine_dynamic_as_of_date(
    purchase_history_df: pd.DataFrame,
    horizon_days: int = 30,
) -> Tuple[date, str, Tuple[date, date]]:
    """
    Calculate evaluation date, target month partition, and rolling outreach horizon.
    
    Continuous-Observation Logic:
    - T_max is the latest valid transaction date in purchase history.
    - Evaluation date (as_of) is T_max + 1 day (e.g. if sales run through 13-Oct-2026, as_of = 14-Oct-2026).
    - Rolling outreach horizon covers [as_of, as_of + horizon_days - 1] (e.g. 14-Oct-2026 to 12-Nov-2026).
    - Target month partition remains as_of.strftime("%Y-%m") for month-level reporting.
    """
    if "invoice_date" in purchase_history_df.columns:
        valid_dates = pd.to_datetime(purchase_history_df["invoice_date"], errors="coerce").dropna()
        if not valid_dates.empty:
            max_dt = valid_dates.max().date()
            as_of = max_dt + timedelta(days=1)
            target_month = as_of.strftime("%Y-%m")
            horizon_end = as_of + timedelta(days=horizon_days - 1)
            return as_of, target_month, (as_of, horizon_end)

    # Fallback to current month/day
    today = date.today()
    target_month = today.strftime("%Y-%m")
    horizon_end = today + timedelta(days=horizon_days - 1)
    return today, target_month, (today, horizon_end)


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
        rolling_horizon = (as_of_date, as_of_date + timedelta(days=29))
    else:
        as_of_date, target_month, rolling_horizon = determine_dynamic_as_of_date(purchase_history_df)
        if target_month_override:
            target_month = target_month_override

    target_month_label = datetime.strptime(target_month, "%Y-%m").strftime("%B %Y")
    print(f"\n[3/4] Evaluating Unified Refill Decisions...")
    print(f"      As-Of Evaluation Date: {as_of_date.isoformat()}")
    print(f"      Target Prediction Month: {target_month_label} ({target_month})")
    print(f"      Rolling 30-Day Outreach Horizon: {rolling_horizon[0].isoformat()} to {rolling_horizon[1].isoformat()}")

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
            horizon_queue = pm.get_date_range_review_queue(
                db, start_date=rolling_horizon[0], end_date=rolling_horizon[1]
            )
            print(f"      Saved {len(decisions_list):,} decisions to database.")
            print(f"      Synced reminder cycles.")
            print(f"      Target Month ({target_month_label}) Scheduled Reminders: {len(target_queue):,}")
            print(f"      Rolling 30-Day Active Reminders ({rolling_horizon[0]} to {rolling_horizon[1]}): {len(horizon_queue):,}")
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
        default=None,
        help="Path to the item / salt master (default: data/refillcare/SALT ITEMS LIST.csv, or REFILLCARE_SALT_CATALOG).",
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
