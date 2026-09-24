"""End-to-end data processing pipeline for RefillCare.

Orchestrates loading, cleaning, invoice aggregation, SALT enrichment,
purchase history building, interval computation, and validation.
Saves processed artifacts to data/refillcare/processed/ in Parquet and JSON formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import logging
import pandas as pd

from refillcare.data.cleaning import (
    load_raw_transactions,
    clean_transactions,
    aggregate_invoice_items,
)
from refillcare.data.enrichment import (
    load_salt_master,
    enrich_with_salt,
)
from refillcare.data.history import (
    create_purchase_history,
    compute_interval_statistics,
)
from refillcare.data.validation import (
    validate_dataset,
    generate_validation_report,
)

logger = logging.getLogger("refillcare.pipeline")


def run_refillcare_data_pipeline(
    transactions_path: Optional[Union[str, Path]] = None,
    salt_master_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Execute the full RefillCare data preparation pipeline.

    Args:
        transactions_path: Path to customer_data_fields.csv.
        salt_master_path: Path to SALT WISE ITEMS.xlsx.
        output_dir: Directory to store processed datasets.
        save_artifacts: Whether to write outputs to disk.

    Returns:
        Dict[str, Any]: Results dictionary containing processed DataFrames and validation metrics.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    tx_path = Path(transactions_path) if transactions_path else base_dir / "data" / "refillcare" / "customer_data_fields.csv"
    salt_path = Path(salt_master_path) if salt_master_path else base_dir / "data" / "refillcare" / "SALT WISE ITEMS.xlsx"
    out_dir = Path(output_dir) if output_dir else base_dir / "data" / "refillcare" / "processed"

    print("================================================================")
    print("STARTING REFILLCARE DATA PIPELINE (PHASE 2)")
    print("================================================================")
    print(f"Transaction source: {tx_path}")
    print(f"SALT master source: {salt_path}")
    print(f"Output directory:   {out_dir}")

    # 1. Load Raw Transactions
    print("\n[1/6] Loading raw transactions...")
    raw_df = load_raw_transactions(tx_path)
    print(f"      Loaded {len(raw_df):,} raw transaction rows.")

    # 2. Clean Transactions
    print("\n[2/6] Cleaning transactions (removing exact duplicates, missing keys, dropping mfgDate)...")
    clean_df = clean_transactions(raw_df)
    print(f"      Cleaned dataset contains {len(clean_df):,} rows.")

    # 3. Aggregate Duplicate Invoices
    print("\n[3/6] Aggregating invoice lines by (customerId + invoice_number + invoice_date + itemId)...")
    aggregated_df = aggregate_invoice_items(clean_df)
    print(f"      Aggregated to {len(aggregated_df):,} unique purchase events.")

    # 4. SALT Master Loading & Enrichment
    salt_df = None
    if salt_path.exists():
        print("\n[4/6] Loading SALT master and enriching purchase events...")
        salt_df = load_salt_master(salt_path)
        print(f"      Loaded {len(salt_df):,} unique master items.")
        enriched_events_df = enrich_with_salt(aggregated_df, salt_df, item_code_col="itemCode")
        matched_count = int(enriched_events_df["salt_composition"].notna().sum())
        print(f"      Enrichment complete. Events with active SALT: {matched_count:,} / {len(enriched_events_df):,}")
    else:
        print("\n[4/6] SALT master not found; proceeding without SALT enrichment.")
        enriched_events_df = aggregated_df

    # 5. Purchase History & Interval Calculation
    print("\n[5/6] Building customer-medicine purchase history and intervals...")
    history_df = create_purchase_history(enriched_events_df)
    interval_stats = compute_interval_statistics(history_df)
    print(f"      History generated: {len(history_df):,} events across {history_df.groupby(['customerId', 'itemId']).ngroups:,} customer-medicine histories.")
    print(f"      Intervals calculated: {interval_stats['total_intervals']:,}")
    print(f"      Median refill interval: {interval_stats['median_days']} days (Mean: {interval_stats['mean_days']} days)")
    print(f"      Recurring refill cycles (15-120d): {interval_stats['recurring_15_120_pct']}%")

    # 6. Validation & Quality Checks
    print("\n[6/6] Executing data quality validation...")
    validation_report = validate_dataset(
        raw_df=raw_df,
        clean_df=clean_df,
        aggregated_df=aggregated_df,
        history_df=history_df,
        salt_df=salt_df,
    )
    validation_report["interval_statistics"] = interval_stats
    print(f"      Validation Status: {validation_report['status']}")

    # Save outputs
    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)

        clean_tx_path = out_dir / "clean_transactions.parquet"
        history_path = out_dir / "purchase_history.parquet"
        report_path = out_dir / "data_quality_report.json"

        try:
            clean_df.to_parquet(clean_tx_path, index=False)
            history_df.to_parquet(history_path, index=False)
            print(f"\nArtifacts saved successfully (Parquet):")
            print(f"  - {clean_tx_path}")
            print(f"  - {history_path}")
        except Exception as e:
            print(f"\nParquet write failed ({e}); falling back to CSV...")
            clean_tx_csv = out_dir / "clean_transactions.csv"
            history_csv = out_dir / "purchase_history.csv"
            clean_df.to_csv(clean_tx_csv, index=False)
            history_df.to_csv(history_csv, index=False)
            print(f"  - {clean_tx_csv}")
            print(f"  - {history_csv}")

        # Save JSON report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, default=str)
        print(f"  - {report_path}")

    print("\n================================================================")
    print("REFILLCARE DATA PIPELINE EXECUTION FINISHED")
    print("================================================================")

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "aggregated_df": aggregated_df,
        "history_df": history_df,
        "salt_df": salt_df,
        "validation_report": validation_report,
        "interval_statistics": interval_stats,
    }


if __name__ == "__main__":
    run_refillcare_data_pipeline()
