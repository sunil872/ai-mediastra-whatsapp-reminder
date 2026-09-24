"""CLI runner for RefillCare V1 Unified Refill Decision Generation & Persistence."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal
from database.fetch_sales import fetch_purchase_history
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.engine.persistence import RefillPersistenceManager
from model.prediction_history import PredictionHistoryManager
from model.save_load_models import load_model_bundle
from model.predictor import predict_refill_cycles


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("RefillCare V1 - Unified Refill Decision Generation & Persistence")
    print("=" * 60)

    df = fetch_purchase_history()
    if df.empty:
        print("[ERROR] No purchase history available for predictions.")
        sys.exit(1)

    print(f"Loaded {len(df):,} transactions.")
    pred_date = date.today()

    print(f"Evaluating decisions with UnifiedRefillDecisionEngine (as of {pred_date})...")
    engine = UnifiedRefillDecisionEngine()
    decisions_df, decisions_list = engine.evaluate_all_customer_items(df, as_of_date=pred_date)

    db = SessionLocal()
    try:
        pm = RefillPersistenceManager()
        saved_decisions = pm.save_refill_decisions(db, decisions_list)
        sync_metrics = pm.sync_reminder_cycles(db, decisions_list)

        eligible_count = sum(1 for d in decisions_list if d.is_eligible)
        review_count = len(decisions_list) - eligible_count

        print(f"[OK] Evaluated {len(decisions_list):,} total customer-item histories.")
        print(f"[OK] Generated {eligible_count:,} eligible refill decisions.")
        print(f"[INFO] {review_count:,} customer-item histories routed to Clinical Review Queue.")
        print(f"[OK] Persisted {saved_decisions:,} RefillDecisions in database.")
        print(f"[OK] Lifecycle Sync: {sync_metrics['cycles_created']:,} cycles, {sync_metrics['stages_created']:,} stages created.")
        if sync_metrics["stages_superseded"] > 0:
            print(f"[INFO] Repurchase Reset: {sync_metrics['stages_superseded']:,} stages superseded by new purchases.")

    finally:
        db.close()


if __name__ == "__main__":
    main()

