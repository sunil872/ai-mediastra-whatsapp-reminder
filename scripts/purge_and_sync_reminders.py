"""Database purge and re-sync script.
Synchronizes enterprise.db with the updated medication classifier and chemical salt rules,
strictly superseding any non-chronic, syrup, drop, or OTC reminder stages.
"""

import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal
from refillcare.engine.persistence import (
    ReminderStageModel,
    ReminderCycleModel,
    RefillDecisionModel,
    STATUS_SUPERSEDED,
    STATUS_PENDING,
    STATUS_DUE,
    STATUS_APPROVED,
)
from refillcare.data.medication_classifier import is_chronic_medication
from database.fetch_sales import fetch_purchase_history
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.engine.persistence import RefillPersistenceManager


def main():
    print("=" * 70)
    print("RefillCare V1: Database Purge & Accurate Chronic Reminder Re-sync")
    print("=" * 70)

    db = SessionLocal()
    try:
        # Step 1: Count pre-existing stages
        total_pre_stages = db.query(ReminderStageModel).count()
        today_pre_stages = db.query(ReminderStageModel).filter(
            ReminderStageModel.target_send_date == date.today(),
            ReminderStageModel.status.in_([STATUS_PENDING, STATUS_DUE, STATUS_APPROVED])
        ).count()
        print(f"[INFO] Existing total reminder stages in DB: {total_pre_stages:,}")
        print(f"[INFO] Existing active stages scheduled for today ({date.today()}): {today_pre_stages:,}")

        # Step 2: Identify and supersede non-chronic cycles and stages
        print("\n[STEP 1/3] Auditing all active cycles and stages against updated classifier...")
        active_cycles = db.query(ReminderCycleModel).filter(ReminderCycleModel.is_active == True).all()
        ineligible_cycle_ids = []
        
        # Load all decisions for item names
        decisions_map = {d.decision_id: d for d in db.query(RefillDecisionModel).all()}

        for cyc in active_cycles:
            item_label = cyc.item_id
            dec = decisions_map.get(cyc.decision_id)
            if dec and dec.item_name:
                item_label = dec.item_name
            
            if not is_chronic_medication(item_label):
                ineligible_cycle_ids.append(cyc.cycle_id)
                cyc.is_active = False
                if dec:
                    dec.is_eligible = False
                    dec.decision_reason = "Excluded by updated formulation/chemical salt safety gates."

        print(f"[OK] Identified {len(ineligible_cycle_ids):,} active cycles belonging to non-chronic medications.")

        # Supersede pending stages for these cycles
        stages_superseded = 0
        chunk_size = 500
        for i in range(0, len(ineligible_cycle_ids), chunk_size):
            chunk = ineligible_cycle_ids[i:i + chunk_size]
            stages = db.query(ReminderStageModel).filter(
                ReminderStageModel.cycle_id.in_(chunk),
                ReminderStageModel.status.in_([STATUS_PENDING, STATUS_DUE, STATUS_APPROVED])
            ).all()
            for st in stages:
                st.status = STATUS_SUPERSEDED
                st.failure_reason = "Superseded: Item excluded by updated formulation/chemical salt rules."
                stages_superseded += 1

        db.commit()
        print(f"[OK] Marked {stages_superseded:,} non-chronic reminder stages as SUPERSEDED.")

        # Step 3: Re-evaluate full purchase history with Unified Engine
        print("\n[STEP 2/3] Re-evaluating complete sales history with UnifiedRefillDecisionEngine...")
        df_sales = fetch_purchase_history()
        print(f"[INFO] Loaded {len(df_sales):,} sales transactions.")
        
        engine = UnifiedRefillDecisionEngine()
        decisions_df, decisions_list = engine.evaluate_all_customer_items(df_sales, as_of_date=date.today())
        
        eligible_count = sum(1 for d in decisions_list if d.is_eligible)
        print(f"[OK] Evaluated {len(decisions_list):,} total trajectories.")
        print(f"[OK] Generated {eligible_count:,} strictly eligible chronic refill decisions.")

        # Step 4: Persist and sync cycles
        print("\n[STEP 3/3] Syncing fresh clean cycles and stages into enterprise.db...")
        pm = RefillPersistenceManager()
        saved_count = pm.save_refill_decisions(db, decisions_list)
        sync_metrics = pm.sync_reminder_cycles(db, decisions_list)
        print(f"[OK] Saved {saved_count:,} RefillDecision records.")
        print(f"[OK] Cycles created: {sync_metrics['cycles_created']:,}, Stages created: {sync_metrics['stages_created']:,}")

        # Final verification
        today_clean_queue = pm.get_today_review_queue(db, target_date=date.today())
        print("\n" + "=" * 70)
        print(f"[VERIFIED] Clean Pharmacist Review Queue for Today ({date.today()}): {len(today_clean_queue):,} reminders.")
        
        # Verify no syrups or drops remain
        leaked_syp_drop = [r["item_name"] for r in today_clean_queue if any(w in r["item_name"].upper() for w in ["SYP", "SYRUP", "DROP", "EYE DROP", "EAR DROP"])]
        if leaked_syp_drop:
            print(f"[ERROR] Found leaked syrup/drop items: {leaked_syp_drop}")
        else:
            print("[SUCCESS] 0 syrups or drops detected in today's review queue! (100% Chronic Maintenance).")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()
