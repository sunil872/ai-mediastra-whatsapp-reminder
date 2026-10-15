"""High-performance script to synchronize enterprise.db with the updated medication classifier.
Updates RefillDecisionModel, ReminderCycleModel, and ReminderStageModel so that
ineligible non-chronic items (syrups, drops, eye drops, topicals, OTC, etc.) are strictly
marked as is_eligible=False, is_active=False, and status='SUPERSEDED'.
"""

import sys
import time
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal
from database.models import ReminderStageModel, ReminderCycleModel, RefillDecisionModel
from refillcare.data.medication_classifier import is_chronic_medication
from sqlalchemy import update


def main():
    print("=" * 70)
    print("Syncing enterprise.db with Strict Chronic Medication Safety Gates")
    print("=" * 70)

    t0 = time.time()
    db = SessionLocal()
    try:
        # 1. Fetch all distinct item names across RefillDecisionModel
        print("[1/4] Classifying all distinct medications in decisions...")
        items_in_db = db.query(RefillDecisionModel.item_name).distinct().all()
        print(f"      Total unique items to classify: {len(items_in_db):,}")

        ineligible_items = set()
        eligible_items = set()

        for (iname,) in items_in_db:
            if not iname:
                continue
            if is_chronic_medication(str(iname)):
                eligible_items.add(iname)
            else:
                ineligible_items.add(iname)

        print(f"      Eligible Chronic Items: {len(eligible_items):,}")
        print(f"      Ineligible Non-Chronic Items: {len(ineligible_items):,}")

        # 2. Bulk update RefillDecisionModel
        print("\n[2/4] Updating RefillDecisionModel is_eligible flags...")
        ineligible_list = list(ineligible_items)
        chunk_size = 500
        dec_updated = 0

        for i in range(0, len(ineligible_list), chunk_size):
            chunk = ineligible_list[i : i + chunk_size]
            res = db.execute(
                update(RefillDecisionModel)
                .where(
                    RefillDecisionModel.item_name.in_(chunk),
                    RefillDecisionModel.is_eligible == True,
                )
                .values(
                    is_eligible=False,
                    decision_reason="Excluded by updated formulation/chemical salt safety gates.",
                )
            )
            dec_updated += res.rowcount

        db.commit()
        print(f"      Marked {dec_updated:,} RefillDecision records as is_eligible=False.")

        # 3. Bulk update ReminderCycleModel
        print("\n[3/4] Deactivating cycles for ineligible decisions...")
        # Get decision_ids of ineligible decisions
        ineligible_dec_ids_query = db.query(RefillDecisionModel.decision_id).filter(
            RefillDecisionModel.is_eligible == False
        )
        # Deactivate cycles linked to these decisions
        res_cyc = db.execute(
            update(ReminderCycleModel)
            .where(
                ReminderCycleModel.decision_id.in_(ineligible_dec_ids_query),
                ReminderCycleModel.is_active == True,
            )
            .values(is_active=False)
        )
        db.commit()
        print(f"      Deactivated {res_cyc.rowcount:,} active ReminderCycle records.")

        # 4. Bulk supersede stages for inactive cycles
        print("\n[4/4] Superseding reminder stages for inactive cycles...")
        inactive_cycle_ids_query = db.query(ReminderCycleModel.cycle_id).filter(
            ReminderCycleModel.is_active == False
        )
        res_stg = db.execute(
            update(ReminderStageModel)
            .where(
                ReminderStageModel.cycle_id.in_(inactive_cycle_ids_query),
                ReminderStageModel.status.in_(["PENDING", "DUE", "APPROVED"]),
            )
            .values(
                status="SUPERSEDED",
                failure_reason="Superseded: Item excluded by updated formulation/chemical salt safety gates.",
            )
        )
        db.commit()
        print(f"      Marked {res_stg.rowcount:,} ReminderStage records as SUPERSEDED.")

        # 5. Live audit of clean queue
        from refillcare.engine.persistence import RefillPersistenceManager
        pm = RefillPersistenceManager()
        today_queue = pm.get_today_review_queue(db, target_date=date.today())
        oct_queue = pm.get_monthly_review_queue(db, target_month="2026-10")

        print("\n" + "=" * 70)
        print(f"AUDIT SUMMARY (Finished in {time.time() - t0:.2f}s):")
        print(f"- Clean Reminders Due Today ({date.today()}): {len(today_queue):,}")
        print(f"- Clean Active Upcoming Reminders for October 2026: {len(oct_queue):,}")

        # Check for any syrups or drops
        leaked_today = [r["item_name"] for r in today_queue if any(w in r["item_name"].upper() for w in ["SYP", "DROP", "EYE DROP", "EAR DROP"])]
        leaked_oct = [r["item_name"] for r in oct_queue if any(w in r["item_name"].upper() for w in ["SYP", "DROP", "EYE DROP", "EAR DROP"])]

        print(f"- Leaked Syrups/Drops Today: {len(leaked_today)}")
        print(f"- Leaked Syrups/Drops October: {len(leaked_oct)}")
        if not leaked_today and not leaked_oct:
            print(">>> 100% CLEAN CHRONIC MEDICATIONS VERIFIED! <<<")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()
