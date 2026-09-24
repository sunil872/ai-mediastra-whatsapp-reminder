"""CLI runner for Daily Reminder Scheduling & Export."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reminder.storage import RefillCareStorage
from reminder.reminder_engine import RefillReminderEngine


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("RefillCare - Reminder Scheduler & 10-Column CSV Exporter")
    print("=" * 60)

    from database.connection import SessionLocal
    from refillcare.engine.persistence import RefillPersistenceManager

    db = SessionLocal()
    try:
        today_date = date.today()
        today_str = today_date.strftime("%Y-%m-%d")
        pm = RefillPersistenceManager()
        queue = pm.get_today_review_queue(db, target_date=today_date)
        due_today = pd.DataFrame(queue) if queue else pd.DataFrame()

        print(f"Reminders scheduled & active for today ({today_str}): {len(due_today):,}")
    finally:
        db.close()

    if due_today.empty:
        print(f"[INFO] No reminders due for today ({today_str}).")
        return

    # Build 10-column delivery CSV
    csv_bytes = RefillReminderEngine.build_10_column_export_csv(due_today)
    candidate_files = [
        PROJECT_ROOT / f"reminder_list_{today_str}.csv",
        PROJECT_ROOT / f"reminder_list_{today_str}_validated.csv",
        PROJECT_ROOT / f"reminder_list_{today_str}_export.csv",
        PROJECT_ROOT / f"reminder_list_{today_str}_latest.csv",
    ]
    
    saved = False
    for candidate in candidate_files:
        try:
            with open(candidate, "wb") as f:
                f.write(csv_bytes)
            print(f"[OK] Exported delivery CSV: {candidate.name}")
            saved = True
            break
        except PermissionError:
            continue
            
    if not saved:
        print("[WARNING] Could not overwrite open CSV files. Please close Excel and re-run.")


if __name__ == "__main__":
    main()
