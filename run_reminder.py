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
    exports_dir = PROJECT_ROOT / "exports"
    exports_dir.mkdir(exist_ok=True)

    export_file = exports_dir / f"reminder_list_{today_str}.csv"
    saved = False
    try:
        with open(export_file, "wb") as f:
            f.write(csv_bytes)
        print(f"[OK] Exported delivery CSV: exports/{export_file.name}")
        saved = True
    except PermissionError:
        # If open in Excel, fallback to timestamped file in exports/
        from datetime import datetime
        ts = datetime.now().strftime("%H%M%S")
        fallback_file = exports_dir / f"reminder_list_{today_str}_{ts}.csv"
        try:
            with open(fallback_file, "wb") as f:
                f.write(csv_bytes)
            print(f"[OK] Exported delivery CSV (fallback): exports/{fallback_file.name}")
            saved = True
        except Exception as e:
            print(f"[WARNING] Could not save export CSV to exports/: {e}")

    if not saved:
        print("[WARNING] Could not write CSV file. Please close any open files and re-run.")


if __name__ == "__main__":
    main()
