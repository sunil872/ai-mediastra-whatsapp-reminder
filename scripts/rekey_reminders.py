"""Cut reminders over to the new patient / therapy keys (see refillcare/engine/rekey.py).

Set the switches in .env first, then:

    python scripts/rekey_reminders.py            # dry run: what would change
    python scripts/rekey_reminders.py --apply    # backs up the SQLite DB, then re-keys

Run once, right after switching REFILLCARE_PATIENT_IDENTITY / REFILLCARE_ITEM_GROUPING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from database.connection import SessionLocal  # noqa: E402
from refillcare.engine.rekey import backup_sqlite_db, rekey_reminder_cycles  # noqa: E402
from refillcare.ml.service import load_history  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = ap.parse_args()

    if args.apply:
        backup = backup_sqlite_db()
        print(f"Backup: {backup}" if backup else "Not SQLite: take a database backup before continuing.", file=sys.stderr if not backup else sys.stdout)

    db = SessionLocal()
    try:
        print(json.dumps(rekey_reminder_cycles(db, load_history(), apply=args.apply), indent=2, default=str))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
