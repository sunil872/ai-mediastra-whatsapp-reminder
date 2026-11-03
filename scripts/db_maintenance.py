"""SQLite housekeeping for the local / single-store database (on AWS RDS this is not needed).

    python scripts/db_maintenance.py                 # report sizes, fold the WAL file back (checkpoint)
    python scripts/db_maintenance.py --vacuum        # also compact the file (online backup taken first)

Stop Streamlit / the API first: a running reader keeps the WAL file from being folded back.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1) if path.exists() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="SQLite file (default: the configured store database)")
    ap.add_argument("--vacuum", action="store_true", help="compact the file after an online backup")
    args = ap.parse_args()

    if args.db:
        db = Path(args.db)
    else:
        from database.connection import SQLITE_PATH

        db = Path(SQLITE_PATH)
    if not db.exists():
        print(f"Not found: {db}")
        return 1
    wal = db.with_name(db.name + "-wal")
    print(f"{db.name}: {_mb(db)} MB, WAL {_mb(wal)} MB")

    con = sqlite3.connect(str(db), timeout=30)
    busy, log_pages, done = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    print("WAL checkpoint:", "blocked by an open reader - stop Streamlit / the API and re-run" if busy else f"ok ({done} pages)")

    if args.vacuum:
        backup = db.with_name(f"{db.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.db")
        with sqlite3.connect(str(backup)) as dst:
            con.backup(dst)  # consistent copy even while the file is in use
        print(f"Backup: {backup} ({_mb(backup)} MB)")
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"After VACUUM: {_mb(db)} MB (delete the backup once the app is verified)")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
