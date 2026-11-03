"""Retrain the Path A model in THIS Python environment and promote it only if the gate approves.

    .venv\\Scripts\\python scripts\\retrain_production.py              # backup + retrain + gate + promote
    .venv\\Scripts\\python scripts\\retrain_production.py --no-promote # measure only

Run it after uploading the latest sales, with Streamlit / the API stopped and >= 4 GB RAM free.
A model must be trained with the same scikit-learn the app runs (a pickle from another version
does not load, and the app silently falls back to the baseline engine - shown as a red warning).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_FREE_GB = 4.0


def _backup_sqlite() -> None:
    from database.connection import IS_SQLITE, SQLITE_PATH

    if not IS_SQLITE:
        print("PostgreSQL: rely on the RDS snapshot taken before this run (no file backup).")
        return
    db = Path(SQLITE_PATH)
    dst = db.with_name(f"{db.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.db")
    with sqlite3.connect(str(db), timeout=30) as src, sqlite3.connect(str(dst)) as out:
        src.backup(out)
    print(f"Backup: {dst} ({dst.stat().st_size / 1e6:.0f} MB) - delete it once the new model is verified.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-promote", action="store_true", help="evaluate only, never change production")
    ap.add_argument("--force", action="store_true", help="run even with less than 4 GB RAM free")
    args = ap.parse_args()

    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1e9
        if free_gb < MIN_FREE_GB and not args.force:
            print(f"Only {free_gb:.1f} GB RAM free (needs ~{MIN_FREE_GB:.0f} GB). Close other apps or pass --force.")
            return 2
    except ImportError:
        pass

    import sklearn

    from database.connection import SessionLocal
    from refillcare.ml.model_provider import ModelProvider
    from refillcare.ml.service import retrain_cycle

    print(f"scikit-learn {sklearn.__version__} (the new model will be pickled with this version)")
    _backup_sqlite()
    t0 = time.time()
    with SessionLocal() as db:
        res = retrain_cycle(db, promote_best=not args.no_promote, created_by="scripts/retrain_production.py")
        serving = ModelProvider.from_registry(db).describe()
    gate = res.get("gate") or {}
    print(json.dumps({
        "selected_on_validation": res.get("selected_on_validation"),
        "gate_approved": gate.get("approved"),
        "test_window": res.get("test"),
        "gate_failed_checks": [f"{c.get('check')}: {c.get('detail')}" for c in (gate.get("checks") or []) if not c.get("passed")],
        "promoted": res.get("promoted"),
        "serving_now": serving,
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2, default=str))
    if serving.get("degraded"):
        print("Still on the baseline: the gate did not approve. Review the ML tab (gate checks) before promoting by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
