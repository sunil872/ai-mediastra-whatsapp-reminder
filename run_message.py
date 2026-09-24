"""CLI runner for WhatsApp Reminder Messaging (Strict DRY-RUN mode)."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reminder.storage import RefillCareStorage
from reminder.send_message import dispatch_refill_message
from reminder.message_logs import append_message_audit_log


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("RefillCare - WhatsApp Dispatch (Strict DRY-RUN Simulation)")
    print("=" * 60)

    storage = RefillCareStorage()
    snapshots_list = storage.get_prediction_snapshots()
    if not snapshots_list:
        print("[WARN] No prediction snapshots found.")
        sys.exit(1)

    df = pd.DataFrame(snapshots_list)
    today_str = date.today().strftime("%Y-%m-%d")
    due_today = df[(df["reminder_date"].astype(str) == today_str) & (df["mobile_status"] == "Valid")] if "reminder_date" in df.columns and "mobile_status" in df.columns else pd.DataFrame()

    print(f"Eligible delivery-ready records for today ({today_str}): {len(due_today):,}")
    print("Simulating message dispatch in DRY-RUN mode (No real messages sent)...")

    success_cnt = 0
    for _, r in due_today.head(10).iterrows():
        res = dispatch_refill_message(
            to_phone=str(r.get("phone_number", "")),
            customer_name=str(r.get("customer_name", "Patient")),
            medication_name=str(r.get("item_name", "Medication")),
            refill_date=str(r.get("expected_refill_date", "")),
            dry_run=True,
        )
        if res.get("success"):
            success_cnt += 1
            append_message_audit_log(
                phone_number=str(r.get("phone_number", "")),
                customer_name=str(r.get("customer_name", "Patient")),
                medication_name=str(r.get("item_name", "Medication")),
                status="DRY_RUN_SUCCESS",
                message_id=res.get("message_id"),
                is_dry_run=True,
            )

    print(f"[OK] Simulated {success_cnt} dry-run reminder dispatches successfully.")


if __name__ == "__main__":
    main()
