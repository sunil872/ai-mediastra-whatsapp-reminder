import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
import app_refillcare as app
from reminder.scheduler import RefillReminderScheduler

model_bundle = app.load_refill_model()
recent_data = app.load_refill_data()

el_df, inel_df, metrics = app.prepare_prediction_overview(recent_data, model_bundle)
scheduler = RefillReminderScheduler()
for _, row in el_df.iterrows():
    scheduler.schedule_refill_cycle(row.to_dict())

sched_df = scheduler.to_dataframe()
due_df = sched_df[sched_df["reminder_date"] == "2026-09-21"].copy()
due_df["mobile_status"] = due_df["MOBILE_NO"].apply(app.determine_mobile_status)

# Sort by stage (-3, -1, 0, 2, 5)
due_df = due_df.sort_values(by=["reminder_stage", "customerName"]).reset_index(drop=True)

print("=" * 100)
print(f"REFILLCARE ACTIVE REMINDERS DUE TODAY (21-09-2026)")
print(f"Total Customers Due: {len(due_df)}")
print("=" * 100)

for idx, r in due_df.iterrows():
    stage_str = f"{r['reminder_stage']:+d} days" if r['reminder_stage'] != 0 else "0 days (Due Today)"
    print(f"[{idx+1}] Customer: {r['customerName']} (ID: {r['customerId']})")
    print(f"    Mobile: {r['MOBILE_NO']} [{r['mobile_status']}]")
    print(f"    Medication: {r['itemName']}")
    print(f"    Last Purchase Date: {r['latest_purchase_date']}")
    print(f"    Expected Refill Date: {r['expected_refill_date']}")
    print(f"    Reminder Stage: {stage_str}")
    print(f"    Message: \"{r['message']}\"")
    print("-" * 100)
