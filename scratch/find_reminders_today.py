import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import joblib
import warnings
warnings.filterwarnings("ignore")

import app_refillcare as app
from reminder.scheduler import RefillReminderScheduler, REMINDER_STAGES, format_reminder_message

def get_due_customers(target_date_str="2026-09-21"):
    model_bundle = app.load_refill_model()
    recent_data = app.load_refill_data()
    hist_data = app.load_purchase_history()

    print(f"Target Date: {target_date_str}")
    
    # 1. Evaluate on active dataset (test.parquet)
    print("\n--- Evaluating Active Dashboard Dataset (test.parquet) ---")
    el_df_active, inel_df_active, m_active = app.prepare_prediction_overview(recent_data, model_bundle)
    
    sched_active = RefillReminderScheduler()
    for _, row in el_df_active.iterrows():
        sched_active.schedule_refill_cycle(row.to_dict())
    
    all_active = sched_active.to_dataframe()
    due_active = all_active[all_active["reminder_date"] == target_date_str].copy()
    print(f"Total eligible histories: {len(el_df_active)}")
    print(f"Total scheduled records (6 stages): {len(all_active)}")
    print(f"Reminders due on {target_date_str}: {len(due_active)}")
    
    if not due_active.empty:
        print("\nStage breakdown (active dataset):")
        print(due_active["reminder_stage"].value_counts().to_dict())
        
        # Add mobile status
        due_active["Mobile Status"] = due_active["MOBILE_NO"].apply(app.determine_mobile_status)
        
        # Save active due CSV
        out_path = "scratch/due_reminders_2026-09-21.csv"
        due_active.to_csv(out_path, index=False)
        print(f"Saved due records to {out_path}")
        
        # Print detailed list
        display_cols = ["customerId", "customerName", "MOBILE_NO", "Mobile Status", "itemName", "latest_purchase_date", "expected_refill_date", "reminder_stage", "reminder_date"]
        print("\nList of customers due for reminder today (21-09-2026):")
        print(due_active[display_cols].to_string())

if __name__ == "__main__":
    get_due_customers("2026-09-21")
