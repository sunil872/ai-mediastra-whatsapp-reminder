"""RefillCare Phase 12 — Controlled Pilot Execution & Monitoring Test Suite.

Verifies:
1. Pilot configuration defaults and strict manual safeguards (auto_send disabled).
2. Operational funnel metrics computation (candidates, eligible, due, reviewed, accepted, failed).
3. Post-reminder refill matching strictly on (customerId, itemId).
4. Shared phone isolation across multiple distinct patients/medicines.
5. Proper right-censoring handling (window open vs window closed, non-causal association).
6. Prediction accuracy metrics on observed refills (MAE, RMSE, MedAE, +-1d, +-3d, +-7d).
7. Reminder fatigue analysis (multi-stage exposure, +2d/+5d followups).
8. Audit reconciliation between in-memory scheduler and SQLite audit database.
9. Masked phone compliance (zero raw phone exposure in reports).
10. Daily summary report generation and CSV export structure.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
import pandas as pd
import pytest

from refillcare.evaluation.pilot_monitoring import (
    PilotConfiguration,
    compute_pilot_funnel,
    match_post_reminder_refills,
    evaluate_pilot_prediction_accuracy,
    analyze_reminder_fatigue,
    reconcile_audit_state,
    generate_daily_pilot_summary,
)
from reminder.storage import RefillCareStorage
from reminder.scheduler import RefillReminderScheduler


@pytest.fixture
def temp_storage():
    """Create a clean isolated temporary SQLite storage instance."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    storage = RefillCareStorage(db_path=db_path)
    yield storage
    if os.path.exists(db_path):
        os.remove(db_path)


def test_pilot_configuration_safeguards():
    """Verify pilot configuration defaults strictly enforce manual/dry-run safety."""
    config = PilotConfiguration()
    d = config.to_dict()

    assert d["duration_days"] == 30
    assert d["mode"] == "controlled_manual"
    assert d["auto_send_enabled"] is False
    assert d["auto_retry_enabled"] is False
    assert d["approved_stages"] == [-7, -3, -1, 0, 2, 5]
    assert d["max_reminders_per_cycle"] == 3


def test_compute_pilot_funnel():
    """Verify end-to-end operational funnel stage calculations."""
    eligible_df = pd.DataFrame([
        {"customerId": "C1", "itemId": "I1", "Pilot Tier": "Tier A (Strong Pilot)"},
        {"customerId": "C2", "itemId": "I2", "Pilot Tier": "Tier A (Strong Pilot)"},
        {"customerId": "C3", "itemId": "I3", "Pilot Tier": "Tier B (Review Required)"},
    ])
    ineligible_df = pd.DataFrame([
        {"customerId": "C4", "itemId": "I4", "Reason for Ineligibility": "Cold-start"},
    ])
    audit_df = pd.DataFrame([
        {
            "reminder_id": "R1",
            "status": "accepted",
            "is_dry_run": 1,
            "reminder_date": "2026-06-01",
        },
        {
            "reminder_id": "R2",
            "status": "accepted",
            "is_dry_run": 0,
            "reminder_date": "2026-06-01",
        },
        {
            "reminder_id": "R3",
            "status": "failed",
            "is_dry_run": 1,
            "reminder_date": "2026-06-01",
        },
        {
            "reminder_id": "R4",
            "status": "duplicate_prevented",
            "is_dry_run": 1,
            "reminder_date": "2026-06-01",
        },
    ])

    funnel = compute_pilot_funnel(
        audit_df=audit_df,
        eligible_df=eligible_df,
        ineligible_df=ineligible_df,
        current_date="2026-06-01",
    )

    assert funnel["total_candidates"] == 4
    assert funnel["eligible_candidates"] == 3
    assert funnel["cold_start_excluded"] == 1
    assert funnel["tier_a_count"] == 2
    assert funnel["tier_b_count"] == 1
    assert funnel["dispatch_attempted"] == 3
    assert funnel["provider_accepted"] == 2
    assert funnel["provider_accepted_dry"] == 1
    assert funnel["provider_accepted_live"] == 1
    assert funnel["provider_failed"] == 1
    assert funnel["duplicates_prevented"] == 1


def test_match_post_reminder_refills_strict_identity():
    """Verify matching is strictly by (customerId, itemId) and temporal order."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "REM_101",
            "customer_id": "CUST_001",
            "item_id": "ITEM_A",
            "customer_name": "Alice Smith",
            "item_name": "Metformin 500mg",
            "phone_last4": "1234",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
            "is_dry_run": 1,
        },
        {
            "reminder_id": "REM_102",
            "customer_id": "CUST_002",
            "item_id": "ITEM_B",
            "customer_name": "Bob Jones",
            "item_name": "Amlodipine 5mg",
            "phone_last4": "1234",  # Same phone number as Alice (shared phone scenario)
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
            "is_dry_run": 1,
        },
    ])

    # Transactions: Only Alice purchases ITEM_A on 2026-06-03. Bob does NOT purchase ITEM_B.
    # Someone else purchases ITEM_B on 2026-06-02 (CUST_999).
    transactions_df = pd.DataFrame([
        {"customerId": "CUST_001", "itemId": "ITEM_A", "invoice_date": "2026-06-03"},
        {"customerId": "CUST_999", "itemId": "ITEM_B", "invoice_date": "2026-06-02"},
    ])

    matched = match_post_reminder_refills(
        audit_df=audit_df,
        transactions_df=transactions_df,
        observation_window_days=30,
        reference_date="2026-06-15",
    )

    assert len(matched) == 2

    # Alice should have an observed refill on 2026-06-03 (2 days error from expected 2026-06-01)
    alice_row = matched[matched["customerId"] == "CUST_001"].iloc[0]
    assert alice_row["refill_status"] == "Observed Refill"
    assert alice_row["actual_refill_date"] == "2026-06-03"
    assert alice_row["days_to_refill"] == 2
    assert alice_row["prediction_error_days"] == 2
    assert alice_row["absolute_error_days"] == 2
    assert alice_row["is_accurate_7d"] == True
    assert alice_row["is_accurate_3d"] == True
    assert alice_row["is_accurate_1d"] == False

    # Bob should be Pending Observation because reference date is 2026-06-15 (14 days elapsed <= 30)
    # Crucially, Bob must NOT match CUST_999's purchase or Alice's purchase despite shared phone!
    bob_row = matched[matched["customerId"] == "CUST_002"].iloc[0]
    assert bob_row["refill_status"] == "Pending Observation (Window Open)"
    assert pd.isna(bob_row["actual_refill_date"])


def test_right_censoring_handling():
    """Verify right-censoring properly distinguishes pending observation from closed window."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "REM_RECENT",
            "customer_id": "C1",
            "item_id": "I1",
            "reminder_date": "2026-06-10",
            "expected_refill_date": "2026-06-10",
            "reminder_stage": "0",
            "status": "accepted",
        },
        {
            "reminder_id": "REM_OLD",
            "customer_id": "C2",
            "item_id": "I2",
            "reminder_date": "2026-04-01",
            "expected_refill_date": "2026-04-01",
            "reminder_stage": "0",
            "status": "accepted",
        },
    ])

    transactions_df = pd.DataFrame(columns=["customerId", "itemId", "invoice_date"])

    # Reference date: 2026-06-20
    matched = match_post_reminder_refills(
        audit_df=audit_df,
        transactions_df=transactions_df,
        observation_window_days=30,
        reference_date="2026-06-20",
    )

    recent_rec = matched[matched["customerId"] == "C1"].iloc[0]
    old_rec = matched[matched["customerId"] == "C2"].iloc[0]

    # C1 is 10 days elapsed <= 30 -> Window Open
    assert recent_rec["refill_status"] == "Pending Observation (Window Open)"

    # C2 is 80 days elapsed > 30 -> Window Closed
    assert old_rec["refill_status"] == "No Observed Refill (Window Closed)"


def test_evaluate_pilot_prediction_accuracy():
    """Verify prediction error metrics on observed post-reminder refills."""
    matched_df = pd.DataFrame([
        {
            "refill_status": "Observed Refill",
            "prediction_error_days": 1,
            "absolute_error_days": 1,
        },
        {
            "refill_status": "Observed Refill",
            "prediction_error_days": -3,
            "absolute_error_days": 3,
        },
        {
            "refill_status": "Observed Refill",
            "prediction_error_days": 8,
            "absolute_error_days": 8,
        },
        {
            "refill_status": "Pending Observation (Window Open)",
            "prediction_error_days": None,
            "absolute_error_days": None,
        },
    ])

    metrics = evaluate_pilot_prediction_accuracy(matched_df)

    assert metrics["evaluated_refill_count"] == 3
    # abs errors: [1, 3, 8] -> Mean = 4.0, Median = 3.0
    assert metrics["mae_days"] == 4.0
    assert metrics["median_absolute_error_days"] == 3.0
    # within 1d: 1/3 = 33.33%
    assert metrics["within_1_day_pct"] == 33.33
    # within 3d: 2/3 = 66.67%
    assert metrics["within_3_days_pct"] == 66.67
    # within 7d: 2/3 = 66.67%
    assert metrics["within_7_days_pct"] == 66.67


def test_analyze_reminder_fatigue():
    """Verify reminder distribution and fatigue monitoring metrics."""
    audit_df = pd.DataFrame([
        {"customer_id": "C1", "reminder_stage": "-3 days"},
        {"customer_id": "C1", "reminder_stage": "0 days"},
        {"customer_id": "C1", "reminder_stage": "+2 days"},
        {"customer_id": "C2", "reminder_stage": "0 days"},
        {"customer_id": "C3", "reminder_stage": "+5 days"},
    ])

    fatigue = analyze_reminder_fatigue(audit_df)

    assert fatigue["total_dispatches"] == 5
    assert fatigue["unique_customers"] == 3
    assert fatigue["avg_reminders_per_customer"] == 1.67
    assert fatigue["max_reminders_single_customer"] == 3
    assert fatigue["customers_with_3_plus_reminders"] == 1
    assert fatigue["followup_stage_count"] == 2  # +2 days and +5 days


def test_reconcile_audit_state(temp_storage):
    """Verify reconciliation between in-memory scheduler and persistent SQLite database."""
    # Seed DB with one reminder
    temp_storage.insert_or_update_scheduled_reminder({
        "reminder_id": "REM_SYNCED",
        "customer_id": "C1",
        "item_id": "I1",
        "customer_name": "Test Customer",
        "item_name": "Test Med",
        "MOBILE_NO": "9876543210",
        "reminder_date": "2026-06-01",
        "expected_refill_date": "2026-06-01",
        "reminder_stage": "0",
        "message": "Reminder text",
    })

    scheduler_list = [
        {"reminder_id": "REM_SYNCED"},
        {"reminder_id": "REM_MISSING_IN_DB"},
    ]

    reconciliation = reconcile_audit_state(scheduler_list, temp_storage)

    assert reconciliation["scheduler_count"] == 2
    assert reconciliation["database_audit_count"] == 1
    assert reconciliation["in_sync_count"] == 1
    assert reconciliation["missing_in_db"] == 1
    assert reconciliation["stale_sending_count"] == 0
    assert reconciliation["audit_health"] == "ATTENTION_REQUIRED"


def test_generate_daily_pilot_summary():
    """Verify daily summary export dataframe generation."""
    eligible_df = pd.DataFrame([{"customerId": "C1", "itemId": "I1", "Pilot Tier": "Tier A (Strong Pilot)"}])
    audit_df = pd.DataFrame([{"reminder_id": "R1", "status": "accepted", "is_dry_run": 1, "reminder_date": "2026-06-01"}])
    
    summary_df = generate_daily_pilot_summary(
        audit_df=audit_df,
        eligible_df=eligible_df,
        report_date="2026-06-01",
    )

    assert not summary_df.empty
    assert "Report Date" in summary_df.columns
    assert "Total Candidates" in summary_df.columns
    assert "Provider Accepted (Dry-Run)" in summary_df.columns
    assert "Post-Reminder Refill Rate (%)" in summary_df.columns
    assert summary_df["Report Date"].iloc[0] == "2026-06-01"


def test_masked_phone_safety_in_monitoring():
    """Confirm monitoring output tables never expose full phone numbers."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "R1",
            "customer_id": "C1",
            "item_id": "I1",
            "customer_name": "Alice",
            "item_name": "Metformin",
            "phone_last4": "9999",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
            "is_dry_run": 1,
        }
    ])

    matched = match_post_reminder_refills(audit_df, pd.DataFrame(), reference_date="2026-06-05")
    
    assert "phone_masked" in matched.columns
    assert "MOBILE_NO" not in matched.columns
    # Ensure value is just the masked last 4 digits
    assert matched["phone_masked"].iloc[0] == "9999"
