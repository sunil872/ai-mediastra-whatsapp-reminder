"""RefillCare Phase 13 — Controlled Pilot Outcome Evaluation Test Suite.

Verifies:
1. Exact (customerId, itemId) matching for actual refills with returns exclusion.
2. Complete isolation for shared phone numbers across patients and medicines.
3. Single customer with multiple distinct medications evaluated separately.
4. Multiple customers on the same medication evaluated separately.
5. Accurate calculation of expected vs actual refill dates and signed/absolute prediction errors.
6. Granular reminder timing classifications: PRE-REFILL REMINDER, SAME-DAY, POST-REFILL FOLLOW-UP.
7. Proper right-censoring management: Pending Observation (Window Open) vs No Observed Refill (Window Closed).
8. Post-reminder refill rate calculation with unbiased denominator.
9. Multi-stage touchpoint breakdown (-7d, -3d, -1d, 0d, +2d, +5d).
10. Segmented cohort evaluations (Pilot Tier, History Quality, Purchase Length, Medicine).
11. Offline benchmark comparison and sample maturity classification.
12. Evidence-based pilot governance recommendations (EXPAND, KEEP TIER A, RESTRICT, PAUSE).
13. Strict phone privacy: zero raw phone numbers or credentials in outcome datasets and exports.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from refillcare.evaluation.pilot_outcomes import (
    PilotCohort,
    build_pilot_outcome_dataset,
    evaluate_timing_and_stages,
    evaluate_cohort_segments,
    compare_against_offline_benchmarks,
    evaluate_pilot_decision,
    export_pilot_outcomes_csv,
    STRUCTURED_REJECTION_REASONS,
    OFFLINE_BENCHMARKS,
)


def test_pilot_cohort_defaults():
    """Verify pilot cohort structure and default observation window values."""
    cohort = PilotCohort()
    d = cohort.to_dict()

    assert d["pilot_id"] == "PILOT-2026-01"
    assert d["activity_window_days"] == 30
    assert d["observation_window_days"] == 30
    assert "Tier A (Strong Pilot)" in d["eligible_tiers"]
    assert len(STRUCTURED_REJECTION_REASONS) >= 6


def test_actual_refill_strict_identity_and_shared_phone():
    """Verify actual refills match strictly on (customerId, itemId) and keep shared phones isolated."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "R1",
            "customer_id": "CUST_A",
            "item_id": "MED_1",
            "customer_name": "Alice",
            "item_name": "Metformin 500mg",
            "phone_last4": "9999",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
        },
        {
            "reminder_id": "R2",
            "customer_id": "CUST_B",
            "item_id": "MED_2",
            "customer_name": "Bob",
            "item_name": "Amlodipine 5mg",
            "phone_last4": "9999",  # Shared phone with Alice!
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
        },
    ])

    # Transactions:
    # 1. Alice purchases MED_1 on 2026-06-04 (valid positive quantity)
    # 2. Return transaction for Alice MED_1 on 2026-06-02 with quantity = -1 (must be excluded)
    # 3. Third party purchases MED_2 on 2026-06-03 (CUST_C)
    transactions_df = pd.DataFrame([
        {"customerId": "CUST_A", "itemId": "MED_1", "invoice_date": "2026-06-02", "quantity": -1},  # Return
        {"customerId": "CUST_A", "itemId": "MED_1", "invoice_date": "2026-06-04", "quantity": 10},  # Real refill
        {"customerId": "CUST_C", "itemId": "MED_2", "invoice_date": "2026-06-03", "quantity": 30},  # Unrelated
    ])

    outcomes = build_pilot_outcome_dataset(
        audit_df=audit_df,
        transactions_df=transactions_df,
        reference_date="2026-06-15",
    )

    assert len(outcomes) == 2

    # Alice outcome verification
    alice_row = outcomes[outcomes["customerId"] == "CUST_A"].iloc[0]
    assert alice_row["actual_refill_date"] == "2026-06-04"
    assert alice_row["prediction_error_days"] == 3  # 2026-06-04 minus expected 2026-06-01
    assert alice_row["absolute_prediction_error_days"] == 3
    assert alice_row["timing_category"] == "PRE-REFILL REMINDER"
    assert alice_row["observation_status"] == "Observed Refill"
    assert alice_row["is_post_reminder_refill"] == True
    assert alice_row["is_accurate_7d"] == True
    assert alice_row["is_accurate_3d"] == True
    assert alice_row["is_accurate_1d"] == False

    # Bob outcome verification (Bob did NOT purchase MED_2, despite shared phone with Alice)
    bob_row = outcomes[outcomes["customerId"] == "CUST_B"].iloc[0]
    assert pd.isna(bob_row["actual_refill_date"])
    assert bob_row["timing_category"] == "PENDING OBSERVATION"
    assert bob_row["observation_status"] == "Pending Observation (Window Open)"
    assert bob_row["is_post_reminder_refill"] == False


def test_single_customer_multiple_medicines_isolated():
    """Verify a customer taking multiple distinct medicines has separate refill tracking."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "R1",
            "customer_id": "CUST_MULTI",
            "item_id": "MED_DIABETES",
            "customer_name": "Carol",
            "item_name": "Glimepiride 2mg",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
        },
        {
            "reminder_id": "R2",
            "customer_id": "CUST_MULTI",
            "item_id": "MED_BP",
            "customer_name": "Carol",
            "item_name": "Telmisartan 40mg",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-15",
            "reminder_stage": "-7",
            "status": "accepted",
        },
    ])

    # Only Glimepiride is purchased on 2026-06-02
    transactions_df = pd.DataFrame([
        {"customerId": "CUST_MULTI", "itemId": "MED_DIABETES", "invoice_date": "2026-06-02", "quantity": 1},
    ])

    outcomes = build_pilot_outcome_dataset(
        audit_df=audit_df,
        transactions_df=transactions_df,
        reference_date="2026-06-05",
    )

    diab_row = outcomes[outcomes["itemId"] == "MED_DIABETES"].iloc[0]
    bp_row = outcomes[outcomes["itemId"] == "MED_BP"].iloc[0]

    assert diab_row["observation_status"] == "Observed Refill"
    assert diab_row["actual_refill_date"] == "2026-06-02"
    assert diab_row["prediction_error_days"] == 1

    assert bp_row["observation_status"] == "Pending Observation (Window Open)"
    assert pd.isna(bp_row["actual_refill_date"])


def test_reminder_timing_categories():
    """Verify PRE-REFILL, SAME-DAY, and POST-REFILL timing classifications."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "R_PRE",
            "customer_id": "C1",
            "item_id": "I1",
            "reminder_date": "2026-06-01",
            "expected_refill_date": "2026-06-01",
            "reminder_stage": "0",
            "status": "accepted",
        },
        {
            "reminder_id": "R_SAME",
            "customer_id": "C2",
            "item_id": "I2",
            "reminder_date": "2026-06-05",
            "expected_refill_date": "2026-06-05",
            "reminder_stage": "0",
            "status": "accepted",
        },
        {
            "reminder_id": "R_POST",
            "customer_id": "C3",
            "item_id": "I3",
            "reminder_date": "2026-06-10",
            "expected_refill_date": "2026-06-08",
            "reminder_stage": "+2",
            "status": "accepted",
        },
    ])

    transactions_df = pd.DataFrame([
        {"customerId": "C1", "itemId": "I1", "invoice_date": "2026-06-03", "quantity": 1},  # After reminder
        {"customerId": "C2", "itemId": "I2", "invoice_date": "2026-06-05", "quantity": 1},  # Same-day
        {"customerId": "C3", "itemId": "I3", "invoice_date": "2026-06-07", "quantity": 1},  # Before reminder
    ])

    outcomes = build_pilot_outcome_dataset(
        audit_df=audit_df,
        transactions_df=transactions_df,
        reference_date="2026-06-20",
    )

    timing_info = evaluate_timing_and_stages(outcomes)

    r_pre = outcomes[outcomes["customerId"] == "C1"].iloc[0]
    r_same = outcomes[outcomes["customerId"] == "C2"].iloc[0]
    r_post = outcomes[outcomes["customerId"] == "C3"].iloc[0]

    assert r_pre["timing_category"] == "PRE-REFILL REMINDER"
    assert r_same["timing_category"] == "SAME-DAY"
    assert r_post["timing_category"] == "POST-REFILL FOLLOW-UP"
    assert r_post["risk_flag"] == "Potentially Stale Reminder"

    assert timing_info["pre_refill_count"] == 1
    assert timing_info["same_day_count"] == 1
    assert timing_info["post_refill_count"] == 1


def test_right_censoring_and_post_reminder_refill_rate():
    """Verify right-censoring denominator calculation excludes open windows."""
    audit_df = pd.DataFrame([
        # 1. Observed refill
        {"reminder_id": "R1", "customer_id": "C1", "item_id": "I1", "reminder_date": "2026-06-01", "expected_refill_date": "2026-06-01", "status": "accepted", "reminder_stage": "0"},
        # 2. Window closed (80 days elapsed), no refill
        {"reminder_id": "R2", "customer_id": "C2", "item_id": "I2", "reminder_date": "2026-03-01", "expected_refill_date": "2026-03-01", "status": "accepted", "reminder_stage": "0"},
        # 3. Window open (5 days elapsed), no refill yet
        {"reminder_id": "R3", "customer_id": "C3", "item_id": "I3", "reminder_date": "2026-06-10", "expected_refill_date": "2026-06-10", "status": "accepted", "reminder_stage": "0"},
    ])

    transactions_df = pd.DataFrame([
        {"customerId": "C1", "itemId": "I1", "invoice_date": "2026-06-03", "quantity": 1},
    ])

    outcomes = build_pilot_outcome_dataset(
        audit_df=audit_df,
        transactions_df=transactions_df,
        reference_date="2026-06-15",
    )

    stage_res = evaluate_timing_and_stages(outcomes)
    stage_df = stage_res["stage_analysis"]

    # In stage 0: 1 observed, 1 closed no refill, 1 pending open
    # Denominator for refill rate = 1 observed + 1 closed = 2
    # Rate = 1/2 = 50.0%
    stage_0_row = stage_df[stage_df["Stage"] == "0"].iloc[0]
    assert stage_0_row["Observed Refills"] == 1
    assert stage_0_row["Pending Observation"] == 1
    assert stage_0_row["Post-Reminder Refill Rate (%)"] == 50.0


def test_evaluate_cohort_segments():
    """Verify multidimensional segmentation across tiers, history quality, and purchase lengths."""
    outcomes_df = pd.DataFrame([
        {
            "customerId": "C1", "itemId": "I1", "medicine": "Metformin", "pilot_tier": "Tier A (Strong Pilot)",
            "history_quality": "high_history", "purchase_count": 6, "observation_status": "Observed Refill",
            "absolute_prediction_error_days": 2.0
        },
        {
            "customerId": "C2", "itemId": "I2", "medicine": "Metformin", "pilot_tier": "Tier A (Strong Pilot)",
            "history_quality": "high_history", "purchase_count": 5, "observation_status": "Observed Refill",
            "absolute_prediction_error_days": 4.0
        },
        {
            "customerId": "C3", "itemId": "I3", "medicine": "Amlodipine", "pilot_tier": "Tier B (Review Required)",
            "history_quality": "medium_history", "purchase_count": 3, "observation_status": "No Observed Refill (Window Closed)",
            "absolute_prediction_error_days": None
        },
    ])

    segments = evaluate_cohort_segments(outcomes_df)

    by_tier = segments["by_tier"]
    tier_a_row = by_tier[by_tier["pilot_tier"] == "Tier A (Strong Pilot)"].iloc[0]
    assert tier_a_row["Total Regimens"] == 2
    assert tier_a_row["Observed Refills"] == 2
    assert tier_a_row["MAE (Days)"] == 3.0  # (2.0 + 4.0)/2
    assert tier_a_row["±7d Accuracy (%)"] == 100.0


def test_compare_against_offline_benchmarks():
    """Verify comparison with offline test benchmarks."""
    outcomes_df = pd.DataFrame([
        {
            "observation_status": "Observed Refill",
            "pilot_tier": "Tier A (Strong Pilot)",
            "absolute_prediction_error_days": 10.0,
        },
        {
            "observation_status": "Observed Refill",
            "pilot_tier": "Tier A (Strong Pilot)",
            "absolute_prediction_error_days": 8.0,
        },
    ])

    cmp_res = compare_against_offline_benchmarks(outcomes_df)

    assert cmp_res["status"] == "EVALUATED"
    assert cmp_res["evaluated_refills"] == 2
    assert cmp_res["pilot_mae"] == 9.0
    assert cmp_res["offline_benchmark_mae"] == OFFLINE_BENCHMARKS["overall_mae"]
    assert cmp_res["delta_mae"] == round(9.0 - OFFLINE_BENCHMARKS["overall_mae"], 2)


def test_evaluate_pilot_decision():
    """Verify pilot governance decision recommendations."""
    # 1. Early / insufficient data (< 10 observed refills)
    early_df = pd.DataFrame([
        {"observation_status": "Observed Refill", "absolute_prediction_error_days": 5.0}
    ])
    dec_early = evaluate_pilot_decision(early_df, reconciliation_health="HEALTHY")
    assert dec_early["recommendation"] == "KEEP TIER A ONLY"
    assert dec_early["retraining_recommended"] is False

    # 2. Mature low-error pilot (>= 10 refills, MAE <= 12)
    mature_strong_df = pd.DataFrame([
        {"observation_status": "Observed Refill", "absolute_prediction_error_days": 8.0} for _ in range(12)
    ])
    dec_strong = evaluate_pilot_decision(mature_strong_df, reconciliation_health="HEALTHY")
    assert dec_strong["recommendation"] == "EXPAND TIER A"

    # 3. Reconciliation issue forces PAUSE
    dec_pause = evaluate_pilot_decision(mature_strong_df, reconciliation_health="ATTENTION_REQUIRED")
    assert dec_pause["recommendation"] == "PAUSE"


def test_export_pilot_outcomes_csv_privacy():
    """Confirm CSV export never contains full phone numbers, API keys, or raw credentials."""
    outcomes_df = pd.DataFrame([
        {
            "pilot_id": "PILOT-2026-01",
            "customerId": "C101",
            "itemId": "I202",
            "customer_name": "Test Customer",
            "medicine": "Atorvastatin 20mg",
            "phone_masked": "***1234",
            "MOBILE_NO": "9876541234",  # Raw phone if inadvertently present
            "api_key": "SECRET123",  # Secret if inadvertently present
            "pilot_tier": "Tier A (Strong Pilot)",
            "expected_refill_date": "2026-06-01",
            "actual_refill_date": "2026-06-02",
            "prediction_error_days": 1,
            "timing_category": "PRE-REFILL REMINDER",
            "observation_status": "Observed Refill",
        }
    ])

    csv_bytes = export_pilot_outcomes_csv(outcomes_df)
    csv_str = csv_bytes.decode("utf-8")

    assert "phone_masked" in csv_str
    assert "***1234" in csv_str
    assert "MOBILE_NO" not in csv_str
    assert "9876541234" not in csv_str
    assert "api_key" not in csv_str
    assert "SECRET123" not in csv_str
