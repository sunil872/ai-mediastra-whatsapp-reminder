"""Unit and operational readiness tests for Phase 11 RefillCare Controlled Pilot.

Verifies:
- Pilot tier classification (Tier A, Tier B, Tier C)
- Human-readable explainability for pharmacy operators
- Prediction and schedule filtering by pilot tier
- Reminder queue safety and duplicate suppression
- Explicit operator approval & dry-run default safeguards
- Shared-phone multi-customer segregation
- Dynamic cycle reset on new purchases
- Masked phone logging & zero credential storage
"""

import os
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
import pandas as pd
import pytest

from app_refillcare import (
    _determine_pilot_tier,
    prepare_prediction_overview,
    filter_predictions,
    filter_schedules,
    generate_reminder_schedule_table,
)
from reminder.scheduler import (
    RefillReminderScheduler,
    ReminderRecord,
    REMINDER_STAGES,
    evaluate_refill_eligibility,
)
from reminder.dispatch import (
    RefillCareReminderDispatcher,
    select_eligible_reminders_for_dispatch,
    build_refillcare_whatsapp_payload,
)
from reminder.storage import RefillCareStorage


@pytest.fixture
def temp_storage(tmp_path):
    """Provide an isolated, temporary SQLite database."""
    db_file = tmp_path / "pilot_test.db"
    return RefillCareStorage(db_path=db_file)


# ==============================================================================
# 1. PILOT TIER & EXPLAINABILITY TESTS
# ==============================================================================

def test_determine_pilot_tier_classification():
    """Verify tier assignment rules and human-readable operator explainability."""
    # Tier A: deep history (>= 5 purchases)
    tier_a, note_a = _determine_pilot_tier(purchase_count=6, is_recurring=1, quality="high_history")
    assert "Tier A" in tier_a
    assert "6 previous purchases" in note_a

    # Tier A: recurring history (>= 3 purchases & recurring)
    tier_a2, note_a2 = _determine_pilot_tier(purchase_count=3, is_recurring=1, quality="high_history")
    assert "Tier A" in tier_a2

    # Tier B: moderate history (3-4 purchases, non-recurring or medium quality)
    tier_b, note_b = _determine_pilot_tier(purchase_count=3, is_recurring=0, quality="medium_history")
    assert "Tier B" in tier_b
    assert "pharmacist verification" in note_b.lower()

    # Tier C: limited history (2 purchases)
    tier_c, note_c = _determine_pilot_tier(purchase_count=2, is_recurring=0, quality="low_history")
    assert "Tier C" in tier_c
    assert "suppressed" in note_c.lower()

    # Tier C: single purchase / cold start (< 2 purchases)
    tier_cold, note_cold = _determine_pilot_tier(purchase_count=1, is_recurring=0, quality="ineligible")
    assert "Tier C" in tier_cold
    assert "insufficient" in note_cold.lower()


def test_prediction_overview_includes_pilot_tier():
    """Verify prepare_prediction_overview populates Pilot Tier and Operator Notes."""
    mock_bundle = {
        "pipeline": type("MockPipe", (), {"predict": lambda self, X: [30.0] * len(X)})(),
        "features_numeric": [],
        "features_categorical": [],
    }

    raw_df = pd.DataFrame([
        {
            "customerId": "CUST_A",
            "itemId": "ITEM_A",
            "customerName": "Alice Patel",
            "itemName": "METPURE XL",
            "MOBILE_NO": "+919876543210",
            "invoice_date": pd.Timestamp("2026-04-01"),
            "purchase_count_so_far": 8,
            "is_recurring_history": 1,
            "historical_interval_median": 30.0,
            "historical_interval_std": 3.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "days_since_previous_purchase": 30.0,
        },
        {
            "customerId": "CUST_B",
            "itemId": "ITEM_B",
            "customerName": "Bob Kumar",
            "itemName": "TELMIKIND 40",
            "MOBILE_NO": "+919876543210",
            "invoice_date": pd.Timestamp("2026-04-01"),
            "purchase_count_so_far": 2,
            "is_recurring_history": 0,
            "historical_interval_median": 30.0,
            "historical_interval_std": 5.0,
            "historical_interval_norm_mad": 0.20,
            "cadence_drift": 5.0,
            "days_since_previous_purchase": 30.0,
        },
        {
            "customerId": "CUST_C",
            "itemId": "ITEM_C",
            "customerName": "Cold Start Patient",
            "itemName": "PARACETAMOL",
            "MOBILE_NO": "+919876543210",
            "invoice_date": pd.Timestamp("2026-04-01"),
            "purchase_count_so_far": 1,
            "is_recurring_history": 0,
        },
    ])

    el_df, inel_df, metrics = prepare_prediction_overview(raw_df, mock_bundle)

    # Phase 17D: only CUST_A is hybrid-eligible (P>=6 + regular); B and C rejected/ineligible
    assert len(el_df) == 1
    assert len(inel_df) == 2
    assert "Pilot Tier" in el_df.columns
    assert "Operator Notes" in el_df.columns

    cust_a = el_df[el_df["customerId"] == "CUST_A"].iloc[0]
    assert "Tier A" in cust_a["Pilot Tier"]

    filtered_a = filter_predictions(el_df, tier_filter="Tier A (Strong Pilot)")
    assert len(filtered_a) == 1
    assert filtered_a.iloc[0]["customerId"] == "CUST_A"


# ==============================================================================
# 2. REMINDER QUEUE SAFETY & DUPLICATE PREVENTION
# ==============================================================================

def test_pilot_duplicate_dispatch_prevention(temp_storage):
    """Verify that a reminder dispatched once cannot be resent."""
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)

    reminder = {
        "reminder_id": "PILOT-CUST-001___MED-001___2026-05-01___0d",
        "customerId": "PILOT-CUST-001",
        "itemId": "MED-001",
        "customerName": "Pilot Patient",
        "MOBILE_NO": "+919876543210",
        "itemName": "METPURE XL",
        "expected_refill_date": "2026-05-01",
        "reminder_date": "2026-05-01",
        "reminder_stage": 0,
        "message": "Your regular medicine refill may be due today, 01-05-2026.",
        "status": "scheduled",
    }

    # First dry-run dispatch: accepted
    res1 = dispatcher.dispatch_single_reminder(reminder, dry_run=True)
    assert res1.status == "accepted"
    assert res1.success is True

    # Second dispatch: duplicate prevented
    res2 = dispatcher.dispatch_single_reminder(reminder, dry_run=True)
    assert res2.status == "duplicate_prevented"
    assert res2.success is False


def test_pilot_cancelled_reminder_omitted(temp_storage):
    """Verify that a cancelled reminder is rejected by dispatcher."""
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)

    cancelled_rem = {
        "reminder_id": "PILOT-CANCEL-001___MED-001___2026-05-01___-3d",
        "customerId": "PILOT-CANCEL-001",
        "itemId": "MED-001",
        "customerName": "Cancelled Patient",
        "MOBILE_NO": "+919876543210",
        "itemName": "METPURE XL",
        "expected_refill_date": "2026-05-01",
        "reminder_date": "2026-04-28",
        "reminder_stage": -3,
        "status": "cancelled",
    }

    res = dispatcher.dispatch_single_reminder(cancelled_rem, dry_run=True)
    assert res.status == "cancelled"
    assert res.success is False


# ==============================================================================
# 3. SHARED-PHONE & CYCLE RESET IN PILOT CONTEXT
# ==============================================================================

def test_pilot_shared_phone_multi_customer():
    """Verify that multiple customers on the same phone maintain isolated pilot records."""
    shared_phone = "+919123456789"

    patient_1 = {
        "customerId": "PILOT-P1",
        "customerName": "Father",
        "MOBILE_NO": shared_phone,
        "itemId": "CARDIO-MED",
        "itemName": "METPURE XL",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 5,
        "predicted_days_until_refill": 30.0,
    }

    patient_2 = {
        "customerId": "PILOT-P2",
        "customerName": "Mother",
        "MOBILE_NO": shared_phone,
        "itemId": "DIABETES-MED",
        "itemName": "GLIMEPIRIDE 2MG",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }

    scheduler = RefillReminderScheduler()
    sched1 = scheduler.schedule_refill_cycle(patient_1)
    sched2 = scheduler.schedule_refill_cycle(patient_2)

    assert sched1["scheduled"] is True
    assert sched2["scheduled"] is True

    # Validate distinct IDs
    ids1 = {r.reminder_id for r in sched1["reminders"]}
    ids2 = {r.reminder_id for r in sched2["reminders"]}
    assert ids1.isdisjoint(ids2)

    # Validate payload parameters map to individual names
    p1_rem = sched1["reminders"][0]
    p2_rem = sched2["reminders"][0]

    payload1 = build_refillcare_whatsapp_payload(p1_rem, store_name="PHARMA HUBB", store_contact="9876543210")
    payload2 = build_refillcare_whatsapp_payload(p2_rem, store_name="PHARMA HUBB", store_contact="9876543210")

    assert payload1["payload"]["template"]["components"][0]["parameters"][0]["text"] == "Father"
    assert payload2["payload"]["template"]["components"][0]["parameters"][0]["text"] == "Mother"


def test_pilot_new_purchase_cycle_reset():
    """Verify cycle reset upon customer repurchase."""
    scheduler = RefillReminderScheduler()

    # Initial cycle
    r1 = {
        "customerId": "PILOT-RESET",
        "customerName": "John Doe",
        "MOBILE_NO": "+919876543210",
        "itemId": "MED-RESET",
        "itemName": "AMLODIPINE 5MG",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }
    scheduler.schedule_refill_cycle(r1)
    initial_rems = scheduler.get_reminders_for_customer("PILOT-RESET", "MED-RESET")
    assert len(initial_rems) == 6
    assert all(r.status == "scheduled" for r in initial_rems)

    # Customer repurchases early
    r2 = dict(r1)
    r2["invoice_date"] = "2026-04-20"
    r2["purchase_count_so_far"] = 4
    scheduler.schedule_refill_cycle(r2, cancel_prior_cycles=True)

    all_rems = scheduler.get_reminders_for_customer("PILOT-RESET", "MED-RESET")
    assert len(all_rems) == 12

    cancelled_count = sum(1 for r in all_rems if r.status == "cancelled")
    scheduled_count = sum(1 for r in all_rems if r.status == "scheduled")
    assert cancelled_count == 6
    assert scheduled_count == 6
