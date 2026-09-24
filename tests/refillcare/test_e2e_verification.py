"""RefillCare Phase 9 — End-to-End System Verification Tests.

Verifies the complete existing RefillCare flow:
Purchase History -> Feature Engineering -> Model Prediction -> Expected Refill Date
-> Reminder Scheduler -> WhatsApp Payload Generation -> Dry-Run Dispatch -> SQLite Audit.

Covers all 9 required verification scenarios with isolated synthetic fixtures and mocks:
- Scenario 1: Recurring Customer End-to-End Flow
- Scenario 2: Cold Start Customer Exclusion
- Scenario 3: Shared Phone Number Isolation
- Scenario 4: New Purchase Cycle Reset
- Scenario 5: WhatsApp Template Payload Specification
- Scenario 6: Dry-Run Dispatch Safety & Audit
- Scenario 7: Duplicate Dispatch Prevention
- Scenario 8: Failed Provider Response Handling & Retryability
- Scenario 9: SQLite Database Persistence Across Restarts
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from refillcare.data.history import create_purchase_history
from refillcare.features.engineering import build_feature_dataset, TARGET_COLUMN
from refillcare.models.prediction import predict_refill_date, generate_batch_predictions
from refillcare.models.training import (
    build_model_pipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from refillcare.config.whatsapp_template import (
    WHATSAPP_TEMPLATE_NAME,
    WHATSAPP_TEMPLATE_LANGUAGE,
    WHATSAPP_TEMPLATE_VARIABLES,
    WHATSAPP_TEMPLATE_BODY,
    build_template_components,
)
from reminder.scheduler import (
    RefillReminderScheduler,
    ReminderRecord,
    REMINDER_STAGES,
    calculate_expected_refill_date,
    evaluate_refill_eligibility,
    format_reminder_message,
)
from reminder.dispatch import (
    RefillCareReminderDispatcher,
    DispatchResult,
    build_refillcare_whatsapp_payload,
    select_eligible_reminders_for_dispatch,
    mask_phone_for_ui,
    sanitize_template_text,
)
from reminder.storage import (
    RefillCareStorage,
    _extract_phone_last4,
)


@pytest.fixture
def trained_synthetic_model_bundle():
    """Build a trained synthetic model bundle for deterministic end-to-end testing."""
    np.random.seed(42)
    n = 100
    df_train = pd.DataFrame({
        "customerId": [f"CUST_{i % 5}" for i in range(n)],
        "itemId": [f"ITEM_{i % 3}" for i in range(n)],
        "itemCode": [f"CODE_{i % 3}" for i in range(n)],
        "itemName": ["METPURE XL" if i % 2 == 0 else "TELMIKIND" for i in range(n)],
        "therapeuticCategory": ["Cardiology" for _ in range(n)],
        "generic_name": ["Metoprolol" for _ in range(n)],
        "salt_composition": ["Metoprolol Succinate" for _ in range(n)],
        "salt_category": ["Anti-hypertensive" for _ in range(n)],
        "salt_itemcat": ["Cardio" for _ in range(n)],
        "salt_pack": ["10 Tablets" for _ in range(n)],
        "invoice_date": [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i * 3) for i in range(n)],
        "purchase_count_so_far": np.random.randint(1, 10, size=n),
        "days_since_first_purchase": np.random.randint(0, 300, size=n),
        "days_since_previous_purchase": np.random.uniform(25, 35, size=n),
        "last_purchase_interval": np.random.uniform(25, 35, size=n),
        "historical_interval_median": np.random.uniform(28, 32, size=n),
        "historical_interval_mean": np.random.uniform(28, 32, size=n),
        "historical_interval_std": np.random.uniform(1, 5, size=n),
        "historical_interval_min": np.random.uniform(20, 28, size=n),
        "historical_interval_max": np.random.uniform(32, 45, size=n),
        "historical_interval_cv": np.random.uniform(0.05, 0.2, size=n),
        "quantity": np.random.randint(10, 60, size=n),
        "freeQuantity": np.zeros(n),
        "avg_historical_quantity": np.random.uniform(10, 60, size=n),
        "quantity_vs_avg_ratio": np.ones(n),
        "netAmount": np.random.uniform(100, 500, size=n),
        "gstAmount": np.random.uniform(5, 50, size=n),
        "rate": np.random.uniform(10, 50, size=n),
        "mrp": np.random.uniform(15, 60, size=n),
        "discountPercent": np.zeros(n),
        "purchase_month": np.random.randint(1, 13, size=n),
        "purchase_day_of_week": np.random.randint(0, 7, size=n),
        "purchase_day_of_month": np.random.randint(1, 29, size=n),
        "purchase_day_of_year": np.random.randint(1, 365, size=n),
        "purchase_quarter": np.random.randint(1, 5, size=n),
        "is_weekend": np.zeros(n),
        "is_first_purchase": np.zeros(n),
        "has_multiple_prior_purchases": np.ones(n),
        "is_recurring_history": np.ones(n),
        TARGET_COLUMN: np.random.uniform(28, 32, size=n),
    })

    regressor = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    pipe = build_model_pipeline(regressor)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipe.fit(df_train[feature_cols], df_train[TARGET_COLUMN].values)

    return {
        "pipeline": pipe,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
    }


@pytest.fixture
def temp_storage(tmp_path):
    """Provide an isolated, temporary SQLite storage instance."""
    db_file = tmp_path / "test_refillcare_e2e.db"
    return RefillCareStorage(db_path=db_file)


# ==============================================================================
# TEST SCENARIO 1 — RECURRING CUSTOMER END-TO-END FLOW
# ==============================================================================

def test_scenario_1_recurring_customer_end_to_end(trained_synthetic_model_bundle, temp_storage):
    """Verify complete flow for a qualifying recurring customer from history to scheduled reminders."""
    # 1. Isolated synthetic test fixture — Phase 17D requires P>=6 + regular cadence
    customer_id = "TEST-CUSTOMER-001"
    customer_name = "Sunil"
    item_id = "TEST-ITEM-001"
    item_name = "METPURE XL"
    test_phone = "+919876543210"

    base_event = {
        "customerId": customer_id,
        "customerName": customer_name,
        "MOBILE_NO": test_phone,
        "itemId": item_id,
        "itemName": item_name,
        "itemCode": "MET-XL-50",
        "quantity": 30,
        "freeQuantity": 0,
        "netAmount": 300.0,
        "gstAmount": 15.0,
        "rate": 10.0,
        "mrp": 12.0,
        "discountPercent": 0.0,
        "therapeuticCategory": "Cardiology",
        "generic_name": "Metoprolol",
        "salt_composition": "Metoprolol Succinate",
        "salt_category": "Anti-hypertensive",
        "salt_itemcat": "Cardio",
        "salt_pack": "10 Tablets",
    }
    # Six visits ~30 days apart (eligible under hybrid_routing_v17d)
    purchase_dates = [
        "2025-11-01",
        "2025-12-01",
        "2025-12-31",
        "2026-01-30",
        "2026-03-01",
        "2026-03-31",
    ]
    raw_events = pd.DataFrame([
        {
            **base_event,
            "invoice_number": f"INV-{i+1:03d}",
            "invoice_date": pd.Timestamp(d),
        }
        for i, d in enumerate(purchase_dates)
    ])

    # Step 1: Verify history creation
    history_df = create_purchase_history(raw_events)
    assert len(history_df) == 6
    assert history_df["purchase_seq"].tolist() == [1, 2, 3, 4, 5, 6]
    assert pd.isna(history_df["days_since_previous_purchase"].iloc[0])
    assert history_df["days_since_previous_purchase"].iloc[1] == 30

    # Step 2: Verify feature engineering (leakage-safe historical info only)
    feature_df = build_feature_dataset(history_df)
    latest_event = feature_df.iloc[-1].to_dict()
    assert latest_event["purchase_count_so_far"] == 6
    assert latest_event["is_recurring_history"] == 1
    assert latest_event["has_multiple_prior_purchases"] == 1
    assert latest_event["historical_interval_norm_mad"] <= 0.50
    assert latest_event["cadence_drift"] <= 10.0
    assert pd.isna(latest_event[TARGET_COLUMN])  # Final purchase has no future target

    # Step 3: Verify Phase 17D hybrid prediction (core regular -> personal median)
    pred_res = predict_refill_date(trained_synthetic_model_bundle, latest_event)
    assert pred_res["customerId"] == customer_id
    assert pred_res["itemId"] == item_id
    assert pred_res["prediction_status"] == "eligible"
    assert pred_res["refill_confidence"] == "HIGH"
    pred_days = pred_res["predicted_days_until_refill"]
    assert pred_days > 0  # Step 4: valid predicted interval
    assert pred_res["expected_refill_date"] is not None

    # Step 5: Verify expected refill date calculation
    calc_res = calculate_expected_refill_date("2026-03-31", pred_days)
    assert calc_res["is_valid"] is True
    exp_date = calc_res["expected_refill_date"]
    assert exp_date == date(2026, 3, 31) + timedelta(days=round(pred_days))

    # Step 6: Verify reminder scheduler produces exactly 6 stages
    scheduler = RefillReminderScheduler()
    sched_record = {
        "customerId": customer_id,
        "customerName": customer_name,
        "MOBILE_NO": test_phone,
        "itemId": item_id,
        "itemName": item_name,
        "invoice_date": "2026-03-31",
        "purchase_count_so_far": 6,
        "is_recurring_history": 1,
        "predicted_days_until_refill": pred_days,
        "prediction_status": pred_res["prediction_status"],
        "refill_confidence": pred_res["refill_confidence"],
    }
    sched_res = scheduler.schedule_refill_cycle(sched_record)
    assert sched_res["scheduled"] is True
    assert sched_res["reminder_count"] == 6

    # Step 7: Verify deterministic reminder IDs and exact stages
    stages_found = [r.reminder_stage for r in sched_res["reminders"]]
    assert stages_found == [-7, -3, -1, 0, 2, 5]

    for rem in sched_res["reminders"]:
        # Step 8: Each reminder contains correct customerId + itemId
        assert rem.customerId == customer_id
        assert rem.itemId == item_id
        assert rem.customerName == customer_name
        assert rem.MOBILE_NO == test_phone
        expected_id = f"{customer_id}___{item_id}___{exp_date.isoformat()}___{rem.reminder_stage:+d}d"
        assert rem.reminder_id == expected_id

    # Step 9: Verify no reminder is created for an invalid prediction
    invalid_record = dict(sched_record)
    invalid_record["predicted_days_until_refill"] = -10.0
    invalid_res = scheduler.schedule_refill_cycle(invalid_record)
    assert invalid_res["scheduled"] is False
    assert len(invalid_res["reminders"]) == 0


# ==============================================================================
# TEST SCENARIO 2 — COLD START CUSTOMER
# ==============================================================================

def test_scenario_2_cold_start_customer(temp_storage):
    """Verify that single-purchase cold-start customers are classified ineligible with zero dispatches."""
    cold_start_record = {
        "customerId": "TEST-COLD-001",
        "customerName": "First Time Patient",
        "MOBILE_NO": "+919876543210",
        "itemId": "TEST-ITEM-COLD",
        "itemName": "PARACETAMOL 650",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 1,
        "predicted_days_until_refill": 30.0,
    }

    # Verify eligibility classification
    eligibility = evaluate_refill_eligibility(cold_start_record)
    assert eligibility["is_eligible"] is False
    assert eligibility["status"] == "ineligible"
    assert "Cold-start" in eligibility["reason"]
    assert eligibility["history_quality"] == "ineligible"

    # Verify scheduler produces no reminders
    scheduler = RefillReminderScheduler()
    sched_res = scheduler.schedule_refill_cycle(cold_start_record)
    assert sched_res["scheduled"] is False
    assert len(sched_res["reminders"]) == 0

    # Verify dispatcher filter skips cold start records
    eligible_list = select_eligible_reminders_for_dispatch(sched_res["reminders"])
    assert len(eligible_list) == 0

    # Verify no records in SQLite audit
    df_audit = temp_storage.get_audit_dataframe()
    assert len(df_audit) == 0


# ==============================================================================
# TEST SCENARIO 3 — SHARED PHONE NUMBER ISOLATION
# ==============================================================================

def test_scenario_3_shared_phone_number():
    """Verify that multiple customers sharing the same phone number maintain separate identities."""
    shared_phone = "+919988776655"

    cust_a_record = {
        "customerId": "TEST-CUSTOMER-A",
        "customerName": "Alice Patel",
        "MOBILE_NO": shared_phone,
        "itemId": "TEST-ITEM-CARDIO",
        "itemName": "METPURE XL",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }

    cust_b_record = {
        "customerId": "TEST-CUSTOMER-B",
        "customerName": "Bob Patel",
        "MOBILE_NO": shared_phone,
        "itemId": "TEST-ITEM-DIABETES",
        "itemName": "GLIMEPIRIDE 2MG",
        "invoice_date": "2026-04-01",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }

    scheduler = RefillReminderScheduler()
    sched_a = scheduler.schedule_refill_cycle(cust_a_record)
    sched_b = scheduler.schedule_refill_cycle(cust_b_record)

    assert sched_a["scheduled"] is True
    assert sched_b["scheduled"] is True

    reminders_a = sched_a["reminders"]
    reminders_b = sched_b["reminders"]

    # Reminders must be strictly disjoint and keyed by (customerId, itemId)
    ids_a = {r.reminder_id for r in reminders_a}
    ids_b = {r.reminder_id for r in reminders_b}
    assert ids_a.isdisjoint(ids_b)

    for r in reminders_a:
        assert r.customerId == "TEST-CUSTOMER-A"
        assert r.itemId == "TEST-ITEM-CARDIO"
        assert "TEST-CUSTOMER-A" in r.reminder_id

    for r in reminders_b:
        assert r.customerId == "TEST-CUSTOMER-B"
        assert r.itemId == "TEST-ITEM-DIABETES"
        assert "TEST-CUSTOMER-B" in r.reminder_id


# ==============================================================================
# TEST SCENARIO 4 — NEW PURCHASE RESETS CYCLE
# ==============================================================================

def test_scenario_4_new_purchase_resets_cycle():
    """Verify that a new purchase cancels pending old-cycle reminders and starts a new independent cycle."""
    scheduler = RefillReminderScheduler()

    # Initial Cycle: purchase on 2026-03-01 with 30-day predicted interval (expected refill: 2026-03-31)
    cycle1_record = {
        "customerId": "TEST-RESET-001",
        "customerName": "Rajesh Sharma",
        "MOBILE_NO": "+919876543210",
        "itemId": "TEST-ITEM-RESET",
        "itemName": "AMLODIPINE 5MG",
        "invoice_date": "2026-03-01",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }
    sched1 = scheduler.schedule_refill_cycle(cycle1_record)
    assert sched1["scheduled"] is True
    assert sched1["expected_refill_date"] == "2026-03-31"

    # Verify all cycle 1 reminders are initially scheduled
    c1_reminders = scheduler.get_reminders_for_customer("TEST-RESET-001", "TEST-ITEM-RESET")
    assert len(c1_reminders) == 6
    assert all(r.status == "scheduled" for r in c1_reminders)
    c1_ids = [r.reminder_id for r in c1_reminders]

    # Customer makes a new purchase early on 2026-03-20 with a 30-day predicted interval (expected: 2026-04-19)
    cycle2_record = {
        "customerId": "TEST-RESET-001",
        "customerName": "Rajesh Sharma",
        "MOBILE_NO": "+919876543210",
        "itemId": "TEST-ITEM-RESET",
        "itemName": "AMLODIPINE 5MG",
        "invoice_date": "2026-03-20",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }
    sched2 = scheduler.schedule_refill_cycle(cycle2_record, cancel_prior_cycles=True)
    assert sched2["scheduled"] is True
    assert sched2["cancelled_prior_reminders"] == 6
    assert sched2["expected_refill_date"] == "2026-04-19"

    # Check that old reminders are cancelled and new reminders are scheduled
    all_reminders = scheduler.get_reminders_for_customer("TEST-RESET-001", "TEST-ITEM-RESET")
    assert len(all_reminders) == 12  # 6 cancelled from cycle 1 + 6 scheduled from cycle 2

    for r in all_reminders:
        if r.reminder_id in c1_ids:
            assert r.status == "cancelled"
            assert "2026-03-31" in r.reminder_id
        else:
            assert r.status == "scheduled"
            assert "2026-04-19" in r.reminder_id


# ==============================================================================
# TEST SCENARIO 5 — APPROVED WHATSAPP TEMPLATE PAYLOAD SPECIFICATION
# ==============================================================================

def test_scenario_5_whatsapp_payload_specification():
    """Verify that build_refillcare_whatsapp_payload generates the exact approved 6-variable schema."""
    reminder = ReminderRecord(
        reminder_id="TEST-CUSTOMER-001___TEST-ITEM-001___2026-05-02___-7d",
        customerId="TEST-CUSTOMER-001",
        itemId="TEST-ITEM-001",
        customerName="Sunil",
        MOBILE_NO="+919876543210",
        itemName="METPURE XL",
        latest_purchase_date="2026-04-02",
        predicted_interval_days=30.0,
        expected_refill_date="2026-05-02",
        reminder_stage=-7,
        reminder_date="2026-04-25",
        message="Your regular medicine refill may be due in about 7 days, around 02-05-2026.",
        status="scheduled",
    )

    build_res = build_refillcare_whatsapp_payload(
        reminder=reminder,
        store_name="PHARMA HUBB",
        store_contact="9876500000",
    )

    assert build_res["is_valid"] is True
    payload = build_res["payload"]

    # 1. Product and recipient
    assert payload["messaging_product"] == "whatsapp"
    assert payload["to"] == "919876543210"
    assert payload["type"] == "template"

    # 2. Template metadata
    template = payload["template"]
    assert template["name"] == WHATSAPP_TEMPLATE_NAME
    assert template["name"] == "refillcare_medicine_reminder"
    assert template["language"]["code"] == WHATSAPP_TEMPLATE_LANGUAGE
    assert template["language"]["code"] == "en"

    # 3. Exactly 1 BODY component with exactly 6 parameters
    components = template["components"]
    assert len(components) == 1
    body_comp = components[0]
    assert body_comp["type"] == "BODY"

    params = body_comp["parameters"]
    assert len(params) == 6
    assert len(WHATSAPP_TEMPLATE_VARIABLES) == 6

    # Parameter ordering:
    # {{1}} -> Customer Name
    assert params[0]["text"] == "Sunil"
    # {{2}} -> Store Name
    assert params[1]["text"] == "PHARMA HUBB"
    # {{3}} -> Reminder-stage message (generated by scheduler)
    assert params[2]["text"] == "Your regular medicine refill may be due in about 7 days, around 02-05-2026."
    # {{4}} -> Medicine Name
    assert params[3]["text"] == "METPURE XL"
    # {{5}} -> Store Contact
    assert params[4]["text"] == "9876500000"
    # {{6}} -> Store Name
    assert params[5]["text"] == "PHARMA HUBB"


# ==============================================================================
# TEST SCENARIO 6 — DRY-RUN DISPATCH SAFETY & AUDIT
# ==============================================================================

def test_scenario_6_dry_run_dispatch_safety_and_audit(temp_storage):
    """Verify dry-run dispatch makes zero external HTTP calls and accurately records SQLite audit state."""
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)

    reminder = {
        "reminder_id": "TEST-DRYRUN-001___ITEM-001___2026-05-01___-3d",
        "customerId": "TEST-DRYRUN-001",
        "itemId": "ITEM-001",
        "customerName": "Test Customer",
        "MOBILE_NO": "+919876543210",
        "itemName": "METPURE XL",
        "expected_refill_date": "2026-05-01",
        "reminder_date": "2026-04-28",
        "reminder_stage": -3,
        "message": "Your regular medicine refill may be due in about 3 days, around 01-05-2026.",
        "status": "scheduled",
    }

    # Pre-insert scheduled reminder
    temp_storage.insert_or_update_scheduled_reminder(reminder)

    with patch("requests.post") as mock_http_post:
        dispatch_result = dispatcher.dispatch_single_reminder(reminder, dry_run=True)

        # 1. Zero external HTTP requests
        mock_http_post.assert_not_called()

        # 2. Result properties
        assert dispatch_result.dry_run is True
        assert dispatch_result.status == "accepted"
        assert dispatch_result.success is True
        assert "[DRY RUN]" in dispatch_result.message
        assert dispatch_result.provider_message_id.startswith("dry_run_")

        # 3. SQLite audit state
        db_rec = temp_storage.get_reminder(reminder["reminder_id"])
        assert db_rec is not None
        assert db_rec["status"] == "accepted"
        assert db_rec["is_dry_run"] == 1
        assert db_rec["attempt_count"] == 1
        assert db_rec["provider_message_id"] == dispatch_result.provider_message_id
        assert db_rec["phone_last4"] == "3210"  # Full phone is NOT stored


# ==============================================================================
# TEST SCENARIO 7 — DUPLICATE DISPATCH PREVENTION
# ==============================================================================

def test_scenario_7_duplicate_dispatch_prevention(temp_storage):
    """Verify that dispatching the same reminder twice is blocked with duplicate_prevented status."""
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)

    reminder = {
        "reminder_id": "TEST-DUP-001___ITEM-DUP___2026-05-01___0d",
        "customerId": "TEST-DUP-001",
        "itemId": "ITEM-DUP",
        "customerName": "Duplicate Test User",
        "MOBILE_NO": "+919876543210",
        "itemName": "TELMIKIND 40",
        "expected_refill_date": "2026-05-01",
        "reminder_date": "2026-05-01",
        "reminder_stage": 0,
        "message": "Your regular medicine refill may be due today, 01-05-2026.",
        "status": "scheduled",
    }

    with patch("requests.post") as mock_http_post:
        # First dispatch: accepted
        res1 = dispatcher.dispatch_single_reminder(reminder, dry_run=True)
        assert res1.status == "accepted"
        assert res1.success is True

        # Second dispatch: duplicate prevented
        res2 = dispatcher.dispatch_single_reminder(reminder, dry_run=True)
        assert res2.status == "duplicate_prevented"
        assert res2.success is False
        assert res2.error_category == "Duplicate"
        assert "Duplicate Prevention" in res2.message

        mock_http_post.assert_not_called()


# ==============================================================================
# TEST SCENARIO 8 — FAILED PROVIDER RESPONSE HANDLING & RETRYABILITY
# ==============================================================================

def test_scenario_8_failed_provider_response(temp_storage):
    """Verify failed API response transitions through sending -> failed and records safe audit info."""
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)

    reminder = {
        "reminder_id": "TEST-FAIL-001___ITEM-FAIL___2026-05-01___2d",
        "customerId": "TEST-FAIL-001",
        "itemId": "ITEM-FAIL",
        "customerName": "Failure Test Customer",
        "MOBILE_NO": "+919876543210",
        "itemName": "GLIMEPIRIDE 1MG",
        "expected_refill_date": "2026-05-01",
        "reminder_date": "2026-05-03",
        "reminder_stage": 2,
        "message": "Your expected refill date was 01-05-2026. If you still need your medicine, please contact us.",
        "status": "scheduled",
    }

    temp_storage.insert_or_update_scheduled_reminder(reminder)

    # Mock failed HTTP response
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": "Internal Provider Server Error"}

    with patch.dict(os.environ, {"XINNO_API_KEY": "test_secret_key", "XINNO_WABA_NUMBER": "919999999999"}):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            res = dispatcher.dispatch_single_reminder(reminder, dry_run=False)

            assert mock_post.called
            assert res.status == "failed"
            assert res.success is False
            assert "API Error" in res.message
            assert res.status_code == 500

    # Verify SQLite audit record
    db_rec = temp_storage.get_reminder(reminder["reminder_id"])
    assert db_rec is not None
    assert db_rec["status"] == "failed"
    assert db_rec["attempt_count"] == 1
    assert db_rec["last_attempt_at"] is not None
    assert "Internal Provider Server Error" in db_rec["error_message"]

    # Security check: verify no credentials or full phone numbers in SQLite
    assert db_rec["phone_last4"] == "3210"
    assert "test_secret_key" not in str(db_rec.values())

    # Verify that the reminder is eligible for manual retry (status == 'failed')
    failed_reminders = temp_storage.get_all_reminders(status="failed")
    assert len(failed_reminders) == 1
    assert failed_reminders[0]["reminder_id"] == reminder["reminder_id"]

    allowed, reason = temp_storage.is_send_allowed(reminder["reminder_id"])
    assert allowed is True
    assert "retry" in reason.lower()


# ==============================================================================
# TEST SCENARIO 9 — SQLITE PERSISTENCE ACROSS RESTARTS
# ==============================================================================

def test_scenario_9_sqlite_persistence_across_restarts(tmp_path):
    """Verify SQLite database persistence and state survival across distinct storage instances."""
    db_file = tmp_path / "persistence_restart_test.db"

    # Instance 1: Create table, insert scheduled reminder, execute dry-run dispatch
    storage1 = RefillCareStorage(db_path=db_file)
    dispatcher1 = RefillCareReminderDispatcher(storage=storage1)

    reminder = {
        "reminder_id": "TEST-PERSIST-001___ITEM-P___2026-06-01___0d",
        "customerId": "TEST-PERSIST-001",
        "itemId": "ITEM-P",
        "customerName": "Persistent Customer",
        "MOBILE_NO": "+919876543210",
        "itemName": "ATORVASTATIN 10MG",
        "expected_refill_date": "2026-06-01",
        "reminder_date": "2026-06-01",
        "reminder_stage": 0,
        "message": "Your regular medicine refill may be due today, 01-06-2026.",
        "status": "scheduled",
    }
    dispatcher1.dispatch_single_reminder(reminder, dry_run=True)

    # Close/dereference storage1
    del dispatcher1
    del storage1

    # Instance 2: Open new connection to the same SQLite file
    storage2 = RefillCareStorage(db_path=db_file)
    restored_rec = storage2.get_reminder(reminder["reminder_id"])

    assert restored_rec is not None
    assert restored_rec["reminder_id"] == reminder["reminder_id"]
    assert restored_rec["customer_id"] == "TEST-PERSIST-001"
    assert restored_rec["item_id"] == "ITEM-P"
    assert restored_rec["status"] == "accepted"
    assert restored_rec["is_dry_run"] == 1
    assert restored_rec["attempt_count"] == 1
    assert restored_rec["phone_last4"] == "3210"

    # Verify metrics calculation persists
    metrics = storage2.get_dashboard_metrics()
    assert metrics["total_accepted_dry"] == 1
    assert metrics["total_attempts"] == 1


# ==============================================================================
# SECURITY AUDIT & PHONE MASKING TESTS
# ==============================================================================

def test_security_phone_masking_and_no_credentials():
    """Verify phone masking and credential safety across all utility functions."""
    assert mask_phone_for_ui("+919876543210") == "***3210"
    assert mask_phone_for_ui("9876543210") == "***3210"
    assert _extract_phone_last4("+919876543210") == "3210"
    assert _extract_phone_last4("3210") == "3210"
    assert _extract_phone_last4("") == ""


# ==============================================================================
# STREAMLIT APPLICATION IMPORT VERIFICATION
# ==============================================================================

def test_streamlit_app_refillcare_import():
    """Verify app_refillcare imports cleanly with required helper utilities."""
    import app_refillcare as app

    assert hasattr(app, "find_artifact_path")
    assert hasattr(app, "load_refill_model")
    assert hasattr(app, "load_refill_data")
    assert hasattr(app, "load_purchase_history")
    assert hasattr(app, "prepare_prediction_overview")
    assert hasattr(app, "generate_reminder_schedule_table")
    assert hasattr(app, "filter_predictions")
    assert hasattr(app, "filter_schedules")
    assert hasattr(app, "get_customer_history_summary")
