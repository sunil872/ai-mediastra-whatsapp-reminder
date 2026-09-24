"""Unit and integration tests for RefillCare Phase 16 Micro-Pilot Engine.

Verifies:
1. Micro-pilot batch size selection (5, 10, 20 only; default 5).
2. Strict Tier A filtering and compound identity isolation (customerId + itemId).
3. Shared phone numbers remain separated as delivery destinations only.
4. Pre-send safety checks block invalid records, template errors, and stale cycles.
5. Cycle-reset / repurchase detection blocks stale reminders.
6. Explicit human operator approval requirement.
7. Default Dry-Run execution with zero external requests and persistent SQLite audit.
8. Live send execution boundaries and provider status recording ("accepted", "failed").
9. Operational stop conditions detection (duplicates, repurchases, abnormal failures).
10. Outcome tracking with right-censoring / pending observation windows.
11. Privacy masking verification (no full phone numbers in logs/reports).
12. Micro-Pilot summary report generation and export.
"""

import os
import pytest
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from reminder.storage import RefillCareStorage
from reminder.dispatch import REFILLCARE_TEMPLATE_NAME, mask_phone_for_ui
from refillcare.evaluation.micro_pilot import (
    MicroPilotCandidate,
    MicroPilotReport,
    MicroPilotStopCondition,
    prepare_micro_pilot_batch,
    execute_micro_pilot_batch,
    check_micro_pilot_stop_conditions,
    evaluate_micro_pilot_outcomes,
    generate_micro_pilot_summary_report,
    ALLOWED_MICRO_PILOT_BATCH_SIZES,
    DEFAULT_MICRO_PILOT_BATCH_SIZE,
)


@pytest.fixture
def temp_storage(tmp_path):
    """Provide isolated temporary SQLite database for testing."""
    db_file = tmp_path / "test_micropilot_audit.db"
    return RefillCareStorage(db_path=db_file)


@pytest.fixture
def sample_schedule_df():
    """Construct realistic sample reminder schedule DataFrame with Tier A & B records."""
    records = []
    # 8 Tier A candidates (high history >=5 purchases)
    for i in range(1, 9):
        records.append({
            "reminder_id": f"CUST_{i:03d}___ITEM_101___2026-06-01___+0d",
            "customerId": f"CUST_{i:03d}",
            "itemId": "ITEM_101",
            "Customer Name": f"Patient {i}",
            "Medicine": "Metformin 500mg",
            "Delivery Phone": f"987654321{i}",
            "Last Purchase Date": "2026-05-01",
            "Purchase Count": 6,
            "Predicted Interval (Days)": 31.0,
            "Expected Refill Date": "2026-06-01",
            "Reminder Date": "2026-06-01",
            "raw_reminder_date": "2026-06-01",
            "reminder_stage": 0,
            "History Quality": "high_history",
            "Pilot Tier": "Tier A (Strong Pilot)",
            "Status": "scheduled",
        })

    # 2 Tier B candidates (lower history)
    for i in range(9, 11):
        records.append({
            "reminder_id": f"CUST_{i:03d}___ITEM_101___2026-06-01___+0d",
            "customerId": f"CUST_{i:03d}",
            "itemId": "ITEM_101",
            "Customer Name": f"Patient {i}",
            "Medicine": "Metformin 500mg",
            "Delivery Phone": f"987654321{i}",
            "Last Purchase Date": "2026-05-01",
            "Purchase Count": 2,
            "Predicted Interval (Days)": 30.0,
            "Expected Refill Date": "2026-06-01",
            "Reminder Date": "2026-06-01",
            "raw_reminder_date": "2026-06-01",
            "reminder_stage": 0,
            "History Quality": "low_history",
            "Pilot Tier": "Tier B (Review Required)",
            "Status": "scheduled",
        })

    return pd.DataFrame(records)


def test_micro_pilot_batch_size_and_tier_a_filtering(sample_schedule_df, temp_storage):
    """Verify micro-pilot batch preparation restricts to small sizes and Tier A only."""
    # Test batch size 5 (default)
    candidates_5, total_avail = prepare_micro_pilot_batch(
        sample_schedule_df,
        storage=temp_storage,
        batch_size=5,
    )
    assert len(candidates_5) == 5
    assert total_avail == 8  # 8 Tier A candidates available
    for c in candidates_5:
        assert c.pilot_tier == "Tier A (Strong Pilot)"
        assert c.history_quality == "high_history"

    # Test batch size 10 (returns all 8 available Tier A)
    candidates_10, _ = prepare_micro_pilot_batch(
        sample_schedule_df,
        storage=temp_storage,
        batch_size=10,
    )
    assert len(candidates_10) == 8

    # Test invalid batch size defaults to 5
    candidates_invalid, _ = prepare_micro_pilot_batch(
        sample_schedule_df,
        storage=temp_storage,
        batch_size=100,  # invalid
    )
    assert len(candidates_invalid) == 5


def test_shared_phone_compound_isolation(temp_storage):
    """Verify patients sharing the same phone remain strictly isolated compound identities."""
    shared_schedule = pd.DataFrame([
        {
            "reminder_id": "CUST_A___ITEM_ALPHA___2026-06-01___+0d",
            "customerId": "CUST_A",
            "itemId": "ITEM_ALPHA",
            "Customer Name": "Alice Family",
            "Medicine": "Amlodipine 5mg",
            "Delivery Phone": "9876543210",
            "Last Purchase Date": "2026-05-01",
            "Purchase Count": 5,
            "Predicted Interval (Days)": 30.0,
            "Expected Refill Date": "2026-06-01",
            "Reminder Date": "2026-06-01",
            "raw_reminder_date": "2026-06-01",
            "reminder_stage": 0,
            "History Quality": "high_history",
            "Pilot Tier": "Tier A (Strong Pilot)",
            "Status": "scheduled",
        },
        {
            "reminder_id": "CUST_B___ITEM_BETA___2026-06-01___+0d",
            "customerId": "CUST_B",
            "itemId": "ITEM_BETA",
            "Customer Name": "Bob Family",
            "Medicine": "Metformin 500mg",
            "Delivery Phone": "9876543210",  # Same phone
            "Last Purchase Date": "2026-05-01",
            "Purchase Count": 5,
            "Predicted Interval (Days)": 30.0,
            "Expected Refill Date": "2026-06-01",
            "Reminder Date": "2026-06-01",
            "raw_reminder_date": "2026-06-01",
            "reminder_stage": 0,
            "History Quality": "high_history",
            "Pilot Tier": "Tier A (Strong Pilot)",
            "Status": "scheduled",
        },
    ])

    candidates, _ = prepare_micro_pilot_batch(shared_schedule, storage=temp_storage, batch_size=5)
    assert len(candidates) == 2
    assert candidates[0].customerId != candidates[1].customerId
    assert candidates[0].itemId != candidates[1].itemId
    assert candidates[0].reminder_id != candidates[1].reminder_id
    assert candidates[0].phone_masked == candidates[1].phone_masked == "***3210"


def test_human_operator_approval_requirement(sample_schedule_df, temp_storage):
    """Verify only explicitly approved candidates can be dispatched."""
    candidates, _ = prepare_micro_pilot_batch(sample_schedule_df, storage=temp_storage, batch_size=5)

    # Operator approves only candidate 0 and 1
    approved_ids = [candidates[0].reminder_id, candidates[1].reminder_id]

    res = execute_micro_pilot_batch(
        candidates=candidates,
        approved_candidate_ids=approved_ids,
        is_live=False,  # Dry Run
        storage=temp_storage,
    )

    assert res["total_approved"] == 2
    assert res["accepted_count"] == 2
    assert res["dry_run_count"] == 2
    assert res["live_count"] == 0

    # Verify audit persistence in SQLite
    audit_records = pd.DataFrame(temp_storage.get_all_reminders())
    assert len(audit_records) == 2
    assert set(audit_records["reminder_id"]) == set(approved_ids)


def test_dry_run_mode_creates_zero_live_requests(sample_schedule_df, temp_storage):
    """Verify Dry-Run mode makes zero external provider calls."""
    candidates, _ = prepare_micro_pilot_batch(sample_schedule_df, storage=temp_storage, batch_size=5)
    approved_ids = [c.reminder_id for c in candidates]

    mock_called = False

    def mock_provider(payload):
        nonlocal mock_called
        mock_called = True
        return {"success": True, "id": "TEST_123"}

    res = execute_micro_pilot_batch(
        candidates=candidates,
        approved_candidate_ids=approved_ids,
        is_live=False,  # DRY RUN
        storage=temp_storage,
        provider_callable=mock_provider,
    )

    # Provider should NEVER be called in Dry-Run mode
    assert mock_called is False
    assert res["dry_run_count"] == 5
    assert res["live_count"] == 0


def test_live_send_boundary_and_provider_status_recording(sample_schedule_df, temp_storage):
    """Verify Live Send calls provider and records 'accepted' / 'failed' without claiming 'delivered'."""
    candidates, _ = prepare_micro_pilot_batch(sample_schedule_df, storage=temp_storage, batch_size=5)
    approved_ids = [candidates[0].reminder_id, candidates[1].reminder_id]

    call_count = 0

    def mock_live_provider(payload):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"success": True, "messages": [{"id": "wamid.HBgL"}]}
        else:
            return {"success": False, "error": {"message": "Invalid recipient"}}

    res = execute_micro_pilot_batch(
        candidates=candidates,
        approved_candidate_ids=approved_ids,
        is_live=True,  # LIVE SEND
        storage=temp_storage,
        provider_callable=mock_live_provider,
    )

    assert call_count == 2
    assert res["live_count"] == 2
    assert res["accepted_count"] == 1
    assert res["failed_count"] == 1

    audit_records = pd.DataFrame(temp_storage.get_all_reminders())
    assert len(audit_records) == 2
    assert "accepted" in audit_records["status"].values
    assert "failed" in audit_records["status"].values
    # Ensure status is never "delivered"
    assert "delivered" not in audit_records["status"].values


def test_repurchase_cycle_reset_safety_check(temp_storage):
    """Verify newer repurchase blocks stale scheduled reminder from dispatch."""
    schedule_df = pd.DataFrame([{
        "reminder_id": "CUST_STALE___ITEM_A___2026-06-01___+0d",
        "customerId": "CUST_STALE",
        "itemId": "ITEM_A",
        "Customer Name": "Stale Patient",
        "Medicine": "Metformin 500mg",
        "Delivery Phone": "9876543210",
        "Last Purchase Date": "2026-05-01",
        "Purchase Count": 5,
        "Predicted Interval (Days)": 30.0,
        "Expected Refill Date": "2026-06-01",
        "Reminder Date": "2026-06-01",
        "raw_reminder_date": "2026-06-01",
        "reminder_stage": 0,
        "History Quality": "high_history",
        "Pilot Tier": "Tier A (Strong Pilot)",
        "Status": "scheduled",
    }])

    # Transactions showing a newer purchase occurred on 2026-05-20 (after baseline 2026-05-01)
    tx_df = pd.DataFrame([
        {
            "customerId": "CUST_STALE",
            "itemId": "ITEM_A",
            "invoice_date": "2026-05-20",
            "quantity": 30,
        }
    ])

    candidates, _ = prepare_micro_pilot_batch(
        schedule_df,
        transactions_df=tx_df,
        storage=temp_storage,
        batch_size=5,
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.pre_send_safety_status == "FAIL"
    assert cand.safety_code == "FAIL_ALREADY_PURCHASED"
    assert cand.is_safe_to_send is False

    # Attempting to dispatch blocked candidate
    res = execute_micro_pilot_batch(
        candidates=candidates,
        approved_candidate_ids=[cand.reminder_id],
        is_live=False,
        storage=temp_storage,
        transactions_df=tx_df,
    )
    assert res["blocked_count"] == 1
    assert res["accepted_count"] == 0


def test_micro_pilot_stop_conditions_surveillance(temp_storage):
    """Verify stop conditions trigger on duplicates, newer purchases, or high failure rates."""
    # Construct candidates with duplicate reminder ID
    candidates = [
        MicroPilotCandidate(
            reminder_id="DUP_ID", customerId="C1", itemId="I1",
            customer_name="P1", medicine="M1", phone_masked="***1234",
            phone_normalized="919876543210", latest_purchase_date="2026-05-01",
            predicted_interval_days=30.0, expected_refill_date="2026-06-01",
            reminder_stage=0, reminder_date="2026-06-01", history_quality="high_history",
            pilot_tier="Tier A (Strong Pilot)", purchase_count=5,
            pre_send_safety_status="PASS", safety_reason="OK", safety_code="SAFE",
            is_safe_to_send=True,
        ),
        MicroPilotCandidate(
            reminder_id="DUP_ID", customerId="C2", itemId="I2",
            customer_name="P2", medicine="M2", phone_masked="***1234",
            phone_normalized="919876543210", latest_purchase_date="2026-05-01",
            predicted_interval_days=30.0, expected_refill_date="2026-06-01",
            reminder_stage=0, reminder_date="2026-06-01", history_quality="high_history",
            pilot_tier="Tier A (Strong Pilot)", purchase_count=5,
            pre_send_safety_status="PASS", safety_reason="OK", safety_code="SAFE",
            is_safe_to_send=True,
        ),
    ]

    stop_conds = check_micro_pilot_stop_conditions(candidates, storage=temp_storage)
    dup_cond = next(c for c in stop_conds if c.name == "Duplicate Send Prevention")
    assert dup_cond.triggered is True
    assert dup_cond.severity == "CRITICAL_STOP"


def test_micro_pilot_outcome_evaluation_and_right_censoring(temp_storage):
    """Verify outcome evaluation handles actual refill matching and 30-day right-censoring."""
    audit_df = pd.DataFrame([
        {
            "reminder_id": "REM_REFILLED",
            "customer_id": "C_REFILLED",
            "item_id": "I_1",
            "customer_name": "Refilled Patient",
            "medicine": "Metformin 500mg",
            "phone": "919876543210",
            "reminder_date": "2026-05-01",
            "expected_refill_date": "2026-05-05",
            "reminder_stage": -3,
            "is_dry_run": 0,
            "status": "accepted",
        },
        {
            "reminder_id": "REM_RECENT_PENDING",
            "customer_id": "C_PENDING",
            "item_id": "I_2",
            "customer_name": "Recent Patient",
            "medicine": "Amlodipine 5mg",
            "phone": "919876543211",
            "reminder_date": "2026-05-25",  # 5 days ago from ref_date 2026-05-30
            "expected_refill_date": "2026-05-28",
            "reminder_stage": 0,
            "is_dry_run": 0,
            "status": "accepted",
        },
    ])

    # Transactions: only C_REFILLED made a subsequent purchase
    tx_df = pd.DataFrame([
        {
            "customerId": "C_REFILLED",
            "itemId": "I_1",
            "invoice_date": "2026-05-04",  # Purchased on 2026-05-04 (1 day before expected)
            "quantity": 30,
        }
    ])

    outcomes = evaluate_micro_pilot_outcomes(
        audit_df=audit_df,
        transactions_df=tx_df,
        reference_date="2026-05-30",
        observation_window_days=30,
    )

    assert len(outcomes) == 2

    # Verify refilled outcome
    row1 = outcomes[outcomes["customerId"] == "C_REFILLED"].iloc[0]
    assert bool(row1["has_post_reminder_purchase"]) is True
    assert row1["observation_status"] == "OBSERVED_REFILL"
    assert row1["prediction_error_days"] == -1.0  # 2026-05-04 vs 2026-05-05
    assert bool(row1["is_accurate_3d"]) is True

    # Verify right-censored pending observation
    row2 = outcomes[outcomes["customerId"] == "C_PENDING"].iloc[0]
    assert bool(row2["has_post_reminder_purchase"]) is False
    assert row2["observation_status"] == "PENDING_OBSERVATION"


def test_privacy_masking_compliance(sample_schedule_df, temp_storage):
    """Verify phone numbers are never stored or presented in cleartext."""
    candidates, _ = prepare_micro_pilot_batch(sample_schedule_df, storage=temp_storage, batch_size=5)

    for c in candidates:
        assert c.phone_masked.startswith("***")
        assert len(c.phone_masked) == 7

    # Run execution and verify report markdown
    res = execute_micro_pilot_batch(
        candidates=candidates,
        approved_candidate_ids=[c.reminder_id for c in candidates],
        is_live=False,
        storage=temp_storage,
    )

    report = generate_micro_pilot_summary_report(
        candidates=candidates,
        dispatched_summary=res,
        total_tier_a_available=8,
    )
    md_text = report.to_markdown()

    # Verify zero 10-digit raw phones in markdown text
    assert "987654321" not in md_text
    assert "MICROPILOT-2026-01" in md_text
