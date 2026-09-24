"""Unit tests for Phase 8 RefillCare Persistent Reminder State & Audit Storage."""

import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from reminder.storage import RefillCareStorage, _extract_phone_last4
from reminder.dispatch import (
    DispatchResult,
    RefillCareReminderDispatcher,
    select_eligible_reminders_for_dispatch,
)
from reminder.scheduler import ReminderRecord


@pytest.fixture
def temp_storage():
    """Create a temporary SQLite database for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_refillcare.db"
        storage = RefillCareStorage(db_path=db_path)
        yield storage


def test_storage_database_initialization_and_tables(temp_storage):
    """Verify that SQLite file is created and tables/indices are initialized."""
    assert temp_storage.db_path.exists()
    with temp_storage._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reminder_audit';")
        assert cursor.fetchone() is not None

        # Check column names
        cursor.execute("PRAGMA table_info(reminder_audit);")
        cols = [row["name"] for row in cursor.fetchall()]
        assert "reminder_id" in cols
        assert "customer_id" in cols
        assert "item_id" in cols
        assert "phone_last4" in cols
        assert "status" in cols
        assert "attempt_count" in cols
        assert "provider_message_id" in cols
        assert "error_message" in cols
        assert "is_dry_run" in cols


def test_insert_and_retrieve_scheduled_reminder(temp_storage):
    """Verify inserting a scheduled reminder and fetching it by reminder_id."""
    rem_data = {
        "reminder_id": "C001___M001___2026-09-09___-7d",
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Jane Doe",
        "itemName": "METFORMIN 500MG",
        "MOBILE_NO": "9849012345",
        "expected_refill_date": "2026-09-09",
        "reminder_date": "2026-09-02",
        "reminder_stage": "-7d",
        "history_quality": "high_history",
        "status": "scheduled",
    }
    temp_storage.insert_or_update_scheduled_reminder(rem_data)

    retrieved = temp_storage.get_reminder("C001___M001___2026-09-09___-7d")
    assert retrieved is not None
    assert retrieved["customer_id"] == "C001"
    assert retrieved["item_id"] == "M001"
    assert retrieved["customer_name"] == "Jane Doe"
    assert retrieved["phone_last4"] == "2345"
    assert retrieved["status"] == "scheduled"
    assert retrieved["attempt_count"] == 0


def test_no_full_phone_number_or_credentials_stored(temp_storage):
    """Ensure full phone number and credentials are never stored in SQLite."""
    full_phone = "919849012345"
    rem_data = {
        "reminder_id": "C002___M002___2026-09-10___-3d",
        "customerId": "C002",
        "itemId": "M002",
        "customerName": "John Secret",
        "itemName": "ATORVASTATIN 10MG",
        "MOBILE_NO": full_phone,
        "expected_refill_date": "2026-09-10",
        "reminder_date": "2026-09-07",
        "reminder_stage": "-3d",
        "status": "scheduled",
    }
    temp_storage.insert_or_update_scheduled_reminder(rem_data)

    rec = temp_storage.get_reminder("C002___M002___2026-09-10___-3d")
    assert rec["phone_last4"] == "2345"
    assert full_phone not in str(rec.values())


def test_attempt_count_increment_and_status_transitions(temp_storage):
    """Verify attempt_count increments on dispatch start and records outcomes."""
    rem_data = {
        "reminder_id": "C003___M003___2026-09-11___0d",
        "customerId": "C003",
        "itemId": "M003",
        "customerName": "Alice Smith",
        "itemName": "LOSARTAN 50MG",
        "MOBILE_NO": "9876543210",
        "expected_refill_date": "2026-09-11",
        "reminder_date": "2026-09-11",
        "reminder_stage": "0d",
    }

    # First attempt: start
    att1 = temp_storage.record_dispatch_start(rem_data, is_dry_run=False)
    assert att1 == 1

    rec1 = temp_storage.get_reminder("C003___M003___2026-09-11___0d")
    assert rec1["status"] == "sending"
    assert rec1["attempt_count"] == 1

    # First attempt: failure
    fail_res = DispatchResult(
        reminder_id="C003___M003___2026-09-11___0d",
        customerId="C003",
        itemId="M003",
        customerName="Alice Smith",
        phone_masked="***3210",
        reminder_stage="0d",
        expected_refill_date="2026-09-11",
        status="failed",
        success=False,
        message="Simulated temporary network timeout",
        error_category="Network Timeout",
        dry_run=False,
    )
    temp_storage.record_dispatch_result(fail_res, is_dry_run=False)

    rec_failed = temp_storage.get_reminder("C003___M003___2026-09-11___0d")
    assert rec_failed["status"] == "failed"
    assert rec_failed["error_category"] == "Network Timeout"
    assert rec_failed["error_message"] == "Simulated temporary network timeout"
    assert rec_failed["attempt_count"] == 1

    # Second attempt: manual retry start
    att2 = temp_storage.record_dispatch_start(rem_data, is_dry_run=False)
    assert att2 == 2

    # Second attempt: success
    success_res = DispatchResult(
        reminder_id="C003___M003___2026-09-11___0d",
        customerId="C003",
        itemId="M003",
        customerName="Alice Smith",
        phone_masked="***3210",
        reminder_stage="0d",
        expected_refill_date="2026-09-11",
        status="accepted",
        success=True,
        message="Xinno API accepted reminder.",
        provider_message_id="wamid.HBgLMTIzNDU",
        dry_run=False,
    )
    temp_storage.record_dispatch_result(success_res, is_dry_run=False)

    rec_succ = temp_storage.get_reminder("C003___M003___2026-09-11___0d")
    assert rec_succ["status"] == "accepted"
    assert rec_succ["attempt_count"] == 2
    assert rec_succ["provider_message_id"] == "wamid.HBgLMTIzNDU"


def test_idempotency_and_accepted_reminder_cannot_be_resent(temp_storage):
    """Verify that an accepted live reminder is blocked from duplicate dispatch."""
    rem_id = "C004___M004___2026-09-12___-7d"
    success_res = DispatchResult(
        reminder_id=rem_id,
        customerId="C004",
        itemId="M004",
        customerName="Bob Brown",
        phone_masked="***9999",
        reminder_stage="-7d",
        expected_refill_date="2026-09-12",
        status="accepted",
        success=True,
        message="Accepted by WhatsApp provider.",
        provider_message_id="prov_id_12345",
        dry_run=False,
    )
    temp_storage.record_dispatch_result(success_res, is_dry_run=False)

    allowed, reason = temp_storage.is_send_allowed(rem_id)
    assert allowed is False
    assert "already accepted" in reason.lower()

    # Now verify with RefillCareReminderDispatcher using this storage
    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)
    rem_dict = {
        "reminder_id": rem_id,
        "customerId": "C004",
        "itemId": "M004",
        "customerName": "Bob Brown",
        "MOBILE_NO": "9849099999",
        "expected_refill_date": "2026-09-12",
        "reminder_stage": -7,
        "message": "Your regular refill is due.",
        "status": "scheduled",
    }
    dispatch_res = dispatcher.dispatch_single_reminder(rem_dict, dry_run=False)
    assert dispatch_res.status == "duplicate_prevented"
    assert dispatch_res.success is False


def test_cancelled_reminder_cannot_be_sent(temp_storage):
    """Verify that cancelled reminders are never dispatched."""
    rem_id = "C005___M005___2026-09-15___-3d"
    cancel_res = DispatchResult(
        reminder_id=rem_id,
        customerId="C005",
        itemId="M005",
        customerName="Charlie Cox",
        phone_masked="***1111",
        reminder_stage="-3d",
        expected_refill_date="2026-09-15",
        status="cancelled",
        success=False,
        message="Cancelled due to early refill purchase.",
        dry_run=False,
    )
    temp_storage.record_dispatch_result(cancel_res, is_dry_run=False)

    allowed, reason = temp_storage.is_send_allowed(rem_id)
    assert allowed is False
    assert "cancelled" in reason.lower()


def test_failed_reminder_can_be_manually_retried(temp_storage):
    """Verify that failed reminders are eligible for manual retry."""
    rem_id = "C006___M006___2026-09-16___+2d"
    fail_res = DispatchResult(
        reminder_id=rem_id,
        customerId="C006",
        itemId="M006",
        customerName="Diana Prince",
        phone_masked="***2222",
        reminder_stage="+2d",
        expected_refill_date="2026-09-16",
        status="failed",
        success=False,
        message="HTTP 503 Service Unavailable",
        error_category="Server Error",
        dry_run=False,
    )
    temp_storage.record_dispatch_result(fail_res, is_dry_run=False)

    allowed, reason = temp_storage.is_send_allowed(rem_id)
    assert allowed is True
    assert "retry" in reason.lower()


def test_multiple_customers_sharing_phone_remain_separate(temp_storage):
    """Verify that distinct customers sharing a phone number maintain isolated audit entries."""
    shared_phone = "9849000000"
    rem1 = {
        "reminder_id": "C_FATHER___M001___2026-09-20___-7d",
        "customerId": "C_FATHER",
        "itemId": "M001",
        "customerName": "Father Customer",
        "itemName": "TELMISARTAN",
        "MOBILE_NO": shared_phone,
        "expected_refill_date": "2026-09-20",
        "reminder_date": "2026-09-13",
        "reminder_stage": "-7d",
        "status": "scheduled",
    }
    rem2 = {
        "reminder_id": "C_SON___M002___2026-09-20___-7d",
        "customerId": "C_SON",
        "itemId": "M002",
        "customerName": "Son Customer",
        "itemName": "CETIRIZINE",
        "MOBILE_NO": shared_phone,
        "expected_refill_date": "2026-09-20",
        "reminder_date": "2026-09-13",
        "reminder_stage": "-7d",
        "status": "scheduled",
    }
    temp_storage.insert_or_update_scheduled_reminder(rem1)
    temp_storage.insert_or_update_scheduled_reminder(rem2)

    rec1 = temp_storage.get_reminder("C_FATHER___M001___2026-09-20___-7d")
    rec2 = temp_storage.get_reminder("C_SON___M002___2026-09-20___-7d")

    assert rec1 is not None and rec2 is not None
    assert rec1["customer_id"] == "C_FATHER"
    assert rec2["customer_id"] == "C_SON"
    assert rec1["reminder_id"] != rec2["reminder_id"]
    assert rec1["phone_last4"] == rec2["phone_last4"] == "0000"


def test_dry_run_audit_record_and_distinction(temp_storage):
    """Verify dry_run records are created and clearly distinguished from live sends."""
    rem_dict = {
        "reminder_id": "C007___M007___2026-09-25___0d",
        "customerId": "C007",
        "itemId": "M007",
        "customerName": "Evan Wright",
        "itemName": "GLIMEPIRIDE 2MG",
        "MOBILE_NO": "9849077777",
        "expected_refill_date": "2026-09-25",
        "reminder_date": "2026-09-25",
        "reminder_stage": 0,
        "message": "Your refill is due today.",
        "status": "scheduled",
    }

    dispatcher = RefillCareReminderDispatcher(storage=temp_storage)
    res = dispatcher.dispatch_single_reminder(rem_dict, dry_run=True)

    assert res.status == "accepted"
    assert res.dry_run is True

    audit_rec = temp_storage.get_reminder("C007___M007___2026-09-25___0d")
    assert audit_rec is not None
    assert audit_rec["is_dry_run"] == 1
    assert audit_rec["status"] == "accepted"


def test_persistence_survives_new_storage_instance(temp_storage):
    """Verify data persists when a new RefillCareStorage instance connects to the same file."""
    rem_id = "C008___M008___2026-09-30___-1d"
    temp_storage.insert_or_update_scheduled_reminder({
        "reminder_id": rem_id,
        "customerId": "C008",
        "itemId": "M008",
        "customerName": "Frank Castle",
        "itemName": "AMLODIPINE 5MG",
        "MOBILE_NO": "9849088888",
        "expected_refill_date": "2026-09-30",
        "reminder_date": "2026-09-29",
        "reminder_stage": "-1d",
        "status": "scheduled",
    })

    # Create new instance pointing to exact same database path
    new_storage_instance = RefillCareStorage(db_path=temp_storage.db_path)
    retrieved = new_storage_instance.get_reminder(rem_id)
    assert retrieved is not None
    assert retrieved["customer_name"] == "Frank Castle"
    assert retrieved["phone_last4"] == "8888"


def test_get_audit_dataframe_and_csv_preparation(temp_storage):
    """Verify conversion of audit records to DataFrame with filtering."""
    temp_storage.insert_or_update_scheduled_reminder({
        "reminder_id": "C009___M009___2026-10-01___-7d",
        "customerId": "C009",
        "itemId": "M009",
        "customerName": "Grace Hopper",
        "itemName": "PANTOPRAZOLE 40MG",
        "MOBILE_NO": "9849011111",
        "expected_refill_date": "2026-10-01",
        "reminder_date": "2026-09-24",
        "reminder_stage": "-7d",
        "status": "scheduled",
    })

    df = temp_storage.get_audit_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Reminder ID" in df.columns
    assert "Customer" in df.columns
    assert "Phone Last-4" in df.columns
    assert df["Customer"].iloc[0] == "Grace Hopper"
    assert df["Phone Last-4"].iloc[0] == "***1111"

    # Test filtering
    filtered_df = temp_storage.get_audit_dataframe(customer_query="Grace")
    assert len(filtered_df) == 1
    assert filtered_df["Customer"].iloc[0] == "Grace Hopper"

    filtered_none = temp_storage.get_audit_dataframe(customer_query="NonExistent")
    assert len(filtered_none) == 0


def test_dashboard_metrics_calculation(temp_storage):
    """Verify aggregation counts in get_dashboard_metrics."""
    temp_storage.insert_or_update_scheduled_reminder({
        "reminder_id": "C010___M010___2026-10-05___0d",
        "customerId": "C010",
        "itemId": "M010",
        "customerName": "Henry Ford",
        "itemName": "PARACETAMOL",
        "MOBILE_NO": "9849022222",
        "expected_refill_date": "2026-10-05",
        "reminder_date": "2026-10-05",
        "reminder_stage": "0d",
        "status": "scheduled",
    })

    metrics = temp_storage.get_dashboard_metrics(today_date="2026-10-05")
    assert metrics["total_scheduled"] == 1
    assert metrics["today_due"] == 1
    assert metrics["total_accepted_live"] == 0
