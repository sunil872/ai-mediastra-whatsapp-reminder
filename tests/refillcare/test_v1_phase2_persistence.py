"""Unit & Integration Tests for V1 Phase 2: Persistence, Review Queue & Controlled Dispatch."""
from datetime import date, datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from database.connection import Base
from database.models import (
    RefillDecisionModel,
    ReminderCycleModel,
    ReminderStageModel,
    WhatsAppDeliveryLogModel,
)
from refillcare.engine.decision_types import (
    PATH_A,
    PATH_B,
    RefillDecision,
    STABILITY_HIGH,
    STATUS_APPROVED,
    STATUS_CANCELLED,
    STATUS_DUE,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SENT,
    STATUS_SUPERSEDED,
)
from refillcare.engine.persistence import RefillPersistenceManager
from api.main import app
from database.connection import get_db


from sqlalchemy.pool import StaticPool

@pytest.fixture
def test_db():
    """In-memory SQLite database session fixture."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """FastAPI TestClient with overridden database session."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_decision_persistence_insert_and_update(test_db):
    pm = RefillPersistenceManager()
    dec1 = RefillDecision(
        customer_id="C1",
        customer_name="Test Customer",
        mobile_no="919876543210",
        item_id="I1",
        item_name="TELMISARTAN",
        customer_item_key="C1_I1",
        path=PATH_A,
        purchase_count=8,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.15,
        cadence_drift=1.0,
        dos_days=30.0,
        units_purchased=30.0,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 1),
        expected_refill_date=date(2026, 8, 31),
        decision_reason="Path A High Stability",
        cycle_id="RC-C1-I1-20260801",
    )

    # 1. Insert
    count = pm.save_refill_decisions(test_db, [dec1])
    assert count == 1

    rec = test_db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_item_key == "C1_I1").first()
    assert rec is not None
    assert rec.is_eligible is True
    assert rec.predicted_interval_days == 30

    # 2. Update with modified interval
    dec1.predicted_interval_days = 28
    pm.save_refill_decisions(test_db, [dec1])

    rec_updated = test_db.query(RefillDecisionModel).filter(RefillDecisionModel.customer_item_key == "C1_I1").first()
    assert rec_updated.predicted_interval_days == 28


def test_reminder_cycle_and_stage_persistence_idempotency(test_db):
    pm = RefillPersistenceManager()
    dec = RefillDecision(
        customer_id="C2",
        customer_name="John Idempotent",
        mobile_no="919876543211",
        item_id="I2",
        item_name="METFORMIN",
        customer_item_key="C2_I2",
        path=PATH_A,
        purchase_count=10,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.1,
        cadence_drift=0.5,
        dos_days=None,
        units_purchased=None,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 1),
        expected_refill_date=date(2026, 8, 31),
        decision_reason="Path A High Stability",
        cycle_id="RC-C2-I2-20260801",
    )

    # Run 1: Create cycles and stages
    metrics1 = pm.sync_reminder_cycles(test_db, [dec])
    assert metrics1["cycles_created"] == 1
    assert metrics1["stages_created"] == 7

    stages_count = test_db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == dec.cycle_id).count()
    assert stages_count == 7

    # Run 2: Re-run same operation (prove idempotency)
    metrics2 = pm.sync_reminder_cycles(test_db, [dec])
    assert metrics2["cycles_created"] == 0
    assert metrics2["stages_created"] == 0
    assert metrics2["duplicate_stages_prevented"] == 7

    # Verify total stages still equals 7 (no duplication)
    stages_count_after = test_db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == dec.cycle_id).count()
    assert stages_count_after == 7


def test_repurchase_cycle_reset_supersedes_unsent_stages(test_db):
    pm = RefillPersistenceManager()

    # Old Cycle: Purchase on 2026-08-01 -> Refill on 2026-08-31
    dec_old = RefillDecision(
        customer_id="C3",
        customer_name="Patient Repurchase",
        mobile_no="919876543212",
        item_id="I3",
        item_name="ATORVASTATIN",
        customer_item_key="C3_I3",
        path=PATH_A,
        purchase_count=6,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.1,
        cadence_drift=0.0,
        dos_days=None,
        units_purchased=None,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 1),
        expected_refill_date=date(2026, 8, 31),
        decision_reason="Path A Old Cycle",
        cycle_id="RC-C3-I3-20260801",
    )
    pm.sync_reminder_cycles(test_db, [dec_old])

    # Mark Stage -7 as already SENT
    old_stage_minus7 = (
        test_db.query(ReminderStageModel)
        .filter(ReminderStageModel.cycle_id == dec_old.cycle_id, ReminderStageModel.stage_offset == -7)
        .first()
    )
    old_stage_minus7.status = STATUS_SENT
    old_stage_minus7.sent_at = datetime(2026, 8, 24, 10, 0, 0)
    test_db.commit()

    # New Cycle: Patient bought again early on 2026-08-26 -> New Refill on 2026-09-25
    dec_new = RefillDecision(
        customer_id="C3",
        customer_name="Patient Repurchase",
        mobile_no="919876543212",
        item_id="I3",
        item_name="ATORVASTATIN",
        customer_item_key="C3_I3",
        path=PATH_A,
        purchase_count=7,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.1,
        cadence_drift=0.0,
        dos_days=None,
        units_purchased=None,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 26),
        expected_refill_date=date(2026, 9, 25),
        decision_reason="Path A New Cycle",
        cycle_id="RC-C3-I3-20260826",
    )
    metrics_new = pm.sync_reminder_cycles(test_db, [dec_new])

    assert metrics_new["cycles_superseded"] == 1
    assert metrics_new["stages_superseded"] == 6  # 6 unsent stages superseded

    # Verify old SENT record was preserved
    stage_sent = (
        test_db.query(ReminderStageModel)
        .filter(ReminderStageModel.cycle_id == dec_old.cycle_id, ReminderStageModel.stage_offset == -7)
        .first()
    )
    assert stage_sent.status == STATUS_SENT
    assert stage_sent.sent_at is not None

    # Verify other 6 stages of old cycle are SUPERSEDED_BY_PURCHASE
    superseded_stages = (
        test_db.query(ReminderStageModel)
        .filter(ReminderStageModel.cycle_id == dec_old.cycle_id, ReminderStageModel.status == STATUS_SUPERSEDED)
        .all()
    )
    assert len(superseded_stages) == 6
    assert "Customer repurchased on 2026-08-26" in superseded_stages[0].failure_reason


def test_invalid_state_transitions_rejected(test_db):
    pm = RefillPersistenceManager()
    dec = RefillDecision(
        customer_id="C4",
        customer_name="Invalid State Test",
        mobile_no="919876543213",
        item_id="I4",
        item_name="AMLODIPINE",
        customer_item_key="C4_I4",
        path=PATH_A,
        purchase_count=6,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.1,
        cadence_drift=0.0,
        dos_days=None,
        units_purchased=None,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 1),
        expected_refill_date=date(2026, 8, 31),
        decision_reason="Test",
        cycle_id="RC-C4-I4-20260801",
    )
    pm.sync_reminder_cycles(test_db, [dec])

    # 1. Attempt to dispatch PENDING stage without approval -> must raise ValueError
    pending_stage = test_db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == dec.cycle_id).first()
    assert pending_stage.status == STATUS_PENDING
    with pytest.raises(ValueError, match="Cannot dispatch reminder in state 'PENDING'. Stage must be APPROVED by pharmacist first."):
        pm.dispatch_reminder_stage(test_db, pending_stage.reminder_id, is_dry_run=True)

    # 2. Transition to SUPERSEDED
    pending_stage.status = STATUS_SUPERSEDED
    test_db.commit()

    # Attempt to approve or dispatch superseded stage -> must raise ValueError
    with pytest.raises(ValueError, match="Cannot approve reminder in state 'SUPERSEDED_BY_PURCHASE'"):
        pm.approve_reminder_stage(test_db, pending_stage.reminder_id)

    with pytest.raises(ValueError, match="Cannot dispatch reminder in state 'SUPERSEDED_BY_PURCHASE'"):
        pm.dispatch_reminder_stage(test_db, pending_stage.reminder_id, is_dry_run=True)



def test_fastapi_review_queue_and_actions(client, test_db):
    pm = RefillPersistenceManager()
    target_date = date(2026, 8, 24)

    dec = RefillDecision(
        customer_id="C5",
        customer_name="Alice API",
        mobile_no="919876543214",
        item_id="I5",
        item_name="ROSUVASTATIN",
        customer_item_key="C5_I5",
        path=PATH_A,
        purchase_count=6,
        is_eligible=True,
        stability_tier=STABILITY_HIGH,
        cadence_median=30.0,
        cadence_norm_mad=0.1,
        cadence_drift=0.0,
        dos_days=None,
        units_purchased=None,
        prediction_method="PERSONAL_HISTORICAL_MEDIAN",
        predicted_interval_days=30,
        last_purchase_date=date(2026, 8, 1),
        expected_refill_date=date(2026, 8, 31),  # Day -7 is 2026-08-24
        decision_reason="Path A High Stability",
        cycle_id="RC-C5-I5-20260801",
    )
    pm.save_refill_decisions(test_db, [dec])
    pm.sync_reminder_cycles(test_db, [dec])

    # 1. GET /api/reminders/today
    res = client.get(f"/api/reminders/today?target_date={target_date.strftime('%Y-%m-%d')}")
    assert res.status_code == 200
    queue = res.json()
    assert len(queue) == 1
    rem_id = queue[0]["reminder_id"]
    assert queue[0]["customer_name"] == "Alice API"
    assert queue[0]["stage_offset"] == -7

    # 2. POST /api/reminders/{reminder_id}/approve
    approve_res = client.post(f"/api/reminders/{rem_id}/approve")
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "APPROVED"

    # 3. POST /api/reminders/{reminder_id}/dispatch (DRY-RUN)
    dispatch_res = client.post(f"/api/reminders/{rem_id}/dispatch", json={"is_dry_run": True})
    assert dispatch_res.status_code == 200
    assert dispatch_res.json()["status"] == "SENT"
    assert dispatch_res.json()["is_dry_run"] is True

    # 4. Verify Audit Log was recorded
    audit_count = test_db.query(WhatsAppDeliveryLogModel).filter(WhatsAppDeliveryLogModel.reminder_id == rem_id).count()
    assert audit_count == 1

    # 5. GET /api/customers/{customer_id}/refill-history
    history_res = client.get("/api/customers/C5/refill-history")
    assert history_res.status_code == 200
    hist_data = history_res.json()
    assert hist_data["total_cycles"] == 1
    assert len(hist_data["cycles"][0]["stages"]) == 7
