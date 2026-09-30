"""Unit tests for RefillCare 3-Tier WhatsApp Service & Dispatcher."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base,
    RefillDecisionModel,
    ReminderCycleModel,
    ReminderStageModel,
    WhatsAppDeliveryLogModel,
)
from refillcare.whatsapp.client import (
    RefillWhatsAppClient,
    mask_phone_for_log,
    normalize_e164_whatsapp_phone,
)
from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher
from refillcare.whatsapp.template_formatter import (
    build_dynamic_tier_text,
    build_template_parameters,
    format_full_message_preview,
    get_tier_for_stage_offset,
)


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_get_tier_for_stage_offset():
    """Verify exact stage offset to 3-tier classification."""
    assert get_tier_for_stage_offset(-7) == "DUE"
    assert get_tier_for_stage_offset(-3) == "DUE"
    assert get_tier_for_stage_offset(-1) == "DUE"
    assert get_tier_for_stage_offset(0) == "DUE"
    assert get_tier_for_stage_offset(2) == "FOLLOWUP"
    assert get_tier_for_stage_offset(5) == "FOLLOWUP"
    assert get_tier_for_stage_offset(40) == "LAPSED"
    assert get_tier_for_stage_offset(45) == "LAPSED"


def test_build_dynamic_tier_text_due_stage():
    """Verify DUE_REFILL function message."""
    txt_due = build_dynamic_tier_text(
        tier="DUE",
        medications="Metformin 500mg",
        expected_refill_date="2026-09-30",
        stage_offset=0,
    )
    assert "Your regular medicine refill is due soon." in txt_due


def test_build_dynamic_tier_text_followup_and_lapsed():
    """Verify REFILL_FOLLOW_UP and LAPSED_REENGAGEMENT function messages."""
    # REFILL_FOLLOW_UP
    txt_followup = build_dynamic_tier_text(
        tier="FOLLOWUP",
        medications="Atchol F",
        expected_refill_date="2026-09-25",
        stage_offset=5,
    )
    assert "We’re following up regarding your regular medicine refill." in txt_followup

    # LAPSED_REENGAGEMENT (+45 Days)
    txt_lapsed = build_dynamic_tier_text(
        tier="LAPSED",
        medications="Rosuva Gold 10mg",
        expected_refill_date="2026-08-15",
        stage_offset=45,
    )
    assert "We’re checking in regarding your regular medicine refill." in txt_lapsed


def test_build_template_parameters():
    """Verify parameters array structure conforms to 6-variable template format."""
    params6 = build_template_parameters(
        customer_name="Anil Kumar",
        tier="DUE",
        medications="Metformin 500mg",
        expected_refill_date="2026-09-30",
        store_name="PHARMA HUBB",
        store_contact="+91 9966473474",
        param_count=6,
    )
    assert len(params6) == 6
    assert params6[0]["text"] == "Anil Kumar"
    assert params6[1]["text"] == "PHARMA HUBB"
    assert params6[2]["text"] == "Your regular medicine refill is due soon."
    assert params6[3]["text"] == "Metformin 500mg"
    assert params6[4]["text"] == "+91 9966473474"
    assert params6[5]["text"] == "PHARMA HUBB"

    # Multi-med bundle
    params_multi = build_template_parameters(
        customer_name="Sunil Sharma",
        tier="FOLLOWUP",
        medications=["Atchol F", "Telmisartan 40mg"],
        expected_refill_date="2026-09-30",
        store_name="PHARMA HUBB",
        store_contact="+91 9966473474",
        param_count=6,
    )
    assert len(params_multi) == 6
    assert params_multi[2]["text"] == "We’re following up regarding your regular medicine refill."
    assert params_multi[3]["text"] == "Atchol F, Telmisartan 40mg"


def test_normalize_phone_and_masking():
    """Verify E.164 normalization and safe log masking."""
    assert normalize_e164_whatsapp_phone("9640568227") == "919640568227"
    assert normalize_e164_whatsapp_phone("+919640568227") == "919640568227"
    assert normalize_e164_whatsapp_phone("09640568227") == "919640568227"
    assert mask_phone_for_log("919640568227") == "91******8227"


def test_client_dry_run_send():
    """Verify client dry-run produces complete validated response without HTTP call."""
    client = RefillWhatsAppClient(
        api_key="mock_key",
        waba_number="919515473474",
        template_name="refillcare_reminder_unified",
    )

    res = client.send_refill_reminder(
        phone_number="9640568227",
        customer_name="Anil Kumar",
        medications="Atchol F",
        expected_refill_date="2026-09-30",
        tier="DUE",
        dry_run=True,
    )

    assert res["success"] is True
    assert res["status_code"] == 200
    assert res["is_dry_run"] is True
    assert "919640568227" in res["response"]["payload"]["to"]
    assert res["masked_phone"] == "91******8227"


def test_dispatcher_batch_execution(in_memory_db):
    """Verify dispatcher batch execution updates database stages and appends delivery logs."""
    # 1. Seed test database
    dec = RefillDecisionModel(
        decision_id="DEC_001",
        cycle_id="CYC_001",
        customer_id="CUST_001",
        customer_name="Sunil Sharma",
        mobile_no="9581473474",
        item_id="MED_001",
        item_name="Metformin 500mg",
        customer_item_key="CUST_001_MED_001",
        path="PATH_A",
        is_eligible=True,
        last_purchase_date=date(2026, 8, 31),
        expected_refill_date=date(2026, 9, 30),
    )
    cyc = ReminderCycleModel(
        cycle_id="CYC_001",
        customer_item_key="CUST_001_MED_001",
        customer_id="CUST_001",
        item_id="MED_001",
        decision_id="DEC_001",
        last_purchase_date=date(2026, 8, 31),
        expected_refill_date=date(2026, 9, 30),
        is_active=True,
    )
    stg = ReminderStageModel(
        reminder_id="REM_STAGE_001",
        cycle_id="CYC_001",
        customer_id="CUST_001",
        item_id="MED_001",
        customer_item_key="CUST_001_MED_001",
        stage_offset=0,
        target_send_date=date(2026, 9, 30),
        expected_refill_date=date(2026, 9, 30),
        status="PENDING",
    )
    in_memory_db.add(dec)
    in_memory_db.add(cyc)
    in_memory_db.add(stg)
    in_memory_db.commit()

    dispatcher = RefillWhatsAppDispatcher()
    res = dispatcher.dispatch_batch(
        db=in_memory_db,
        target_date=date(2026, 9, 30),
        tier_filter="DUE",
        dry_run=True,
    )

    assert res["dispatched_count"] == 1
    assert res["success_count"] == 1

    # Verify stage was updated in DB
    updated_stg = in_memory_db.query(ReminderStageModel).filter_by(reminder_id="REM_STAGE_001").first()
    assert updated_stg.status == "DRY_RUN_SUCCESS"

    # Verify delivery log was written
    logs = in_memory_db.query(WhatsAppDeliveryLogModel).all()
    assert len(logs) == 1
    assert logs[0].customer_name == "Sunil Sharma"
    assert logs[0].status == "DRY_RUN_SUCCESS"
