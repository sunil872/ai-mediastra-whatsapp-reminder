"""Unit & Integration Tests for V1 Single-Store Unified Refill Decision Engine & Lifecycle."""
from datetime import date, datetime, timedelta
import pandas as pd
import pytest

from refillcare.engine.decision_types import (
    PATH_A,
    PATH_B,
    PATH_INELIGIBLE,
    PRED_HISTORICAL_MEDIAN,
    PRED_PATH_B_RECENCY,
    RefillDecision,
    STABILITY_HIGH,
    STABILITY_MEDIUM_SAFE,
    STABILITY_UNSTABLE,
    STAGE_OFFSETS,
    STATUS_DUE,
    STATUS_PENDING,
    STATUS_SUPERSEDED,
)
from refillcare.engine.reminder_lifecycle import ReminderLifecycleManager
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine


def test_path_a_regular_customer_high_stability():
    engine = UnifiedRefillDecisionEngine()
    # 7 purchases evenly spaced every 30 days
    dates = [
        date(2026, 1, 1),
        date(2026, 1, 31),
        date(2026, 3, 2),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_101",
        customer_name="John Doe",
        mobile_no="9876543210",
        item_id="MED_40MG",
        item_name="TELMISARTAN 40MG",
        dates=dates,
        as_of_date=date(2026, 7, 1),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_A
    assert dec.stability_tier in (STABILITY_HIGH, STABILITY_MEDIUM_SAFE)
    assert dec.prediction_method == PRED_HISTORICAL_MEDIAN
    assert dec.predicted_interval_days == 30
    assert dec.expected_refill_date == date(2026, 7, 30)
    assert dec.customer_item_key == "CUST_101_MED_40MG"
    assert dec.cycle_id == "RC-CUST_101-MED_40MG-20260630"


def test_path_a_unstable_customer_rejected():
    engine = UnifiedRefillDecisionEngine()
    # Highly erratic intervals: 5d, 120d, 10d, 200d, 15d, 90d
    dates = [
        date(2025, 1, 1),
        date(2025, 1, 6),
        date(2025, 5, 6),
        date(2025, 5, 16),
        date(2025, 11, 28),
        date(2025, 12, 13),
        date(2026, 3, 13),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_102",
        customer_name="Jane Erratic",
        mobile_no="9876543211",
        item_id="MED_10MG",
        item_name="ATORVASTATIN 10MG",
        dates=dates,
        as_of_date=date(2026, 4, 1),
    )

    assert dec.is_eligible is False
    assert dec.path == PATH_A
    assert dec.stability_tier == STABILITY_UNSTABLE
    assert dec.expected_refill_date is None


def test_path_b_3m_recurrence_eligible():
    engine = UnifiedRefillDecisionEngine()
    # 2 purchases across 2 distinct months within last 90 days
    dates = [
        date(2026, 5, 10),
        date(2026, 6, 12),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_201",
        customer_name="Bob Recent",
        mobile_no="9876543212",
        item_id="MED_500MG",
        item_name="METFORMIN 500MG",
        dates=dates,
        as_of_date=date(2026, 6, 20),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_B
    assert dec.prediction_method == PRED_PATH_B_RECENCY
    assert dec.predicted_interval_days == 33
    assert dec.expected_refill_date == date(2026, 7, 15)


def test_path_b_stale_recency_rejected():
    engine = UnifiedRefillDecisionEngine()
    # Last purchase > 180 days ago
    dates = [
        date(2025, 1, 10),
        date(2025, 2, 12),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_202",
        customer_name="Old Patient",
        mobile_no="9876543213",
        item_id="MED_500MG",
        item_name="METFORMIN 500MG",
        dates=dates,
        as_of_date=date(2026, 6, 20),
    )

    assert dec.is_eligible is False
    assert dec.path == PATH_INELIGIBLE
    assert "Stale recency" in dec.decision_reason


def test_7_stage_reminder_creation_and_offsets():
    engine = UnifiedRefillDecisionEngine()
    mgr = ReminderLifecycleManager()

    dates = [
        date(2026, 1, 1),
        date(2026, 1, 31),
        date(2026, 3, 2),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_301",
        customer_name="Alice Smith",
        mobile_no="9876543214",
        item_id="MED_20MG",
        item_name="ROSUVASTATIN 20MG",
        dates=dates,
        as_of_date=date(2026, 7, 1),
    )

    cycle = mgr.create_cycle_from_decision(dec)
    assert cycle is not None
    assert len(cycle.stages) == 7
    assert [s.stage_offset for s in cycle.stages] == STAGE_OFFSETS

    # Expected refill date is 2026-07-30
    exp = date(2026, 7, 30)
    assert cycle.stages[0].stage_offset == -7
    assert cycle.stages[0].target_send_date == exp - timedelta(days=7)  # 2026-07-23
    assert cycle.stages[1].stage_offset == -3
    assert cycle.stages[1].target_send_date == exp - timedelta(days=3)  # 2026-07-27
    assert cycle.stages[2].stage_offset == -1
    assert cycle.stages[2].target_send_date == exp - timedelta(days=1)  # 2026-07-29
    assert cycle.stages[3].stage_offset == 0
    assert cycle.stages[3].target_send_date == exp                      # 2026-07-30
    assert cycle.stages[4].stage_offset == 2
    assert cycle.stages[4].target_send_date == exp + timedelta(days=2)  # 2026-08-01
    assert cycle.stages[5].stage_offset == 5
    assert cycle.stages[5].target_send_date == exp + timedelta(days=5)  # 2026-08-04
    assert cycle.stages[6].stage_offset == 40
    assert cycle.stages[6].target_send_date == exp + timedelta(days=40) # 2026-09-08
    assert "Care Check-in" in cycle.stages[6].message_text


def test_repurchase_cycle_reset():
    engine = UnifiedRefillDecisionEngine()
    mgr = ReminderLifecycleManager()

    dates = [
        date(2026, 1, 1),
        date(2026, 1, 31),
        date(2026, 3, 2),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_401",
        customer_name="Carlos Repurchase",
        mobile_no="9876543215",
        item_id="MED_100MG",
        item_name="SERTRALINE 100MG",
        dates=dates,
        as_of_date=date(2026, 7, 1),
    )

    cycle = mgr.create_cycle_from_decision(dec)
    assert cycle.is_active is True

    # Patient visits store early on 2026-07-25 and repurchases
    new_purchase_date = date(2026, 7, 25)
    was_reset = mgr.process_repurchase_reset(cycle, new_purchase_date)

    assert was_reset is True
    assert cycle.is_active is False
    assert cycle.superseded_by_purchase_date == new_purchase_date
    for st in cycle.stages:
        assert st.status == STATUS_SUPERSEDED
        assert "Customer repurchased on 2026-07-25" in (st.failure_reason or "")


def test_daily_dispatch_queue_and_idempotency():
    engine = UnifiedRefillDecisionEngine()
    mgr = ReminderLifecycleManager()

    dates = [
        date(2026, 1, 1),
        date(2026, 1, 31),
        date(2026, 3, 2),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_501",
        customer_name="Diana Queue",
        mobile_no="9876543216",
        item_id="MED_5MG",
        item_name="AMLODIPINE 5MG",
        dates=dates,
        as_of_date=date(2026, 7, 1),
    )

    cycle = mgr.create_cycle_from_decision(dec)
    dec_lookup = {dec.customer_item_key: dec}

    # Query queue for Day -7 (2026-07-23)
    dispatch_df, due_stages = mgr.build_daily_dispatch_queue(
        active_cycles=[cycle],
        dispatch_date=date(2026, 7, 23),
        decisions_lookup=dec_lookup,
    )

    assert len(due_stages) == 1
    assert len(dispatch_df) == 1
    assert due_stages[0].stage_offset == -7
    assert due_stages[0].status == STATUS_DUE
    assert dispatch_df["MOBILE_NO"].iloc[0] == "919876543216"
    assert "PHARMA HUBB" in dispatch_df["message_text"].iloc[0]

    # Query same date again (Idempotency test)
    dispatch_df_2, due_stages_2 = mgr.build_daily_dispatch_queue(
        active_cycles=[cycle],
        dispatch_date=date(2026, 7, 23),
        decisions_lookup=dec_lookup,
    )
    # Stage is already marked DUE / no duplicate pending stage
    assert len(due_stages_2) == 0


def test_path_a_bulk_purchase_divergence_routes_to_review():
    engine = UnifiedRefillDecisionEngine()
    # 7 purchases every 30 days
    dates = [
        date(2026, 1, 1),
        date(2026, 1, 31),
        date(2026, 3, 2),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]
    # Latest purchase is 6 boxes of 30 tabs = 180 days supply
    quantities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0]
    packings = ["1X30", "1X30", "1X30", "1X30", "1X30", "1X30", "1X30"]

    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_BULK",
        customer_name="Bulk Buyer",
        mobile_no="9876543217",
        item_id="MED_30TAB",
        item_name="METFORMIN 500MG 1X30",
        dates=dates,
        quantities=quantities,
        packings=packings,
        as_of_date=date(2026, 7, 1),
    )

    # Must be routed to Pharmacist Review due to bulk purchase safety guardrail
    assert dec.is_eligible is False
    assert dec.path == PATH_A
    assert dec.prediction_method == PRED_HISTORICAL_MEDIAN
    assert dec.expected_refill_date is None
    assert "BULK_PURCHASE_DIVERGENCE" in dec.decision_reason


def test_path_b_6m_fallback_recurrence_eligible():
    engine = UnifiedRefillDecisionEngine()
    # 3 purchases across 3 distinct months in last 180 days (Jan, Feb, May; only May in last 90d -> fails 3M rule, passes 6M fallback)
    dates = [
        date(2026, 1, 15),  # Month 1 (Jan - 137d ago)
        date(2026, 2, 20),  # Month 2 (Feb - 101d ago)
        date(2026, 5, 10),  # Month 3 (May - 22d ago)
    ]
    quantities = [1.0, 1.0, 1.0]
    packings = ["1X30", "1X30", "1X30"]

    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_6M",
        customer_name="Six Month Recurrent",
        mobile_no="9876543218",
        item_id="MED_30TAB",
        item_name="AMLODIPINE 5MG 1X30",
        dates=dates,
        quantities=quantities,
        packings=packings,
        as_of_date=date(2026, 6, 1),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_B
    assert dec.prediction_method == PRED_PATH_B_RECENCY
    assert "6M_RECURRING" in dec.decision_reason
    assert dec.predicted_interval_days is not None
    assert dec.expected_refill_date is not None


def test_cross_path_isolation():
    engine = UnifiedRefillDecisionEngine()

    # 1. 6 purchases -> Strictly Path A, never Path B
    dates_6 = [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 3),
        date(2026, 4, 2),
        date(2026, 5, 2),
        date(2026, 6, 1),
    ]
    dec_a = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_A",
        customer_name="Path A Patient",
        mobile_no="9876543219",
        item_id="MED_1",
        item_name="MEDICINE A",
        dates=dates_6,
        as_of_date=date(2026, 6, 10),
    )
    assert dec_a.path == PATH_A
    assert dec_a.purchase_count == 6

    # 2. 5 purchases -> Strictly Path B, never Path A
    dates_5 = [
        date(2026, 2, 1),
        date(2026, 3, 3),
        date(2026, 4, 2),
        date(2026, 5, 2),
        date(2026, 6, 1),
    ]
    dec_b = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_B",
        customer_name="Path B Patient",
        mobile_no="9876543220",
        item_id="MED_2",
        item_name="MEDICINE B",
        dates=dates_5,
        as_of_date=date(2026, 6, 10),
    )
    assert dec_b.path == PATH_B
    assert dec_b.purchase_count == 5


def test_december_30th_purchase_schedule_and_dates():
    """Verify exact 7-stage reminder schedule for a purchase made on December 30th."""
    engine = UnifiedRefillDecisionEngine()
    mgr = ReminderLifecycleManager()

    # 6 prior monthly purchases + 7th purchase on 30-Dec-2026 (cadence = 30d)
    dates = [
        date(2026, 7, 3),
        date(2026, 8, 2),
        date(2026, 9, 1),
        date(2026, 10, 1),
        date(2026, 10, 31),
        date(2026, 11, 30),
        date(2026, 12, 30),
    ]
    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_DEC30",
        customer_name="Mr. Sharma",
        mobile_no="9876543210",
        item_id="MED_TELMA40",
        item_name="TELMISARTAN 40MG",
        dates=dates,
        as_of_date=date(2026, 12, 31),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_A
    assert dec.predicted_interval_days == 30
    # Expected Refill Date (Day 0) is 2027-01-29 (30 days from 30-Dec-2026)
    assert dec.expected_refill_date == date(2027, 1, 29)

    cycle = mgr.create_cycle_from_decision(dec)
    assert cycle is not None
    assert len(cycle.stages) == 7

    # Stage 1: Day -7 (2027-01-22)
    assert cycle.stages[0].stage_offset == -7
    assert cycle.stages[0].target_send_date == date(2027, 1, 22)

    # Stage 2: Day -3 (2027-01-26)
    assert cycle.stages[1].stage_offset == -3
    assert cycle.stages[1].target_send_date == date(2027, 1, 26)

    # Stage 3: Day -1 (2027-01-28)
    assert cycle.stages[2].stage_offset == -1
    assert cycle.stages[2].target_send_date == date(2027, 1, 28)

    # Stage 4: Day 0 (2027-01-29)
    assert cycle.stages[3].stage_offset == 0
    assert cycle.stages[3].target_send_date == date(2027, 1, 29)

    # Stage 5: Day +2 (2027-01-31)
    assert cycle.stages[4].stage_offset == 2
    assert cycle.stages[4].target_send_date == date(2027, 1, 31)

    # Stage 6: Day +5 (2027-02-03)
    assert cycle.stages[5].stage_offset == 5
    assert cycle.stages[5].target_send_date == date(2027, 2, 3)

    # Stage 7: Day +40 (2027-03-10)
    assert cycle.stages[6].stage_offset == 40
    assert cycle.stages[6].target_send_date == date(2027, 3, 10)


def test_repurchase_after_churn_reactivates_new_cycle():
    """Verify that a customer who lapses past Day +75 is churned, but repurchasing on April 25 reactivates fresh."""
    engine = UnifiedRefillDecisionEngine()
    mgr = ReminderLifecycleManager()

    dates = [
        date(2026, 7, 3),
        date(2026, 8, 2),
        date(2026, 9, 1),
        date(2026, 10, 1),
        date(2026, 10, 31),
        date(2026, 11, 30),
        date(2026, 12, 30),
    ]

    # Check state on 20-Apr-2027 (which is >75 days after 29-Jan-2027 due date)
    dec_churned = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_REACTIVATE",
        customer_name="Mr. Sharma",
        mobile_no="9876543210",
        item_id="MED_TELMA40",
        item_name="TELMISARTAN 40MG",
        dates=dates,
        as_of_date=date(2027, 4, 20),
    )
    # Gated as churned inactive
    assert dec_churned.is_eligible is False
    assert "EXCLUDED_CHURNED_INACTIVE" in dec_churned.decision_reason

    # On 25-Apr-2027, Mr. Sharma returns and buys a new 30-day supply!
    dates_with_new_purchase = dates + [date(2027, 4, 25)]
    dec_reactivated = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_REACTIVATE",
        customer_name="Mr. Sharma",
        mobile_no="9876543210",
        item_id="MED_TELMA40",
        item_name="TELMISARTAN 40MG",
        dates=dates_with_new_purchase,
        as_of_date=date(2027, 4, 25),
    )

    # Cleanly reactivated with new expected refill date 25-May-2027!
    assert dec_reactivated.is_eligible is True
    assert dec_reactivated.path == PATH_A
    assert dec_reactivated.purchase_count == 8
    assert dec_reactivated.last_purchase_date == date(2027, 4, 25)
    assert dec_reactivated.expected_refill_date == date(2027, 5, 25)

    new_cycle = mgr.create_cycle_from_decision(dec_reactivated)
    assert new_cycle is not None
    assert new_cycle.stages[0].target_send_date == date(2027, 5, 18)  # Day -7
    assert new_cycle.stages[3].target_send_date == date(2027, 5, 25)  # Day 0
    assert new_cycle.stages[6].target_send_date == date(2027, 7, 4)   # Day +40


def test_phone_number_sanitization_in_csv_export():
    """Verify phone numbers with trailing .0 or spaces are cleanly stripped in CSV export."""
    from reminder.reminder_engine import RefillReminderEngine

    sample_df = pd.DataFrame([
        {
            "customerId": "C100",
            "customerName": "Ramesh Kumar",
            "MOBILE_NO": "9876543210.0",
            "itemId": "I001",
            "itemName": "TELMA 40MG",
            "last_purchase_date": "2026-09-01 00:00:00",
            "estimated_days_of_supply": 30.0,
            "expected_refill_date": "2026-10-01T00:00:00",
            "target_send_date": "2026-09-24",
        }
    ])

    csv_bytes = RefillReminderEngine.build_10_column_export_csv(sample_df)
    csv_str = csv_bytes.decode("utf-8")

    assert "9876543210.0" not in csv_str
    assert "9876543210" in csv_str
    assert "2026-09-01" in csv_str
    assert " 00:00:00" not in csv_str
    assert "T00:00:00" not in csv_str


def test_partial_purchase_quantity_scaling_narasimulu_scenario():
    """Verify that when a patient usually buys 20 tabs every 23d, but buys only 10 tabs,
    the interval scales down to 10 days rather than predicting 23 days."""
    engine = UnifiedRefillDecisionEngine()

    dates = [
        date(2026, 1, 31),
        date(2026, 3, 3),
        date(2026, 4, 1),
        date(2026, 5, 1),
        date(2026, 5, 26),
        date(2026, 6, 29),
        date(2026, 8, 27),  # 59-day gap, bought only 1 strip (10 tabs)
    ]
    quantities = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    packings = ["1X10", "1X10", "1X10", "1X10", "1X10", "1X10", "1X10"]

    dec = engine.evaluate_customer_item_trajectory(
        customer_id="NARASIMULU",
        customer_name="NARASIMULU",
        mobile_no="919441113276",
        item_id="5977",
        item_name="REVLAMER 400MG TAB",
        dates=dates,
        quantities=quantities,
        packings=packings,
        as_of_date=date(2026, 9, 24),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_A
    assert dec.cadence_median == 27.0 or dec.cadence_median > 20.0
    # Must scale down from ~27d to 10d matching the 10 tablets purchased!
    assert dec.predicted_interval_days == 10
    assert dec.expected_refill_date == date(2026, 9, 6)
    assert "Partial Purchase Scaled" in dec.decision_reason


def test_multi_pack_quantity_scaling_scenario():
    """Verify that when a patient with a 30-day cadence buys 2 months of supply (60 tabs),
    the interval scales up to 60 days rather than sending a reminder at 30 days."""
    engine = UnifiedRefillDecisionEngine()

    dates = [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 3),
        date(2026, 4, 2),
        date(2026, 5, 2),
        date(2026, 6, 1),
        date(2026, 7, 1),
    ]
    # Previously bought 1 pack of 30 tabs, on latest visit bought 2 packs = 60 tabs
    quantities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    packings = ["1X30", "1X30", "1X30", "1X30", "1X30", "1X30", "1X30"]

    dec = engine.evaluate_customer_item_trajectory(
        customer_id="CUST_2MONTHS",
        customer_name="Patient TwoMonths",
        mobile_no="919876543299",
        item_id="MED_30TAB",
        item_name="TELMA 40MG TAB 1X30",
        dates=dates,
        quantities=quantities,
        packings=packings,
        as_of_date=date(2026, 7, 2),
    )

    assert dec.is_eligible is True
    assert dec.path == PATH_A
    # Scaled interval ~ 60 days (30d cadence * 2.0 ratio)
    assert dec.predicted_interval_days == 60
    assert dec.expected_refill_date == date(2026, 7, 1) + timedelta(days=60)
    assert "Multi-Pack Scaled" in dec.decision_reason



