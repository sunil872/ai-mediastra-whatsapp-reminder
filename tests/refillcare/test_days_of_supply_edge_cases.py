"""Comprehensive edge-case validation tests for Days-of-Supply and consumption rate calculations.

Validates all 12 operational edge-case scenarios and Balraju-style customer patterns
to ensure deterministic, safe behavior across pharmacy transactions.
"""

from datetime import date, timedelta
import pytest
import pandas as pd
import numpy as np

from refillcare.data.packing import parse_pack_units, compute_total_units_purchased, enrich_total_units_purchased
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_daily_consumption,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
    compute_expanding_consumption_rates,
)
from reminder.scheduler import (
    calculate_expected_refill_date,
    evaluate_refill_eligibility,
    RefillReminderScheduler,
    format_reminder_message,
)
from refillcare.models.hybrid_strategy import evaluate_hybrid_eligibility


# ==============================================================================
# SCENARIO 1: NORMAL MONTHLY PURCHASE
# ==============================================================================
def test_scenario_1_normal_monthly_purchase():
    """Scenario 1: 4 × 1X15 = 60 tablets, historical rate ≈ 2 tabs/day -> supply ≈ 30 days."""
    history = [
        {"invoice_date": "2026-06-01", "quantity": 4, "packing": "1X15"},  # 60 units
        {"invoice_date": "2026-07-01", "quantity": 4, "packing": "1X15"},  # 30 days later, bought next 60 units
    ]
    rate_res = calculate_historical_consumption_rate(history)
    assert rate_res["status"] == "valid"
    assert pytest.approx(rate_res["historical_consumption_rate"], rel=1e-2) == 2.0

    # Latest purchase on 2026-07-01: 4 strips of 1X15 = 60 units
    total_units = compute_total_units_purchased(4, "1X15")
    assert total_units == 60.0

    dos_res = calculate_estimated_days_of_supply(total_units, rate_res["historical_consumption_rate"])
    assert dos_res["status"] == "valid"
    assert pytest.approx(dos_res["estimated_days_of_supply"], rel=1e-2) == 30.0

    # Refill Date: 2026-07-01 + 30 days = 2026-07-31
    exp_res = calculate_expected_refill_date("2026-07-01", dos_res["estimated_days_of_supply"])
    assert exp_res["is_valid"] is True
    assert exp_res["expected_refill_date"] == date(2026, 7, 31)

    # Primary Reminder Date: 2026-07-31 - 2 days = 2026-07-29
    target_res = calculate_target_refill_interval(dos_res["estimated_days_of_supply"], buffer_days=2.0)
    assert target_res["status"] == "valid"
    assert target_res["target_refill_interval"] == 28.0
    rem_date = date(2026, 7, 1) + timedelta(days=int(round(target_res["target_refill_interval"])))
    assert rem_date == date(2026, 7, 29)


# ==============================================================================
# SCENARIO 2: BULK PURCHASE
# ==============================================================================
def test_scenario_2_bulk_purchase():
    """Scenario 2: 8 × 1X15 = 120 tablets, rate ≈ 2 tabs/day -> supply ≈ 60 days."""
    total_units = compute_total_units_purchased(8, "1X15")
    assert total_units == 120.0

    daily_consumption = 2.0
    dos_res = calculate_estimated_days_of_supply(total_units, daily_consumption)
    assert dos_res["status"] == "valid"
    assert pytest.approx(dos_res["estimated_days_of_supply"], rel=1e-2) == 60.0

    # Refill Date: 2026-08-01 + 60 days = 2026-09-30
    exp_res = calculate_expected_refill_date("2026-08-01", dos_res["estimated_days_of_supply"])
    assert exp_res["is_valid"] is True
    assert exp_res["expected_refill_date"] == date(2026, 9, 30)

    # Primary Reminder Date: 2026-09-30 - 2 days = 2026-09-28
    target_res = calculate_target_refill_interval(dos_res["estimated_days_of_supply"], buffer_days=2.0)
    assert target_res["target_refill_interval"] == 58.0


# ==============================================================================
# SCENARIO 3: PARTIAL / TOP-UP PURCHASE
# ==============================================================================
def test_scenario_3_partial_topup_purchase():
    """Scenario 3: 1 × 1X15 = 15 tablets, rate ≈ 2 tabs/day -> supply ≈ 7.5 days."""
    total_units = compute_total_units_purchased(1, "1X15")
    assert total_units == 15.0

    daily_consumption = 2.0
    dos_res = calculate_estimated_days_of_supply(total_units, daily_consumption)
    assert dos_res["status"] == "valid"
    assert pytest.approx(dos_res["estimated_days_of_supply"], rel=1e-2) == 7.5

    # Refill Date: 2026-08-01 + 8 days (round(7.5) = 8) = 2026-08-09
    exp_res = calculate_expected_refill_date("2026-08-01", dos_res["estimated_days_of_supply"])
    assert exp_res["is_valid"] is True
    assert exp_res["days_offset"] == 8
    assert exp_res["expected_refill_date"] == date(2026, 8, 9)

    # Primary Reminder Date with buffer: max(1, 7.5 - 2.0) = 5.5 days offset (round(5.5) = 6)
    target_res = calculate_target_refill_interval(dos_res["estimated_days_of_supply"], buffer_days=2.0)
    assert target_res["target_refill_interval"] == 5.5


# ==============================================================================
# SCENARIO 4: DIFFERENT PACK FORMATS
# ==============================================================================
def test_scenario_4_pack_formats_parsing():
    """Scenario 4: Verify 1X15, 1X10, 10 TAB, 30 CAP, 100ML, 1PC parse deterministically."""
    # 1X15
    assert parse_pack_units("1X15") == 15.0
    assert compute_total_units_purchased(2, "1X15") == 30.0

    # 1X10
    assert parse_pack_units("1X10") == 10.0
    assert compute_total_units_purchased(3, "1X10") == 30.0

    # 10 TAB
    assert parse_pack_units("10 TAB") == 10.0
    assert parse_pack_units("10 TABLETS") == 10.0
    assert compute_total_units_purchased(3, "10 TAB") == 30.0

    # 30 CAP
    assert parse_pack_units("30 CAP") == 30.0
    assert parse_pack_units("30 CAPSULES") == 30.0
    assert compute_total_units_purchased(1, "30 CAP") == 30.0

    # 100ML
    assert parse_pack_units("100ML") == 100.0
    assert compute_total_units_purchased(1, "100ML") == 100.0

    # 1PC
    assert parse_pack_units("1PC") == 1.0
    assert parse_pack_units("1 PC") == 1.0
    assert compute_total_units_purchased(5, "1PC") == 5.0


# ==============================================================================
# SCENARIO 5: MISSING OR UNPARSEABLE PACKING
# ==============================================================================
def test_scenario_5_missing_or_unparseable_packing():
    """Scenario 5: Invalid/missing packing safely returns None and does not generate false supply."""
    assert parse_pack_units(None) is None
    assert parse_pack_units("") is None
    assert parse_pack_units("UNKNOWN_PACK") is None
    assert parse_pack_units("10X4.4GM") is None  # Ambiguous multi-dimensional

    # Total units must be None
    assert compute_total_units_purchased(4, "UNKNOWN_PACK") is None
    assert compute_total_units_purchased(4, None) is None

    # Supply calculation must return unavailable
    dos_res = calculate_estimated_days_of_supply(None, 2.0)
    assert dos_res["status"] == "unavailable"
    assert dos_res["estimated_days_of_supply"] is None


# ==============================================================================
# SCENARIO 6: ZERO AND NEGATIVE QUANTITIES (RETURNS)
# ==============================================================================
def test_scenario_6_zero_and_negative_quantities_returns():
    """Scenario 6: Quantity = 0 and negative returns are not treated as positive consumption."""
    # Qty 0
    assert compute_total_units_purchased(0, "1X15") == 0.0
    dos_zero = calculate_estimated_days_of_supply(0.0, 2.0)
    assert dos_zero["status"] == "valid"
    assert dos_zero["estimated_days_of_supply"] == 0.0

    # Qty -2 (Returns)
    assert compute_total_units_purchased(-2, "1X15") == -30.0
    dos_neg = calculate_estimated_days_of_supply(-30.0, 2.0)
    assert dos_neg["status"] == "unavailable"
    assert dos_neg["estimated_days_of_supply"] is None
    assert "negative" in dos_neg["reason"]

    # In history, returns must NOT be counted as consumption
    history_with_return = [
        {"invoice_date": "2026-01-01", "quantity": 4, "packing": "1X15"},   # +60 units
        {"invoice_date": "2026-01-15", "quantity": -1, "packing": "1X15"},  # -15 return (must be skipped)
        {"invoice_date": "2026-01-31", "quantity": 4, "packing": "1X15"},   # +60 units (30d elapsed)
    ]
    rate_res = calculate_historical_consumption_rate(history_with_return)
    assert rate_res["status"] == "valid"
    assert rate_res["usable_purchase_count"] == 2  # Only the 2 positive visits
    assert rate_res["total_consumed_units"] == 60.0
    assert pytest.approx(rate_res["historical_consumption_rate"], rel=1e-2) == 2.0


# ==============================================================================
# SCENARIO 7: INSUFFICIENT HISTORY (COLD START)
# ==============================================================================
def test_scenario_7_insufficient_history_cold_start():
    """Scenario 7: Customers with 1 purchase return insufficient_data and are flagged as ineligible."""
    single_purchase = [
        {"invoice_date": "2026-08-01", "quantity": 4, "packing": "1X15"},
    ]
    rate_res = calculate_historical_consumption_rate(single_purchase)
    assert rate_res["status"] == "insufficient_data"
    assert rate_res["historical_consumption_rate"] is None
    assert rate_res["usable_purchase_count"] == 1

    elig = evaluate_refill_eligibility({"purchase_count_so_far": 1, "predicted_days_until_refill": 30.0})
    assert elig["is_eligible"] is False
    assert elig["history_quality"] == "ineligible"
    assert "Cold-start" in elig["reason"]


# ==============================================================================
# SCENARIO 8: IRREGULAR PURCHASE HISTORY
# ==============================================================================
def test_scenario_8_irregular_purchase_history():
    """Scenario 8: Irregular purchase cadence is handled safely by Phase 17D regularity gating."""
    # Irregular intervals with large drift
    record = {
        "customerId": "CUST_IRREGULAR",
        "itemId": "ITEM_001",
        "purchase_count_so_far": 6,
        "historical_interval_median": 30.0,
        "historical_interval_norm_mad": 0.65,  # > 0.50 (Irregular)
        "cadence_drift": 18.0,                 # > 10.0 (High drift)
    }
    decision = evaluate_hybrid_eligibility(record)
    assert decision["is_eligible"] is False
    assert decision["route"] == "rejected"
    assert decision["confidence"] == "REJECTED"
    assert "Irregular cadence" in decision["rejection_reason"]


# ==============================================================================
# SCENARIO 9: MISSING AND INVALID DATES
# ==============================================================================
def test_scenario_9_missing_and_invalid_dates():
    """Scenario 9: Missing or unparseable dates safely fail without crashing."""
    # Null date
    res_none = calculate_expected_refill_date(None, 30.0)
    assert res_none["is_valid"] is False
    assert res_none["status"] == "invalid_prediction"
    assert res_none["expected_refill_date"] is None

    # Invalid string
    res_inv = calculate_expected_refill_date("NOT_A_DATE", 30.0)
    assert res_inv["is_valid"] is False
    assert res_inv["expected_refill_date"] is None

    # History with null dates
    bad_history = [
        {"invoice_date": None, "quantity": 4, "packing": "1X15"},
        {"invoice_date": "2026-08-01", "quantity": 4, "packing": "1X15"},
    ]
    rate_res = calculate_historical_consumption_rate(bad_history)
    assert rate_res["status"] == "insufficient_data"


# ==============================================================================
# SCENARIO 10: REFILL DATE CALCULATION
# ==============================================================================
def test_scenario_10_refill_date_exact_addition():
    """Scenario 10: expected_refill_date = latest_purchase_date + round(estimated_days_of_supply)."""
    p_date = date(2026, 8, 1)
    dos = 30.0
    res = calculate_expected_refill_date(p_date, dos)
    assert res["is_valid"] is True
    assert res["expected_refill_date"] == date(2026, 8, 31)

    # Fractional days (e.g. 15.4 days -> +15 days, 15.6 days -> +16 days)
    res_frac1 = calculate_expected_refill_date(p_date, 15.4)
    assert res_frac1["expected_refill_date"] == date(2026, 8, 16)

    res_frac2 = calculate_expected_refill_date(p_date, 15.6)
    assert res_frac2["expected_refill_date"] == date(2026, 8, 17)


# ==============================================================================
# SCENARIO 11: PRIMARY REMINDER DATE (2-DAY BUFFER)
# ==============================================================================
def test_scenario_11_primary_reminder_date_two_day_buffer():
    """Scenario 11: reminder_date = expected_refill_date - 2 days (never alters expected_refill_date)."""
    p_date = date(2026, 8, 1)
    dos = 30.0
    exp_res = calculate_expected_refill_date(p_date, dos)
    target_res = calculate_target_refill_interval(dos, buffer_days=2.0)

    expected_refill = exp_res["expected_refill_date"]
    reminder_date = p_date + timedelta(days=int(round(target_res["target_refill_interval"])))

    assert expected_refill == date(2026, 8, 31)
    assert reminder_date == date(2026, 8, 29)
    assert (expected_refill - reminder_date).days == 2


# ==============================================================================
# SCENARIO 12: MULTI-STAGE REMINDERS SYNCHRONIZATION
# ==============================================================================
def test_scenario_12_multi_stage_reminders_all_anchored_to_expected_refill_date():
    """Scenario 12: All 6 stages (-7, -3, -1, 0, +2, +5) are anchored to the same expected refill date."""
    scheduler = RefillReminderScheduler()
    record = {
        "customerId": "CUST_STAGE_TEST",
        "itemId": "MED_101",
        "customerName": "Test Patient",
        "itemName": "Telmisartan 40mg",
        "MOBILE_NO": "9876543210",
        "latest_purchase_date": "2026-08-01",
        "predicted_days_until_refill": 30.0,
        "estimated_days_of_supply": 30.0,
        "purchase_count_so_far": 5,
    }
    sched_result = scheduler.schedule_refill_cycle(record)
    assert sched_result["scheduled"] is True
    assert sched_result["reminder_count"] == 6

    reminders = sched_result["reminders"]
    stage_dates = {r.reminder_stage: r.reminder_date for r in reminders}
    expected_refill = "2026-08-31"

    assert stage_dates[-7] == "2026-08-24"  # 31 - 7
    assert stage_dates[-3] == "2026-08-28"  # 31 - 3
    assert stage_dates[-1] == "2026-08-30"  # 31 - 1
    assert stage_dates[0] == "2026-08-31"   # 31 + 0
    assert stage_dates[2] == "2026-09-02"   # 31 + 2
    assert stage_dates[5] == "2026-09-05"   # 31 + 5


# ==============================================================================
# BALRAJU-STYLE REAL PHARMACY PATTERN
# ==============================================================================
def test_balraju_style_pharmacy_customer_scenario():
    """Balraju-style pattern: packing=1X15, quantity=4, total units=60, inferred rate ≈ 2/day, DOS ≈ 30 days."""
    # Historical purchases over 3 visits at ~30 day intervals
    balraju_history = [
        {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15", "itemName": "Glycomet GP 1"},
        {"invoice_date": "2026-05-01", "quantity": 4, "packing": "1X15", "itemName": "Glycomet GP 1"},
        {"invoice_date": "2026-05-31", "quantity": 4, "packing": "1X15", "itemName": "Glycomet GP 1"},
    ]

    # 1. Historical consumption rate
    rate_res = calculate_historical_consumption_rate(balraju_history)
    assert rate_res["status"] == "valid"
    assert rate_res["usable_purchase_count"] == 3
    assert rate_res["total_elapsed_days"] == 60  # 60 days total elapsed
    assert rate_res["total_consumed_units"] == 120.0  # 2 cycles of 60 units consumed
    assert pytest.approx(rate_res["historical_consumption_rate"], rel=1e-2) == 2.0  # exactly 2.0 units/day

    # 2. Latest purchase on 2026-05-31: 4 strips of 1X15 = 60 units
    latest_units = compute_total_units_purchased(4, "1X15")
    assert latest_units == 60.0

    # 3. Estimated Days of Supply
    dos_res = calculate_estimated_days_of_supply(latest_units, rate_res["historical_consumption_rate"])
    assert dos_res["status"] == "valid"
    assert pytest.approx(dos_res["estimated_days_of_supply"], rel=1e-2) == 30.0

    # 4. Expected Refill Date: 2026-05-31 + 30 days = 2026-06-30
    exp_res = calculate_expected_refill_date("2026-05-31", dos_res["estimated_days_of_supply"])
    assert exp_res["is_valid"] is True
    assert exp_res["expected_refill_date"] == date(2026, 6, 30)

    # 5. Primary Reminder Date: 2026-06-30 - 2 days = 2026-06-28
    target_res = calculate_target_refill_interval(dos_res["estimated_days_of_supply"], buffer_days=2.0)
    assert target_res["target_refill_interval"] == 28.0
    primary_rem_date = date(2026, 5, 31) + timedelta(days=int(round(target_res["target_refill_interval"])))
    assert primary_rem_date == date(2026, 6, 28)
