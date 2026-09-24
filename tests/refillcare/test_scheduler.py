"""Unit tests for Phase 5 RefillCare Reminder Scheduling Engine."""

from datetime import date
import pytest
import pandas as pd
import numpy as np

from reminder.scheduler import (
    REMINDER_STAGES,
    ReminderRecord,
    RefillReminderScheduler,
    evaluate_refill_eligibility,
    format_reminder_message,
    calculate_expected_refill_date,
    get_daily_due_reminders,
)


def test_calculate_expected_refill_date_standard_and_zero_days():
    """Verify expected refill date computation with normal and zero-day intervals."""
    # Standard 30-day interval
    res = calculate_expected_refill_date("2026-08-10", 30.0)
    assert res["is_valid"] is True
    assert res["latest_purchase_date"] == date(2026, 8, 10)
    assert res["expected_refill_date"] == date(2026, 9, 9)
    assert res["days_offset"] == 30

    # Zero-day interval (same-day prediction)
    res_zero = calculate_expected_refill_date("2026-08-10", 0.0)
    assert res_zero["is_valid"] is True
    assert res_zero["expected_refill_date"] == date(2026, 8, 10)
    assert res_zero["days_offset"] == 0

    # Indian format string DD/MM/YYYY
    res_ind = calculate_expected_refill_date("10/08/2026", 15.0)
    assert res_ind["is_valid"] is True
    assert res_ind["latest_purchase_date"] == date(2026, 8, 10)
    assert res_ind["expected_refill_date"] == date(2026, 8, 25)


def test_calculate_expected_refill_date_invalid_inputs():
    """Verify invalid/negative/null predictions are rejected safely."""
    # Negative interval
    res_neg = calculate_expected_refill_date("2026-08-10", -5.0)
    assert res_neg["is_valid"] is False
    assert res_neg["status"] == "invalid_prediction"

    # NaN / Inf
    assert calculate_expected_refill_date("2026-08-10", np.nan)["is_valid"] is False
    assert calculate_expected_refill_date("2026-08-10", float("inf"))["is_valid"] is False

    # Invalid date string
    assert calculate_expected_refill_date("invalid-date", 30.0)["is_valid"] is False


def test_six_reminder_stages_and_offsets():
    """Verify all six reminder stages and their exact calendar offsets."""
    assert REMINDER_STAGES == [-7, -3, -1, 0, 2, 5]

    scheduler = RefillReminderScheduler()
    event = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "John Doe",
        "MOBILE_NO": "9876543210",
        "itemName": "Medicine A",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }

    res = scheduler.schedule_refill_cycle(event)
    assert res["scheduled"] is True
    assert len(res["reminders"]) == 6

    rem_map = {r.reminder_stage: r.reminder_date for r in res["reminders"]}
    assert rem_map[-7] == "2026-09-02"
    assert rem_map[-3] == "2026-09-06"
    assert rem_map[-1] == "2026-09-08"
    assert rem_map[0] == "2026-09-09"
    assert rem_map[2] == "2026-09-11"
    assert rem_map[5] == "2026-09-14"


def test_exact_reminder_message_copy():
    """Verify dynamic message text matches client-specified templates exactly."""
    exp_date = "2026-09-09"  # Formats to 09-09-2026

    msg_m7 = format_reminder_message(-7, exp_date)
    assert msg_m7 == "Your regular medicine refill may be due in about 7 days, around 09-09-2026."

    msg_m3 = format_reminder_message(-3, exp_date)
    assert msg_m3 == "Your regular medicine refill may be due in about 3 days, around 09-09-2026."

    msg_m1 = format_reminder_message(-1, exp_date)
    assert msg_m1 == "Your regular medicine refill may be due tomorrow, 09-09-2026."

    msg_0 = format_reminder_message(0, exp_date)
    assert msg_0 == "Your regular medicine refill may be due today, 09-09-2026."

    msg_p2 = format_reminder_message(2, exp_date)
    assert msg_p2 == "Your expected refill date was 09-09-2026. If you still need your medicine, please contact us."

    msg_p5 = format_reminder_message(5, exp_date)
    assert msg_p5 == "This is a follow-up regarding your expected medicine refill on 09-09-2026. Please contact us if you still need your medicine."


def test_customer_item_identity_and_shared_phone():
    """Verify shared phone numbers do not merge distinct customer accounts."""
    scheduler = RefillReminderScheduler()

    # Customer 1 (Father) on phone 9999999999
    event_c1 = {
        "customerId": "CUST_FATHER",
        "itemId": "M001",
        "customerName": "Father",
        "MOBILE_NO": "9999999999",
        "itemName": "Diabetes Med",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }

    # Customer 2 (Mother) on SAME phone 9999999999
    event_c2 = {
        "customerId": "CUST_MOTHER",
        "itemId": "M001",
        "customerName": "Mother",
        "MOBILE_NO": "9999999999",
        "itemName": "Diabetes Med",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 20.0,
    }

    res1 = scheduler.schedule_refill_cycle(event_c1)
    res2 = scheduler.schedule_refill_cycle(event_c2)

    assert res1["scheduled"] is True
    assert res2["scheduled"] is True

    c1_rems = scheduler.get_reminders_for_customer("CUST_FATHER")
    c2_rems = scheduler.get_reminders_for_customer("CUST_MOTHER")

    assert len(c1_rems) == 6
    assert len(c2_rems) == 6
    assert c1_rems[0].expected_refill_date == "2026-09-09"
    assert c2_rems[0].expected_refill_date == "2026-08-30"


def test_multiple_medicines_same_customer_isolation():
    """Verify multiple medicines for the same patient have separate cycles."""
    scheduler = RefillReminderScheduler()

    event_med1 = {
        "customerId": "C001",
        "itemId": "MED_BP",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "BP Tablet",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }

    event_med2 = {
        "customerId": "C001",
        "itemId": "MED_THYROID",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Thyroid Tablet",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 60.0,
    }

    scheduler.schedule_refill_cycle(event_med1)
    scheduler.schedule_refill_cycle(event_med2)

    med1_rems = scheduler.get_reminders_for_customer("C001", "MED_BP")
    med2_rems = scheduler.get_reminders_for_customer("C001", "MED_THYROID")

    assert len(med1_rems) == 6
    assert len(med2_rems) == 6
    assert med1_rems[0].expected_refill_date == "2026-09-09"
    assert med2_rems[0].expected_refill_date == "2026-10-09"


def test_duplicate_protection_idempotency():
    """Calling schedule_refill_cycle multiple times must be deterministic and idempotent."""
    scheduler = RefillReminderScheduler()

    event = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }

    # Run twice
    res1 = scheduler.schedule_refill_cycle(event)
    res2 = scheduler.schedule_refill_cycle(event)

    assert len(scheduler._reminders) == 6
    all_rems = scheduler.get_reminders_for_customer("C001", "M001")
    assert len(all_rems) == 6


def test_cycle_reset_on_new_purchase():
    """Verify that a new purchase cancels pending reminders from the prior cycle."""
    scheduler = RefillReminderScheduler()

    # Initial purchase on 2026-08-10 (Exp: 2026-09-09)
    event1 = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }
    scheduler.schedule_refill_cycle(event1)

    # All 6 reminders scheduled initially
    rems_initial = scheduler.get_reminders_for_customer("C001", "M001")
    assert all(r.status == "scheduled" for r in rems_initial)

    # Customer repurchases early on 2026-09-08
    event2 = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "invoice_date": "2026-09-08",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }
    res2 = scheduler.schedule_refill_cycle(event2)

    assert res2["cancelled_prior_reminders"] == 6
    assert res2["expected_refill_date"] == "2026-10-08"

    # Prior cycle reminders must be cancelled
    all_rems = scheduler.get_reminders_for_customer("C001", "M001")
    assert len(all_rems) == 12

    old_cycle_rems = [r for r in all_rems if r.expected_refill_date == "2026-09-09"]
    new_cycle_rems = [r for r in all_rems if r.expected_refill_date == "2026-10-08"]

    assert len(old_cycle_rems) == 6
    assert all(r.status == "cancelled" for r in old_cycle_rems)

    assert len(new_cycle_rems) == 6
    assert all(r.status == "scheduled" for r in new_cycle_rems)


def test_cold_start_exclusion():
    """Verify single purchase / cold-start records are flagged as ineligible."""
    scheduler = RefillReminderScheduler()

    cold_start_event = {
        "customerId": "C_NEW",
        "itemId": "M_NEW",
        "customerName": "First Time Buyer",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine X",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 1,  # Cold start
        "predicted_days_until_refill": 30.0,
    }

    res = scheduler.schedule_refill_cycle(cold_start_event)
    assert res["scheduled"] is False
    assert res["status"] == "ineligible"
    assert "Cold-start" in res["reason"]
    assert len(res["reminders"]) == 0


def test_daily_due_reminders_filtering():
    """Verify get_due_reminders correctly filters only scheduled reminders for target date."""
    scheduler = RefillReminderScheduler()

    event = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 4,
        "predicted_days_until_refill": 30.0,
    }
    scheduler.schedule_refill_cycle(event)

    # On 2026-09-02, Stage -7 should be due
    due_02 = scheduler.get_due_reminders("2026-09-02")
    assert len(due_02) == 1
    assert due_02[0].reminder_stage == -7
    assert due_02[0].status == "scheduled"

    # On 2026-09-09, Stage 0 should be due
    due_09 = scheduler.get_due_reminders("2026-09-09")
    assert len(due_09) == 1
    assert due_09[0].reminder_stage == 0

    # On non-scheduled date (e.g. 2026-09-03), 0 should be due
    due_none = scheduler.get_due_reminders("2026-09-03")
    assert len(due_none) == 0

    # Standalone functional helper test
    due_helper = get_daily_due_reminders(list(scheduler._reminders.values()), "2026-09-02")
    assert len(due_helper) == 1
    assert due_helper[0].reminder_stage == -7


def test_scheduler_to_dataframe():
    """Verify exporting scheduled records to a clean Pandas DataFrame."""
    scheduler = RefillReminderScheduler()
    df_empty = scheduler.to_dataframe()
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0

    event = {
        "customerId": "C001",
        "itemId": "M001",
        "customerName": "Patient A",
        "MOBILE_NO": "9849012345",
        "itemName": "Medicine A",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 3,
        "predicted_days_until_refill": 30.0,
    }
    scheduler.schedule_refill_cycle(event)

    df_populated = scheduler.to_dataframe()
    assert len(df_populated) == 6
    assert "reminder_id" in df_populated.columns
    assert "expected_refill_date" in df_populated.columns
    assert "message" in df_populated.columns
