"""RefillCare Reminder Eligibility & Scheduling Engine.

This module implements the business logic layer that converts XGBoost refill
predictions into deterministic, duplicate-free, multi-stage reminder schedules
with automated cycle resets upon new customer purchases.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union
import math
import pandas as pd
import numpy as np


# Exact 6 reminder stages specified for RefillCare
REMINDER_STAGES: List[int] = [-7, -3, -1, 0, 2, 5]


@dataclass
class ReminderRecord:
    """Individual scheduled reminder instance."""

    reminder_id: str
    customerId: str
    itemId: str
    customerName: str
    MOBILE_NO: str
    itemName: str
    latest_purchase_date: str  # YYYY-MM-DD
    predicted_interval_days: float
    expected_refill_date: str  # YYYY-MM-DD
    reminder_stage: int  # -7, -3, -1, 0, 2, 5
    reminder_date: str  # YYYY-MM-DD
    message: str
    status: str = "scheduled"  # "scheduled", "cancelled", "completed"
    history_quality: str = "medium_history"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert reminder record to standard dictionary."""
        return asdict(self)


def _parse_to_date(d: Union[str, date, datetime, pd.Timestamp]) -> date:
    """Safely parse input date into a datetime.date object.

    Handles ISO formats (YYYY-MM-DD), Indian formats (DD/MM/YYYY, DD-MM-YYYY),
    as well as Timestamp and datetime objects.
    """
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        s = d.strip()
        try:
            if len(s) >= 10 and s[4] in ("-", "/") and s[:4].isdigit():
                return pd.to_datetime(s, dayfirst=False).date()
            else:
                return pd.to_datetime(s, dayfirst=True).date()
        except Exception as e:
            raise ValueError(f"Unable to parse date string: {d}") from e
    raise TypeError(f"Unsupported date type: {type(d)}")


def calculate_expected_refill_date(
    latest_purchase_date: Union[str, date, datetime, pd.Timestamp],
    predicted_interval_days: float,
) -> Dict[str, Any]:
    """Calculate expected refill date with strict validation.

    Formula:
    expected_refill_date = latest_purchase_date + round(predicted_interval_days)

    Args:
        latest_purchase_date: Date of the canonical purchase visit.
        predicted_interval_days: Model-predicted days until the next refill.

    Returns:
        Dict with status, parsed dates, and offset.
    """
    if predicted_interval_days is None or math.isnan(predicted_interval_days) or math.isinf(predicted_interval_days):
        return {
            "is_valid": False,
            "status": "invalid_prediction",
            "reason": "Predicted interval is NaN, infinite, or null.",
            "expected_refill_date": None,
        }

    if predicted_interval_days < 0:
        return {
            "is_valid": False,
            "status": "invalid_prediction",
            "reason": f"Negative predicted interval ({predicted_interval_days} days).",
            "expected_refill_date": None,
        }

    try:
        p_date = _parse_to_date(latest_purchase_date)
    except Exception as e:
        return {
            "is_valid": False,
            "status": "invalid_prediction",
            "reason": f"Invalid purchase date: {e}",
            "expected_refill_date": None,
        }

    # Handle zero-day predictions explicitly
    days_offset = int(round(float(predicted_interval_days)))
    exp_date = p_date + timedelta(days=days_offset)

    return {
        "is_valid": True,
        "status": "valid",
        "latest_purchase_date": p_date,
        "predicted_interval_days": float(predicted_interval_days),
        "days_offset": days_offset,
        "expected_refill_date": exp_date,
    }


def format_reminder_message(
    stage: int,
    expected_refill_date: Union[str, date, datetime, pd.Timestamp],
) -> str:
    """Generate exact client-approved dynamic reminder message copy.

    Templates format date as DD-MM-YYYY.

    Stages:
    -7: "Your regular medicine refill may be due in about 7 days, around DD-MM-YYYY."
    -3: "Your regular medicine refill may be due in about 3 days, around DD-MM-YYYY."
    -1: "Your regular medicine refill may be due tomorrow, DD-MM-YYYY."
     0: "Your regular medicine refill may be due today, DD-MM-YYYY."
    +2: "Your expected refill date was DD-MM-YYYY. If you still need your medicine, please contact us."
    +5: "This is a follow-up regarding your expected medicine refill on DD-MM-YYYY. Please contact us if you still need your medicine."
    """
    exp_d = _parse_to_date(expected_refill_date)
    formatted_date = exp_d.strftime("%d-%m-%Y")

    if stage == -7:
        return f"Your regular medicine refill may be due in about 7 days, around {formatted_date}."
    elif stage == -3:
        return f"Your regular medicine refill may be due in about 3 days, around {formatted_date}."
    elif stage == -1:
        return f"Your regular medicine refill may be due tomorrow, {formatted_date}."
    elif stage == 0:
        return f"Your regular medicine refill may be due today, {formatted_date}."
    elif stage == 2:
        return f"Your expected refill date was {formatted_date}. If you still need your medicine, please contact us."
    elif stage == 5:
        return f"This is a follow-up regarding your expected medicine refill on {formatted_date}. Please contact us if you still need your medicine."
    else:
        if stage < 0:
            return f"Your regular medicine refill may be due in {abs(stage)} days, around {formatted_date}."
        else:
            return f"Your expected refill date was {formatted_date}. Please contact us if you still need your medicine."


def evaluate_refill_eligibility(
    record: Union[Dict[str, Any], pd.Series],
    min_purchase_count: int = 2,
) -> Dict[str, Any]:
    """Determine whether a customer-medicine purchase history is eligible for reminders.

    Conceptual Statuses:
    - eligible: Multi-purchase history with valid prediction.
    - ineligible: Cold-start single purchase (<2 purchases).
    - invalid_prediction: Null, negative, or invalid model output.

    History Quality Indicator (Rule-based, not statistical probability):
    - high_history: purchase_count >= 5, or (>=3 with recurring pattern)
    - medium_history: purchase_count 3-4, or (=2 with recurring pattern)
    - low_history: purchase_count == 2
    - ineligible: purchase_count < 2
    """
    if isinstance(record, pd.Series):
        rec = record.to_dict()
    else:
        rec = dict(record)

    # 1. Check purchase count / cold start
    purchase_count = rec.get("purchase_count_so_far", rec.get("purchase_seq", 1))
    try:
        purchase_count = int(purchase_count) if pd.notna(purchase_count) else 1
    except (ValueError, TypeError):
        purchase_count = 1

    if purchase_count < min_purchase_count:
        return {
            "status": "ineligible",
            "is_eligible": False,
            "reason": f"Cold-start history: purchase count is {purchase_count} (< {min_purchase_count}).",
            "history_quality": "ineligible",
            "purchase_count": purchase_count,
        }

    # Phase 17D hybrid rejection (prediction layer)
    pred_status = str(rec.get("prediction_status", "") or "").lower()
    refill_conf = str(rec.get("refill_confidence", "") or "").upper()
    if pred_status == "rejected" or refill_conf in ("REJECTED", "INVALID"):
        reason = rec.get("rejection_reason") or "Rejected by Phase 17D hybrid eligibility."
        return {
            "status": "ineligible",
            "is_eligible": False,
            "reason": str(reason),
            "history_quality": "ineligible",
            "purchase_count": purchase_count,
        }

    # 2. Check prediction validity
    pred_days = rec.get("predicted_days_until_refill", rec.get("predicted_interval_days"))
    if pred_days is None or pd.isna(pred_days) or math.isinf(pred_days):
        return {
            "status": "invalid_prediction",
            "is_eligible": False,
            "reason": "Predicted interval is missing, null, or infinite.",
            "history_quality": "invalid_prediction",
            "purchase_count": purchase_count,
        }

    try:
        pred_val = float(pred_days)
    except (ValueError, TypeError):
        return {
            "status": "invalid_prediction",
            "is_eligible": False,
            "reason": f"Could not parse predicted interval value: {pred_days}",
            "history_quality": "invalid_prediction",
            "purchase_count": purchase_count,
        }

    if pred_val < 0:
        return {
            "status": "invalid_prediction",
            "is_eligible": False,
            "reason": f"Negative predicted interval: {pred_val} days.",
            "history_quality": "invalid_prediction",
            "purchase_count": purchase_count,
        }

    # 3. Check date validity
    p_date_raw = rec.get("invoice_date", rec.get("latest_purchase_date", rec.get("current_purchase_date")))
    if p_date_raw is None or pd.isna(p_date_raw):
        return {
            "status": "invalid_prediction",
            "is_eligible": False,
            "reason": "Missing invoice/purchase date.",
            "history_quality": "invalid_prediction",
            "purchase_count": purchase_count,
        }

    # 4. History Quality Assignment
    is_recurring = int(rec.get("is_recurring_history", 0) or 0)
    if purchase_count >= 5 or (purchase_count >= 3 and is_recurring == 1):
        quality = "high_history"
    elif purchase_count >= 3 or (purchase_count == 2 and is_recurring == 1):
        quality = "medium_history"
    else:
        quality = "low_history"

    return {
        "status": "eligible",
        "is_eligible": True,
        "reason": "History meets eligibility criteria with valid prediction.",
        "history_quality": quality,
        "purchase_count": purchase_count,
        "predicted_interval_days": pred_val,
        "parsed_purchase_date": p_date_raw,
    }


class RefillReminderScheduler:
    """In-memory scheduling engine for RefillCare reminder cycles."""

    def __init__(self, stages: Optional[List[int]] = None):
        """Initialize scheduler with target reminder stages."""
        self.stages = stages if stages is not None else list(REMINDER_STAGES)
        # Store records keyed by reminder_id: customerId_itemId_expectedDate_stage
        self._reminders: Dict[str, ReminderRecord] = {}
        # Track active cycle by (customerId, itemId) -> expected_refill_date
        self._active_cycles: Dict[str, str] = {}

    @staticmethod
    def _build_reminder_id(customer_id: str, item_id: str, exp_date: str, stage: int) -> str:
        """Construct deterministic unique reminder identifier."""
        return f"{customer_id}___{item_id}___{exp_date}___{stage:+d}d"

    @staticmethod
    def _cycle_key(customer_id: str, item_id: str) -> str:
        """Construct canonical customer-medicine history key."""
        return f"{customer_id}:::{item_id}"

    def cancel_active_cycle(self, customer_id: str, item_id: str, reason: str = "new_purchase") -> int:
        """Cancel all pending/scheduled reminders for a customer + medicine cycle.

        Args:
            customer_id: Customer unique identifier.
            item_id: Medicine unique identifier.
            reason: Explanation for cancellation.

        Returns:
            int: Count of reminders cancelled.
        """
        cancelled_count = 0
        for rem in self._reminders.values():
            if rem.customerId == str(customer_id) and rem.itemId == str(item_id) and rem.status == "scheduled":
                rem.status = "cancelled"
                cancelled_count += 1
        return cancelled_count

    def schedule_refill_cycle(
        self,
        record: Union[Dict[str, Any], pd.Series],
        cancel_prior_cycles: bool = True,
    ) -> Dict[str, Any]:
        """Process a purchase event or prediction record and generate scheduled reminders.

        Workflow:
        1. Evaluate eligibility & validate prediction.
        2. If eligible, cancel any prior scheduled reminders for this (customerId, itemId).
        3. Calculate expected refill date.
        4. Generate 6 reminder stage records (-7, -3, -1, 0, +2, +5).
        5. Store deterministically with duplicate protection.

        Args:
            record: Dictionary or Series with customer, item, date, and prediction fields.
            cancel_prior_cycles: Whether to auto-cancel pending reminders for same (customerId, itemId).

        Returns:
            Dict containing scheduling outcome and created ReminderRecord objects.
        """
        eligibility = evaluate_refill_eligibility(record)
        if not eligibility["is_eligible"]:
            return {
                "status": eligibility["status"],
                "scheduled": False,
                "reason": eligibility["reason"],
                "reminders": [],
            }

        customer_id = str(record.get("customerId", "UNKNOWN"))
        item_id = str(record.get("itemId", "UNKNOWN"))
        customer_name = str(record.get("customerName", "Valued Customer"))
        MOBILE_NO = str(record.get("MOBILE_NO", record.get("phone1", ""))).strip()
        item_name = str(record.get("itemName", "Prescribed Medication"))

        p_date_raw = record.get("invoice_date", record.get("latest_purchase_date", record.get("current_purchase_date")))
        pred_days = eligibility["predicted_interval_days"]

        # Calculate expected refill date
        date_calc = calculate_expected_refill_date(p_date_raw, pred_days)
        if not date_calc["is_valid"]:
            return {
                "status": "invalid_prediction",
                "scheduled": False,
                "reason": date_calc["reason"],
                "reminders": [],
            }

        p_date = date_calc["latest_purchase_date"]
        exp_date = date_calc["expected_refill_date"]
        exp_date_str = exp_date.isoformat()
        p_date_str = p_date.isoformat()

        # Refill Cycle Reset: cancel previous active cycle for (customerId, itemId)
        cancelled_prior = 0
        if cancel_prior_cycles:
            cancelled_prior = self.cancel_active_cycle(customer_id, item_id, reason="new_purchase")

        created_reminders: List[ReminderRecord] = []
        for stage in self.stages:
            rem_date = exp_date + timedelta(days=stage)
            rem_date_str = rem_date.isoformat()
            rem_id = self._build_reminder_id(customer_id, item_id, exp_date_str, stage)
            msg = format_reminder_message(stage, exp_date)

            # Duplicate protection: if this exact reminder already exists, update/preserve
            if rem_id in self._reminders:
                existing = self._reminders[rem_id]
                created_reminders.append(existing)
            else:
                new_rem = ReminderRecord(
                    reminder_id=rem_id,
                    customerId=customer_id,
                    itemId=item_id,
                    customerName=customer_name,
                    MOBILE_NO=MOBILE_NO,
                    itemName=item_name,
                    latest_purchase_date=p_date_str,
                    predicted_interval_days=pred_days,
                    expected_refill_date=exp_date_str,
                    reminder_stage=stage,
                    reminder_date=rem_date_str,
                    message=msg,
                    status="scheduled",
                    history_quality=eligibility["history_quality"],
                )
                self._reminders[rem_id] = new_rem
                created_reminders.append(new_rem)

        # Update active cycle pointer
        self._active_cycles[self._cycle_key(customer_id, item_id)] = exp_date_str

        return {
            "status": "scheduled",
            "scheduled": True,
            "customerId": customer_id,
            "itemId": item_id,
            "expected_refill_date": exp_date_str,
            "cancelled_prior_reminders": cancelled_prior,
            "reminder_count": len(created_reminders),
            "reminders": created_reminders,
        }

    def get_due_reminders(
        self,
        target_date: Union[str, date, datetime, pd.Timestamp],
        filter_status: str = "scheduled",
    ) -> List[ReminderRecord]:
        """Identify which reminders are due on a specific calendar date.

        Args:
            target_date: Date to filter for (YYYY-MM-DD or date object).
            filter_status: Status filter (default 'scheduled').

        Returns:
            List[ReminderRecord]: Matching reminder records.
        """
        tgt = _parse_to_date(target_date).isoformat()
        return [
            rem for rem in self._reminders.values()
            if rem.reminder_date == tgt and (filter_status is None or rem.status == filter_status)
        ]

    def get_reminders_for_customer(
        self,
        customer_id: str,
        item_id: Optional[str] = None,
    ) -> List[ReminderRecord]:
        """Retrieve all reminder records for a customer, optionally filtered by medicine."""
        cid = str(customer_id)
        iid = str(item_id) if item_id is not None else None
        return [
            rem for rem in self._reminders.values()
            if rem.customerId == cid and (iid is None or rem.itemId == iid)
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Export all current reminder records to a Pandas DataFrame."""
        if not self._reminders:
            return pd.DataFrame(columns=[
                "reminder_id", "customerId", "itemId", "customerName", "MOBILE_NO",
                "itemName", "latest_purchase_date", "predicted_interval_days",
                "expected_refill_date", "reminder_stage", "reminder_date",
                "message", "status", "history_quality", "created_at"
            ])
        return pd.DataFrame([r.to_dict() for r in self._reminders.values()])


def get_daily_due_reminders(
    reminders: List[Union[ReminderRecord, Dict[str, Any]]],
    target_date: Union[str, date, datetime, pd.Timestamp],
) -> List[Union[ReminderRecord, Dict[str, Any]]]:
    """Pure functional helper to filter due reminders on a given date."""
    tgt = _parse_to_date(target_date).isoformat()
    due = []
    for r in reminders:
        if isinstance(r, ReminderRecord):
            if r.reminder_date == tgt and r.status == "scheduled":
                due.append(r)
        elif isinstance(r, dict):
            if r.get("reminder_date") == tgt and r.get("status") == "scheduled":
                due.append(r)
    return due


def run_scheduler_demo() -> Dict[str, Any]:
    """Deterministic in-memory demonstration of RefillCare Phase 5 scheduling.

    Scenario:
    - Latest Purchase: 2026-08-10 (10-08-2026)
    - Predicted Interval: 30 days
    - Expected Refill Date: 2026-09-09 (09-09-2026)
    """
    scheduler = RefillReminderScheduler()

    sample_event = {
        "customerId": "CUST_DEMO_001",
        "itemId": "ITEM_TELMI_40",
        "customerName": "Demo Patient",
        "MOBILE_NO": "9849012345",
        "itemName": "TELMISARTAN 40MG",
        "invoice_date": "2026-08-10",
        "purchase_count_so_far": 4,
        "is_recurring_history": 1,
        "predicted_days_until_refill": 30.0,
    }

    res = scheduler.schedule_refill_cycle(sample_event)
    print("=== RefillCare Phase 5 Scheduler Demonstration ===")
    print(f"Customer ID:           {sample_event['customerId']}")
    print(f"Medicine ID:           {sample_event['itemId']}")
    print(f"Latest Purchase Date:  {sample_event['invoice_date']} (10-08-2026)")
    print(f"Predicted Interval:    {sample_event['predicted_days_until_refill']} days")
    print(f"Expected Refill Date:  {res['expected_refill_date']} (09-09-2026)")
    print("\nGenerated 6 Reminder Stages:")
    print("-" * 85)
    for rem in res["reminders"]:
        print(f"Stage {rem.reminder_stage:+2d}d | Date: {rem.reminder_date} | Status: {rem.status:9s} | Copy:")
        print(f"  \"{rem.message}\"")
    print("-" * 85)

    # Demo 2: Query due reminders for a specific date (e.g. Stage -7 on 2026-09-02)
    due_02 = scheduler.get_due_reminders("2026-09-02")
    print(f"\nDue Reminders on 2026-09-02: {len(due_02)} found (Stage {due_02[0].reminder_stage}d)")

    # Demo 3: New purchase on 2026-09-08 resets cycle
    print("\n--- Simulating New Purchase on 2026-09-08 (Cycle Reset) ---")
    new_event = {
        "customerId": "CUST_DEMO_001",
        "itemId": "ITEM_TELMI_40",
        "customerName": "Demo Patient",
        "MOBILE_NO": "9849012345",
        "itemName": "TELMISARTAN 40MG",
        "invoice_date": "2026-09-08",
        "purchase_count_so_far": 5,
        "is_recurring_history": 1,
        "predicted_days_until_refill": 30.0,
    }
    res_new = scheduler.schedule_refill_cycle(new_event)
    print(f"Cancelled prior reminders: {res_new['cancelled_prior_reminders']}")
    print(f"New Expected Refill Date:  {res_new['expected_refill_date']} (08-10-2026)")

    # Check status of all reminders for patient + medicine
    all_rems = scheduler.get_reminders_for_customer("CUST_DEMO_001", "ITEM_TELMI_40")
    print("\nAll Reminders in Registry for Patient + Medicine:")
    for r in all_rems:
        print(f"  Exp: {r.expected_refill_date} | Stage: {r.reminder_stage:+2d}d | RemDate: {r.reminder_date} | Status: {r.status}")

    return {
        "initial_schedule": res,
        "new_schedule": res_new,
        "all_reminders": all_rems,
    }


if __name__ == "__main__":
    run_scheduler_demo()
