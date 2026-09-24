"""V1 6-Stage Reminder Lifecycle State Machine & Idempotent Scheduler.

Manages:
1. 6-Stage Lifecycle (-7d, -3d, -1d, 0d, +2d, +5d)
2. State Transitions (PENDING -> DUE -> SENT / SUPERSEDED_BY_PURCHASE)
3. New-Purchase Cycle Reset
4. Idempotent Dispatch Queue Construction
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from refillcare.engine.decision_types import (
    RefillDecision,
    ReminderCycleState,
    ReminderStage,
    STAGE_OFFSETS,
    STATUS_DUE,
    STATUS_PENDING,
    STATUS_SENT,
    STATUS_SUPERSEDED,
)


def get_stage_message_template(
    stage_offset: int,
    customer_name: str,
    store_name: str,
    medicine_name: str,
    expected_refill_date: date,
    store_contact: str = "9876543210",
) -> str:
    """Format approved deterministic message for each of the 6 reminder stages."""
    date_str = expected_refill_date.strftime("%d-%m-%Y")
    
    if stage_offset == -7:
        msg = f"Your regular medicine refill for {medicine_name} may be due in about 7 days, around {date_str}."
    elif stage_offset == -3:
        msg = f"Your regular medicine refill for {medicine_name} is due in 3 days, around {date_str}."
    elif stage_offset == -1:
        msg = f"Reminder: Your medicine refill for {medicine_name} is due tomorrow, {date_str}."
    elif stage_offset == 0:
        msg = f"Your medicine refill for {medicine_name} is due today ({date_str})."
    elif stage_offset == 2:
        msg = f"Follow-up: Your medicine refill for {medicine_name} was due on {date_str}. Please let us know if you need assistance."
    elif stage_offset == 5:
        msg = f"Final check: We noticed you haven't refilled your {medicine_name} which was due on {date_str}. Please contact us if you need help."
    elif stage_offset == 40:
        msg = f"Care Check-in: We noticed your medicine refill for {medicine_name} was due on {date_str}. Have you refilled elsewhere, or would you like us to assist with home delivery?"
    else:
        msg = f"Your regular medicine refill for {medicine_name} is due around {date_str}."

    return (
        f"Dear {customer_name},\n\n"
        f"You are a valued customer of {store_name}.\n"
        f"{msg}\n\n"
        f"If you have any questions, please contact us at {store_contact}.\n\n"
        f"Team\n"
        f"{store_name}"
    )


class ReminderLifecycleManager:
    """Production lifecycle manager for RefillCare reminder cycles."""

    def __init__(self, store_name: str = "PHARMA HUBB", store_contact: str = "9876543210"):
        self.store_name = store_name
        self.store_contact = store_contact

    def create_cycle_from_decision(self, decision: RefillDecision) -> Optional[ReminderCycleState]:
        """Create a 6-stage reminder cycle from an eligible RefillDecision."""
        if not decision.is_eligible or not decision.expected_refill_date:
            return None

        stages: List[ReminderStage] = []
        for offset in STAGE_OFFSETS:
            target_date = decision.expected_refill_date + timedelta(days=offset)
            msg_text = get_stage_message_template(
                stage_offset=offset,
                customer_name=decision.customer_name,
                store_name=self.store_name,
                medicine_name=decision.item_name,
                expected_refill_date=decision.expected_refill_date,
                store_contact=self.store_contact,
            )

            stage = ReminderStage(
                cycle_id=decision.cycle_id,
                customer_id=decision.customer_id,
                item_id=decision.item_id,
                customer_item_key=decision.customer_item_key,
                stage_offset=offset,
                target_send_date=target_date,
                expected_refill_date=decision.expected_refill_date,
                status=STATUS_PENDING,
                message_text=msg_text,
            )
            stages.append(stage)

        cycle = ReminderCycleState(
            cycle_id=decision.cycle_id,
            customer_item_key=decision.customer_item_key,
            customer_id=decision.customer_id,
            item_id=decision.item_id,
            last_purchase_date=decision.last_purchase_date,
            expected_refill_date=decision.expected_refill_date,
            stages=stages,
            is_active=True,
        )
        return cycle

    def process_repurchase_reset(
        self,
        cycle: ReminderCycleState,
        latest_transaction_date: date,
    ) -> bool:
        """
        Check if customer made a new purchase after the cycle's last_purchase_date.
        If so, cancels all remaining pending stages immediately.

        Returns:
            True if cycle was reset / superseded, False otherwise.
        """
        if latest_transaction_date > cycle.last_purchase_date:
            cancelled = cycle.cancel_pending_stages(latest_transaction_date)
            return cancelled > 0
        return False

    def build_daily_dispatch_queue(
        self,
        active_cycles: List[ReminderCycleState],
        dispatch_date: date,
        decisions_lookup: Optional[Dict[str, RefillDecision]] = None,
    ) -> Tuple[pd.DataFrame, List[ReminderStage]]:
        """
        Identify all reminder stages due for dispatch on dispatch_date.

        Guarantees:
        - Strict idempotency (unique cycle_id + stage_offset)
        - Only active, non-superseded cycles
        - Valid phone number check
        """
        due_stages: List[ReminderStage] = []
        seen_keys: Set[Tuple[str, int]] = set()
        records: List[Dict[str, Any]] = []

        for cycle in active_cycles:
            if not cycle.is_active:
                continue

            for stage in cycle.stages:
                if stage.status == STATUS_PENDING and stage.target_send_date == dispatch_date:
                    dedup_key = (stage.cycle_id, stage.stage_offset)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)

                    stage.status = STATUS_DUE
                    due_stages.append(stage)

                    # Lookup decision details if available
                    dec = decisions_lookup.get(stage.customer_item_key) if decisions_lookup else None
                    phone = dec.mobile_no if dec else None
                    c_name = dec.customer_name if dec else "Valued Customer"
                    i_name = dec.item_name if dec else stage.item_id

                    records.append({
                        "cycle_id": stage.cycle_id,
                        "customer_item_key": stage.customer_item_key,
                        "customerId": stage.customer_id,
                        "customerName": c_name,
                        "MOBILE_NO": phone,
                        "itemId": stage.item_id,
                        "itemName": i_name,
                        "stage_offset": stage.stage_offset,
                        "target_send_date": stage.target_send_date.strftime("%Y-%m-%d"),
                        "expected_refill_date": stage.expected_refill_date.strftime("%Y-%m-%d"),
                        "status": stage.status,
                        "message_text": stage.message_text,
                    })

        dispatch_df = pd.DataFrame(records)
        return dispatch_df, due_stages
