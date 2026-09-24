"""Persistence and Lifecycle State Management Repository for V1 RefillCare.

Handles:
- RefillDecisionModel persistence and updates
- ReminderCycleModel & ReminderStageModel lifecycle sync
- Repurchase Cycle Auto-Reset (SUPERSEDED_BY_PURCHASE)
- Database-Level Idempotency on (cycle_id, stage_offset)
- State Machine Transition Enforcement
- Pharmacist Review Queue & Controlled Dispatch Auditing
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from database.models import (
    RefillDecisionModel,
    ReminderCycleModel,
    ReminderStageModel,
    WhatsAppDeliveryLogModel,
)
from refillcare.engine.decision_types import (
    RefillDecision,
    STAGE_OFFSETS,
    STATUS_APPROVED,
    STATUS_CANCELLED,
    STATUS_DUE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SENT,
    STATUS_SUPERSEDED,
)
from refillcare.engine.reminder_lifecycle import get_stage_message_template
from services.xinno_whatsapp import send_template_message
from utils.validators import mask_phone, normalize_to_whatsapp_number


class RefillPersistenceManager:
    """Enterprise database-backed persistence manager for RefillCare V1."""

    def __init__(self, store_name: str = "PHARMA HUBB", store_contact: str = "9876543210"):
        self.store_name = store_name
        self.store_contact = store_contact

    def save_refill_decisions(self, db: Session, decisions: List[RefillDecision]) -> int:
        """
        Persist a batch of RefillDecision objects idempotently using bulk lookups.

        Returns:
            Number of decision records upserted.
        """
        if not decisions:
            return 0

        now = datetime.utcnow()
        existing_map = {r.decision_id: r for r in db.query(RefillDecisionModel).all()}
        saved_count = 0
        new_records = []

        for dec in decisions:
            dec_id = f"DEC-{dec.customer_item_key}-{dec.last_purchase_date.strftime('%Y%m%d')}"
            existing = existing_map.get(dec_id)

            if existing:
                existing.path = dec.path
                existing.purchase_count = dec.purchase_count
                existing.is_eligible = dec.is_eligible
                existing.stability_tier = dec.stability_tier
                existing.cadence_median = dec.cadence_median
                existing.cadence_norm_mad = dec.cadence_norm_mad
                existing.cadence_drift = dec.cadence_drift
                existing.dos_days = dec.dos_days
                existing.units_purchased = dec.units_purchased
                existing.prediction_method = dec.prediction_method
                existing.predicted_interval_days = dec.predicted_interval_days
                existing.expected_refill_date = dec.expected_refill_date
                existing.decision_reason = dec.decision_reason
                existing.updated_at = now
            else:
                new_rec = RefillDecisionModel(
                    decision_id=dec_id,
                    cycle_id=dec.cycle_id,
                    customer_id=dec.customer_id,
                    customer_name=dec.customer_name,
                    mobile_no=dec.mobile_no,
                    item_id=dec.item_id,
                    item_name=dec.item_name,
                    customer_item_key=dec.customer_item_key,
                    path=dec.path,
                    purchase_count=dec.purchase_count,
                    is_eligible=dec.is_eligible,
                    stability_tier=dec.stability_tier,
                    cadence_median=dec.cadence_median,
                    cadence_norm_mad=dec.cadence_norm_mad,
                    cadence_drift=dec.cadence_drift,
                    dos_days=dec.dos_days,
                    units_purchased=dec.units_purchased,
                    prediction_method=dec.prediction_method,
                    predicted_interval_days=dec.predicted_interval_days,
                    last_purchase_date=dec.last_purchase_date,
                    expected_refill_date=dec.expected_refill_date,
                    decision_reason=dec.decision_reason,
                    created_at=now,
                    updated_at=now,
                )
                new_records.append(new_rec)
                existing_map[dec_id] = new_rec
            saved_count += 1

        if new_records:
            db.add_all(new_records)
        db.commit()
        return saved_count

    def sync_reminder_cycles(
        self,
        db: Session,
        decisions: List[RefillDecision],
    ) -> Dict[str, int]:
        """
        Synchronize reminder cycles and 6-stage schedules into the database.

        Enforces:
        - Repurchase auto-reset: marks previous cycle stages SUPERSEDED_BY_PURCHASE.
        - Database-level uniqueness on (cycle_id, stage_offset).
        - Idempotent execution (running N times produces 1 schedule).

        Returns:
            Dict with counts of cycles_created, stages_created, cycles_superseded, stages_superseded.
        """
        metrics = {
            "cycles_created": 0,
            "stages_created": 0,
            "cycles_superseded": 0,
            "stages_superseded": 0,
            "duplicate_stages_prevented": 0,
        }

        if not decisions:
            return metrics

        now = datetime.utcnow()

        # 1. Batch load existing active cycles grouped by customer_item_key
        active_cycles_by_key = defaultdict(list)
        for c in db.query(ReminderCycleModel).filter(ReminderCycleModel.is_active == True).all():
            active_cycles_by_key[c.customer_item_key].append(c)

        # 2. Batch load all cycle_ids and existing stages
        existing_cycles = {c.cycle_id: c for c in db.query(ReminderCycleModel).all()}
        existing_stages = {
            (s.cycle_id, s.stage_offset)
            for s in db.query(ReminderStageModel.cycle_id, ReminderStageModel.stage_offset).all()
        }

        # 3. Repurchase Reset & Ineligibility Sync Logic
        superseded_cycle_info = []
        for dec in decisions:
            if not dec.is_eligible or not dec.expected_refill_date:
                # Cancel any existing active cycles for newly ineligible or churned decisions
                for old_cycle in active_cycles_by_key.get(dec.customer_item_key, []):
                    if old_cycle.is_active:
                        old_cycle.is_active = False
                        old_cycle.superseded_at = now
                        metrics["cycles_superseded"] += 1
                        superseded_cycle_info.append((old_cycle.cycle_id, dec.last_purchase_date, dec.decision_reason))
                continue

            for old_cycle in active_cycles_by_key.get(dec.customer_item_key, []):
                if dec.last_purchase_date > old_cycle.last_purchase_date and old_cycle.is_active:
                    old_cycle.is_active = False
                    old_cycle.superseded_at = now
                    old_cycle.superseded_by_purchase_date = dec.last_purchase_date
                    metrics["cycles_superseded"] += 1
                    superseded_cycle_info.append((old_cycle.cycle_id, dec.last_purchase_date, f"Customer repurchased on {dec.last_purchase_date.strftime('%Y-%m-%d')}"))

        # Bulk update superseded stages
        for cycle_id, purch_date, reason in superseded_cycle_info:
            pending_stages = (
                db.query(ReminderStageModel)
                .filter(
                    ReminderStageModel.cycle_id == cycle_id,
                    ReminderStageModel.status.in_([STATUS_PENDING, STATUS_DUE, STATUS_APPROVED]),
                )
                .all()
            )
            for st in pending_stages:
                st.status = STATUS_SUPERSEDED
                st.failure_reason = reason
                st.updated_at = now
                metrics["stages_superseded"] += 1

        # 4. Create missing cycles and stages
        new_cycles = []
        new_stages = []

        for dec in decisions:
            if not dec.is_eligible or not dec.expected_refill_date:
                continue

            if dec.cycle_id not in existing_cycles:
                dec_id = f"DEC-{dec.customer_item_key}-{dec.last_purchase_date.strftime('%Y%m%d')}"
                new_cycle = ReminderCycleModel(
                    cycle_id=dec.cycle_id,
                    customer_item_key=dec.customer_item_key,
                    customer_id=dec.customer_id,
                    item_id=dec.item_id,
                    decision_id=dec_id,
                    last_purchase_date=dec.last_purchase_date,
                    expected_refill_date=dec.expected_refill_date,
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
                new_cycles.append(new_cycle)
                existing_cycles[dec.cycle_id] = new_cycle
                metrics["cycles_created"] += 1

            for offset in STAGE_OFFSETS:
                if (dec.cycle_id, offset) in existing_stages:
                    metrics["duplicate_stages_prevented"] += 1
                    continue

                target_date = dec.expected_refill_date + timedelta(days=offset)
                msg_text = get_stage_message_template(
                    stage_offset=offset,
                    customer_name=dec.customer_name,
                    store_name=self.store_name,
                    medicine_name=dec.item_name,
                    expected_refill_date=dec.expected_refill_date,
                    store_contact=self.store_contact,
                )
                rem_id = f"REM-{dec.cycle_id}-{offset:+d}"

                new_stage = ReminderStageModel(
                    reminder_id=rem_id,
                    cycle_id=dec.cycle_id,
                    customer_id=dec.customer_id,
                    item_id=dec.item_id,
                    customer_item_key=dec.customer_item_key,
                    stage_offset=offset,
                    target_send_date=target_date,
                    expected_refill_date=dec.expected_refill_date,
                    status=STATUS_PENDING,
                    message_text=msg_text,
                    created_at=now,
                    updated_at=now,
                )
                new_stages.append(new_stage)
                existing_stages.add((dec.cycle_id, offset))
                metrics["stages_created"] += 1

        if new_cycles:
            db.add_all(new_cycles)
            db.flush()
        if new_stages:
            db.add_all(new_stages)

        db.commit()
        return metrics

    def get_today_review_queue(
        self,
        db: Session,
        target_date: date,
        path_filter: Optional[str] = None,
        stability_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query all reminder stages due for dispatch on target_date.
        Joins with RefillDecisionModel to provide rich explainability.
        """
        query = (
            db.query(ReminderStageModel, RefillDecisionModel)
            .join(
                ReminderCycleModel,
                ReminderStageModel.cycle_id == ReminderCycleModel.cycle_id,
            )
            .outerjoin(
                RefillDecisionModel,
                ReminderCycleModel.decision_id == RefillDecisionModel.decision_id,
            )
            .filter(
                ReminderStageModel.target_send_date == target_date,
                ReminderCycleModel.is_active == True,
                RefillDecisionModel.is_eligible == True,
                ReminderStageModel.status.in_([STATUS_PENDING, STATUS_DUE, STATUS_APPROVED]),
            )
        )

        if path_filter:
            query = query.filter(RefillDecisionModel.path == path_filter)
        if stability_filter:
            query = query.filter(RefillDecisionModel.stability_tier == stability_filter)

        results = query.all()
        queue: List[Dict[str, Any]] = []

        for stage, dec in results:
            c_name = dec.customer_name if dec else "Valued Customer"
            i_name = dec.item_name if dec else stage.item_id
            phone = dec.mobile_no if dec else None
            p_masked = mask_phone(phone) if phone else "MISSING"

            queue.append({
                "reminder_id": stage.reminder_id,
                "cycle_id": stage.cycle_id,
                "customer_item_key": stage.customer_item_key,
                "customer_id": stage.customer_id,
                "customer_name": c_name,
                "phone_number": phone,
                "masked_phone": p_masked,
                "item_id": stage.item_id,
                "item_name": i_name,
                "purchase_count": dec.purchase_count if dec else 0,
                "last_purchase_date": dec.last_purchase_date.strftime("%Y-%m-%d") if (dec and dec.last_purchase_date) else None,
                "stage_offset": stage.stage_offset,
                "target_send_date": stage.target_send_date.strftime("%Y-%m-%d"),
                "expected_refill_date": stage.expected_refill_date.strftime("%Y-%m-%d"),
                "status": stage.status,
                "path": dec.path if dec else "UNKNOWN",
                "stability_tier": dec.stability_tier if dec else "UNKNOWN",
                "prediction_method": dec.prediction_method if dec else "NONE",
                "historical_median_days": dec.cadence_median if dec else None,
                "predicted_interval_days": dec.predicted_interval_days if dec else None,
                "estimated_days_of_supply": dec.dos_days if (dec and dec.dos_days) else (dec.predicted_interval_days if dec else 30.0),
                "decision_reason": dec.decision_reason if dec else "",
                "message_text": stage.message_text,
            })

        return queue

    def approve_reminder_stage(self, db: Session, reminder_id: str) -> Dict[str, Any]:
        """Mark a reminder stage as APPROVED for dispatch."""
        stage = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id == reminder_id).first()
        if not stage:
            raise ValueError(f"Reminder stage '{reminder_id}' not found.")

        # Valid transitions: PENDING or DUE -> APPROVED
        if stage.status in (STATUS_SUPERSEDED, STATUS_SENT, STATUS_CANCELLED):
            raise ValueError(f"Cannot approve reminder in state '{stage.status}'.")

        stage.status = STATUS_APPROVED
        stage.updated_at = datetime.utcnow()
        db.commit()
        return {"reminder_id": reminder_id, "status": STATUS_APPROVED, "success": True}

    def reject_reminder_stage(self, db: Session, reminder_id: str, reason: str = "Pharmacist rejected") -> Dict[str, Any]:
        """Mark a reminder stage as REJECTED."""
        stage = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id == reminder_id).first()
        if not stage:
            raise ValueError(f"Reminder stage '{reminder_id}' not found.")

        if stage.status in (STATUS_SUPERSEDED, STATUS_SENT):
            raise ValueError(f"Cannot reject reminder in state '{stage.status}'.")

        stage.status = STATUS_REJECTED
        stage.failure_reason = reason
        stage.updated_at = datetime.utcnow()
        db.commit()
        return {"reminder_id": reminder_id, "status": STATUS_REJECTED, "reason": reason, "success": True}

    def dispatch_reminder_stage(
        self,
        db: Session,
        reminder_id: str,
        is_dry_run: bool = True,
        authorizer: str = "pharmacist",
    ) -> Dict[str, Any]:
        """
        Dispatch reminder via Xinno gateway or simulation in strict DRY-RUN mode.
        Requires the reminder stage to be in APPROVED state first.
        """
        stage = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id == reminder_id).first()
        if not stage:
            raise ValueError(f"Reminder stage '{reminder_id}' not found.")

        if stage.status != STATUS_APPROVED:
            raise ValueError(f"Cannot dispatch reminder in state '{stage.status}'. Stage must be APPROVED by pharmacist first.")

        # Fetch associated decision for customer details
        dec = (
            db.query(RefillDecisionModel)
            .join(ReminderCycleModel, RefillDecisionModel.decision_id == ReminderCycleModel.decision_id)
            .filter(ReminderCycleModel.cycle_id == stage.cycle_id)
            .first()
        )

        phone = dec.mobile_no if dec else None
        c_name = dec.customer_name if dec else "Valued Customer"
        i_name = dec.item_name if dec else stage.item_id

        if not phone:
            stage.status = STATUS_FAILED
            stage.failure_reason = "Missing valid phone number"
            db.commit()
            return {"reminder_id": reminder_id, "status": STATUS_FAILED, "error": "Missing phone number"}

        # Perform dispatch
        dispatch_res = send_template_message(
            phone_number=phone,
            customer_name=c_name,
            store_name=self.store_name,
            dry_run=is_dry_run,
        )

        # Audit Log
        now = datetime.utcnow()
        audit_rec = WhatsAppDeliveryLogModel(
            reminder_id=reminder_id,
            phone_number=phone,
            customer_name=c_name,
            template_name="refillcare_medicine_reminder",
            xinno_message_id=dispatch_res.get("provider_msg_id"),
            is_dry_run=is_dry_run,
            status="DRY_RUN_SUCCESS" if is_dry_run else ("SUCCESS" if dispatch_res.get("success") else "FAILED"),
            http_status_code=200 if dispatch_res.get("success") else 400,
            response_payload=str(dispatch_res),
            error_detail=dispatch_res.get("error"),
            dispatched_at=now,
        )
        db.add(audit_rec)

        if dispatch_res.get("success"):
            stage.status = STATUS_SENT
            stage.sent_at = now
            stage.provider_msg_id = dispatch_res.get("provider_msg_id")
        else:
            stage.status = STATUS_FAILED
            stage.failure_reason = dispatch_res.get("error")

        stage.updated_at = now
        db.commit()

        return {
            "reminder_id": reminder_id,
            "status": stage.status,
            "is_dry_run": is_dry_run,
            "provider_msg_id": stage.provider_msg_id,
            "success": dispatch_res.get("success"),
        }
