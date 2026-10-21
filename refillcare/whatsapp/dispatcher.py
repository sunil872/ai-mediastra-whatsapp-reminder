"""RefillCare WhatsApp Batch & Single Dispatcher with Database Auditing.

Manages automated outreach execution, database state transitions, and audit logs.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import (
    ReminderCycleModel,
    ReminderStageModel,
    WhatsAppDeliveryLogModel,
)
from refillcare.data.monthly_ingestion import determine_mobile_status, format_display_phone_10digits
from refillcare.whatsapp.client import RefillWhatsAppClient
from refillcare.whatsapp.consent import load_opt_out_index
from refillcare.whatsapp.template_formatter import get_tier_for_stage_offset

logger = logging.getLogger("refillcare_whatsapp_dispatcher")

MAX_LIVE_ATTEMPTS = 3


def live_requires_approval() -> bool:
    """Live sends require pharmacist APPROVED stages unless explicitly disabled."""
    import os

    return os.getenv("REFILLCARE_LIVE_REQUIRES_APPROVAL", "true").strip().lower() in ("1", "true", "yes", "on")


def already_delivered_live(db: Session, reminder_id: str) -> bool:
    return (
        db.query(WhatsAppDeliveryLogModel)
        .filter(
            WhatsAppDeliveryLogModel.reminder_id == reminder_id,
            WhatsAppDeliveryLogModel.is_dry_run == False,  # noqa: E712
            WhatsAppDeliveryLogModel.status == "SUCCESS",
        )
        .first()
        is not None
    )


def live_attempts(db: Session, reminder_id: str) -> int:
    return (
        db.query(WhatsAppDeliveryLogModel)
        .filter(WhatsAppDeliveryLogModel.reminder_id == reminder_id, WhatsAppDeliveryLogModel.is_dry_run == False)  # noqa: E712
        .count()
    )


class RefillWhatsAppDispatcher:
    """Enterprise batch and individual dispatcher for RefillCare WhatsApp reminders."""

    def __init__(self, client: Optional[RefillWhatsAppClient] = None):
        self.client = client or RefillWhatsAppClient()

    def dispatch_stage_record(
        self,
        db: Session,
        stage: ReminderStageModel,
        customer_name: str,
        phone_number: str,
        medication_name: str,
        expected_refill_date: date,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Dispatch a single database stage record and commit audit logs."""
        tier = get_tier_for_stage_offset(stage.stage_offset)

        res = self.client.send_refill_reminder(
            phone_number=phone_number,
            customer_name=customer_name,
            medications=medication_name,
            expected_refill_date=expected_refill_date,
            tier=tier,
            stage_offset=stage.stage_offset,
            dry_run=dry_run,
            reminder_id=stage.reminder_id,
        )

        is_ok = res.get("success", False)
        status_code = res.get("status_code", 200 if dry_run else 500)
        msg_id = res.get("message_id")

        # 1. Update Database Stage Model - only live sends change state; a dry run is audit-only
        if not dry_run:
            stage.status = "SENT" if is_ok else "FAILED"
            if is_ok:
                stage.sent_at = datetime.utcnow()
                stage.provider_msg_id = msg_id
            else:
                stage.failure_reason = res.get("message")
            stage.updated_at = datetime.utcnow()

        # 2. Append to WhatsAppDeliveryLogModel
        log_entry = WhatsAppDeliveryLogModel(
            reminder_id=stage.reminder_id,
            phone_number=phone_number,
            customer_name=customer_name,
            template_name=self.client.template_name,
            xinno_message_id=msg_id,
            is_dry_run=dry_run,
            status=("DRY_RUN_SUCCESS" if is_ok else "DRY_RUN_FAILED") if dry_run else ("SUCCESS" if is_ok else "FAILED"),
            http_status_code=status_code,
            response_payload=str(res.get("response", {}))[:4000],
            error_detail=res.get("message") if not is_ok else None,
            dispatched_at=datetime.utcnow(),
        )
        db.add(log_entry)
        db.commit()

        return res

    def dispatch_batch(
        self,
        db: Session,
        target_date: Optional[date] = None,
        target_month: Optional[str] = None,
        tier_filter: str = "ALL",
        dry_run: bool = True,
        batch_limit: int = 50,
        rate_limit_delay_sec: float = 0.05,
    ) -> Dict[str, Any]:
        """Dispatch a filtered batch of scheduled reminder stages.

        Args:
            db: Active SQLAlchemy database session.
            target_date: Optional specific target send date.
            target_month: Optional target send month string (YYYY-MM).
            tier_filter: "ALL", "DUE", "FOLLOWUP", or "LAPSED".
            dry_run: If True, simulates dispatch without live HTTP calls.
            batch_limit: Maximum records to process in single batch.
            rate_limit_delay_sec: Throttle delay between live requests.
        """
        # Query active scheduled stages
        if dry_run:
            allowed_status = ["PENDING", "DUE", "APPROVED"]
        elif live_requires_approval():
            allowed_status = ["APPROVED", "FAILED"]
        else:
            allowed_status = ["PENDING", "DUE", "APPROVED", "FAILED"]
        query = db.query(ReminderStageModel).join(
            ReminderCycleModel, ReminderStageModel.cycle_id == ReminderCycleModel.cycle_id
        ).filter(
            ReminderCycleModel.is_active == True,
            ReminderStageModel.status.in_(allowed_status),
        )

        if target_date:
            query = query.filter(ReminderStageModel.target_send_date == target_date)
        elif target_month and target_month != "ALL":
            try:
                y, m = map(int, target_month.split("-"))
                start_d = date(y, m, 1)
                end_d = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
                query = query.filter(
                    ReminderStageModel.target_send_date >= start_d,
                    ReminderStageModel.target_send_date < end_d,
                )
            except Exception:
                pass

        # Filter by tier
        tier_upper = str(tier_filter).strip().upper()
        if tier_upper == "DUE":
            query = query.filter(ReminderStageModel.stage_offset <= 0)
        elif tier_upper in ("FOLLOWUP", "FOLLOW_UP"):
            query = query.filter(ReminderStageModel.stage_offset.in_([2, 5]))
        elif tier_upper in ("LAPSED", "+45 DAYS", "RE-ENGAGEMENT"):
            query = query.filter(ReminderStageModel.stage_offset.in_([40, 45]))

        all_stages = query.all()

        opt_outs = load_opt_out_index(db)

        # Filter only valid mobile numbers and opted-in customers
        valid_candidates = []
        skipped_already_sent = 0
        skipped_opted_out = 0
        for stg in all_stages:
            dec = getattr(stg.cycle, "decision", None)
            if dec is None or not dec.is_eligible:
                continue

            if opt_outs.blocks(dec.customer_id, dec.mobile_no):
                skipped_opted_out += 1
                continue

            if not dry_run:
                if already_delivered_live(db, stg.reminder_id):
                    skipped_already_sent += 1
                    continue
                if stg.status == "FAILED" and live_attempts(db, stg.reminder_id) >= MAX_LIVE_ATTEMPTS:
                    continue
            phone = dec.mobile_no if dec and dec.mobile_no else ""
            cust_name = dec.customer_name if dec and dec.customer_name else "Valued Customer"
            med_name = dec.item_name if dec and dec.item_name else "Prescription Medication"

            mob_stat = determine_mobile_status(phone)
            if mob_stat == "Valid":
                valid_candidates.append((stg, cust_name, phone, med_name, stg.expected_refill_date))

        # Take batch slice
        batch_slice = valid_candidates[:batch_limit]

        logs: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        for stg, cust_name, phone, med_name, exp_dt in batch_slice:
            res = self.dispatch_stage_record(
                db=db,
                stage=stg,
                customer_name=cust_name,
                phone_number=phone,
                medication_name=med_name,
                expected_refill_date=exp_dt,
                dry_run=dry_run,
            )

            is_ok = res.get("success", False)
            if is_ok:
                success_count += 1
            else:
                failed_count += 1

            tier_resolved = get_tier_for_stage_offset(stg.stage_offset)
            logs.append({
                "reminder_id": stg.reminder_id,
                "customer_name": cust_name,
                "phone_number": format_display_phone_10digits(phone),
                "medication_name": med_name,
                "expected_refill_date": exp_dt.strftime("%d-%m-%Y"),
                "stage_offset": stg.stage_offset,
                "tier": tier_resolved,
                "status": "DRY_RUN_SUCCESS" if dry_run else ("SUCCESS" if is_ok else "FAILED"),
                "message_id": res.get("message_id", "DRY_RUN"),
                "message": res.get("message", ""),
            })

            if not dry_run and rate_limit_delay_sec > 0:
                time.sleep(rate_limit_delay_sec)

        return {
            "total_eligible": len(valid_candidates),
            "total_candidates": len(all_stages),
            "valid_candidates": len(valid_candidates),
            "dispatched_count": len(batch_slice),
            "success_count": success_count,
            "failed_count": failed_count,
            "is_dry_run": dry_run,
            "skipped_already_sent": skipped_already_sent,
            "requires_approval": (not dry_run) and live_requires_approval(),
            "tier_filter": tier_filter,
            "delivery_logs": logs,
            "delivery_summary": logs,
        }

    def dispatch_medsync_bundle_record(
        self,
        db: Session,
        bundle: Any,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Dispatch a synchronized multi-medication WhatsApp reminder.

        Live sends are idempotent per bundle_id and per covered reminder stage, require the
        covered stages to be APPROVED (unless disabled) and mark them SENT on success.
        """
        stage_ids = list(getattr(bundle, "stage_reminder_ids", []) or [])
        if not dry_run:
            if not stage_ids:
                return {"success": False, "status_code": None, "message": "Live bundle send refused: bundle carries no reminder stage ids."}
            if already_delivered_live(db, bundle.bundle_id) or any(already_delivered_live(db, sid) for sid in stage_ids):
                return {"success": False, "status_code": None, "message": "Skipped: already delivered live (idempotency guard).", "skipped": True}
            stages = db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id.in_(stage_ids)).all()
            allowed = {"APPROVED"} if live_requires_approval() else {"PENDING", "DUE", "APPROVED"}
            not_ready = [st.reminder_id for st in stages if st.status not in allowed]
            if not stages or not_ready:
                return {"success": False, "status_code": None, "message": f"Live bundle send refused: stages not approved ({len(not_ready)}).", "skipped": True}
        phone = bundle.mobile_no or ""
        cust_name = bundle.customer_name or "Valued Customer"
        med_names = [item.item_name for item in bundle.synced_items] if bundle.synced_items else [bundle.anchor_item_name]
        refill_dt = bundle.anchor_refill_date
        tier = getattr(bundle, "lifecycle_tier", "DUE")
        stage_offset = 0 if tier == "DUE" else (5 if tier == "FOLLOWUP" else 45)

        res = self.client.send_refill_reminder(
            phone_number=phone,
            customer_name=cust_name,
            medications=med_names,
            expected_refill_date=refill_dt,
            tier=tier,
            stage_offset=stage_offset,
            dry_run=dry_run,
            reminder_id=bundle.bundle_id,
        )

        is_ok = res.get("success", False)
        status_code = res.get("status_code", 200 if dry_run else 500)
        msg_id = res.get("message_id")

        log_entry = WhatsAppDeliveryLogModel(
            reminder_id=bundle.bundle_id,
            phone_number=phone,
            customer_name=cust_name,
            template_name=self.client.template_name,
            xinno_message_id=msg_id,
            is_dry_run=dry_run,
            status=("DRY_RUN_SUCCESS" if is_ok else "DRY_RUN_FAILED") if dry_run else ("SUCCESS" if is_ok else "FAILED"),
            http_status_code=status_code,
            response_payload=str(res.get("response", {}))[:4000],
            error_detail=res.get("message") if not is_ok else None,
            dispatched_at=datetime.utcnow(),
        )
        db.add(log_entry)
        if not dry_run and stage_ids:
            now = datetime.utcnow()
            for st in db.query(ReminderStageModel).filter(ReminderStageModel.reminder_id.in_(stage_ids)).all():
                st.status = "SENT" if is_ok else "FAILED"
                st.updated_at = now
                if is_ok:
                    st.sent_at = now
                    st.provider_msg_id = msg_id
                else:
                    st.failure_reason = res.get("message")
        db.commit()

        return res

    def dispatch_medsync_batch(
        self,
        db: Session,
        bundles: List[Any],
        dry_run: bool = True,
        batch_limit: int = 50,
        rate_limit_delay_sec: float = 0.05,
    ) -> Dict[str, Any]:
        """Dispatch a batch of Med-Sync synchronized bundles."""
        opt_outs = load_opt_out_index(db)
        valid_bundles = []
        for b in bundles:
            if determine_mobile_status(b.mobile_no or "") != "Valid":
                continue
            if opt_outs.blocks(getattr(b, "customer_id", None), b.mobile_no):
                continue
            valid_bundles.append(b)

        batch_slice = valid_bundles[:batch_limit]

        logs: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        skipped = 0
        for b in batch_slice:
            res = self.dispatch_medsync_bundle_record(db=db, bundle=b, dry_run=dry_run)
            is_ok = res.get("success", False)
            if res.get("skipped"):
                skipped += 1
                continue
            if is_ok:
                success_count += 1
            else:
                failed_count += 1

            med_names = [item.item_name for item in b.synced_items] if b.synced_items else [b.anchor_item_name]
            logs.append({
                "reminder_id": b.bundle_id,
                "customer_name": b.customer_name or "Valued Customer",
                "phone_number": format_display_phone_10digits(b.mobile_no or ""),
                "medication_name": ", ".join(med_names),
                "expected_refill_date": b.anchor_refill_date.strftime("%d-%m-%Y") if isinstance(b.anchor_refill_date, (date, datetime)) else str(b.anchor_refill_date),
                "stage_offset": 0 if getattr(b, "lifecycle_tier", "DUE") == "DUE" else (5 if getattr(b, "lifecycle_tier", "DUE") == "FOLLOWUP" else 45),
                "tier": getattr(b, "lifecycle_tier", "DUE"),
                "status": "DRY_RUN_SUCCESS" if dry_run else ("SUCCESS" if is_ok else "FAILED"),
                "message_id": res.get("message_id", "DRY_RUN"),
                "message": res.get("message", ""),
            })

            if not dry_run and rate_limit_delay_sec > 0:
                time.sleep(rate_limit_delay_sec)

        return {
            "total_eligible": len(valid_bundles),
            "total_candidates": len(bundles),
            "valid_candidates": len(valid_bundles),
            "dispatched_count": len(batch_slice),
            "success_count": success_count,
            "failed_count": failed_count,
            "is_dry_run": dry_run,
            "skipped_not_approved_or_already_sent": skipped,
            "delivery_logs": logs,
            "delivery_summary": logs,
        }
