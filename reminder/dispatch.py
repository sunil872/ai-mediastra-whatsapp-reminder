"""RefillCare Controlled WhatsApp Reminder Dispatch Engine.

Bridges the Phase 5 reminder schedule queue to the approved Xinno WhatsApp
API service with strict manual confirmation, dry-run safety, duplicate protection,
phone masking, and credential sanitization.
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import requests
import pandas as pd
from dotenv import load_dotenv

# Reusable phone normalizer and maskers from existing services
from services.xinno_whatsapp import (
    normalize_phone_number,
    _mask_phone_for_log,
    _format_safe_exception_details,
    DEFAULT_API_URL,
)
from reminder.scheduler import (
    ReminderRecord,
    format_reminder_message,
    _parse_to_date,
)
from refillcare.config.whatsapp_template import (
    WHATSAPP_TEMPLATE_NAME,
    WHATSAPP_TEMPLATE_LANGUAGE,
    WHATSAPP_TEMPLATE_POLICY,
    WHATSAPP_TEMPLATE_VARIABLES,
    build_template_components,
)

logger = logging.getLogger("refillcare.dispatch")

# Ensure .env is loaded
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

# Approved RefillCare WhatsApp Template Name
REFILLCARE_TEMPLATE_NAME = WHATSAPP_TEMPLATE_NAME


@dataclass
class DispatchResult:
    """Standardized, safe result for an individual reminder dispatch attempt."""

    reminder_id: str
    customerId: str
    itemId: str
    customerName: str
    phone_masked: str
    reminder_stage: str
    expected_refill_date: str
    status: str  # "accepted", "failed", "cancelled", "duplicate_prevented", "invalid"
    success: bool
    message: str
    provider_message_id: Optional[str] = None
    status_code: Optional[int] = None
    error_category: Optional[str] = None
    dry_run: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert dispatch result to standard dictionary."""
        return asdict(self)


def sanitize_template_text(val: Any, default: str = "") -> str:
    """Sanitize text parameters for WhatsApp templates.

    Meta and WhatsApp APIs strictly reject newlines, carriage returns, or tabs
    inside template body parameter strings.
    """
    if val is None or pd_isna(val):
        return default
    text = str(val).strip()
    # Replace newlines, returns, and tabs with single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple consecutive spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else default


def pd_isna(val: Any) -> bool:
    """Check if value is null/NaN safely without heavy pandas dependency in inner loops."""
    if val is None:
        return True
    if isinstance(val, float) and (val != val or str(val) == "nan"):
        return True
    return False


def mask_phone_for_ui(phone: str) -> str:
    """Mask phone number for safe UI display (e.g. 919849012345 -> ***2345)."""
    digits = "".join(c for c in str(phone) if c.isdigit())
    if len(digits) < 4:
        return "***"
    return f"***{digits[-4:]}"


def build_refillcare_whatsapp_payload(
    reminder: Union[ReminderRecord, Dict[str, Any]],
    store_name: Optional[str] = None,
    store_contact: Optional[str] = None,
    template_name: Optional[str] = None,
    template_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct deterministic Xinno WhatsApp template payload for RefillCare.

    Template: refillcare_medicine_reminder
    Parameters:
      {{1}} -> Customer Name
      {{2}} -> Store Name
      {{3}} -> Dynamic stage reminder message
      {{4}} -> Medicine List / Name
      {{5}} -> Store Contact / Phone
    """
    rec = reminder.to_dict() if isinstance(reminder, ReminderRecord) else dict(reminder)

    customer_id = sanitize_template_text(rec.get("customerId", ""))
    item_id = sanitize_template_text(rec.get("itemId", ""))
    customer_name = sanitize_template_text(rec.get("customerName", ""), default="Valued Customer")
    medicine_name = sanitize_template_text(rec.get("itemName", ""), default="Prescribed Medicine")
    raw_phone = str(rec.get("MOBILE_NO", rec.get("phone1", ""))).strip()
    exp_date_raw = rec.get("expected_refill_date", "")
    stage_val = rec.get("reminder_stage", 0)

    # Validate Customer & Medicine Identifiers
    if not customer_id or customer_id == "UNKNOWN":
        return {
            "is_valid": False,
            "error": "Missing or invalid customerId.",
            "error_category": "Validation Error",
        }

    if not item_id or item_id == "UNKNOWN":
        return {
            "is_valid": False,
            "error": "Missing or invalid itemId.",
            "error_category": "Validation Error",
        }

    # Validate & Normalize Phone Number
    if not raw_phone:
        return {
            "is_valid": False,
            "error": "Destination phone number is empty.",
            "error_category": "Validation Error",
        }

    try:
        normalized_phone = normalize_phone_number(raw_phone)
    except ValueError as val_err:
        return {
            "is_valid": False,
            "error": f"Invalid destination phone number: {val_err}",
            "error_category": "Validation Error",
        }

    # Validate Expected Refill Date
    if not exp_date_raw:
        return {
            "is_valid": False,
            "error": "Missing expected refill date.",
            "error_category": "Validation Error",
        }

    try:
        parsed_exp_date = _parse_to_date(exp_date_raw)
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Invalid expected refill date: {e}",
            "error_category": "Validation Error",
        }

    # Extract or generate exact stage message
    msg = rec.get("message")
    if not msg:
        try:
            # Parse stage integer if given as string like '-7 days' or -7
            if isinstance(stage_val, str):
                m = re.search(r"[-+]?\d+", stage_val)
                stage_int = int(m.group(0)) if m else 0
            else:
                stage_int = int(stage_val)
            msg = format_reminder_message(stage_int, parsed_exp_date)
        except Exception:
            msg = f"Your regular medicine refill may be due around {parsed_exp_date.strftime('%d-%m-%Y')}."

    sanitized_msg = sanitize_template_text(msg)
    if not sanitized_msg:
        return {
            "is_valid": False,
            "error": "Constructed reminder message is empty.",
            "error_category": "Validation Error",
        }

    # Store configurations from env or arguments
    eff_store_name = sanitize_template_text(
        store_name or os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB"),
        default="PHARMA HUBB"
    )
    eff_store_contact = sanitize_template_text(
        store_contact or os.getenv("STORE_CONTACT_NUMBER", "our pharmacy"),
        default="our pharmacy"
    )
    eff_template_name = sanitize_template_text(
        template_name or os.getenv("REFILLCARE_TEMPLATE_NAME", REFILLCARE_TEMPLATE_NAME),
        default=REFILLCARE_TEMPLATE_NAME
    )
    eff_lang = sanitize_template_text(
        template_language or os.getenv("REFILLCARE_TEMPLATE_LANGUAGE", "en"),
        default="en"
    )

    # 6 Body Parameters as per Approved RefillCare Template Specification
    components = build_template_components(
        customer_name=customer_name,
        store_name=eff_store_name,
        stage_message=sanitized_msg,
        medicine=medicine_name,
        store_contact=eff_store_contact,
    )

    payload = {
        "messaging_product": "whatsapp",
        "to": normalized_phone,
        "type": "template",
        "template": {
            "language": {
                "policy": WHATSAPP_TEMPLATE_POLICY,
                "code": eff_lang
            },
            "name": eff_template_name,
            "components": components,
        }
    }

    return {
        "is_valid": True,
        "normalized_phone": normalized_phone,
        "masked_phone": _mask_phone_for_log(normalized_phone),
        "ui_masked_phone": mask_phone_for_ui(normalized_phone),
        "customer_name": customer_name,
        "medicine_name": medicine_name,
        "stage_message": sanitized_msg,
        "template_name": eff_template_name,
        "payload": payload,
        "raw_record": rec,
    }


def select_eligible_reminders_for_dispatch(
    reminders: Union[pd.DataFrame, List[Union[ReminderRecord, Dict[str, Any]]]],
    target_date: Optional[Union[str, date]] = None,
) -> List[Dict[str, Any]]:
    """Filter reminders eligible for dispatch based on status, date, and required keys.

    Accepts a pandas DataFrame, list of ReminderRecords, or list of dictionaries.
    """
    eligible: List[Dict[str, Any]] = []
    target_iso = _parse_to_date(target_date).isoformat() if target_date else None

    if isinstance(reminders, pd.DataFrame):
        records_to_process = reminders.to_dict("records")
    elif isinstance(reminders, (list, tuple)):
        records_to_process = [
            r.to_dict() if isinstance(r, ReminderRecord) else dict(r)
            for r in reminders
        ]
    else:
        return eligible

    for rec in records_to_process:
        status = str(rec.get("status") or rec.get("Status") or "scheduled").strip().lower()
        if status != "scheduled":
            continue

        # Check reminder date if target_date is specified
        if target_iso is not None:
            r_date_raw = rec.get("raw_reminder_date") or rec.get("reminder_date") or rec.get("Reminder Date") or ""
            try:
                r_date_iso = _parse_to_date(r_date_raw).isoformat()
            except Exception:
                continue
            if r_date_iso != target_iso:
                continue

        # Must have identifiers and phone
        cid = str(rec.get("customerId") or rec.get("Customer ID") or "").strip()
        iid = str(rec.get("itemId") or rec.get("Medicine ID") or "").strip()
        phone = str(rec.get("MOBILE_NO") or rec.get("phone1") or rec.get("Delivery Phone") or "").strip()

        if cid and iid and phone and cid.upper() != "UNKNOWN" and iid.upper() != "UNKNOWN":
            # Ensure canonical keys are populated in the dict
            normalized_rec = dict(rec)
            normalized_rec["customerId"] = cid
            normalized_rec["itemId"] = iid
            normalized_rec["MOBILE_NO"] = phone
            normalized_rec["customerName"] = rec.get("customerName") or rec.get("Customer") or cid
            normalized_rec["itemName"] = rec.get("itemName") or rec.get("Medicine") or iid
            normalized_rec["expected_refill_date"] = rec.get("expected_refill_date") or rec.get("Expected Refill Date") or ""
            normalized_rec["reminder_date"] = rec.get("reminder_date") or rec.get("raw_reminder_date") or rec.get("Reminder Date") or ""
            normalized_rec["reminder_stage"] = rec.get("reminder_stage") or rec.get("Reminder Stage") or 0
            normalized_rec["message"] = rec.get("message") or rec.get("Reminder Message") or ""
            normalized_rec["status"] = "scheduled"
            eligible.append(normalized_rec)

    return eligible


class RefillCareReminderDispatcher:
    """Stateful dispatch engine with duplicate prevention, SQLite audit logging, and dry-run safety."""

    def __init__(self, storage: Optional[Any] = None):
        """Initialize dispatch engine with state tracking and optional SQLite persistence."""
        # Maps reminder_id -> DispatchResult for idempotent in-memory tracking
        self._dispatched: Dict[str, DispatchResult] = {}
        self.storage = storage

    @property
    def dispatched_records(self) -> Dict[str, DispatchResult]:
        """Return read-only dictionary of dispatched records."""
        return dict(self._dispatched)

    def is_already_dispatched(self, reminder_id: str) -> bool:
        """Check if reminder has already been successfully dispatched."""
        res = self._dispatched.get(reminder_id)
        if res and res.status in ("accepted", "sent") and res.success:
            return True

        if self.storage is not None:
            db_rec = self.storage.get_reminder(reminder_id)
            if db_rec and db_rec.get("status") in ("accepted", "sent") and not db_rec.get("is_dry_run", 0):
                return True

        return False

    def dispatch_single_reminder(
        self,
        reminder: Union[ReminderRecord, Dict[str, Any]],
        dry_run: bool = True,
        store_name: Optional[str] = None,
        store_contact: Optional[str] = None,
    ) -> DispatchResult:
        """Dispatch a single RefillCare reminder with safety checks and dry-run support.

        Args:
            reminder: ReminderRecord or dictionary.
            dry_run: If True, validates payload without making external HTTP requests.
            store_name: Medical store name override.
            store_contact: Store contact number override.

        Returns:
            DispatchResult: Normalized, safe dispatch result without sensitive credentials.
        """
        rec = reminder.to_dict() if isinstance(reminder, ReminderRecord) else dict(reminder)
        rem_id = str(rec.get("reminder_id", "UNKNOWN_ID"))
        cid = str(rec.get("customerId", "UNKNOWN"))
        iid = str(rec.get("itemId", "UNKNOWN"))
        c_name = str(rec.get("customerName", "Valued Customer"))
        stage_str = str(rec.get("reminder_stage", ""))
        exp_date_str = str(rec.get("expected_refill_date", ""))
        raw_phone = str(rec.get("MOBILE_NO", rec.get("phone1", "")))
        rem_status = str(rec.get("status", "scheduled"))

        # 1. Check if reminder was cancelled in Phase 5
        if rem_status == "cancelled":
            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(raw_phone),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="cancelled",
                success=False,
                message="Reminder was cancelled (e.g. customer repurchased early). Omitted from dispatch.",
                error_category="Business Logic",
                dry_run=dry_run,
            )
            return result

        # 2. Duplicate Check: If already dispatched/accepted, do NOT send again
        if self.is_already_dispatched(rem_id):
            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(raw_phone),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="duplicate_prevented",
                success=False,
                message="Duplicate Prevention: This exact reminder ID was already dispatched.",
                error_category="Duplicate",
                dry_run=dry_run,
            )
            return result

        # 3. Build & Validate Payload
        build_res = build_refillcare_whatsapp_payload(
            reminder=reminder,
            store_name=store_name,
            store_contact=store_contact,
        )

        if not build_res["is_valid"]:
            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(raw_phone),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="invalid",
                success=False,
                message=build_res["error"],
                error_category=build_res.get("error_category", "Validation Error"),
                dry_run=dry_run,
            )
            return result

        masked_phone = build_res["masked_phone"]
        payload = build_res["payload"]

        # 4. Record Dispatch Attempt Start in Storage (Transitions status to 'sending')
        if self.storage is not None:
            self.storage.record_dispatch_start(rec, is_dry_run=dry_run)

        # 5. DRY-RUN MODE: Zero external HTTP requests
        if dry_run:
            logger.info(f"[DRY RUN] RefillCare reminder validated for {masked_phone} (ID: {rem_id})")
            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(build_res["normalized_phone"]),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="accepted",
                success=True,
                message="[DRY RUN] WhatsApp template payload validated successfully. No HTTP request sent.",
                provider_message_id=f"dry_run_{rem_id[:12]}",
                dry_run=True,
            )
            self._dispatched[rem_id] = result
            if self.storage is not None:
                self.storage.record_dispatch_result(result, is_dry_run=True)
            return result

        # 6. LIVE SENDING MODE (Requires explicit user confirmation in UI)
        api_url = os.getenv("XINNO_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
        api_key = os.getenv("XINNO_API_KEY", "").strip()
        waba_number = os.getenv("XINNO_WABA_NUMBER", "").strip()

        if not api_key:
            res = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(build_res["normalized_phone"]),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="failed",
                success=False,
                message="Configuration Error: XINNO_API_KEY is not configured.",
                error_category="Configuration Error",
                dry_run=False,
            )
            if self.storage is not None:
                self.storage.record_dispatch_result(res, is_dry_run=False)
            return res

        if not waba_number:
            res = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(build_res["normalized_phone"]),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="failed",
                success=False,
                message="Configuration Error: XINNO_WABA_NUMBER is not configured.",
                error_category="Configuration Error",
                dry_run=False,
            )
            if self.storage is not None:
                self.storage.record_dispatch_result(res, is_dry_run=False)
            return res

        headers = {
            "wabaNumber": waba_number,
            "Key": api_key,
            "Content-Type": "application/json",
        }

        try:
            http_resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
            try:
                resp_json = http_resp.json()
            except Exception:
                resp_json = {}

            has_err = isinstance(resp_json, dict) and ("error" in resp_json or "errors" in resp_json)
            is_ok = (http_resp.status_code in (200, 201)) and not has_err

            provider_mid = None
            if isinstance(resp_json, dict):
                msgs = resp_json.get("messages")
                if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
                    provider_mid = msgs[0].get("id")
                provider_mid = provider_mid or resp_json.get("messageId") or resp_json.get("id")

            if is_ok:
                msg_text = "Xinno API accepted the reminder message."
                status_text = "accepted"
            else:
                err_info = resp_json.get("error") or resp_json.get("errors") or f"HTTP status {http_resp.status_code}"
                msg_text = f"API Error: {err_info}"
                status_text = "failed"

            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(build_res["normalized_phone"]),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status=status_text,
                success=is_ok,
                status_code=http_resp.status_code,
                message=msg_text,
                provider_message_id=str(provider_mid) if provider_mid else None,
                error_category="API Error" if not is_ok else None,
                dry_run=False,
            )
            self._dispatched[rem_id] = result
            if self.storage is not None:
                self.storage.record_dispatch_result(result, is_dry_run=False)
            return result

        except Exception as exc:
            diag = _format_safe_exception_details(exc, api_key)
            result = DispatchResult(
                reminder_id=rem_id,
                customerId=cid,
                itemId=iid,
                customerName=c_name,
                phone_masked=mask_phone_for_ui(build_res["normalized_phone"]),
                reminder_stage=stage_str,
                expected_refill_date=exp_date_str,
                status="failed",
                success=False,
                message=f"Network Error ({diag['error_category']}): {diag['error_detail']}",
                error_category=diag["error_category"],
                dry_run=False,
            )
            self._dispatched[rem_id] = result
            if self.storage is not None:
                self.storage.record_dispatch_result(result, is_dry_run=False)
            return result

    def dispatch_batch_reminders(
        self,
        reminders: List[Union[ReminderRecord, Dict[str, Any]]],
        dry_run: bool = True,
        store_name: Optional[str] = None,
        store_contact: Optional[str] = None,
    ) -> List[DispatchResult]:
        """Dispatch a batch of due reminders with in-batch deduplication."""
        results: List[DispatchResult] = []
        seen_in_batch = set()

        for rem in reminders:
            rec = rem.to_dict() if isinstance(rem, ReminderRecord) else dict(rem)
            rem_id = str(rec.get("reminder_id", "UNKNOWN"))

            # In-batch duplicate check
            if rem_id in seen_in_batch:
                dup_res = DispatchResult(
                    reminder_id=rem_id,
                    customerId=str(rec.get("customerId", "")),
                    itemId=str(rec.get("itemId", "")),
                    customerName=str(rec.get("customerName", "Valued Customer")),
                    phone_masked=mask_phone_for_ui(str(rec.get("MOBILE_NO", rec.get("phone1", "")))),
                    reminder_stage=str(rec.get("reminder_stage", "")),
                    expected_refill_date=str(rec.get("expected_refill_date", "")),
                    status="duplicate_prevented",
                    success=False,
                    message="Duplicate in batch: Omitted redundant dispatch attempt.",
                    error_category="Duplicate",
                    dry_run=dry_run,
                )
                results.append(dup_res)
                continue

            seen_in_batch.add(rem_id)
            res = self.dispatch_single_reminder(
                reminder=rem,
                dry_run=dry_run,
                store_name=store_name,
                store_contact=store_contact,
            )
            results.append(res)

        return results

    def to_dataframe(self) -> Any:
        """Export dispatch records to a pandas DataFrame."""
        import pandas as pd
        if not self._dispatched:
            return pd.DataFrame(columns=[
                "reminder_id", "customerId", "itemId", "customerName",
                "phone_masked", "reminder_stage", "expected_refill_date",
                "status", "success", "message", "provider_message_id",
                "status_code", "error_category", "dry_run", "timestamp"
            ])
        return pd.DataFrame([r.to_dict() for r in self._dispatched.values()])
