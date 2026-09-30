"""Xinno WhatsApp REST Client for RefillCare Reminders.

Handles 3-Tier dynamic template message delivery via Xinno CPaaS API.
Supports strict Dry-Run simulation and safe credential isolation.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

from refillcare.whatsapp.template_formatter import (
    build_dynamic_tier_text,
    build_template_parameters,
    format_full_message_preview,
    get_tier_for_stage_offset,
)

logger = logging.getLogger("refillcare_whatsapp")

# Explicitly load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

DEFAULT_API_URL = "https://whatsapp.xinno.in/REST/directApi/message"
DEFAULT_TEMPLATE_NAME = "refillcare_medicine_reminder"
DEFAULT_STORE_CONTACT = "+91 9966473474"


def normalize_e164_whatsapp_phone(phone_number: str, country_code: str = "91") -> str:
    """Normalize phone number to 12-digit format required by Xinno (e.g. 919876543210)."""
    if not phone_number:
        raise ValueError("Phone number cannot be empty.")

    cleaned = "".join(c for c in str(phone_number) if c.isdigit())
    if not cleaned:
        raise ValueError(f"Invalid phone number containing no digits: '{phone_number}'")

    if len(cleaned) == 10:
        return f"{country_code}{cleaned}"
    elif len(cleaned) == 11 and cleaned.startswith("0"):
        return f"{country_code}{cleaned[1:]}"
    elif len(cleaned) == 12 and cleaned.startswith(country_code):
        return cleaned
    return cleaned


def mask_phone_for_log(phone: str) -> str:
    """Mask middle digits for safe logging (e.g. 919876543210 -> 91******3210)."""
    digits = "".join(c for c in str(phone) if c.isdigit())
    if len(digits) < 6:
        return "***"
    return f"{digits[:2]}{'*' * (len(digits) - 6)}{digits[-4:]}"


def log_refill_send_attempt(
    customer_name: str,
    phone: str,
    template_name: str,
    tier: str,
    success: bool,
    status_code: Optional[int],
    message: str,
    message_id: Optional[str] = None,
    is_dry_run: bool = True,
) -> None:
    """Safely log dispatch attempt to logs/whatsapp_refill_send.log."""
    try:
        log_dir = _PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "whatsapp_refill_send.log"

        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)

        log_inst = logging.getLogger("refillcare_whatsapp_file")
        log_inst.setLevel(logging.INFO)
        if not log_inst.handlers:
            log_inst.addHandler(file_handler)

        mode_str = "DRY_RUN" if is_dry_run else "LIVE"
        status_str = "SUCCESS" if success else "FAILED"
        masked_phone = mask_phone_for_log(phone)
        mid = message_id or "-"

        log_inst.info(
            f"[{mode_str}_{status_str}] Tier='{tier}', Customer='{customer_name}', Phone='{masked_phone}', "
            f"Template='{template_name}', Status={status_code}, MessageId='{mid}', Message='{message}'"
        )
    except Exception as log_err:
        logger.warning(f"Failed to write to RefillCare WhatsApp log file: {log_err}")


class RefillWhatsAppClient:
    """Enterprise Xinno WhatsApp client tailored for RefillCare multi-tier reminders."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        waba_number: Optional[str] = None,
        template_name: Optional[str] = None,
        template_language: str = "en",
        store_name: Optional[str] = None,
        store_contact: Optional[str] = None,
        param_count: int = 6,
    ):
        load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
        self.api_url = (api_url or os.getenv("XINNO_API_URL", DEFAULT_API_URL)).strip() or DEFAULT_API_URL
        self.api_key = (api_key or os.getenv("XINNO_API_KEY", "")).strip()
        self.waba_number = (waba_number or os.getenv("XINNO_WABA_NUMBER", "")).strip()
        
        # Check specific RefillCare template env var or fallback
        self.template_name = (
            template_name
            or os.getenv("WHATSAPP_REFILL_TEMPLATE_NAME", "")
            or os.getenv("WHATSAPP_TEMPLATE_NAME", DEFAULT_TEMPLATE_NAME)
        ).strip() or DEFAULT_TEMPLATE_NAME

        self.template_language = template_language or os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en")
        self.store_name = (store_name or os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB")).strip() or "PHARMA HUBB"
        self.store_contact = (store_contact or os.getenv("MEDICAL_STORE_CONTACT", DEFAULT_STORE_CONTACT)).strip() or DEFAULT_STORE_CONTACT
        self.param_count = param_count

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Safe non-secret diagnostic view for UI and health-checks."""
        return {
            "api_url": self.api_url,
            "api_key_configured": bool(self.api_key),
            "waba_number": self.waba_number or "Not Configured",
            "template_name": self.template_name,
            "template_language": self.template_language,
            "store_name": self.store_name,
            "store_contact": self.store_contact,
            "param_count": self.param_count,
        }

    def preview_message(
        self,
        customer_name: str,
        medications: Union[str, List[str]],
        expected_refill_date: Union[date, str],
        tier: Optional[str] = None,
        stage_offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate structured preview and formatted text for UI inspection."""
        resolved_tier = tier or (get_tier_for_stage_offset(stage_offset) if stage_offset is not None else "DUE")
        
        dynamic_text = build_dynamic_tier_text(
            tier=resolved_tier,
            medications=medications,
            expected_refill_date=expected_refill_date,
            stage_offset=stage_offset,
            store_name=self.store_name,
            store_contact=self.store_contact,
        )

        parameters = build_template_parameters(
            customer_name=customer_name,
            tier=resolved_tier,
            medications=medications,
            expected_refill_date=expected_refill_date,
            stage_offset=stage_offset,
            store_name=self.store_name,
            store_contact=self.store_contact,
            param_count=self.param_count,
        )

        visual_preview = format_full_message_preview(
            customer_name=customer_name,
            tier=resolved_tier,
            medications=medications,
            expected_refill_date=expected_refill_date,
            stage_offset=stage_offset,
            store_name=self.store_name,
            store_contact=self.store_contact,
        )

        return {
            "tier": resolved_tier,
            "template_name": self.template_name,
            "customer_name": customer_name,
            "dynamic_body_text": dynamic_text,
            "parameters": parameters,
            "visual_preview": visual_preview,
        }

    def send_refill_reminder(
        self,
        phone_number: str,
        customer_name: str,
        medications: Union[str, List[str]],
        expected_refill_date: Union[date, str],
        tier: Optional[str] = None,
        stage_offset: Optional[int] = None,
        dry_run: bool = True,
        reminder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch or simulate a 3-tier templated WhatsApp reminder message."""
        # 1. Normalize phone
        try:
            normalized_to = normalize_e164_whatsapp_phone(phone_number)
        except ValueError as val_err:
            log_refill_send_attempt(
                customer_name=customer_name,
                phone=phone_number,
                template_name=self.template_name,
                tier=tier or "DUE",
                success=False,
                status_code=400,
                message=str(val_err),
                is_dry_run=dry_run,
            )
            return {
                "success": False,
                "status_code": 400,
                "message": f"Validation Error: {val_err}",
                "response": None,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(phone_number),
            }

        # 2. Resolve Tier and Parameters
        resolved_tier = tier or (get_tier_for_stage_offset(stage_offset) if stage_offset is not None else "DUE")
        parameters = build_template_parameters(
            customer_name=customer_name,
            tier=resolved_tier,
            medications=medications,
            expected_refill_date=expected_refill_date,
            stage_offset=stage_offset,
            store_name=self.store_name,
            store_contact=self.store_contact,
            param_count=self.param_count,
        )

        # 3. Construct Xinno JSON Payload
        payload = {
            "messaging_product": "whatsapp",
            "to": normalized_to,
            "type": "template",
            "template": {
                "language": {
                    "policy": "deterministic",
                    "code": self.template_language,
                },
                "name": self.template_name,
                "components": [
                    {
                        "type": "body",
                        "parameters": parameters,
                    }
                ],
            },
        }

        masked_headers = {
            "wabaNumber": self.waba_number or "[NOT_CONFIGURED]",
            "Key": "***MASKED***",
            "Content-Type": "application/json",
        }

        # 4. Dry-Run Simulation Mode
        if dry_run:
            simulated_id = f"DRY_RUN_{resolved_tier}_{normalized_to[-4:]}"
            msg = f"[DRY RUN] Validated 3-Tier payload for {normalized_to} (Tier: {resolved_tier}). No HTTP request sent."
            log_refill_send_attempt(
                customer_name=customer_name,
                phone=normalized_to,
                template_name=self.template_name,
                tier=resolved_tier,
                success=True,
                status_code=200,
                message=msg,
                message_id=simulated_id,
                is_dry_run=True,
            )
            return {
                "success": True,
                "status_code": 200,
                "message": msg,
                "message_id": simulated_id,
                "is_dry_run": True,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(normalized_to),
                "tier": resolved_tier,
                "response": {
                    "url": self.api_url,
                    "headers": masked_headers,
                    "payload": payload,
                    "dry_run": True,
                },
            }

        # 5. Live Mode Validation
        if not self.api_key:
            return {
                "success": False,
                "status_code": None,
                "message": "Configuration Error: XINNO_API_KEY is not set in .env.",
                "response": None,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(normalized_to),
            }

        if not self.waba_number:
            return {
                "success": False,
                "status_code": None,
                "message": "Configuration Error: XINNO_WABA_NUMBER is not set in .env.",
                "response": None,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(normalized_to),
            }

        # 6. Execute Live HTTP Request
        live_headers = {
            "wabaNumber": self.waba_number,
            "Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            http_res = requests.post(
                self.api_url,
                json=payload,
                headers=live_headers,
                timeout=15,
            )
            raw_text = http_res.text
            try:
                res_json = http_res.json()
            except Exception:
                res_json = {"raw": raw_text}

            # Extract message ID if returned
            msg_id = None
            if isinstance(res_json, dict):
                msg_id = res_json.get("message_id") or res_json.get("messageId") or res_json.get("id")
                if not msg_id and "messages" in res_json and isinstance(res_json["messages"], list) and res_json["messages"]:
                    msg_id = res_json["messages"][0].get("id")

            is_ok = 200 <= http_res.status_code < 300
            msg = "Message successfully accepted by Xinno." if is_ok else f"Xinno HTTP Error: {http_res.status_code}"

            log_refill_send_attempt(
                customer_name=customer_name,
                phone=normalized_to,
                template_name=self.template_name,
                tier=resolved_tier,
                success=is_ok,
                status_code=http_res.status_code,
                message=msg,
                message_id=msg_id,
                is_dry_run=False,
            )

            return {
                "success": is_ok,
                "status_code": http_res.status_code,
                "message": msg,
                "message_id": msg_id,
                "is_dry_run": False,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(normalized_to),
                "tier": resolved_tier,
                "response": res_json,
            }

        except Exception as exc:
            err_msg = f"Network Connection Error: {str(exc)}"
            log_refill_send_attempt(
                customer_name=customer_name,
                phone=normalized_to,
                template_name=self.template_name,
                tier=resolved_tier,
                success=False,
                status_code=500,
                message=err_msg,
                is_dry_run=False,
            )
            return {
                "success": False,
                "status_code": 500,
                "message": err_msg,
                "response": None,
                "reminder_id": reminder_id,
                "masked_phone": mask_phone_for_log(normalized_to),
                "tier": resolved_tier,
            }
