"""RefillCare 3-Tier WhatsApp Communication Package."""
from __future__ import annotations

from refillcare.whatsapp.client import (
    RefillWhatsAppClient,
    log_refill_send_attempt,
    mask_phone_for_log,
    normalize_e164_whatsapp_phone,
)
from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher
from refillcare.whatsapp.template_formatter import (
    REFILL_FUNCTION_DUE,
    REFILL_FUNCTION_FOLLOWUP,
    REFILL_FUNCTION_LAPSED,
    REFILL_FUNCTION_LABELS,
    REFILL_FUNCTION_MESSAGES,
    build_dynamic_tier_text,
    build_medicine_list_string,
    build_template_parameters,
    format_full_message_preview,
    get_function_from_tier,
    get_refill_function,
    get_tier_for_stage_offset,
)

__all__ = [
    "RefillWhatsAppClient",
    "RefillWhatsAppDispatcher",
    "REFILL_FUNCTION_DUE",
    "REFILL_FUNCTION_FOLLOWUP",
    "REFILL_FUNCTION_LAPSED",
    "REFILL_FUNCTION_LABELS",
    "REFILL_FUNCTION_MESSAGES",
    "build_dynamic_tier_text",
    "build_medicine_list_string",
    "build_template_parameters",
    "format_full_message_preview",
    "get_function_from_tier",
    "get_refill_function",
    "get_tier_for_stage_offset",
    "normalize_e164_whatsapp_phone",
    "mask_phone_for_log",
    "log_refill_send_attempt",
]
