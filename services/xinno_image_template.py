"""
Xinno WhatsApp API Integration Service — Image + Text Template

Handles payload construction, validation, and message delivery
for Image + Text template messages via the Xinno WhatsApp API.

DOCUMENTATION SOURCE OF TRUTH:
docs/WhatsappAPIDocument.json  (Request: "Template- Image")

API Endpoint:  POST /REST/directApi/message
Headers:       wabaNumber, Key, Content-Type: application/json
Body structure (from Postman collection):
{
    "to": "{{ToMobile}}",
    "type": "template",
    "template": {
        "language": {
            "policy": "deterministic",
            "code": "en"
        },
        "name": "{{template_name}}",
        "components": [
            {
                "type": "header",
                "parameters": [
                    {
                        "type": "image",
                        "image": {
                            "link": "{{image_url}}"
                        }
                    }
                ]
            },
            {
                "type": "body",
                "parameters": [
                    {"type": "text", "text": "{{1}}"},
                    {"type": "text", "text": "{{2}}"},
                    ...
                ]
            }
        ]
    }
}

NOTE: The Postman "Template- Image" entry does NOT include
"messaging_product": "whatsapp".  This service follows the
documented payload exactly.  If Xinno requires that field,
it can be added to the payload without changing the rest of
the structure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from utils.image_campaign import build_image_template_variables

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("xinno_image_template")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

# Matches the existing text-template service default
DEFAULT_API_URL = "https://cpaasreseller.notify24x7.com/REST/directApi/message"


# ---------------------------------------------------------------------------
# Internal helpers (same patterns as xinno_whatsapp.py)
# ---------------------------------------------------------------------------

def _mask_phone_for_log(phone: str) -> str:
    """Mask middle digits for safe logging (e.g. 917659935016 -> 91******5016)."""
    digits = "".join(c for c in str(phone) if c.isdigit())
    if len(digits) < 6:
        return "***"
    return f"{digits[:2]}{'*' * (len(digits) - 6)}{digits[-4:]}"


def _format_safe_exception_details(exc: Exception, api_key: str) -> Dict[str, str]:
    """Categorise network errors safely, masking any API key in the message."""
    exc_type = type(exc).__name__
    raw_msg = str(exc)

    safe_msg = raw_msg.replace(api_key, "***MASKED***") if api_key and api_key in raw_msg else raw_msg

    lowered = safe_msg.lower()
    if "nameresolutionerror" in lowered or "getaddrinfo failed" in lowered or "dns" in lowered:
        category = "DNS Resolution Error"
    elif "ssl" in lowered or "certificate" in lowered:
        category = "TLS/SSL Certificate Error"
    elif "timed out" in lowered or "timeout" in lowered:
        category = "Connection Timeout"
    elif "connection refused" in lowered:
        category = "TCP Connection Refused"
    else:
        category = "Network Connection Error"

    return {
        "error_type": exc_type,
        "error_category": category,
        "error_detail": safe_msg,
    }


def _log_send_attempt(
    customer_name: str,
    phone: str,
    template_name: str,
    success: bool,
    status_code: Optional[int],
    message: str,
    message_id: Optional[str] = None,
    xinno_status: Optional[str] = None,
) -> None:
    """Append a safe log line to logs/whatsapp_send.log.  Never logs API keys."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "whatsapp_send.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)

        log_inst = logging.getLogger("xinno_image_template_file")
        log_inst.setLevel(logging.INFO)
        if not log_inst.handlers:
            log_inst.addHandler(file_handler)

        status_str = "SUCCESS" if success else "FAILED"
        masked_phone = _mask_phone_for_log(phone)
        xinno_status = xinno_status or ("accepted" if success else "failed")
        mid = message_id or "-"
        log_inst.info(
            f"[{status_str}] [IMAGE-TEMPLATE] Customer='{customer_name}', "
            f"Phone='{masked_phone}', Template='{template_name}', "
            f"Status={status_code}, XinnoStatus='{xinno_status}', "
            f"MessageId='{mid}', Message='{message}'"
        )
    except Exception as log_err:
        logger.warning(f"Failed to write to log file: {log_err}")


# ---------------------------------------------------------------------------
# Normalise phone (re-use from existing service)
# ---------------------------------------------------------------------------

def normalize_phone_number(phone_number: str, country_code: str = "91") -> str:
    """Delegate to the existing normaliser in xinno_whatsapp.py."""
    from services.xinno_whatsapp import normalize_phone_number as _normalize
    return _normalize(phone_number, country_code)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_image_template_message(
    phone_number: str,
    customer_name: str,
    store_name: str,
    branch: str = "",
    medicine_list: str = "",
    contact_no: str = "",
    manager_contact: str = "",
    image_url: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Construct and send an Image + Text template message via the Xinno API.

    Payload follows docs/WhatsappAPIDocument.json "Template- Image".

    The 8 body variables (in order):
        1. customer_name  2. store_name  3. branch
        4. medicine_list  5. contact_no  6. manager_contact
        7. store_name     8. branch

    Args:
        phone_number:    Recipient mobile number.
        customer_name:   Customer name (body parameter 1).
        store_name:      Medical store name (body parameters 2 & 7).
        branch:          Branch / location (body parameters 3 & 8).
        medicine_list:   Pre-formatted bullet medicine list (body parameter 4).
        contact_no:      Store contact number (body parameter 5).
        manager_contact: Manager contact number (body parameter 6).
        image_url:       Publicly accessible image URL for the header.
                         Falls back to XINNO_IMAGE_URL env var if omitted.
        dry_run:         If True, constructs payload without HTTP call.

    Returns:
        Structured response dict (same shape as send_template_message).
    """
    # 1. Load environment
    _proj_root = Path(__file__).resolve().parent.parent
    _dotenv = _proj_root / ".env"
    load_dotenv(dotenv_path=_dotenv, override=True)

    api_url = os.getenv("XINNO_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    api_key = os.getenv("XINNO_API_KEY", "").strip()
    waba_number = os.getenv("XINNO_WABA_NUMBER", "").strip()
    template_name = os.getenv("XINNO_IMAGE_TEMPLATE_NAME", "").strip()
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"
    effective_image_url = (image_url or os.getenv("XINNO_IMAGE_URL", "")).strip()

    # 2. Validate configuration
    if not template_name:
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: XINNO_IMAGE_TEMPLATE_NAME is not set in environment.",
            "response": None,
        }

    if not effective_image_url:
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: Image URL is required. Set XINNO_IMAGE_URL in environment or provide image_url parameter.",
            "response": None,
        }

    if not (effective_image_url.startswith("https://") or effective_image_url.startswith("http://")):
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: Image URL must be a publicly accessible HTTP/HTTPS URL, not a local filesystem path.",
            "response": None,
        }

    # 3. Validate inputs
    if not customer_name or not str(customer_name).strip():
        return {
            "success": False,
            "status_code": None,
            "message": "Validation Error: Customer name is required.",
            "response": None,
        }

    if not store_name or not str(store_name).strip():
        return {
            "success": False,
            "status_code": None,
            "message": "Validation Error: Store name is required.",
            "response": None,
        }

    # 4. Normalise phone
    try:
        normalized_phone = normalize_phone_number(phone_number)
    except ValueError as val_err:
        return {
            "success": False,
            "status_code": None,
            "message": f"Validation Error: {str(val_err)}",
            "response": None,
        }

    # 5. Build customer dict for variable generator
    customer_dict = {
        "Name": str(customer_name).strip(),
        "Branch": str(branch).strip(),
        "Medicine List": str(medicine_list).strip(),
        "Contact No.": str(contact_no).strip(),
        "Manager Contact": str(manager_contact).strip(),
    }
    body_parameters = build_image_template_variables(customer_dict, store_name)

    # 6. Construct JSON payload per docs/WhatsappAPIDocument.json "Template- Image"
    payload = {
        "to": normalized_phone,
        "type": "template",
        "template": {
            "language": {
                "policy": "deterministic",
                "code": template_language,
            },
            "name": template_name,
            "components": [
                {
                    "type": "header",
                    "parameters": [
                        {
                            "type": "image",
                            "image": {
                                "link": effective_image_url,
                            },
                        }
                    ],
                },
                {
                    "type": "body",
                    "parameters": body_parameters,
                },
            ],
        },
    }

    # 7. Construct headers
    effective_waba = waba_number if waba_number else "[XINNO_WABA_NUMBER_NOT_SET]"
    raw_headers = {
        "wabaNumber": effective_waba,
        "Key": api_key if api_key else "[XINNO_API_KEY_NOT_SET]",
        "Content-Type": "application/json",
    }
    masked_headers = {
        "wabaNumber": effective_waba,
        "Key": "***MASKED***",
        "Content-Type": "application/json",
    }

    # 8. DRY RUN
    if dry_run:
        logger.info(f"[DRY RUN] Image template message created for {normalized_phone}")
        return {
            "success": True,
            "status_code": None,
            "message": "[DRY RUN] Payload successfully constructed. No HTTP request was sent.",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": True,
            },
        }

    # 9. Live-mode credential checks
    if not api_key:
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: XINNO_API_KEY is not set in environment.",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
            },
        }

    if not waba_number:
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: XINNO_WABA_NUMBER is not set in environment.",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
            },
        }

    # 10. Execute HTTP request (live mode)
    try:
        http_response = requests.post(
            api_url,
            headers=raw_headers,
            json=payload,
            timeout=15,
        )

        try:
            resp_data = http_response.json()
        except Exception:
            resp_data = http_response.text

        has_error = isinstance(resp_data, dict) and ("error" in resp_data or "errors" in resp_data)
        is_success = (http_response.status_code in (200, 201)) and not has_error

        if has_error:
            err_obj = resp_data.get("error") or resp_data.get("errors")
            if isinstance(err_obj, dict):
                err_code = err_obj.get("code", "Unknown")
                err_msg = err_obj.get("message", str(err_obj))
                msg_text = f"Meta/Xinno API Error (#{err_code}): {err_msg}"
            else:
                msg_text = f"Meta/Xinno API Error: {str(err_obj)}"
        elif is_success:
            msg_text = "Xinno API accepted the request."
        else:
            msg_text = f"API returned status {http_response.status_code}"

        # Best-effort message id extraction
        message_id = None
        if isinstance(resp_data, dict):
            messages = resp_data.get("messages")
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                message_id = messages[0].get("id")
            message_id = message_id or resp_data.get("messageId") or resp_data.get("id")

        _log_send_attempt(
            customer_name=customer_name,
            phone=normalized_phone,
            template_name=template_name,
            success=is_success,
            status_code=http_response.status_code,
            message=msg_text,
            message_id=str(message_id) if message_id else None,
            xinno_status="accepted" if is_success else "failed",
        )

        return {
            "success": is_success,
            "status_code": http_response.status_code,
            "message": msg_text,
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "api_response": resp_data,
                "dry_run": False,
            },
        }

    except requests.exceptions.Timeout as time_err:
        diag = _format_safe_exception_details(time_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Connection Error ({diag['error_category']}): Request to Xinno API timed out (15s).",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag,
            },
        }
    except requests.exceptions.ConnectionError as conn_err:
        diag = _format_safe_exception_details(conn_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Connection Error ({diag['error_category']}): {diag['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag,
            },
        }
    except requests.exceptions.RequestException as req_err:
        diag = _format_safe_exception_details(req_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"HTTP Error ({diag['error_category']}): {diag['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag,
            },
        }
    except Exception as exc:
        diag = _format_safe_exception_details(exc, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Unexpected Error ({diag['error_category']}): {diag['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag,
            },
        }
