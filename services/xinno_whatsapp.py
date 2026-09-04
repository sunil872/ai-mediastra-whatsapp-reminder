"""
Xinno WhatsApp API Integration Service

Handles payload construction, phone number normalization, and message delivery
via the Xinno WhatsApp API (Template- Text format).

DOCUMENTATION SOURCE OF TRUTH:
docs/WhatsappAPIDocument.json (Request: "Template- Text")

API Endpoint: POST /REST/directApi/message
Headers: wabaNumber, Key, Content-Type: application/json
Body structure:
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
                "type": "body",
                "parameters": [
                    {"type": "text", "text": "{{customer_name}}"},
                    {"type": "text", "text": "{{store_name}}"},
                    {"type": "text", "text": "{{store_name}}"}
                ]
            }
        ]
    }
}
"""

import os
import logging
from pathlib import Path
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger("xinno_whatsapp")

# Explicitly load .env from project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

# Default API URL from Postman collection in docs/WhatsappAPIDocument.json
DEFAULT_API_URL = "https://cpaasreseller.notify24x7.com/REST/directApi/message"


def _mask_phone_for_log(phone: str) -> str:
    """Mask middle digits for safe logging (e.g. 917659935016 -> 91******5016)."""
    digits = "".join(c for c in str(phone) if c.isdigit())
    if len(digits) < 6:
        return "***"
    return f"{digits[:2]}{'*' * (len(digits) - 6)}{digits[-4:]}"


def get_config_diagnostic() -> Dict[str, str]:
    """
    Safe configuration diagnostic.
    Shows non-secret values (URL, WABA, template, language, store).
    NEVER exposes API key — only 'configured' / 'not configured'.
    """
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DOTENV_PATH = _PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

    api_key_set = bool(os.getenv("XINNO_API_KEY", "").strip())
    return {
        "XINNO_API_URL": os.getenv("XINNO_API_URL", "").strip() or "not configured",
        "XINNO_API_KEY": "configured" if api_key_set else "not configured",
        "XINNO_WABA_NUMBER": os.getenv("XINNO_WABA_NUMBER", "").strip() or "not configured",
        "WHATSAPP_TEMPLATE_NAME": os.getenv("WHATSAPP_TEMPLATE_NAME", "").strip() or "not configured",
        "WHATSAPP_TEMPLATE_LANGUAGE": os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en",
        "MEDICAL_STORE_NAME": os.getenv("MEDICAL_STORE_NAME", "").strip() or "not configured",
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
):
    """
    Log send attempt details safely to logs/whatsapp_send.log.
    Guarantees NO credentials or API keys are logged. Phone is masked.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "whatsapp_send.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)

        log_inst = logging.getLogger("xinno_whatsapp_file")
        log_inst.setLevel(logging.INFO)
        if not log_inst.handlers:
            log_inst.addHandler(file_handler)

        status_str = "SUCCESS" if success else "FAILED"
        masked_phone = _mask_phone_for_log(phone)
        xinno_status = xinno_status or ("accepted" if success else "failed")
        mid = message_id or "-"
        log_inst.info(
            f"[{status_str}] Customer='{customer_name}', Phone='{masked_phone}', "
            f"Template='{template_name}', Status={status_code}, "
            f"XinnoStatus='{xinno_status}', MessageId='{mid}', Message='{message}'"
        )
    except Exception as log_err:
        logger.warning(f"Failed to write to log file: {log_err}")


def normalize_phone_number(phone_number: str, country_code: str = "91") -> str:
    """
    Normalize phone number to country-code format (e.g., 917659935016)
    as required by the Xinno API (ToMobile parameter).

    NOTE: Indian 10-digit mobile numbers are prepended with country code '91'.
    Verification with a controlled test on your Xinno account is recommended
    before live bulk sending to confirm account-specific format requirements.

    Args:
        phone_number: Raw phone number string.
        country_code: Country code prefix without '+' (default: '91').

    Returns:
        Normalized phone number string with country code.
    """
    if not phone_number:
        raise ValueError("Phone number cannot be empty.")

    # Remove all non-digit characters except leading '+'
    cleaned = str(phone_number).strip()
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]

    # Remove spaces, hyphens, brackets, dots
    cleaned = "".join(c for c in cleaned if c.isdigit())

    if not cleaned:
        raise ValueError(f"Invalid phone number containing no digits: '{phone_number}'")

    # If 10 digits, prepend country code
    if len(cleaned) == 10:
        return f"{country_code}{cleaned}"

    # If 11 digits starting with '0', strip leading '0' and prepend country code
    if len(cleaned) == 11 and cleaned.startswith("0"):
        return f"{country_code}{cleaned[1:]}"

    # If 12 digits starting with country code, return as is
    if len(cleaned) == 12 and cleaned.startswith(country_code):
        return cleaned

    # If number already includes country code or another length, return digits
    return cleaned


def _format_safe_exception_details(exc: Exception, api_key: str) -> Dict[str, str]:
    """
    Format exception details safely without exposing secrets.
    Categorizes the network failure (DNS / TCP / SSL / Timeout / Generic Connection).
    """
    exc_type = type(exc).__name__
    raw_msg = str(exc)

    # Sanitize API key from exception text if present
    if api_key and api_key in raw_msg:
        safe_msg = raw_msg.replace(api_key, "***MASKED***")
    else:
        safe_msg = raw_msg

    # Categorize error
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


def send_template_message(
    phone_number: str,
    customer_name: str,
    store_name: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Construct and send a template text message using the Xinno WhatsApp API.

    Args:
        phone_number: Recipient mobile number.
        customer_name: Customer name (Body Parameter 1).
        store_name: Medical store name (Body Parameter 2). Falls back to env var if omitted.
        dry_run: If True, constructs and validates payload/headers without making HTTP call.

    Returns:
        Structured response dictionary:
        {
            "success": True/False,
            "status_code": int or None,
            "message": str,
            "response": dict (safe, masked headers and payload)
        }
    """
    # 1. Load configuration from environment variables
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DOTENV_PATH = _PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

    api_url = os.getenv("XINNO_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
    api_key = os.getenv("XINNO_API_KEY", "").strip()
    waba_number = os.getenv("XINNO_WABA_NUMBER", "").strip()
    template_name = os.getenv("WHATSAPP_TEMPLATE_NAME", "").strip()
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"
    default_store_name = os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB"

    # Determine store name
    effective_store_name = (store_name or default_store_name or "PHARMA HUBB").strip()

    # 2. Configuration & Input Validation
    if not template_name or template_name == "Your template name here":
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: WHATSAPP_TEMPLATE_NAME is not set in environment or is using placeholder 'Your template name here'.",
            "response": None
        }

    if not customer_name or not str(customer_name).strip():
        return {
            "success": False,
            "status_code": None,
            "message": "Validation Error: Customer name is required.",
            "response": None
        }

    if not effective_store_name:
        return {
            "success": False,
            "status_code": None,
            "message": "Validation Error: Medical store name is required.",
            "response": None
        }

    # Normalize phone number
    try:
        normalized_phone = normalize_phone_number(phone_number)
    except ValueError as val_err:
        return {
            "success": False,
            "status_code": None,
            "message": f"Validation Error: {str(val_err)}",
            "response": None
        }

    # 3. Construct JSON Payload according to Xinno WhatsApp API documentation
    payload = {
        "messaging_product": "whatsapp",
        "to": normalized_phone,
        "type": "template",
        "template": {
            "language": {
                "policy": "deterministic",
                "code": template_language
            },
            "name": template_name,
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {
                            "type": "text",
                            "text": str(customer_name).strip()
                        },
                        {
                            "type": "text",
                            "text": effective_store_name
                        },
                        {
                            "type": "text",
                            "text": effective_store_name
                        }
                    ]
                }
            ]
        }
    }

    # Construct headers
    effective_waba = waba_number if waba_number else "[XINNO_WABA_NUMBER_NOT_SET]"
    raw_headers = {
        "wabaNumber": effective_waba,
        "Key": api_key if api_key else "[XINNO_API_KEY_NOT_SET]",
        "Content-Type": "application/json"
    }

    # Safe headers for display/logging (Key is ALWAYS masked)
    masked_headers = {
        "wabaNumber": effective_waba,
        "Key": "***MASKED***",
        "Content-Type": "application/json"
    }

    # 4. DRY RUN Mode
    if dry_run:
        logger.info(f"[DRY RUN] Template message created for {normalized_phone}")
        return {
            "success": True,
            "status_code": None,
            "message": "[DRY RUN] Payload successfully constructed. No HTTP request was sent.",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": True
            }
        }

    # 5. Live sending mode checks
    if not api_key:
        return {
            "success": False,
            "status_code": None,
            "message": "Configuration Error: XINNO_API_KEY is not set in environment.",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False
            }
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
                "dry_run": False
            }
        }

    # 6. Execute HTTP Request (Live Mode)
    try:
        http_response = requests.post(
            api_url,
            headers=raw_headers,
            json=payload,
            timeout=15
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

        # Best-effort message id for safe logging (never logs API key)
        message_id = None
        if isinstance(resp_data, dict):
            messages = resp_data.get("messages")
            if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                message_id = messages[0].get("id")
            message_id = message_id or resp_data.get("messageId") or resp_data.get("id")

        # Safe File Logging (NO credentials logged)
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
                "dry_run": False
            }
        }

    except requests.exceptions.Timeout as time_err:
        diag_info = _format_safe_exception_details(time_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Connection Error ({diag_info['error_category']}): Request to Xinno WhatsApp API timed out (15s).",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag_info
            }
        }
    except requests.exceptions.ConnectionError as conn_err:
        diag_info = _format_safe_exception_details(conn_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Connection Error ({diag_info['error_category']}): {diag_info['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag_info
            }
        }
    except requests.exceptions.RequestException as req_err:
        diag_info = _format_safe_exception_details(req_err, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"HTTP Error ({diag_info['error_category']}): {diag_info['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag_info
            }
        }
    except Exception as exc:
        diag_info = _format_safe_exception_details(exc, api_key)
        return {
            "success": False,
            "status_code": None,
            "message": f"Unexpected Error ({diag_info['error_category']}): {diag_info['error_detail']}",
            "response": {
                "url": api_url,
                "headers": masked_headers,
                "payload": payload,
                "dry_run": False,
                "error_details": diag_info
            }
        }
