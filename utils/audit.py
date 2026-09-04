"""
Phase 7: Send-result audit records, session history helpers, CSV export.

No database. No live sending. Phone masking for display/logs/export.
Never includes API keys or credentials.
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from utils.validators import WHATSAPP_TEMPLATE_NAME
except ModuleNotFoundError:
    # Running as `python utils/audit.py` — script dir is on sys.path, not project root
    from validators import WHATSAPP_TEMPLATE_NAME

AUDIT_COLUMNS = [
    "timestamp",
    "customer_name",
    "original_phone",
    "normalized_phone",
    "template_name",
    "template_language",
    "pharmacy_name",
    "dry_run",
    "success",
    "status_code",
    "message",
    "message_id",
    "message_status",
    "error",
    "attempt_id",
    "bulk_attempt_id",
]

_SECRET_PATTERNS = re.compile(
    r"(api[_-]?key|authorization|x-api-key|bearer\s+\S+|XINNO_API_KEY)",
    re.IGNORECASE,
)


def mask_phone_for_audit(phone: str) -> str:
    """
    Mask middle digits for logs/UI/export.
    Example: 917659935016 -> 917******016
    """
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    if len(digits) < 6:
        return "***"
    if len(digits) >= 10:
        return f"{digits[:3]}{'*' * (len(digits) - 6)}{digits[-3:]}"
    return f"{digits[:2]}{'*' * (len(digits) - 4)}{digits[-2:]}"


def extract_message_id(api_response: Any) -> Optional[str]:
    """Best-effort WhatsApp/Xinno message id extraction."""
    if api_response is None:
        return None
    if isinstance(api_response, dict):
        if api_response.get("messageId"):
            return str(api_response["messageId"])
        if api_response.get("id") and not isinstance(api_response.get("id"), (dict, list)):
            # Prefer messages[].id when present
            pass
        messages = api_response.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("id"):
                return str(first["id"])
        if api_response.get("id") and not isinstance(api_response.get("id"), (dict, list)):
            return str(api_response["id"])
    if isinstance(api_response, list):
        for item in api_response:
            mid = extract_message_id(item)
            if mid:
                return mid
    return None


def extract_message_status(api_response: Any) -> Optional[str]:
    """Best-effort message_status from Xinno/Meta response."""
    if api_response is None:
        return None
    if isinstance(api_response, dict):
        messages = api_response.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            status = messages[0].get("message_status") or messages[0].get("status")
            if status is not None:
                return str(status)
        if api_response.get("message_status"):
            return str(api_response["message_status"])
        if api_response.get("status") is True:
            return "accepted"
        if api_response.get("status") is False:
            return "failed"
    if isinstance(api_response, list) and api_response:
        return extract_message_status(api_response[0])
    return None


def _safe_error_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    # Strip anything that looks like a secret key value pattern from env placeholders
    if _SECRET_PATTERNS.search(s):
        s = _SECRET_PATTERNS.sub("[REDACTED]", s)
    # Never leave raw Key header values
    s = re.sub(r'"Key"\s*:\s*"[^"]*"', '"Key": "***MASKED***"', s)
    return s


def build_audit_record(
    *,
    customer_name: str,
    original_phone: str,
    normalized_phone: str,
    template_name: str = WHATSAPP_TEMPLATE_NAME,
    template_language: str = "en",
    pharmacy_name: str = "PHARMA HUBB",
    dry_run: bool,
    send_response: Optional[Dict[str, Any]] = None,
    attempt_id: Optional[str] = None,
    bulk_attempt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a structured send-result audit record.
    Never includes API keys or auth headers.
    """
    send_response = send_response or {}
    response_block = send_response.get("response") if isinstance(send_response, dict) else None
    api_response = None
    if isinstance(response_block, dict):
        api_response = response_block.get("api_response")

    success = bool(send_response.get("success")) if send_response else False
    status_code = send_response.get("status_code")
    raw_message = send_response.get("message") or ""

    message_id = extract_message_id(api_response)
    message_status = extract_message_status(api_response)
    if message_status is None and success:
        message_status = "accepted" if not dry_run else "dry_run"
    if message_status is None and not success and send_response:
        message_status = "failed"

    if dry_run and success:
        display_message = "Dry-run only — no WhatsApp message was sent."
        error = None
    elif success:
        display_message = "WhatsApp request accepted by Xinno."
        error = None
    else:
        display_message = "WhatsApp request failed."
        error = _safe_error_text(raw_message) or "WhatsApp request failed."

    record = {
        "attempt_id": attempt_id or str(uuid.uuid4()),
        "bulk_attempt_id": bulk_attempt_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "customer_name": str(customer_name or ""),
        "original_phone": str(original_phone or ""),
        "normalized_phone": str(normalized_phone or ""),
        "template_name": str(template_name or WHATSAPP_TEMPLATE_NAME),
        "template_language": str(template_language or "en"),
        "pharmacy_name": str(pharmacy_name or "PHARMA HUBB"),
        "dry_run": bool(dry_run),
        "success": success,
        "status_code": status_code,
        "message": display_message,
        "message_id": message_id,
        "message_status": message_status,
        "error": error,
    }

    # Absolute guarantee: no API key / credential fields
    for banned in ("api_key", "API_KEY", "Key", "authorization", "XINNO_API_KEY"):
        record.pop(banned, None)
    return record


def audit_record_for_display(record: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of audit record with masked phones for UI/logging."""
    out = dict(record)
    out["original_phone_masked"] = mask_phone_for_audit(record.get("original_phone", ""))
    out["normalized_phone_masked"] = mask_phone_for_audit(record.get("normalized_phone", ""))
    return out


def append_send_history(
    history: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Append one audit record if attempt_id is new.
    Prevents duplicate entries on Streamlit reruns.
    """
    if history is None:
        history = []
    attempt_id = record.get("attempt_id")
    if attempt_id and any(r.get("attempt_id") == attempt_id for r in history):
        return history
    history.append(dict(record))
    return history


def history_to_dataframe(
    history: List[Dict[str, Any]],
    mask_phones: bool = True,
) -> pd.DataFrame:
    """Build a compact history table (phones masked by default)."""
    if not history:
        return pd.DataFrame(
            columns=[
                "Time",
                "Customer",
                "Phone",
                "Template",
                "Result",
                "API Status",
                "Message ID",
            ]
        )

    rows = []
    for r in history:
        phone = r.get("normalized_phone", "")
        if mask_phones:
            phone = mask_phone_for_audit(phone)
        result = "Accepted" if r.get("success") else "Failed"
        if r.get("dry_run"):
            result = f"Dry-run ({'OK' if r.get('success') else 'Failed'})"
        rows.append({
            "Time": r.get("timestamp", ""),
            "Customer": r.get("customer_name", ""),
            "Phone": phone,
            "Template": r.get("template_name", ""),
            "Result": result,
            "API Status": r.get("message_status") or ("accepted" if r.get("success") else "failed"),
            "Message ID": r.get("message_id") or "",
        })
    return pd.DataFrame(rows)


def history_to_csv_bytes(
    history: List[Dict[str, Any]],
    mask_phones: bool = True,
) -> bytes:
    """
    Export session history as CSV bytes.
    Phones masked. No credentials.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in history:
        row = {col: r.get(col) for col in AUDIT_COLUMNS}
        if mask_phones:
            row["original_phone"] = mask_phone_for_audit(str(row.get("original_phone") or ""))
            row["normalized_phone"] = mask_phone_for_audit(str(row.get("normalized_phone") or ""))
        # Ensure secrets never appear
        for key in list(row.keys()):
            if key and "key" in key.lower() and key != "attempt_id":
                row.pop(key, None)
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def log_audit_record_safely(record: Dict[str, Any], log_path: Optional[str] = None) -> str:
    """
    Format a safe one-line log entry (phones masked, no API key).
    Optionally append to logs/whatsapp_send.log when log_path provided.
    """
    display = audit_record_for_display(record)
    line = (
        f"[AUDIT] attempt_id={display.get('attempt_id')} "
        f"customer='{display.get('customer_name')}' "
        f"phone='{display.get('normalized_phone_masked')}' "
        f"template='{display.get('template_name')}' "
        f"dry_run={display.get('dry_run')} "
        f"success={display.get('success')} "
        f"status={display.get('status_code')} "
        f"message_id='{display.get('message_id') or '-'}' "
        f"error='{display.get('error') or '-'}'"
    )
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] INFO: {line}\n"
                )
        except OSError:
            pass
    return line


if __name__ == "__main__":
    # Quick self-check only — not a live Xinno send.
    sample = build_audit_record(
        customer_name="Sunil",
        original_phone="76599 35016",
        normalized_phone="917659935016",
        dry_run=True,
        send_response={
            "success": True,
            "status_code": None,
            "message": "[DRY RUN] ok",
            "response": {"dry_run": True},
        },
    )
    print("audit.py self-check OK")
    print(f"  template : {sample['template_name']}")
    print(f"  dry_run  : {sample['dry_run']}")
    print(f"  masked   : {mask_phone_for_audit(sample['normalized_phone'])}")
    print("Use: streamlit run app.py  OR  python -m pytest test_phase7_audit.py")
