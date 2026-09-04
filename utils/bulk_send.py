"""
Controlled bulk WhatsApp send helpers.

Sequential only — one customer at a time. No threads, no async fan-out, no retries.
Uses the existing send_template_message service (injected for tests).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from utils.validators import (
    WHATSAPP_TEMPLATE_NAME,
    generate_message,
    get_template_variable_mapping,
)
from utils.audit import build_audit_record, mask_phone_for_audit


SendFn = Callable[..., Dict[str, Any]]


def confirmation_required(confirmed: bool) -> bool:
    """Bulk send is allowed only when explicit confirmation is True."""
    return bool(confirmed)


def get_eligible_customers(valid_df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Return Status=Valid customers only (duplicates are already excluded from valid_df).
    Each dict is self-contained: Name + Normalized Phone from the SAME row.
    """
    if valid_df is None or valid_df.empty:
        return []
    eligible: List[Dict[str, str]] = []
    for _, row in valid_df.iterrows():
        if str(row.get("Status", "Valid")).strip() != "Valid":
            continue
        name = str(row.get("Name", "")).strip()
        normalized = str(row.get("Normalized Phone", "")).strip()
        if not name or not normalized:
            continue
        eligible.append({
            "Name": name,
            "Phone number": str(row.get("Phone number", normalized)),
            "Original Phone": str(row.get("Original Phone", "")),
            "Normalized Phone": normalized,
            "Status": "Valid",
        })
    return eligible


def build_bulk_summary(
    total_rows: int,
    valid_count: int,
    invalid_count: int,
    duplicate_count: int,
    eligible_count: int,
    template_name: str = WHATSAPP_TEMPLATE_NAME,
    template_language: str = "en",
    pharmacy_name: str = "PHARMA HUBB",
) -> Dict[str, Any]:
    """Preview summary shown before bulk send."""
    return {
        "total_uploaded_rows": int(total_rows),
        "valid_customers": int(valid_count),
        "invalid_customers": int(invalid_count),
        "duplicate_customers": int(duplicate_count),
        "eligible_for_sending": int(eligible_count),
        "template_name": template_name,
        "template_language": template_language,
        "pharmacy_name": pharmacy_name,
    }


def build_sample_message_previews(
    eligible: List[Dict[str, str]],
    store_name: str,
    limit: int = 3,
) -> List[Dict[str, str]]:
    """Dynamic personalized previews — never hard-codes a customer name."""
    samples: List[Dict[str, str]] = []
    for customer in eligible[: max(0, limit)]:
        name = customer["Name"]
        mapping = get_template_variable_mapping(name, store_name)
        samples.append({
            "customer_name": name,
            "normalized_phone": customer["Normalized Phone"],
            "original_phone": customer.get("Original Phone", ""),
            "message_preview": generate_message(name, store_name),
            "var1": mapping["{{1}}"],
            "var2": mapping["{{2}}"],
            "var3": mapping["{{3}}"],
        })
    return samples


def bulk_confirmation_ready(confirmed: bool) -> bool:
    """Explicit confirmation required before production bulk send."""
    return confirmation_required(confirmed)


def new_bulk_attempt_id() -> str:
    return str(uuid.uuid4())


def execute_bulk_send(
    eligible: List[Dict[str, str]],
    *,
    store_name: str,
    template_name: str = WHATSAPP_TEMPLATE_NAME,
    template_language: str = "en",
    send_fn: SendFn,
    dry_run: bool,
    bulk_attempt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sequentially send to each eligible customer.

    - Independent payload per customer (same-row name + phone)
    - Continues after failures (no stop-on-error)
    - No automatic retries
    - Returns summary + per-customer audit records

    dry_run must be False only for explicitly confirmed production bulk send.
    Tests should pass dry_run=True or a mocked send_fn.
    """
    bulk_id = bulk_attempt_id or new_bulk_attempt_id()
    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0
    skipped = 0

    if not eligible:
        return {
            "bulk_attempt_id": bulk_id,
            "eligible": 0,
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "dry_run": dry_run,
            "records": [],
            "message": "No valid customers are available for sending.",
        }

    for customer in eligible:
        name = customer["Name"]
        phone = customer["Normalized Phone"]
        # Safety: never send without both fields from the same row
        if not name or not phone:
            skipped += 1
            continue

        send_response = send_fn(
            phone_number=phone,
            customer_name=name,
            store_name=store_name,
            dry_run=dry_run,
        )
        attempt_id = str(uuid.uuid4())
        record = build_audit_record(
            customer_name=name,
            original_phone=customer.get("Original Phone", ""),
            normalized_phone=phone,
            template_name=template_name,
            template_language=template_language,
            pharmacy_name=store_name,
            dry_run=dry_run,
            send_response=send_response,
            attempt_id=attempt_id,
            bulk_attempt_id=bulk_id,
        )
        results.append(record)
        if record.get("success"):
            successful += 1
        else:
            failed += 1

    return {
        "bulk_attempt_id": bulk_id,
        "eligible": len(eligible),
        "attempted": successful + failed,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "dry_run": dry_run,
        "records": results,
        "message": "BULK SEND COMPLETED" if not dry_run else "BULK DRY-RUN COMPLETED",
    }


def bulk_results_table(records: List[Dict[str, Any]], mask_phones: bool = True) -> pd.DataFrame:
    """UI/export-friendly results table."""
    rows = []
    for r in records:
        phone = r.get("normalized_phone", "")
        if mask_phones:
            phone = mask_phone_for_audit(phone)
        rows.append({
            "Customer": r.get("customer_name", ""),
            "Phone": phone,
            "Status": "Accepted" if r.get("success") else "Failed",
            "Message ID": r.get("message_id") or "",
            "Error": r.get("error") or "",
            "Bulk Attempt ID": r.get("bulk_attempt_id") or "",
        })
    return pd.DataFrame(
        rows,
        columns=["Customer", "Phone", "Status", "Message ID", "Error", "Bulk Attempt ID"],
    )
