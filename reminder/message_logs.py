"""Message Delivery Logs & Audit Trail Module."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from config.config import WHATSAPP_LOG_PATH


def append_message_audit_log(
    phone_number: str,
    customer_name: str,
    medication_name: str,
    status: str,
    message_id: Optional[str] = None,
    is_dry_run: bool = True,
    error_detail: Optional[str] = None,
) -> None:
    """Log individual message dispatch event to persistent audit log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "phone_number": phone_number,
        "customer_name": customer_name,
        "medication_name": medication_name,
        "status": status,
        "message_id": message_id or "DRY_RUN_ID",
        "is_dry_run": is_dry_run,
        "error_detail": error_detail,
    }

    try:
        with open(WHATSAPP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        pass
