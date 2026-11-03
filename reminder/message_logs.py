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
        # Patient identity is never written to the audit file in clear (last 4 digits / initial only)
        "phone_number": f"******{str(phone_number or '')[-4:]}",
        "customer_name": (str(customer_name or "").strip()[:1] + "***") if customer_name else None,
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
