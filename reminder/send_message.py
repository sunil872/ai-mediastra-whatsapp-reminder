"""WhatsApp Message Dispatcher Module (Xinno Gateway).

Enforces strictly controlled DRY-RUN execution mode.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from config.config import WHATSAPP_DRY_RUN
from services.xinno_whatsapp import send_template_message


def dispatch_refill_message(
    to_phone: str,
    customer_name: str,
    medication_name: str,
    refill_date: str,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Dispatch templated WhatsApp reminder message via Xinno gateway.

    Args:
        to_phone: 10-digit Indian phone number.
        customer_name: Recipient patient name.
        medication_name: Prescription medication name.
        refill_date: Formatted expected refill date.
        dry_run: Force dry-run simulation mode (defaults to True).

    Returns:
        Dict[str, Any]: Dispatch result status and message provider ID.
    """
    # Enforce global safety default if not explicitly provided
    is_dry = WHATSAPP_DRY_RUN if dry_run is None else dry_run

    return send_template_message(
        phone_number=to_phone,
        customer_name=customer_name,
        dry_run=is_dry,
    )
