"""3-Tier Dynamic WhatsApp Template Message Formatter for RefillCare.

Provides deterministic message composition across:
- DUE_REFILL (Stages -7d, -3d, -1d, 0d): "Your regular medicine refill is due soon."
- REFILL_FOLLOW_UP (Stages +2d, +5d): "We’re following up regarding your regular medicine refill."
- LAPSED_REENGAGEMENT (Stage +45d): "We’re checking in regarding your regular medicine refill."

Template Structure:
Dear *{{1}}*,

This is a friendly reminder from *{{2}}* regarding your regular medicine refill.

{{3}}

💊 Medicines:
{{4}}

If you need a refill, please contact us or place your order with our pharmacy.

📞 Contact: {{5}}

Thank you for choosing *{{6}}*.
🙏 We are happy to serve you.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Union

# Constant Refill Functions
REFILL_FUNCTION_DUE = "DUE_REFILL"
REFILL_FUNCTION_FOLLOWUP = "REFILL_FOLLOW_UP"
REFILL_FUNCTION_LAPSED = "LAPSED_REENGAGEMENT"

REFILL_FUNCTION_MESSAGES = {
    REFILL_FUNCTION_DUE: "Your regular medicine refill is due soon.",
    REFILL_FUNCTION_FOLLOWUP: "We’re following up regarding your regular medicine refill.",
    REFILL_FUNCTION_LAPSED: "We’re checking in regarding your regular medicine refill.",
}

REFILL_FUNCTION_LABELS = {
    REFILL_FUNCTION_DUE: "Due / Advance (DUE_REFILL)",
    REFILL_FUNCTION_FOLLOWUP: "Follow-up (REFILL_FOLLOW_UP)",
    REFILL_FUNCTION_LAPSED: "+45d Re-engagement (LAPSED_REENGAGEMENT)",
}


def get_refill_function(stage_offset: Union[int, float, str]) -> str:
    """Classify stage offset into one of 3 explicit refill functions."""
    try:
        offset = int(stage_offset)
    except (TypeError, ValueError):
        offset = 0

    if offset <= 0:
        return REFILL_FUNCTION_DUE
    elif offset in (2, 5):
        return REFILL_FUNCTION_FOLLOWUP
    elif offset in (40, 45):
        return REFILL_FUNCTION_LAPSED
    return REFILL_FUNCTION_DUE


def get_tier_for_stage_offset(stage_offset: int) -> str:
    """Classify stage offset into one of 3 isolated lifecycle tiers: DUE, FOLLOWUP, or LAPSED."""
    func = get_refill_function(stage_offset)
    if func == REFILL_FUNCTION_FOLLOWUP:
        return "FOLLOWUP"
    elif func == REFILL_FUNCTION_LAPSED:
        return "LAPSED"
    return "DUE"


def get_function_from_tier(tier: str) -> str:
    """Convert tier label/string into standardized refill function."""
    t = str(tier).strip().upper()
    if t in ("LAPSED", "RE-ENGAGEMENT", "REENGAGEMENT", "LAPSED_REENGAGEMENT", "TIER_3"):
        return REFILL_FUNCTION_LAPSED
    elif t in ("FOLLOWUP", "FOLLOW_UP", "REFILL_FOLLOW_UP", "TIER_2"):
        return REFILL_FUNCTION_FOLLOWUP
    return REFILL_FUNCTION_DUE


def build_medicine_list_string(medications: Union[str, List[str]]) -> str:
    """Format single or multiple medications into a clean comma-separated string."""
    if isinstance(medications, list):
        clean_meds = [str(m).strip() for m in medications if str(m).strip()]
        return ", ".join(clean_meds) if clean_meds else "Prescription Medication"
    raw_str = str(medications).strip()
    return raw_str if raw_str else "Prescription Medication"


def build_dynamic_tier_text(
    tier: str,
    medications: Union[str, List[str]],
    expected_refill_date: Union[date, str],
    stage_offset: Optional[int] = None,
    store_name: str = "PHARMA HUBB",
    store_contact: str = "+91 9966473474",
) -> str:
    """Return the exact dynamic message string for {{3}} based on the 3 functions."""
    if stage_offset is not None:
        func = get_refill_function(stage_offset)
    else:
        func = get_function_from_tier(tier)

    return REFILL_FUNCTION_MESSAGES.get(func, REFILL_FUNCTION_MESSAGES[REFILL_FUNCTION_DUE])


def build_template_parameters(
    customer_name: str,
    tier: str,
    medications: Union[str, List[str]],
    expected_refill_date: Union[date, str],
    stage_offset: Optional[int] = None,
    store_name: str = "PHARMA HUBB",
    store_contact: str = "+91 9966473474",
    param_count: int = 6,
) -> List[Dict[str, str]]:
    """Build the exact Xinno template components parameters array.

    Supported Parameter Formats:
    - 6 Params (Official RefillCare Template `refillcare_medicine_reminder`):
      {{1}} Customer Name
      {{2}} Store Name
      {{3}} Function Message (DUE_REFILL / REFILL_FOLLOW_UP / LAPSED_REENGAGEMENT)
      {{4}} Medicines List (comma-separated)
      {{5}} Contact Number
      {{6}} Store Name
    - 3 Params (Legacy format): [{{1}}: Customer Name, {{2}}: Dynamic Message, {{3}}: Store Name]
    - 2 Params: [{{1}}: Customer Name, {{2}}: Dynamic Message]
    """
    clean_cust = str(customer_name).strip() or "Valued Customer"
    clean_store = str(store_name).strip() or "PHARMA HUBB"
    clean_contact = str(store_contact).strip() or "+91 9966473474"
    med_list_str = build_medicine_list_string(medications)
    
    tier_msg = build_dynamic_tier_text(
        tier=tier,
        medications=medications,
        expected_refill_date=expected_refill_date,
        stage_offset=stage_offset,
        store_name=clean_store,
        store_contact=clean_contact,
    )

    if param_count == 6:
        texts = [
            clean_cust,       # {{1}}
            clean_store,      # {{2}}
            tier_msg,         # {{3}}
            med_list_str,     # {{4}}
            clean_contact,    # {{5}}
            clean_store,      # {{6}}
        ]
    elif param_count == 4:
        texts = [clean_cust, tier_msg, clean_contact, clean_store]
    elif param_count == 2:
        texts = [clean_cust, tier_msg]
    else:  # 3 params
        texts = [clean_cust, tier_msg, clean_store]

    return [{"type": "text", "text": t} for t in texts]


def format_full_message_preview(
    customer_name: str,
    tier: str,
    medications: Union[str, List[str]],
    expected_refill_date: Union[date, str],
    stage_offset: Optional[int] = None,
    store_name: str = "PHARMA HUBB",
    store_contact: str = "+91 9966473474",
) -> str:
    """Generate a high-fidelity visual simulation of the official WhatsApp template."""
    clean_cust = str(customer_name).strip() or "Valued Customer"
    clean_store = str(store_name).strip() or "PHARMA HUBB"
    clean_contact = str(store_contact).strip() or "+91 9966473474"
    med_list_str = build_medicine_list_string(medications)

    func = get_refill_function(stage_offset) if stage_offset is not None else get_function_from_tier(tier)
    tier_msg = REFILL_FUNCTION_MESSAGES.get(func, REFILL_FUNCTION_MESSAGES[REFILL_FUNCTION_DUE])

    tier_badge = (
        "🟢 DUE_REFILL" if func == REFILL_FUNCTION_DUE
        else ("🟡 REFILL_FOLLOW_UP" if func == REFILL_FUNCTION_FOLLOWUP else "🔴 LAPSED_REENGAGEMENT")
    )

    return (
        f"📱 [WhatsApp Template: refillcare_medicine_reminder — {tier_badge}]\n"
        f"────────────────────────────────────────────────────────────\n"
        f"Dear *{clean_cust}*,\n\n"
        f"This is a friendly reminder from *{clean_store}* regarding your regular medicine refill.\n\n"
        f"{tier_msg}\n\n"
        f"💊 Medicines:\n"
        f"{med_list_str}\n\n"
        f"If you need a refill, please contact us or place your order with our pharmacy.\n\n"
        f"📞 Contact: {clean_contact}\n\n"
        f"Thank you for choosing *{clean_store}*.\n"
        f"🙏 We are happy to serve you.\n"
        f"────────────────────────────────────────────────────────────"
    )
