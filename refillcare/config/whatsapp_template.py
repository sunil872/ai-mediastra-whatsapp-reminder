"""RefillCare Approved WhatsApp Business Template Configuration.

Central single source of truth for the approved RefillCare WhatsApp reminder template
specifications, metadata, variable ordering, and reference payload format.
"""

from __future__ import annotations
from typing import Dict, Any, List

# Approved WhatsApp Template Metadata
WHATSAPP_TEMPLATE_NAME: str = "refillcare_medicine_reminder"
WHATSAPP_TEMPLATE_CATEGORY: str = "Utility"
WHATSAPP_TEMPLATE_TYPE: str = "text"
WHATSAPP_TEMPLATE_LANGUAGE: str = "en"
WHATSAPP_TEMPLATE_POLICY: str = "deterministic"

# Exact Approved WhatsApp Template Body Text
WHATSAPP_TEMPLATE_BODY: str = """Dear *{{1}}*,

This is a friendly reminder from *{{2}}* regarding your regular medicine refill.

{{3}}

💊 Medicines:
{{4}}

If you need a refill, please contact us or place your order with our pharmacy.

📞 Contact: {{5}}

Thank you for choosing *{{6}}*.
🙏 We are happy to serve you."""

# Exact 6-Variable Order Mapping (Fixed)
WHATSAPP_TEMPLATE_VARIABLES: Dict[str, str] = {
    "1": "customer_name",
    "2": "store_name",
    "3": "reminder_stage_message",
    "4": "medicine",
    "5": "store_contact",
    "6": "store_name",
}

# Variable Descriptions
WHATSAPP_VARIABLE_DESCRIPTIONS: Dict[str, str] = {
    "1": "Customer / Patient Name (sanitized string)",
    "2": "Medical Store Name (e.g. PHARMA HUBB)",
    "3": "Dynamic Reminder-Stage Clinical Message from Scheduler",
    "4": "Medication Name / Prescribed Item List",
    "5": "Store Contact Number for ordering",
    "6": "Medical Store Name (closing greeting: Thank you for choosing *{{6}}*)",
}


def get_template_metadata() -> Dict[str, Any]:
    """Return dictionary of approved template configuration."""
    return {
        "name": WHATSAPP_TEMPLATE_NAME,
        "category": WHATSAPP_TEMPLATE_CATEGORY,
        "type": WHATSAPP_TEMPLATE_TYPE,
        "language": WHATSAPP_TEMPLATE_LANGUAGE,
        "policy": WHATSAPP_TEMPLATE_POLICY,
        "body": WHATSAPP_TEMPLATE_BODY,
        "variables": WHATSAPP_TEMPLATE_VARIABLES,
        "total_variables": len(WHATSAPP_TEMPLATE_VARIABLES),
    }


def build_template_components(
    customer_name: str,
    store_name: str,
    stage_message: str,
    medicine: str,
    store_contact: str,
) -> List[Dict[str, Any]]:
    """Build the exact 6-parameter BODY component array for the WhatsApp API payload.

    Args:
        customer_name: Variable {{1}}
        store_name: Variable {{2}} and Variable {{6}}
        stage_message: Variable {{3}}
        medicine: Variable {{4}}
        store_contact: Variable {{5}}

    Returns:
        List containing the BODY component with 6 ordered parameters.
    """
    return [
        {
            "type": "BODY",
            "parameters": [
                {"type": "text", "text": str(customer_name)},
                {"type": "text", "text": str(store_name)},
                {"type": "text", "text": str(stage_message)},
                {"type": "text", "text": str(medicine)},
                {"type": "text", "text": str(store_contact)},
                {"type": "text", "text": str(store_name)},
            ],
        }
    ]
