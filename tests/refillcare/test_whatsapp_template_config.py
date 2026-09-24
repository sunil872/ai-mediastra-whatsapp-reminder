"""Unit tests for RefillCare Approved WhatsApp Business Template Configuration."""

from refillcare.config.whatsapp_template import (
    WHATSAPP_TEMPLATE_NAME,
    WHATSAPP_TEMPLATE_CATEGORY,
    WHATSAPP_TEMPLATE_TYPE,
    WHATSAPP_TEMPLATE_LANGUAGE,
    WHATSAPP_TEMPLATE_POLICY,
    WHATSAPP_TEMPLATE_BODY,
    WHATSAPP_TEMPLATE_VARIABLES,
    WHATSAPP_VARIABLE_DESCRIPTIONS,
    get_template_metadata,
    build_template_components,
)


def test_template_metadata_constants():
    """Verify core template metadata constants."""
    assert WHATSAPP_TEMPLATE_NAME == "refillcare_medicine_reminder"
    assert WHATSAPP_TEMPLATE_CATEGORY == "Utility"
    assert WHATSAPP_TEMPLATE_TYPE == "text"
    assert WHATSAPP_TEMPLATE_LANGUAGE == "en"
    assert WHATSAPP_TEMPLATE_POLICY == "deterministic"


def test_template_variable_count_and_ordering():
    """Verify that exactly 6 variables are defined in the exact fixed order."""
    assert len(WHATSAPP_TEMPLATE_VARIABLES) == 6
    expected_mapping = {
        "1": "customer_name",
        "2": "store_name",
        "3": "reminder_stage_message",
        "4": "medicine",
        "5": "store_contact",
        "6": "store_name",
    }
    assert WHATSAPP_TEMPLATE_VARIABLES == expected_mapping


def test_template_body_text_content():
    """Verify that template body matches the approved wording and variable placeholders."""
    assert "Dear *{{1}}*," in WHATSAPP_TEMPLATE_BODY
    assert "This is a friendly reminder from *{{2}}* regarding your regular medicine refill." in WHATSAPP_TEMPLATE_BODY
    assert "{{3}}" in WHATSAPP_TEMPLATE_BODY
    assert "💊 Medicines:\n{{4}}" in WHATSAPP_TEMPLATE_BODY
    assert "📞 Contact: {{5}}" in WHATSAPP_TEMPLATE_BODY
    assert "Thank you for choosing *{{6}}*." in WHATSAPP_TEMPLATE_BODY
    assert "🙏 We are happy to serve you." in WHATSAPP_TEMPLATE_BODY


def test_get_template_metadata_dictionary():
    """Verify the metadata helper returns a complete, valid dictionary."""
    meta = get_template_metadata()
    assert meta["name"] == "refillcare_medicine_reminder"
    assert meta["category"] == "Utility"
    assert meta["total_variables"] == 6
    assert meta["variables"]["6"] == "store_name"


def test_build_template_components():
    """Verify construction of 6-parameter BODY component array."""
    components = build_template_components(
        customer_name="Alice Patient",
        store_name="CITY PHARMACY",
        stage_message="Your regular refill is due tomorrow, 10-09-2026.",
        medicine="METFORMIN 500MG",
        store_contact="9876500000",
    )

    assert len(components) == 1
    body_comp = components[0]
    assert body_comp["type"] == "BODY"
    params = body_comp["parameters"]
    assert len(params) == 6

    assert params[0] == {"type": "text", "text": "Alice Patient"}
    assert params[1] == {"type": "text", "text": "CITY PHARMACY"}
    assert params[2] == {"type": "text", "text": "Your regular refill is due tomorrow, 10-09-2026."}
    assert params[3] == {"type": "text", "text": "METFORMIN 500MG"}
    assert params[4] == {"type": "text", "text": "9876500000"}
    assert params[5] == {"type": "text", "text": "CITY PHARMACY"}
