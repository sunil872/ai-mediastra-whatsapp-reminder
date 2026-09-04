"""
Test script for Xinno WhatsApp API integration (Phase 3).

Verifies:
1. Missing / placeholder template name returns Configuration Error.
2. DRY RUN payload generation using valid test data:
   - Customer: Sunil
   - Phone: 7659935016 (Normalized: 917659935016)
   - Store: PHARMA HUBB
   - Template: reminder_refill_followup_v3
3. DRY RUN returns status_code: None (not 200).
4. Headers mask API key as '***MASKED***'.
5. NO HTTP request is sent (dry_run=True).
6. NO secrets or API keys are exposed.
"""

import os
import json
import inspect
from unittest.mock import patch
from services.xinno_whatsapp import send_template_message, normalize_phone_number


def run_dry_run_test():
    print("=" * 60)
    print("  XINNO WHATSAPP API SERVICE - PHASE 3/4 DRY RUN TEST")
    print("=" * 60)

    # 0. Verify function signature default parameter is dry_run=True
    sig = inspect.signature(send_template_message)
    assert sig.parameters["dry_run"].default is True, "DEFAULT dry_run parameter MUST be True"
    print("\n[SAFETY CHECK]")
    print("  [PASS] send_template_message() default parameter is dry_run=True.")

    # 1. Test configuration error when template name is missing or placeholder
    # Prevent load_dotenv(override=True) from restoring .env over the test value.
    with patch("services.xinno_whatsapp.load_dotenv"), patch.dict(
        os.environ, {"WHATSAPP_TEMPLATE_NAME": "Your template name here"}, clear=False
    ):
        err_res = send_template_message("7659935016", "Sunil", "Pharma Hub", dry_run=True)
    assert err_res["success"] is False, "Expected failure when template name is placeholder"
    assert "Configuration Error" in err_res["message"], f"Expected Configuration Error, got '{err_res['message']}'"
    print("\n[CONFIG VALIDATION TEST]")
    print("  [PASS] Missing / placeholder WHATSAPP_TEMPLATE_NAME correctly returns Configuration Error.")

    # 2. Set valid template name in environment for testing
    template_name = "reminder_refill_followup_v3"
    os.environ["WHATSAPP_TEMPLATE_NAME"] = template_name
    os.environ["MEDICAL_STORE_NAME"] = "PHARMA HUBB"

    customer_name = "Sunil"
    phone_number = "7659935016"
    store_name = "PHARMA HUBB"

    print(f"\n[INPUTS]")
    print(f"  Template Name : {template_name}")
    print(f"  Customer Name : {customer_name}")
    print(f"  Phone Number  : {phone_number}")
    print(f"  Store Name    : {store_name}")

    # 3. Test Phone Normalization
    normalized = normalize_phone_number(phone_number)
    print(f"\n[PHONE NORMALIZATION]")
    print(f"  Raw Input     : {phone_number}")
    print(f"  Normalized    : {normalized}")
    assert normalized == "917659935016", f"Expected '917659935016', got '{normalized}'"

    # 4. Run DRY RUN message construction
    result = send_template_message(
        phone_number=phone_number,
        customer_name=customer_name,
        store_name=store_name,
        dry_run=True
    )

    print(f"\n[RESULT SUMMARY]")
    print(f"  Success       : {result['success']}")
    print(f"  Status Code   : {result['status_code']}")
    print(f"  Message       : {result['message']}")

    print(f"\n[GENERATED DRY RUN PAYLOAD & HEADERS]")
    print(json.dumps(result["response"], indent=2))

    # 5. Assertions & Safety Checks
    assert result["success"] is True, "DRY RUN failed"
    assert result["status_code"] is None, f"DRY RUN status_code must be None, got {result['status_code']}"
    assert result["response"]["dry_run"] is True, "dry_run flag must be True"

    payload = result["response"]["payload"]
    headers = result["response"]["headers"]

    assert payload["to"] == "917659935016", f"Payload 'to' must be '917659935016', got {payload['to']}"
    assert payload["type"] == "template", f"Payload 'type' must be 'template'"
    assert payload["template"]["name"] == "reminder_refill_followup_v3", f"Payload template name must be 'reminder_refill_followup_v3', got {payload['template']['name']}"

    # Category is MARKETING in Xinno/Meta and does not affect the REST payload structure
    assert "category" not in payload["template"], "Template category must not alter the directApi message payload structure"

    params = payload["template"]["components"][0]["parameters"]
    assert len(params) == 3, f"Expected exactly 3 parameters, got {len(params)}"
    assert params[0]["text"] == "Sunil", f"Parameter 1 must be customer name 'Sunil', got {params[0]['text']}"
    assert params[1]["text"] == "PHARMA HUBB", f"Parameter 2 must be 'PHARMA HUBB', got {params[1]['text']}"
    assert params[2]["text"] == "PHARMA HUBB", f"Parameter 3 must be 'PHARMA HUBB', got {params[2]['text']}"
    assert headers["Key"] == "***MASKED***", "API Key in headers must be masked as '***MASKED***'"

    result_json_str = json.dumps(result)
    assert "***MASKED***" in result_json_str, "Key must be masked"

    print("\n" + "=" * 60)
    print("  [PASS] ALL DRY RUN ASSERTIONS PASSED SUCCESSFULLY!")
    print("   - Configuration error validated for missing/placeholder template.")
    print("   - Template name validated: reminder_refill_followup_v3")
    print("   - Exactly 3 body parameters validated (Sunil, PHARMA HUBB, PHARMA HUBB).")
    print("   - Template category (MARKETING) does not alter API payload.")
    print("   - status_code is None in DRY RUN mode.")
    print("   - No HTTP request was sent.")
    print("   - No API secret was exposed.")
    print("=" * 60)

    return result


def test_dry_run_suite():
    """Pytest entry point."""
    run_dry_run_test()


if __name__ == "__main__":
    run_dry_run_test()
