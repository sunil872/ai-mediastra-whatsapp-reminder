"""
Step 18: Controlled Bulk Live WhatsApp Test Execution Script.

Sends exactly ONE live campaign execution to the 3 verified recipients in
data/image_campaign_sample.csv:
1. Sunil -> 917659935016
2. Tarun -> 918688504571
3. Ram   -> 917661087360

Safety Gates:
- Strictly exactly 3 eligible recipients
- Template 'refill_reminder_image', Language 'en'
- Single-line sanitized parameter values
- Public HTTPS Cloudinary Image URL
- No retry on failure
- No second execution
- Masked audit logging without secrets
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
proj_root = Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=proj_root / ".env", override=True)

import pandas as pd

from services.xinno_image_template import send_image_template_message
from utils.audit import log_audit_record_safely
from utils.bulk_send import execute_bulk_send
from utils.column_aliases import normalize_dataframe_columns
from utils.image_campaign import (
    build_image_template_variables,
    validate_and_group_customers,
    validate_image_url,
)


def run_controlled_bulk_live_test():
    csv_path = proj_root / "data" / "image_campaign_sample.csv"
    log_path = proj_root / "logs" / "whatsapp_send.log"

    print("=" * 75)
    print("STEP 18: CONTROLLED BULK LIVE WHATSAPP TEST (3 RECIPIENTS)")
    print("=" * 75)

    # 1. PRE-LIVE VALIDATION
    print("\n--- Phase 1: Pre-Live Validation ---")
    raw_df = pd.read_csv(csv_path)
    print(f"Raw rows loaded: {len(raw_df)}")

    norm_df, alias_res = normalize_dataframe_columns(raw_df)
    if not alias_res.is_valid:
        print(f"FATAL: Alias resolution failed: {alias_res.errors}")
        sys.exit(1)
    print("Alias Resolution: SUCCESS")

    grouped_customers, invalid_df = validate_and_group_customers(norm_df)
    print(f"Grouped customers count: {len(grouped_customers)}")
    print(f"Eligible recipients count: {len(grouped_customers)}")
    print(f"Invalid rows count: {len(invalid_df)}")

    if len(grouped_customers) != 3 or len(invalid_df) != 0:
        print(f"FATAL: Expected exactly 3 eligible recipients and 0 invalid rows. Got {len(grouped_customers)} and {len(invalid_df)}.")
        sys.exit(1)

    expected_recipients = {
        "Sunil": "917659935016",
        "Tarun": "918688504571",
        "Ram": "917661087360",
    }

    for cust in grouped_customers:
        name = cust["Name"]
        phone = cust["Normalized Phone"]
        if name not in expected_recipients or expected_recipients[name] != phone:
            print(f"FATAL: Unexpected recipient {name} -> {phone}")
            sys.exit(1)
        # Verify single-line medicine list
        meds = cust["Medicine List"]
        if "\n" in meds or "\r" in meds or "\t" in meds or "  " in meds:
            print(f"FATAL: Medicine list for {name} contains invalid characters: {repr(meds)}")
            sys.exit(1)
        print(f"  Verified Recipient: {name} -> {phone} | Branch: {cust['Branch']} | Medicines: {meds}")

    # 2. IMAGE URL & TEMPLATE VALIDATION
    image_url = os.getenv(
        "XINNO_IMAGE_URL",
        "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template",
    ).strip()
    is_valid_img, img_err = validate_image_url(image_url)
    if not is_valid_img:
        print(f"FATAL: Invalid image URL: {img_err}")
        sys.exit(1)

    template_name = os.getenv("XINNO_IMAGE_TEMPLATE_NAME", "refill_reminder_image").strip()
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"
    store_name = os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB"

    if template_name != "refill_reminder_image":
        print(f"FATAL: Invalid template name: {template_name}")
        sys.exit(1)

    print(f"Template Name: {template_name} ({template_language})")
    print(f"Header Image: {image_url}")
    print(f"Store Name: {store_name}")

    # 3. LIVE SEND EXECUTION
    print("\n--- Phase 2: Controlled Live Send (dry_run=False) ---")
    print("Initiating exactly 1 bulk send execution to 3 recipients...")

    customer_lookup = {
        (c["Name"].strip().lower(), c["Normalized Phone"]): c
        for c in grouped_customers
    }

    real_request_count = 0
    live_results = []

    def _live_send_fn(phone_number, customer_name, store_name, dry_run):
        nonlocal real_request_count
        real_request_count += 1
        lookup_key = (
            str(customer_name).strip().lower(),
            str(phone_number).strip(),
        )
        cust = customer_lookup.get(lookup_key, {})
        print(f"  [Sending {real_request_count}/3] Calling Xinno API for {customer_name} ({phone_number})...")
        res = send_image_template_message(
            phone_number=phone_number,
            customer_name=customer_name,
            store_name=store_name,
            branch=cust.get("Branch", ""),
            medicine_list=cust.get("Medicine List", ""),
            contact_no=cust.get("Contact No.", ""),
            manager_contact=cust.get("Manager Contact", ""),
            image_url=image_url,
            dry_run=False,  # REAL LIVE SEND
        )
        live_results.append({
            "customer_name": customer_name,
            "phone_number": phone_number,
            "response": res,
        })
        return res

    bulk_attempt_id = f"step18_live_bulk_{uuid.uuid4().hex[:8]}"
    summary = execute_bulk_send(
        grouped_customers,
        store_name=store_name,
        template_name=template_name,
        template_language=template_language,
        send_fn=_live_send_fn,
        dry_run=False,
        bulk_attempt_id=bulk_attempt_id,
    )

    print(f"\nBulk Send Execution Finished.")
    print(f"Real Xinno HTTP Requests Made: {real_request_count}")
    print(f"Total Attempted: {summary['attempted']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    # 4. RESPONSE INSPECTION
    print("\n--- Phase 3: Response Inspection ---")
    for idx, item in enumerate(live_results, 1):
        name = item["customer_name"]
        phone = item["phone_number"]
        res = item["response"]
        success = res.get("success")
        status_code = res.get("status_code")
        raw_resp = res.get("response") or {}

        # Extract wamid and status from Meta/Xinno response format
        messages = raw_resp.get("messages", []) if isinstance(raw_resp, dict) else []
        wamid = messages[0].get("id") if messages and isinstance(messages[0], dict) else "-"
        msg_status = messages[0].get("message_status") if messages and isinstance(messages[0], dict) else ("accepted" if success else "failed")

        print(f"\n[Recipient #{idx}] {name} ({phone}):")
        print(f"  HTTP Status Code: {status_code}")
        print(f"  Xinno / Meta Status: {msg_status}")
        print(f"  Message ID (wamid): {wamid}")
        print(f"  Success Flag: {success}")
        print(f"  API Message: {res.get('message')}")
        if not success:
            print(f"  Error details: {raw_resp}")

    # 5. AUDIT LOGGING
    print("\n--- Phase 4: Audit Logging ---")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for rec in summary["records"]:
        log_audit_record_safely(rec, str(log_path))

    print(f"Audit log recorded safely in: {log_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_controlled_bulk_live_test()
