"""
Verification Script: Bulk Image Campaign Dry-Run Simulation.

Executes a local dry-run simulation of the bulk image campaign pipeline on
data/image_campaign_sample.csv.

CRITICAL: ZERO real HTTP or WhatsApp requests are sent by this script.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so utils/services can be resolved
proj_root = Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import pandas as pd

from services.xinno_image_template import send_image_template_message
from utils.audit import log_audit_record_safely
from utils.bulk_send import execute_bulk_send
from utils.image_campaign import (
    IMAGE_CAMPAIGN_REQUIRED_COLUMNS,
    build_image_template_variables,
    check_image_campaign_columns,
    normalize_image_campaign_columns,
    validate_and_group_customers,
    validate_image_url,
)


def run_bulk_dry_run_verification():
    csv_path = proj_root / "data" / "image_campaign_sample.csv"
    log_path = proj_root / "logs" / "whatsapp_send.log"

    print("=" * 70)
    print("BULK IMAGE CAMPAIGN DRY-RUN VERIFICATION")
    print("=" * 70)

    # 1. Load CSV
    df = pd.read_csv(csv_path)
    raw_rows_count = len(df)
    print(f"\n1. Raw rows loaded: {raw_rows_count}")

    # 2. Normalize and check columns
    normalized_df = normalize_image_campaign_columns(df)
    missing = check_image_campaign_columns(normalized_df)
    print(f"2. Required columns present: {missing == []}")

    # 3. Validate and Group
    grouped_customers, invalid_df = validate_and_group_customers(normalized_df)
    print(f"3. Grouped customer count: {len(grouped_customers)}")
    print(f"4. Eligible customer count: {len(grouped_customers)}")
    print(f"5. Invalid rows count: {len(invalid_df)}")

    # 4. Customer Details & Medicine lists
    print("\n--- Grouped Customers Summary ---")
    for idx, cust in enumerate(grouped_customers, 1):
        print(f"\nCustomer #{idx}:")
        print(f"  Name: {cust['Name']}")
        print(f"  Normalized Phone: {cust['Normalized Phone']}")
        print(f"  Branch: {cust['Branch']}")
        print(f"  Medicine List: {cust['Medicine List']}")
        print(f"  Contact No.: {cust['Contact No.']}")
        print(f"  Manager Contact: {cust['Manager Contact']}")

    # 5. Validate Image URL
    image_url = os.getenv(
        "XINNO_IMAGE_URL",
        "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template",
    ).strip()
    is_valid_img, img_err = validate_image_url(image_url)
    print(f"\n6. Image URL: {image_url}")
    print(f"   Image URL Valid: {is_valid_img} (Error: '{img_err}')")

    # 6. Template & Store info
    template_name = os.getenv("XINNO_IMAGE_TEMPLATE_NAME", "refill_reminder_image").strip()
    template_language = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip()
    store_name = "PHARMA HUBB"
    print(f"7. Template Name: {template_name}")
    print(f"8. Template Language: {template_language}")
    print(f"9. Store Name: {store_name}")

    # 7. Bulk Dry-Run Execution
    customer_lookup = {
        (c["Name"].strip().lower(), c["Normalized Phone"]): c
        for c in grouped_customers
    }

    generated_payloads = []

    def _send_fn(phone_number, customer_name, store_name, dry_run):
        lookup_key = (
            str(customer_name).strip().lower(),
            str(phone_number).strip(),
        )
        cust = customer_lookup.get(lookup_key, {})
        res = send_image_template_message(
            phone_number=phone_number,
            customer_name=customer_name,
            store_name=store_name,
            branch=cust.get("Branch", ""),
            medicine_list=cust.get("Medicine List", ""),
            contact_no=cust.get("Contact No.", ""),
            manager_contact=cust.get("Manager Contact", ""),
            image_url=image_url,
            dry_run=dry_run,
        )
        if res.get("response", {}).get("payload"):
            generated_payloads.append(res["response"]["payload"])
        return res

    summary = execute_bulk_send(
        grouped_customers,
        store_name=store_name,
        template_name=template_name,
        template_language=template_language,
        send_fn=_send_fn,
        dry_run=True,
        bulk_attempt_id="bulk_dry_run_verification",
    )

    print("\n--- Bulk Dry-Run Summary ---")
    print(f"Total Recipients Attempted: {summary['attempted']}")
    print(f"Successful Dry-Run Simulations: {summary['successful']}")
    print(f"Failed Simulations: {summary['failed']}")
    print(f"Payloads Generated: {len(generated_payloads)}")

    # 8. Inspect Generated Payloads
    print("\n--- Generated Payloads Inspection ---")
    for idx, p in enumerate(generated_payloads, 1):
        print(f"\n[Payload #{idx}] Destination: {p['to']}")
        print(f"  Template: {p['template']['name']} ({p['template']['language']['code']})")
        print(f"  Header Image: {p['template']['components'][0]['parameters'][0]['image']['link']}")
        print("  Body Parameters (8 variables):")
        for v_idx, param in enumerate(p['template']['components'][1]['parameters'], 1):
            print(f"    {{{{{v_idx}}}}}: {repr(param['text'])}")

    # 9. Log Audit Records safely
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for rec in summary["records"]:
        log_audit_record_safely(rec, str(log_path))

    print(f"\n10. Audit log entries recorded safely in: {log_path}")
    print("11. Real Xinno HTTP requests made: 0 (DRY-RUN ONLY)")
    print("=" * 70)


if __name__ == "__main__":
    run_bulk_dry_run_verification()
