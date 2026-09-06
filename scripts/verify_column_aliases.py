"""
Verification Script: Production CSV Column Alias System Check.

Validates alias resolution across alternate real-world client CSV formats
and verifies dry-run simulation.

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
from utils.column_aliases import (
    COLUMN_ALIASES,
    REQUIRED_CANONICAL_FIELDS,
    normalize_dataframe_columns,
    resolve_column_aliases,
)
from utils.image_campaign import (
    build_image_template_variables,
    validate_and_group_customers,
    validate_image_url,
)


def run_column_alias_verification():
    csv_path = proj_root / "data" / "image_campaign_sample.csv"
    log_path = proj_root / "logs" / "whatsapp_send.log"

    print("=" * 75)
    print("PRODUCTION CSV COLUMN ALIASES VERIFICATION")
    print("=" * 75)

    # 1. Alias System Stats
    print("\n--- Central Column Alias Configuration ---")
    total_aliases = sum(len(aliases) for aliases in COLUMN_ALIASES.values())
    print(f"Canonical Fields: {list(COLUMN_ALIASES.keys())}")
    print(f"Total Supported Aliases: {total_aliases}")

    # 2. Test Alternate CSV Formats
    print("\n--- Alternate Real-World Client Formats ---")
    formats = {
        "FORMAT 1 (Sample CSV)": [
            "Name", "Phone number", "Medicine", "Branch",
            "Customer Medication List", "Contact No.", "Manager Contact"
        ],
        "FORMAT 2 (Pharmacy/Client)": [
            "Patient Name", "WhatsApp Number", "Medication",
            "Outlet", "Store Contact Number", "Manager Mobile"
        ],
        "FORMAT 3 (Excel-style)": [
            " customer_name", "mobile_number", "medicine_name",
            "branch_name", "contact_number", "manager_number"
        ],
        "FORMAT 4 (Mixed Capitalization/Spacing)": [
            "CUSTOMER NAME", "Mobile No", "Medicine Name",
            "Branch / Location", "Store Phone", "Manager Contact No"
        ],
    }

    for fname, cols in formats.items():
        res = resolve_column_aliases(cols)
        print(f"  [{fname}] Valid: {res.is_valid} | Mapped: {len(res.detected_mappings)} fields")

    # 3. Test Missing and Ambiguous Edge Cases
    print("\n--- Validation & Safety Gates Check ---")
    missing_test = resolve_column_aliases(["Patient Name", "WhatsApp Number", "Outlet"])
    print(f"  Missing Column Detection: {'PASSED' if not missing_test.is_valid and 'medicine' in missing_test.missing_required else 'FAILED'}")

    ambig_test = resolve_column_aliases(["Patient Name", "WhatsApp Number", "Mobile No", "Medication", "Outlet"])
    print(f"  Ambiguity Detection: {'PASSED' if not ambig_test.is_valid and 'phone' in ambig_test.ambiguities else 'FAILED'}")

    # 4. Dry-Run Execution on Sample CSV
    print("\n--- Bulk Dry-Run on Official Sample CSV ---")
    raw_df = pd.read_csv(csv_path)
    print(f"Raw rows loaded: {len(raw_df)}")

    norm_df, alias_res = normalize_dataframe_columns(raw_df)
    print(f"Alias Resolution Valid: {alias_res.is_valid}")
    if alias_res.unmapped_columns:
        print(f"Unmapped Extra Columns (safely ignored): {alias_res.unmapped_columns}")

    grouped_customers, invalid_df = validate_and_group_customers(norm_df)
    print(f"Grouped customer count: {len(grouped_customers)}")
    print(f"Eligible customer count: {len(grouped_customers)}")
    print(f"Invalid rows count: {len(invalid_df)}")

    image_url = os.getenv(
        "XINNO_IMAGE_URL",
        "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template",
    ).strip()
    is_valid_img, _ = validate_image_url(image_url)
    print(f"Image URL Valid: {is_valid_img}")

    store_name = "PHARMA HUBB"
    template_name = "refill_reminder_image"
    template_language = "en"

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
        bulk_attempt_id="column_alias_verification",
    )

    print(f"Total attempted: {summary['attempted']}")
    print(f"Successful simulations: {summary['successful']}")
    print(f"Failed simulations: {summary['failed']}")
    print(f"Payloads generated: {len(generated_payloads)}")

    for idx, cust in enumerate(grouped_customers, 1):
        print(f"\nRecipient #{idx}:")
        print(f"  Name: {cust['Name']}")
        print(f"  Destination: {cust['Normalized Phone']}")
        print(f"  Branch: {cust['Branch']}")
        print(f"  Medicines: {cust['Medicine List']}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    for rec in summary["records"]:
        log_audit_record_safely(rec, str(log_path))

    print(f"\nAudit entries safely recorded in: {log_path}")
    print("Real Xinno HTTP requests made: 0 (DRY-RUN ONLY)")
    print("=" * 75)


if __name__ == "__main__":
    run_column_alias_verification()
