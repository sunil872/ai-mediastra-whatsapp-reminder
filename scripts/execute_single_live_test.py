"""
Controlled Single Live Send Execution Script

Performs exactly ONE real WhatsApp send for the test customer:
  Sunil / 917659935016
using the approved template 'refill_reminder_image' and verified Cloudinary image URL.

Never retries automatically.
Never logs or exposes API secrets.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Ensure repo root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.image_campaign import (
    normalize_image_campaign_columns,
    validate_and_group_customers,
    validate_image_url,
)
from services.xinno_image_template import send_image_template_message


def run_live_test():
    # 1. Load environment
    load_dotenv(dotenv_path=_ROOT / ".env", override=True)
    
    csv_path = _ROOT / "data" / "image_campaign_single_test_customer.csv"
    if not csv_path.exists():
        print(f"ERROR: Test CSV file not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path, dtype=str)
    df = normalize_image_campaign_columns(df)
    grouped, invalid = validate_and_group_customers(df)
    
    # 2. Safety assertions
    assert len(grouped) == 1, f"Expected exactly 1 customer, got {len(grouped)}"
    assert len(invalid) == 0, f"Expected 0 invalid records, got {len(invalid)}"
    
    customer = grouped[0]
    assert customer["Name"].strip().lower() == "sunil", f"Unexpected customer name: {customer['Name']}"
    assert customer["Normalized Phone"] == "917659935016", f"Unexpected phone: {customer['Normalized Phone']}"
    
    store_name = os.getenv("MEDICAL_STORE_NAME", "PHARMA HUBB").strip() or "PHARMA HUBB"
    template_name = os.getenv("XINNO_IMAGE_TEMPLATE_NAME", "refill_reminder_image").strip()
    assert template_name == "refill_reminder_image", f"Unexpected template name: {template_name}"
    
    image_url = os.getenv(
        "XINNO_IMAGE_URL",
        "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template",
    ).strip()
    
    valid_url, url_err = validate_image_url(image_url)
    assert valid_url, f"Invalid image URL: {url_err}"
    assert "cloudinary.com" in image_url, f"Image URL is not Cloudinary: {image_url}"
    
    print("=" * 60)
    print("PRE-SEND VALIDATION PASSED:")
    print(f"Customer:       {customer['Name']}")
    print(f"Phone:          {customer['Normalized Phone']}")
    print(f"Branch:         {customer['Branch']}")
    print(f"Medicines:      {customer['Medicine Count']} items")
    print(f"Template:       {template_name}")
    print(f"Image URL:      {image_url}")
    print(f"Dry Run:        FALSE (Executing 1 live send)")
    print("=" * 60)
    
    # 3. Execute exactly ONE live send
    res = send_image_template_message(
        phone_number=customer["Normalized Phone"],
        customer_name=customer["Name"],
        store_name=store_name,
        branch=customer.get("Branch", ""),
        medicine_list=customer.get("Medicine List", ""),
        contact_no=customer.get("Contact No.", ""),
        manager_contact=customer.get("Manager Contact", ""),
        image_url=image_url,
        dry_run=False,
    )
    
    print("\nXINNO API LIVE RESPONSE RESULT:")
    print(f"Success:      {res.get('success')}")
    print(f"Status Code:  {res.get('status_code')}")
    print(f"Message:      {res.get('message')}")
    
    api_resp = res.get("response", {}).get("api_response")
    print(f"API Response: {json.dumps(api_resp, indent=2) if api_resp else 'None'}")
    print("=" * 60)
    
    return res


if __name__ == "__main__":
    run_live_test()
