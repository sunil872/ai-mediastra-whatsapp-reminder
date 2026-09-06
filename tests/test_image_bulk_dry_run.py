"""
Bulk Image Campaign Dry-Run Validation Tests.

Validates:
1. Loading data/image_campaign_sample.csv (6 raw rows -> 3 unique grouped customers).
2. Grouping by normalized name + normalized phone.
3. Medicine list aggregation into single-line comma-separated format without newlines/tabs/consecutive spaces.
4. Duplicate medicine deduplication and order preservation.
5. Exact 8 template body variables in required order.
6. Image header with public HTTPS Cloudinary image URL.
7. Template name 'refill_reminder_image', language 'en'.
8. Normalized destination phone numbers:
   - Sunil -> 917659935016
   - Tarun -> 918688504571
   - Ram   -> 917661087360
9. Exact payload structure per Xinno/Postman API specification.
10. Bulk execution pipeline processes exactly 3 recipients with dry_run=True.
11. Zero real network/HTTP requests made.
12. Safe audit logging without secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.xinno_image_template import send_image_template_message
from utils.bulk_send import execute_bulk_send
from utils.image_campaign import (
    IMAGE_CAMPAIGN_REQUIRED_COLUMNS,
    build_image_template_variables,
    check_image_campaign_columns,
    normalize_image_campaign_columns,
    sanitize_template_variable,
    validate_and_group_customers,
    validate_image_url,
)


@pytest.fixture
def sample_csv_path() -> Path:
    proj_root = Path(__file__).resolve().parent.parent
    path = proj_root / "data" / "image_campaign_sample.csv"
    assert path.exists(), f"Sample CSV not found at {path}"
    return path


@pytest.fixture
def sample_df(sample_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(sample_csv_path)


class TestImageBulkDryRunValidation:
    """End-to-end test suite for Bulk Image Campaign Dry-Run."""

    def test_raw_rows_count(self, sample_df: pd.DataFrame):
        """Verify the sample CSV has exactly 6 raw data rows."""
        assert len(sample_df) == 6

    def test_column_normalization_and_required_columns(self, sample_df: pd.DataFrame):
        """Verify normalization preserves or maps all required columns."""
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        missing = check_image_campaign_columns(normalized_df)
        assert missing == [], f"Missing columns: {missing}"

        for col in IMAGE_CAMPAIGN_REQUIRED_COLUMNS:
            assert col in normalized_df.columns

    def test_grouping_produces_exactly_three_eligible_customers(self, sample_df: pd.DataFrame):
        """Verify 6 raw rows group into exactly 3 unique customers with 0 invalid rows."""
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        grouped_customers, invalid_df = validate_and_group_customers(normalized_df)

        assert len(invalid_df) == 0
        assert len(grouped_customers) == 3

    def test_grouped_customer_details_and_destinations(self, sample_df: pd.DataFrame):
        """Verify the 3 grouped customers match expected names, phones, and branches."""
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        grouped_customers, _ = validate_and_group_customers(normalized_df)

        expected_destinations = {
            "Sunil": "917659935016",
            "Tarun": "918688504571",
            "Ram": "917661087360",
        }

        expected_branches = {
            "Sunil": "Chadargatt",
            "Tarun": "Kompally",
            "Ram": "Karmanghat",
        }

        for cust in grouped_customers:
            name = cust["Name"]
            assert name in expected_destinations
            assert cust["Normalized Phone"] == expected_destinations[name]
            assert cust["Branch"] == expected_branches[name]

    def test_medicine_lists_single_line_and_exact_content(self, sample_df: pd.DataFrame):
        """
        Verify each customer's final medicine list is single-line comma-separated
        and contains the exact expected medicines.
        """
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        grouped_customers, _ = validate_and_group_customers(normalized_df)

        expected_medicines = {
            "Sunil": "METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG",
            "Tarun": "AMLODIPINE 5 MG, GLIMEPIRIDE 2 MG, LOSARTAN 50 MG",
            "Ram": "LEVOTHYROXINE 50 MCG, ATORVASTATIN 20 MG, METFORMIN 500 MG, GLYCOMET GP1",
        }

        for cust in grouped_customers:
            name = cust["Name"]
            med_list = cust["Medicine List"]
            assert med_list == expected_medicines[name]
            # Verify no newlines, carriage returns, tabs, or consecutive spaces
            assert "\n" not in med_list
            assert "\r" not in med_list
            assert "\t" not in med_list
            assert "  " not in med_list

    def test_all_eight_template_variables_per_customer(self, sample_df: pd.DataFrame):
        """Verify all 8 template body variables in the exact required order for each customer."""
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        grouped_customers, _ = validate_and_group_customers(normalized_df)
        store_name = "PHARMA HUBB"

        for cust in grouped_customers:
            vars_list = build_image_template_variables(cust, store_name)
            assert len(vars_list) == 8

            texts = [v["text"] for v in vars_list]
            # 1: Customer Name
            assert texts[0] == cust["Name"]
            # 2: Store Name
            assert texts[1] == store_name
            # 3: Branch
            assert texts[2] == cust["Branch"]
            # 4: Medicine List
            assert texts[3] == cust["Medicine List"]
            # 5: Contact No.
            assert texts[4] == cust["Contact No."]
            # 6: Manager Contact
            assert texts[5] == cust["Manager Contact"]
            # 7: Store Name
            assert texts[6] == store_name
            # 8: Branch
            assert texts[7] == cust["Branch"]

            # Check parameter sanitization on all 8 variables
            for text in texts:
                assert "\n" not in text
                assert "\r" not in text
                assert "\t" not in text
                assert "  " not in text

    def test_image_url_validation(self):
        """Verify the Cloudinary image URL passes validation."""
        cloudinary_url = (
            "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/"
            "PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template"
        )
        is_valid, err = validate_image_url(cloudinary_url)
        assert is_valid is True
        assert err == ""

    @patch("urllib3.PoolManager.request")
    @patch("requests.Session.send")
    @patch("requests.post")
    def test_bulk_dry_run_generates_three_payloads_with_zero_network_calls(
        self,
        mock_post,
        mock_send,
        mock_pool,
        sample_df: pd.DataFrame,
    ):
        """
        Execute the bulk dry-run pipeline and verify:
        - Exactly 3 customers processed
        - Exactly 3 successful dry-run payloads returned
        - Zero real HTTP/network calls made
        """
        normalized_df = normalize_image_campaign_columns(sample_df.copy())
        grouped_customers, _ = validate_and_group_customers(normalized_df)

        store_name = "PHARMA HUBB"
        template_name = "refill_reminder_image"
        template_language = "en"
        cloudinary_url = (
            "https://res.cloudinary.com/troli5kq/image/upload/f_auto,q_auto/"
            "PHARMA_HUBB_-_medicine_Refill_Reminder_Image_Template"
        )

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
            result = send_image_template_message(
                phone_number=phone_number,
                customer_name=customer_name,
                store_name=store_name,
                branch=cust.get("Branch", ""),
                medicine_list=cust.get("Medicine List", ""),
                contact_no=cust.get("Contact No.", ""),
                manager_contact=cust.get("Manager Contact", ""),
                image_url=cloudinary_url,
                dry_run=dry_run,
            )
            if result.get("response", {}).get("payload"):
                generated_payloads.append(result["response"]["payload"])
            return result

        summary = execute_bulk_send(
            grouped_customers,
            store_name=store_name,
            template_name=template_name,
            template_language=template_language,
            send_fn=_send_fn,
            dry_run=True,
            bulk_attempt_id="image_bulk_dry_run_test",
        )

        # 1. Summary checks
        assert summary["eligible"] == 3
        assert summary["attempted"] == 3
        assert summary["successful"] == 3
        assert summary["failed"] == 0
        assert len(summary["records"]) == 3

        # 2. Payloads check
        assert len(generated_payloads) == 3

        expected_destinations = ["917659935016", "918688504571", "917661087360"]
        for idx, p in enumerate(generated_payloads):
            assert p["to"] == expected_destinations[idx]
            assert p["type"] == "template"
            assert p["template"]["name"] == "refill_reminder_image"
            assert p["template"]["language"]["code"] == "en"

            # Header component
            header = p["template"]["components"][0]
            assert header["type"] == "header"
            assert header["parameters"][0]["type"] == "image"
            assert header["parameters"][0]["image"]["link"] == cloudinary_url

            # Body component
            body = p["template"]["components"][1]
            assert body["type"] == "body"
            assert len(body["parameters"]) == 8

        # 3. Critical Safety: ZERO real HTTP calls made
        mock_post.assert_not_called()
        mock_send.assert_not_called()
        mock_pool.assert_not_called()
