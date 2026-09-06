"""
Tests for the Image + Text WhatsApp campaign.

Covers:
  1. One medicine
  2. Two medicines
  3. Three medicines
  4. More than three medicines
  5. Same customer across multiple rows
  6. Duplicate medicine removal
  7. Correct 8-variable ordering
  8. Dynamic medicine-list generation
  9. Image-template payload structure (dry run)
 10. Missing image URL / configuration
 11. Missing image-template name / configuration
 12. Invalid phone number
 13. Duplicate customer handling (grouping by phone)
 14. No real HTTP request during tests

NO real WhatsApp messages are sent.  All Xinno calls use dry_run=True
or mocked send functions.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from utils.image_campaign import (
    IMAGE_CAMPAIGN_REQUIRED_COLUMNS,
    build_image_campaign_preview_table,
    build_image_sample_previews,
    build_image_template_variables,
    build_medicine_list,
    check_image_campaign_columns,
    generate_image_campaign_preview,
    get_variable_labels,
    missing_image_columns_message,
    normalize_image_campaign_columns,
    validate_and_group_customers,
)
from services.xinno_image_template import send_image_template_message


# ===================================================================
# Helpers
# ===================================================================

def _make_df(rows, columns=None):
    """Build a DataFrame from a list of row-tuples."""
    cols = columns or [
        "Name", "Phone number", "Medicine", "Branch",
        "Contact No.", "Manager Contact",
    ]
    return pd.DataFrame(rows, columns=cols)


_ENV_PATCH = {
    "XINNO_IMAGE_TEMPLATE_NAME": "refill_reminder_image",
    "XINNO_IMAGE_URL": "https://example.com/test_image.jpg",
    "XINNO_API_KEY": "test_key_12345",
    "XINNO_WABA_NUMBER": "919515473474",
    "WHATSAPP_TEMPLATE_LANGUAGE": "en",
    "MEDICAL_STORE_NAME": "PHARMA HUBB",
}


# ===================================================================
# 1. One medicine
# ===================================================================

class TestOneMedicine:
    def test_single_medicine_list(self):
        result = build_medicine_list(["METFORMIN 500 MG"])
        assert result == "METFORMIN 500 MG"
        assert "\n" not in result

    def test_single_medicine_grouping(self):
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt",
             "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Medicine Count"] == 1
        assert grouped[0]["Medicine List"] == "METFORMIN 500 MG"


# ===================================================================
# 2. Two medicines
# ===================================================================

class TestTwoMedicines:
    def test_two_medicine_list(self):
        result = build_medicine_list(["METFORMIN 500 MG", "TELMISARTAN 40 MG"])
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG"
        assert "\n" not in result

    def test_two_medicine_grouping(self):
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Sunil", "7659935016", "TELMISARTAN 40 MG", "Chadargatt",
             "9581473474", "9885473474"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Medicine Count"] == 2
        assert grouped[0]["Medicine List"] == "METFORMIN 500 MG, TELMISARTAN 40 MG"


# ===================================================================
# 3. Three medicines
# ===================================================================

class TestThreeMedicines:
    def test_three_medicine_list(self):
        meds = ["METFORMIN 500 MG", "TELMISARTAN 40 MG", "ATORVASTATIN 10 MG"]
        result = build_medicine_list(meds)
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG"
        assert "\n" not in result
        assert "\t" not in result

    def test_three_medicine_grouping(self):
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Sunil", "7659935016", "TELMISARTAN 40 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt",
             "9581473474", "9885473474"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Medicine Count"] == 3
        assert grouped[0]["Medicine List"] == "METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG"


# ===================================================================
# 4. More than three medicines
# ===================================================================

class TestMoreThanThreeMedicines:
    def test_four_medicine_list(self):
        meds = [
            "LEVOTHYROXINE 50 MCG", "ATORVASTATIN 20 MG",
            "METFORMIN 500 MG", "GLYCOMET GP1",
        ]
        result = build_medicine_list(meds)
        assert result == "LEVOTHYROXINE 50 MCG, ATORVASTATIN 20 MG, METFORMIN 500 MG, GLYCOMET GP1"
        assert "\n" not in result

    def test_five_medicines_grouping(self):
        df = _make_df([
            ("Ram", "7661087360", "LEVOTHYROXINE 50 MCG", "Branch A",
             "1111111111", "2222222222"),
            ("Ram", "7661087360", "ATORVASTATIN 20 MG", "Branch A",
             "1111111111", "2222222222"),
            ("Ram", "7661087360", "METFORMIN 500 MG", "Branch A",
             "1111111111", "2222222222"),
            ("Ram", "7661087360", "GLYCOMET GP1", "Branch A",
             "1111111111", "2222222222"),
            ("Ram", "7661087360", "PARACETAMOL 500 MG", "Branch A",
             "1111111111", "2222222222"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Medicine Count"] == 5
        assert grouped[0]["Medicine List"] == "LEVOTHYROXINE 50 MCG, ATORVASTATIN 20 MG, METFORMIN 500 MG, GLYCOMET GP1, PARACETAMOL 500 MG"


# ===================================================================
# 5. Same customer across multiple rows
# ===================================================================

class TestSameCustomerMultipleRows:
    def test_groups_by_phone(self):
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Sunil", "7659935016", "TELMISARTAN 40 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt",
             "9581473474", "9885473474"),
            ("Tarun", "8688504571", "AMLODIPINE 5 MG", "Chadargatt",
             "9581473474", "9885473474"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 2  # Sunil + Tarun
        sunil = next(c for c in grouped if c["Name"] == "Sunil")
        assert sunil["Medicine Count"] == 3
        tarun = next(c for c in grouped if c["Name"] == "Tarun")
        assert tarun["Medicine Count"] == 1

    def test_same_branch_and_contacts(self):
        """Branch and contact info are preserved when consistent."""
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "Branch1", "1111", "2222"),
            ("Sunil", "7659935016", "MED B", "Branch1", "1111", "2222"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert len(invalid) == 0
        assert grouped[0]["Branch"] == "Branch1"
        assert grouped[0]["Contact No."] == "1111"
        assert grouped[0]["Manager Contact"] == "2222"

    def test_phone_normalization_groups(self):
        """Different phone formats for same number should group together."""
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "B1", "C1", "M1"),
            ("Sunil", "917659935016", "MED B", "B1", "C1", "M1"),
            ("Sunil", "+91 76599 35016", "MED C", "B1", "C1", "M1"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Medicine Count"] == 3


# ===================================================================
# 6. Duplicate medicine removal
# ===================================================================

class TestDuplicateMedicineRemoval:
    def test_exact_duplicate(self):
        result = build_medicine_list([
            "METFORMIN 500 MG", "METFORMIN 500 MG", "TELMISARTAN 40 MG",
        ])
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG"
        assert "\n" not in result

    def test_case_insensitive_duplicate(self):
        result = build_medicine_list([
            "metformin 500 mg", "METFORMIN 500 MG", "Telmisartan 40 MG",
        ])
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG"

    def test_duplicate_in_grouping(self):
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "B1", "C1", "M1"),
            ("Sunil", "7659935016", "METFORMIN 500 MG", "B1", "C1", "M1"),
            ("Sunil", "7659935016", "TELMISARTAN 40 MG", "B1", "C1", "M1"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert grouped[0]["Medicine Count"] == 2
        assert grouped[0]["Medicine List"] == "METFORMIN 500 MG, TELMISARTAN 40 MG"

    def test_empty_medicine_ignored(self):
        result = build_medicine_list(["METFORMIN 500 MG", "", "  ", "nan", "None"])
        assert result == "METFORMIN 500 MG"


# ===================================================================
# 7. Correct 8-variable ordering
# ===================================================================

class TestVariableOrdering:
    def test_exactly_8_variables(self):
        customer = {
            "Name": "Sunil",
            "Branch": "Chadargatt",
            "Medicine List": "METFORMIN 500 MG, TELMISARTAN 40 MG",
            "Contact No.": "9581473474",
            "Manager Contact": "9885473474",
        }
        variables = build_image_template_variables(customer, "PHARMA HUBB")
        assert len(variables) == 8

    def test_exact_order(self):
        customer = {
            "Name": "Sunil",
            "Branch": "Chadargatt",
            "Medicine List": "METFORMIN 500 MG",
            "Contact No.": "9581473474",
            "Manager Contact": "9885473474",
        }
        variables = build_image_template_variables(customer, "PHARMA HUBB")
        assert variables[0]["text"] == "Sunil"             # 1. Customer Name
        assert variables[1]["text"] == "PHARMA HUBB"       # 2. Store Name
        assert variables[2]["text"] == "Chadargatt"        # 3. Branch
        assert variables[3]["text"] == "METFORMIN 500 MG"  # 4. Medicine List
        assert variables[4]["text"] == "9581473474"        # 5. Contact No.
        assert variables[5]["text"] == "9885473474"        # 6. Manager Contact
        assert variables[6]["text"] == "PHARMA HUBB"       # 7. Store Name (repeat)
        assert variables[7]["text"] == "Chadargatt"        # 8. Branch (repeat)

    def test_all_type_text(self):
        customer = {
            "Name": "Test", "Branch": "B",
            "Medicine List": "X", "Contact No.": "C",
            "Manager Contact": "M",
        }
        variables = build_image_template_variables(customer, "S")
        for var in variables:
            assert var["type"] == "text"

    def test_variable_labels(self):
        labels = get_variable_labels()
        assert len(labels) == 8
        assert labels[0] == "Customer Name"
        assert labels[3] == "Medicine List"
        assert labels[6] == "Store Name"


# ===================================================================
# 8. Dynamic medicine-list generation
# ===================================================================

class TestDynamicMedicineList:
    def test_comma_separated_format(self):
        result = build_medicine_list(["A", "B", "C"])
        assert result == "A, B, C"
        assert "\n" not in result

    def test_single_variable(self):
        """Medicine list is one string, not separate variables."""
        customer = {
            "Name": "Sunil", "Branch": "B",
            "Medicine List": "MED A, MED B, MED C",
            "Contact No.": "C", "Manager Contact": "M",
        }
        variables = build_image_template_variables(customer, "S")
        # Variable 4 (index 3) contains the full medicine list
        med_var = variables[3]["text"]
        assert "\n" not in med_var  # single line in one variable
        assert med_var == "MED A, MED B, MED C"

    def test_empty_medicine_list(self):
        result = build_medicine_list([])
        assert result == ""

    def test_preserves_order(self):
        meds = ["ZZZZZ", "AAAAA", "MMMMM"]
        result = build_medicine_list(meds)
        assert result == "ZZZZZ, AAAAA, MMMMM"

    def test_uppercases_medicines(self):
        result = build_medicine_list(["metformin 500 mg"])
        assert result == "METFORMIN 500 MG"

    def test_sanitization_removes_newlines_and_tabs(self):
        result = build_medicine_list(["METFORMIN 500 MG\t", "TELMISARTAN   40  MG"])
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG"
        assert "\n" not in result
        assert "\t" not in result
        assert "   " not in result

    def test_multi_line_cell_splits_to_comma_separated(self):
        result = build_medicine_list(["METFORMIN 500 MG\nTELMISARTAN 40 MG"])
        assert result == "METFORMIN 500 MG, TELMISARTAN 40 MG"
        assert "\n" not in result


# ===================================================================
# 9. Image-template payload structure (dry run)
# ===================================================================

class TestImageTemplatePayload:
    def test_dry_run_payload_structure(self):
        """Verify the full payload matches docs/WhatsappAPIDocument.json Template-Image."""
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                branch="Chadargatt",
                medicine_list="METFORMIN 500 MG, TELMISARTAN 40 MG",
                contact_no="9581473474",
                manager_contact="9885473474",
                image_url="https://example.com/image.jpg",
                dry_run=True,
            )

        assert result["success"] is True
        assert result["status_code"] is None
        assert result["response"]["dry_run"] is True

        payload = result["response"]["payload"]
        assert payload["to"] == "917659935016"
        assert payload["type"] == "template"

        template = payload["template"]
        assert template["language"]["policy"] == "deterministic"
        assert template["language"]["code"] == "en"
        assert template["name"] == "refill_reminder_image"

        components = template["components"]
        assert len(components) == 2

        # Header: image
        header = components[0]
        assert header["type"] == "header"
        assert len(header["parameters"]) == 1
        assert header["parameters"][0]["type"] == "image"
        assert header["parameters"][0]["image"]["link"] == "https://example.com/image.jpg"

        # Body: 8 text parameters
        body = components[1]
        assert body["type"] == "body"
        assert len(body["parameters"]) == 8
        for param in body["parameters"]:
            assert param["type"] == "text"
            assert "text" in param
            assert "\n" not in param["text"]

    def test_api_key_masked_in_headers(self):
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/image.jpg",
                dry_run=True,
            )
        headers = result["response"]["headers"]
        assert headers["Key"] == "***MASKED***"
        # Verify no API key leaked anywhere
        result_str = json.dumps(result)
        assert "test_key_12345" not in result_str

    def test_body_parameter_values(self):
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                branch="Chadargatt",
                medicine_list="METFORMIN 500 MG",
                contact_no="9581473474",
                manager_contact="9885473474",
                image_url="https://example.com/image.jpg",
                dry_run=True,
            )
        params = result["response"]["payload"]["template"]["components"][1]["parameters"]
        assert params[0]["text"] == "Sunil"
        assert params[1]["text"] == "PHARMA HUBB"
        assert params[2]["text"] == "Chadargatt"
        assert params[3]["text"] == "METFORMIN 500 MG"
        assert params[4]["text"] == "9581473474"
        assert params[5]["text"] == "9885473474"
        assert params[6]["text"] == "PHARMA HUBB"
        assert params[7]["text"] == "Chadargatt"

    def test_phone_normalized_in_payload(self):
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="+91 76599 35016",
                customer_name="Test",
                store_name="Store",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        assert result["response"]["payload"]["to"] == "917659935016"


# ===================================================================
# 10. Missing image URL / configuration
# ===================================================================

class TestMissingImageUrl:
    def test_no_image_url_env_or_param(self):
        env = dict(_ENV_PATCH)
        env.pop("XINNO_IMAGE_URL", None)
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, env, clear=False):
            # Also ensure env doesn't have it
            os.environ.pop("XINNO_IMAGE_URL", None)
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                dry_run=True,
            )
        assert result["success"] is False
        assert "Image URL" in result["message"]

    def test_empty_image_url(self):
        env = dict(_ENV_PATCH)
        env["XINNO_IMAGE_URL"] = ""
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, env, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="",
                dry_run=True,
            )
        assert result["success"] is False
        assert "Image URL" in result["message"]

    def test_local_path_rejected(self):
        env = dict(_ENV_PATCH)
        env["XINNO_IMAGE_URL"] = "C:\\images\\header.jpg"
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, env, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="C:\\images\\header.jpg",
                dry_run=True,
            )
        assert result["success"] is False
        assert "publicly accessible HTTP/HTTPS URL" in result["message"]


# ===================================================================
# 11. Missing image-template name / configuration
# ===================================================================

class TestMissingTemplateName:
    def test_no_template_name(self):
        env = dict(_ENV_PATCH)
        env.pop("XINNO_IMAGE_TEMPLATE_NAME", None)
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, env, clear=False):
            os.environ.pop("XINNO_IMAGE_TEMPLATE_NAME", None)
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        assert result["success"] is False
        assert "XINNO_IMAGE_TEMPLATE_NAME" in result["message"]

    def test_empty_template_name(self):
        env = dict(_ENV_PATCH)
        env["XINNO_IMAGE_TEMPLATE_NAME"] = ""
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, env, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        assert result["success"] is False
        assert "XINNO_IMAGE_TEMPLATE_NAME" in result["message"]


# ===================================================================
# 12. Invalid phone number
# ===================================================================

class TestInvalidPhone:
    def test_invalid_phone_in_sender(self):
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        assert result["success"] is False
        assert "Validation Error" in result["message"]

    def test_invalid_phone_in_grouping(self):
        df = _make_df([
            ("Sunil", "INVALID", "MED A", "B", "C", "M"),
            ("Tarun", "8688504571", "MED B", "B", "C", "M"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1  # only Tarun
        assert len(invalid) == 1
        assert invalid.iloc[0]["Name"] == "Sunil"

    def test_letters_in_phone(self):
        df = _make_df([
            ("Test", "765abc5016", "MED", "B", "C", "M"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 0
        assert len(invalid) == 1

    def test_empty_name_invalid(self):
        df = _make_df([
            ("", "7659935016", "MED", "B", "C", "M"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 0
        assert len(invalid) == 1


# ===================================================================
# 13. Duplicate customer handling (grouping)
# ===================================================================

class TestDuplicateCustomerHandling:
    def test_multiple_customers_grouped_separately(self):
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "B1", "C1", "M1"),
            ("Sunil", "7659935016", "MED B", "B1", "C1", "M1"),
            ("Tarun", "8688504571", "MED C", "B2", "C2", "M2"),
            ("Tarun", "8688504571", "MED D", "B2", "C2", "M2"),
            ("Ram", "7661087360", "MED E", "B3", "C3", "M3"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 3
        names = [c["Name"] for c in grouped]
        assert "Sunil" in names
        assert "Tarun" in names
        assert "Ram" in names

    def test_groups_count_medicines_correctly(self):
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "B", "C", "M"),
            ("Sunil", "7659935016", "MED B", "B", "C", "M"),
            ("Sunil", "7659935016", "MED C", "B", "C", "M"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert grouped[0]["Medicine Count"] == 3

    def test_eligible_dict_structure(self):
        """Grouped customers have all fields needed by execute_bulk_send."""
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "B", "C", "M"),
        ])
        grouped, _ = validate_and_group_customers(df)
        cust = grouped[0]
        assert "Name" in cust
        assert "Normalized Phone" in cust
        assert "Status" in cust
        assert cust["Status"] == "Valid"

    def test_single_row_per_customer(self):
        df = _make_df([
            ("Sunil", "7659935016", "MED A", "B", "C", "M"),
            ("Tarun", "8688504571", "MED B", "B", "C", "M"),
        ])
        grouped, _ = validate_and_group_customers(df)
        assert len(grouped) == 2
        for c in grouped:
            assert c["Medicine Count"] == 1


# ===================================================================
# 14. No real HTTP request during tests
# ===================================================================

class TestNoRealHttp:
    def test_dry_run_default(self):
        """send_image_template_message defaults to dry_run=True."""
        import inspect
        sig = inspect.signature(send_image_template_message)
        assert sig.parameters["dry_run"].default is True

    def test_dry_run_no_http(self):
        """Dry run returns without making HTTP calls."""
        with patch("services.xinno_image_template.load_dotenv"), \
             patch("services.xinno_image_template.requests.post") as mock_post, \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        mock_post.assert_not_called()
        assert result["success"] is True
        assert "DRY RUN" in result["message"]

    def test_mocked_send_fn_in_bulk(self):
        """execute_bulk_send with mocked send_fn never touches Xinno."""
        from utils.bulk_send import execute_bulk_send

        call_log = []

        def mock_send(phone_number, customer_name, store_name, dry_run):
            call_log.append({
                "phone": phone_number,
                "name": customer_name,
                "dry_run": dry_run,
            })
            return {
                "success": True,
                "status_code": None,
                "message": "[MOCK] OK",
                "response": {"dry_run": dry_run},
            }

        eligible = [
            {"Name": "Sunil", "Phone number": "917659935016",
             "Original Phone": "7659935016", "Normalized Phone": "917659935016",
             "Status": "Valid"},
        ]

        result = execute_bulk_send(
            eligible,
            store_name="PHARMA HUBB",
            template_name="test_template",
            send_fn=mock_send,
            dry_run=True,
        )
        assert result["successful"] == 1
        assert len(call_log) == 1
        assert call_log[0]["dry_run"] is True


# ===================================================================
# Column normalization tests
# ===================================================================

class TestColumnNormalization:
    def test_standard_columns(self):
        df = pd.DataFrame(columns=[
            "Name", "Phone number", "Medicine", "Branch",
            "Contact No.", "Manager Contact",
        ])
        result = normalize_image_campaign_columns(df)
        missing = check_image_campaign_columns(result)
        assert missing == []

    def test_alias_columns(self):
        df = pd.DataFrame(columns=[
            "customer_name", "mobile", "medication",
            "branch_name", "contact_no", "manager_phone",
        ])
        result = normalize_image_campaign_columns(df)
        missing = check_image_campaign_columns(result)
        assert missing == []

    def test_customer_medication_list_alias(self):
        """'customer medication list' maps to 'Medicine'."""
        df = pd.DataFrame(columns=[
            "Name", "Phone number", "Customer Medication List",
            "Branch", "Contact No.", "Manager Contact",
        ])
        result = normalize_image_campaign_columns(df)
        assert "Medicine" in result.columns

    def test_missing_columns_message(self):
        msg = missing_image_columns_message(["Medicine", "Branch"])
        assert "Medicine" in msg
        assert "Branch" in msg


# ===================================================================
# Preview tests
# ===================================================================

class TestPreview:
    def test_preview_table(self):
        grouped = [
            {"Name": "Sunil", "Original Phone": "7659935016",
             "Normalized Phone": "917659935016", "Branch": "B1",
             "Medicine Count": 3, "Status": "Valid"},
        ]
        invalid = pd.DataFrame(columns=[
            "Row", "Name", "Original Phone", "Normalized Phone",
            "Medicine", "Reason", "Status",
        ])
        table = build_image_campaign_preview_table(grouped, invalid)
        assert len(table) == 1
        assert table.iloc[0]["Name"] == "Sunil"

    def test_sample_previews(self):
        grouped = [
            {"Name": "Sunil", "Original Phone": "7659935016",
             "Normalized Phone": "917659935016", "Branch": "Chadargatt",
             "Medicine List": "- MED A\n- MED B", "Medicine Count": 2,
             "Contact No.": "9581473474", "Manager Contact": "9885473474",
             "Status": "Valid"},
        ]
        samples = build_image_sample_previews(grouped, "PHARMA HUBB", limit=1)
        assert len(samples) == 1
        assert samples[0]["customer_name"] == "Sunil"
        assert len(samples[0]["variables"]) == 8
        assert "message_preview" in samples[0]

    def test_rendered_preview_contains_values(self):
        customer = {
            "Name": "Sunil", "Branch": "Chadargatt",
            "Medicine List": "- MED A", "Contact No.": "C",
            "Manager Contact": "M",
        }
        preview = generate_image_campaign_preview(customer, "PHARMA HUBB")
        assert "Sunil" in preview
        assert "PHARMA HUBB" in preview
        assert "Chadargatt" in preview
        assert "- MED A" in preview


# ===================================================================
# Step 4 exact sample data test suite
# ===================================================================

class TestStep4SampleData:
    """End-to-end tests using the exact Step 4 customer dataset."""

    SAMPLE_DATA = [
        ("Sunil", "7659935016", "METFORMIN 500 MG, TELMISARTAN 40 MG", "Chadargatt", "9581473474", "9885473474"),
        ("Tarun", "8688504571", "AMLODIPINE 5 MG, GLIMEPIRIDE 2 MG", "Kompally", "9123456780", "9876543210"),
        ("Tarun", "8688504571", "LOSARTAN 50 MG", "Kompally", "9123456780", "9876543210"),
        ("Ram", "7661087360", "LEVOTHYROXINE 50 MCG, ATORVASTATIN 20 MG", "Karmanghat", "9345678901", "9898989898"),
        ("Ram", "7661087360", "METFORMIN 500 MG, GLYCOMET GP1", "Karmanghat", "9345678901", "9898989898"),
        ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt", "9581473474", "9885473474"),
    ]

    def test_grouping_produces_3_customers(self):
        df = _make_df(self.SAMPLE_DATA)
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 3
        assert len(invalid) == 0

    def test_sunil_medicines(self):
        df = _make_df(self.SAMPLE_DATA)
        grouped, _ = validate_and_group_customers(df)
        sunil = next(c for c in grouped if c["Name"] == "Sunil")
        assert sunil["Medicine Count"] == 3
        expected = "METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG"
        assert sunil["Medicine List"] == expected
        assert "\n" not in sunil["Medicine List"]

    def test_tarun_medicines(self):
        df = _make_df(self.SAMPLE_DATA)
        grouped, _ = validate_and_group_customers(df)
        tarun = next(c for c in grouped if c["Name"] == "Tarun")
        assert tarun["Medicine Count"] == 3
        expected = "AMLODIPINE 5 MG, GLIMEPIRIDE 2 MG, LOSARTAN 50 MG"
        assert tarun["Medicine List"] == expected

    def test_ram_medicines(self):
        df = _make_df(self.SAMPLE_DATA)
        grouped, _ = validate_and_group_customers(df)
        ram = next(c for c in grouped if c["Name"] == "Ram")
        assert ram["Medicine Count"] == 4
        expected = "LEVOTHYROXINE 50 MCG, ATORVASTATIN 20 MG, METFORMIN 500 MG, GLYCOMET GP1"
        assert ram["Medicine List"] == expected

    def test_exact_8_variables_ordering(self):
        df = _make_df(self.SAMPLE_DATA)
        grouped, _ = validate_and_group_customers(df)
        sunil = next(c for c in grouped if c["Name"] == "Sunil")
        vars_ = build_image_template_variables(sunil, store_name="PHARMA HUBB")
        assert len(vars_) == 8
        assert vars_[0]["text"] == "Sunil"             # 1. Customer Name
        assert vars_[1]["text"] == "PHARMA HUBB"       # 2. Store Name
        assert vars_[2]["text"] == "Chadargatt"        # 3. Branch
        assert vars_[3]["text"] == sunil["Medicine List"]  # 4. Medicine list (single var)
        assert vars_[4]["text"] == "9581473474"        # 5. Contact No.
        assert vars_[5]["text"] == "9885473474"        # 6. Manager Contact
        assert vars_[6]["text"] == "PHARMA HUBB"       # 7. Store Name
        assert vars_[7]["text"] == "Chadargatt"        # 8. Branch

    def test_sample_csv_file_on_disk(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "image_campaign_sample.csv")
        df = pd.read_csv(csv_path, dtype=str)
        df = normalize_image_campaign_columns(df)
        missing = check_image_campaign_columns(df)
        assert missing == []
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 3
        assert len(invalid) == 0


# ===================================================================
# Step 5: Name + Phone grouping and Conflict Safety
# ===================================================================

class TestStep5NameAndPhoneGrouping:
    """Tests for customer identity based on (normalized_name, normalized_phone)."""

    def test_same_name_same_phone_different_medicines_combined(self):
        """1. Same name + same phone + different medicines -> 1 customer, medicines combined."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "TELMISARTAN 40 MG", "Chadargatt", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert len(invalid) == 0
        assert grouped[0]["Name"] == "Sunil"
        assert grouped[0]["Medicine Count"] == 2
        assert "METFORMIN 500 MG" in grouped[0]["Medicine List"]
        assert "TELMISARTAN 40 MG" in grouped[0]["Medicine List"]
        assert "\n" not in grouped[0]["Medicine List"]

    def test_same_name_different_case_same_phone(self):
        """2. Same name with different letter case + same phone -> 1 customer."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("SUNIL", "7659935016", "TELMISARTAN 40 MG", "Chadargatt", "9581473474", "9885473474"),
            ("sunil ", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert len(invalid) == 0
        assert grouped[0]["Name"] == "Sunil"
        assert grouped[0]["Medicine Count"] == 3

    def test_same_name_same_phone_duplicate_medicine_removed(self):
        """3. Same name + same phone + duplicate medicine -> 1 customer, duplicate removed."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert len(invalid) == 0
        assert grouped[0]["Medicine Count"] == 1
        assert grouped[0]["Medicine List"] == "METFORMIN 500 MG"

    def test_different_name_same_phone_separate_customers(self):
        """4. Different name + same phone -> TWO separate customers with own medicine lists."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Ravi", "7659935016", "AMLODIPINE 5 MG", "Chadargatt", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 2
        assert len(invalid) == 0

        sunil = next(c for c in grouped if c["Name"] == "Sunil")
        ravi = next(c for c in grouped if c["Name"] == "Ravi")

        assert sunil["Medicine List"] == "METFORMIN 500 MG"
        assert sunil["Medicine Count"] == 1
        assert sunil["Normalized Phone"] == "917659935016"

        assert ravi["Medicine List"] == "AMLODIPINE 5 MG"
        assert ravi["Medicine Count"] == 1
        assert ravi["Normalized Phone"] == "917659935016"

    def test_same_name_same_phone_different_branch_flags_conflict(self):
        """5. Same name + same phone + different branch -> data conflict; excluded from sending."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Kompally", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 0
        assert len(invalid) == 2
        assert "Data conflict" in invalid.iloc[0]["Reason"]
        assert "Conflicting Branch" in invalid.iloc[0]["Reason"]

    def test_same_name_same_phone_different_contact_flags_conflict(self):
        """6. Same name + same phone + different contact number -> data conflict; excluded."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt", "9123456780", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 0
        assert len(invalid) == 2
        assert "Data conflict" in invalid.iloc[0]["Reason"]
        assert "Conflicting Contact No." in invalid.iloc[0]["Reason"]

    def test_same_name_same_phone_different_manager_flags_conflict(self):
        """Same name + same phone + different manager contact -> data conflict; excluded."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Chadargatt", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Chadargatt", "9581473474", "9876543210"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 0
        assert len(invalid) == 2
        assert "Data conflict" in invalid.iloc[0]["Reason"]
        assert "Conflicting Manager Contact" in invalid.iloc[0]["Reason"]

    def test_shared_phone_with_one_valid_and_one_conflicted(self):
        """Ravi is valid, while Sunil has conflicting branch on same phone."""
        df = _make_df([
            ("Sunil", "7659935016", "METFORMIN 500 MG", "Branch A", "9581473474", "9885473474"),
            ("Sunil", "7659935016", "ATORVASTATIN 10 MG", "Branch B", "9581473474", "9885473474"),
            ("Ravi", "7659935016", "AMLODIPINE 5 MG", "Branch C", "9581473474", "9885473474"),
        ])
        grouped, invalid = validate_and_group_customers(df)
        assert len(grouped) == 1
        assert grouped[0]["Name"] == "Ravi"
        assert len(invalid) == 2
        assert all(invalid["Name"] == "Sunil")


# ===================================================================
# Step 7: Latest Approved Xinno Image Template (refill_reminder_image)
# ===================================================================

class TestStep7ApprovedTemplate:
    """Verification for the approved template: refill_reminder_image."""

    APPROVED_TEMPLATE_NAME = "refill_reminder_image"

    def test_approved_template_name_used_in_payload(self):
        """a, b, c, d, e, f, h: Verify payload structure with refill_reminder_image."""
        with patch("services.xinno_image_template.load_dotenv"), \
             patch("services.xinno_image_template.requests.post") as mock_post, \
             patch.dict(os.environ, _ENV_PATCH, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                branch="Chadargatt",
                medicine_list="METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG",
                contact_no="9581473474",
                manager_contact="9885473474",
                image_url="https://example.com/refill_header.jpg",
                dry_run=True,
            )

        # h. No real HTTP request made
        mock_post.assert_not_called()
        assert result["success"] is True

        payload = result["response"]["payload"]

        # c. "type": "template"
        assert payload["type"] == "template"

        # a & b. Template name is refill_reminder_image
        assert payload["template"]["name"] == self.APPROVED_TEMPLATE_NAME
        assert payload["template"]["language"]["code"] == "en"

        components = payload["template"]["components"]
        assert len(components) == 2

        # d. Header component with image
        header = next(c for c in components if c["type"] == "header")
        assert header["parameters"][0]["type"] == "image"
        assert header["parameters"][0]["image"]["link"] == "https://example.com/refill_header.jpg"

        # e. Body component with exactly 8 parameters
        body = next(c for c in components if c["type"] == "body")
        params = body["parameters"]
        assert len(params) == 8

        # f. Exact 8 parameters in order
        assert params[0] == {"type": "text", "text": "Sunil"}               # 1. Customer Name
        assert params[1] == {"type": "text", "text": "PHARMA HUBB"}         # 2. Store Name
        assert params[2] == {"type": "text", "text": "Chadargatt"}          # 3. Branch
        assert params[3] == {"type": "text", "text": "METFORMIN 500 MG, TELMISARTAN 40 MG, ATORVASTATIN 10 MG"} # 4. Combined Meds
        assert params[4] == {"type": "text", "text": "9581473474"}          # 5. Contact No.
        assert params[5] == {"type": "text", "text": "9885473474"}          # 6. Manager Contact
        assert params[6] == {"type": "text", "text": "PHARMA HUBB"}         # 7. Store Name
        assert params[7] == {"type": "text", "text": "Chadargatt"}          # 8. Branch

    def test_dynamic_template_name_from_env(self):
        """Image sender uses whatever template name is configured, without hard-coding."""
        custom_env = dict(_ENV_PATCH)
        custom_env["XINNO_IMAGE_TEMPLATE_NAME"] = "custom_dynamic_template_v9"
        with patch("services.xinno_image_template.load_dotenv"), \
             patch.dict(os.environ, custom_env, clear=False):
            result = send_image_template_message(
                phone_number="7659935016",
                customer_name="Sunil",
                store_name="PHARMA HUBB",
                image_url="https://example.com/img.jpg",
                dry_run=True,
            )
        assert result["response"]["payload"]["template"]["name"] == "custom_dynamic_template_v9"

    def test_rendered_preview_matches_approved_template_text(self):
        """Preview template renders the exact approved template text."""
        customer = {
            "Name": "Sunil",
            "Branch": "Chadargatt",
            "Medicine List": "METFORMIN 500 MG, TELMISARTAN 40 MG",
            "Contact No.": "9581473474",
            "Manager Contact": "9885473474",
        }
        preview = generate_image_campaign_preview(customer, "PHARMA HUBB")
        assert "Dear *Sunil* Garu," in preview
        assert "*PHARMA HUBB*," in preview
        assert "*Chadargatt*." in preview
        assert "📋 Our records indicate that it may be time to refill your medication(s):" in preview
        assert "METFORMIN 500 MG, TELMISARTAN 40 MG" in preview
        assert "📞 9581473474" in preview
        assert "📞 9885473474" in preview
        assert "🧡 *Team*" in preview
        assert "📍 *Chadargatt*" in preview
        assert "We look forward to serving you." in preview



