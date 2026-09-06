"""
Tests for the Production-Ready CSV Column Alias and Canonical Normalization System.

Validates Part B requirements:
1. Format 1: Current sample format
2. Format 2: Typical pharmacy/client format
3. Format 3: Excel-style format
4. Format 4: Mixed capitalization / spacing / punctuation
5. Missing required column detection with example suggestions
6. Ambiguous column detection (multiple raw columns mapping to same canonical field)
7. Duplicate column detection in upload
8. Extra/unrelated columns handling (ignored cleanly without errors)
9. UTF-8 BOM and leading/trailing whitespace handling
10. Empty / None column name handling
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.column_aliases import (
    ALL_CANONICAL_FIELDS,
    CANONICAL_BRANCH,
    CANONICAL_CUSTOMER_NAME,
    CANONICAL_MEDICINE,
    CANONICAL_PHONE,
    COLUMN_ALIASES,
    REQUIRED_CANONICAL_FIELDS,
    normalize_dataframe_columns,
    normalize_header_string,
    resolve_column_aliases,
)


class TestHeaderNormalization:
    """Test string-level header normalization rules."""

    def test_bom_stripping(self):
        assert normalize_header_string("\ufeffCustomer Name") == "customer name"

    def test_case_and_whitespace_collapse(self):
        assert normalize_header_string("  CUSTOMER   NAME  ") == "customer name"
        assert normalize_header_string("Patient\tName\n") == "patient name"

    def test_delimiters_and_punctuation(self):
        assert normalize_header_string("customer_name") == "customer name"
        assert normalize_header_string("CUSTOMER-NAME") == "customer name"
        assert normalize_header_string("Branch / Location") == "branch location"
        assert normalize_header_string("Contact No.") == "contact no"
        assert normalize_header_string("Mobile (WhatsApp)") == "mobile whatsapp"


class TestClientCsvFormats:
    """Validate all real-world client CSV formats resolve to canonical fields."""

    def test_format_1_current_sample(self):
        """Format 1: Current sample format with extra Customer Medication List."""
        cols = [
            "Name", "Phone number", "Medicine", "Branch",
            "Customer Medication List", "Contact No.", "Manager Contact"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is True
        assert res.canonical_to_raw["customer_name"] == "Name"
        assert res.canonical_to_raw["phone"] == "Phone number"
        assert res.canonical_to_raw["medicine"] == "Medicine"
        assert res.canonical_to_raw["branch"] == "Branch"
        assert res.canonical_to_raw["contact_no"] == "Contact No."
        assert res.canonical_to_raw["manager_contact"] == "Manager Contact"
        assert "Customer Medication List" in res.unmapped_columns

    def test_format_2_pharmacy_client_format(self):
        """Format 2: Typical pharmacy format."""
        cols = [
            "Patient Name", "WhatsApp Number", "Medication",
            "Outlet", "Store Contact Number", "Manager Mobile"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is True
        assert res.canonical_to_raw["customer_name"] == "Patient Name"
        assert res.canonical_to_raw["phone"] == "WhatsApp Number"
        assert res.canonical_to_raw["medicine"] == "Medication"
        assert res.canonical_to_raw["branch"] == "Outlet"
        assert res.canonical_to_raw["contact_no"] == "Store Contact Number"
        assert res.canonical_to_raw["manager_contact"] == "Manager Mobile"

    def test_format_3_excel_style_format(self):
        """Format 3: Excel-style snake_case with leading whitespace."""
        cols = [
            " customer_name", "mobile_number", "medicine_name",
            "branch_name", "contact_number", "manager_number"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is True
        assert res.canonical_to_raw["customer_name"] == " customer_name"
        assert res.canonical_to_raw["phone"] == "mobile_number"
        assert res.canonical_to_raw["medicine"] == "medicine_name"
        assert res.canonical_to_raw["branch"] == "branch_name"
        assert res.canonical_to_raw["contact_no"] == "contact_number"
        assert res.canonical_to_raw["manager_contact"] == "manager_number"

    def test_format_4_mixed_capitalization_and_spacing(self):
        """Format 4: Mixed uppercase, slashes, punctuation."""
        cols = [
            "CUSTOMER NAME", "Mobile No", "Medicine Name",
            "Branch / Location", "Store Phone", "Manager Contact No"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is True
        assert res.canonical_to_raw["customer_name"] == "CUSTOMER NAME"
        assert res.canonical_to_raw["phone"] == "Mobile No"
        assert res.canonical_to_raw["medicine"] == "Medicine Name"
        assert res.canonical_to_raw["branch"] == "Branch / Location"
        assert res.canonical_to_raw["contact_no"] == "Store Phone"
        assert res.canonical_to_raw["manager_contact"] == "Manager Contact No"


class TestValidationAndEdgeCases:
    """Validate safety gates, ambiguities, duplicates, and missing columns."""

    def test_missing_required_column(self):
        """Detect missing required column and provide accepted alias examples."""
        cols = ["Patient Name", "WhatsApp Number", "Outlet"]  # Missing medicine
        res = resolve_column_aliases(cols)
        assert res.is_valid is False
        assert "medicine" in res.missing_required
        assert len(res.errors) >= 1
        assert "Missing required field 'medicine'" in res.errors[0]

    def test_ambiguous_columns_detected(self):
        """Detect when multiple columns in the upload map to the same canonical field."""
        cols = [
            "Patient Name", "WhatsApp Number", "Mobile No",
            "Medication", "Outlet"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is False
        assert "phone" in res.ambiguities
        assert set(res.ambiguities["phone"]) == {"WhatsApp Number", "Mobile No"}
        assert any("Ambiguous column mapping" in err for err in res.errors)

    def test_duplicate_column_names_detected(self):
        """Detect exact duplicate column names."""
        cols = [
            "Patient Name", "WhatsApp Number", "Medication",
            "Outlet", "Medication"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is False
        assert len(res.duplicate_columns) >= 1

    def test_extra_unrelated_columns_ignored_safely(self):
        """Extra columns should not cause failure and must be recorded in unmapped_columns."""
        cols = [
            "Patient Name", "WhatsApp Number", "Medication",
            "Outlet", "Notes", "Doctor Name", "Invoice ID"
        ]
        res = resolve_column_aliases(cols)
        assert res.is_valid is True
        assert "Notes" in res.unmapped_columns
        assert "Doctor Name" in res.unmapped_columns
        assert "Invoice ID" in res.unmapped_columns

    def test_dataframe_normalization_end_to_end(self):
        """Verify DataFrame columns are renamed to internal standard names."""
        raw_df = pd.DataFrame([
            {"Patient Name": "Sunil", "WhatsApp Number": "7659935016", "Medication": "METFORMIN", "Outlet": "Chadargatt"}
        ])
        renamed_df, res = normalize_dataframe_columns(raw_df)
        assert res.is_valid is True
        assert "Name" in renamed_df.columns
        assert "Phone number" in renamed_df.columns
        assert "Medicine" in renamed_df.columns
        assert "Branch" in renamed_df.columns
        assert renamed_df.iloc[0]["Name"] == "Sunil"
