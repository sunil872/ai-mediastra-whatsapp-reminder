"""Unit tests for RefillCare packaging and pack-size parser.

Tests parse_pack_units across valid, invalid, empty, ambiguous, and edge-case packaging inputs.
"""

import pytest
import numpy as np
import pandas as pd
from refillcare.data.packing import parse_pack_units


class TestParsePackUnitsValid:
    """Tests for recognized and standard packing formats."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("1X15", 15.0),
            ("1x15", 15.0),
            ("1*15", 15.0),
            ("1 X 15", 15.0),
            ("1 x 15", 15.0),
            ("1X10", 10.0),
            ("1x10", 10.0),
            ("1X30", 30.0),
            ("1X20", 20.0),
            ("2X10", 20.0),
            ("10X10", 100.0),
            ("1X14", 14.0),
            ("1X7", 7.0),
            ("1X120", 120.0),
            ("1X100", 100.0),
            ("1X4", 4.0),
            ("1X5", 5.0),
            ("1X1", 1.0),
        ],
    )
    def test_multipack_solid_formats(self, input_val, expected):
        """Test standard multiplier formats (1X15, 2X10, etc.)."""
        assert parse_pack_units(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("10 TAB", 10.0),
            ("10TAB", 10.0),
            ("10 TABS", 10.0),
            ("10 TABLETS", 10.0),
            ("10 TAB.", 10.0),
            ("30 CAP", 30.0),
            ("30CAP", 30.0),
            ("30 CAPS", 30.0),
            ("30 CAPSULES", 30.0),
            ("1 PC", 1.0),
            ("1PC", 1.0),
            ("1 PCS", 1.0),
            ("1 PIECE", 1.0),
            ("1 PIECES", 1.0),
            ("1 STRIP", 1.0),
            ("10 STRIP", 10.0),
            ("1 VIAL", 1.0),
            ("1 AMP", 1.0),
            ("1 BOTTLE", 1.0),
            ("TAB 10", 10.0),
            ("CAP 30", 30.0),
            ("STRIP 1", 1.0),
        ],
    )
    def test_discrete_unit_labeled_formats(self, input_val, expected):
        """Test unit labeled formats (10 TAB, 30 CAP, 1 PC, etc.)."""
        assert parse_pack_units(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("100ML", 100.0),
            ("100 ML", 100.0),
            ("200ML", 200.0),
            ("10ML", 10.0),
            ("15ML", 15.0),
            ("5ML", 5.0),
            ("60ML", 60.0),
            ("400GM", 400.0),
            ("400 GM", 400.0),
            ("100GM", 100.0),
            ("75GM", 75.0),
            ("30GM", 30.0),
            ("15GM", 15.0),
            ("10GM", 10.0),
            ("500MG", 500.0),
            ("1L", 1.0),
            ("1 L", 1.0),
            ("1KG", 1.0),
            ("1X100ML", 100.0),
            ("1X50GM", 50.0),
            ("1X10ML", 10.0),
        ],
    )
    def test_volume_and_weight_formats(self, input_val, expected):
        """Test standard liquid volumes and topical weights."""
        assert parse_pack_units(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("1", 1.0),
            ("10", 10.0),
            ("15", 15.0),
            ("30", 30.0),
            ("100", 100.0),
            (15, 15.0),
            (10.0, 10.0),
            (1, 1.0),
        ],
    )
    def test_plain_numeric_inputs(self, input_val, expected):
        """Test plain numeric string and integer/float inputs."""
        assert parse_pack_units(input_val) == expected


class TestParsePackUnitsNormalization:
    """Tests for whitespace, casing, and typographical artifacts."""

    def test_whitespace_and_casing(self):
        assert parse_pack_units("  1x15  ") == 15.0
        assert parse_pack_units("\t10 tab\n") == 10.0
        assert parse_pack_units(" 100 ml ") == 100.0

    def test_typographical_artifacts(self):
        assert parse_pack_units("`1X14`") == 14.0
        assert parse_pack_units("'1X15'") == 15.0
        assert parse_pack_units('"1X30"') == 30.0
        assert parse_pack_units("1X14`") == 14.0


class TestParsePackUnitsInvalidAndEmpty:
    """Tests for empty, null, and non-positive inputs."""

    @pytest.mark.parametrize(
        "input_val",
        [
            None,
            "",
            "   ",
            "\t\n",
            pd.NA,
            np.nan,
            0,
            "0",
            -5,
            "-5",
            "0X10",
            "-1X15",
            False,  # bools must not be treated as int 0/1
            True,
        ],
    )
    def test_null_empty_and_zero_inputs(self, input_val):
        """Verify null, empty, boolean, and zero inputs return None."""
        assert parse_pack_units(input_val) is None


class TestParsePackUnitsAmbiguousAndUnsupported:
    """Tests for ambiguous multi-dimensional or unsupported strings that must safely return None."""

    @pytest.mark.parametrize(
        "input_val",
        [
            "10X4.4GM",    # ambiguous (10 sachets vs 4.4gm)
            "5X3ML",       # ambiguous (5 ampoules vs 3ml)
            "1X5X2ML",     # nested 3-tier packing
            "1X5X3ML",
            "200MDI",      # device metered doses
            "120MDI",
            "1KIT",        # composite kit
            "1PFS",        # prefilled syringe
            "1S",          # ambiguous single letter
            "UNKNOWN",     # pure text
            "N/A",
            "MISC",
            "SYRUP",
            "CREAM",
            "COMBO PACK",
        ],
    )
    def test_ambiguous_and_unsupported_formats(self, input_val):
        """Verify ambiguous and composite strings return None without guessing."""
        assert parse_pack_units(input_val) is None


class TestComputeTotalUnitsPurchased:
    """Tests for compute_total_units_purchased helper."""

    @pytest.mark.parametrize(
        "qty,packing,expected",
        [
            (4, "1X15", 60.0),
            (8, "1X15", 120.0),
            (1, "1X15", 15.0),
            (2, "10 TAB", 20.0),
            (3, "30 CAP", 90.0),
            (2, "100ML", 200.0),
            (5, "20GM", 100.0),
            (10, "1X10", 100.0),
            ("4", "1X15", 60.0),
            (4.0, "1X15", 60.0),
        ],
    )
    def test_valid_single_and_multi_pack_purchases(self, qty, packing, expected):
        """Test calculation across various valid quantities and pack types."""
        from refillcare.data.packing import compute_total_units_purchased
        assert compute_total_units_purchased(qty, packing) == expected

    @pytest.mark.parametrize(
        "qty,packing,expected",
        [
            (-3, "1X15", -45.0),     # 3 strips returned
            (-1, "10 TAB", -10.0),   # 1 pack returned
            ("-2", "1X30", -60.0),
            (0, "1X15", 0.0),
        ],
    )
    def test_return_and_adjustment_transactions(self, qty, packing, expected):
        """Test calculation preserves negative sign for return transactions."""
        from refillcare.data.packing import compute_total_units_purchased
        assert compute_total_units_purchased(qty, packing) == expected

    @pytest.mark.parametrize(
        "qty,packing",
        [
            (4, None),
            (4, ""),
            (4, "   "),
            (4, pd.NA),
            (4, np.nan),
            (4, "10X4.4GM"),
            (4, "UNKNOWN"),
            (4, "200MDI"),
            (None, "1X15"),
            ("", "1X15"),
            ("invalid_qty", "1X15"),
            (pd.NA, "1X15"),
            (np.nan, "1X15"),
            (False, "1X15"),
            (True, "1X15"),
        ],
    )
    def test_missing_and_invalid_inputs_return_none(self, qty, packing):
        """Verify missing/invalid quantity or packing returns None without guessing."""
        from refillcare.data.packing import compute_total_units_purchased
        assert compute_total_units_purchased(qty, packing) is None


class TestEnrichTotalUnitsPurchased:
    """Tests for DataFrame enrichment with total units."""

    def test_dataframe_enrichment_preserves_originals(self):
        """Verify enrich_total_units_purchased preserves existing columns and calculates total units."""
        from refillcare.data.packing import enrich_total_units_purchased

        df = pd.DataFrame([
            {"customerId": "C1", "itemId": "I1", "quantity": 4, "packing": "1X15"},
            {"customerId": "C2", "itemId": "I2", "quantity": 8, "packing": "1X15"},
            {"customerId": "C3", "itemId": "I3", "quantity": 1, "packing": "1X15"},
            {"customerId": "C4", "itemId": "I4", "quantity": -3, "packing": "1X15"},
            {"customerId": "C5", "itemId": "I5", "quantity": 2, "packing": "10 TAB"},
            {"customerId": "C6", "itemId": "I6", "quantity": 4, "packing": "UNKNOWN"},
            {"customerId": "C7", "itemId": "I7", "quantity": 4, "packing": None},
            {"customerId": "C8", "itemId": "I8", "quantity": None, "packing": "1X15"},
        ])

        enriched = enrich_total_units_purchased(df)

        # Check existing columns preserved
        assert "customerId" in enriched.columns
        assert "quantity" in enriched.columns
        assert "packing" in enriched.columns
        assert len(enriched) == 8

        # Check new columns
        assert "total_units_purchased" in enriched.columns
        assert "units_per_pack" in enriched.columns

        # Verify values
        assert enriched.loc[0, "total_units_purchased"] == 60.0
        assert enriched.loc[1, "total_units_purchased"] == 120.0
        assert enriched.loc[2, "total_units_purchased"] == 15.0
        assert enriched.loc[3, "total_units_purchased"] == -45.0
        assert enriched.loc[4, "total_units_purchased"] == 20.0
        assert pd.isna(enriched.loc[5, "total_units_purchased"])
        assert pd.isna(enriched.loc[6, "total_units_purchased"])
        assert pd.isna(enriched.loc[7, "total_units_purchased"])

    def test_empty_dataframe_enrichment(self):
        """Verify empty DataFrame returns gracefully with target columns."""
        from refillcare.data.packing import enrich_total_units_purchased

        empty_df = pd.DataFrame(columns=["quantity", "packing"])
        enriched = enrich_total_units_purchased(empty_df)

        assert enriched.empty
        assert "total_units_purchased" in enriched.columns
        assert "units_per_pack" in enriched.columns

