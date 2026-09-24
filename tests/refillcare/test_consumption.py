"""Unit tests for RefillCare historical consumption rate calculations.

Tests calculate_historical_consumption_rate and compute_expanding_consumption_rates across
standard refill cycles, cold-start histories, edge cases, and leakage safety.
"""

import pytest
import pandas as pd
import numpy as np

from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_daily_consumption,
    compute_expanding_consumption_rates,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)
from refillcare.data.packing import parse_pack_units, compute_total_units_purchased



class TestHistoricalConsumptionRateCalculation:
    """Tests for single-history consumption rate calculation."""

    def test_example_four_strips_sixty_units_thirty_days(self):
        """User example: 4 strips of 1X15 (60 units) consumed over ~30 days yields ~2.0 units/day."""
        history = [
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},  # 60 units
            {"invoice_date": "2026-05-01", "quantity": 4, "packing": "1X15"},  # 30 days later, bought next 60 units
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "valid"
        assert result["usable_purchase_count"] == 2
        assert result["total_elapsed_days"] == 30
        assert result["total_consumed_units"] == 60.0
        assert pytest.approx(result["historical_consumption_rate"], rel=1e-2) == 2.0

    def test_multi_visit_consistent_cadence(self):
        """Verify multi-visit history across 4 monthly cycles."""
        history = [
            {"invoice_date": "2026-01-01", "quantity": 4, "packing": "1X15"},  # 60 units
            {"invoice_date": "2026-01-31", "quantity": 4, "packing": "1X15"},  # 30 days, 60 units
            {"invoice_date": "2026-03-02", "quantity": 4, "packing": "1X15"},  # 30 days, 60 units
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},  # 30 days, 60 units
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "valid"
        assert result["usable_purchase_count"] == 4
        assert result["total_elapsed_days"] == 90
        assert result["total_consumed_units"] == 180.0  # 3 cycles of 60 units consumed
        assert pytest.approx(result["historical_consumption_rate"], rel=1e-2) == 2.0

    def test_once_daily_regimen(self):
        """Test a once-daily regimen: 1 strip (30 tabs) every 30 days -> 1.0 unit/day."""
        history = [
            {"invoice_date": "2026-01-01", "quantity": 1, "packing": "1X30"},
            {"invoice_date": "2026-01-31", "quantity": 1, "packing": "1X30"},
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "valid"
        assert result["total_elapsed_days"] == 30
        assert result["total_consumed_units"] == 30.0
        assert pytest.approx(result["historical_consumption_rate"], rel=1e-2) == 1.0

    def test_cold_start_single_purchase_returns_insufficient_data(self):
        """Verify single purchase returns insufficient_data status."""
        history = [
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "insufficient_data"
        assert result["historical_consumption_rate"] is None
        assert "at least 2" in result["reason"]

    def test_same_day_purchases_zero_elapsed_days(self):
        """Verify multiple purchases on the exact same date return insufficient_data."""
        history = [
            {"invoice_date": "2026-04-01", "quantity": 2, "packing": "1X15"},
            {"invoice_date": "2026-04-01", "quantity": 2, "packing": "1X15"},
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "insufficient_data"
        assert result["historical_consumption_rate"] is None
        assert "0 days" in result["reason"]

    def test_unparseable_packing_skipped(self):
        """Verify unparseable packing rows are excluded from usable purchase count."""
        history = [
            {"invoice_date": "2026-01-01", "quantity": 4, "packing": "1X15"},
            {"invoice_date": "2026-02-01", "quantity": 4, "packing": "UNKNOWN_PACK"},
        ]

        result = calculate_historical_consumption_rate(history)

        assert result["status"] == "insufficient_data"
        assert result["historical_consumption_rate"] is None


class TestExpandingConsumptionRates:
    """Tests for vectorized expanding history computation."""

    def test_expanding_rate_leakage_safety(self):
        """Verify expanding rate at step k uses only history up to step k."""
        df = pd.DataFrame([
            {"customerId": "C1", "itemId": "M1", "invoice_date": "2026-01-01", "quantity": 4, "packing": "1X15"},
            {"customerId": "C1", "itemId": "M1", "invoice_date": "2026-01-31", "quantity": 4, "packing": "1X15"},
            {"customerId": "C1", "itemId": "M1", "invoice_date": "2026-03-02", "quantity": 4, "packing": "1X15"},
            # Different customer / item
            {"customerId": "C2", "itemId": "M2", "invoice_date": "2026-01-01", "quantity": 1, "packing": "1X30"},
        ])

        rates = compute_expanding_consumption_rates(df)

        # Row 0: Cold start -> NaN
        assert np.isnan(rates.iloc[0])
        # Row 1: 60 units / 30 days -> 2.0
        assert pytest.approx(rates.iloc[1], rel=1e-2) == 2.0
        # Row 2: 120 units / 60 days -> 2.0
        assert pytest.approx(rates.iloc[2], rel=1e-2) == 2.0
        # Row 3: Different customer (C2) cold start -> NaN
        assert np.isnan(rates.iloc[3])


class TestEstimatedDaysOfSupply:
    """Tests for calculate_estimated_days_of_supply."""

    # ---- 1. Normal 30-day supply ----
    def test_normal_thirty_day_supply(self):
        """60 units / 2 units per day = 30 days."""
        result = calculate_estimated_days_of_supply(60, 2.0)
        assert result["status"] == "valid"
        assert result["estimated_days_of_supply"] == pytest.approx(30.0)
        assert result["current_units"] == 60
        assert result["historical_consumption_rate"] == 2.0
        assert result["reason"] is None

    # ---- 2. Bulk purchase ----
    def test_bulk_purchase_sixty_day_supply(self):
        """120 units / 2 units per day = 60 days."""
        result = calculate_estimated_days_of_supply(120, 2.0)
        assert result["status"] == "valid"
        assert result["estimated_days_of_supply"] == pytest.approx(60.0)

    # ---- 3. Partial / top-up purchase ----
    def test_partial_topup_purchase(self):
        """15 units / 2 units per day = 7.5 days."""
        result = calculate_estimated_days_of_supply(15, 2.0)
        assert result["status"] == "valid"
        assert result["estimated_days_of_supply"] == pytest.approx(7.5)

    # ---- 4. Zero units (supply exhausted) ----
    def test_zero_units_exhausted_supply(self):
        """0 units / 2 units per day = 0 days — supply is exhausted."""
        result = calculate_estimated_days_of_supply(0, 2.0)
        assert result["status"] == "valid"
        assert result["estimated_days_of_supply"] == pytest.approx(0.0)

    # ---- 5. Missing rate (None) ----
    def test_missing_rate_returns_unavailable(self):
        """When historical_consumption_rate is None, result is unavailable."""
        result = calculate_estimated_days_of_supply(60, None)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None
        assert "missing" in result["reason"].lower()

    # ---- 6. Zero rate ----
    def test_zero_rate_returns_unavailable(self):
        """Division by zero must be prevented."""
        result = calculate_estimated_days_of_supply(60, 0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None
        assert "zero" in result["reason"].lower()

    # ---- 7. Negative rate ----
    def test_negative_rate_returns_unavailable(self):
        """Negative consumption rate is not valid."""
        result = calculate_estimated_days_of_supply(60, -2.0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    # ---- 8. Negative units ----
    def test_negative_units_returns_unavailable(self):
        """Negative current_units is not valid."""
        result = calculate_estimated_days_of_supply(-10, 2.0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None
        assert "negative" in result["reason"].lower()

    # ---- 9. NaN / Inf inputs ----
    def test_nan_units_returns_unavailable(self):
        result = calculate_estimated_days_of_supply(float("nan"), 2.0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    def test_inf_rate_returns_unavailable(self):
        result = calculate_estimated_days_of_supply(60, float("inf"))
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    # ---- 10. Non-numeric string inputs ----
    def test_string_units_returns_unavailable(self):
        result = calculate_estimated_days_of_supply("abc", 2.0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    def test_string_rate_returns_unavailable(self):
        result = calculate_estimated_days_of_supply(60, "abc")
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    # ---- Decimal precision preservation ----
    def test_decimal_precision_preserved(self):
        """100 units / 3 units per day should produce a precise decimal result."""
        result = calculate_estimated_days_of_supply(100, 3.0)
        assert result["status"] == "valid"
        assert result["estimated_days_of_supply"] == pytest.approx(33.3333, rel=1e-3)

    # ---- Both inputs None ----
    def test_both_none_returns_unavailable(self):
        result = calculate_estimated_days_of_supply(None, None)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None

    # ---- Missing units ----
    def test_missing_units_returns_unavailable(self):
        """When current_units is None, result is unavailable."""
        result = calculate_estimated_days_of_supply(None, 2.0)
        assert result["status"] == "unavailable"
        assert result["estimated_days_of_supply"] is None
        assert "missing" in result["reason"].lower()


class TestTargetRefillInterval:
    """Tests for calculate_target_refill_interval (2-day adherence reminder buffer)."""

    def test_normal_thirty_day_supply_yields_twenty_eight_day_reminder(self):
        """User example: 30 days supply - 2 days buffer = 28 days target refill interval."""
        result = calculate_target_refill_interval(30.0, buffer_days=2.0)
        assert result["status"] == "valid"
        assert result["target_refill_interval"] == pytest.approx(28.0)
        assert result["estimated_days_of_supply"] == 30.0
        assert result["buffer_days"] == 2.0
        assert result["reason"] is None

    def test_bulk_sixty_day_supply(self):
        """Bulk: 60 days supply - 2 days buffer = 58 days target refill interval."""
        result = calculate_target_refill_interval(60.0, buffer_days=2.0)
        assert result["status"] == "valid"
        assert result["target_refill_interval"] == pytest.approx(58.0)

    def test_partial_seven_point_five_day_supply(self):
        """Partial: 7.5 days supply - 2 days buffer = 5.5 days target refill interval."""
        result = calculate_target_refill_interval(7.5, buffer_days=2.0)
        assert result["status"] == "valid"
        assert result["target_refill_interval"] == pytest.approx(5.5)

    def test_floor_limit_at_least_one_day(self):
        """Very small supply: 1.5 days supply - 2 days buffer = max(1.0, -0.5) = 1.0 day."""
        result = calculate_target_refill_interval(1.5, buffer_days=2.0)
        assert result["status"] == "valid"
        assert result["target_refill_interval"] == 1.0

    def test_missing_supply_returns_unavailable(self):
        result = calculate_target_refill_interval(None)
        assert result["status"] == "unavailable"
        assert result["target_refill_interval"] is None
        assert "missing" in result["reason"].lower()

    def test_zero_or_negative_supply_returns_unavailable(self):
        result_zero = calculate_target_refill_interval(0.0)
        assert result_zero["status"] == "unavailable"
        assert result_zero["target_refill_interval"] is None

        result_neg = calculate_target_refill_interval(-5.0)
        assert result_neg["status"] == "unavailable"
        assert result_neg["target_refill_interval"] is None

    def test_nan_or_string_supply_returns_unavailable(self):
        result_nan = calculate_target_refill_interval(float("nan"))
        assert result_nan["status"] == "unavailable"
        assert result_nan["target_refill_interval"] is None

        result_str = calculate_target_refill_interval("invalid")
        assert result_str["status"] == "unavailable"
        assert result_str["target_refill_interval"] is None


class TestEstimatedDaysOfSupplyRequirements:
    """Explicit tests for all user requirements for Estimated Days of Supply calculation."""

    def test_1x15_packing_and_units_calculation(self):
        """Verify parsing 1X15 and computing total units for single and multiple quantities."""
        pack_units = parse_pack_units("1X15")
        assert pack_units == 15.0

        # 4 strips * 15 tablets = 60 tablets
        total_60 = compute_total_units_purchased(4, "1X15")
        assert total_60 == 60.0

        # 1 strip * 15 tablets = 15 tablets
        total_15 = compute_total_units_purchased(1, "1X15")
        assert total_15 == 15.0

    def test_1x10_packing_and_units_calculation(self):
        """Verify parsing 1X10 and computing total units."""
        pack_units = parse_pack_units("1X10")
        assert pack_units == 10.0

        # 3 strips * 10 tablets = 30 tablets
        total_30 = compute_total_units_purchased(3, "1X10")
        assert total_30 == 30.0

        # 1 strip * 10 tablets = 10 tablets
        total_10 = compute_total_units_purchased(1, "1X10")
        assert total_10 == 10.0

    def test_multiple_packs_parsing_and_units(self):
        """Verify parsing multiple pack formats: 2X10, 10X10, 10 TAB, 30 CAP, 100ML."""
        assert parse_pack_units("2X10") == 20.0
        assert parse_pack_units("10X10") == 100.0
        assert parse_pack_units("10 TAB") == 10.0
        assert parse_pack_units("30 CAP") == 30.0
        assert parse_pack_units("100ML") == 100.0

        # 4 boxes of 2X10 = 80 units
        assert compute_total_units_purchased(4, "2X10") == 80.0
        # 2 bottles of 100ML = 200 units
        assert compute_total_units_purchased(2, "100ML") == 200.0

    def test_sixty_tablets_thirty_days_consumption_and_supply(self):
        """Verify exact user example: 4 strips x 15 tablets = 60 tablets over 30 days -> 2.0/day -> 30 days supply."""
        history = [
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},  # 60 units
            {"invoice_date": "2026-05-01", "quantity": 4, "packing": "1X15"},  # 30 days later, buys another 60 units
        ]

        # 1. Estimate daily consumption velocity from historical purchases
        rate_res = calculate_estimated_daily_consumption(history)
        assert rate_res["status"] == "valid"
        assert rate_res["estimated_daily_consumption"] == 2.0
        assert rate_res["historical_consumption_rate"] == 2.0
        assert rate_res["total_consumed_units"] == 60.0
        assert rate_res["total_elapsed_days"] == 30

        # 2. Calculate estimated days of supply for current purchase: 60 / 2.0 = 30 days
        supply_res = calculate_estimated_days_of_supply(
            current_total_units=60.0,
            estimated_daily_consumption=rate_res["estimated_daily_consumption"],
        )
        assert supply_res["status"] == "valid"
        assert supply_res["estimated_days_of_supply"] == 30.0

        # 3. Apply 2-day reminder lead-time policy: 30 - 2 = 28 days
        target_res = calculate_target_refill_interval(supply_res["estimated_days_of_supply"], buffer_days=2.0)
        assert target_res["status"] == "valid"
        assert target_res["target_refill_interval"] == 28.0

    def test_partial_purchase_supply_estimation(self):
        """Verify partial / top-up purchase (1 strip of 15 tablets with daily consumption of 2.0/day = 7.5 days supply)."""
        daily_rate = 2.0
        partial_units = compute_total_units_purchased(1, "1X15")  # 15 tablets
        assert partial_units == 15.0

        supply_res = calculate_estimated_days_of_supply(
            current_total_units=partial_units,
            estimated_daily_consumption=daily_rate,
        )
        assert supply_res["status"] == "valid"
        assert supply_res["estimated_days_of_supply"] == 7.5

        # Reminder lead-time: 7.5 - 2 = 5.5 days
        target_res = calculate_target_refill_interval(supply_res["estimated_days_of_supply"], buffer_days=2.0)
        assert target_res["status"] == "valid"
        assert target_res["target_refill_interval"] == 5.5

    def test_bulk_purchase_supply_estimation(self):
        """Verify bulk purchase (8 strips of 15 tablets = 120 tablets with daily consumption of 2.0/day = 60 days supply)."""
        daily_rate = 2.0
        bulk_units = compute_total_units_purchased(8, "1X15")  # 120 tablets
        assert bulk_units == 120.0

        supply_res = calculate_estimated_days_of_supply(
            current_total_units=bulk_units,
            estimated_daily_consumption=daily_rate,
        )
        assert supply_res["status"] == "valid"
        assert supply_res["estimated_days_of_supply"] == 60.0

        # Reminder lead-time: 60 - 2 = 58 days
        target_res = calculate_target_refill_interval(supply_res["estimated_days_of_supply"], buffer_days=2.0)
        assert target_res["status"] == "valid"
        assert target_res["target_refill_interval"] == 58.0

    def test_insufficient_history_handling(self):
        """Verify single purchase (cold start) or 0 elapsed days returns unavailable status safely."""
        single_purchase_history = [
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},
        ]
        rate_res = calculate_estimated_daily_consumption(single_purchase_history)
        assert rate_res["status"] == "insufficient_data"
        assert rate_res["estimated_daily_consumption"] is None

        # Supply calculation safely returns unavailable when rate is None
        supply_res = calculate_estimated_days_of_supply(
            current_total_units=60.0,
            estimated_daily_consumption=rate_res["estimated_daily_consumption"],
        )
        assert supply_res["status"] == "unavailable"
        assert supply_res["estimated_days_of_supply"] is None

    def test_safety_client_wording_no_clinical_dosage_claim(self):
        """Verify that returned metadata explicitly uses purchasing/consumption terminology and no clinical claims."""
        history = [
            {"invoice_date": "2026-01-01", "quantity": 4, "packing": "1X15"},
            {"invoice_date": "2026-01-31", "quantity": 4, "packing": "1X15"},
        ]
        rate_res = calculate_estimated_daily_consumption(history)
        supply_res = calculate_estimated_days_of_supply(60.0, rate_res["estimated_daily_consumption"])

        # Check key names
        assert "estimated_daily_consumption" in rate_res
        assert "estimated_days_of_supply" in supply_res
        assert "clinical_dosage" not in rate_res
        assert "prescribed_dosage" not in supply_res

