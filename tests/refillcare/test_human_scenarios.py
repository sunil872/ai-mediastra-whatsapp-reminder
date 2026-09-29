"""Generalized Behavioral Regression Test Suite for RefillCare V1 Path A.

Validates that the multi-signal consensus engine handles all real-world human
purchasing archetypes organically across generalized signals without customer hardcoding.
"""

from __future__ import annotations

from datetime import date, timedelta
import pytest

from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.engine.decision_types import (
    PATH_A,
    PATH_B,
    STABILITY_HIGH,
    STABILITY_MEDIUM_SAFE,
    STABILITY_UNSTABLE,
)
from refillcare.models.supply_hybrid import (
    calculate_estimated_residual_inventory,
    calculate_dimensionally_correct_supply_bounds,
)


class TestGeneralizedHumanScenarios:
    """Test suite verifying generalized behavioral archetypes across all customer-items."""

    def test_01_normal_recurring_customer(self):
        """Stable 30-unit recurring customer with ~30d gaps retains expected ~30d cadence."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 1, 31),
            date(2026, 3, 2),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 5, 31),
            date(2026, 6, 30),
        ]
        quantities = [1.0] * 7
        packings = ["1X30"] * 7

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="GEN_RECURRING",
            customer_name="Stable Customer",
            mobile_no="919876543210",
            item_id="MED_30",
            item_name="AMLODIPINE 5MG TAB 1X30",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 7, 1),
        )

        assert dec.is_eligible is True
        assert dec.path == PATH_A
        assert dec.stability_tier in (STABILITY_HIGH, STABILITY_MEDIUM_SAFE)
        # Regular recurring customer predicted interval matches ~30 days
        assert 28 <= dec.predicted_interval_days <= 32
        assert dec.expected_refill_date == date(2026, 6, 30) + timedelta(days=dec.predicted_interval_days)

    def test_02_multi_pack_stocking_up(self):
        """Customer buying 90 units (3 packs of 30) scales up to ~90 days, not stuck at 30d median."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 2, 1),
            date(2026, 3, 3),
            date(2026, 4, 2),
            date(2026, 5, 2),
            date(2026, 6, 1),
            date(2026, 7, 1),  # Bought 3 packs = 90 tabs for extended travel
        ]
        quantities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
        packings = ["1X30"] * 7

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="GEN_STOCKING",
            customer_name="Traveler Customer",
            mobile_no="919876543211",
            item_id="MED_30",
            item_name="METOPROLOL 50MG TAB 1X30",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 7, 2),
        )

        # 90 tabs triggers bulk guardrail or scales to >= 60 days
        if dec.is_eligible:
            assert dec.predicted_interval_days >= 60
        else:
            assert "BULK_PURCHASE_DIVERGENCE" in dec.decision_reason or "review required" in dec.decision_reason.lower()

    def test_03_emergency_partial_strip_purchase(self):
        """Customer buying 1 strip of 10 instead of usual 30 is capped by physical supply."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 2, 1),
            date(2026, 3, 1),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 6, 1),
            date(2026, 7, 1),  # Bought only 1 strip of 10 tabs (cash-flow bridge)
        ]
        quantities = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0]
        packings = ["1X10"] * 7

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="GEN_PARTIAL",
            customer_name="Partial Buyer",
            mobile_no="919876543212",
            item_id="MED_10",
            item_name="ROSUVASTATIN 10MG TAB 1X10",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 7, 2),
        )

        assert dec.is_eligible is True
        assert dec.path == PATH_A
        # 10 tablets physically cannot last 30 days; must scale down to <= 15 days
        assert dec.predicted_interval_days <= 15
        assert dec.expected_refill_date <= date(2026, 7, 1) + timedelta(days=15)

    def test_04_early_topup_with_residual_carryover(self):
        """Customer visiting at 18 days after 30u buy and buying 60u incorporates carryover stock."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 2, 1),
            date(2026, 3, 4),
            date(2026, 4, 3),
            date(2026, 5, 4),
            date(2026, 6, 5),
            date(2026, 7, 6),
            date(2026, 8, 3),   # Bought 30 units (gap 28d)
            date(2026, 8, 21),  # Early visit at 18 days! Bought 60 units
        ]
        quantities = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0]
        packings = ["1X15"] * 8

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="GEN_EARLY_TOPUP",
            customer_name="Early Topup Customer",
            mobile_no="919876543213",
            item_id="MED_15",
            item_name="GLIMEPIRIDE 2MG TAB 1X15",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 8, 22),
        )

        assert dec.is_eligible is True
        assert dec.path == PATH_A
        # With 60 new units + ~8 residual units = ~68 units, predicted interval must be >= 45 days
        assert dec.predicted_interval_days >= 45
        assert dec.expected_refill_date >= date(2026, 8, 21) + timedelta(days=45)

    def test_05_post_lapse_restart(self):
        """Customer returning after a 60-day gap resets residual inventory to zero."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 2, 1),
            date(2026, 3, 2),
            date(2026, 4, 1),
            date(2026, 5, 2),
            date(2026, 7, 2),  # 61-day gap (bought outside / skipped)
        ]
        quantities = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0]  # Restart buy: 1 strip of 10
        packings = ["1X10"] * 6

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="GEN_POST_LAPSE",
            customer_name="Post Lapse Customer",
            mobile_no="919876543214",
            item_id="MED_10",
            item_name="METFORMIN 1000MG TAB 1X10",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 7, 3),
        )

        assert dec.is_eligible is True
        # 10 units after lapse must not predict lifetime median (30d); capped at <= 15 days
        assert dec.predicted_interval_days <= 15

    def test_06_dimensional_correctness_invariant(self):
        """Assert that supply bounds are strictly derived in days via units / (units/day)."""
        effective_units = 68.0  # units
        consumption_velocity = 1.22  # units/day

        supply_days, lower_bound, upper_bound = calculate_dimensionally_correct_supply_bounds(
            effective_units=effective_units,
            consumption_velocity=consumption_velocity,
            lower_factor=0.65,
            upper_factor=1.50,
            min_lower_days=7.0,
            min_upper_days=10.0,
        )

        # 68 units / 1.22 units/day = 55.7 days
        assert 54.0 <= supply_days <= 57.0
        # Lower bound = 55.7 * 0.65 = 36.2 days
        assert 35.0 <= lower_bound <= 38.0
        # Upper bound = 55.7 * 1.50 = 83.6 days
        assert 80.0 <= upper_bound <= 86.0
        # Dimensional check: Lower bound MUST NOT equal 68 * 0.65 (44.2)!
        assert lower_bound != round(68.0 * 0.65, 1)

    def test_07_residual_inventory_zero_after_30_days(self):
        """Residual inventory must be strictly 0.0 if prior gap exceeds 30 days."""
        # Gap of 35 days with previous buy of 30 units
        res = calculate_estimated_residual_inventory(
            previous_units=30.0,
            elapsed_days=35.0,
            consumption_velocity=1.0,
        )
        assert res == 0.0

        # Gap of 18 days with previous buy of 30 units and 1.0 unit/day rate
        res_early = calculate_estimated_residual_inventory(
            previous_units=30.0,
            elapsed_days=18.0,
            consumption_velocity=1.0,
        )
        # Leftover = 30 - 18 = 12.0 units
        assert res_early == 12.0


class TestRepresentativeCaseStudies:
    """Representative regression case studies from actual pharmacy operational incidents."""

    def test_case_study_murlikrishna(self):
        """Customer 4: Murlikrishna (Reclide XR 60mg).
        
        Bought 60 tabs on 2026-08-21 just 18 days after buying 30 tabs on 2026-08-03.
        Old Engine: Predicted 2026-09-19 (29d median), firing an overdue Day +5 alert on Sept 24.
        New Engine: Incorporates residual stock, predicting >= 45 days (expected refill >= Oct 5).
        Result: Day +5 alert on Sept 24 is successfully suppressed.
        """
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 3, 13),
            date(2026, 4, 2),
            date(2026, 5, 4),
            date(2026, 6, 4),
            date(2026, 7, 3),
            date(2026, 8, 3),   # Bought 2 packs (30 tabs)
            date(2026, 8, 21),  # Bought 4 packs (60 tabs) at 18 days
        ]
        quantities = [3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 4.0]
        packings = ["1X15"] * 7

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="919441723455_MURLIKRISHNA",
            customer_name="MURLIKRISHNA",
            mobile_no="919441723455",
            item_id="1148",
            item_name="RECLIDE XR 60MG TAB 1X15",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 9, 24),
        )

        assert dec.is_eligible is True
        assert dec.path == PATH_A
        # Must predict >= 45 days, never 29 days
        assert dec.predicted_interval_days >= 45
        assert dec.expected_refill_date >= date(2026, 10, 5)

        # Verify downstream reminder lifecycle on Sept 24:
        # Since expected refill date is >= Oct 5, on Sept 24 the patient is at Day -11 or earlier.
        # No Day +5 alert can fire!
        days_offset_on_sept_24 = (date(2026, 9, 24) - dec.expected_refill_date).days
        assert days_offset_on_sept_24 < 0  # Still before refill date!

    def test_case_study_narasimulu(self):
        """Customer 1: Narasimulu (Revlamer 400mg).
        
        Bought only 1 strip (10 tabs) on 2026-08-27 after a 59-day lapse.
        Old Engine: Assigned 23-day median, leaving patient without reminders until Sept 16.
        New Engine: Capped by physical 10-tablet supply limit (predicted interval <= 15 days).
        Expected Refill: <= 2026-09-11.
        """
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 31),
            date(2026, 3, 3),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 5, 26),
            date(2026, 6, 29),
            date(2026, 8, 27),  # 59-day gap, bought 1 strip of 10
        ]
        quantities = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
        packings = ["1X10"] * 7

        dec = engine.evaluate_customer_item_trajectory(
            customer_id="919441113276_NARASIMULU",
            customer_name="NARASIMULU",
            mobile_no="919441113276",
            item_id="5977",
            item_name="REVLAMER 400MG TAB 1X10",
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=date(2026, 9, 24),
        )

        assert dec.is_eligible is True
        assert dec.path == PATH_A
        # Must recognize 10-tablet physical constraint (predicted interval <= 15 days)
        assert dec.predicted_interval_days <= 15
        assert dec.expected_refill_date <= date(2026, 9, 11)
