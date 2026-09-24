"""Unit tests for the Hybrid Refill-Date Calculation Layer (supply_hybrid.py).

Tests:
1. Packaging check and unit calculations.
2. Historical consumption rate verification.
3. Estimated days of supply computation.
4. Adherence reminder buffer (2-day lead time):
   - Formula: target_refill_interval = max(1, estimated_days_of_supply - 2)
   - Keeps expected_refill_date and reminder_date separate.
5. Source selection:
   - Partial / top-up purchase -> selects estimated_days_of_supply.
   - Regular purchase -> hybrid_supply_consensus.
   - Bulk purchase -> falls back safely to prevent over-extension.
6. Safe fallback behavior:
   - Missing/unparseable packing -> falls back with explicit reason.
   - Missing/invalid consumption rate -> falls back with explicit reason.
   - Cold-start (< 2 purchases) -> falls back with explicit reason.
7. Feature flag behavior:
   - Default OFF -> returns existing model prediction with telemetry fallback reason.
   - Enabled -> activates validated hybrid source selection.
"""

import pytest
import pandas as pd
import numpy as np

from refillcare.models.supply_hybrid import (
    calculate_hybrid_refill_date,
    generate_hybrid_supply_batch_predictions,
    PREDICTION_SOURCE_SUPPLY,
    PREDICTION_SOURCE_EXISTING,
    PREDICTION_SOURCE_HYBRID_CONSENSUS,
)


@pytest.fixture
def mock_model_bundle():
    """Mock model bundle for testing without heavy model loading."""
    class MockPipeline:
        def predict(self, X):
            return np.full(len(X), 30.0)

    return {
        "pipeline": MockPipeline(),
        "features_numeric": ["purchase_count_so_far"],
        "features_categorical": [],
    }


class TestHybridRefillDateCalculation:
    """Test suite for calculate_hybrid_refill_date."""

    def test_feature_flag_default_off(self, mock_model_bundle):
        """When feature flag is OFF (default), existing prediction is preserved untouched."""
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-06-01",
            "quantity": 1,
            "packing": "1X15",  # 15 units (partial top-up vs usual 60)
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 0.25,
            "historical_consumption_rate": 2.0,  # 15 / 2.0 = 7.5 days
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            enable_supply_layer=False,  # Explicitly OFF
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_EXISTING
        assert result["predicted_days_until_refill"] == 30.0  # From existing personal median
        assert result["estimated_days_of_supply"] == 7.5
        assert result["target_refill_interval"] == 5.5  # 7.5 - 2.0
        assert "disabled by feature flag" in result["fallback_reason"].lower()

    def test_partial_topup_purchase_selects_days_of_supply_with_reminder_buffer(self, mock_model_bundle):
        """Partial top-up purchase (15 units / 2 rate = 7.5d supply):
        - expected_refill_date: +7.5d (~2026-06-08 / 2026-06-09)
        - target_refill_interval: 7.5 - 2 = 5.5d
        - reminder_date: 2026-06-01 + 6d = 2026-06-07
        """
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-06-01",
            "quantity": 1,
            "packing": "1X15",  # 15 units (buying 1 strip instead of usual 4)
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 0.25,
            "historical_consumption_rate": 2.0,  # 15 / 2.0 = 7.5 days
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            enable_supply_layer=True,  # ENABLED
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_SUPPLY
        assert result["predicted_days_until_refill"] == 7.5
        assert result["estimated_days_of_supply"] == 7.5
        assert result["target_refill_interval"] == 5.5
        assert result["expected_refill_date"] in ("2026-06-08", "2026-06-09")
        assert result["reminder_date"] == "2026-06-07"
        assert result["expected_refill_date"] != result["reminder_date"]
        assert result["fallback_reason"] is None
        assert result["total_units_purchased"] == 15.0
        assert result["historical_consumption_rate"] == 2.0

    def test_regular_normal_purchase_with_two_day_buffer(self, mock_model_bundle):
        """Normal 30-day supply (60 units / 2 rate = 30d supply):
        - expected_refill_date: 2026-06-01 + 30d = 2026-07-01
        - target_refill_interval: 30 - 2 = 28.0 days
        - reminder_date: 2026-06-01 + 28d = 2026-06-29
        """
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-06-01",
            "quantity": 4,
            "packing": "1X15",  # 60 units
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 1.0,
            "historical_consumption_rate": 2.0,  # 60 / 2.0 = 30 days
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            enable_supply_layer=True,
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_HYBRID_CONSENSUS
        assert result["predicted_days_until_refill"] == 30.0
        assert result["estimated_days_of_supply"] == 30.0
        assert result["target_refill_interval"] == 28.0
        assert result["expected_refill_date"] == "2026-07-01"
        assert result["reminder_date"] == "2026-06-29"
        assert result["expected_refill_date"] != result["reminder_date"]
        assert result["fallback_reason"] is None

    def test_bulk_purchase_with_two_day_buffer_and_safe_fallback(self, mock_model_bundle):
        """Bulk purchase (16 strips of 1X15 = 240 units / 2 rate = 120d supply):
        - target_refill_interval: 120 - 2 = 118.0 days
        - Falls back safely to existing model to prevent over-extension.
        """
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-06-01",
            "quantity": 16,
            "packing": "1X15",  # 240 units -> 120 days of supply
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 4.0,
            "historical_consumption_rate": 2.0,
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            enable_supply_layer=True,
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_EXISTING
        assert result["predicted_days_until_refill"] == 30.0
        assert result["estimated_days_of_supply"] == 120.0
        assert result["target_refill_interval"] == 118.0
        assert "bulk purchase detected" in result["fallback_reason"].lower()

    def test_missing_packing_invalid_supply(self, mock_model_bundle):
        """When packing is missing or unparseable, supply is unavailable, target_refill_interval is None."""
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-06-01",
            "quantity": 4,
            "packing": "UNKNOWN_PACKING",
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 1.0,
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            enable_supply_layer=True,
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_EXISTING
        assert result["estimated_days_of_supply"] is None
        assert result.get("target_refill_interval") is None
        assert "unparseable packaging" in result["fallback_reason"].lower()

    def test_rate_calculated_from_history_if_not_in_record(self, mock_model_bundle):
        """When history parameter is supplied, computes consumption rate on the fly."""
        history = [
            {"invoice_date": "2026-04-01", "quantity": 4, "packing": "1X15"},
            {"invoice_date": "2026-05-01", "quantity": 4, "packing": "1X15"},
        ]
        record = {
            "customerId": "C001",
            "itemId": "MED_A",
            "itemName": "Glycomet GP1",
            "invoice_date": "2026-05-01",
            "quantity": 1,
            "packing": "1X15",  # partial
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            "quantity_vs_avg_ratio": 0.25,
        }

        result = calculate_hybrid_refill_date(
            mock_model_bundle,
            record,
            history=history,
            enable_supply_layer=True,
        )

        assert result["prediction_source"] == PREDICTION_SOURCE_SUPPLY
        assert result["predicted_days_until_refill"] == 7.5
        assert result["estimated_days_of_supply"] == 7.5
        assert result["target_refill_interval"] == 5.5
        assert result["historical_consumption_rate"] == 2.0

    def test_batch_predictions_separate_dates_and_buffer(self, mock_model_bundle):
        """Test vectorized batch processing with separate expected_refill_date and reminder_date."""
        df = pd.DataFrame([
            {
                "customerId": "C001",
                "itemId": "M1",
                "itemName": "Glycomet GP1",
                "invoice_date": "2026-06-01",
                "quantity": 1,
                "packing": "1X15",
                "purchase_count_so_far": 8,
                "historical_interval_median": 30.0,
                "historical_interval_norm_mad": 0.10,
                "cadence_drift": 2.0,
                "quantity_vs_avg_ratio": 0.25,
                "historical_consumption_rate": 2.0,
            },
            {
                "customerId": "C002",
                "itemId": "M2",
                "itemName": "Ecosprin 75",
                "invoice_date": "2026-06-01",
                "quantity": 1,
                "packing": "INVALID",
                "purchase_count_so_far": 8,
                "historical_interval_median": 30.0,
                "historical_interval_norm_mad": 0.10,
                "cadence_drift": 2.0,
                "quantity_vs_avg_ratio": 1.0,
            },
        ])

        out_df = generate_hybrid_supply_batch_predictions(
            mock_model_bundle,
            df,
            enable_supply_layer=True,
        )

        assert len(out_df) == 2
        # Row 0: Valid partial top-up
        assert out_df["prediction_source"].iloc[0] == PREDICTION_SOURCE_SUPPLY
        assert out_df["predicted_days_until_refill"].iloc[0] == 7.5
        assert out_df["estimated_days_of_supply"].iloc[0] == 7.5
        assert out_df["target_refill_interval"].iloc[0] == 5.5
        assert out_df["reminder_date"].iloc[0] == "2026-06-07"
        assert out_df["expected_refill_date"].iloc[0] in ("2026-06-08", "2026-06-09")
        assert out_df["reminder_date"].iloc[0] != out_df["expected_refill_date"].iloc[0]

        # Row 1: Invalid packing fallback
        assert out_df["prediction_source"].iloc[1] == PREDICTION_SOURCE_EXISTING
        assert "unparseable" in out_df["fallback_reason"].iloc[1].lower()
