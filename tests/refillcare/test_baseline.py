"""Unit tests for RefillCare baseline evaluation and metrics.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.evaluation.baseline import (
    HistoricalMedianBaseline,
    compute_prediction_metrics,
    evaluate_baseline_predictions,
)


def test_baseline_predictions():
    """Verify baseline predictions use personal median or fallback correctly."""
    train_df = pd.DataFrame({
        "historical_interval_median": [30.0, 45.0, np.nan],
        "target_days_until_next_purchase": [30.0, 45.0, 30.0],
    })

    baseline = HistoricalMedianBaseline()
    baseline.fit(train_df)

    # Global training fallback should be median of target [30, 45, 30] -> 30.0
    assert baseline.fallback_median == 30.0

    test_df = pd.DataFrame({
        "historical_interval_median": [60.0, np.nan],
    })
    preds = baseline.predict(test_df)

    assert preds[0] == 60.0  # Used personal median
    assert preds[1] == 30.0  # Used fallback


def test_compute_prediction_metrics_perfect():
    """Verify metrics for perfect predictions (MAE=0, RMSE=0, R2=1.0, 100% within tolerance)."""
    y_true = np.array([30.0, 45.0, 60.0])
    y_pred = np.array([30.0, 45.0, 60.0])

    metrics = compute_prediction_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["within_1_day_pct"] == 100.0
    assert metrics["within_3_days_pct"] == 100.0
    assert metrics["within_7_days_pct"] == 100.0


def test_compute_prediction_metrics_with_errors():
    """Verify metrics calculation with known errors."""
    y_true = np.array([30.0, 30.0, 30.0, 30.0])
    y_pred = np.array([30.0, 31.0, 35.0, 40.0])  # Errors: [0, 1, 5, 10]

    metrics = compute_prediction_metrics(y_true, y_pred)
    assert metrics["mae"] == (0 + 1 + 5 + 10) / 4  # 4.0
    assert metrics["within_1_day_pct"] == 50.0      # 2 out of 4 (diffs: 0, 1)
    assert metrics["within_3_days_pct"] == 50.0      # 2 out of 4 (diffs: 0, 1)
    assert metrics["within_7_days_pct"] == 75.0      # 3 out of 4 (diffs: 0, 1, 5)


def test_evaluate_baseline_predictions_pipeline():
    """Verify complete baseline evaluation workflow across train, val, and test."""
    train_df = pd.DataFrame({
        "historical_interval_median": [30.0, 30.0],
        "target_days_until_next_purchase": [30.0, 30.0],
    })
    val_df = pd.DataFrame({
        "historical_interval_median": [30.0, np.nan],
        "target_days_until_next_purchase": [30.0, 35.0],
    })
    test_df = pd.DataFrame({
        "historical_interval_median": [30.0],
        "target_days_until_next_purchase": [32.0],
    })

    results = evaluate_baseline_predictions(train_df, val_df, test_df)

    assert "train" in results
    assert "validation" in results
    assert "test" in results
    assert results["train"]["mae"] == 0.0
    assert results["validation"]["count"] == 2
    assert results["test"]["count"] == 1
