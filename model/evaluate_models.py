"""Model Evaluation and Metric Extraction Module."""

from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from refillcare.models.evaluation import (
    evaluate_regression_model,
    analyze_performance_by_interval_group,
    analyze_performance_by_history_length,
)


def evaluate_models(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics: MAE, RMSE, bias, within bounds."""
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    within_1 = float(np.mean(np.abs(diff) <= 1.0) * 100)
    within_3 = float(np.mean(np.abs(diff) <= 3.0) * 100)
    within_7 = float(np.mean(np.abs(diff) <= 7.0) * 100)

    return {
        "mae_days": round(mae, 2),
        "rmse_days": round(rmse, 2),
        "mean_signed_bias_days": round(bias, 2),
        "within_1_day_pct": round(within_1, 1),
        "within_3_days_pct": round(within_3, 1),
        "within_7_days_pct": round(within_7, 1),
    }
