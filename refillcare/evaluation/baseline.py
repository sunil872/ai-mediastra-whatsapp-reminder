"""Baseline evaluation module for RefillCare refill prediction.

Implements the Historical Median Interval baseline and computes regression
and tolerance metrics (MAE, RMSE, R2, accuracy within +-1, +-3, +-7 days).
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd


class HistoricalMedianBaseline:
    """Historical Median Interval baseline model for refill timing prediction.

    Strategy:
    1. If the customer+item history has prior intervals (historical_interval_median is not NaN),
       predict the customer's personal historical median interval.
    2. If no personal history exists yet (cold start / first purchase event),
       fall back to the global training median interval.
    """

    def __init__(self, fallback_median: float = 29.0):
        self.fallback_median = fallback_median
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, target_col: str = "target_days_until_next_purchase") -> "HistoricalMedianBaseline":
        """Fit the baseline model by calculating global fallback median from training data."""
        if target_col in train_df.columns:
            valid_targets = train_df[target_col].dropna()
            if len(valid_targets) > 0:
                self.fallback_median = float(valid_targets.median())
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate baseline predictions for purchase events."""
        if "historical_interval_median" in df.columns:
            # Use personal historical median if available, otherwise fallback
            preds = df["historical_interval_median"].copy()
            preds = preds.fillna(self.fallback_median).values
        else:
            preds = np.full(len(df), self.fallback_median, dtype=np.float64)

        return np.asarray(preds, dtype=np.float64)


def compute_prediction_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """Compute regression and window-accuracy metrics.

    Metrics:
    - MAE: Mean Absolute Error (lower is better)
    - RMSE: Root Mean Squared Error (lower is better)
    - R2: Coefficient of Determination (higher is better)
    - within_1_day_pct: Percentage of predictions within +-1 day of actual
    - within_3_days_pct: Percentage of predictions within +-3 days of actual
    - within_7_days_pct: Percentage of predictions within +-7 days of actual

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics and target summary statistics.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Filter out NaNs if any exist
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]

    n = len(y_t)
    if n == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
        }

    errors = y_p - y_t
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # R-squared calculation
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    within_1 = float((abs_errors <= 1.0).sum() / n * 100.0)
    within_3 = float((abs_errors <= 3.0).sum() / n * 100.0)
    within_7 = float((abs_errors <= 7.0).sum() / n * 100.0)

    # Target statistics
    t_mean = float(np.mean(y_t))
    t_median = float(np.median(y_t))
    t_std = float(np.std(y_t, ddof=1)) if n > 1 else 0.0
    t_min = float(np.min(y_t))
    t_max = float(np.max(y_t))

    # Percentiles
    p5, p25, p75, p90, p95 = np.percentile(y_t, [5, 25, 75, 90, 95])

    return {
        "count": int(n),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "within_1_day_pct": round(within_1, 2),
        "within_3_days_pct": round(within_3, 2),
        "within_7_days_pct": round(within_7, 2),
        "target_summary": {
            "mean": round(t_mean, 2),
            "median": round(t_median, 2),
            "std": round(t_std, 2),
            "min": round(t_min, 2),
            "max": round(t_max, 2),
            "p5": round(float(p5), 2),
            "p25": round(float(p25), 2),
            "p75": round(float(p75), 2),
            "p90": round(float(p90), 2),
            "p95": round(float(p95), 2),
        },
    }


def evaluate_baseline_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target_days_until_next_purchase",
) -> Dict[str, Any]:
    """Fit baseline on train set and evaluate across Train, Validation, and Test sets."""
    baseline = HistoricalMedianBaseline()
    baseline.fit(train_df, target_col=target_col)

    results = {
        "fallback_median_days": baseline.fallback_median,
        "train": compute_prediction_metrics(train_df[target_col], baseline.predict(train_df)),
        "validation": compute_prediction_metrics(val_df[target_col], baseline.predict(val_df)),
        "test": compute_prediction_metrics(test_df[target_col], baseline.predict(test_df)),
    }

    return results
