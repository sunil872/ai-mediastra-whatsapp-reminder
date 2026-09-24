"""Model evaluation, metrics calculation, subgroup performance, and interpretability for RefillCare."""

from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd


def evaluate_regression_model(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    clip_min: float = 1.0,
) -> Dict[str, Any]:
    """Calculate comprehensive regression and window-accuracy metrics with prediction clipping.

    Args:
        y_true: Ground truth target days until next purchase.
        y_pred: Raw model predictions.
        clip_min: Minimum allowable prediction in days (default: 1.0 day to prevent negative or zero refill dates).

    Returns:
        Dict[str, Any]: Full evaluation metrics report.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Clean NaNs if present
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
            "median_abs_error": None,
            "within_1_day_pct": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
        }

    # Clip predictions to valid physiological lower bound
    y_p_clipped = np.clip(y_p, clip_min, None)

    errors = y_p_clipped - y_t
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    med_ae = float(np.median(abs_errors))

    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    within_1 = float((abs_errors <= 1.0).sum() / n * 100.0)
    within_3 = float((abs_errors <= 3.0).sum() / n * 100.0)
    within_7 = float((abs_errors <= 7.0).sum() / n * 100.0)

    return {
        "count": int(n),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "median_abs_error": round(med_ae, 2),
        "within_1_day_pct": round(within_1, 2),
        "within_3_days_pct": round(within_3, 2),
        "within_7_days_pct": round(within_7, 2),
        "actual_target": {
            "mean": round(float(np.mean(y_t)), 2),
            "median": round(float(np.median(y_t)), 2),
            "min": round(float(np.min(y_t)), 2),
            "max": round(float(np.max(y_t)), 2),
        },
        "predicted_target": {
            "mean": round(float(np.mean(y_p_clipped)), 2),
            "median": round(float(np.median(y_p_clipped)), 2),
            "min": round(float(np.min(y_p_clipped)), 2),
            "max": round(float(np.max(y_p_clipped)), 2),
            "negative_raw_predictions_clipped": int((y_p < clip_min).sum()),
        },
    }


def analyze_performance_by_interval_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Analyze model accuracy across short, regular, longer, and long refill intervals.

    Groups:
    - Short: 0–15 days
    - Regular: 16–45 days
    - Longer: 46–90 days
    - Long: >90 days
    """
    y_t = df[y_true_col].values
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)

    groups = {
        "short_0_15d": (y_t >= 0) & (y_t <= 15),
        "regular_16_45d": (y_t >= 16) & (y_t <= 45),
        "longer_46_90d": (y_t >= 46) & (y_t <= 90),
        "long_gt_90d": y_t > 90,
    }

    results = {}
    for name, mask in groups.items():
        sub_t = y_t[mask]
        sub_p = y_p[mask]
        k = len(sub_t)
        if k == 0:
            results[name] = {"count": 0, "mae": None, "rmse": None, "within_7_days_pct": None}
            continue

        abs_err = np.abs(sub_p - sub_t)
        results[name] = {
            "count": int(k),
            "mae": round(float(np.mean(abs_err)), 2),
            "rmse": round(float(np.sqrt(np.mean((sub_p - sub_t) ** 2))), 2),
            "within_7_days_pct": round(float((abs_err <= 7.0).sum() / k * 100.0), 2),
        }

    return results


def analyze_performance_by_history_length(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred: np.ndarray,
    history_seq_col: str = "purchase_count_so_far",
) -> Dict[str, Dict[str, Any]]:
    """Analyze model performance as patient history grows deeper.

    Groups:
    - 2 purchases (first repeat opportunity)
    - 3–5 purchases (moderate history)
    - >5 purchases (deep chronic history)
    """
    y_t = df[y_true_col].values
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    seqs = df[history_seq_col].values if history_seq_col in df.columns else np.full(len(df), 2)

    groups = {
        "2_purchases": (seqs <= 2),
        "3_to_5_purchases": (seqs >= 3) & (seqs <= 5),
        "gt_5_purchases": (seqs > 5),
    }

    results = {}
    for name, mask in groups.items():
        sub_t = y_t[mask]
        sub_p = y_p[mask]
        k = len(sub_t)
        if k == 0:
            results[name] = {"count": 0, "mae": None, "rmse": None, "within_7_days_pct": None}
            continue

        abs_err = np.abs(sub_p - sub_t)
        results[name] = {
            "count": int(k),
            "mae": round(float(np.mean(abs_err)), 2),
            "rmse": round(float(np.sqrt(np.mean((sub_p - sub_t) ** 2))), 2),
            "within_7_days_pct": round(float((abs_err <= 7.0).sum() / k * 100.0), 2),
        }

    return results


def extract_feature_importances(
    pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and sort feature importances from a fitted Scikit-Learn or XGBoost pipeline."""
    model = pipeline.named_steps.get("model", pipeline)
    preprocessor = pipeline.named_steps.get("preprocessor", None)

    # Get feature names from preprocessor if available
    feature_names = []
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in preprocessor.get_feature_names_out()]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is None:
        return pd.DataFrame()

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return imp_df.head(top_n)
