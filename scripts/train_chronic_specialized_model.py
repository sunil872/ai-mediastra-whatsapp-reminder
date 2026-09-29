"""Enterprise Chronic-Specialized Model Training & Benchmark Pipeline.

Trains and evaluates a high-precision gradient-boosted refill prediction model
specifically tuned for Chronic Maintenance Therapies, using Median-Optimized loss
functions (MAE / Huber) and acute medication purging.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Workspace root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refillcare.data.medication_classifier import (
    classify_medication,
    compute_item_population_chronic_stats,
    is_chronic_medication,
)
from refillcare.data.transaction_classifier import classify_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainChronicModel")

PROCESSED_DIR = PROJECT_ROOT / "data" / "refillcare" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Clean predictive features
PREDICTIVE_FEATURES = [
    "quantity", "freeQuantity", "rate", "mrp", "discountPercent",
    "days_since_previous_purchase", "purchase_count_so_far", "days_since_first_purchase",
    "last_purchase_interval", "historical_interval_median", "historical_interval_mean",
    "historical_interval_std", "historical_interval_min", "historical_interval_max",
    "historical_interval_cv", "avg_historical_quantity", "quantity_vs_avg_ratio",
    "is_first_purchase", "has_multiple_prior_purchases", "is_recurring_history",
    "purchase_month", "purchase_day_of_week", "purchase_day_of_month", "purchase_quarter", "is_weekend"
]


def load_and_curate_chronic_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str]]:
    """Load and isolate clean chronic-only maintenance transactions with minimal memory overhead."""
    raw_train_p = PROCESSED_DIR / "train.parquet"
    raw_val_p = PROCESSED_DIR / "validation.parquet"
    raw_test_p = PROCESSED_DIR / "test.parquet"

    target_col = "target_days_until_next_purchase"
    base_cols = ["itemName", target_col]
    
    # Check which predictive features exist in parquet
    sample_df = pd.read_parquet(raw_train_p)
    feature_cols = [c for c in PREDICTIVE_FEATURES if c in sample_df.columns]
    load_cols = [c for c in base_cols + feature_cols if c in sample_df.columns]
    if "transactionNumber" in sample_df.columns:
        load_cols.append("transactionNumber")
    elif "invoice_number" in sample_df.columns:
        load_cols.append("invoice_number")
    del sample_df

    logger.info(f"Loading {len(load_cols)} columns from parquet partitions...")
    train_df = pd.read_parquet(raw_train_p, columns=load_cols)
    val_df = pd.read_parquet(raw_val_p, columns=load_cols)
    test_df = pd.read_parquet(raw_test_p, columns=load_cols)

    initial_counts = {
        "train_initial": len(train_df),
        "val_initial": len(val_df),
        "test_initial": len(test_df),
    }
    logger.info(f"Initial raw partition sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    item_col = "itemName"
    all_unique_items = pd.concat([
        train_df[item_col].dropna().astype(str),
        val_df[item_col].dropna().astype(str),
        test_df[item_col].dropna().astype(str),
    ]).unique()

    logger.info(f"Classifying {len(all_unique_items):,} unique medication names...")
    chronic_map = {
        item: is_chronic_medication(str(item))
        for item in all_unique_items
    }

    def _filter_partition(df: pd.DataFrame) -> pd.DataFrame:
        med_series = df[item_col].astype(str)
        is_chronic = med_series.map(chronic_map).fillna(False)
        
        tx_col = "transactionNumber" if "transactionNumber" in df.columns else ("invoice_number" if "invoice_number" in df.columns else None)
        if tx_col:
            not_b2b = ~df[tx_col].astype(str).str.strip().str.upper().str.startswith("SB/")
            return df[is_chronic & not_b2b]
        return df[is_chronic]

    logger.info("Filtering partitions by chronic eligibility and B2B removal...")
    train_chronic = _filter_partition(train_df)
    val_chronic = _filter_partition(val_df)
    test_chronic = _filter_partition(test_df)

    # Outlier Capping on Training Targets (Filter extreme lapsed gaps > 120 days from continuous training)
    train_filtered = train_chronic[
        (train_chronic[target_col] >= 3.0) & (train_chronic[target_col] <= 120.0)
    ]

    curation_summary = {
        **initial_counts,
        "train_chronic_filtered": len(train_filtered),
        "val_chronic": len(val_chronic),
        "test_chronic": len(test_chronic),
        "retention_rate_pct": round(len(train_filtered) / max(1, len(train_df)) * 100, 2),
    }
    logger.info(f"Chronic-curated partition sizes: Train={len(train_filtered):,}, Val={len(val_chronic):,}, Test={len(test_chronic):,}")
    return train_filtered, val_chronic, test_chronic, curation_summary, feature_cols


def extract_features_and_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Isolate numeric features and target arrays as float32."""
    target_col = "target_days_until_next_purchase"

    logger.info(f"Extracting {len(feature_cols)} clean predictive feature columns...")

    X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive clinical and machine learning accuracy metrics."""
    errors = np.abs(y_true - y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    med_ae = float(np.median(errors))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    acc_1d = float(np.mean(errors <= 1.0) * 100)
    acc_3d = float(np.mean(errors <= 3.0) * 100)
    acc_7d = float(np.mean(errors <= 7.0) * 100)
    acc_14d = float(np.mean(errors <= 14.0) * 100)

    return {
        "count": int(len(y_true)),
        "mae": round(mae, 2),
        "median_abs_error": round(med_ae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "within_1_day_pct": round(acc_1d, 2),
        "within_3_days_pct": round(acc_3d, 2),
        "within_7_days_pct": round(acc_7d, 2),
        "within_14_days_pct": round(acc_14d, 2),
        "actual_mean": round(float(np.mean(y_true)), 2),
        "actual_median": round(float(np.median(y_true)), 2),
        "predicted_mean": round(float(np.mean(y_pred)), 2),
        "predicted_median": round(float(np.median(y_pred)), 2),
    }


def train_and_benchmark():
    """Execute complete end-to-end training and comparative benchmarking."""
    train_df, val_df, test_df, curation_summary, feature_cols = load_and_curate_chronic_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = extract_features_and_targets(
        train_df, val_df, test_df, feature_cols
    )

    # 1. Historical Median Baseline Model
    logger.info("Computing Historical Median Baseline benchmark...")
    if "historical_interval_median" in test_df.columns:
        baseline_test_preds = test_df["historical_interval_median"].fillna(30.0).replace(0, 30.0).to_numpy(dtype=np.float32)
    else:
        baseline_test_preds = np.full(len(y_test), 30.0, dtype=np.float32)
    baseline_metrics = evaluate_predictions(y_test, baseline_test_preds)

    # 2. Train Chronic-Specialized 3-Head Quantile Regressors (P10, P50, P90)
    logger.info("Training Quantile P50 (Median) Regressor...")
    xgb_p50 = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    xgb_p50.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    logger.info("Training Quantile P10 (Lower Bound / Early Refill) Regressor...")
    xgb_p10 = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    xgb_p10.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    logger.info("Training Quantile P90 (Upper Bound / Late Threshold) Regressor...")
    xgb_p90 = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    xgb_p90.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Predictions & Monotonicity Enforcement
    raw_p50 = xgb_p50.predict(X_test)
    raw_p10 = xgb_p10.predict(X_test)
    raw_p90 = xgb_p90.predict(X_test)

    test_preds_p50 = np.clip(raw_p50, 5.0, 90.0)
    test_preds_p10 = np.clip(np.minimum(raw_p10, test_preds_p50), 3.0, 85.0)
    test_preds_p90 = np.clip(np.maximum(raw_p90, test_preds_p50), 7.0, 120.0)

    chronic_xgb_metrics = evaluate_predictions(y_test, test_preds_p50)
    
    # Quantile Coverage Evaluation (What % of actual test targets fall between P10 and P90?)
    within_quantile_envelope = float(np.mean((y_test >= test_preds_p10) & (y_test <= test_preds_p90)) * 100)
    avg_envelope_span = float(np.mean(test_preds_p90 - test_preds_p10))

    quantile_summary = {
        "p10_p90_coverage_pct": round(within_quantile_envelope, 2),
        "avg_uncertainty_span_days": round(avg_envelope_span, 2),
        "p10_mean": round(float(np.mean(test_preds_p10)), 2),
        "p50_mean": round(float(np.mean(test_preds_p50)), 2),
        "p90_mean": round(float(np.mean(test_preds_p90)), 2),
    }
    logger.info(f"Quantile Uncertainty Envelope: Coverage={within_quantile_envelope:.2f}%, Avg Span={avg_envelope_span:.1f} days")

    # 3. History-Depth Stratified Evaluation
    depth_col = "purchase_count_so_far" if "purchase_count_so_far" in test_df.columns else None
    stratified_metrics = {}
    if depth_col:
        depth_buckets = {
            "1-2 Purchases (Early Developing)": (test_df[depth_col] <= 2).to_numpy(),
            "3-5 Purchases (Stabilizing)": ((test_df[depth_col] >= 3) & (test_df[depth_col] <= 5)).to_numpy(),
            "6-10 Purchases (Recurring Chronic)": ((test_df[depth_col] >= 6) & (test_df[depth_col] <= 10)).to_numpy(),
            ">10 Purchases (High-Stability Chronic)": (test_df[depth_col] > 10).to_numpy(),
        }
        for bucket_name, mask in depth_buckets.items():
            if mask.sum() > 0:
                y_sub = y_test[mask]
                p_sub = test_preds_p50[mask]
                b_sub = baseline_test_preds[mask]
                stratified_metrics[bucket_name] = {
                    "count": int(mask.sum()),
                    "model_metrics": evaluate_predictions(y_sub, p_sub),
                    "baseline_metrics": evaluate_predictions(y_sub, b_sub),
                }

    # Top Feature Importances
    importances = xgb_p50.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:15]
    top_features = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp]

    benchmark_report = {
        "curation_summary": curation_summary,
        "overall_test_comparison": {
            "HistoricalMedianBaseline": baseline_metrics,
            "ChronicSpecializedXGBoost": chronic_xgb_metrics,
        },
        "quantile_uncertainty_envelope": quantile_summary,
        "stratified_by_history_depth": stratified_metrics,
        "top_features": top_features,
    }

    # Save serialized model bundle with multi-quantile models
    bundle = {
        "model": xgb_p50,
        "model_p10": xgb_p10,
        "model_p50": xgb_p50,
        "model_p90": xgb_p90,
        "pipeline": xgb_p50,
        "feature_cols": feature_cols,
        "features_numeric": feature_cols,
        "model_name": "ChronicQuantileXGBoost_P10_P50_P90",
        "trained_date": "2026-09-28",
        "benchmark": chronic_xgb_metrics,
        "quantile_envelope": quantile_summary,
    }
    model_output_path = MODEL_DIR / "chronic_refill_model.joblib"
    joblib.dump(bundle, model_output_path)
    logger.info(f"Saved chronic multi-quantile model bundle to {model_output_path}")

    # Also sync to default refill_model.joblib for production loader compatibility
    default_model_path = MODEL_DIR / "refill_model.joblib"
    joblib.dump(bundle, default_model_path)
    logger.info(f"Synchronized production model to {default_model_path}")

    report_output_path = PROCESSED_DIR / "chronic_model_benchmark_report.json"
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    logger.info(f"Saved benchmark report to {report_output_path}")

    return benchmark_report


if __name__ == "__main__":
    train_and_benchmark()
