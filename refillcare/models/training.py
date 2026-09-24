"""Training pipeline for RefillCare machine learning regression models.

Trains, tunes, compares, and serializes tree-based models (Random Forest, XGBoost, HistGradientBoosting)
for refill interval prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb

from refillcare.models.evaluation import (
    evaluate_regression_model,
    analyze_performance_by_interval_group,
    analyze_performance_by_history_length,
    extract_feature_importances,
)
from refillcare.evaluation.baseline import HistoricalMedianBaseline

NUMERIC_FEATURES = [
    "purchase_count_so_far",
    "days_since_first_purchase",
    "days_since_previous_purchase",
    "historical_interval_median",
    "historical_interval_mean",
    "historical_interval_std",
    "historical_interval_min",
    "historical_interval_max",
    "historical_interval_cv",
    "quantity",
    "freeQuantity",
    "avg_historical_quantity",
    "quantity_vs_avg_ratio",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_day_of_month",
    "purchase_day_of_year",
    "purchase_quarter",
    "is_weekend",
    "is_first_purchase",
    "has_multiple_prior_purchases",
    "is_recurring_history",
]

CATEGORICAL_FEATURES = [
    "salt_category",
    "salt_itemcat",
]

TARGET_COL = "target_days_until_next_purchase"


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numeric and nominal categorical preprocessing."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(regressor) -> Pipeline:
    """Combine feature preprocessor and estimator into a unified scikit-learn pipeline."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor),
    ])
    return pipeline


def train_and_evaluate_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filter_positive_target: bool = True,
) -> Dict[str, Any]:
    """Train baseline and ML models on Train set, compare on Validation set, and test selected model.

    Models:
    1. Historical Median Baseline
    2. Random Forest Regressor
    3. XGBoost Regressor
    4. HistGradientBoosting Regressor
    """
    # Filter to genuine refill targets (> 0) to avoid training on same-day multi-invoice events
    if filter_positive_target:
        train_clean = train_df[train_df[TARGET_COL] > 0].copy()
        val_clean = val_df[val_df[TARGET_COL] > 0].copy()
        test_clean = test_df[test_df[TARGET_COL] > 0].copy()
    else:
        train_clean = train_df.copy()
        val_clean = val_df.copy()
        test_clean = test_df.copy()

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in train_clean.columns]
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in train_clean.columns]
    if missing:
        print(f"Warning: skipping missing feature columns: {missing}")
    X_train = train_clean[feature_cols]
    y_train = train_clean[TARGET_COL].values

    X_val = val_clean[feature_cols]
    y_val = val_clean[TARGET_COL].values

    X_test = test_clean[feature_cols]
    y_test = test_clean[TARGET_COL].values

    results: Dict[str, Any] = {
        "candidate_models": {},
        "selected_model_name": None,
        "validation_comparison": {},
    }

    print(f"Training dataset size: {len(X_train):,} rows ({X_train.shape[1]} raw input features)")
    print(f"Validation dataset size: {len(X_val):,} rows")
    print(f"Test dataset size: {len(X_test):,} rows")

    # 1. Historical Median Baseline
    print("\n--- 1. Evaluating Baseline ---")
    baseline = HistoricalMedianBaseline()
    baseline.fit(train_clean, target_col=TARGET_COL)
    baseline_val_preds = baseline.predict(val_clean)
    baseline_val_metrics = evaluate_regression_model(y_val, baseline_val_preds)
    results["candidate_models"]["HistoricalMedianBaseline"] = {
        "validation_metrics": baseline_val_metrics,
        "pipeline": None,
    }
    print(f"  Baseline Val MAE: {baseline_val_metrics['mae']}d | RMSE: {baseline_val_metrics['rmse']}d | Within +-7d: {baseline_val_metrics['within_7_days_pct']}%")

    # 2. Random Forest Regressor
    print("\n--- 2. Training Random Forest Regressor ---")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf_pipe = build_model_pipeline(rf)
    rf_pipe.fit(X_train, y_train)
    rf_val_preds = rf_pipe.predict(X_val)
    rf_val_metrics = evaluate_regression_model(y_val, rf_val_preds)
    results["candidate_models"]["RandomForest"] = {
        "validation_metrics": rf_val_metrics,
        "training_time_sec": round(time.time() - t0, 2),
        "pipeline": rf_pipe,
    }
    print(f"  Random Forest trained in {time.time()-t0:.2f}s")
    print(f"  RF Val MAE: {rf_val_metrics['mae']}d | RMSE: {rf_val_metrics['rmse']}d | Within +-7d: {rf_val_metrics['within_7_days_pct']}%")

    # 3. XGBoost Regressor (Phase 17D improved objective: quantile median)
    print("\n--- 3. Training XGBoost Regressor ---")
    t0 = time.time()
    from refillcare.models.hybrid_strategy import build_improved_xgboost_regressor
    xgb_reg = build_improved_xgboost_regressor(n_jobs=-1)
    xgb_pipe = build_model_pipeline(xgb_reg)
    xgb_pipe.fit(X_train, y_train)
    xgb_val_preds = xgb_pipe.predict(X_val)
    xgb_val_metrics = evaluate_regression_model(y_val, xgb_val_preds)
    results["candidate_models"]["XGBoost"] = {
        "validation_metrics": xgb_val_metrics,
        "training_time_sec": round(time.time() - t0, 2),
        "pipeline": xgb_pipe,
    }
    print(f"  XGBoost trained in {time.time()-t0:.2f}s")
    print(f"  XGB Val MAE: {xgb_val_metrics['mae']}d | RMSE: {xgb_val_metrics['rmse']}d | Within +-7d: {xgb_val_metrics['within_7_days_pct']}%")

    # 4. HistGradientBoosting Regressor
    print("\n--- 4. Training HistGradientBoosting Regressor ---")
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
    )
    hgb_pipe = build_model_pipeline(hgb)
    hgb_pipe.fit(X_train, y_train)
    hgb_val_preds = hgb_pipe.predict(X_val)
    hgb_val_metrics = evaluate_regression_model(y_val, hgb_val_preds)
    results["candidate_models"]["HistGradientBoosting"] = {
        "validation_metrics": hgb_val_metrics,
        "training_time_sec": round(time.time() - t0, 2),
        "pipeline": hgb_pipe,
    }
    print(f"  HistGradientBoosting trained in {time.time()-t0:.2f}s")
    print(f"  HGB Val MAE: {hgb_val_metrics['mae']}d | RMSE: {hgb_val_metrics['rmse']}d | Within +-7d: {hgb_val_metrics['within_7_days_pct']}%")

    # Select Best Model based on Validation MAE
    model_maes = {
        name: data["validation_metrics"]["mae"]
        for name, data in results["candidate_models"].items()
        if data["pipeline"] is not None
    }
    best_name = min(model_maes, key=model_maes.get)
    best_pipeline = results["candidate_models"][best_name]["pipeline"]
    results["selected_model_name"] = best_name
    print(f"\n[SELECTED BEST MODEL] {best_name} (Validation MAE: {model_maes[best_name]} days)")

    # Final Evaluation on TEST SET (once only)
    print(f"\n--- 5. Final Evaluation on TEST SET ({best_name}) ---")
    test_preds = best_pipeline.predict(X_test)
    test_metrics = evaluate_regression_model(y_test, test_preds)
    results["test_evaluation"] = {
        "model_name": best_name,
        "metrics": test_metrics,
    }
    print(f"  Test MAE:  {test_metrics['mae']} days")
    print(f"  Test RMSE: {test_metrics['rmse']} days")
    print(f"  Test Within +-3d: {test_metrics['within_3_days_pct']}%")
    print(f"  Test Within +-7d: {test_metrics['within_7_days_pct']}%")

    # Subgroup Performance Analysis on Test set
    results["test_subgroups_by_interval"] = analyze_performance_by_interval_group(
        test_clean, TARGET_COL, test_preds
    )
    results["test_subgroups_by_history"] = analyze_performance_by_history_length(
        test_clean, TARGET_COL, test_preds
    )

    # Feature Importance
    results["top_features"] = extract_feature_importances(best_pipeline, top_n=15).to_dict(orient="records")

    return results


def save_model_bundle(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_dir: Optional[Union[str, Path]] = None,
    filename: str = "refill_model.joblib",
) -> Path:
    """Serialize model pipeline and metadata to disk."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    m_dir = Path(model_dir) if model_dir else base_dir / "data" / "refillcare" / "processed" / "models"
    m_dir.mkdir(parents=True, exist_ok=True)
    out_path = m_dir / filename

    bundle = {
        "pipeline": pipeline,
        "metadata": metadata,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
    }
    joblib.dump(bundle, out_path)
    return out_path


def load_model_bundle(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load serialized model bundle from disk."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    path = Path(model_path) if model_path else base_dir / "data" / "refillcare" / "processed" / "models" / "refill_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at: {path}")
    return joblib.load(path)


def run_phase4_training_pipeline(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run full Phase 4 training, evaluation, reporting, and model serialization."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    t_path = Path(train_path) if train_path else base_dir / "data" / "refillcare" / "processed" / "train.parquet"
    v_path = Path(val_path) if val_path else base_dir / "data" / "refillcare" / "processed" / "validation.parquet"
    te_path = Path(test_path) if test_path else base_dir / "data" / "refillcare" / "processed" / "test.parquet"
    out_dir = Path(output_dir) if output_dir else base_dir / "data" / "refillcare" / "processed"

    print("================================================================")
    print("STARTING REFILLCARE MODEL TRAINING PIPELINE (PHASE 4)")
    print("================================================================")

    train_df = pd.read_parquet(t_path)
    val_df = pd.read_parquet(v_path)
    test_df = pd.read_parquet(te_path)

    results = train_and_evaluate_all_models(train_df, val_df, test_df)

    # Save best model
    best_name = results["selected_model_name"]
    best_pipeline = results["candidate_models"][best_name]["pipeline"]
    model_metadata = {
        "model_name": best_name,
        "train_rows": len(train_df),
        "validation_mae": results["candidate_models"][best_name]["validation_metrics"]["mae"],
        "test_mae": results["test_evaluation"]["metrics"]["mae"],
        "trained_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "train_date_min": str(pd.to_datetime(train_df["invoice_date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train_df["invoice_date"]).max().date()),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "dataset_span": "2020-12-24 to 2026-08-31",
    }
    saved_path = save_model_bundle(best_pipeline, model_metadata, model_dir=out_dir / "models")
    print(f"\nModel artifact serialized to: {saved_path}")

    # Prepare JSON serializable report
    report = {
        "phase": "PHASE_4",
        "status": "COMPLETE",
        "selected_model": best_name,
        "validation_comparison": {
            k: v["validation_metrics"] for k, v in results["candidate_models"].items()
        },
        "test_results": results["test_evaluation"]["metrics"],
        "test_subgroups_by_interval": results["test_subgroups_by_interval"],
        "test_subgroups_by_history_length": results["test_subgroups_by_history"],
        "top_features": results["top_features"],
    }

    report_path = out_dir / "phase4_model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Phase 4 Evaluation Report written to: {report_path}")

    print("\n================================================================")
    print("REFILLCARE MODEL TRAINING PIPELINE COMPLETED")
    print("================================================================")

    return report


if __name__ == "__main__":
    run_phase4_training_pipeline()
