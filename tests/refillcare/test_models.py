"""Unit tests for RefillCare machine learning models, evaluation, and prediction.

Uses synthetic test fixtures only (no real customer data).
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from refillcare.models.training import (
    build_preprocessor,
    build_model_pipeline,
    save_model_bundle,
    load_model_bundle,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from refillcare.models.evaluation import (
    evaluate_regression_model,
    analyze_performance_by_interval_group,
    analyze_performance_by_history_length,
    extract_feature_importances,
)
from refillcare.models.prediction import (
    predict_refill_date,
    generate_batch_predictions,
)


@pytest.fixture
def synthetic_train_val():
    """Create a minimal synthetic training and validation dataset."""
    np.random.seed(42)
    n = 60
    data = {
        "customerId": [f"CUST_{i % 5}" for i in range(n)],
        "itemId": [f"ITEM_{i % 3}" for i in range(n)],
        "itemName": ["Medicine Alpha" if i % 2 == 0 else "Medicine Beta" for i in range(n)],
        "invoice_date": [pd.Timestamp("2025-06-01") + pd.Timedelta(days=i * 5) for i in range(n)],
        "purchase_count_so_far": np.full(n, 8),
        "days_since_first_purchase": np.random.randint(0, 200, size=n),
        "days_since_previous_purchase": np.random.uniform(15, 45, size=n),
        "historical_interval_median": np.full(n, 30.0),
        "historical_interval_mean": np.full(n, 30.0),
        "historical_interval_std": np.full(n, 3.0),
        "historical_interval_min": np.full(n, 25.0),
        "historical_interval_max": np.full(n, 35.0),
        "historical_interval_cv": np.full(n, 0.1),
        "historical_interval_mad": np.full(n, 2.0),
        "historical_interval_norm_mad": np.full(n, 0.10),
        "recent3_interval_median": np.full(n, 30.0),
        "cadence_drift": np.full(n, 2.0),
        "quantity": np.random.randint(10, 60, size=n),
        "freeQuantity": np.zeros(n),
        "avg_historical_quantity": np.random.uniform(10, 60, size=n),
        "quantity_vs_avg_ratio": np.ones(n),
        "purchase_month": np.random.randint(1, 13, size=n),
        "purchase_day_of_week": np.random.randint(0, 7, size=n),
        "purchase_day_of_month": np.random.randint(1, 29, size=n),
        "purchase_day_of_year": np.random.randint(1, 365, size=n),
        "purchase_quarter": np.random.randint(1, 5, size=n),
        "is_weekend": np.random.choice([0, 1], size=n),
        "is_first_purchase": np.zeros(n),
        "has_multiple_prior_purchases": np.ones(n),
        "is_recurring_history": np.ones(n),
        "salt_category": np.random.choice(["TABLETS", "CAPSULES"], size=n),
        "salt_itemcat": np.random.choice(["ORAL", "TOPICAL"], size=n),
        "target_days_until_next_purchase": np.random.uniform(25, 35, size=n),
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:40].copy()
    val_df = df.iloc[40:].copy()
    return train_df, val_df


def test_build_model_pipeline_fit_predict(synthetic_train_val):
    """Verify preprocessor + XGBoost pipeline trains and predicts properly."""
    train_df, val_df = synthetic_train_val
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    pipeline = build_model_pipeline(model)

    pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])
    preds = pipeline.predict(val_df[feature_cols])

    assert len(preds) == len(val_df)
    assert not np.isnan(preds).any()


def test_evaluate_regression_model_clipping():
    """Verify evaluate_regression_model clips negative predictions to minimum 1.0 day."""
    y_true = np.array([30.0, 30.0, 30.0])
    y_pred_with_negatives = np.array([-5.0, 0.0, 30.0])

    metrics = evaluate_regression_model(y_true, y_pred_with_negatives, clip_min=1.0)

    # Predictions [-5.0, 0.0, 30.0] become [1.0, 1.0, 30.0]
    # Errors: |1-30|=29, |1-30|=29, |30-30|=0 -> MAE = (29+29+0)/3 = 19.33
    assert metrics["mae"] == round((29 + 29 + 0) / 3, 2)
    assert metrics["predicted_target"]["min"] >= 1.0
    assert metrics["predicted_target"]["negative_raw_predictions_clipped"] == 2


def test_subgroup_analysis(synthetic_train_val):
    """Verify subgroup analysis across interval groups and history depths."""
    _, val_df = synthetic_train_val
    preds = np.full(len(val_df), 30.0)

    interval_analysis = analyze_performance_by_interval_group(val_df, TARGET_COL, preds)
    assert "short_0_15d" in interval_analysis
    assert "regular_16_45d" in interval_analysis
    assert "longer_46_90d" in interval_analysis
    assert "long_gt_90d" in interval_analysis

    history_analysis = analyze_performance_by_history_length(val_df, TARGET_COL, preds)
    assert "2_purchases" in history_analysis
    assert "3_to_5_purchases" in history_analysis
    assert "gt_5_purchases" in history_analysis


def test_predict_refill_date_and_serialization(synthetic_train_val, tmp_path):
    """Verify single prediction calculation and joblib model bundle serialization."""
    train_df, _ = synthetic_train_val
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    pipeline = build_model_pipeline(model)
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])

    # Save bundle
    saved_file = save_model_bundle(pipeline, {"name": "test_rf"}, model_dir=tmp_path)
    assert saved_file.exists()

    # Load bundle
    loaded_bundle = load_model_bundle(saved_file)
    assert loaded_bundle["pipeline"] is not None

    # Predict refill date (Phase 17D hybrid — core regular uses personal median)
    sample_row = train_df.iloc[0]
    res = predict_refill_date(loaded_bundle, sample_row)

    assert "predicted_days_until_refill" in res
    assert "expected_refill_date" in res
    assert res["prediction_status"] == "eligible"
    assert res["predicted_days_until_refill"] >= 1.0

    # Date math verification
    cur_date = pd.to_datetime(sample_row["invoice_date"]).date()
    exp_date = pd.to_datetime(res["expected_refill_date"]).date()
    days_diff = (exp_date - cur_date).days
    assert days_diff == int(round(res["predicted_days_until_refill"]))


def test_batch_predictions(synthetic_train_val):
    """Verify generate_batch_predictions adds columns properly."""
    train_df, val_df = synthetic_train_val
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    pipeline = build_model_pipeline(model)
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])

    bundle = {
        "pipeline": pipeline,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
    }

    batch_res = generate_batch_predictions(bundle, val_df)
    assert "predicted_days_until_refill" in batch_res.columns
    assert "expected_refill_date" in batch_res.columns
    assert "refill_confidence" in batch_res.columns
    assert "prediction_status" in batch_res.columns
    assert "estimated_days_of_supply" in batch_res.columns
    assert "estimated_daily_consumption" in batch_res.columns
    assert len(batch_res) == len(val_df)
    assert (batch_res["prediction_status"] == "eligible").all()


def test_predict_refill_date_with_estimated_supply(synthetic_train_val):
    """Verify predict_refill_date populates estimated supply and daily consumption fields."""
    train_df, _ = synthetic_train_val
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    pipeline = build_model_pipeline(model)
    pipeline.fit(train_df[feature_cols], train_df[TARGET_COL])

    bundle = {
        "pipeline": pipeline,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
    }

    # Case 1: Row with valid packing and consumption rate
    row_with_supply = train_df.iloc[0].to_dict()
    row_with_supply["packing"] = "1X15"
    row_with_supply["quantity"] = 2  # 30 tablets
    row_with_supply["historical_consumption_rate"] = 1.0  # 1 tablet/day

    res = predict_refill_date(bundle, row_with_supply)
    assert res["prediction_status"] == "eligible"
    assert res["total_units_purchased"] == 30.0
    assert res["estimated_daily_consumption"] == 1.0
    assert res["estimated_days_of_supply"] == 30.0
    assert "reminder_date" in res

    # Case 2: Row with unparseable packing -> fallback with None supply metrics, no crash
    row_unparseable = train_df.iloc[0].to_dict()
    row_unparseable["packing"] = "UNKNOWN_PACK"
    row_unparseable["quantity"] = 2

    res_unparseable = predict_refill_date(bundle, row_unparseable)
    assert res_unparseable["prediction_status"] == "eligible"
    assert res_unparseable["estimated_days_of_supply"] is None
    assert res_unparseable["estimated_daily_consumption"] is None
    assert "expected_refill_date" in res_unparseable

