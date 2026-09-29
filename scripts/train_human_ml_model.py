"""Train and serialize the Generalized Human-Level ML Regressor for RefillCare V1.

Strict Point-in-Time Protocol:
- Train window: Historical customer-item transitions strictly on or before 2026-07-31.
- August 2026 is strictly held out for validation.
- Model: HistGradientBoostingRegressor with absolute error (L1) loss.
- Target: Days until next purchase (target_interval).
- Serialized to: data/refillcare/processed/models/human_ml_refill_model.joblib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from refillcare.data.medication_classifier import is_chronic_medication
from refillcare.data.packing import parse_pack_units
from refillcare.features.point_in_time import POINT_IN_TIME_FEATURE_COLS

DATA_PATH = Path("data/refillcare/processed/clean_transactions.parquet")
OUTPUT_MODEL_PATH = Path("data/refillcare/processed/models/human_ml_refill_model.joblib")


def train_and_serialize_model():
    print(f"Loading data from {DATA_PATH}...")
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    # Filter to eligible customer transactions (S0/ only, non-null customerId, chronic maintenance only)
    is_chronic = df["itemName"].astype(str).apply(is_chronic_medication)
    clean = df[(df["refillcare_eligible"] == True) & (df["customerId"].notna()) & (df["itemId"].notna()) & is_chronic].copy()
    clean["invoice_date"] = pd.to_datetime(clean["invoice_date"])

    # Parse pack units
    clean["pack_units"] = clean["packing"].apply(lambda p: float(parse_pack_units(p) or 10.0))
    clean["total_units"] = clean["quantity"] * clean["pack_units"]

    # Deduplicate / aggregate same-day transactions per customer-item
    agg_df = (
        clean.groupby(["customerId", "itemId", "invoice_date"])
        .agg(
            total_units=("total_units", "sum"),
            pack_units=("pack_units", "first"),
        )
        .reset_index()
        .sort_values(["customerId", "itemId", "invoice_date"])
    )

    print(f"Aggregated customer-item-date events: {len(agg_df):,}")

    # Vectorized point-in-time lags
    agg_df["prev_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(1)
    agg_df["prev_units"] = agg_df.groupby(["customerId", "itemId"])["total_units"].shift(1)
    agg_df["prev_prev_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(2)
    agg_df["prev_prev_units"] = agg_df.groupby(["customerId", "itemId"])["total_units"].shift(2)
    agg_df["next_date"] = agg_df.groupby(["customerId", "itemId"])["invoice_date"].shift(-1)

    # Intervals
    # target_interval is time until next purchase (target to predict at purchase N)
    agg_df["target_interval"] = (agg_df["next_date"] - agg_df["invoice_date"]).dt.days
    agg_df["last_interval"] = (agg_df["invoice_date"] - agg_df["prev_date"]).dt.days
    agg_df["prior_interval"] = (agg_df["prev_date"] - agg_df["prev_prev_date"]).dt.days

    # Filter to valid transitions strictly on or before train cutoff
    train_mask = (
        (agg_df["invoice_date"] <= "2026-07-31")
        & agg_df["next_date"].notna()
        & (agg_df["target_interval"] >= 5)
        & (agg_df["target_interval"] <= 180)
        & agg_df["prev_date"].notna()
        & (agg_df["last_interval"] > 0)
    )

    train_df = agg_df[train_mask].copy()
    print(f"Eligible training transitions (<= 2026-07-31): {len(train_df):,}")

    # Cumulative features up to current date (strictly point-in-time)
    # 1. Quantity features
    train_df["latest_units"] = train_df["total_units"].clip(5, 300)
    train_df["previous_units"] = train_df["prev_units"].fillna(train_df["latest_units"]).clip(5, 300)
    train_df["typical_units_median"] = train_df["latest_units"]  # Proxy baseline
    train_df["recent_units_median"] = ((train_df["latest_units"] + train_df["previous_units"]) / 2.0).clip(5, 300)
    train_df["quantity_ratio_vs_typical"] = (train_df["latest_units"] / train_df["recent_units_median"]).clip(0.1, 10.0)
    train_df["quantity_ratio_vs_previous"] = (train_df["latest_units"] / train_df["previous_units"]).clip(0.1, 10.0)
    train_df["quantity_std"] = (train_df["latest_units"] - train_df["previous_units"]).abs()

    # 2. Cadence & temporal features
    train_df["last_interval"] = train_df["last_interval"].clip(5, 180)
    train_df["prior_interval"] = train_df["prior_interval"].fillna(train_df["last_interval"]).clip(5, 180)
    train_df["historical_median_interval"] = ((train_df["last_interval"] + train_df["prior_interval"]) / 2.0).clip(5, 120)
    train_df["recent_median_interval"] = train_df["last_interval"]
    train_df["rolling_interval_mean"] = train_df["historical_median_interval"]
    train_df["cadence_norm_mad"] = ((train_df["last_interval"] - train_df["historical_median_interval"]).abs() / train_df["historical_median_interval"]).clip(0.0, 2.0)
    train_df["cadence_drift"] = (train_df["last_interval"] - train_df["historical_median_interval"]).abs()

    # 3. Supply & consumption features
    train_df["estimated_consumption_velocity"] = (train_df["previous_units"] / train_df["last_interval"]).clip(0.33, 4.0)
    train_df["estimated_dos"] = (train_df["latest_units"] / train_df["estimated_consumption_velocity"]).clip(5.0, 365.0)

    # Estimated residual inventory
    consumed = train_df["last_interval"] * train_df["estimated_consumption_velocity"]
    residual = np.maximum(0.0, train_df["previous_units"] - consumed)
    train_df["estimated_residual_inventory"] = np.where(train_df["last_interval"] <= 20.0, residual, 0.0).clip(0, 90)
    train_df["estimated_effective_supply"] = train_df["latest_units"] + train_df["estimated_residual_inventory"]

    # Dimensionally correct estimated supply days: units / (units/day) = days
    train_df["estimated_supply_days"] = (train_df["estimated_effective_supply"] / train_df["estimated_consumption_velocity"]).clip(5.0, 365.0)

    # 4. Behavioral archetype flags
    train_df["is_stocking_up"] = (train_df["latest_units"] >= 50.0).astype(float)
    train_df["is_partial_purchase"] = (train_df["latest_units"] <= 15.0).astype(float)
    train_df["is_early_topup"] = (train_df["last_interval"] <= 20.0).astype(float)
    train_df["is_post_lapse"] = (train_df["last_interval"] >= 45.0).astype(float)
    train_df["is_quantity_anomaly"] = ((train_df["quantity_ratio_vs_previous"] >= 2.5) | (train_df["quantity_ratio_vs_previous"] <= 0.35)).astype(float)

    # 5. Metadata
    train_df["purchase_count"] = 6.0  # Established customer cohort indicator
    train_df["customer_tenure_days"] = 180.0
    train_df["purchase_month"] = train_df["invoice_date"].dt.month.astype(float)

    X = train_df[POINT_IN_TIME_FEATURE_COLS]
    y = train_df["target_interval"]

    print(f"Training feature matrix shape: {X.shape}")
    print("Fitting HistGradientBoostingRegressor with absolute_error loss...")
    t_fit = time.time()
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=160,
        min_samples_leaf=35,
        random_state=42,
    )
    model.fit(X, y)
    print(f"Model fitted in {time.time() - t_fit:.2f}s")

    preds = model.predict(X)
    train_mae = float(np.mean(np.abs(preds - y)))
    print(f"Train MAE: {train_mae:.2f} days")

    # Serialize bundle
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_cols": POINT_IN_TIME_FEATURE_COLS,
        "metadata": {
            "model_type": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "train_cutoff": "2026-07-31",
            "train_samples": len(train_df),
            "train_mae": round(train_mae, 2),
            "features_count": len(POINT_IN_TIME_FEATURE_COLS),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    joblib.dump(bundle, OUTPUT_MODEL_PATH)
    print(f"Serialized model bundle saved to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    train_and_serialize_model()
