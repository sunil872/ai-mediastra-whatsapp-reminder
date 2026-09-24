"""Supply-First Refill Predictor Module.

Hierarchical Resolution:
1. Physical Pack + Historical Consumption Velocity -> Estimated Days of Supply (Authoritative when valid).
2. Personal Historical Median for established recurring pairs (Path A).
3. ML Model (LightGBM/XGBoost) for eligible variable patterns.
4. Cold-start (< 2 purchases) routed to Clinical Review Queue.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from preprocessing.feature_table import build_customer_item_feature_table
from reminder.scheduler import calculate_expected_refill_date, evaluate_refill_eligibility


def predict_refill_cycles(
    history_df: pd.DataFrame,
    model_bundle: Optional[Dict[str, Any]] = None,
    prediction_date: Optional[date] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate refill predictions using the authoritative supply-first hierarchy.

    Args:
        history_df: Historical purchase transactions dataframe.
        model_bundle: Optional loaded model bundle (.pkl or .joblib).
        prediction_date: Prediction anchor date (defaults to today).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (eligible_df, ineligible_df)
    """
    if history_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if prediction_date is None:
        prediction_date = date.today()

    feature_table = build_customer_item_feature_table(history_df)
    if feature_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    cid = feature_table["customerId"].values
    iid = feature_table["itemId"].values
    n_purchases = feature_table["total_purchases"].values
    last_dts = pd.to_datetime(feature_table["last_purchase_date"])
    last_dts_str = last_dts.dt.strftime("%Y-%m-%d").values
    pred_dt_str = prediction_date.isoformat()

    dos = feature_table["estimated_days_of_supply"].values
    med = feature_table["median_interval"].values
    auth = feature_table["authoritative_refill_days"].values
    # Extract Path B distinct month metrics
    distinct_3m = feature_table["distinct_months_3m"].values if "distinct_months_3m" in feature_table.columns else np.zeros(len(cid))
    distinct_6m = feature_table["distinct_months_6m"].values if "distinct_months_6m" in feature_table.columns else np.zeros(len(cid))

    # Determine final duration and source
    is_path_a = (n_purchases >= 6)
    is_path_b_3m = (n_purchases < 6) & (distinct_3m >= 2)
    is_path_b_6m = (n_purchases < 6) & ~is_path_b_3m & (distinct_6m >= 3)
    is_path_b = is_path_b_3m | is_path_b_6m
    is_dos = ~pd.isna(dos) & (dos >= 5) & (dos <= 180)
    is_auth = ~is_dos & ~is_path_a & ~is_path_b & ~pd.isna(auth)

    final_interval = np.where(
        is_dos,
        dos,
        np.where(
            is_path_a & ~pd.isna(med),
            med,
            np.where(
                is_path_b & ~pd.isna(med) & (med >= 10) & (med <= 180),
                med,
                np.where(is_auth, auth, 30.0),
            ),
        ),
    )

    pred_source = np.where(
        is_path_b & is_dos,
        "Path B - Days of Supply (Pack-Aware)",
        np.where(
            is_path_b_3m,
            "Path B (3M Recency: >=2 Distinct Months)",
            np.where(
                is_path_b_6m,
                "Path B (6M Recency: >=3 Distinct Months)",
                np.where(
                    is_dos,
                    "Days of Supply (Pack-Aware)",
                    np.where(
                        is_path_a,
                        "Path A (Personal Historical Median)",
                        np.where(is_auth, "Hybrid Heuristic", "Standard 30-Day Refill"),
                    ),
                ),
            ),
        ),
    )

    # Dates
    int_days = np.round(final_interval).astype(int)
    exp_dts = last_dts + pd.to_timedelta(int_days, unit="D")
    rem_dts = exp_dts - pd.to_timedelta(2, unit="D")

    exp_dts_str = exp_dts.dt.strftime("%Y-%m-%d").values
    rem_dts_str = rem_dts.dt.strftime("%Y-%m-%d").values

    snap_ids = [f"snap_{c}_{i}_{pred_dt_str}_{l}" for c, i, l in zip(cid, iid, last_dts_str)]
    pilot_tiers = np.where(
        n_purchases >= 6,
        "Tier A (Strong Pilot)",
        np.where(is_path_b, "Tier B (Path B Active Pilot)", "Tier C (Cold-Start Review Queue)"),
    )

    full_df = pd.DataFrame({
        "snapshot_id": snap_ids,
        "customerId": cid,
        "itemId": iid,
        "customerName": feature_table["customerName"].values,
        "itemName": feature_table["itemName"].values,
        "MOBILE_NO": feature_table["MOBILE_NO"].values,
        "mobile_status": feature_table["mobile_status"].values,
        "total_purchases": n_purchases,
        "last_purchase_date": last_dts_str,
        "prediction_date": pred_dt_str,
        "estimated_days_of_supply": np.round(final_interval, 1),
        "expected_refill_date": exp_dts_str,
        "reminder_date": rem_dts_str,
        "prediction_source": pred_source,
        "pilot_tier": pilot_tiers,
        "status": "pending",
        "distinct_months_3m": distinct_3m,
        "distinct_months_6m": distinct_6m,
    })

    # Eligibility partition: Path A (>=6) OR Path B (recent recurrence) OR valid DOS
    is_eligible = (
        ((n_purchases >= 6) | is_path_b | is_dos)
        & (n_purchases >= 2)
        & (last_dts.dt.date <= prediction_date)
    )

    eligible_df = full_df[is_eligible].copy()
    ineligible_df = full_df[~is_eligible].copy()

    if not ineligible_df.empty:
        ineligible_df["Reason for Ineligibility"] = np.where(
            ineligible_df["total_purchases"] < 2,
            "Cold-start / Single Purchase (< 2 purchases)",
            np.where(
                pd.to_datetime(ineligible_df["last_purchase_date"]).dt.date > prediction_date,
                "Future last purchase date",
                "Insufficient distinct monthly recurrence (<2 distinct months in 3m, <3 in 6m)",
            ),
        )

    return eligible_df, ineligible_df
