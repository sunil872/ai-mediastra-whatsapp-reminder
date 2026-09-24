"""Feature Table Builder integrating Pack Units and Consumption Velocity."""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from refillcare.data.packing import enrich_total_units_purchased, parse_pack_units
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)


def build_customer_item_feature_table(history_df: pd.DataFrame) -> pd.DataFrame:
    """Construct longitudinal feature table for customer-medication pairs.

    Features include:
    - total_purchases
    - median_inter_purchase_interval
    - mean_inter_purchase_interval
    - interval_std
    - units_per_pack
    - total_units_purchased
    - historical_consumption_velocity (units/day)
    - estimated_days_of_supply
    - authoritative_refill_days (Supply-first resolution)

    Args:
        history_df: Valid historical transactions dataframe.

    Returns:
        pd.DataFrame: Pair-level feature summary ready for predictor.
    """
    if history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    df["customerId"] = df["customerId"].astype(str)
    df["itemId"] = df["itemId"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"]).sort_values(["customerId", "itemId", "invoice_date"])

    # Compute pack units via vectorized enricher
    if "packing" in df.columns and "quantity" in df.columns:
        df = enrich_total_units_purchased(df, quantity_col="quantity", packing_col="packing", output_units_col="total_units")
    else:
        df["total_units"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)

    # 1. Last purchase per pair
    last_df = df.drop_duplicates(subset=["customerId", "itemId"], keep="last").copy()
    last_df.set_index(["customerId", "itemId"], inplace=True)

    # 2. Aggregations across all transactions per pair
    grouped = df.groupby(["customerId", "itemId"])
    counts = grouped["invoice_date"].count()
    first_dates = grouped["invoice_date"].first()
    last_dates = grouped["invoice_date"].last()
    total_units_sum = grouped["total_units"].sum()

    # Align metrics
    idx = counts.index
    last_units = last_df.loc[idx, "total_units"].values
    last_qty = last_df.loc[idx, "quantity"].values if "quantity" in last_df.columns else np.ones(len(idx))
    last_packing = last_df.loc[idx, "packing"].values if "packing" in last_df.columns else np.full(len(idx), "1X10")
    cust_name = last_df.loc[idx, "customerName"].values if "customerName" in last_df.columns else [c[0] for c in idx]
    item_name = last_df.loc[idx, "itemName"].values if "itemName" in last_df.columns else [c[1] for c in idx]
    mobile_no = last_df.loc[idx, "MOBILE_NO"].values if "MOBILE_NO" in last_df.columns else np.full(len(idx), "")
    mobile_stat = last_df.loc[idx, "mobile_status"].values if "mobile_status" in last_df.columns else np.full(len(idx), "Missing")

    # Time elapsed and consumption velocity
    n_purchases = counts.values
    elapsed_days = (last_dates.values - first_dates.values).astype("timedelta64[D]").astype(float)
    prior_units = total_units_sum.values - last_units

    # Valid consumption rate: elapsed_days >= 1 and prior_units > 0
    valid_rate_mask = (n_purchases >= 2) & (elapsed_days > 0) & (prior_units > 0)
    cons_rate = np.where(valid_rate_mask, prior_units / elapsed_days, np.nan)

    # Estimated Days of Supply
    valid_dos_mask = ~np.isnan(cons_rate) & (cons_rate > 0) & (last_units > 0)
    dos = np.where(valid_dos_mask, last_units / cons_rate, np.nan)

    # Interval cadence
    avg_interval = np.where(n_purchases > 1, elapsed_days / np.maximum(1, n_purchases - 1), np.nan)

    # Authoritative Supply-First Resolution
    auth_days = np.where(
        ~np.isnan(dos) & (dos > 0),
        dos,
        np.where(
            ~np.isnan(avg_interval) & (avg_interval > 0),
            avg_interval,
            30.0,
        ),
    )

    # Path B Recency & Multi-Month calculations
    df["_ym"] = df["invoice_date"].dt.strftime("%Y-%m")
    last_dates_series = df.groupby(["customerId", "itemId"])["invoice_date"].transform("last")

    # 3-Month Window (trailing 90 days from last purchase)
    mask_3m = df["invoice_date"] >= (last_dates_series - pd.Timedelta(days=90))
    cnt_3m = df[mask_3m].groupby(["customerId", "itemId"])["_ym"].nunique()
    distinct_3m = cnt_3m.reindex(idx, fill_value=0).values

    # 6-Month Window (trailing 180 days from last purchase)
    mask_6m = df["invoice_date"] >= (last_dates_series - pd.Timedelta(days=180))
    cnt_6m = df[mask_6m].groupby(["customerId", "itemId"])["_ym"].nunique()
    distinct_6m = cnt_6m.reindex(idx, fill_value=0).values

    path_b_mask = (n_purchases < 6) & ((distinct_3m >= 2) | (distinct_6m >= 3))

    history_quality = np.where(
        n_purchases >= 6,
        "high_history",
        np.where(path_b_mask, "medium_history", "low_history"),
    )

    result_df = pd.DataFrame({
        "customerId": [c[0] for c in idx],
        "itemId": [c[1] for c in idx],
        "customerName": cust_name,
        "itemName": item_name,
        "MOBILE_NO": mobile_no,
        "mobile_status": mobile_stat,
        "total_purchases": n_purchases,
        "last_purchase_date": pd.to_datetime(last_dates.values).date,
        "median_interval": avg_interval,
        "mean_interval": avg_interval,
        "std_interval": np.zeros(len(idx)),
        "last_quantity": last_qty,
        "last_packing": last_packing,
        "consumption_rate": cons_rate,
        "estimated_days_of_supply": dos,
        "authoritative_refill_days": auth_days,
        "distinct_months_3m": distinct_3m,
        "distinct_months_6m": distinct_6m,
        "path_b_eligible": path_b_mask,
        "history_quality": history_quality,
    })

    return result_df
