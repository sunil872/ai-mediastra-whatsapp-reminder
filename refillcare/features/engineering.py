"""Leakage-safe feature engineering module for RefillCare.

Constructs historical, interval, calendar, and item-context features from canonical
purchase histories strictly using information available on or before each purchase event.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from refillcare.data.packing import enrich_total_units_purchased


FEATURE_COLUMNS_NUMERIC = [
    "purchase_count_so_far",
    "days_since_first_purchase",
    "days_since_previous_purchase",
    "last_purchase_interval",
    "historical_interval_median",
    "historical_interval_mean",
    "historical_interval_std",
    "historical_interval_min",
    "historical_interval_max",
    "historical_interval_cv",
    "historical_interval_mad",
    "historical_interval_norm_mad",
    "recent3_interval_median",
    "cadence_drift",
    # Phase 17F Path A eligibility diagnostics (not used by production model features)
    "pct_extreme_lt10_or_gt180",
    "pct_in_refill_band_15_120",
    "pct_within_50pct_of_median",
    "max_over_median",
    "quantity",
    "freeQuantity",
    "avg_historical_quantity",
    "quantity_vs_avg_ratio",
    "netAmount",
    "gstAmount",
    "rate",
    "mrp",
    "discountPercent",
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

FEATURE_COLUMNS_CATEGORICAL = [
    "itemId",
    "itemCode",
    "itemName",
    "therapeuticCategory",
    "generic_name",
    "salt_composition",
    "salt_category",
    "salt_itemcat",
    "salt_pack",
]

TARGET_COLUMN = "target_days_until_next_purchase"

SORT_KEY = ["customerId", "itemId", "invoice_date", "invoice_number"]


def build_feature_dataset(history_df: pd.DataFrame) -> pd.DataFrame:
    """Construct leakage-safe features and supervised targets from purchase history.

    Leakage Protection Principles:
    1. For every purchase event i at date t_i, all features use ONLY information up to event i.
    2. Historical interval statistics (median, mean, std, min, max, cv) are computed over intervals
       that concluded on or before event i (i.e. interval between i-1 and i, and earlier).
    3. Future interval (t_{i+1} - t_i) is strictly assigned as the target variable
       'target_days_until_next_purchase' and is excluded from all feature columns.
    4. The final purchase in any history has target_days_until_next_purchase = NaN.

    Args:
        history_df: Canonical purchase history DataFrame from Phase 2.

    Returns:
        pd.DataFrame: Feature dataset containing all engineered features and target.
    """
    if history_df.empty:
        return history_df.copy()

    df = history_df.copy()

    # Ensure chronological sort
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], dayfirst=True)

    df = df.sort_values(by=SORT_KEY, ascending=True).reset_index(drop=True)

    # Enrich physical units if packing column is available
    if "packing" in df.columns:
        df = enrich_total_units_purchased(df)

    # 1. Target Construction: Days Until Next Purchase (forward shift by 1)
    grouped = df.groupby(["customerId", "itemId"], sort=False)
    next_date = grouped["invoice_date"].shift(-1)
    df[TARGET_COLUMN] = (next_date - df["invoice_date"]).dt.days

    # 2. Historical Timeline & Sequences
    df["purchase_count_so_far"] = grouped.cumcount() + 1
    if "first_purchase_date" not in df.columns:
        df["first_purchase_date"] = grouped["invoice_date"].transform("min")

    df["days_since_first_purchase"] = (df["invoice_date"] - df["first_purchase_date"]).dt.days

    # Alias last purchase interval
    df["last_purchase_interval"] = df["days_since_previous_purchase"]

    # 3. Fast Contiguous Computation for Historical Interval Statistics
    # Because df is sorted by customerId + itemId, consecutive rows with the same key form contiguous groups.
    n = len(df)
    cust_ids = df["customerId"].values
    item_ids = df["itemId"].values
    intervals = df["days_since_previous_purchase"].values
    quantities = df["quantity"].values

    hist_median = np.full(n, np.nan, dtype=np.float64)
    hist_mean = np.full(n, np.nan, dtype=np.float64)
    hist_std = np.full(n, np.nan, dtype=np.float64)
    hist_min = np.full(n, np.nan, dtype=np.float64)
    hist_max = np.full(n, np.nan, dtype=np.float64)
    hist_cv = np.full(n, np.nan, dtype=np.float64)
    hist_mad = np.full(n, np.nan, dtype=np.float64)
    hist_norm_mad = np.full(n, np.nan, dtype=np.float64)
    recent3_med = np.full(n, np.nan, dtype=np.float64)
    cadence_drift = np.full(n, np.nan, dtype=np.float64)
    pct_extreme = np.full(n, np.nan, dtype=np.float64)
    pct_in_band = np.full(n, np.nan, dtype=np.float64)
    pct_within_50 = np.full(n, np.nan, dtype=np.float64)
    max_over_med = np.full(n, np.nan, dtype=np.float64)
    avg_qty = np.full(n, np.nan, dtype=np.float64)

    current_intervals = []
    current_quantities = []
    prev_key = None

    for i in range(n):
        key = (cust_ids[i], item_ids[i])
        if key != prev_key:
            current_intervals = []
            current_quantities = []
            prev_key = key

        # Track quantity expanding history
        q_val = quantities[i]
        current_quantities.append(q_val)
        avg_qty[i] = np.mean(current_quantities)

        # Track interval expanding history
        int_val = intervals[i]
        if not np.isnan(int_val):
            current_intervals.append(int_val)

        k = len(current_intervals)
        if k >= 1:
            arr = np.array(current_intervals, dtype=np.float64)
            med = float(np.median(arr))
            hist_median[i] = med
            m = np.mean(arr)
            hist_mean[i] = m
            hist_min[i] = np.min(arr)
            hist_max[i] = np.max(arr)
            mad = float(np.median(np.abs(arr - med)))
            hist_mad[i] = mad
            hist_norm_mad[i] = (mad / med) if med > 0 else np.nan
            r3 = float(np.median(arr[-3:]))
            recent3_med[i] = r3
            cadence_drift[i] = abs(r3 - med)
            pct_extreme[i] = float(np.mean((arr < 10.0) | (arr > 180.0)) * 100.0)
            pct_in_band[i] = float(np.mean((arr >= 15.0) & (arr <= 120.0)) * 100.0)
            if med > 0:
                pct_within_50[i] = float(np.mean(np.abs(arr - med) <= 0.50 * med) * 100.0)
                max_over_med[i] = float(arr.max() / med)
            if k >= 2:
                s = np.std(arr, ddof=1)
                hist_std[i] = s
                hist_cv[i] = (s / m) if m > 0 else 0.0

    df["historical_interval_median"] = hist_median
    df["historical_interval_mean"] = hist_mean
    df["historical_interval_std"] = hist_std
    df["historical_interval_min"] = hist_min
    df["historical_interval_max"] = hist_max
    df["historical_interval_cv"] = hist_cv
    df["historical_interval_mad"] = hist_mad
    df["historical_interval_norm_mad"] = hist_norm_mad
    df["recent3_interval_median"] = recent3_med
    df["cadence_drift"] = cadence_drift
    df["pct_extreme_lt10_or_gt180"] = pct_extreme
    df["pct_in_refill_band_15_120"] = pct_in_band
    df["pct_within_50pct_of_median"] = pct_within_50
    df["max_over_median"] = max_over_med
    df["avg_historical_quantity"] = avg_qty
    safe_avg_qty = np.where(avg_qty > 0, avg_qty, 1.0)
    df["quantity_vs_avg_ratio"] = np.where(avg_qty > 0, quantities / safe_avg_qty, 1.0)

    # 4. Pattern & Regularity Flags
    df["is_first_purchase"] = (df["purchase_count_so_far"] == 1).astype(int)
    df["has_multiple_prior_purchases"] = (df["purchase_count_so_far"] >= 3).astype(int)
    df["is_recurring_history"] = (
        (df["purchase_count_so_far"] >= 3)
        & (df["historical_interval_median"] >= 15.0)
        & (df["historical_interval_median"] <= 120.0)
    ).astype(int)

    # 5. Calendar Features
    df["purchase_month"] = df["invoice_date"].dt.month
    df["purchase_day_of_week"] = df["invoice_date"].dt.dayofweek
    df["purchase_day_of_month"] = df["invoice_date"].dt.day
    df["purchase_day_of_year"] = df["invoice_date"].dt.dayofyear
    df["purchase_quarter"] = df["invoice_date"].dt.quarter
    df["is_weekend"] = df["purchase_day_of_week"].isin([5, 6]).astype(int)

    # 6. Quality & Eligibility Flags
    df["has_next_purchase"] = df[TARGET_COLUMN].notna()
    df["is_positive_quantity"] = df["quantity"] > 0
    df["is_same_day_target"] = (df[TARGET_COLUMN] == 0)
    df["is_supervised_eligible"] = df["has_next_purchase"] & df["is_positive_quantity"]

    # 7. Temporal Partitioning Column
    # Dataset span: 2020-12-24 → 2026-08-31 (~5.8 years)
    # Hold out the final 4 months for validation + test.
    train_mask = df["invoice_date"] <= "2026-04-30"
    val_mask = (df["invoice_date"] >= "2026-05-01") & (df["invoice_date"] <= "2026-06-30")
    test_mask = (df["invoice_date"] >= "2026-07-01") & (df["invoice_date"] <= "2026-08-31")

    df["split_set"] = "unassigned"
    df.loc[train_mask, "split_set"] = "train"
    df.loc[val_mask, "split_set"] = "validation"
    df.loc[test_mask, "split_set"] = "test"

    return df


def split_dataset_temporally(
    feature_df: pd.DataFrame,
    supervised_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature dataset chronologically into Train, Validation, and Test sets.

    Temporal Boundaries (aligned to ~5.8-year dataset ending 2026-08-31):
    - Train:      invoice_date <= 2026-04-30 (history through Apr 2026)
    - Validation: 2026-05-01 <= invoice_date <= 2026-06-30 (2 months)
    - Test:       2026-07-01 <= invoice_date <= 2026-08-31 (2 months)

    Args:
        feature_df: DataFrame generated by build_feature_dataset.
        supervised_only: If True, filters to is_supervised_eligible rows.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    df = feature_df[feature_df["is_supervised_eligible"]].copy() if supervised_only else feature_df.copy()

    train_df = df[df["split_set"] == "train"].reset_index(drop=True)
    val_df = df[df["split_set"] == "validation"].reset_index(drop=True)
    test_df = df[df["split_set"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


def extract_feature_target_matrices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y, guaranteeing no target leakage.

    Args:
        df: Feature DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' missing from DataFrame.")

    y = df[TARGET_COLUMN].copy()

    # Features to select (numeric + categorical)
    cols_to_use = [c for c in FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL if c in df.columns]
    X = df[cols_to_use].copy()

    return X, y
