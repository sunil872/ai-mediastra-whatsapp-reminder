"""Purchase history creation and interval calculation module for RefillCare.

Builds chronological customer + medicine purchase timelines and computes
backward-looking purchase intervals without forward data leakage.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


HISTORY_GROUP_KEY = ["customerId", "itemId"]
SORT_ORDER = ["customerId", "itemId", "invoice_date", "invoice_number"]


def create_purchase_history(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build chronological purchase history and calculate backward-looking purchase intervals.

    Rules & Business Logic:
    1. Identity: customerId + itemId is the unique patient-medication history timeline.
       MOBILE_NO is NOT used as identity, keeping histories for shared phones strictly isolated.
    2. Chronological Ordering: Sorted by (customerId, itemId, invoice_date, invoice_number).
    3. Interval Calculation:
       - previous_purchase_date = immediate prior purchase date for this customer + item.
       - days_since_previous_purchase = (current invoice_date - previous_purchase_date).
       - For the first purchase in a history: previous_purchase_date is NaT, interval is NaN.
    4. Same-Day Purchases:
       - Lines within the same invoice were already aggregated in step 2.
       - Different invoices on the same date for the same (customerId, itemId) result in interval = 0 days.
    5. Data Leakage Prevention:
       - No forward-looking features (e.g., next_purchase_date, days_until_next_purchase) are added.
       - Only backward-looking history (purchase_seq, previous_purchase_date, days_since_previous_purchase)
         is computed.

    Args:
        events_df: Aggregated purchase events DataFrame.

    Returns:
        pd.DataFrame: Chronologically ordered purchase history with interval columns.
    """
    if events_df.empty:
        return events_df.copy()

    # Check required columns
    required = ["customerId", "itemId", "invoice_date", "invoice_number"]
    for col in required:
        if col not in events_df.columns:
            raise KeyError(f"Required column '{col}' missing from purchase events DataFrame.")

    # Copy and ensure invoice_date is datetime
    df = events_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], dayfirst=True)

    # Sort strictly chronologically
    df = df.sort_values(by=SORT_ORDER, ascending=True).reset_index(drop=True)

    # Group by customerId + itemId
    grouped = df.groupby(HISTORY_GROUP_KEY, sort=False)

    # 1. Sequential purchase number for this customer + item (1-indexed)
    df["purchase_seq"] = grouped.cumcount() + 1

    # 2. Total lifetime purchases in dataset for this customer + item
    df["total_purchases"] = grouped["invoice_date"].transform("count")

    # 3. First and latest purchase dates in dataset
    df["first_purchase_date"] = grouped["invoice_date"].transform("min")
    df["latest_purchase_date"] = grouped["invoice_date"].transform("max")

    # 4. Previous purchase date (strictly backward-looking)
    df["previous_purchase_date"] = grouped["invoice_date"].shift(1)

    # 5. Days since previous purchase
    # Note: NaT - NaT yields NaT, dt.days yields NaN
    interval_delta = df["invoice_date"] - df["previous_purchase_date"]
    df["days_since_previous_purchase"] = interval_delta.dt.days

    return df


def compute_interval_statistics(history_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics on backward-looking purchase intervals.

    Args:
        history_df: Purchase history DataFrame with 'days_since_previous_purchase'.

    Returns:
        Dict[str, Any]: Descriptive statistics of refill intervals.
    """
    if history_df.empty or "days_since_previous_purchase" not in history_df.columns:
        return {
            "total_intervals": 0,
            "median_days": None,
            "mean_days": None,
            "std_days": None,
            "min_days": None,
            "max_days": None,
            "recurring_15_120_pct": None,
        }

    # Extract non-null intervals (excluding first purchases)
    valid_intervals = history_df["days_since_previous_purchase"].dropna()

    if len(valid_intervals) == 0:
        return {
            "total_intervals": 0,
            "median_days": None,
            "mean_days": None,
            "std_days": None,
            "min_days": None,
            "max_days": None,
            "recurring_15_120_pct": None,
        }

    recurring_count = ((valid_intervals >= 15) & (valid_intervals <= 120)).sum()
    recurring_pct = round((recurring_count / len(valid_intervals)) * 100.0, 2)

    return {
        "total_intervals": int(len(valid_intervals)),
        "median_days": float(valid_intervals.median()),
        "mean_days": round(float(valid_intervals.mean()), 2),
        "std_days": round(float(valid_intervals.std()), 2) if len(valid_intervals) > 1 else 0.0,
        "min_days": int(valid_intervals.min()),
        "max_days": int(valid_intervals.max()),
        "recurring_15_120_pct": recurring_pct,
    }
