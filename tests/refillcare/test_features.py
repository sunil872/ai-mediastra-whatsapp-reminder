"""Unit tests for RefillCare feature engineering and data leakage prevention.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.features.engineering import (
    build_feature_dataset,
    split_dataset_temporally,
    extract_feature_target_matrices,
    TARGET_COLUMN,
)


@pytest.fixture
def synthetic_history():
    """Create a synthetic purchase history timeline with 3 distinct customers."""
    return pd.DataFrame([
        # Customer 1, Item A (4 purchases: Jan 1, Jan 31, Mar 2, Apr 1)
        # Intervals: [30, 30, 30]
        {
            "customerId": "CUST_1",
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "itemName": "Medicine Alpha",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 30,
            "freeQuantity": 0,
            "netAmount": 300.0,
            "gstAmount": 36.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9876543210",
            "days_since_previous_purchase": np.nan,
        },
        {
            "customerId": "CUST_1",
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "itemName": "Medicine Alpha",
            "invoice_number": "INV_2",
            "invoice_date": pd.Timestamp("2025-07-01"),  # +30d
            "quantity": 30,
            "freeQuantity": 0,
            "netAmount": 300.0,
            "gstAmount": 36.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9876543210",
            "days_since_previous_purchase": 30.0,
        },
        {
            "customerId": "CUST_1",
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "itemName": "Medicine Alpha",
            "invoice_number": "INV_3",
            "invoice_date": pd.Timestamp("2025-08-05"),  # +35d
            "quantity": 30,
            "freeQuantity": 0,
            "netAmount": 300.0,
            "gstAmount": 36.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9876543210",
            "days_since_previous_purchase": 35.0,
        },
        {
            "customerId": "CUST_1",
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "itemName": "Medicine Alpha",
            "invoice_number": "INV_4",
            "invoice_date": pd.Timestamp("2025-09-04"),  # +30d
            "quantity": 30,
            "freeQuantity": 0,
            "netAmount": 300.0,
            "gstAmount": 36.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9876543210",
            "days_since_previous_purchase": 30.0,
        },
        # Customer 2 sharing the SAME phone number (single purchase - cold start)
        {
            "customerId": "CUST_2",
            "itemId": "ITEM_A",
            "itemCode": "ITEM_A",
            "itemName": "Medicine Alpha",
            "invoice_number": "INV_5",
            "invoice_date": pd.Timestamp("2025-06-15"),
            "quantity": 10,
            "freeQuantity": 0,
            "netAmount": 100.0,
            "gstAmount": 12.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9876543210",  # Shared phone!
            "days_since_previous_purchase": np.nan,
        },
        # Customer 3 in Validation period (May–Jun 2026)
        {
            "customerId": "CUST_3",
            "itemId": "ITEM_B",
            "itemCode": "ITEM_B",
            "itemName": "Medicine Beta",
            "invoice_number": "INV_6",
            "invoice_date": pd.Timestamp("2026-05-10"),
            "quantity": 20,
            "freeQuantity": 0,
            "netAmount": 200.0,
            "gstAmount": 24.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9123456789",
            "days_since_previous_purchase": np.nan,
        },
        {
            "customerId": "CUST_3",
            "itemId": "ITEM_B",
            "itemCode": "ITEM_B",
            "itemName": "Medicine Beta",
            "invoice_number": "INV_7",
            "invoice_date": pd.Timestamp("2026-06-10"),  # +31d
            "quantity": 20,
            "freeQuantity": 0,
            "netAmount": 200.0,
            "gstAmount": 24.0,
            "rate": 10.0,
            "mrp": 12.0,
            "discountPercent": 0.0,
            "MOBILE_NO": "9123456789",
            "days_since_previous_purchase": 31.0,
        },
    ])


def test_target_is_next_purchase_interval(synthetic_history):
    """Verify target_days_until_next_purchase is date(i+1) - date(i)."""
    df = build_feature_dataset(synthetic_history)
    cust1 = df[df["customerId"] == "CUST_1"].reset_index(drop=True)

    # June 1 -> July 1: 30 days
    assert cust1.loc[0, TARGET_COLUMN] == 30.0
    # July 1 -> Aug 5: 35 days
    assert cust1.loc[1, TARGET_COLUMN] == 35.0
    # Aug 5 -> Sep 4: 30 days
    assert cust1.loc[2, TARGET_COLUMN] == 30.0


def test_final_and_single_purchase_have_no_supervised_target(synthetic_history):
    """Verify the final purchase in a history has NaN target and is not supervised eligible."""
    df = build_feature_dataset(synthetic_history)
    cust1 = df[df["customerId"] == "CUST_1"].reset_index(drop=True)
    cust2 = df[df["customerId"] == "CUST_2"].reset_index(drop=True)

    # Cust 1 final purchase (row 3)
    assert pd.isna(cust1.loc[3, TARGET_COLUMN])
    assert not cust1.loc[3, "has_next_purchase"]
    assert not cust1.loc[3, "is_supervised_eligible"]

    # Cust 2 single purchase
    assert pd.isna(cust2.loc[0, TARGET_COLUMN])
    assert not cust2.loc[0, "has_next_purchase"]
    assert not cust2.loc[0, "is_supervised_eligible"]


def test_leakage_safe_historical_interval_statistics(synthetic_history):
    """Verify historical interval statistics ONLY use intervals prior to or at the current purchase."""
    df = build_feature_dataset(synthetic_history)
    cust1 = df[df["customerId"] == "CUST_1"].reset_index(drop=True)

    # Event 1 (June 1): 0 prior intervals
    assert pd.isna(cust1.loc[0, "historical_interval_median"])
    assert pd.isna(cust1.loc[0, "historical_interval_mean"])
    assert cust1.loc[0, "purchase_count_so_far"] == 1

    # Event 2 (July 1): 1 prior interval [30]
    assert cust1.loc[1, "historical_interval_median"] == 30.0
    assert cust1.loc[1, "historical_interval_mean"] == 30.0
    assert cust1.loc[1, "historical_interval_min"] == 30.0
    assert cust1.loc[1, "historical_interval_max"] == 30.0
    assert pd.isna(cust1.loc[1, "historical_interval_std"])  # 1 interval -> std is NaN

    # Event 3 (Aug 5): 2 prior intervals [30, 35]
    assert cust1.loc[2, "historical_interval_median"] == 32.5
    assert cust1.loc[2, "historical_interval_mean"] == 32.5
    assert cust1.loc[2, "historical_interval_min"] == 30.0
    assert cust1.loc[2, "historical_interval_max"] == 35.0
    assert np.isclose(cust1.loc[2, "historical_interval_std"], np.std([30, 35], ddof=1))

    # Expanding average quantity must be persisted as a feature column
    assert "avg_historical_quantity" in cust1.columns
    assert cust1.loc[0, "avg_historical_quantity"] == cust1.loc[0, "quantity"]
    assert np.isclose(
        cust1.loc[2, "avg_historical_quantity"],
        cust1.loc[:2, "quantity"].mean(),
    )

def test_shared_phone_does_not_merge_histories(synthetic_history):
    """Verify Customer 1 and Customer 2 sharing phone 9876543210 have distinct feature histories."""
    df = build_feature_dataset(synthetic_history)
    cust2 = df[df["customerId"] == "CUST_2"].iloc[0]

    # Cust 2 bought on June 15, between Cust 1's June 1 and July 1 purchases
    # Cust 2 must still have purchase_count_so_far = 1 and NaN historical interval
    assert cust2["purchase_count_so_far"] == 1
    assert pd.isna(cust2["historical_interval_median"])
    assert cust2["is_first_purchase"] == 1


def test_temporal_split_ranges(synthetic_history):
    """Verify train, validation, and test temporal splitting with no chronological overlap."""
    df = build_feature_dataset(synthetic_history)
    train, val, test = split_dataset_temporally(df, supervised_only=False)

    # Check dates
    if len(train) > 0:
        assert train["invoice_date"].max() <= pd.Timestamp("2026-04-30")
    if len(val) > 0:
        assert val["invoice_date"].min() >= pd.Timestamp("2026-05-01")
        assert val["invoice_date"].max() <= pd.Timestamp("2026-06-30")
    if len(test) > 0:
        assert test["invoice_date"].min() >= pd.Timestamp("2026-07-01")
        assert test["invoice_date"].max() <= pd.Timestamp("2026-08-31")


def test_no_forbidden_columns_in_feature_matrix(synthetic_history):
    """Verify extracted feature matrix X contains no target or forward leakage columns."""
    df = build_feature_dataset(synthetic_history)
    supervised_df = df[df["is_supervised_eligible"]]
    X, y = extract_feature_target_matrices(supervised_df)

    assert TARGET_COLUMN not in X.columns
    assert "next_purchase_date" not in X.columns
    assert "days_until_next_purchase" not in X.columns
    # Ensure no negative target intervals
    assert (y >= 0).all()
