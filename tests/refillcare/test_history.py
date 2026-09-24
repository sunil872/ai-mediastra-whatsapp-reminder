"""Unit tests for purchase history creation and interval calculation.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.data.history import (
    create_purchase_history,
    compute_interval_statistics,
)


def test_customer_item_identity_and_shared_phones():
    """Verify that different customerId sharing the same MOBILE_NO remain separate histories."""
    events = pd.DataFrame([
        {
            "customerId": "PATIENT_A",
            "itemId": "MED_101",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 30,
            "MOBILE_NO": "9876543210",  # Shared phone
        },
        {
            "customerId": "PATIENT_B",
            "itemId": "MED_101",
            "invoice_number": "INV_2",
            "invoice_date": pd.Timestamp("2025-06-15"),
            "quantity": 30,
            "MOBILE_NO": "9876543210",  # Shared phone
        },
        {
            "customerId": "PATIENT_A",
            "itemId": "MED_101",
            "invoice_number": "INV_3",
            "invoice_date": pd.Timestamp("2025-07-01"),
            "quantity": 30,
            "MOBILE_NO": "9876543210",
        },
    ])

    history = create_purchase_history(events)

    # Patient A should have 2 purchases: June 1 and July 1 (interval: 30 days)
    patient_a = history[history["customerId"] == "PATIENT_A"].reset_index(drop=True)
    assert len(patient_a) == 2
    assert patient_a.loc[0, "purchase_seq"] == 1
    assert pd.isna(patient_a.loc[0, "days_since_previous_purchase"])
    assert patient_a.loc[1, "purchase_seq"] == 2
    assert patient_a.loc[1, "days_since_previous_purchase"] == 30

    # Patient B should have 1 purchase: June 15 (interval: NaN, not affected by Patient A)
    patient_b = history[history["customerId"] == "PATIENT_B"].reset_index(drop=True)
    assert len(patient_b) == 1
    assert patient_b.loc[0, "purchase_seq"] == 1
    assert pd.isna(patient_b.loc[0, "days_since_previous_purchase"])


def test_first_purchase_has_no_previous_interval():
    """Verify that the first chronological purchase in any history has NaN interval and NaT previous date."""
    events = pd.DataFrame([
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 10,
        },
        {
            "customerId": "CUST_1",
            "itemId": "MED_B",
            "invoice_number": "INV_2",
            "invoice_date": pd.Timestamp("2025-06-10"),
            "quantity": 20,
        },
    ])

    history = create_purchase_history(events)

    # Both are first purchases for their respective items
    for _, row in history.iterrows():
        assert row["purchase_seq"] == 1
        assert pd.isna(row["previous_purchase_date"])
        assert pd.isna(row["days_since_previous_purchase"])


def test_purchase_interval_calculation_and_ordering():
    """Verify accurate interval computation and strictly chronological ordering."""
    # Provide events out of order to ensure sorting works
    events = pd.DataFrame([
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_3",
            "invoice_date": pd.Timestamp("2025-08-30"),
            "quantity": 30,
        },
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 30,
        },
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_2",
            "invoice_date": pd.Timestamp("2025-07-01"),
            "quantity": 30,
        },
    ])

    history = create_purchase_history(events)

    assert list(history["invoice_number"]) == ["INV_1", "INV_2", "INV_3"]
    assert list(history["purchase_seq"]) == [1, 2, 3]

    # June 1 -> July 1 = 30 days
    assert history.loc[1, "days_since_previous_purchase"] == 30
    # July 1 -> Aug 30 = 60 days
    assert history.loc[2, "days_since_previous_purchase"] == 60


def test_same_day_purchases():
    """Verify that multiple invoices on the same date result in interval = 0 days."""
    events = pd.DataFrame([
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_10",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 10,
        },
        {
            "customerId": "CUST_1",
            "itemId": "MED_A",
            "invoice_number": "INV_11",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 5,
        },
    ])

    history = create_purchase_history(events)
    assert len(history) == 2
    assert history.loc[0, "purchase_seq"] == 1
    assert pd.isna(history.loc[0, "days_since_previous_purchase"])

    assert history.loc[1, "purchase_seq"] == 2
    assert history.loc[1, "days_since_previous_purchase"] == 0


def test_compute_interval_statistics():
    """Verify interval statistics calculation."""
    events = pd.DataFrame([
        {"customerId": "C1", "itemId": "M1", "invoice_number": "1", "invoice_date": pd.Timestamp("2025-01-01")},
        {"customerId": "C1", "itemId": "M1", "invoice_number": "2", "invoice_date": pd.Timestamp("2025-01-31")},  # 30d
        {"customerId": "C1", "itemId": "M1", "invoice_number": "3", "invoice_date": pd.Timestamp("2025-03-02")},  # 30d
        {"customerId": "C1", "itemId": "M1", "invoice_number": "4", "invoice_date": pd.Timestamp("2025-04-01")},  # 30d
    ])
    history = create_purchase_history(events)
    stats = compute_interval_statistics(history)

    assert stats["total_intervals"] == 3
    assert stats["median_days"] == 30.0
    assert stats["mean_days"] == 30.0
    assert stats["min_days"] == 30
    assert stats["max_days"] == 30
    assert stats["recurring_15_120_pct"] == 100.0
