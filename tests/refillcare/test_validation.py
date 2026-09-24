"""Unit tests for RefillCare data validation module.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.data.validation import (
    validate_dataset,
    generate_validation_report,
)


def test_validation_passes_on_clean_data():
    """Verify validate_dataset returns PASS status on clean, consistent datasets."""
    clean_df = pd.DataFrame([
        {
            "customerId": "C1",
            "itemId": "M1",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-06-01"),
            "quantity": 10,
        },
        {
            "customerId": "C1",
            "itemId": "M1",
            "invoice_number": "INV_2",
            "invoice_date": pd.Timestamp("2025-07-01"),
            "quantity": 10,
        },
    ])

    history_df = clean_df.copy()
    history_df["days_since_previous_purchase"] = [np.nan, 30.0]

    report = validate_dataset(
        clean_df=clean_df,
        aggregated_df=clean_df,
        history_df=history_df,
    )

    assert report["status"] == "PASS"
    assert report["metrics"]["negative_intervals"] == 0
    assert report["metrics"]["clean_missing_customerId"] == 0


def test_validation_detects_negative_intervals():
    """Verify validate_dataset flags FAIL when negative purchase intervals exist."""
    history_df = pd.DataFrame([
        {
            "customerId": "C1",
            "itemId": "M1",
            "invoice_number": "INV_1",
            "invoice_date": pd.Timestamp("2025-07-01"),
            "days_since_previous_purchase": -15.0,  # Corrupted interval
        }
    ])

    report = validate_dataset(history_df=history_df)

    assert report["status"] == "FAIL"
    assert report["metrics"]["negative_intervals"] == 1
    assert any("negative purchase intervals" in issue for issue in report["issues_found"])


def test_generate_validation_report_output():
    """Verify human-readable report formatting."""
    report = {
        "status": "PASS",
        "metrics": {"total_records": 100},
        "issues_found": [],
    }
    formatted = generate_validation_report(report)
    assert "REFILLCARE DATA QUALITY VALIDATION: PASS" in formatted
    assert "total_records: 100" in formatted
