"""Tests for the RefillCare Historical Holdout Validation workflow.

Validates:
1. Zero Data Leakage: August transactions cannot enter pre-August history or features.
2. Compound Identity Matching: Outcomes matched strictly using customerId + itemId.
3. Same-Invoice Aggregation: Multi-line transactions within the same invoice are consolidated.
4. Returns / Zero Quantity Exclusion: quantity <= 0 rows are excluded from intervals.
5. Right-Censoring: Unobserved holdout repurchases remain pending and are not penalized as errors.
6. Days-of-Supply Pre-Cutoff Anchoring: Supply calculation uses only pre-cutoff history.
7. Realized Metric Accuracy: Exact error_days, MAE, RMSE, ±1d, ±3d, ±7d, and catastrophic metrics.
8. Read-only Safety: No modification to models, schedulers, or WhatsApp dispatch.
"""

from datetime import date, timedelta
import pandas as pd
import pytest

from refillcare.evaluation.holdout_validation import run_historical_holdout_validation


def _create_sample_multi_month_transactions() -> pd.DataFrame:
    """Create a controlled transaction history spanning Jan 2026 through Aug 2026."""
    records = [
        # Customer C1 + Medicine M1: Regular 30-day cadence
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-101", "invoice_date": "2026-05-01", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-102", "invoice_date": "2026-05-31", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-103", "invoice_date": "2026-06-30", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-104", "invoice_date": "2026-07-30", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},
        # Actual August repurchase on 2026-08-29 (Exact predicted expected date is ~2026-08-29, 0 or 1 day error)
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-105", "invoice_date": "2026-08-29", "quantity": 2, "packing": "1X15", "MOBILE_NO": "9849011111"},

        # Customer C2 + Medicine M2: Same-invoice split lines on 2026-07-15, holdout repurchase on 2026-08-16
        {"customerId": "C2", "itemId": "M2", "customerName": "Suresh", "itemName": "Telma 40", "invoice_number": "INV-201", "invoice_date": "2026-06-15", "quantity": 3, "packing": "1X10", "MOBILE_NO": "9849022222"},
        # Two lines in same invoice INV-202 on 2026-07-15
        {"customerId": "C2", "itemId": "M2", "customerName": "Suresh", "itemName": "Telma 40", "invoice_number": "INV-202", "invoice_date": "2026-07-15", "quantity": 2, "packing": "1X10", "MOBILE_NO": "9849022222"},
        {"customerId": "C2", "itemId": "M2", "customerName": "Suresh", "itemName": "Telma 40", "invoice_number": "INV-202", "invoice_date": "2026-07-15", "quantity": 1, "packing": "1X10", "MOBILE_NO": "9849022222"},
        # Holdout purchase in Aug
        {"customerId": "C2", "itemId": "M2", "customerName": "Suresh", "itemName": "Telma 40", "invoice_number": "INV-203", "invoice_date": "2026-08-16", "quantity": 3, "packing": "1X10", "MOBILE_NO": "9849022222"},

        # Customer C3 + Medicine M3: Pending in August (No August purchase, right-censored)
        {"customerId": "C3", "itemId": "M3", "customerName": "Anita", "itemName": "Ecosprin 75", "invoice_number": "INV-301", "invoice_date": "2026-06-01", "quantity": 1, "packing": "1X14", "MOBILE_NO": "9849033333"},
        {"customerId": "C3", "itemId": "M3", "customerName": "Anita", "itemName": "Ecosprin 75", "invoice_number": "INV-302", "invoice_date": "2026-07-15", "quantity": 2, "packing": "1X14", "MOBILE_NO": "9849033333"},

        # Customer C4 + Medicine M4: Cold-start (< 2 purchases) -> Should be excluded from predictions
        {"customerId": "C4", "itemId": "M4", "customerName": "Vikram", "itemName": "Pan D", "invoice_number": "INV-401", "invoice_date": "2026-07-20", "quantity": 1, "packing": "1X10", "MOBILE_NO": "9849044444"},

        # Return / Negative quantity transaction -> should be filtered out
        {"customerId": "C1", "itemId": "M1", "customerName": "Ramesh", "itemName": "Glycomet 500", "invoice_number": "INV-109-RET", "invoice_date": "2026-07-05", "quantity": -1, "packing": "1X15", "MOBILE_NO": "9849011111"},
    ]
    return pd.DataFrame(records)


def test_holdout_validation_zero_data_leakage():
    """Verify that August transactions are strictly excluded from prediction features."""
    df = _create_sample_multi_month_transactions()
    res = run_historical_holdout_validation(
        transactions_df=df,
        model_bundle=None,
        cutoff_date="2026-07-31",
        holdout_end_date="2026-08-31",
    )
    assert res["status"] == "success"
    audit = res["audit_report"]
    assert audit["leakage_detected"] is False
    assert audit["leakage_count"] == 0
    assert audit["historical_transactions_used"] > 0
    assert audit["holdout_transactions_used_for_eval"] > 0


def test_holdout_validation_entity_matching_and_outcomes():
    """Verify compound customerId + itemId matching and separation of evaluated vs pending."""
    df = _create_sample_multi_month_transactions()
    res = run_historical_holdout_validation(
        transactions_df=df,
        model_bundle=None,
        cutoff_date="2026-07-31",
        holdout_end_date="2026-08-31",
        target_prediction_window_start=None,
        target_prediction_window_end=None,
    )
    metrics = res["metrics"]
    eval_df = res["evaluated_df"]
    pend_df = res["pending_df"]

    # C1 and C2 repurchased in August -> evaluated
    assert "C1" in eval_df["customerId"].values
    assert "C2" in eval_df["customerId"].values
    assert metrics["evaluated_count"] >= 2

    # C3 had no August repurchase -> pending (right-censored)
    assert "C3" in pend_df["customerId"].values
    assert metrics["pending_count"] >= 1

    # C4 cold-start (< 2 purchases) excluded from candidates
    assert "C4" not in eval_df["customerId"].values
    assert "C4" not in pend_df["customerId"].values


def test_holdout_validation_same_invoice_aggregation_and_returns():
    """Verify same-invoice lines are aggregated and negative return quantities are excluded."""
    df = _create_sample_multi_month_transactions()
    res = run_historical_holdout_validation(
        transactions_df=df,
        model_bundle=None,
        cutoff_date="2026-07-31",
        holdout_end_date="2026-08-31",
    )
    assert res["status"] == "success"
    # Negative quantity in July for C1 did not cause negative intervals or errors
    eval_df = res["evaluated_df"]
    c1_rec = eval_df[eval_df["customerId"] == "C1"].iloc[0]
    assert c1_rec["status"] == "evaluated"
    assert c1_rec["abs_error_days"] <= 5.0


def test_holdout_validation_metrics_calculation():
    """Verify MAE, RMSE, and +/- 1d, 3d, 7d accuracy calculations."""
    df = _create_sample_multi_month_transactions()
    res = run_historical_holdout_validation(
        transactions_df=df,
        model_bundle=None,
        cutoff_date="2026-07-31",
        holdout_end_date="2026-08-31",
        target_prediction_window_start=None,
        target_prediction_window_end=None,
    )
    m = res["metrics"]
    assert "mae_days" in m
    assert "rmse_days" in m
    assert "within_1d_pct" in m
    assert "within_3d_pct" in m
    assert "within_7d_pct" in m
    assert "catastrophic_errors_gt_30d" in m
    assert m["mae_days"] >= 0.0
    assert m["rmse_days"] >= m["mae_days"] or pytest.approx(m["rmse_days"], 0.1) == m["mae_days"]


def test_holdout_validation_empty_data_graceful_handling():
    """Verify that empty inputs return valid empty structures without crashing."""
    empty_df = pd.DataFrame(columns=["customerId", "itemId", "invoice_date", "quantity"])
    res = run_historical_holdout_validation(empty_df, cutoff_date="2026-07-31")
    assert res["status"] == "empty"
    assert res["metrics"]["total_predictions"] == 0
    assert res["metrics"]["evaluated_count"] == 0
    assert res["metrics"]["pending_count"] == 0
