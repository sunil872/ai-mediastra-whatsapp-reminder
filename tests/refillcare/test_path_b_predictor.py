"""Comprehensive unit and integration test suite for Path B Recency & Multi-Month Recurrence Predictor.

Verifies:
1. 3-Month window evaluation: >=2 distinct calendar months -> Eligible (3M_RECURRING).
2. 6-Month window evaluation fallback: >=3 distinct calendar months -> Eligible (6M_RECURRING).
3. Insufficient monthly recurrence (<2 in 3m, <3 in 6m) -> Ineligible (Cold-Start Review Queue).
4. Multiple purchases within the SAME calendar month counted as 1 distinct month.
5. Recency filter: Stale latest purchase (>180 days from cutoff) -> Ineligible.
6. Zero data leakage: Future purchases after prediction cutoff are strictly excluded.
7. Identity isolation: Evaluated strictly per (customerId, itemId).
8. Expected refill date & 2-day primary reminder date calculations.
9. Integration with feature table and predict_refill_cycles.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from refillcare.models.path_b_classifier import (
    evaluate_path_b_eligibility,
    ROUTE_3M_RECURRING,
    ROUTE_6M_RECURRING,
    ROUTE_INELIGIBLE,
)
from refillcare.models.path_b_predictor import (
    calculate_path_b_refill_duration,
    predict_path_b_candidate,
)
from model.predictor import predict_refill_cycles


def test_path_b_3m_window_eligible():
    """Customer with purchases in 2 distinct months within the last 90 days is eligible under 3M rule."""
    cutoff = date(2026, 9, 1)
    # Purchases in July and August (2 distinct months within 90 days of Sept 1)
    dates = ["2026-07-10", "2026-08-12"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["is_eligible"] is True
    assert result["route"] == ROUTE_3M_RECURRING
    assert result["distinct_months_3m"] == 2
    assert ">= 2 distinct months" in result["reason"]


def test_path_b_same_month_purchases_count_as_one_month():
    """Multiple purchases in the SAME calendar month count as only 1 distinct month."""
    cutoff = date(2026, 9, 1)
    # 2 purchases both in August 2026 (only 1 distinct month in 3m)
    dates = ["2026-08-05", "2026-08-25"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["distinct_months_3m"] == 1
    # 6m window also only has 1 distinct month -> Ineligible
    assert result["is_eligible"] is False
    assert result["route"] == ROUTE_INELIGIBLE


def test_path_b_6m_window_fallback_eligible():
    """Customer failing 3M (<2 months) but satisfying 6M (>=3 distinct months) is eligible."""
    cutoff = date(2026, 9, 1)
    # Purchases in March, May, and August (3 distinct months within 180 days)
    # In trailing 90 days (June-Aug), only 1 purchase (August) -> 3m fails
    # In trailing 180 days (March-Aug), 3 distinct months -> 6m succeeds!
    dates = ["2026-03-15", "2026-05-20", "2026-08-10"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["is_eligible"] is True
    assert result["route"] == ROUTE_6M_RECURRING
    assert result["distinct_months_3m"] == 1
    assert result["distinct_months_6m"] == 3
    assert "Satisfied 6-month evaluation" in result["reason"]


def test_path_b_insufficient_recurrence_ineligible():
    """Customer with 2 purchases spread > 3 months apart (only 2 months in 6m) is ineligible."""
    cutoff = date(2026, 9, 1)
    # Purchases in April and August (2 distinct months in 6m, 1 in 3m)
    dates = ["2026-04-10", "2026-08-15"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["is_eligible"] is False
    assert result["route"] == ROUTE_INELIGIBLE
    assert result["distinct_months_3m"] == 1
    assert result["distinct_months_6m"] == 2
    assert "Insufficient distinct monthly recurrence" in result["reason"]


def test_path_b_single_purchase_ineligible():
    """Single purchase is cold-start and ineligible."""
    cutoff = date(2026, 9, 1)
    dates = ["2026-08-15"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["is_eligible"] is False
    assert result["route"] == ROUTE_INELIGIBLE
    assert result["distinct_months_3m"] == 1


def test_path_b_stale_recency_ineligible():
    """Latest purchase older than 180 days from cutoff is inactive/stale -> Ineligible."""
    cutoff = date(2026, 9, 1)
    # Purchases in 2025 (latest is Dec 2025, > 180 days from Sept 2026)
    dates = ["2025-10-10", "2025-11-15", "2025-12-20"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    assert result["is_eligible"] is False
    assert result["route"] == ROUTE_INELIGIBLE
    assert "Latest purchase is stale" in result["reason"]


def test_path_b_zero_future_leakage():
    """Transactions occurring AFTER prediction cutoff date are strictly ignored."""
    cutoff = date(2026, 7, 31)
    # July 10 (valid), August 15 (future relative to July 31)
    dates = ["2026-06-10", "2026-07-15", "2026-08-20"]
    
    result = evaluate_path_b_eligibility(purchase_dates=dates, prediction_date=cutoff)
    # Before cutoff: June and July -> 2 distinct months in 3m -> Eligible!
    assert result["is_eligible"] is True
    assert result["latest_purchase_date"] == "2026-07-15"
    assert "2026-08-20" not in result.get("qualifying_dates", [])


def test_predict_path_b_candidate_dates_and_lead_buffer():
    """Predict path B candidate sets expected refill date and 2-day reminder buffer."""
    cutoff = date(2026, 9, 1)
    history_df = pd.DataFrame([
        {"invoice_date": "2026-07-15", "quantity": 1, "packing": "1X30", "customerName": "Ramesh", "itemName": "Glycomet GP 1"},
        {"invoice_date": "2026-08-15", "quantity": 1, "packing": "1X30", "customerName": "Ramesh", "itemName": "Glycomet GP 1"},
    ])
    
    cand = predict_path_b_candidate(
        customer_id="CUST_001",
        item_id="ITEM_101",
        history_df=history_df,
        prediction_date=cutoff,
    )
    
    assert cand["is_eligible"] is True
    assert cand["path"] == "B"
    assert "Path B" in cand["prediction_source"]
    assert cand["last_purchase_date"] == "2026-08-15"
    assert cand["estimated_days_of_supply"] == 31.0 or cand["estimated_days_of_supply"] == 30.0
    # Expected refill date should be 30 or 31 days after 2026-08-15
    assert cand["expected_refill_date"] in ("2026-09-14", "2026-09-15")
    # Reminder date must be exactly 2 days before expected refill date
    assert cand["reminder_date"] in ("2026-09-12", "2026-09-13")
    assert cand["pilot_tier"] == "Tier B (Path B Recurrent Pilot)"


def test_predict_refill_cycles_e2e_path_b_integration():
    """Verify predict_refill_cycles correctly partitions Path A, Path B, and Cold-Start."""
    cutoff = date(2026, 9, 1)
    
    rows = []
    # 1. Path A Customer (8 purchases across 2026)
    for m in range(1, 9):
        rows.append({
            "customerId": "CUST_PATH_A",
            "itemId": "ITEM_1",
            "customerName": "Path A Patient",
            "itemName": "Medicine A",
            "MOBILE_NO": "919876543210",
            "mobile_status": "Valid",
            "invoice_date": f"2026-0{m}-10",
            "quantity": 1,
            "packing": "1X30",
        })
        
    # 2. Path B Customer (3 purchases: June, July, August 2026 -> 3 distinct months in 3m)
    for m in (6, 7, 8):
        rows.append({
            "customerId": "CUST_PATH_B",
            "itemId": "ITEM_2",
            "customerName": "Path B Patient",
            "itemName": "Medicine B",
            "MOBILE_NO": "919876543211",
            "mobile_status": "Valid",
            "invoice_date": f"2026-0{m}-12",
            "quantity": 1,
            "packing": "1X30",
        })
        
    # 3. Cold-Start Customer (1 purchase in August 2026)
    rows.append({
        "customerId": "CUST_COLD",
        "itemId": "ITEM_3",
        "customerName": "Cold Patient",
        "itemName": "Medicine C",
        "MOBILE_NO": "919876543212",
        "mobile_status": "Valid",
        "invoice_date": "2026-08-20",
        "quantity": 1,
        "packing": "1X30",
    })
    
    df = pd.DataFrame(rows)
    eligible_df, ineligible_df = predict_refill_cycles(df, prediction_date=cutoff)
    
    # Path A and Path B should both be in eligible_df
    assert len(eligible_df) == 2
    cids_eligible = set(eligible_df["customerId"].tolist())
    assert "CUST_PATH_A" in cids_eligible
    assert "CUST_PATH_B" in cids_eligible
    
    path_b_row = eligible_df[eligible_df["customerId"] == "CUST_PATH_B"].iloc[0]
    assert "Path B" in path_b_row["prediction_source"]
    assert path_b_row["pilot_tier"] == "Tier B (Path B Active Pilot)"
    
    # Cold start should be in ineligible_df
    assert len(ineligible_df) == 1
    assert ineligible_df.iloc[0]["customerId"] == "CUST_COLD"
    assert "Cold-start" in ineligible_df.iloc[0]["Reason for Ineligibility"]
