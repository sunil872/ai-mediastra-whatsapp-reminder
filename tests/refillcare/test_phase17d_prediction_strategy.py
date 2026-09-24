"""Regression tests for Phase 17D approved hybrid prediction strategy.

Covers: low-history, regular, irregular, shared phones, cycle reset,
expected refill date, and prediction rejection. No WhatsApp sends.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from refillcare.models.hybrid_strategy import (
    STRATEGY_NAME,
    MIN_PURCHASES,
    evaluate_hybrid_eligibility,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
)
from refillcare.models.prediction import predict_refill_date, generate_batch_predictions
from refillcare.models.training import (
    build_model_pipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from reminder.scheduler import (
    RefillReminderScheduler,
    calculate_expected_refill_date,
    evaluate_refill_eligibility,
)


def _base_features(**overrides):
    row = {
        "customerId": "CUST_A",
        "itemId": "ITEM_1",
        "itemName": "Medicine Alpha",
        "MOBILE_NO": "+919111111111",
        "invoice_date": pd.Timestamp("2026-07-15"),
        "purchase_count_so_far": 8,
        "days_since_first_purchase": 210,
        "days_since_previous_purchase": 30,
        "historical_interval_median": 30.0,
        "historical_interval_mean": 30.0,
        "historical_interval_std": 3.0,
        "historical_interval_min": 25.0,
        "historical_interval_max": 35.0,
        "historical_interval_cv": 0.1,
        "historical_interval_mad": 2.0,
        "historical_interval_norm_mad": 0.10,
        "recent3_interval_median": 30.0,
        "cadence_drift": 2.0,
        "quantity": 30,
        "freeQuantity": 0,
        "avg_historical_quantity": 30.0,
        "quantity_vs_avg_ratio": 1.0,
        "purchase_month": 7,
        "purchase_day_of_week": 2,
        "purchase_day_of_month": 15,
        "purchase_day_of_year": 196,
        "purchase_quarter": 3,
        "is_weekend": 0,
        "is_first_purchase": 0,
        "has_multiple_prior_purchases": 1,
        "is_recurring_history": 1,
        "salt_category": "TABLETS",
        "salt_itemcat": "ORAL",
    }
    row.update(overrides)
    return row


@pytest.fixture
def model_bundle():
    np.random.seed(0)
    n = 40
    rows = []
    for i in range(n):
        r = _base_features(
            customerId=f"C{i % 4}",
            itemId=f"I{i % 2}",
            purchase_count_so_far=8,
            historical_interval_median=28 + (i % 5),
        )
        r[TARGET_COL] = 30.0
        rows.append(r)
    train = pd.DataFrame(rows)
    for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if c not in train.columns:
            train[c] = 0
    pipe = build_model_pipeline(RandomForestRegressor(n_estimators=8, max_depth=3, random_state=0))
    pipe.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET_COL])
    return {
        "pipeline": pipe,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "metadata": {"strategy": STRATEGY_NAME},
    }


def test_low_history_customers_rejected(model_bundle):
    for p in (1, 2, 3, 4, 5):
        row = _base_features(purchase_count_so_far=p)
        decision = evaluate_hybrid_eligibility(row)
        assert decision["is_eligible"] is False
        assert decision["route"] == ROUTE_REJECTED
        res = predict_refill_date(model_bundle, row)
        assert res["prediction_status"] == "rejected"
        assert res["refill_confidence"] == CONF_REJECTED
        assert res["predicted_days_until_refill"] is None
        assert res["expected_refill_date"] is None
        assert str(MIN_PURCHASES) in res["rejection_reason"] or "purchase_count" in res["rejection_reason"].lower()
        if p <= 3:
            assert res["pack_default_days"] == 30.0


def test_regular_customers_use_personal_median(model_bundle):
    row = _base_features(
        purchase_count_so_far=10,
        historical_interval_median=28.0,
        historical_interval_norm_mad=0.12,
        cadence_drift=3.0,
    )
    decision = evaluate_hybrid_eligibility(row)
    assert decision["is_eligible"] is True
    assert decision["is_core_regular"] is True
    assert decision["route"] == ROUTE_CORE
    res = predict_refill_date(model_bundle, row)
    assert res["prediction_status"] == "eligible"
    assert res["prediction_route"] == ROUTE_CORE
    assert res["refill_confidence"] == CONF_HIGH
    assert res["predicted_days_until_refill"] == 28.0
    assert res["customerId"] == "CUST_A"
    assert res["itemId"] == "ITEM_1"


def test_irregular_customers_rejected_or_secondary(model_bundle):
    irregular_reject = _base_features(
        purchase_count_so_far=9,
        historical_interval_norm_mad=0.80,
        cadence_drift=20.0,
    )
    d1 = evaluate_hybrid_eligibility(irregular_reject)
    assert d1["is_eligible"] is False
    r1 = predict_refill_date(model_bundle, irregular_reject)
    assert r1["prediction_status"] == "rejected"
    assert "Irregular" in r1["rejection_reason"] or "NormMAD" in r1["rejection_reason"]

    irregular_secondary = _base_features(
        purchase_count_so_far=9,
        historical_interval_median=32.0,
        historical_interval_norm_mad=0.40,
        cadence_drift=8.0,
    )
    d2 = evaluate_hybrid_eligibility(irregular_secondary)
    assert d2["is_eligible"] is True
    assert d2["route"] == ROUTE_SECONDARY
    r2 = predict_refill_date(model_bundle, irregular_secondary)
    assert r2["prediction_status"] == "eligible"
    assert r2["prediction_route"] == ROUTE_SECONDARY
    assert r2["refill_confidence"] == CONF_MEDIUM
    assert r2["predicted_days_until_refill"] >= 1.0


def test_shared_phone_numbers_keep_customer_item_identity(model_bundle):
    shared = "+919988776655"
    alice = _base_features(
        customerId="ALICE",
        itemId="MED_A",
        MOBILE_NO=shared,
        historical_interval_median=30.0,
        historical_interval_norm_mad=0.1,
        cadence_drift=1.0,
    )
    bob = _base_features(
        customerId="BOB",
        itemId="MED_B",
        MOBILE_NO=shared,
        historical_interval_median=45.0,
        historical_interval_norm_mad=0.1,
        cadence_drift=2.0,
    )
    pred_a = predict_refill_date(model_bundle, alice)
    pred_b = predict_refill_date(model_bundle, bob)
    assert pred_a["customerId"] == "ALICE"
    assert pred_b["customerId"] == "BOB"
    assert pred_a["itemId"] == "MED_A"
    assert pred_b["itemId"] == "MED_B"
    assert pred_a["predicted_days_until_refill"] == 30.0
    assert pred_b["predicted_days_until_refill"] == 45.0

    batch = generate_batch_predictions(model_bundle, pd.DataFrame([alice, bob]))
    assert list(batch["customerId"]) == ["ALICE", "BOB"]
    assert list(batch["predicted_days_until_refill"]) == [30.0, 45.0]


def test_new_purchase_cycle_reset_uses_identity(model_bundle):
    row = _base_features(
        customerId="CUST_RESET",
        itemId="ITEM_RESET",
        historical_interval_median=30.0,
        historical_interval_norm_mad=0.1,
        cadence_drift=1.0,
        invoice_date=pd.Timestamp("2026-06-01"),
    )
    pred = predict_refill_date(model_bundle, row)
    assert pred["prediction_status"] == "eligible"

    sched = RefillReminderScheduler()
    cycle1 = {
        **row,
        "predicted_days_until_refill": pred["predicted_days_until_refill"],
        "prediction_status": pred["prediction_status"],
        "refill_confidence": pred["refill_confidence"],
        "invoice_date": "2026-06-01",
    }
    res1 = sched.schedule_refill_cycle(cycle1)
    assert res1["scheduled"] is True

    cycle2 = dict(cycle1)
    cycle2["invoice_date"] = "2026-07-01"
    res2 = sched.schedule_refill_cycle(cycle2)
    assert res2["scheduled"] is True
    assert res2["cancelled_prior_reminders"] == 6
    all_rems = sched.get_reminders_for_customer("CUST_RESET", "ITEM_RESET")
    assert all(r.customerId == "CUST_RESET" and r.itemId == "ITEM_RESET" for r in all_rems)
    new_cycle = [r for r in all_rems if r.status == "scheduled"]
    assert any(r.latest_purchase_date == "2026-07-01" for r in new_cycle)


def test_expected_refill_date_calculation(model_bundle):
    row = _base_features(
        invoice_date=pd.Timestamp("2026-07-01"),
        historical_interval_median=30.0,
        historical_interval_norm_mad=0.1,
        cadence_drift=1.0,
    )
    pred = predict_refill_date(model_bundle, row)
    assert pred["expected_refill_date"] == "2026-07-31"
    calc = calculate_expected_refill_date("2026-07-01", pred["predicted_days_until_refill"])
    assert calc["is_valid"] is True
    assert calc["expected_refill_date"] == date(2026, 7, 31)


def test_prediction_rejection_blocks_scheduler(model_bundle):
    row = _base_features(purchase_count_so_far=2)
    pred = predict_refill_date(model_bundle, row)
    assert pred["prediction_status"] == "rejected"

    elig = evaluate_refill_eligibility({
        **row,
        "predicted_days_until_refill": pred["predicted_days_until_refill"],
        "prediction_status": pred["prediction_status"],
        "refill_confidence": pred["refill_confidence"],
        "rejection_reason": pred["rejection_reason"],
    })
    assert elig["is_eligible"] is False

    sched = RefillReminderScheduler()
    res = sched.schedule_refill_cycle({
        **row,
        "predicted_days_until_refill": pred["predicted_days_until_refill"],
        "prediction_status": pred["prediction_status"],
        "refill_confidence": pred["refill_confidence"],
        "rejection_reason": pred["rejection_reason"],
    })
    assert res["scheduled"] is False
    assert len(res["reminders"]) == 0


def test_batch_mixed_routes(model_bundle):
    rows = [
        _base_features(customerId="LOW", purchase_count_so_far=2),
        _base_features(
            customerId="CORE",
            purchase_count_so_far=12,
            historical_interval_median=30.0,
            historical_interval_norm_mad=0.1,
            cadence_drift=1.0,
        ),
        _base_features(
            customerId="SEC",
            purchase_count_so_far=8,
            historical_interval_median=40.0,
            historical_interval_norm_mad=0.42,
            cadence_drift=9.0,
        ),
    ]
    out = generate_batch_predictions(model_bundle, pd.DataFrame(rows))
    assert out.loc[0, "prediction_status"] == "rejected"
    assert out.loc[1, "prediction_route"] == ROUTE_CORE
    assert out.loc[1, "predicted_days_until_refill"] == 30.0
    assert out.loc[2, "prediction_route"] == ROUTE_SECONDARY
    assert out.loc[2, "refill_confidence"] == CONF_MEDIUM
    assert pd.isna(out.loc[0, "expected_refill_date"]) or out.loc[0, "expected_refill_date"] is None
    assert out.loc[1, "expected_refill_date"] is not None
