"""Regression tests for Phase 17F Path A classifier (non-production).

Canonical interval patterns from Phase 17E. Does not send WhatsApp,
does not overwrite the production model, and does not enable Path B.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from refillcare.features.engineering import build_feature_dataset
from refillcare.models.hybrid_strategy import (
    STRATEGY_NAME as HYBRID_STRATEGY_NAME,
    evaluate_hybrid_eligibility,
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
)
from refillcare.models.path_a_classifier import (
    STRATEGY_NAME as PATH_A_STRATEGY_NAME,
    USE_PATH_A_CLASSIFIER_V17F,
    LABEL_HIGH,
    LABEL_MEDIUM,
    LABEL_UNSTABLE,
    classify_path_a,
    classify_path_a_from_intervals,
    compute_path_a_interval_diagnostics,
    evaluate_eligibility,
    is_path_a_classifier_enabled,
)


STABLE_INTERVALS = [30, 30, 31, 29, 30, 32]
VARIABLE_INTERVALS = [30, 60, 45, 30, 90, 30, 60, 30, 90]
UNSTABLE_INTERVALS = [5, 180, 12, 240, 3, 150]
# Low NormMAD/Drift but max/median > 2.2 → must NOT be HIGH under Path A v17F
CONTAMINATED_INTERVALS = [30, 30, 30, 30, 30, 30, 30, 90]


def test_feature_flag_defaults_off():
    assert USE_PATH_A_CLASSIFIER_V17F is False or isinstance(USE_PATH_A_CLASSIFIER_V17F, bool)
    # Default environment should keep Path A classifier disabled for production safety
    assert is_path_a_classifier_enabled() is False


def test_stable_pattern_is_high():
    decision = classify_path_a_from_intervals(STABLE_INTERVALS)
    assert decision["path_a_class"] == LABEL_HIGH
    assert decision["label"] == LABEL_HIGH
    assert decision["confidence"] == CONF_HIGH
    assert decision["is_eligible"] is True
    assert decision["strategy"] == PATH_A_STRATEGY_NAME
    assert decision["path"] == "A"


def test_recurring_variable_pattern_is_medium():
    decision = classify_path_a_from_intervals(VARIABLE_INTERVALS)
    assert decision["path_a_class"] == LABEL_MEDIUM
    assert decision["label"] == LABEL_MEDIUM
    assert decision["confidence"] == CONF_MEDIUM
    assert decision["is_eligible"] is True
    # Baseline hybrid rejects this pattern on Drift > 10 — Path A recovers it
    baseline = evaluate_hybrid_eligibility(
        {
            "purchase_count_so_far": len(VARIABLE_INTERVALS) + 1,
            "prior_intervals": VARIABLE_INTERVALS,
        }
    )
    assert baseline["confidence"] == CONF_REJECTED


def test_chaotic_pattern_is_unstable():
    decision = classify_path_a_from_intervals(UNSTABLE_INTERVALS)
    assert decision["path_a_class"] == LABEL_UNSTABLE
    assert decision["label"] == LABEL_UNSTABLE
    assert decision["confidence"] == CONF_REJECTED
    assert decision["is_eligible"] is False


def test_contaminated_high_example_is_not_high():
    diag = compute_path_a_interval_diagnostics(CONTAMINATED_INTERVALS)
    # Sanity: would look "regular" on NormMAD/Drift alone
    assert diag["norm_mad"] <= 0.35
    assert diag["cadence_drift"] <= 7.0
    assert diag["max_over_median"] > 2.2

    decision = classify_path_a_from_intervals(CONTAMINATED_INTERVALS)
    assert decision["path_a_class"] != LABEL_HIGH
    assert decision["path_a_class"] in {LABEL_MEDIUM, LABEL_UNSTABLE}


def test_missing_metrics_are_unstable():
    decision = classify_path_a(
        {
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            # NormMAD / Drift intentionally missing; no prior_intervals to recompute
        }
    )
    assert decision["path_a_class"] == LABEL_UNSTABLE
    assert decision["confidence"] == CONF_REJECTED
    assert "Missing" in (decision["rejection_reason"] or "")


def test_missing_diagnostics_without_intervals_unstable():
    decision = classify_path_a(
        {
            "purchase_count_so_far": 8,
            "historical_interval_median": 30.0,
            "historical_interval_norm_mad": 0.10,
            "cadence_drift": 2.0,
            # diagnostics missing
        }
    )
    assert decision["path_a_class"] == LABEL_UNSTABLE
    assert "diagnostics" in (decision["rejection_reason"] or "").lower() or "Missing" in (
        decision["rejection_reason"] or ""
    )


def test_purchase_count_below_six_unstable():
    decision = classify_path_a_from_intervals(STABLE_INTERVALS, purchase_count=5)
    assert decision["path_a_class"] == LABEL_UNSTABLE
    assert "Path A" in (decision["rejection_reason"] or "") or "purchase_count" in (
        decision["rejection_reason"] or ""
    )


def test_evaluate_eligibility_defaults_to_baseline_hybrid():
    row = {
        "purchase_count_so_far": 8,
        "historical_interval_median": 30.0,
        "historical_interval_norm_mad": 0.10,
        "cadence_drift": 2.0,
        "prior_intervals": STABLE_INTERVALS,
    }
    decision = evaluate_eligibility(row)  # flag off → baseline
    assert decision["strategy"] == HYBRID_STRATEGY_NAME
    assert decision["confidence"] == CONF_HIGH


def test_evaluate_eligibility_can_force_path_a():
    decision = evaluate_eligibility(
        {
            "purchase_count_so_far": len(VARIABLE_INTERVALS) + 1,
            "prior_intervals": VARIABLE_INTERVALS,
        },
        use_path_a_v17f=True,
    )
    assert decision["strategy"] == PATH_A_STRATEGY_NAME
    assert decision["path_a_class"] == LABEL_MEDIUM


def test_engineering_emits_path_a_diagnostic_columns():
    history = pd.DataFrame(
        {
            "customerId": ["C1"] * 7,
            "itemId": ["I1"] * 7,
            "invoice_number": list(range(1, 8)),
            "invoice_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-31",
                    "2025-03-02",
                    "2025-04-02",
                    "2025-05-01",
                    "2025-06-01",
                    "2025-07-03",
                ]
            ),
            "days_since_previous_purchase": [np.nan, 30, 30, 31, 29, 31, 32],
            "purchase_seq": list(range(1, 8)),
            "quantity": [30] * 7,
            "freeQuantity": [0] * 7,
            "netAmount": [100.0] * 7,
            "gstAmount": [0.0] * 7,
            "rate": [10.0] * 7,
            "mrp": [12.0] * 7,
            "discountPercent": [0.0] * 7,
            "itemCode": ["X"] * 7,
            "itemName": ["Med"] * 7,
            "therapeuticCategory": ["NA"] * 7,
            "generic_name": ["NA"] * 7,
            "salt_composition": ["NA"] * 7,
            "salt_category": ["TABLETS"] * 7,
            "salt_itemcat": ["ORAL"] * 7,
            "salt_pack": ["30S"] * 7,
        }
    )
    feats = build_feature_dataset(history)
    for col in (
        "pct_extreme_lt10_or_gt180",
        "pct_in_refill_band_15_120",
        "pct_within_50pct_of_median",
        "max_over_median",
        "historical_interval_norm_mad",
        "cadence_drift",
    ):
        assert col in feats.columns
    last = feats.iloc[-1]
    assert last["purchase_count_so_far"] == 7
    assert not math.isnan(last["max_over_median"])
    assert last["pct_in_refill_band_15_120"] == 100.0


def test_path_b_not_implemented_low_history_is_unstable_only():
    """Path B must not invent a separate recent-customer route."""
    decision = classify_path_a(
        {
            "purchase_count_so_far": 3,
            "prior_intervals": [30, 30],
        }
    )
    assert decision["path"] == "A"
    assert decision["path_a_class"] == LABEL_UNSTABLE
    assert "Path B" not in (decision.get("rejection_reason") or "")
