"""New prediction features and rules: calendar anchoring, visit context, salt priors / default dose,
short-history shrinkage, relative-target models, label-bias guard."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from refillcare.engine.prediction_rules import (
    EngineRules,
    device_ceiling_days,
    dom_anchor_interval,
    is_normal_quantity,
    path_b_interval,
)
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.features.context import PopulationStats, VisitIndex, build_context, salt_key
from refillcare.features.point_in_time import POINT_IN_TIME_FEATURE_COLS, compute_point_in_time_features_single
from refillcare.models.supply_hybrid import predict_with_bundle


def _monthly_on_day(day: int, n: int, start=(2026, 1)) -> list:
    y, m = start
    out = []
    for _ in range(n):
        out.append(date(y, m, day))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


# ------------------------------------------------------------------ calendar
def test_calendar_anchored_buyer_vs_strict_interval_buyer():
    anchored = _monthly_on_day(5, 8)  # always the 5th: intervals 28-31 days
    f = compute_point_in_time_features_single(anchored, [2] * 8, ["1X15"] * 8, "TELMA 40")
    assert f["dom_spread_days"] < 1 and f["dom_fit_gain_days"] >= 0.3

    strict = [date(2026, 1, 1) + timedelta(days=30 * i) for i in range(8)]  # drifts across dates
    g = compute_point_in_time_features_single(strict, [2] * 8, ["1X15"] * 8, "TELMA 40")
    assert g["dom_fit_gain_days"] <= 0
    assert dom_anchor_interval(strict[-1], 30, g, 8) == (30, False)


def test_dom_anchor_moves_monthly_buyer_to_usual_day():
    hist = _monthly_on_day(5, 7)  # last purchase 5 Jul
    f = compute_point_in_time_features_single(hist, [2] * 7, ["1X15"] * 7, "TELMA 40")
    interval, moved = dom_anchor_interval(hist[-1], 28, f, 7)  # model said 2 Aug
    assert moved and (hist[-1] + timedelta(days=interval)).day in (4, 5, 6)


def test_every_feature_column_is_always_returned():
    f = compute_point_in_time_features_single([date(2026, 1, 1)], [1], ["1X10"], "X")
    assert set(POINT_IN_TIME_FEATURE_COLS) <= set(f)


# ------------------------------------------------------------------ visit context
def test_visit_index_is_point_in_time_and_sees_other_medicines():
    ev = pd.DataFrame({
        "customerId": ["C1"] * 7,
        "itemId": ["A", "A", "A", "B", "B", "B", "A"],
        "date": [date(2026, 1, 1), date(2026, 1, 31), date(2026, 3, 2), date(2026, 1, 10), date(2026, 2, 9), date(2026, 3, 11), date(2026, 4, 1)],
    })
    idx = VisitIndex.from_events(ev)
    ctx = idx.context("C1", "A", date(2026, 3, 2))
    assert ctx["days_since_prev_visit_any_item"] == 21  # B on 9 Feb
    assert ctx["active_items_180d"] == 2.0
    assert ctx["other_item_next_due_days"] == 9.0  # B due 11 Mar (9 Feb + 30)
    # The 11 Mar and 1 Apr purchases are in the future at 2 Mar and must not be visible
    assert idx.context("C1", "A", date(2026, 3, 2))["visits_last_180d"] == 5.0


# ------------------------------------------------------------------ salt priors / default dose
def _salt_events(n_customers=12, gap=30, qty=2, as_salt="TELMISARTAN-40Mg [00]"):
    rows = []
    for c in range(n_customers):
        for k in range(5):
            rows.append({"customerId": f"C{c}", "itemId": "TELMA40", "date": date(2026, 1, 1) + timedelta(days=gap * k),
                         "quantity": qty, "packing": "1X15", "salt": as_salt, "itemName": "TELMA 40"})
    return pd.DataFrame(rows)


def test_population_stats_salt_median_and_default_dose_respect_cutoff():
    ev = _salt_events()
    stats = PopulationStats.from_events(ev)
    s = stats.lookup(salt_key("TELMISARTAN-40Mg [00]"))
    assert s["median_interval"] == 30 and s["units_per_day"] == pytest.approx(1.0)  # 2 x 15 tabs / 30 days
    early = PopulationStats.from_events(ev, as_of=date(2026, 1, 15))
    assert early.lookup(salt_key("TELMISARTAN-40Mg [00]")) is None  # not enough history before cut-off

    f = compute_point_in_time_features_single(
        [date(2026, 3, 1), date(2026, 3, 31), date(2026, 4, 30)], [2, 2, 4], ["1X15"] * 3, "TELMA 40",
        context={"salt_stats": s},
    )
    assert f["dose_supply_days"] == pytest.approx(60, abs=1)  # 60 tabs at the salt's 1 tab/day


def test_path_b_modes():
    prior = {"median_interval": 30.0}
    assert path_b_interval("dos", [40.0, 44.0], 50.0, prior) == (50.0, "dos")
    shr, m = path_b_interval("shrink", [40.0, 44.0], 50.0, prior)
    assert m == "shrinkage" and shr == pytest.approx((2 * 42 + 2 * 30) / 4)
    bl, m = path_b_interval("blend", [40.0, 44.0], 50.0, prior)
    assert m == "blend" and bl == pytest.approx(0.5 * shr + 25)


def test_quantity_archetypes_are_never_routed_to_median():
    assert is_normal_quantity({"quantity_ratio_vs_typical": 1.0})
    assert not is_normal_quantity({"quantity_ratio_vs_typical": 0.5, "is_partial_purchase": 1.0})
    assert not is_normal_quantity({"quantity_ratio_vs_typical": 2.0})


def test_rules_ship_off_by_default(monkeypatch):
    for k in ("REFILLCARE_RULE_HIGH_STABILITY_MEDIAN", "REFILLCARE_RULE_INSULIN_MEDIAN", "REFILLCARE_RULE_DOM_ANCHOR",
              "REFILLCARE_RULE_PATH_B_MODE", "REFILLCARE_RULE_DEVICE_CAP_MODE"):
        monkeypatch.delenv(k, raising=False)
    assert EngineRules.from_env() == EngineRules.legacy()


def test_device_ceiling_observed_never_below_own_median():
    assert device_ceiling_days("strict", 30.0, 42.0) == 30.0
    assert device_ceiling_days("observed", 30.0, 42.0) == 42.0
    assert device_ceiling_days("observed", 30.0, 21.0) == 30.0  # a fast user is still capped at the label
    assert device_ceiling_days("observed", None, 42.0) is None  # not a device


def test_observed_device_cap_reminds_slow_pen_user_at_own_cadence():
    dates = [date(2026, 1, 1) + timedelta(days=42 * i) for i in range(8)]
    kw = dict(customer_id="C1", customer_name="P", mobile_no="919848012345", item_id="BASU",
              item_name="BASUGINE DISPOSABLE PEN 3ML", dates=dates, quantities=[1.0] * 8, packings=["3ML"] * 8,
              as_of_date=dates[-1])
    from refillcare.ml.model_provider import ModelProvider

    strict = UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline(), identity_guard=False).evaluate_customer_item_trajectory(**kw)
    observed = UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline(), identity_guard=False,
                                           rules=EngineRules(device_cap_mode="observed")).evaluate_customer_item_trajectory(**kw)
    assert strict.predicted_interval_days == 30
    assert observed.predicted_interval_days == 42


def test_rolling_folds_are_back_to_back_and_out_of_time():
    from refillcare.ml.service import plan_time_splits, rolling_folds

    days = pd.date_range("2024-01-01", "2026-09-30", freq="D")
    hist = pd.DataFrame({"invoice_date": days})
    plan = plan_time_splits(hist)
    folds = rolling_folds(hist, 3)
    assert len(folds) == 3 and folds[0]["end"] == plan["test"]["end"]
    for newer, older in zip(folds, folds[1:]):
        assert older["end"] == newer["start"] - timedelta(days=1)
    for f in folds:
        assert f["train_end"] < f["start"] - timedelta(days=29)  # validation-sized gap, never sees the fold


# ------------------------------------------------------------------ model plumbing
class _ConstModel:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.array([self.v] * len(X))


def test_log_ratio_target_is_inverted_at_serving():
    bundle = {"model": _ConstModel(np.log(1.2)), "feature_cols": ["historical_median_interval"],
              "metadata": {"target_transform": "log_ratio_to_personal_median"}}
    assert predict_with_bundle(bundle, {"historical_median_interval": 30.0}) == pytest.approx(36.0)
    raw = {"model": _ConstModel(33.0), "feature_cols": ["historical_median_interval"], "metadata": {}}
    assert predict_with_bundle(raw, {"historical_median_interval": 30.0}) == pytest.approx(33.0)


def test_batch_scoring_only_keys_but_uses_all_patient_items():
    rows = []
    for item, off in (("A", 0), ("B", 10)):
        for k in range(7):
            rows.append({"customerId": "C1", "customerName": "P", "itemId": item, "itemName": "TELMA 40 TAB",
                         "invoice_date": pd.Timestamp(date(2026, 1, 1) + timedelta(days=off + 30 * k)),
                         "quantity": 2, "packing": "1X15", "MOBILE_NO": "9876543210", "refillcare_eligible": True})
    df = pd.DataFrame(rows)
    eng = UnifiedRefillDecisionEngine(identity_guard=False)
    _, decs = eng.evaluate_all_customer_items(df, as_of_date=date(2026, 8, 1), only_keys={("C1", "A")})
    assert [d.item_id for d in decs] == ["A"]


def test_training_matrix_drops_events_near_cutoff():
    from refillcare.ml.training import build_training_matrix

    rows = []
    for c in range(30):
        for k in range(14):
            rows.append({"customerId": f"C{c}", "customerName": f"P{c}", "itemId": "TELMA40", "itemName": "TELMA 40 TAB",
                         "invoice_date": pd.Timestamp(date(2025, 6, 1) + timedelta(days=30 * k)), "quantity": 2,
                         "packing": "1X15", "MOBILE_NO": f"98{c:08d}", "refillcare_eligible": True})
    hist = pd.DataFrame(rows)
    cutoff = date(2026, 7, 1)
    mat = build_training_matrix(hist, date(2000, 1, 1), cutoff, label_horizon_days=90)
    assert not mat.empty and max(mat["event_date"]) <= cutoff - timedelta(days=90)
    assert {"usual_purchase_dom", "visits_last_180d", "salt_median_interval"} <= set(mat.columns)
