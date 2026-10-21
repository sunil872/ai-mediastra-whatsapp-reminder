"""ML lifecycle: evidence-based evaluation, promotion gate, registry invariants, outcomes, safety."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.models  # noqa: F401
from database.connection import Base
from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel, WhatsAppDeliveryLogModel
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
from refillcare.ml import registry as reg
from refillcare.ml.evaluation import BASELINE_METHOD, run_backtest
from refillcare.ml.model_provider import ModelProvider
from refillcare.ml.outcomes import outcome_summary, refresh_outcomes
from refillcare.ml.promotion import evaluate_gate
from refillcare.models.supply_hybrid import apply_final_safety_clamp, evaluate_human_hybrid_prediction


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def synthetic_history(n_customers=120, cadence=30, noise=3, start=date(2025, 1, 1), end=date(2026, 9, 30), seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        d = start + timedelta(days=int(rng.integers(0, 20)))
        while d <= end:
            rows.append({
                "customerId": f"C{c:04d}", "customerName": f"P{c}", "itemId": "TELMA40", "itemName": "TELMA 40 TAB",
                "invoice_date": pd.Timestamp(d), "quantity": 2, "packing": "1X15", "MOBILE_NO": f"98{c:08d}",
                "invoice_number": f"S0/{c}-{d}", "refillcare_eligible": True,
            })
            d = d + timedelta(days=int(cadence + rng.integers(-noise, noise + 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ evaluation
def test_backtest_scores_methods_on_identical_paired_events():
    hist = synthetic_history()
    variants = {"engine[baseline]": UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline(), identity_guard=False)}
    rep = run_backtest(hist, variants, date(2026, 6, 1), date(2026, 7, 31), max_events=400)
    paired = rep["paired_comparison"]
    assert paired["n_events"] > 50
    m = paired["methods"]
    assert set(m) == {BASELINE_METHOD, "days_of_supply", "engine[baseline]"}
    assert all(v["n"] == paired["n_events"] for v in m.values())
    # regular 30-day cadence: the personal median should be accurate
    assert m[BASELINE_METHOD]["within_7_days_pct"] > 90
    assert "within_7_days_ci95" in m[BASELINE_METHOD]
    assert rep["event_counts"]["censored_excluded"] >= 0


def test_backtest_never_sees_future_purchases():
    hist = synthetic_history(n_customers=20)
    from refillcare.ml.evaluation import build_evaluation_events, prepare_purchase_events

    sel = build_evaluation_events(prepare_purchase_events(hist), date(2026, 6, 1), date(2026, 6, 30))
    for ev in sel["events"]:
        assert ev["dates"][-1] <= date(2026, 6, 30)
        assert all(d <= ev["dates"][-1] for d in ev["dates"])


# ------------------------------------------------------------------ promotion gate
def _report(cand_w7, base_w7, n=500, ci_low=1.0, mae_diff=-0.5, early_c=15.0, early_b=15.0, overlap=None):
    return {
        "run_id": "EVAL-T",
        "training_overlap_versions": overlap or [],
        "version_by_method": {"engine[cand]": "cand"},
        "paired_comparison": {
            "n_events": n,
            "methods": {
                "engine[cand]": {"within_7_days_pct": cand_w7, "mae_days": 8.0, "early_pct": early_c},
                BASELINE_METHOD: {"within_7_days_pct": base_w7, "mae_days": 8.5, "early_pct": early_b},
                "engine[prod]": {"within_7_days_pct": base_w7 - 1, "mae_days": 9.0, "early_pct": 20.0},
            },
            "diffs": {f"engine[cand] vs {BASELINE_METHOD}": {"within_7_days_diff_pp": cand_w7 - base_w7, "within_7_days_diff_ci95": [ci_low, ci_low + 4], "mae_diff_days": mae_diff}},
        },
        "segments": {},
    }


def test_gate_approves_significant_improvement():
    d = evaluate_gate(_report(65, 60), "engine[cand]", candidate_is_baseline=False, production_method="engine[prod]")
    assert d["approved"], d["failed_checks"]


@pytest.mark.parametrize(
    "kwargs,failed",
    [
        (dict(cand_w7=61, base_w7=60), "beats_baseline_within_7"),
        (dict(cand_w7=65, base_w7=60, ci_low=-0.5), "gain_is_significant"),
        (dict(cand_w7=65, base_w7=60, n=50), "min_paired_events"),
        (dict(cand_w7=65, base_w7=60, mae_diff=2.0), "mae_not_worse"),
        (dict(cand_w7=65, base_w7=60, early_c=30.0), "no_extra_reminder_spam"),
        (dict(cand_w7=65, base_w7=60, overlap=["cand"]), "out_of_time_evaluation"),
    ],
)
def test_gate_rejects_unproven_models(kwargs, failed):
    d = evaluate_gate(_report(**kwargs), "engine[cand]", candidate_is_baseline=False)
    assert not d["approved"] and failed in d["failed_checks"]


def test_gate_rejects_segment_regression():
    r = _report(65, 60)
    r["segments"] = {"therapy": {"insulin": {"n": 60, "methods": {"engine[cand]": {"within_7_days_pct": 40}, BASELINE_METHOD: {"within_7_days_pct": 60}}}}}
    d = evaluate_gate(r, "engine[cand]", candidate_is_baseline=False)
    assert "segment:therapy=insulin" in d["failed_checks"]


# ------------------------------------------------------------------ registry invariants
def test_bootstrap_registry_is_truthful(db):
    reg.ensure_bootstrap_versions(db)
    base = reg.get_version(db, reg.BASELINE_VERSION_ID)
    assert base.model_type == reg.TYPE_BASELINE_MEDIAN and base.artifact_uri is None
    legacy = reg.get_version(db, reg.LEGACY_VERSION_ID)
    assert legacy.is_validated is False
    if legacy.artifact_uri:
        assert not legacy.artifact_uri.startswith(("C:", "/"))  # relative artifact path


def test_promotion_keeps_exactly_one_production_model(db):
    for vid in ("t-a", "t-b"):
        reg.register_version(db, version_id=vid, model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_BASELINE_MEDIAN)
    reg.promote_version(db, "t-a", "admin", {"approved": True})
    reg.promote_version(db, "t-b", "admin", {"approved": True})
    prods = [v for v in reg.list_versions(db) if v.status == reg.STATUS_PRODUCTION]
    assert [v.version_id for v in prods] == ["t-b"]
    assert reg.get_version(db, "t-a").status == reg.STATUS_RETIRED


def test_forced_promotion_is_recorded_and_not_validated(db):
    reg.register_version(db, version_id="t-f", model_family=reg.FAMILY_PATH_A_INTERVAL, model_type=reg.TYPE_BASELINE_MEDIAN)
    v = reg.promote_version(db, "t-f", "admin", {"approved": False, "forced": True, "force_reason": "pilot"})
    assert v.is_validated is False and v.promotion_decision["forced"] is True


# ------------------------------------------------------------------ provenance & safety
def test_every_path_a_decision_records_model_provenance():
    eng = UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline(), identity_guard=False)
    dates = [date(2026, 1, 1) + timedelta(days=30 * i) for i in range(8)]
    dec = eng.evaluate_customer_item_trajectory("C1", "P", "9876543210", "I1", "TELMA 40 TAB", dates, [2] * 8, ["1X15"] * 8, as_of_date=dates[-1])
    assert dec.is_eligible
    assert dec.model_version == "baseline-median-v1"
    assert dec.prediction_source in ("BASELINE_MEDIAN", "BASELINE_CLAMPED")
    assert dec.predicted_interval_days == 30


def test_safety_clamp_caps_insulin_and_global_max():
    assert apply_final_safety_clamp(490, 1000, device_cap_days=30) == (30.0, True)
    assert apply_final_safety_clamp(265, 1000)[0] == 120.0
    assert apply_final_safety_clamp(2, 10)[0] == 5.0


def test_multipack_archetype_cannot_explode():
    feats = {"estimated_effective_supply": 30.0, "estimated_consumption_velocity": 0.3, "quantity_ratio_vs_typical": 10.0, "typical_units_median": 3.0}
    out = evaluate_human_hybrid_prediction(feats, "MEDIUM-SAFE", 49.0, use_ml=True, model_bundle={"model": None, "feature_cols": ["missing_feature"]}, device_cap_days=30.0)
    assert out["predicted_interval_days"] <= 30
    assert out["prediction_source"] in ("HEURISTIC_FALLBACK", "ML_CLAMPED")


def test_identity_conflict_blocks_outreach():
    eng = UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline())
    dates = [date(2026, 1, 1) + timedelta(days=30 * i) for i in range(8)]
    mobiles = ["9000000001"] * 5 + ["9111111111", "9222222222", "9333333333"]
    dec = eng.evaluate_customer_item_trajectory("SRINIVAS", "SRINIVAS", "9333333333", "I1", "TELMA 40 TAB", dates, [2] * 8, ["1X15"] * 8, as_of_date=dates[-1], mobile_history=mobiles)
    assert dec.is_eligible is False and dec.expected_refill_date is None
    assert dec.decision_reason.startswith("IDENTITY_CONFLICT")


# ------------------------------------------------------------------ outcomes
def test_outcomes_classify_on_time_late_and_open(db):
    def add(key, lpd, erd):
        db.add(RefillDecisionModel(
            decision_id=f"DEC-{key}", cycle_id=f"RC-{key}", customer_id=key.split("_")[0], item_id=key.split("_")[1],
            customer_item_key=key, path="PATH_A", purchase_count=8, is_eligible=True, stability_tier="HIGH",
            prediction_method="X", predicted_interval_days=(erd - lpd).days, last_purchase_date=lpd, expected_refill_date=erd,
            model_version="v1", prediction_source="ML",
        ))
    add("A_1", date(2026, 8, 1), date(2026, 8, 31))
    add("B_1", date(2026, 8, 1), date(2026, 8, 31))
    add("C_1", date(2026, 9, 1), date(2026, 10, 1))
    db.commit()
    hist = pd.DataFrame([
        {"customerId": "A", "itemId": "1", "invoice_date": "2026-08-01"}, {"customerId": "A", "itemId": "1", "invoice_date": "2026-09-02"},
        {"customerId": "B", "itemId": "1", "invoice_date": "2026-08-01"}, {"customerId": "B", "itemId": "1", "invoice_date": "2026-09-25"},
        {"customerId": "C", "itemId": "1", "invoice_date": "2026-09-01"},
    ])
    counts = refresh_outcomes(db, hist, as_of=date(2026, 9, 30))
    assert counts["ON_TIME"] == 1 and counts["EARLY"] == 1 and counts["OPEN"] == 1
    summ = outcome_summary(db, window_days=90, as_of=date(2026, 10, 5))
    assert summ["overall"]["resolved"] == 2 and summ["overall"]["within_7_days_pct"] == 50.0


# ------------------------------------------------------------------ dispatch safety & lifecycle
def _seed_stage(db, status="APPROVED", send=date(2026, 10, 10)):
    db.add(RefillDecisionModel(
        decision_id="DEC_1", cycle_id="CYC_1", customer_id="C1", customer_name="Patient One", mobile_no="919876543210",
        item_id="I1", item_name="TELMA 40 TAB", customer_item_key="C1_I1", path="PATH_A", purchase_count=8, is_eligible=True,
        stability_tier="HIGH", prediction_method="X", predicted_interval_days=30,
        last_purchase_date=date(2026, 9, 10), expected_refill_date=date(2026, 10, 10),
    ))
    db.add(ReminderCycleModel(cycle_id="CYC_1", customer_item_key="C1_I1", customer_id="C1", item_id="I1", decision_id="DEC_1",
                              last_purchase_date=date(2026, 9, 10), expected_refill_date=date(2026, 10, 10), is_active=True))
    db.add(ReminderStageModel(reminder_id="REM_1", cycle_id="CYC_1", customer_id="C1", item_id="I1", customer_item_key="C1_I1",
                              stage_offset=0, target_send_date=send, expected_refill_date=date(2026, 10, 10), status=status))
    db.commit()


def test_dry_run_dispatch_never_changes_status(db):
    from refillcare.engine.persistence import RefillPersistenceManager

    _seed_stage(db)
    res = RefillPersistenceManager(store_name="T").dispatch_reminder_stage(db, "REM_1", is_dry_run=True)
    assert res["success"] and res["status"] == "APPROVED"
    assert db.query(WhatsAppDeliveryLogModel).count() == 1


def test_live_dispatch_is_idempotent(db, monkeypatch):
    from refillcare.engine.persistence import RefillPersistenceManager
    from refillcare.whatsapp import client as wa_client

    monkeypatch.setattr(wa_client.RefillWhatsAppClient, "send_refill_reminder", lambda self, **kw: {"success": True, "status_code": 200, "message_id": "MSG1"})
    _seed_stage(db)
    pm = RefillPersistenceManager(store_name="T")
    assert pm.dispatch_reminder_stage(db, "REM_1", is_dry_run=False)["status"] == "SENT"
    stage = db.query(ReminderStageModel).get("REM_1")
    stage.status = "APPROVED"  # even if someone re-approves...
    db.commit()
    with pytest.raises(ValueError, match="already delivered"):
        pm.dispatch_reminder_stage(db, "REM_1", is_dry_run=False)


def test_batch_live_dispatch_requires_approval(db, monkeypatch):
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher

    _seed_stage(db, status="PENDING")
    res = RefillWhatsAppDispatcher().dispatch_batch(db, target_date=date(2026, 10, 10), dry_run=False)
    assert res["dispatched_count"] == 0 and res["requires_approval"] is True


def test_expire_past_due_stages(db):
    from refillcare.engine.persistence import RefillPersistenceManager

    _seed_stage(db, status="PENDING", send=date(2026, 10, 1))
    assert RefillPersistenceManager().expire_past_due_stages(db, today=date(2026, 10, 5)) == 1
    assert db.query(ReminderStageModel).get("REM_1").status == "EXPIRED"


def test_rollback_reverts_cycles_and_restores_superseded(db):
    from refillcare.engine.persistence import RefillPersistenceManager

    _seed_stage(db, status="PENDING")
    old = ReminderCycleModel(cycle_id="CYC_0", customer_item_key="C1_I1", customer_id="C1", item_id="I1", decision_id=None,
                             last_purchase_date=date(2026, 8, 10), expected_refill_date=date(2026, 9, 10), is_active=False,
                             superseded_by_purchase_date=date(2026, 9, 10))
    db.add(old)
    db.add(ReminderStageModel(reminder_id="REM_0", cycle_id="CYC_0", customer_id="C1", item_id="I1", customer_item_key="C1_I1",
                              stage_offset=0, target_send_date=date(2026, 9, 10), expected_refill_date=date(2026, 9, 10), status="SUPERSEDED_BY_PURCHASE"))
    db.commit()
    m = RefillPersistenceManager().revert_rolled_back_purchases(db, {"C1_I1": {date(2026, 9, 10)}})
    assert m["cycles_rolled_back"] == 1 and m["cycles_reactivated"] == 1
    assert db.query(ReminderCycleModel).get("CYC_1").is_active is False
    assert db.query(ReminderStageModel).get("REM_1").status == "ROLLED_BACK"
    assert db.query(ReminderCycleModel).get("CYC_0").is_active is True
    assert db.query(ReminderStageModel).get("REM_0").status == "PENDING"


# ------------------------------------------------------------------ dynamic train / validation / test windows
def test_split_plan_moves_with_new_data_and_never_overlaps():
    from refillcare.ml.service import plan_time_splits

    hist = synthetic_history(n_customers=20)
    p1 = plan_time_splits(hist, test_days=30, validation_days=30, censor_buffer_days=45)
    assert p1["data_end"] <= date(2026, 9, 30)
    assert p1["test"]["end"] == p1["data_end"] - timedelta(days=45)
    assert p1["train"]["end"] < p1["validation"]["start"] <= p1["validation"]["end"] < p1["test"]["start"]

    # A new month of sales arrives -> every window moves forward by the same amount.
    newer = synthetic_history(n_customers=20, end=date(2026, 10, 31))
    p2 = plan_time_splits(newer, test_days=30, validation_days=30, censor_buffer_days=45)
    shift = p2["data_end"] - p1["data_end"]
    assert shift.days > 0
    for w in ("train", "validation", "test"):
        assert p2[w]["end"] - p1[w]["end"] == shift


def test_split_plan_rejects_too_little_history():
    from refillcare.ml.service import plan_time_splits

    with pytest.raises(ValueError):
        plan_time_splits(synthetic_history(n_customers=5, start=date(2026, 6, 1)))


def test_retrain_cycle_uses_planned_windows_and_is_idempotent(db, tmp_path, monkeypatch):
    from refillcare.ml.service import plan_time_splits, retrain_cycle

    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    hist = synthetic_history(n_customers=60, start=date(2024, 6, 1))
    plan = plan_time_splits(hist)
    res = retrain_cycle(db, history_df=hist, promote_best=True, max_eval_events=500, max_train_events=5000)

    cand = reg.get_version(db, res["candidate_version"])
    assert res["candidate_trained_now"] is True
    assert cand.train_window_end == plan["train"]["end"] < plan["validation"]["start"]
    assert res["validation"]["window"]["start"] == str(plan["validation"]["start"])
    assert res["test"]["window"]["start"] == str(plan["test"]["start"])
    assert res["selected_on_validation"] != reg.LEGACY_VERSION_ID
    # Too few paired events in a 30-day window of synthetic data -> the gate must refuse promotion.
    assert res["gate"]["approved"] is False and res["promoted"] is False

    again = retrain_cycle(db, history_df=hist, max_eval_events=500, max_train_events=5000)
    assert again["candidate_trained_now"] is False and again["candidate_version"] == cand.version_id


def test_freshness_experiment_cutoffs_follow_the_data(db, tmp_path, monkeypatch):
    from refillcare.ml.service import freshness_experiment, plan_time_splits

    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    hist = synthetic_history(n_customers=60, start=date(2024, 6, 1))
    plan = plan_time_splits(hist)
    res = freshness_experiment(db, history_df=hist, step_days=30, max_eval_events=500, max_train_events=5000)
    assert res["fresh_model"]["train_cutoff"] == plan["test"]["start"] - timedelta(days=1)
    assert res["older_model"]["train_cutoff"] == plan["test"]["start"] - timedelta(days=31)
    assert res["results"]["window"]["start"] == str(plan["test"]["start"])
    assert res["fresh_minus_older_within_7_pp"] is not None
