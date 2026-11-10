"""ML lifecycle: evidence-based evaluation, promotion gate, registry invariants, outcomes, safety."""
from __future__ import annotations

from dataclasses import replace as dataclasses_replace
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


class _FakeWhatsAppClient:
    """Records sends; never makes an HTTP call (no real WhatsApp messages in tests)."""

    template_name = "refillcare_medicine_reminder"

    def __init__(self):
        self.sent = []

    def send_refill_reminder(self, phone_number, **kw):
        self.sent.append(phone_number)
        return {"success": True, "status_code": 200, "message": "ok", "message_id": f"MSG_{len(self.sent)}"}


def test_batch_live_dispatch_with_pharmacist_confirmation_approves_and_sends(db):
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher

    _seed_stage(db, status="PENDING")
    fake = _FakeWhatsAppClient()
    res = RefillWhatsAppDispatcher(client=fake).dispatch_batch(
        db, target_date=date(2026, 10, 10), dry_run=False, batch_limit=None, approve_before_send=True, approved_by="tester")
    assert res["success_count"] == 1 and res["requires_approval"] is False
    assert db.query(ReminderStageModel).get("REM_1").status == "SENT"
    assert fake.sent == ["919876543210"]


def _bundle(**kw):
    from types import SimpleNamespace

    base = dict(bundle_id="BND_1", customer_id="C1", customer_name="Patient One", mobile_no="919876543210",
                synced_items=[SimpleNamespace(item_name="TELMA 40 TAB")], anchor_item_name="TELMA 40 TAB",
                anchor_refill_date=date(2026, 10, 10), lifecycle_tier="DUE", stage_reminder_ids=["REM_1"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_medsync_send_all_after_confirmation_counts_and_never_double_sends(db):
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher

    _seed_stage(db, status="PENDING")
    fake = _FakeWhatsAppClient()
    disp = RefillWhatsAppDispatcher(client=fake)
    bundles = [_bundle(), _bundle(bundle_id="BND_2", mobile_no="12345", stage_reminder_ids=[])]
    summary = disp.summarize_medsync_send(db, bundles)
    assert summary["to_send"] == 1 and summary["no_valid_mobile"] == 1

    progress = []
    out = disp.dispatch_medsync_batch(db, bundles, dry_run=False, batch_limit=None, approve_before_send=True,
                                      approved_by="tester", on_progress=lambda d, t: progress.append((d, t)))
    assert out["success_count"] == 1 and progress == [(1, 1)]
    assert db.query(ReminderStageModel).get("REM_1").status == "SENT"

    # Clicking "Yes" again the same day sends nothing twice
    again = disp.dispatch_medsync_batch(db, [_bundle()], dry_run=False, batch_limit=None, approve_before_send=True)
    assert again["success_count"] == 0 and len(fake.sent) == 1
    assert disp.summarize_medsync_send(db, [_bundle()])["already_sent"] == 1


def test_medsync_live_send_without_confirmation_still_needs_approval(db):
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher

    _seed_stage(db, status="PENDING")
    fake = _FakeWhatsAppClient()
    out = RefillWhatsAppDispatcher(client=fake).dispatch_medsync_batch(db, [_bundle()], dry_run=False)
    assert out["success_count"] == 0 and fake.sent == []


def test_pharmacist_search_stop_and_resume_by_name_or_mobile(db):
    from refillcare.whatsapp.consent import is_opted_out
    from refillcare.whatsapp.pharmacist_views import search_patients, set_messages_allowed

    _seed_stage(db, status="PENDING")
    by_name = search_patients(db, "patient one")
    by_mobile = search_patients(db, "98765 43210")
    assert [p["customer_id"] for p in by_name] == ["C1"] == [p["customer_id"] for p in by_mobile]
    assert by_name[0]["messages_allowed"] is True
    assert search_patients(db, "x") == []  # too short

    set_messages_allowed(db, allowed=False, mobile_number=by_name[0]["mobile_number"], customer_id="C1")
    assert is_opted_out(db, customer_id="C1", phone="919876543210")
    assert search_patients(db, "Patient")[0]["messages_allowed"] is False
    # The new Send flow skips the stopped patient
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher
    assert RefillWhatsAppDispatcher(client=_FakeWhatsAppClient()).summarize_medsync_send(db, [_bundle()])["opted_out"] == 1

    set_messages_allowed(db, allowed=True, mobile_number="9876543210")
    assert not is_opted_out(db, customer_id="C1", phone="919876543210")


def test_delivery_log_view_shows_live_rows_for_india_day_only(db):
    from datetime import datetime

    from refillcare.whatsapp.pharmacist_views import delivery_log_rows

    def log(rid, status, dry, at, err=None, http=200):
        db.add(WhatsAppDeliveryLogModel(reminder_id=rid, phone_number="919876543210", customer_name="Patient One",
                                        template_name="t", is_dry_run=dry, status=status, http_status_code=http,
                                        error_detail=err, dispatched_at=at))

    log("A", "SUCCESS", False, datetime(2026, 11, 6, 19, 0))   # 7 Nov 00:30 IST -> counted for 7 Nov
    log("B", "FAILED", False, datetime(2026, 11, 7, 5, 0), err="Validation Error: bad number", http=400)
    log("C", "DRY_RUN_SUCCESS", True, datetime(2026, 11, 7, 5, 0))  # test row: never shown
    log("D", "SUCCESS", False, datetime(2026, 11, 7, 19, 0))   # 8 Nov 00:30 IST -> not 7 Nov
    db.commit()

    v = delivery_log_rows(db, day=date(2026, 11, 7))
    assert (v["sent"], v["failed"], len(v["rows"])) == (1, 1, 2)
    failed = delivery_log_rows(db, day=date(2026, 11, 7), status="Failed")["rows"]
    assert len(failed) == 1 and failed[0]["reason"] == "Invalid mobile number"
    assert set(failed[0]) == {"time", "patient", "mobile", "status", "reason"}  # no technical columns


def test_holdout_group_is_stable_about_the_set_share_and_never_messaged(db, monkeypatch):
    from refillcare.whatsapp.dispatcher import RefillWhatsAppDispatcher
    from refillcare.whatsapp.holdout import is_holdout

    monkeypatch.setenv("REFILLCARE_HOLDOUT_PCT", "10")
    phones = [f"98{i:08d}" for i in range(4000)]
    share = sum(is_holdout(phone=p) for p in phones) / len(phones)
    assert 0.08 < share < 0.12
    assert is_holdout(phone="91" + phones[0]) == is_holdout(phone=phones[0])  # same group for 10 / 12 digits
    monkeypatch.setenv("REFILLCARE_HOLDOUT_PCT", "0")
    assert not any(is_holdout(phone=p) for p in phones[:200])

    monkeypatch.setenv("REFILLCARE_HOLDOUT_PCT", "10")
    held = next(p for p in phones if is_holdout(phone=p))
    _seed_stage(db, status="PENDING")
    fake = _FakeWhatsAppClient()
    disp = RefillWhatsAppDispatcher(client=fake)
    b = _bundle(mobile_no=held)
    assert disp.summarize_medsync_send(db, [b])["holdout"] == 1
    out = disp.dispatch_medsync_batch(db, [b], dry_run=False, batch_limit=None, approve_before_send=True)
    assert out["skipped_holdout"] == 1 and fake.sent == []


def test_outcome_summary_reports_holdout_uplift(db):
    from refillcare.ml.outcomes import outcome_summary

    rep = outcome_summary(db, window_days=90, as_of=date(2026, 10, 31))
    assert "holdout_uplift" in rep and rep["holdout_uplift"]["uplift_pp"] is None  # no live sends yet


def _seed_ineligible(db, item="TELMA 40 TAB", last=date(2026, 10, 1)):
    db.add(RefillDecisionModel(
        decision_id=f"DEC-C9_I9-{last:%Y%m%d}", cycle_id=f"CYC-C9_I9-{last:%Y%m%d}", customer_id="C9", customer_name="Patient Nine",
        mobile_no="919812345678", item_id="I9", item_name=item, customer_item_key="C9_I9", path="INELIGIBLE", purchase_count=2,
        is_eligible=False, stability_tier="UNSTABLE", prediction_method="NONE", cadence_median=28.0,
        last_purchase_date=last, expected_refill_date=None, decision_reason="Insufficient purchase history",
    ))
    db.commit()


def test_pharmacist_add_survives_rescoring_moves_on_repurchase_and_can_be_removed(db):
    from refillcare.engine import pharmacist_overrides as po
    from refillcare.engine.persistence import RefillPersistenceManager

    _seed_ineligible(db)
    today = date(2026, 10, 25)
    cands = po.review_candidates(db, today - timedelta(days=7), today + timedelta(days=7))
    assert [c["customer_item_key"] for c in cands] == ["C9_I9"] and cands[0]["medicine_class"] == "Chronic"
    assert cands[0]["usual_gap_days"] == 28 and cands[0]["estimated_due_date"] == "2026-10-29"

    res = po.add_to_reminders(db, "C9_I9", next_refill_date=date(2026, 10, 29), added_by="tester", today=today)
    assert res["added"] and res["stages_created"] == 7
    cyc = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9", ReminderCycleModel.is_active == True).one()  # noqa: E712
    assert cyc.cycle_id.endswith("-PH") and cyc.expected_refill_date == date(2026, 10, 29)
    assert po.review_candidates(db, today - timedelta(days=7), today + timedelta(days=7)) == []  # no longer "missing"

    # Next upload re-scores the patient as NOT eligible again -> pharmacist decision is kept
    pm = RefillPersistenceManager()
    again = [po._decision_from_model(db.query(RefillDecisionModel).get("DEC-C9_I9-20261001"))]
    again[0] = dataclasses_replace(again[0], is_eligible=False, path="INELIGIBLE", expected_refill_date=None, cycle_id="CYC-C9_I9-20261001")
    pm.save_refill_decisions(db, again)
    pm.sync_reminder_cycles(db, again)
    assert db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9", ReminderCycleModel.is_active == True).count() == 1  # noqa: E712

    # Patient buys again on 30 Oct -> next reminder = 30 Oct + 28 days
    newer = [dataclasses_replace(again[0], last_purchase_date=date(2026, 10, 30), cycle_id="CYC-C9_I9-20261030",
                                 is_eligible=False, path="INELIGIBLE", expected_refill_date=None)]  # fresh engine decision
    pm.save_refill_decisions(db, newer)
    pm.sync_reminder_cycles(db, newer)
    act = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9", ReminderCycleModel.is_active == True).one()  # noqa: E712
    assert act.expected_refill_date == date(2026, 11, 27)

    out = po.remove_from_reminders(db, "C9_I9", removed_by="tester")
    assert out["removed"] and out["stages_cancelled"] > 0
    assert db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9", ReminderCycleModel.is_active == True).count() == 0  # noqa: E712


def test_pharmacist_cannot_add_known_non_chronic_and_engine_takeover_retires_addition(db):
    from refillcare.engine import pharmacist_overrides as po
    from refillcare.engine.persistence import RefillPersistenceManager

    _seed_ineligible(db, item="DOLO 650MG TAB")  # paracetamol: acute / PRN by the rules
    assert po.review_candidates(db, date(2026, 10, 18), date(2026, 11, 1)) == []
    with pytest.raises(ValueError):
        po.add_to_reminders(db, "C9_I9", today=date(2026, 10, 25))

    db.query(RefillDecisionModel).delete()
    db.commit()
    _seed_ineligible(db)
    po.add_to_reminders(db, "C9_I9", next_refill_date=date(2026, 10, 29), today=date(2026, 10, 25))
    eng = dataclasses_replace(po._decision_from_model(db.query(RefillDecisionModel).get("DEC-C9_I9-20261001")),
                              path="PATH_A", is_eligible=True, expected_refill_date=date(2026, 10, 30), cycle_id="CYC-C9_I9-20261001")
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, [eng])
    pm.sync_reminder_cycles(db, [eng])
    active = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9", ReminderCycleModel.is_active == True).all()  # noqa: E712
    assert [c.cycle_id for c in active] == ["CYC-C9_I9-20261001"]  # only the engine's cycle, no duplicate
    assert po.list_pharmacist_added(db) == []


def test_pharmacist_bulk_add_many_patients_in_one_click(db):
    from refillcare.engine import pharmacist_overrides as po

    _seed_ineligible(db)  # C9_I9 chronic, last 1 Oct, gap 28 -> due 29 Oct
    db.add(RefillDecisionModel(
        decision_id="DEC-C8_I8-20260901", cycle_id="CYC-C8_I8-20260901", customer_id="C8", customer_name="Patient Eight",
        mobile_no="919812345670", item_id="I8", item_name="GLYCOMET 500 TAB", customer_item_key="C8_I8", path="INELIGIBLE",
        purchase_count=2, is_eligible=False, stability_tier="UNSTABLE", prediction_method="NONE", cadence_median=30.0,
        last_purchase_date=date(2026, 9, 1), expected_refill_date=None, decision_reason="Insufficient purchase history",
    ))
    db.add(RefillDecisionModel(
        decision_id="DEC-C7_I7-20261001", cycle_id="CYC-C7_I7-20261001", customer_id="C7", customer_name="Patient Seven",
        mobile_no="919812345671", item_id="I7", item_name="DOLO 650MG TAB", customer_item_key="C7_I7", path="INELIGIBLE",
        purchase_count=2, is_eligible=False, stability_tier="UNSTABLE", prediction_method="NONE", cadence_median=20.0,
        last_purchase_date=date(2026, 10, 1), expected_refill_date=None, decision_reason="Not chronic",
    ))
    db.commit()

    res = po.add_many_to_reminders(db, ["C9_I9", "C8_I8", "C7_I7", "C9_I9"], added_by="tester", today=date(2026, 10, 25))
    assert res["added"] == 2 and res["stages_created"] == 14  # 7 stages each, duplicate key ignored
    assert [f["customer_item_key"] for f in res["failed"]] == ["C7_I7"]  # non-chronic rejected
    added = {r["customer_item_key"]: r for r in po.list_pharmacist_added(db)}
    assert set(added) == {"C9_I9", "C8_I8"}
    due = {c.customer_item_key: c.expected_refill_date for c in db.query(ReminderCycleModel)
           .filter(ReminderCycleModel.is_active == True).all()}  # noqa: E712
    assert due["C9_I9"] == date(2026, 10, 29)  # own gap
    assert due["C8_I8"] == date(2026, 10, 25)  # 1 Oct (past) moved to today


def test_overview_kpis_count_patients_vs_messages_this_month_and_review_today(db):
    from refillcare.engine import pharmacist_overrides as po
    from refillcare.engine.overview_kpis import overview_kpis

    from datetime import datetime as _dt

    from database.models import SalesTransactionModel

    _seed_ineligible(db)  # C9 TELMA, last 1 Oct, gap 28 -> due 29 Oct
    db.add_all([  # customers monitored = patients with a customer sale (B2B lines do not count)
        SalesTransactionModel(import_batch_id="T", invoice_number="S0/1", invoice_date=_dt(2026, 10, 1), customer_id="C9",
                              item_id="I9", quantity=1.0, transaction_type="CUSTOMER_SALE", refillcare_eligible=True),
        SalesTransactionModel(import_batch_id="T", invoice_number="SB/1", invoice_date=_dt(2026, 10, 1), customer_id="BRANCH",
                              item_id="I9", quantity=9.0, transaction_type="B2B_INTER_STORE", refillcare_eligible=False),
    ])
    db.commit()
    today = date(2026, 10, 29)
    k = overview_kpis(db, today)
    assert k["month"] == "2026-10" and k["customers_monitored"] == 1 and k["sales_lines_stored"] == 2
    assert k["latest_sales_date"] == "2026-10-01" and "patients_with_reminders_month" not in k
    assert k["reminder_messages_month"] == 0
    assert k["patients_needing_review_today"] == 1          # due today, not in reminders
    assert overview_kpis(db, date(2026, 10, 30))["patients_needing_review_today"] == 0  # only that day

    po.add_to_reminders(db, "C9_I9", next_refill_date=date(2026, 11, 10), today=today)
    k = overview_kpis(db, date(2026, 11, 1))
    # one patient, 7 stages (-7 .. +45): 6 fall in November, the +45 one in December
    assert k["reminder_messages_month"] == 6
    assert k["patients_needing_review_today"] == 0


def test_send_rule_not_classified_engine_reminder_is_not_sent_but_reviewable_and_addable(db):
    """User decision 9 Nov 2026: send only rules-chronic or pharmacist-added medicines."""
    from refillcare.engine import pharmacist_overrides as po
    from refillcare.engine.persistence import RefillPersistenceManager, medicine_can_be_sent

    item = "ZZQVITA PLUS TAB"  # unknown to the rules -> "Not classified"
    assert po.medicine_class(item) == po.CLASS_UNCLASSIFIED and not medicine_can_be_sent(item)
    assert medicine_can_be_sent(item, "PHARMACIST_ADDED") and medicine_can_be_sent("TELMA 40 TAB")

    _seed_ineligible(db, item=item)
    dec = dataclasses_replace(po._decision_from_model(db.query(RefillDecisionModel).get("DEC-C9_I9-20261001")),
                              path="PATH_B", is_eligible=True, expected_refill_date=date(2026, 10, 29))
    pm = RefillPersistenceManager()
    pm.save_refill_decisions(db, [dec])
    pm.sync_reminder_cycles(db, [dec])
    today = date(2026, 10, 29)
    assert pm.get_today_review_queue(db, target_date=today) == []          # engine cycle exists, but not sent
    cands = po.review_candidates(db, today, today)
    assert [c["customer_item_key"] for c in cands] == ["C9_I9"]
    assert cands[0]["reason_not_in_reminders"] == po.NOT_CLASSIFIED_SCHEDULED_REASON

    res = po.add_to_reminders(db, "C9_I9", next_refill_date=today, today=today)
    assert res["added"] and not res["already_in_reminders"]
    active = db.query(ReminderCycleModel).filter(ReminderCycleModel.customer_item_key == "C9_I9",
                                                 ReminderCycleModel.is_active == True).all()  # noqa: E712
    assert [c.cycle_id[-3:] for c in active] == ["-PH"]                     # one schedule, no double reminders
    assert len(pm.get_today_review_queue(db, target_date=today)) == 1       # now sent (pharmacist-added)
    assert po.review_candidates(db, today, today) == []


def test_business_metrics_reminder_coverage_and_purchase_kinds():
    from refillcare.ml.evaluation import _metrics, purchase_kind, reminder_coverage

    pred = np.array([30.0, 30.0, 30.0, 30.0])
    actual = np.array([30.0, 22.0, 36.0, 50.0])  # on time, 8 d early, 6 d late, 20 d late
    cov = reminder_coverage(pred, actual)
    # d = 0 -> stage 0 same day; d = -8 -> before the -7 stage (missed); d = +6 -> +5 stage 1 day before; d = +20 -> last stage 15 d before
    assert cov["reminder_before_purchase_pct"] == 75.0
    assert cov["reminder_0_7_days_before_pct"] == 50.0
    m = _metrics(pred, actual)
    assert m["within_14_days_pct"] == 75.0 and m["within_7_days_pct"] == 50.0
    assert purchase_kind(70, 30) == "lapse_or_gap" and purchase_kind(10, 30) == "early_top_up"
    assert purchase_kind(28, 30) == "regular" and purchase_kind(28, None) == "unknown"


def test_backtest_report_has_business_view(db):
    reg.ensure_bootstrap_versions(db)
    hist = synthetic_history(n_customers=25)
    variants = {"engine[baseline]": UnifiedRefillDecisionEngine(model_provider=ModelProvider.baseline(), identity_guard=False)}
    rep = run_backtest(hist, variants, date(2026, 6, 1), date(2026, 7, 31), max_events=400)
    bv = rep["business_view"]
    assert bv["method"] == "engine[baseline]" and "reminder_0_7_days_before_pct" in bv["all_refills"]
    assert "purchase_kind" in rep["segments"]


def test_preview_matches_approved_template_with_garu():
    from refillcare.whatsapp.template_formatter import format_full_message_preview, whatsapp_text_to_html

    text = format_full_message_preview("KRISHNA RAO", "DUE", ["TELMA 20MG TAB"], date(2026, 11, 7), stage_offset=0,
                                       store_name="PHARMA HUBB", store_contact="+91 9966473474")
    assert text.startswith("Dear *KRISHNA RAO* Garu,\n\nThis is a friendly reminder from *PHARMA HUBB*")
    assert "Your regular medicine refill is due soon." in text and "💊 Medicines:\nTELMA 20MG TAB" in text
    assert "[WhatsApp Template" not in text and "{{" not in text
    assert "<b>KRISHNA RAO</b> Garu," in whatsapp_text_to_html(text)
    assert "<script>" not in whatsapp_text_to_html("Dear *<script>*")


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
