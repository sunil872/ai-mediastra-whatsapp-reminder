"""Orchestration: run evaluations, apply the promotion gate, compute ML health."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import MLEvaluationRunModel, RefillDecisionModel
from refillcare.ml import registry as reg
from refillcare.ml.evaluation import engine_method_name, run_backtest
from refillcare.ml.model_provider import LoadedModel, ModelProvider, load_version
from refillcare.ml.promotion import evaluate_gate

logger = logging.getLogger("refillcare.ml.service")


def load_history(path: Optional[Path] = None) -> pd.DataFrame:
    """Purchase history of the configured store (``REFILLCARE_DATA_DIR``), or ``path``."""
    from refillcare.store import StoreConfig

    p = Path(path) if path else StoreConfig.from_env().history_parquet
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


MIN_TRAIN_DAYS = 180


def plan_time_splits(
    history_df: pd.DataFrame,
    test_days: Optional[int] = None,
    validation_days: Optional[int] = None,
    censor_buffer_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Rolling train / validation / test windows derived from the latest sale in the data.

    Nothing is hard-coded: when a new month of sales is ingested every window moves forward.

    label_end   = last sale - censor buffer (later purchases mostly have no observed refill yet)
    test        = the last ``test_days`` up to label_end     -> promotion gate only (never used to choose)
    validation  = ``validation_days`` right before test       -> picks the best model
    train       = everything before validation                -> fits the candidate (targets too)
    """
    test_days = test_days or _env_int("ML_TEST_WINDOW_DAYS", 30)
    validation_days = validation_days or _env_int("ML_VALIDATION_WINDOW_DAYS", 30)
    censor = censor_buffer_days if censor_buffer_days is not None else _env_int("ML_CENSOR_BUFFER_DAYS", 45)

    dts = pd.to_datetime(history_df["invoice_date"], errors="coerce").dropna() if "invoice_date" in history_df else pd.Series(dtype="datetime64[ns]")
    if dts.empty:
        raise ValueError("No purchase history available.")
    data_start, data_end = dts.min().date(), dts.max().date()

    label_end = data_end - timedelta(days=censor)
    test_start = label_end - timedelta(days=test_days - 1)
    val_end = test_start - timedelta(days=1)
    val_start = val_end - timedelta(days=validation_days - 1)
    train_end = val_start - timedelta(days=1)
    if (train_end - data_start).days < MIN_TRAIN_DAYS:
        raise ValueError(
            f"Only {(train_end - data_start).days} days of history before the validation window; "
            f"need >= {MIN_TRAIN_DAYS} to train and evaluate out-of-time."
        )

    def _n(a: date, b: date) -> int:
        return int(((dts >= pd.Timestamp(a)) & (dts <= pd.Timestamp(b))).sum())

    return {
        "data_start": data_start,
        "data_end": data_end,
        "censor_buffer_days": censor,
        "train": {"start": data_start, "end": train_end, "purchases": _n(data_start, train_end)},
        "validation": {"start": val_start, "end": val_end, "purchases": _n(val_start, val_end)},
        "test": {"start": test_start, "end": label_end, "purchases": _n(test_start, label_end)},
    }


def default_evaluation_window(history_df: pd.DataFrame) -> Dict[str, date]:
    """The current test window of :func:`plan_time_splits` (moves with every data refresh)."""
    test = plan_time_splits(history_df)["test"]
    return {"start": test["start"], "end": test["end"]}


def data_freshness(db: Session, history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Has new sales data moved past what the serving model was trained on?"""
    hist = history_df if history_df is not None else load_history()
    reg.ensure_bootstrap_versions(db)
    prod = reg.get_by_status(db, reg.FAMILY_PATH_A_INTERVAL, reg.STATUS_PRODUCTION)
    try:
        plan = plan_time_splits(hist)
    except ValueError as e:
        return {"retrain_recommended": False, "reason": str(e)}
    cutoff = prod.train_window_end if prod is not None else None
    lag = (plan["train"]["end"] - cutoff).days if cutoff else None
    threshold = _env_int("ML_RETRAIN_AFTER_DAYS", 30)
    recommended = prod is None or cutoff is None or (lag is not None and lag >= threshold) or not bool(prod.is_validated)
    return {
        "data_end": plan["data_end"],
        "production_version": prod.version_id if prod else None,
        "production_train_cutoff": cutoff,
        "planned_train_cutoff": plan["train"]["end"],
        "days_of_new_training_data": lag,
        "retrain_after_days": threshold,
        "retrain_recommended": bool(recommended),
        "plan": plan,
    }


def _engine_for(loaded: LoadedModel, rules: Any = None):
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine

    return UnifiedRefillDecisionEngine(model_provider=ModelProvider(loaded, None, resolved=True), identity_guard=False, rules=rules)


def compare_rules(
    db: Session,
    version_id: Optional[str] = None,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    max_events: Optional[int] = 3000,
    history_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """A/B every prediction rule on the same unseen events (test window by default).

    Variants: no rules, each rule alone, all rules. Read-only: nothing is stored or promoted.
    Use the result to decide which ``REFILLCARE_RULE_*`` switches to turn on.
    """
    from refillcare.engine.prediction_rules import EngineRules

    reg.ensure_bootstrap_versions(db)
    v = reg.get_version(db, version_id) if version_id else reg.get_by_status(db, reg.FAMILY_PATH_A_INTERVAL, reg.STATUS_PRODUCTION)
    if v is None:
        raise ValueError(f"Model version '{version_id}' not found.")
    loaded = load_version(v)
    if loaded is None:
        raise ValueError(f"Model version '{v.version_id}' has no usable artifact.")
    hist = history_df if history_df is not None else load_history()
    if eval_start is None or eval_end is None:
        win = default_evaluation_window(hist)
        eval_start, eval_end = eval_start or win["start"], eval_end or win["end"]

    off = EngineRules.legacy()
    variants = {
        "rules:none": off,
        "rules:high_stability_median": EngineRules(high_stability_median=True),
        "rules:insulin_median": EngineRules(insulin_median=True),
        "rules:dom_anchor": EngineRules(dom_anchor=True),
        "rules:path_b_shrink": EngineRules(path_b_mode="shrink"),
        "rules:path_b_blend": EngineRules(path_b_mode="blend"),
        "rules:device_cap_observed": EngineRules(device_cap_mode="observed"),
        "rules:blend+device_cap_observed": EngineRules(path_b_mode="blend", device_cap_mode="observed"),
        "rules:all(blend)": EngineRules(high_stability_median=True, insulin_median=True, dom_anchor=True, path_b_mode="blend", device_cap_mode="observed"),
    }
    engines = {name: _engine_for(loaded, rules) for name, rules in variants.items()}
    report = run_backtest(hist, engines, eval_start, eval_end, candidate_method="rules:none", max_events=max_events, seed=seed)
    methods = (report.get("paired_comparison") or {}).get("methods") or {}
    keys = ("within_7_days_pct", "mae_days", "early_pct", "late_pct")
    out = {name: {k: (methods.get(name) or {}).get(k) for k in keys} for name in variants}
    seg_mae = {
        s: {val: {name: (g.get("methods", {}).get(name) or {}).get("mae_days") for name in variants} | {"n": g.get("n")}
            for val, g in (report.get("segments") or {}).get(s, {}).items()}
        for s in ("therapy",)
    }
    base = out["rules:none"].get("within_7_days_pct")
    for name, m in out.items():
        w = m.get("within_7_days_pct")
        m["within_7_gain_pp_vs_none"] = round(w - base, 2) if (w is not None and base is not None) else None
    seg = {
        s: {val: {name: (g.get("methods", {}).get(name) or {}).get("within_7_days_pct") for name in variants} | {"n": g.get("n")}
            for val, g in (report.get("segments") or {}).get(s, {}).items()}
        for s in ("depth", "therapy", "stability")
    }
    return {
        "model_version": v.version_id,
        "window": report.get("evaluation_window"),
        "n_paired_events": (report.get("paired_comparison") or {}).get("n_events"),
        "coverage": {name: (report.get("coverage") or {}).get(name) for name in variants},
        "variants": out,
        "segments_within_7": seg,
        "segments_mae": seg_mae,
        "how_to_apply": "Set REFILLCARE_RULE_HIGH_STABILITY_MEDIAN / _INSULIN_MEDIAN / _DOM_ANCHOR=true, "
                        "REFILLCARE_RULE_PATH_B_MODE=shrink|blend and REFILLCARE_RULE_DEVICE_CAP_MODE=observed "
                        "for rules that gain; restart the app.",
    }


def run_evaluation(
    db: Session,
    version_ids: Optional[List[str]] = None,
    candidate_version_id: Optional[str] = None,
    eval_start: Optional[date] = None,
    eval_end: Optional[date] = None,
    max_events: Optional[int] = 3000,
    seed: int = 42,
    created_by: str = "system",
    history_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> MLEvaluationRunModel:
    """Backtest the given registry versions (default: production + shadow + candidates) and store the report."""
    reg.ensure_bootstrap_versions(db)
    family = reg.FAMILY_PATH_A_INTERVAL
    if not version_ids:
        versions = [v for v in reg.list_versions(db, family) if v.status != reg.STATUS_RETIRED][:5]
    else:
        versions = [v for v in (reg.get_version(db, vid) for vid in version_ids) if v is not None]
    if not versions:
        raise ValueError("No model versions to evaluate.")

    prod = reg.get_by_status(db, family, reg.STATUS_PRODUCTION)
    if candidate_version_id is None:
        non_prod = [v for v in versions if prod is None or v.version_id != prod.version_id]
        candidate_version_id = (non_prod[0] if non_prod else versions[0]).version_id

    run = MLEvaluationRunModel(
        run_id=run_id or f"EVAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        status="RUNNING",
        model_version=candidate_version_id,
        params={"versions": [v.version_id for v in versions], "max_events": max_events, "seed": seed},
        created_by=created_by,
    )
    db.add(run)
    db.commit()

    try:
        hist = history_df if history_df is not None else load_history()
        if eval_start is None or eval_end is None:
            win = default_evaluation_window(hist)
            eval_start, eval_end = eval_start or win["start"], eval_end or win["end"]
        overlapping = [v.version_id for v in versions if v.train_window_end is not None and v.train_window_end >= eval_start]

        variants: Dict[str, Any] = {}
        version_by_method: Dict[str, str] = {}
        for v in versions:
            loaded = load_version(v)
            if loaded is None:
                logger.warning("Skipping version %s: artifact unusable.", v.version_id)
                continue
            name = engine_method_name(v.version_id)
            variants[name] = _engine_for(loaded)
            version_by_method[name] = v.version_id

        cand_method = engine_method_name(candidate_version_id)
        report = run_backtest(
            hist,
            variants,
            eval_start,
            eval_end,
            candidate_method=cand_method if cand_method in variants else None,
            max_events=max_events,
            seed=seed,
            run_id=run.run_id,
        )
        report["candidate_model_version"] = candidate_version_id
        report["production_model_version"] = prod.version_id if prod else None
        report["version_by_method"] = version_by_method
        report["training_overlap_versions"] = overlapping
        if overlapping:
            report["notes"].append(
                f"LEAKAGE WARNING: {overlapping} were trained on data overlapping the evaluation window; "
                "their scores are optimistic and they cannot pass the promotion gate on this run."
            )
        run.report = report
        run.status = "COMPLETED"
        run.completed_at = datetime.utcnow()
        for v in versions:
            if engine_method_name(v.version_id) in variants:
                v.latest_evaluation_run_id = run.run_id
        db.commit()
    except Exception as e:
        db.rollback()
        run = db.get(MLEvaluationRunModel, run.run_id)
        run.status = "FAILED"
        run.error = repr(e)
        run.completed_at = datetime.utcnow()
        db.commit()
        logger.exception("Evaluation %s failed", run.run_id)
    return run


def gate_for_version(db: Session, version_id: str, evaluation_run_id: Optional[str] = None) -> Dict[str, Any]:
    reg.ensure_bootstrap_versions(db)
    v = reg.get_version(db, version_id)
    if v is None:
        raise ValueError(f"Model version '{version_id}' not found.")
    run_id = evaluation_run_id or v.latest_evaluation_run_id
    run = db.get(MLEvaluationRunModel, run_id) if run_id else None
    if run is None or run.status != "COMPLETED" or not run.report:
        return {
            "approved": False,
            "failed_checks": ["evaluation_required"],
            "checks": [{"check": "evaluation_required", "passed": False, "detail": "Run an evaluation that includes this version first."}],
        }
    prod = reg.get_by_status(db, v.model_family, reg.STATUS_PRODUCTION)
    prod_method = engine_method_name(prod.version_id) if prod else None
    return evaluate_gate(
        run.report,
        candidate_method=engine_method_name(version_id),
        candidate_is_baseline=v.model_type == reg.TYPE_BASELINE_MEDIAN,
        production_method=prod_method,
    )


def promote(
    db: Session,
    version_id: str,
    promoted_by: str,
    evaluation_run_id: Optional[str] = None,
    force: bool = False,
    force_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote ``version_id`` to production - only if the evidence gate approves (or a recorded force)."""
    decision = gate_for_version(db, version_id, evaluation_run_id)
    if not decision.get("approved"):
        if not force:
            return {"promoted": False, "decision": decision}
        if not force_reason:
            raise ValueError("A force promotion requires a written reason.")
        decision = {**decision, "forced": True, "force_reason": force_reason}
    promoted = reg.promote_version(db, version_id, promoted_by, decision)
    return {"promoted": True, "version": reg.to_dict(promoted, db), "decision": decision}


def _within7(run: MLEvaluationRunModel, version_id: str) -> Optional[float]:
    methods = ((run.report or {}).get("paired_comparison") or {}).get("methods") or {}
    m = methods.get(engine_method_name(version_id)) or {}
    return m.get("within_7_days_pct")


def _method_summary(run: MLEvaluationRunModel, version_ids: List[str]) -> Dict[str, Any]:
    methods = ((run.report or {}).get("paired_comparison") or {}).get("methods") or {}
    keys = ("within_7_days_pct", "within_7_days_ci95", "mae_days", "early_pct", "late_pct")
    out = {vid: {k: (methods.get(engine_method_name(vid)) or {}).get(k) for k in keys} for vid in version_ids}
    return {"run_id": run.run_id, "status": run.status, "error": run.error,
            "window": (run.report or {}).get("evaluation_window"),
            "n_paired_events": ((run.report or {}).get("paired_comparison") or {}).get("n_events"),
            "versions": out}


def _model_for_cutoff(db: Session, hist: pd.DataFrame, train_end: date, max_train_events: int, created_by: str):
    """The HGB candidate trained up to ``train_end`` (reused if it exists, otherwise trained now)."""
    from refillcare.ml.training import train_interval_model

    existing = next(
        (v for v in reg.list_versions(db, reg.FAMILY_PATH_A_INTERVAL)
         if v.model_type == reg.TYPE_HIST_GB and v.train_window_end == train_end and v.status != reg.STATUS_RETIRED),
        None,
    )
    if existing is not None:
        return existing, False
    res = train_interval_model(db, hist, train_end=train_end, max_events=max_train_events, created_by=created_by)
    v = reg.get_version(db, res["version"]["version_id"])
    if v is None:
        raise RuntimeError("Training did not register a candidate version.")
    return v, True


def freshness_experiment(
    db: Session,
    history_df: Optional[pd.DataFrame] = None,
    step_days: Optional[int] = None,
    created_by: str = "system",
    max_eval_events: Optional[int] = 3000,
    max_train_events: int = 150000,
) -> Dict[str, Any]:
    """Does retraining on fresher data help? Two models, identical except for the training cutoff.

    Both cutoffs and the scoring window come from :func:`plan_time_splits`, so the experiment moves
    with every new month: the *fresh* model is trained up to the day before the test window, the
    *older* one ``step_days`` earlier; both are scored with production and the baseline on the same
    unseen test window.
    """
    hist = history_df if history_df is not None else load_history()
    plan = plan_time_splits(hist)
    step = step_days or _env_int("ML_FRESHNESS_STEP_DAYS", 30)
    fresh_end = plan["test"]["start"] - timedelta(days=1)
    older_end = fresh_end - timedelta(days=step)
    if (older_end - plan["data_start"]).days < MIN_TRAIN_DAYS:
        raise ValueError("Not enough history for an older training cutoff.")

    reg.ensure_bootstrap_versions(db)
    older, older_new = _model_for_cutoff(db, hist, older_end, max_train_events, created_by)
    fresh, fresh_new = _model_for_cutoff(db, hist, fresh_end, max_train_events, created_by)
    prod = reg.get_by_status(db, reg.FAMILY_PATH_A_INTERVAL, reg.STATUS_PRODUCTION)
    ids: List[str] = []
    for v in (prod, reg.get_version(db, reg.BASELINE_VERSION_ID), older, fresh):
        if v is not None and v.version_id not in ids:
            ids.append(v.version_id)

    run = run_evaluation(db, version_ids=ids, candidate_version_id=fresh.version_id,
                         eval_start=plan["test"]["start"], eval_end=plan["test"]["end"],
                         max_events=max_eval_events, created_by=created_by, history_df=hist,
                         run_id=f"FRESH-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    if run.status != "COMPLETED":
        raise RuntimeError(f"Freshness evaluation failed: {run.error}")
    w_old, w_new = _within7(run, older.version_id), _within7(run, fresh.version_id)
    gain = round(w_new - w_old, 2) if w_old is not None and w_new is not None else None
    return {
        "test_window": {"start": plan["test"]["start"], "end": plan["test"]["end"]},
        "older_model": {"version": older.version_id, "train_cutoff": older_end, "trained_now": older_new},
        "fresh_model": {"version": fresh.version_id, "train_cutoff": fresh_end, "trained_now": fresh_new},
        "results": _method_summary(run, ids),
        "fresh_minus_older_within_7_pp": gain,
        "fresher_data_helps": bool(gain is not None and gain > 0),
        "gate_for_fresh_model": gate_for_version(db, fresh.version_id, run.run_id),
    }


def rolling_folds(history_df: pd.DataFrame, n_folds: Optional[int] = None) -> List[Dict[str, date]]:
    """Back-to-back test months ending at the planned test window (newest first).

    Each fold keeps the same gap as :func:`plan_time_splits`: its model is trained only up to the day
    before a validation-sized gap ahead of the fold, so no fold is ever scored by a model that saw it.
    """
    n_folds = n_folds or _env_int("ML_ROLLING_FOLDS", 3)
    plan = plan_time_splits(history_df)
    test_days = (plan["test"]["end"] - plan["test"]["start"]).days + 1
    gap_days = (plan["validation"]["end"] - plan["validation"]["start"]).days + 1
    folds: List[Dict[str, date]] = []
    for k in range(n_folds):
        end = plan["test"]["end"] - timedelta(days=k * test_days)
        start = end - timedelta(days=test_days - 1)
        train_end = start - timedelta(days=gap_days + 1)
        if (train_end - plan["data_start"]).days < MIN_TRAIN_DAYS:
            break
        folds.append({"start": start, "end": end, "train_end": train_end})
    if not folds:
        raise ValueError("Not enough history for a rolling evaluation.")
    return folds


def rolling_evaluation(
    db: Session,
    history_df: Optional[pd.DataFrame] = None,
    n_folds: Optional[int] = None,
    created_by: str = "system",
    max_eval_events: Optional[int] = 3000,
    max_train_events: int = 150000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Is a gain real or one month's noise? Score the same recipe on several unseen months.

    Per fold: the HGB model trained up to that fold's cutoff (reused if it exists), scored with the
    current rules (``REFILLCARE_RULE_*``) and with no rules, against the baseline engine and the raw
    personal median. Nothing is promoted; fold models are registered as candidates like any other
    training run.
    """
    from refillcare.engine.prediction_rules import EngineRules
    from refillcare.ml.evaluation import BASELINE_METHOD

    hist = history_df if history_df is not None else load_history()
    folds = rolling_folds(hist, n_folds)
    reg.ensure_bootstrap_versions(db)
    baseline_loaded = load_version(reg.get_version(db, reg.BASELINE_VERSION_ID))
    rules_now = EngineRules.from_env()
    methods = {
        "raw_median": BASELINE_METHOD,
        "baseline_engine": "engine[baseline]",
        "model_no_rules": "engine[model:no_rules]",
        "model_current_rules": "engine[model:current_rules]",
    }
    keys = ("within_7_days_pct", "within_7_days_ci95", "mae_days", "early_pct", "late_pct")

    fold_results: List[Dict[str, Any]] = []
    for f in folds:
        model, trained_now = _model_for_cutoff(db, hist, f["train_end"], max_train_events, created_by)
        loaded = load_version(model)
        if loaded is None:
            raise RuntimeError(f"Model {model.version_id} has no usable artifact.")
        engines = {
            methods["baseline_engine"]: _engine_for(baseline_loaded, rules_now),
            methods["model_no_rules"]: _engine_for(loaded, EngineRules.legacy()),
            methods["model_current_rules"]: _engine_for(loaded, rules_now),
        }
        report = run_backtest(hist, engines, f["start"], f["end"], candidate_method=methods["model_current_rules"],
                              max_events=max_eval_events, seed=seed)
        pc = report.get("paired_comparison") or {}
        got = pc.get("methods") or {}
        insulin = (((report.get("segments") or {}).get("therapy") or {}).get("insulin") or {}).get("methods") or {}
        fold_results.append({
            "window": {"start": f["start"], "end": f["end"]},
            "model_version": model.version_id,
            "model_train_cutoff": f["train_end"],
            "model_trained_now": trained_now,
            "n_paired_events": pc.get("n_events"),
            "methods": {label: {k: (got.get(m) or {}).get(k) for k in keys} for label, m in methods.items()},
            "insulin_mae": {label: (insulin.get(m) or {}).get("mae_days") for label, m in methods.items()},
        })

    def _pooled(label: str, key: str) -> Optional[float]:
        pairs = [(r["methods"][label].get(key), r["n_paired_events"] or 0) for r in fold_results]
        pairs = [(v, n) for v, n in pairs if v is not None and n]
        total = sum(n for _, n in pairs)
        return round(sum(v * n for v, n in pairs) / total, 2) if total else None

    def _gains(a: str, b: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for r in fold_results:
            wa, wb = r["methods"][a].get("within_7_days_pct"), r["methods"][b].get("within_7_days_pct")
            out.append(round(wa - wb, 2) if wa is not None and wb is not None else None)
        return out

    pooled = {label: {k: _pooled(label, k) for k in ("within_7_days_pct", "mae_days", "early_pct", "late_pct")}
              for label in methods}
    model_vs_baseline = _gains("model_current_rules", "baseline_engine")
    rules_vs_none = _gains("model_current_rules", "model_no_rules")
    return {
        "folds": fold_results,
        "n_folds": len(fold_results),
        "n_paired_events_total": sum(r["n_paired_events"] or 0 for r in fold_results),
        "current_rules": rules_now.describe(),
        "pooled": pooled,
        "model_vs_baseline_within_7_pp_by_fold": model_vs_baseline,
        "rules_vs_no_rules_within_7_pp_by_fold": rules_vs_none,
        "model_beats_baseline_every_fold": all(g is not None and g > 0 for g in model_vs_baseline),
        "rules_help_every_fold": all(g is not None and g >= 0 for g in rules_vs_none),
    }


def retrain_cycle(
    db: Session,
    history_df: Optional[pd.DataFrame] = None,
    promote_best: bool = False,
    created_by: str = "system",
    max_eval_events: Optional[int] = 3000,
    max_train_events: int = 150000,
) -> Dict[str, Any]:
    """Re-plan the splits from the current data, retrain, select on validation, gate on test.

    1. train windows come from :func:`plan_time_splits` (they move with every new month of data)
    2. fit a candidate on the train window (reused if one already exists for that cutoff)
    3. evaluate production + baseline + candidate on the validation window -> pick the best
    4. evaluate them again on the untouched test window -> promotion gate for the pick
    5. promote only when ``promote_best`` and the gate approves
    """
    hist = history_df if history_df is not None else load_history()
    plan = plan_time_splits(hist)
    reg.ensure_bootstrap_versions(db)
    family = reg.FAMILY_PATH_A_INTERVAL
    train_end = plan["train"]["end"]

    candidate, trained_now = _model_for_cutoff(db, hist, train_end, max_train_events, created_by)

    prod = reg.get_by_status(db, family, reg.STATUS_PRODUCTION)
    ids: List[str] = []
    for v in (prod, reg.get_version(db, reg.BASELINE_VERSION_ID), candidate):
        if v is not None and v.version_id not in ids:
            ids.append(v.version_id)

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    val_run = run_evaluation(db, version_ids=ids, candidate_version_id=candidate.version_id,
                             eval_start=plan["validation"]["start"], eval_end=plan["validation"]["end"],
                             max_events=max_eval_events, created_by=created_by, history_df=hist, run_id=f"VAL-{stamp}")
    if val_run.status != "COMPLETED":
        raise RuntimeError(f"Validation evaluation failed: {val_run.error}")

    # Model selection on validation only (the test window stays untouched until the gate).
    challengers = [vid for vid in ids if prod is None or vid != prod.version_id]
    best = max(challengers, key=lambda vid: _within7(val_run, vid) or -1.0)

    test_run = run_evaluation(db, version_ids=ids, candidate_version_id=best,
                              eval_start=plan["test"]["start"], eval_end=plan["test"]["end"],
                              max_events=max_eval_events, created_by=created_by, history_df=hist, run_id=f"TEST-{stamp}")
    if test_run.status != "COMPLETED":
        raise RuntimeError(f"Test evaluation failed: {test_run.error}")

    gate = gate_for_version(db, best, test_run.run_id)
    result: Dict[str, Any] = {
        "plan": plan,
        "candidate_version": candidate.version_id,
        "candidate_trained_now": trained_now,
        "production_version": prod.version_id if prod else None,
        "validation": _method_summary(val_run, ids),
        "selected_on_validation": best,
        "test": _method_summary(test_run, ids),
        "gate": gate,
        "promoted": False,
    }
    if promote_best and gate.get("approved"):
        result["promotion"] = promote(db, best, promoted_by=created_by, evaluation_run_id=test_run.run_id)
        result["promoted"] = bool(result["promotion"].get("promoted"))
    return result


def ml_health(db: Session, since_days: int = 30) -> Dict[str, Any]:
    """How predictions were actually produced recently (fallback rate = silent-degradation alarm)."""
    from refillcare.models.supply_hybrid import FALLBACK_COUNTERS

    since = datetime.utcnow() - timedelta(days=since_days)
    rows = (
        db.query(RefillDecisionModel.prediction_source, RefillDecisionModel.model_version, func.count())
        .filter(RefillDecisionModel.is_eligible == True, RefillDecisionModel.updated_at >= since)  # noqa: E712
        .group_by(RefillDecisionModel.prediction_source, RefillDecisionModel.model_version)
        .all()
    )
    by_source: Dict[str, int] = {}
    by_version: Dict[str, int] = {}
    for src, ver, cnt in rows:
        by_source[str(src or "UNRECORDED")] = by_source.get(str(src or "UNRECORDED"), 0) + int(cnt)
        by_version[str(ver or "UNRECORDED")] = by_version.get(str(ver or "UNRECORDED"), 0) + int(cnt)
    path_a_total = sum(v for k, v in by_source.items() if k in ("ML", "ML_CLAMPED", "HEURISTIC_FALLBACK", "BASELINE_MEDIAN", "BASELINE_CLAMPED"))
    fallback = by_source.get("HEURISTIC_FALLBACK", 0)
    return {
        "window_days": since_days,
        "serving": ModelProvider.from_registry(db).describe(),
        "eligible_decisions_by_source": by_source,
        "eligible_decisions_by_model_version": by_version,
        "ml_fallback_rate_pct": round(100.0 * fallback / path_a_total, 2) if path_a_total else None,
        "process_fallback_counters": dict(FALLBACK_COUNTERS),
        "alert": bool(path_a_total and fallback / path_a_total > 0.05),
        "data_freshness": data_freshness(db),
    }
