"""READ-ONLY Phase 17D reminder-system business metrics.

Evaluates best Phase 17D prediction strategies from a reminder operations view:
date error, ±3/±7 accuracy, opportunity coverage, exclusions, early/late risk,
and prediction failures — by customer segment.

Does NOT send WhatsApp messages. Does NOT modify production artifacts.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from refillcare.models.training import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
from refillcare.evaluation.baseline import HistoricalMedianBaseline

# Reuse hybrid helpers / thresholds (load by path — scripts/ is not a package)
import importlib.util

_hybrid_path = ROOT / "scripts" / "phase17d_hybrid_comparison.py"
_spec = importlib.util.spec_from_file_location("phase17d_hybrid_comparison", _hybrid_path)
_hybrid = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hybrid)

HYBRID = _hybrid.HYBRID
MODEL_PATH = _hybrid.MODEL_PATH
REPORT_PATH = _hybrid.REPORT_PATH
PROC = _hybrid.PROC
_attach_regularity_from_history = _hybrid._attach_regularity_from_history
_build_preprocessor = _hybrid._build_preprocessor
_tiered_cadence_predict = _hybrid._tiered_cadence_predict
_hybrid_route = _hybrid._hybrid_route

# Failure thresholds (reminder ops)
FAIL_SOFT_DAYS = 7.0   # outside ±7d operational miss
FAIL_HARD_DAYS = 14.0  # severe date miss


def _biz_metrics(y_true, y_pred) -> dict:
    """Reminder-oriented metrics on scored rows only."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t, y_p = y_t[mask], y_p[mask]
    n = int(len(y_t))
    if n == 0:
        return {
            "scored_events": 0,
            "predicted_refill_date_error_mae": None,
            "predicted_refill_date_error_medae": None,
            "within_3_days_pct": None,
            "within_7_days_pct": None,
            "potential_early_reminders_pct": None,
            "potential_late_reminders_pct": None,
            "potential_early_count": 0,
            "potential_late_count": 0,
            "mean_days_early_when_early": None,
            "mean_days_late_when_late": None,
            "prediction_failures_outside_pm7_pct": None,
            "prediction_failures_outside_pm14_pct": None,
            "prediction_failures_outside_pm7_count": 0,
            "prediction_failures_outside_pm14_count": 0,
            "on_time_exact_day_pct": None,
        }

    # signed: pred - actual. Negative => predicted date earlier than true refill (early reminders)
    signed = y_p - y_t
    abs_err = np.abs(signed)
    early = signed < 0
    late = signed > 0
    on_time = signed == 0

    early_mag = -signed[early] if early.any() else np.array([])
    late_mag = signed[late] if late.any() else np.array([])

    return {
        "scored_events": n,
        "predicted_refill_date_error_mae": round(float(np.mean(abs_err)), 2),
        "predicted_refill_date_error_medae": round(float(np.median(abs_err)), 2),
        "within_3_days_pct": round(float((abs_err <= 3).mean() * 100), 2),
        "within_7_days_pct": round(float((abs_err <= 7).mean() * 100), 2),
        "on_time_exact_day_pct": round(float(on_time.mean() * 100), 2),
        "potential_early_reminders_pct": round(float(early.mean() * 100), 2),
        "potential_late_reminders_pct": round(float(late.mean() * 100), 2),
        "potential_early_count": int(early.sum()),
        "potential_late_count": int(late.sum()),
        "mean_days_early_when_early": (
            round(float(early_mag.mean()), 2) if len(early_mag) else None
        ),
        "mean_days_late_when_late": (
            round(float(late_mag.mean()), 2) if len(late_mag) else None
        ),
        "median_days_early_when_early": (
            round(float(np.median(early_mag)), 2) if len(early_mag) else None
        ),
        "median_days_late_when_late": (
            round(float(np.median(late_mag)), 2) if len(late_mag) else None
        ),
        "prediction_failures_outside_pm7_pct": round(
            float((abs_err > FAIL_SOFT_DAYS).mean() * 100), 2
        ),
        "prediction_failures_outside_pm14_pct": round(
            float((abs_err > FAIL_HARD_DAYS).mean() * 100), 2
        ),
        "prediction_failures_outside_pm7_count": int((abs_err > FAIL_SOFT_DAYS).sum()),
        "prediction_failures_outside_pm14_count": int((abs_err > FAIL_HARD_DAYS).sum()),
    }


def _coverage_block(df: pd.DataFrame, pred: np.ndarray, universe_events: int) -> dict:
    scored = ~np.isnan(pred)
    cust = df["customerId"].astype(str)
    item = df["itemId"].astype(str)
    pairs = cust + "|" + item
    all_customers = cust.nunique()
    all_pairs = pairs.nunique()
    scored_customers = cust[scored].nunique()
    scored_pairs = pairs[scored].nunique()
    excl_customers = all_customers - scored_customers
    excl_pairs = all_pairs - scored_pairs
    return {
        "event_universe": universe_events,
        "reminder_opportunity_events": int(scored.sum()),
        "reminder_opportunity_coverage_pct": round(100.0 * scored.sum() / universe_events, 2),
        "events_excluded_by_eligibility": int((~scored).sum()),
        "events_excluded_pct": round(100.0 * (~scored).sum() / universe_events, 2),
        "unique_customers_universe": int(all_customers),
        "unique_customers_with_opportunity": int(scored_customers),
        "customers_excluded_by_eligibility": int(excl_customers),
        "customers_excluded_pct": round(100.0 * excl_customers / all_customers, 2) if all_customers else None,
        "unique_customer_item_pairs_universe": int(all_pairs),
        "unique_pairs_with_opportunity": int(scored_pairs),
        "pairs_excluded_by_eligibility": int(excl_pairs),
        "pairs_excluded_pct": round(100.0 * excl_pairs / all_pairs, 2) if all_pairs else None,
        "prediction_failures_no_emit": int((~scored).sum()),
    }


def _segment_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    p = df["purchase_count_so_far"].to_numpy()
    norm_mad = df["norm_mad"].to_numpy(dtype=np.float64)
    drift = df["cadence_drift"].to_numpy(dtype=np.float64)
    hist_med = df["historical_interval_median"].to_numpy(dtype=np.float64)
    recurring = df["is_recurring_history"].to_numpy() == 1

    depth_ok = p >= HYBRID["min_purchases"]
    cadence_ok = (
        ~np.isnan(hist_med)
        & (hist_med >= HYBRID["cadence_median_min"])
        & (hist_med <= HYBRID["cadence_median_max"])
    )
    broad_ok = (
        depth_ok
        & cadence_ok
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & (norm_mad <= HYBRID["broad_norm_mad_max"])
        & (drift <= HYBRID["broad_drift_max"])
    )
    regular = (
        (p >= 6)
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & (norm_mad <= HYBRID["core_norm_mad_max"])
        & (drift <= HYBRID["core_drift_max"])
    )
    # Irregular among those with enough history to measure dispersion
    irregular = (
        (p >= 6)
        & ~np.isnan(norm_mad)
        & ~np.isnan(drift)
        & ((norm_mad > HYBRID["core_norm_mad_max"]) | (drift > HYBRID["core_drift_max"]))
    )
    low_history = p <= 3
    eligible_recurring = broad_ok & recurring  # hybrid-eligible recurring band

    return {
        "eligible_recurring_customers": eligible_recurring,
        "low_history_customers": low_history,
        "regular_customers": regular,
        "irregular_customers": irregular,
        "all_test_events": np.ones(len(df), dtype=bool),
    }


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_before = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    keep = list(
        dict.fromkeys(
            feature_cols
            + [
                TARGET_COL,
                "invoice_date",
                "purchase_count_so_far",
                "customerId",
                "itemId",
                "is_recurring_history",
                "historical_interval_median",
            ]
        )
    )

    print("Loading partitions (read-only)...", flush=True)
    train = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    train = train[[c for c in keep if c in train.columns]]
    test = test[[c for c in keep if c in test.columns]]
    train = train[train[TARGET_COL] > 0].copy()
    test = test[test[TARGET_COL] > 0].copy()
    n = len(test)
    print(f"Test events (target>0): {n:,}", flush=True)

    print("Attaching regularity features...", flush=True)
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _attach_regularity_from_history(test, history)
    del history
    gc.collect()

    baseline = HistoricalMedianBaseline()
    baseline.fit(train, target_col=TARGET_COL)
    hist_pred = baseline.predict(test)
    cadence_pred = _tiered_cadence_predict(test, baseline.fallback_median)

    print("Scoring production XGBoost...", flush=True)
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feats_prod = bundle.get("features_numeric", NUMERIC_FEATURES) + bundle.get(
        "features_categorical", CATEGORICAL_FEATURES
    )
    X_prod = test.copy()
    for c in feats_prod:
        if c not in X_prod.columns:
            X_prod[c] = np.nan
    current_xgb = np.clip(pipe.predict(X_prod[feats_prod]), 1.0, None)
    del bundle, pipe, X_prod
    gc.collect()

    print("Training in-memory experimental quantile XGB...", flush=True)
    feats = [c for c in feature_cols if c in train.columns]
    pre, _, _ = _build_preprocessor(feats)
    X_train = pre.fit_transform(train[feats])
    X_test = pre.transform(test[feats])
    t0 = time.time()
    exp = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        tree_method="hist",
    )
    exp.fit(X_train, train[TARGET_COL].to_numpy(dtype=np.float64))
    exp_sec = round(time.time() - t0, 2)
    exp_xgb = np.clip(exp.predict(X_test), 1.0, None)
    del exp, X_train, X_test, pre
    gc.collect()

    hybrid_pred, eligible, route_meta = _hybrid_route(test, hist_pred, exp_xgb)
    y_true = test[TARGET_COL].to_numpy(dtype=np.float64)
    segments = _segment_masks(test)

    strategies = {
        "current_xgboost": {
            "label": "Current production XGBoost",
            "pred": current_xgb,
            "uses_eligibility_gate": False,
        },
        "personal_historical_median": {
            "label": "Personal historical median",
            "pred": hist_pred,
            "uses_eligibility_gate": False,
        },
        "best_personal_cadence": {
            "label": "Best personal-cadence (tiered)",
            "pred": cadence_pred,
            "uses_eligibility_gate": False,
        },
        "best_experimental_xgboost": {
            "label": "Best experimental XGBoost (quantile q50)",
            "pred": exp_xgb,
            "uses_eligibility_gate": False,
        },
        "hybrid_routing": {
            "label": "Hybrid routing (eligible only)",
            "pred": hybrid_pred,
            "uses_eligibility_gate": True,
        },
    }

    # Segment sizes
    segment_sizes = {}
    for name, mask in segments.items():
        cust = test.loc[mask, "customerId"].astype(str).nunique()
        segment_sizes[name] = {
            "events": int(mask.sum()),
            "events_pct": round(100.0 * mask.sum() / n, 2),
            "unique_customers": int(cust),
        }

    results = {
        "experiment": "PHASE_17D_BUSINESS_METRICS",
        "read_only": True,
        "whatsapp_messages_sent": 0,
        "definitions": {
            "predicted_refill_date_error": "|predicted_days - actual_days| (calendar date offset error)",
            "potential_early_reminders": "predicted interval < actual (pred date before true next purchase)",
            "potential_late_reminders": "predicted interval > actual (pred date after true next purchase)",
            "prediction_failures_outside_pm7": "|error| > 7 days (misses ±7 reminder window)",
            "prediction_failures_outside_pm14": "|error| > 14 days (severe miss)",
            "prediction_failures_no_emit": "no automated prediction emitted (eligibility reject)",
            "reminder_opportunity_coverage": "% of test events that would receive an automated prediction/reminder opportunity",
            "segments": {
                "eligible_recurring_customers": (
                    "P>=6 & NormMAD<=0.50 & Drift<=10 & hist_median in [15,120] "
                    "& is_recurring_history=1 (hybrid-eligible recurring)"
                ),
                "low_history_customers": "purchase_count_so_far <= 3",
                "regular_customers": "P>=6 & NormMAD<=0.35 & Drift<=7 (Class 1)",
                "irregular_customers": "P>=6 & (NormMAD>0.35 OR Drift>7) (Class 2-like)",
            },
        },
        "controls": {
            "test_events": n,
            "test_date_min": str(pd.to_datetime(test["invoice_date"]).min().date()),
            "test_date_max": str(pd.to_datetime(test["invoice_date"]).max().date()),
            "experimental_xgb_train_sec": exp_sec,
            "hybrid_thresholds": HYBRID,
            "fail_soft_days": FAIL_SOFT_DAYS,
            "fail_hard_days": FAIL_HARD_DAYS,
        },
        "segment_sizes": segment_sizes,
        "strategies": {},
    }

    for key, spec in strategies.items():
        pred = np.asarray(spec["pred"], dtype=np.float64)
        cov = _coverage_block(test, pred, n)
        overall = _biz_metrics(y_true, pred)
        # Universe-level failure rate includes non-emits for gated strategies
        no_emit = int(np.isnan(pred).sum())
        scored = ~np.isnan(pred)
        abs_all = np.full(n, np.nan)
        abs_all[scored] = np.abs(pred[scored] - y_true[scored])
        fail7_scored = int(np.nansum(abs_all > FAIL_SOFT_DAYS))
        # For always-on, no_emit=0; for hybrid, failures = no_emit + outside ±7 among scored
        seg_out = {}
        for seg_name, mask in segments.items():
            # Metrics on emitted predictions within the segment
            m = _biz_metrics(y_true[mask], pred[mask])
            cov_seg = {
                "segment_events": int(mask.sum()),
                "scored_in_segment": int((mask & scored).sum()),
                "opportunity_coverage_within_segment_pct": (
                    round(100.0 * (mask & scored).sum() / mask.sum(), 2) if mask.sum() else None
                ),
                "excluded_within_segment": int((mask & ~scored).sum()),
            }
            seg_out[seg_name] = {**cov_seg, **m}

        entry = {
            "label": spec["label"],
            "uses_eligibility_gate": spec["uses_eligibility_gate"],
            "coverage_and_exclusion": cov,
            "overall_on_emitted_predictions": overall,
            "universe_prediction_failure_summary": {
                "no_prediction_emitted": no_emit,
                "no_prediction_emitted_pct": round(100.0 * no_emit / n, 2),
                "outside_pm7_among_emitted": fail7_scored,
                "outside_pm7_among_emitted_pct_of_universe": round(100.0 * fail7_scored / n, 2),
                "total_ops_miss_no_emit_or_outside_pm7": no_emit + fail7_scored,
                "total_ops_miss_pct_of_universe": round(
                    100.0 * (no_emit + fail7_scored) / n, 2
                ),
            },
            "by_segment": seg_out,
        }
        if key == "hybrid_routing":
            entry["routing"] = route_meta["route_counts"]
        results["strategies"][key] = entry
        print(
            f"{key}: cov={cov['reminder_opportunity_coverage_pct']}% "
            f"MAE={overall['predicted_refill_date_error_mae']} "
            f"±7={overall['within_7_days_pct']}% "
            f"early={overall['potential_early_reminders_pct']}% "
            f"late={overall['potential_late_reminders_pct']}% "
            f"fail>7={overall['prediction_failures_outside_pm7_pct']}%",
            flush=True,
        )

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    report_mtime_after = REPORT_PATH.stat().st_mtime if REPORT_PATH.exists() else None
    results["artifact_safety"] = {
        "refill_model_joblib_mtime_unchanged": model_mtime_before == model_mtime_after,
        "phase4_report_mtime_unchanged": report_mtime_before == report_mtime_after,
        "whatsapp_messages_sent": 0,
        "production_code_modified": False,
    }

    out = PROC / "phase17d_business_metrics_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print("Artifact safety:", results["artifact_safety"], flush=True)


if __name__ == "__main__":
    main()
