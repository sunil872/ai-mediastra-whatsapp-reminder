"""READ-ONLY historical backtest: Existing Hybrid vs Consumption Days-of-Supply.

Compares hybrid_routing_v17d against Historical Consumption + Estimated Days of Supply
on supervised (customerId, itemId) purchase events.

Constraints:
- No production routing changes
- No model retrain / overwrite
- No WhatsApp sends
- Predictions use only information available at/before the current purchase
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refillcare.data.packing import enrich_total_units_purchased
from refillcare.features.consumption import (
    calculate_estimated_days_of_supply,
    compute_expanding_consumption_rates,
)
from refillcare.models.hybrid_strategy import (
    ROUTE_CORE,
    ROUTE_SECONDARY,
    evaluate_hybrid_eligibility,
    personal_median_prediction,
)
from refillcare.models.training import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"
OUT_JSON = ROOT / "docs" / "backtest_results.json"
LARGE_ERROR_DAYS = 30.0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[mask], y_p[mask]
    n = int(len(y_t))
    if n == 0:
        return {
            "label": label,
            "n": 0,
            "mae": None,
            "rmse": None,
            "median_error": None,
            "within_3d_pct": None,
            "within_7d_pct": None,
            "large_error_gt30d": 0,
            "large_error_pct": None,
        }
    err = y_p - y_t
    abs_err = np.abs(err)
    large = int((abs_err > LARGE_ERROR_DAYS).sum())
    return {
        "label": label,
        "n": n,
        "mae": round(float(np.mean(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "median_error": round(float(np.median(abs_err)), 2),
        "within_3d_pct": round(float((abs_err <= 3).mean() * 100), 2),
        "within_7d_pct": round(float((abs_err <= 7).mean() * 100), 2),
        "large_error_gt30d": large,
        "large_error_pct": round(100.0 * large / n, 2),
    }


def _model_predict_days(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    pipeline = bundle["pipeline"]
    num_cols = list(bundle.get("features_numeric", NUMERIC_FEATURES))
    cat_cols = list(bundle.get("features_categorical", CATEGORICAL_FEATURES))
    feature_cols = num_cols + cat_cols
    work = df.copy()
    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
    raw = np.asarray(pipeline.predict(work[feature_cols]), dtype=np.float64)
    return np.round(np.clip(raw, 1.0, None), 1)


def main() -> None:
    model_mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None

    print("Loading supervised training dataset (read-only)...", flush=True)
    ds = pd.read_parquet(PROC / "training_dataset.parquet")
    df = ds[ds["is_supervised_eligible"] & (ds["target_days_until_next_purchase"] > 0)].copy()
    total_supervised = len(df)
    print(f"  supervised rows with target>0: {total_supervised:,}", flush=True)

    print("Enriching pack units...", flush=True)
    df = enrich_total_units_purchased(df)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.sort_values(
        ["customerId", "itemId", "invoice_date", "purchase_seq"]
    ).reset_index(drop=True)

    print("Computing expanding historical consumption rates (leakage-safe)...", flush=True)
    df["historical_consumption_rate"] = compute_expanding_consumption_rates(df)

    print("Computing Estimated Days of Supply predictions...", flush=True)
    units_arr = pd.to_numeric(df["total_units_purchased"], errors="coerce").to_numpy(dtype=np.float64)
    rate_arr = pd.to_numeric(df["historical_consumption_rate"], errors="coerce").to_numpy(dtype=np.float64)
    eds_days = np.full(len(df), np.nan, dtype=np.float64)
    ok = np.isfinite(units_arr) & (units_arr >= 0) & np.isfinite(rate_arr) & (rate_arr > 0)
    eds_days[ok] = np.round(units_arr[ok] / rate_arr[ok], 4)
    df["pred_consumption_dos"] = eds_days

    print("Scoring existing hybrid_routing_v17d (read-only model for MEDIUM)...", flush=True)
    bundle = joblib.load(MODEL_PATH)

    existing_pred = np.full(len(df), np.nan, dtype=np.float64)
    existing_eligible = np.zeros(len(df), dtype=bool)
    is_core = np.zeros(len(df), dtype=bool)
    routes: list[str] = []

    print("  batch XGB score (used only for secondary route)...", flush=True)
    xgb_all = _model_predict_days(bundle, df)

    print("  applying hybrid eligibility (leakage-safe feature columns)...", flush=True)
    records = df.to_dict("records")
    for i, rec in enumerate(records):
        decision = evaluate_hybrid_eligibility(rec)
        routes.append(decision["route"])
        if not decision["is_eligible"]:
            continue
        existing_eligible[i] = True
        is_core[i] = bool(decision["is_core_regular"])
        if decision["route"] == ROUTE_CORE:
            try:
                existing_pred[i] = max(1.0, float(personal_median_prediction(rec)))
            except ValueError:
                existing_eligible[i] = False
                existing_pred[i] = np.nan
        elif decision["route"] == ROUTE_SECONDARY:
            existing_pred[i] = float(xgb_all[i])
        else:
            existing_eligible[i] = False
            existing_pred[i] = np.nan

    df["pred_existing"] = existing_pred
    df["existing_eligible"] = existing_eligible
    df["is_core_regular"] = is_core
    df["hybrid_route"] = routes

    y = df["target_days_until_next_purchase"].to_numpy(dtype=np.float64)
    cons_valid = np.isfinite(df["pred_consumption_dos"].to_numpy())
    exist_valid = np.isfinite(df["pred_existing"].to_numpy()) & df["existing_eligible"].to_numpy()
    both = exist_valid & cons_valid

    print(
        f"  existing eligible: {int(exist_valid.sum()):,} | "
        f"consumption valid: {int(cons_valid.sum()):,} | "
        f"overlap: {int(both.sum()):,}",
        flush=True,
    )

    packing_ambiguous = df["total_units_purchased"].isna().to_numpy()
    ratio = pd.to_numeric(df["quantity_vs_avg_ratio"], errors="coerce").to_numpy()

    # Segment masks (evaluated on head-to-head overlap unless noted)
    seg_regular = both & is_core  # core-regular purchase patterns
    seg_bulk = both & np.isfinite(ratio) & (ratio >= 1.5)
    seg_partial = both & np.isfinite(ratio) & (ratio <= 0.5)
    # Missing/ambiguous packing: existing can still predict; consumption cannot
    seg_missing_pack = exist_valid & packing_ambiguous

    results = {
        "backtest_metadata": {
            "total_supervised_rows": int(total_supervised),
            "existing_eligible_count": int(exist_valid.sum()),
            "consumption_valid_count": int(cons_valid.sum()),
            "both_overlap_count": int(both.sum()),
            "large_error_threshold_days": LARGE_ERROR_DAYS,
            "prediction_leakage_policy": (
                "Both predictors use only information available at/before the current "
                "purchase; actual target is target_days_until_next_purchase."
            ),
            "existing_strategy": "hybrid_routing_v17d (core=personal median, secondary=production XGB read-only)",
            "consumption_strategy": "historical_consumption_rate (expanding) + estimated_days_of_supply",
            "production_changes": "NONE — read-only backtest",
            "whatsapp_sends": 0,
        },
        "overall_head_to_head": {
            "existing": _metrics(y[both], df["pred_existing"].to_numpy()[both], "Existing Hybrid (head-to-head)"),
            "consumption": _metrics(
                y[both], df["pred_consumption_dos"].to_numpy()[both], "Consumption DoS (head-to-head)"
            ),
        },
        "independent_coverage": {
            "existing": {
                **_metrics(y[exist_valid], df["pred_existing"].to_numpy()[exist_valid], "Existing Hybrid (all eligible)"),
                "coverage_pct_of_supervised": round(100.0 * float(exist_valid.sum()) / total_supervised, 2),
            },
            "consumption": {
                **_metrics(
                    y[cons_valid],
                    df["pred_consumption_dos"].to_numpy()[cons_valid],
                    "Consumption DoS (all valid)",
                ),
                "coverage_pct_of_supervised": round(100.0 * float(cons_valid.sum()) / total_supervised, 2),
            },
        },
        "segments": {
            "regular": {
                "count": int(seg_regular.sum()),
                "definition": "Head-to-head overlap AND hybrid core-regular (NormMAD<=0.35, Drift<=7, P>=6, cadence in [15,120])",
                "existing": _metrics(y[seg_regular], df["pred_existing"].to_numpy()[seg_regular], "Existing (Regular)"),
                "consumption": _metrics(
                    y[seg_regular], df["pred_consumption_dos"].to_numpy()[seg_regular], "Consumption (Regular)"
                ),
            },
            "bulk": {
                "count": int(seg_bulk.sum()),
                "definition": "Head-to-head overlap AND quantity_vs_avg_ratio >= 1.5",
                "existing": _metrics(y[seg_bulk], df["pred_existing"].to_numpy()[seg_bulk], "Existing (Bulk)"),
                "consumption": _metrics(
                    y[seg_bulk], df["pred_consumption_dos"].to_numpy()[seg_bulk], "Consumption (Bulk)"
                ),
            },
            "partial_topup": {
                "count": int(seg_partial.sum()),
                "definition": "Head-to-head overlap AND quantity_vs_avg_ratio <= 0.5",
                "existing": _metrics(
                    y[seg_partial], df["pred_existing"].to_numpy()[seg_partial], "Existing (Partial/Top-up)"
                ),
                "consumption": _metrics(
                    y[seg_partial],
                    df["pred_consumption_dos"].to_numpy()[seg_partial],
                    "Consumption (Partial/Top-up)",
                ),
            },
            "missing_packing": {
                "count": int(seg_missing_pack.sum()),
                "definition": "Existing-eligible rows with missing/unparseable packing (no total_units)",
                "existing": _metrics(
                    y[seg_missing_pack],
                    df["pred_existing"].to_numpy()[seg_missing_pack],
                    "Existing (Missing Packing)",
                ),
                "consumption": {
                    "label": "Consumption (Missing Packing)",
                    "n": 0,
                    "mae": None,
                    "rmse": None,
                    "note": "Cannot produce consumption prediction without valid packing",
                },
            },
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}", flush=True)

    model_mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    assert model_mtime_before == model_mtime_after, "Model file mtime changed — abort"
    print("Model mtime unchanged. Backtest complete.", flush=True)

    # Brief stdout summary
    hh = results["overall_head_to_head"]
    print("\nHEAD-TO-HEAD", flush=True)
    for k in ("existing", "consumption"):
        m = hh[k]
        print(
            f"  {m['label']}: n={m['n']} MAE={m['mae']} RMSE={m['rmse']} "
            f"±3d={m['within_3d_pct']}% ±7d={m['within_7d_pct']}% "
            f"large>{LARGE_ERROR_DAYS}d={m['large_error_gt30d']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
