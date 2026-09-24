"""Read-only Phase 17D experiment: 5.8-year vs latest-3-year training window.

Memory-conscious. Does NOT write model artifacts. Prints JSON to stdout.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from refillcare.models.training import NUMERIC_FEATURES, TARGET_COL
from refillcare.models.evaluation import evaluate_regression_model
from refillcare.evaluation.baseline import HistoricalMedianBaseline

PROC = ROOT / "data" / "refillcare" / "processed"

VAL_MIN, VAL_MAX = "2026-05-01", "2026-06-30"
TEST_MIN, TEST_MAX = "2026-07-01", "2026-08-31"
TRAIN_MAX = "2026-04-30"
THREE_YEAR_MIN = "2023-05-01"

KEEP_COLS = list(dict.fromkeys(
    NUMERIC_FEATURES
    + [TARGET_COL, "invoice_date", "purchase_count_so_far", "is_recurring_history"]
))


def _metrics_bundle(y_true, y_pred) -> dict:
    m = evaluate_regression_model(y_true, y_pred)
    return {
        "count": m["count"],
        "mae": m["mae"],
        "medae": m["median_abs_error"],
        "rmse": m["rmse"],
        "within_3_days_pct": m["within_3_days_pct"],
        "within_7_days_pct": m["within_7_days_pct"],
    }


def _cohort_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    y_t = df[TARGET_COL].to_numpy(dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 1.0, None)
    seq = df["purchase_count_so_far"].to_numpy()
    recurring = df["is_recurring_history"].fillna(0).astype(int).to_numpy()

    masks = {
        "2_purchases": seq <= 2,
        "3_to_5_purchases": (seq >= 3) & (seq <= 5),
        "gt_5_purchases": seq > 5,
        "regular_is_recurring_history": recurring == 1,
        "irregular_ge3_not_recurring": (seq >= 3) & (recurring == 0),
    }
    out = {}
    for name, mask in masks.items():
        if not np.any(mask):
            out[name] = {
                "count": 0, "mae": None, "medae": None, "rmse": None,
                "within_3_days_pct": None, "within_7_days_pct": None,
            }
        else:
            out[name] = _metrics_bundle(y_t[mask], y_p[mask])
    return out


def _build_numeric_xgb():
    # Numeric-only to keep memory low (categorical OHE blew up peak RAM).
    # Same XGB hyperparams family as Phase 4, slightly leaner.
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        max_bin=128,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _train_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    feats = [c for c in NUMERIC_FEATURES if c in train_df.columns]

    train_c = train_df.loc[train_df[TARGET_COL] > 0]
    val_c = val_df.loc[val_df[TARGET_COL] > 0]
    test_c = test_df.loc[test_df[TARGET_COL] > 0]

    X_tr = train_c[feats].astype(np.float32)
    y_tr = train_c[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = val_c[feats].astype(np.float32)
    y_va = val_c[TARGET_COL].to_numpy(dtype=np.float32)
    X_te = test_c[feats].astype(np.float32)
    y_te = test_c[TARGET_COL].to_numpy(dtype=np.float32)

    pipe = _build_numeric_xgb()
    pipe.fit(X_tr, y_tr)
    val_pred = pipe.predict(X_va)
    test_pred = pipe.predict(X_te)

    del pipe, X_tr, y_tr, X_va, X_te
    gc.collect()

    baseline = HistoricalMedianBaseline()
    baseline.fit(train_c, target_col=TARGET_COL)
    bl_val = baseline.predict(val_c)
    bl_test = baseline.predict(test_c)

    cohorts = _cohort_metrics(test_c, test_pred)

    return {
        "train_rows_raw": int(len(train_df)),
        "train_rows_positive_target": int(len(train_c)),
        "train_date_min": str(pd.to_datetime(train_df["invoice_date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train_df["invoice_date"]).max().date()),
        "features_used": feats,
        "note": "Numeric features only (no categorical OHE) for memory-safe read-only experiment",
        "xgb_validation": _metrics_bundle(y_va, val_pred),
        "xgb_test": _metrics_bundle(y_te, test_pred),
        "baseline_validation": _metrics_bundle(y_va, bl_val),
        "baseline_test": _metrics_bundle(y_te, bl_test),
        "test_history_depth": {
            k: cohorts[k] for k in ("2_purchases", "3_to_5_purchases", "gt_5_purchases")
        },
        "test_regularity": {
            k: cohorts[k] for k in ("regular_is_recurring_history", "irregular_ge3_not_recurring")
        },
    }


def main() -> None:
    usecols = [c for c in KEEP_COLS]
    print("Loading parquets (selected columns only)...", flush=True)
    train = pd.read_parquet(PROC / "train.parquet", columns=usecols)
    val = pd.read_parquet(PROC / "validation.parquet", columns=usecols)
    test = pd.read_parquet(PROC / "test.parquet", columns=usecols)

    train["invoice_date"] = pd.to_datetime(train["invoice_date"])
    val["invoice_date"] = pd.to_datetime(val["invoice_date"])
    test["invoice_date"] = pd.to_datetime(test["invoice_date"])

    val = val[(val["invoice_date"] >= VAL_MIN) & (val["invoice_date"] <= VAL_MAX)]
    test = test[(test["invoice_date"] >= TEST_MIN) & (test["invoice_date"] <= TEST_MAX)]

    train_a = train[train["invoice_date"] <= TRAIN_MAX]
    train_b = train[(train["invoice_date"] >= THREE_YEAR_MIN) & (train["invoice_date"] <= TRAIN_MAX)]

    print(f"Arm A train rows: {len(train_a):,}", flush=True)
    print(f"Arm B train rows: {len(train_b):,}", flush=True)
    print(f"Val rows: {len(val):,} | Test rows: {len(test):,}", flush=True)

    print("Running Arm A: full 5.8-year train ...", flush=True)
    res_a = _train_eval(train_a, val, test)
    gc.collect()

    print("Running Arm B: latest 3-year train ...", flush=True)
    res_b = _train_eval(train_b, val, test)
    gc.collect()

    report = {
        "experiment": "PHASE_17D_3YEAR_VS_58YEAR",
        "read_only": True,
        "production_artifacts_modified": False,
        "model": "XGBoost numeric-only Pipeline (imputer + XGBRegressor hist)",
        "fixed_windows": {
            "validation": f"{VAL_MIN} to {VAL_MAX}",
            "test": f"{TEST_MIN} to {TEST_MAX}",
            "train_cutoff": TRAIN_MAX,
        },
        "arm_A_full_5_8_year": {
            "definition": f"Train invoice_date <= {TRAIN_MAX} (full available history)",
            **res_a,
        },
        "arm_B_latest_3_year": {
            "definition": f"Train {THREE_YEAR_MIN} <= invoice_date <= {TRAIN_MAX}",
            **res_b,
        },
    }
    out_json = PROC / "phase17d_experiment_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    print(json.dumps({
        "A_test_mae": res_a["xgb_test"]["mae"],
        "B_test_mae": res_b["xgb_test"]["mae"],
        "A_train_rows": res_a["train_rows_positive_target"],
        "B_train_rows": res_b["train_rows_positive_target"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
