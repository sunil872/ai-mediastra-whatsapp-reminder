"""Offline cohort counts: baseline hybrid vs Path A v17F (read-only)."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from refillcare.models.hybrid_strategy import evaluate_hybrid_eligibility, MIN_PURCHASES
from refillcare.models.path_a_classifier import classify_path_a, LABEL_HIGH, LABEL_MEDIUM, LABEL_UNSTABLE
from refillcare.models.training import TARGET_COL

PROC = ROOT / "data" / "refillcare" / "processed"
MODEL_PATH = PROC / "models" / "refill_model.joblib"

_spec = importlib.util.spec_from_file_location(
    "phase17d_hybrid_comparison",
    ROOT / "scripts" / "phase17d_hybrid_comparison.py",
)
_hc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hc)


def main() -> None:
    mtime_before = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    test = pd.read_parquet(PROC / "test.parquet")
    test = test[test[TARGET_COL] > 0].copy()
    history = pd.read_parquet(PROC / "purchase_history.parquet")
    test = _hc._attach_regularity_from_history(test, history)
    test["historical_interval_norm_mad"] = test["norm_mad"]

    # Attach prior intervals for Path A diagnostics
    hist = history[
        ["customerId", "itemId", "invoice_date", "days_since_previous_purchase", "purchase_seq"]
    ].copy()
    hist["invoice_date"] = pd.to_datetime(hist["invoice_date"])
    hist["customerId"] = hist["customerId"].astype(str)
    hist["itemId"] = hist["itemId"].astype(str)
    hist = hist.sort_values(["customerId", "itemId", "invoice_date", "purchase_seq"])

    path_a = test[test["purchase_count_so_far"].astype(int) >= MIN_PURCHASES].copy()
    path_a["customerId"] = path_a["customerId"].astype(str)
    path_a["itemId"] = path_a["itemId"].astype(str)
    path_a["purchase_seq"] = path_a["purchase_count_so_far"].astype(int)
    keys = set(zip(path_a["customerId"], path_a["itemId"]))
    hist["_key"] = list(zip(hist["customerId"], hist["itemId"]))
    hist_sub = hist[hist["_key"].isin(keys)]

    prior_map = {}
    for (cid, iid), g in hist_sub.groupby(["customerId", "itemId"], sort=False):
        intervals = []
        for _, row in g.iterrows():
            seq = int(row["purchase_seq"])
            prior_map[(cid, iid, seq)] = list(intervals)
            d = row["days_since_previous_purchase"]
            if pd.notna(d) and float(d) > 0:
                intervals.append(float(d))

    baseline = Counter()
    proposed = Counter()
    for _, r in path_a.iterrows():
        key = (str(r["customerId"]), str(r["itemId"]), int(r["purchase_seq"]))
        prior = prior_map.get(key, [])
        rec = {
            "purchase_count_so_far": int(r["purchase_count_so_far"]),
            "historical_interval_median": float(r["historical_interval_median"])
            if pd.notna(r["historical_interval_median"])
            else float("nan"),
            "historical_interval_norm_mad": float(r["norm_mad"]) if pd.notna(r["norm_mad"]) else float("nan"),
            "cadence_drift": float(r["cadence_drift"]) if pd.notna(r["cadence_drift"]) else float("nan"),
            "prior_intervals": prior,
        }
        b = evaluate_hybrid_eligibility(rec)
        if b["confidence"] == "HIGH":
            baseline[LABEL_HIGH] += 1
        elif b["confidence"] == "MEDIUM":
            baseline[LABEL_MEDIUM] += 1
        else:
            baseline[LABEL_UNSTABLE] += 1
        p = classify_path_a(rec)
        proposed[p["path_a_class"]] += 1

    mtime_after = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    out = {
        "test_n": int(len(test)),
        "path_a_n": int(len(path_a)),
        "baseline_hybrid_routing_v17d": dict(baseline),
        "proposed_path_a_classifier_v17f": dict(proposed),
        "artifact_safety": {
            "model_mtime_unchanged": mtime_before == mtime_after,
            "model_mtime": mtime_before,
            "whatsapp_messages_sent": 0,
            "path_b_implemented": False,
            "feature_flag_default": False,
        },
    }
    dest = PROC / "phase17f_path_a_cohort_counts.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
