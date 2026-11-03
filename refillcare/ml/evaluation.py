"""Rolling-origin backtest: measure what RefillCare would have predicted vs what patients did.

Protocol (point-in-time correct):
1. Same-day purchases of a customer-item are merged into one purchase event.
2. An *evaluation event* is purchase ``i`` of a customer-item whose date falls inside the
   evaluation window and which has at least ``min_prior_purchases`` earlier purchases.
   Only history up to and including purchase ``i`` is visible to every predictor.
3. Target = days until the next observed purchase. Events whose next purchase is not observed
   (right-censored) or comes after ``max_target_days`` (therapy lapse) are excluded and counted.
4. Predictors scored on the *same paired events*:
   - ``historical_median``: personal median interval (reference baseline)
   - ``days_of_supply``: units bought / estimated consumption velocity (reference)
   - ``engine[<version>]``: the full UnifiedRefillDecisionEngine exactly as it ships, with the
     given registry version serving Path A (eligibility gates, archetypes, safety clamps included).
5. Metrics: MAE, median AE, bias, within +/-3 / +/-7 days, early (reminder > 7 d too early ->
   spam) and late (> 7 d too late -> stock-out risk), bootstrap 95% CIs, plus segments.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("refillcare.ml.evaluation")

BASELINE_METHOD = "historical_median"
DOS_METHOD = "days_of_supply"


def engine_method_name(version_id: str) -> str:
    return f"engine[{version_id}]"


# ----------------------------------------------------------------------------- events
def prepare_purchase_events(history_df: pd.DataFrame) -> pd.DataFrame:
    """Customer-item-date purchase events (same-day rows merged), eligible customer sales only."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    df = history_df.copy()
    if "refillcare_eligible" in df.columns:
        df = df[df["refillcare_eligible"].fillna(True).astype(bool)]
    df = df[df["customerId"].notna() & df["itemId"].notna()]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df[df["invoice_date"].notna()]
    from refillcare.data.identity import normalise_trajectory_keys

    df = normalise_trajectory_keys(df)
    df["customerId"] = df["customerId"].astype(str).str.strip()
    df["itemId"] = df["itemId"].astype(str).str.strip()
    if "quantity" in df.columns:
        from refillcare.data.schema import supply_quantity

        # Same supply as serving: billed + free quantity
        df["quantity"] = supply_quantity(df["quantity"], df["freeQuantity"]) if "freeQuantity" in df.columns else pd.to_numeric(df["quantity"], errors="coerce")
    else:
        df["quantity"] = np.nan
    df["_date"] = df["invoice_date"].dt.normalize()
    salt_col = next((c for c in ("salt_composition", "SALT") if c in df.columns), None)
    agg = {
        "quantity": ("quantity", "sum"),
        "packing": ("packing", "first") if "packing" in df.columns else ("itemId", "first"),
        "itemName": ("itemName", "first") if "itemName" in df.columns else ("itemId", "first"),
        "mobile": ("MOBILE_NO", "last") if "MOBILE_NO" in df.columns else ("itemId", "first"),
    }
    if salt_col:
        agg["salt"] = (salt_col, "first")
    ev = df.groupby(["customerId", "itemId", "_date"], sort=False).agg(**agg).reset_index()
    if "packing" not in df.columns:
        ev["packing"] = None
    if "MOBILE_NO" not in df.columns:
        ev["mobile"] = None
    if not salt_col:
        ev["salt"] = None
    ev = ev.rename(columns={"_date": "date"}).sort_values(["customerId", "itemId", "date"]).reset_index(drop=True)
    return ev


def build_evaluation_events(
    purchase_events: pd.DataFrame,
    eval_start: date,
    eval_end: date,
    min_prior_purchases: int = 2,
    max_target_days: int = 180,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Select evaluation events; returns {"events": [...], "counts": {...}}."""
    counts = {"in_window": 0, "censored_excluded": 0, "lapse_excluded": 0, "insufficient_history": 0, "selected": 0}
    events: List[Dict[str, Any]] = []
    if purchase_events is None or purchase_events.empty:
        return {"events": events, "counts": counts}

    start_ts, end_ts = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    for (cid, iid), g in purchase_events.groupby(["customerId", "itemId"], sort=False):
        dates = list(g["date"])
        in_window = [i for i, d in enumerate(dates) if start_ts <= d <= end_ts]
        if not in_window:
            continue
        qtys = list(g["quantity"])
        packs = list(g["packing"])
        mobiles = list(g["mobile"])
        name = next((n for n in g["itemName"] if isinstance(n, str) and n.strip()), str(iid))
        salt = next((s for s in g["salt"] if isinstance(s, str) and s.strip()), None)
        for i in in_window:
            counts["in_window"] += 1
            if i < min_prior_purchases:
                counts["insufficient_history"] += 1
                continue
            if i + 1 >= len(dates):
                counts["censored_excluded"] += 1
                continue
            target = (dates[i + 1] - dates[i]).days
            if target > max_target_days:
                counts["lapse_excluded"] += 1
                continue
            events.append({
                "customer_id": cid,
                "item_id": iid,
                "item_name": name,
                "salt": salt,
                "dates": [d.date() for d in dates[: i + 1]],
                "quantities": [None if pd.isna(q) else float(q) for q in qtys[: i + 1]],
                "packings": [None if (p is None or (isinstance(p, float) and np.isnan(p))) else str(p) for p in packs[: i + 1]],
                "mobiles": mobiles[: i + 1],
                "actual_days": float(target),
                "history_len": i + 1,
            })

    if max_events is not None and len(events) > max_events:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(events), size=max_events, replace=False)
        events = [events[k] for k in sorted(idx)]
    counts["selected"] = len(events)
    return {"events": events, "counts": counts}


# ----------------------------------------------------------------------------- predictors
def predict_historical_median(ev: Dict[str, Any]) -> Optional[float]:
    d = ev["dates"]
    iv = [(d[k] - d[k - 1]).days for k in range(1, len(d)) if (d[k] - d[k - 1]).days > 0]
    return float(np.median(iv)) if iv else None


def predict_days_of_supply(ev: Dict[str, Any], engine) -> Optional[float]:
    dos, _ = engine.compute_dos(ev["dates"], ev["quantities"], ev["packings"], ev["item_name"], ev.get("salt"))
    return float(dos) if dos is not None and dos > 0 else None


def predict_engine(ev: Dict[str, Any], engine, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dec = engine.evaluate_customer_item_trajectory(
        customer_id=ev["customer_id"],
        customer_name=ev["customer_id"],
        mobile_no=None,
        item_id=ev["item_id"],
        item_name=ev["item_name"],
        dates=ev["dates"],
        quantities=ev["quantities"],
        packings=ev["packings"],
        as_of_date=ev["dates"][-1],
        salt_name=ev.get("salt"),
        mobile_history=None,  # identity guard is an outreach rule, not a forecasting rule
        context=context,
    )
    pred = float(dec.predicted_interval_days) if dec.is_eligible and dec.predicted_interval_days else None
    return {
        "pred": pred,
        "path": dec.path,
        "stability": dec.stability_tier,
        "source": dec.prediction_source or "NONE",
    }


# ----------------------------------------------------------------------------- metrics
def _boot_indices(n: int, rng: np.random.Generator, n_boot: int, groups: Optional[np.ndarray]) -> List[np.ndarray]:
    """Bootstrap resamples. With ``groups`` (patient ids) whole patients are resampled: one patient
    contributes many refills, so resampling single events makes the intervals too narrow."""
    if groups is None:
        return list(rng.integers(0, n, size=(n_boot, n)))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def _metrics(pred: np.ndarray, actual: np.ndarray, rng: Optional[np.random.Generator] = None, n_boot: int = 300,
             groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
    n = int(len(pred))
    if n == 0:
        return {"n": 0}
    err = pred - actual  # + = predicted too late
    ae = np.abs(err)
    w7 = ae <= 7
    out = {
        "n": n,
        "mae_days": round(float(ae.mean()), 2),
        "median_abs_error_days": round(float(np.median(ae)), 2),
        "rmse_days": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias_days": round(float(err.mean()), 2),
        "within_3_days_pct": round(float((ae <= 3).mean() * 100), 2),
        "within_7_days_pct": round(float(w7.mean() * 100), 2),
        "early_pct": round(float((err < -7).mean() * 100), 2),
        "late_pct": round(float((err > 7).mean() * 100), 2),
    }
    if rng is not None and n >= 20:
        idx = _boot_indices(n, rng, n_boot, groups)
        w7_b = np.array([w7[i].mean() for i in idx]) * 100
        mae_b = np.array([ae[i].mean() for i in idx])
        out["within_7_days_ci95"] = [round(float(np.percentile(w7_b, 2.5)), 2), round(float(np.percentile(w7_b, 97.5)), 2)]
        out["mae_days_ci95"] = [round(float(np.percentile(mae_b, 2.5)), 2), round(float(np.percentile(mae_b, 97.5)), 2)]
    return out


def _paired_diff(pred_a: np.ndarray, pred_b: np.ndarray, actual: np.ndarray, rng: np.random.Generator, n_boot: int = 500,
                 groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Bootstrap CI of (A - B) for within-7 (pp) and MAE (days) on paired events (patient clusters)."""
    n = len(actual)
    if n < 20:
        return {"n": n}
    w_a = (np.abs(pred_a - actual) <= 7).astype(float)
    w_b = (np.abs(pred_b - actual) <= 7).astype(float)
    ae_a, ae_b = np.abs(pred_a - actual), np.abs(pred_b - actual)
    idx = _boot_indices(n, rng, n_boot, groups)
    dw = np.array([(w_a[i].mean() - w_b[i].mean()) for i in idx]) * 100
    dm = np.array([(ae_a[i].mean() - ae_b[i].mean()) for i in idx])
    return {
        "n": n,
        "within_7_days_diff_pp": round(float((w_a.mean() - w_b.mean()) * 100), 2),
        "within_7_days_diff_ci95": [round(float(np.percentile(dw, 2.5)), 2), round(float(np.percentile(dw, 97.5)), 2)],
        "mae_diff_days": round(float(ae_a.mean() - ae_b.mean()), 2),
        "mae_diff_ci95": [round(float(np.percentile(dm, 2.5)), 2), round(float(np.percentile(dm, 97.5)), 2)],
    }


def _depth_bucket(n: int) -> str:
    if n < 6:
        return "3-5 purchases"
    if n < 12:
        return "6-11 purchases"
    return "12+ purchases"


# ----------------------------------------------------------------------------- run
def run_backtest(
    history_df: pd.DataFrame,
    engine_variants: Dict[str, Any],
    eval_start: date,
    eval_end: date,
    candidate_method: Optional[str] = None,
    min_prior_purchases: int = 2,
    max_target_days: Optional[int] = None,
    max_events: Optional[int] = 3000,
    seed: int = 42,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Score reference methods and engine variants on identical future events.

    ``max_target_days`` defaults to the censor buffer (ML_CENSOR_BUFFER_DAYS, 90): with the test
    window ending that many days before the last sale, every counted refill is fully observable.

    Args:
        engine_variants: {method_name: UnifiedRefillDecisionEngine} (one per registry version).
        candidate_method: the method under test (used by the promotion gate); defaults to the
            first engine variant.
    """
    from refillcare.data.packing import is_insulin_product
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine
    from refillcare.features.context import PopulationStats, VisitIndex, build_context

    import os

    if max_target_days is None:
        max_target_days = int(os.getenv("ML_CENSOR_BUFFER_DAYS", "90"))
    rng = np.random.default_rng(seed)
    pe = prepare_purchase_events(history_df)
    data_end = pe["date"].max().date() if not pe.empty else None
    sel = build_evaluation_events(pe, eval_start, eval_end, min_prior_purchases, max_target_days, max_events, seed)
    events = sel["events"]

    # Point-in-time context, identical for every method: visit queries only see purchases up to
    # each event's date; salt priors are built from purchases before the evaluation window.
    visit_index = VisitIndex.from_events(pe)
    salt_stats = PopulationStats.from_events(pe, as_of=eval_start - timedelta(days=1))
    ref_engine = UnifiedRefillDecisionEngine(identity_guard=False, population_stats=salt_stats)
    for eng in engine_variants.values():
        eng.population_stats = salt_stats
    rows: List[Dict[str, Any]] = []
    for ev in events:
        ctx = build_context(visit_index, salt_stats, ev["customer_id"], ev["item_id"], ev["dates"][-1], ev.get("salt"), ev["item_name"])
        row: Dict[str, Any] = {
            "actual": ev["actual_days"],
            "patient": ev["customer_id"],
            "depth": _depth_bucket(ev["history_len"]),
            "therapy": "insulin" if is_insulin_product(ev["item_name"], ev["packings"][-1], ev.get("salt")) else "other",
            BASELINE_METHOD: predict_historical_median(ev),
            DOS_METHOD: predict_days_of_supply(ev, ref_engine),
        }
        for name, eng in engine_variants.items():
            r = predict_engine(ev, eng, ctx)
            row[name] = r["pred"]
            row[f"{name}::path"] = r["path"]
            row[f"{name}::stability"] = r["stability"]
            row[f"{name}::source"] = r["source"]
        rows.append(row)

    df = pd.DataFrame(rows)
    methods = [BASELINE_METHOD, DOS_METHOD] + list(engine_variants.keys())
    candidate_method = candidate_method or (list(engine_variants.keys())[0] if engine_variants else BASELINE_METHOD)

    coverage = {}
    for m in methods:
        n_pred = int(df[m].notna().sum()) if (not df.empty and m in df.columns) else 0
        coverage[m] = {"n_predicted": n_pred, "coverage_pct": round(100.0 * n_pred / len(df), 2) if len(df) else 0.0}

    # Paired set: events where every method produced a prediction
    paired = df.dropna(subset=methods) if not df.empty else df
    actual = paired["actual"].to_numpy(dtype=float) if not paired.empty else np.array([])
    groups = paired["patient"].astype(str).to_numpy() if not paired.empty else None
    paired_methods = {m: _metrics(paired[m].to_numpy(dtype=float), actual, rng, groups=groups) for m in methods} if len(actual) else {}
    diffs = {}
    if len(actual):
        for m in methods:
            if m == BASELINE_METHOD:
                continue
            diffs[f"{m} vs {BASELINE_METHOD}"] = _paired_diff(paired[m].to_numpy(float), paired[BASELINE_METHOD].to_numpy(float), actual, rng, groups=groups)
        engine_names = list(engine_variants.keys())
        for other in engine_names:
            if other != candidate_method and candidate_method in engine_variants:
                diffs[f"{candidate_method} vs {other}"] = _paired_diff(
                    paired[candidate_method].to_numpy(float), paired[other].to_numpy(float), actual, rng, groups=groups
                )

    # Segments on the paired set (per candidate engine's own path / stability / source)
    segments: Dict[str, Dict[str, Any]] = {}
    if len(actual):
        seg_cols = {"depth": "depth", "therapy": "therapy"}
        if candidate_method in engine_variants:
            seg_cols.update({
                "path": f"{candidate_method}::path",
                "stability": f"{candidate_method}::stability",
                "prediction_source": f"{candidate_method}::source",
            })
        for seg_name, col in seg_cols.items():
            segments[seg_name] = {}
            for val, g in paired.groupby(col):
                a = g["actual"].to_numpy(float)
                segments[seg_name][str(val)] = {
                    "n": int(len(g)),
                    "methods": {m: _metrics(g[m].to_numpy(float), a) for m in methods},
                }

    report = {
        "run_id": run_id or f"EVAL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "created_at": datetime.utcnow().isoformat(),
        "evaluation_window": {"start": eval_start.isoformat(), "end": eval_end.isoformat()},
        "data_end": data_end.isoformat() if data_end else None,
        "params": {
            "engine_rules": {name: getattr(getattr(eng, "rules", None), "describe", lambda: None)() for name, eng in engine_variants.items()},
            "salt_priors": {"as_of": salt_stats.as_of, "salts": len(salt_stats.by_salt)},
            "min_prior_purchases": min_prior_purchases,
            "max_target_days": max_target_days,
            "max_events": max_events,
            "seed": seed,
        },
        "event_counts": {**sel["counts"], "evaluated": int(len(df)), "paired": int(len(paired))},
        "baseline_method": BASELINE_METHOD,
        "candidate_method": candidate_method,
        "methods": methods,
        "coverage": coverage,
        "paired_comparison": {"n_events": int(len(paired)), "methods": paired_methods, "diffs": diffs},
        "segments": segments,
        "notes": [
            "Metrics are computed only on events where every method produced a prediction (paired).",
            "Censored events (next purchase not yet observed) and lapses beyond max_target_days are excluded; see event_counts.",
            "early_pct = predicted refill > 7 days before the actual purchase (reminder spam risk); late_pct = > 7 days after (stock-out risk).",
        ],
    }
    if data_end and eval_end > data_end - timedelta(days=max_target_days // 3):
        report["notes"].append(
            "Evaluation window ends close to the last sales date: many late refills are censored, which biases metrics toward short intervals."
        )
    return report
