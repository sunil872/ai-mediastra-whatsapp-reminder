"""Promotion gate: a model reaches production only on measured evidence.

Rules (configurable per call, defaults from environment):
- Enough paired evaluation events (``ML_GATE_MIN_PAIRED_EVENTS``, default 200).
- ML model types must beat the historical-median baseline on within-7-days by at least
  ``ML_GATE_MIN_GAIN_PP`` percentage points (default 2.0) AND the lower bound of the bootstrap
  95% CI of that gain must be > 0 (the gain is not noise).
- MAE may not be worse than the baseline by more than ``ML_GATE_MAX_MAE_REGRESSION_DAYS`` (0.5).
- Non-inferiority vs the current production engine: within-7 may not drop by more than
  ``ML_GATE_MAX_PROD_REGRESSION_PP`` (1.0) (skipped when the candidate *is* production).
- No segment with >= ``ML_GATE_SEGMENT_MIN_N`` (50) events may regress vs the baseline by more
  than ``ML_GATE_MAX_SEGMENT_REGRESSION_PP`` (5.0) on within-7, nor by more than
  ``ML_GATE_MAX_SEGMENT_MAE_REGRESSION_DAYS`` (1.5) on MAE - insulin / device therapy always checked.
- The early (spam) rate may not exceed the baseline's by more than 5 pp.
- Baseline-type candidates only need enough events + non-inferiority vs production
  (they are the safe fallback policy).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from refillcare.ml.evaluation import BASELINE_METHOD


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def gate_config() -> Dict[str, float]:
    return {
        "min_paired_events": _env_float("ML_GATE_MIN_PAIRED_EVENTS", 200),
        "min_gain_pp": _env_float("ML_GATE_MIN_GAIN_PP", 2.0),
        "max_mae_regression_days": _env_float("ML_GATE_MAX_MAE_REGRESSION_DAYS", 0.5),
        "max_prod_regression_pp": _env_float("ML_GATE_MAX_PROD_REGRESSION_PP", 1.0),
        "segment_min_n": _env_float("ML_GATE_SEGMENT_MIN_N", 50),
        "max_segment_regression_pp": _env_float("ML_GATE_MAX_SEGMENT_REGRESSION_PP", 5.0),
        "max_segment_mae_regression_days": _env_float("ML_GATE_MAX_SEGMENT_MAE_REGRESSION_DAYS", 1.5),
        "max_early_increase_pp": _env_float("ML_GATE_MAX_EARLY_INCREASE_PP", 5.0),
        # Late = reminder after the patient has already run out (stock-out risk)
        "max_late_increase_pp": _env_float("ML_GATE_MAX_LATE_INCREASE_PP", 3.0),
        # A model may not "win" by abstaining: coverage vs production may drop at most this much
        "max_coverage_drop_pp": _env_float("ML_GATE_MAX_COVERAGE_DROP_PP", 2.0),
    }


def evaluate_gate(
    report: Dict[str, Any],
    candidate_method: str,
    candidate_is_baseline: bool,
    production_method: Optional[str] = None,
    config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cfg = {**gate_config(), **(config or {})}
    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    paired = (report or {}).get("paired_comparison") or {}
    methods = paired.get("methods") or {}
    diffs = paired.get("diffs") or {}
    n = int(paired.get("n_events") or 0)

    check("min_paired_events", n >= cfg["min_paired_events"], f"{n} paired events (need >= {int(cfg['min_paired_events'])})")
    overlap = report.get("training_overlap_versions") or []
    version_by_method = report.get("version_by_method") or {}
    cand_version = version_by_method.get(candidate_method)
    check(
        "out_of_time_evaluation",
        cand_version not in overlap,
        "evaluation window is after the candidate's training cutoff" if cand_version not in overlap
        else "candidate was trained on data overlapping the evaluation window (leakage)",
    )
    cand = methods.get(candidate_method)
    base = methods.get(BASELINE_METHOD)
    if cand is None or base is None:
        check("candidate_present", False, f"'{candidate_method}' or baseline missing from the evaluation report")
        return _decision(checks, cfg, report, candidate_method)

    if not candidate_is_baseline:
        d = diffs.get(f"{candidate_method} vs {BASELINE_METHOD}") or {}
        gain = d.get("within_7_days_diff_pp")
        ci = d.get("within_7_days_diff_ci95") or [None, None]
        check(
            "beats_baseline_within_7",
            gain is not None and gain >= cfg["min_gain_pp"],
            f"within+/-7d gain vs baseline = {gain} pp (need >= {cfg['min_gain_pp']})",
        )
        check(
            "gain_is_significant",
            ci[0] is not None and ci[0] > 0,
            f"95% CI of gain = {ci}",
        )
        mae_diff = d.get("mae_diff_days")
        check(
            "mae_not_worse",
            mae_diff is not None and mae_diff <= cfg["max_mae_regression_days"],
            f"MAE diff vs baseline = {mae_diff} d (max +{cfg['max_mae_regression_days']})",
        )
        early_diff = (cand.get("early_pct") or 0) - (base.get("early_pct") or 0)
        check(
            "no_extra_reminder_spam",
            early_diff <= cfg["max_early_increase_pp"],
            f"early (spam) rate {cand.get('early_pct')}% vs baseline {base.get('early_pct')}%",
        )
        late_diff = (cand.get("late_pct") or 0) - (base.get("late_pct") or 0)
        check(
            "no_extra_late_reminders",
            late_diff <= cfg["max_late_increase_pp"],
            f"late (stock-out) rate {cand.get('late_pct')}% vs baseline {base.get('late_pct')}% (max +{cfg['max_late_increase_pp']} pp)",
        )
        # Segment regressions vs baseline
        for seg_name, seg_vals in (report.get("segments") or {}).items():
            for val, seg in seg_vals.items():
                sm = seg.get("methods") or {}
                if seg.get("n", 0) < cfg["segment_min_n"] and not (seg_name == "therapy" and val == "insulin" and seg.get("n", 0) >= 10):
                    continue
                c7 = (sm.get(candidate_method) or {}).get("within_7_days_pct")
                b7 = (sm.get(BASELINE_METHOD) or {}).get("within_7_days_pct")
                if c7 is None or b7 is None:
                    continue
                check(
                    f"segment:{seg_name}={val}",
                    (c7 - b7) >= -cfg["max_segment_regression_pp"],
                    f"n={seg.get('n')}: {c7}% vs baseline {b7}%",
                )
                cm = (sm.get(candidate_method) or {}).get("mae_days")
                bm = (sm.get(BASELINE_METHOD) or {}).get("mae_days")
                if cm is not None and bm is not None:
                    check(
                        f"segment_mae:{seg_name}={val}",
                        (cm - bm) <= cfg["max_segment_mae_regression_days"],
                        f"n={seg.get('n')}: MAE {cm}d vs baseline {bm}d (max +{cfg['max_segment_mae_regression_days']})",
                    )

    if production_method and production_method != candidate_method and production_method in methods:
        p7 = methods[production_method].get("within_7_days_pct")
        c7 = cand.get("within_7_days_pct")
        check(
            "non_inferior_to_production",
            c7 is not None and p7 is not None and (c7 - p7) >= -cfg["max_prod_regression_pp"],
            f"within+/-7d {c7}% vs current production {p7}%",
        )
        cov = report.get("coverage") or {}
        cc = (cov.get(candidate_method) or {}).get("coverage_pct")
        pc = (cov.get(production_method) or {}).get("coverage_pct")
        if cc is not None and pc is not None:
            check(
                "coverage_not_lower",
                (cc - pc) >= -cfg["max_coverage_drop_pp"],
                f"coverage {cc}% vs current production {pc}% (max -{cfg['max_coverage_drop_pp']} pp)",
            )
    return _decision(checks, cfg, report, candidate_method)


def _decision(checks: List[Dict[str, Any]], cfg: Dict[str, float], report: Dict[str, Any], candidate_method: str) -> Dict[str, Any]:
    failed = [c for c in checks if not c["passed"]]
    return {
        "approved": not failed,
        "candidate_method": candidate_method,
        "evaluation_run_id": (report or {}).get("run_id"),
        "evaluation_window": (report or {}).get("evaluation_window"),
        "checks": checks,
        "failed_checks": [c["check"] for c in failed],
        "config": cfg,
    }
