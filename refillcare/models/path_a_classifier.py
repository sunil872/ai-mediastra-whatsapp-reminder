"""Phase 17F Path A classifier (NON-PRODUCTION, feature-flagged).

Established customers only (purchase_count >= 6). Implements the approved
Phase 17E decision tree for HIGH / MEDIUM / UNSTABLE.

This module does NOT replace hybrid_routing_v17d. Production prediction
continues to use evaluate_hybrid_eligibility unless USE_PATH_A_CLASSIFIER_V17F
is explicitly enabled.

Path B (recent / low-history customers) is intentionally not implemented.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from refillcare.models.hybrid_strategy import (
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
    _as_float,
    _as_int,
    compute_regularity_from_intervals,
    resolve_regularity_fields,
)

STRATEGY_NAME = "path_a_classifier_v17f"
BASELINE_STRATEGY_NAME = "hybrid_routing_v17d"

# Feature flag: OFF by default (non-production). Enable via env or set True locally.
USE_PATH_A_CLASSIFIER_V17F = os.environ.get("REFILLCARE_USE_PATH_A_V17F", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Path A thresholds (Phase 17E approved tree)
MIN_PURCHASES = 6
CADENCE_MEDIAN_MIN = 15.0
CADENCE_MEDIAN_MAX = 120.0

EXTREME_PCT_MAX = 35.0
BAND_PCT_MIN = 55.0
MAX_OVER_MEDIAN_UNSTABLE = 3.5
WITHIN_50_PCT_MIN_FOR_MAX_RATIO = 65.0

HIGH_NORM_MAD_STRICT = 0.30
HIGH_DRIFT_STRICT = 7.0
HIGH_MAX_RATIO_STRICT = 2.2

HIGH_NORM_MAD_ALT = 0.35
HIGH_DRIFT_ALT = 5.0
HIGH_MAX_RATIO_ALT = 2.0

MEDIUM_NORM_MAD_MAX = 0.55
MEDIUM_DRIFT_MAX = 15.0

LABEL_HIGH = "HIGH"
LABEL_MEDIUM = "MEDIUM"
LABEL_UNSTABLE = "UNSTABLE"


def is_path_a_classifier_enabled() -> bool:
    """Return whether the non-production Path A classifier flag is on."""
    return bool(USE_PATH_A_CLASSIFIER_V17F)


def compute_path_a_interval_diagnostics(
    prior_intervals: Sequence[float],
) -> Dict[str, float]:
    """Leakage-safe Path A diagnostics from prior positive intervals only."""
    vals = [
        float(x)
        for x in prior_intervals
        if x is not None and not (isinstance(x, float) and math.isnan(x)) and float(x) > 0
    ]
    reg = compute_regularity_from_intervals(vals)
    if not vals:
        return {
            **reg,
            "pct_extreme_lt10_or_gt180": float("nan"),
            "pct_in_refill_band_15_120": float("nan"),
            "pct_within_50pct_of_median": float("nan"),
            "max_over_median": float("nan"),
        }

    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    return {
        **reg,
        "pct_extreme_lt10_or_gt180": float(np.mean((arr < 10.0) | (arr > 180.0)) * 100.0),
        "pct_in_refill_band_15_120": float(np.mean((arr >= 15.0) & (arr <= 120.0)) * 100.0),
        "pct_within_50pct_of_median": (
            float(np.mean(np.abs(arr - med) <= 0.50 * med) * 100.0) if med > 0 else float("nan")
        ),
        "max_over_median": float(arr.max() / med) if med > 0 else float("nan"),
    }


def resolve_path_a_fields(record: Dict[str, Any]) -> Dict[str, float]:
    """Resolve NormMAD/Drift/hist median plus Path A diagnostic columns.

    Prefers explicit feature columns; recomputes from ``prior_intervals`` when needed.
    """
    reg = resolve_regularity_fields(record)
    extreme = _as_float(record.get("pct_extreme_lt10_or_gt180"))
    band = _as_float(record.get("pct_in_refill_band_15_120"))
    within50 = _as_float(
        record.get("pct_within_50pct_of_median", record.get("pct_within_50pct_median"))
    )
    max_ratio = _as_float(record.get("max_over_median"))

    prior = record.get("prior_intervals")
    need = any(math.isnan(x) for x in (extreme, band, within50, max_ratio))
    need = need or math.isnan(reg["norm_mad"]) or math.isnan(reg["cadence_drift"])
    need = need or math.isnan(reg["historical_interval_median"])
    if prior is not None and need:
        computed = compute_path_a_interval_diagnostics(list(prior))
        if math.isnan(reg["norm_mad"]):
            reg["norm_mad"] = computed["norm_mad"]
        if math.isnan(reg["cadence_drift"]):
            reg["cadence_drift"] = computed["cadence_drift"]
        if math.isnan(reg["historical_interval_median"]):
            reg["historical_interval_median"] = computed["hist_median"]
        if math.isnan(reg["historical_interval_mad"]):
            reg["historical_interval_mad"] = computed["mad"]
        if math.isnan(reg["recent3_interval_median"]):
            reg["recent3_interval_median"] = computed["recent3_median"]
        if math.isnan(extreme):
            extreme = computed["pct_extreme_lt10_or_gt180"]
        if math.isnan(band):
            band = computed["pct_in_refill_band_15_120"]
        if math.isnan(within50):
            within50 = computed["pct_within_50pct_of_median"]
        if math.isnan(max_ratio):
            max_ratio = computed["max_over_median"]

    # Derive max_over_median from hist min/max when possible
    if math.isnan(max_ratio):
        hist_med = reg["historical_interval_median"]
        hist_max = _as_float(record.get("historical_interval_max"))
        if not math.isnan(hist_med) and hist_med > 0 and not math.isnan(hist_max):
            max_ratio = hist_max / hist_med

    return {
        **reg,
        "pct_extreme_lt10_or_gt180": extreme,
        "pct_in_refill_band_15_120": band,
        "pct_within_50pct_of_median": within50,
        "max_over_median": max_ratio,
    }


def _decision(
    *,
    label: str,
    reason: Optional[str],
    purchase_count: int,
    fields: Dict[str, float],
) -> Dict[str, Any]:
    if label == LABEL_HIGH:
        route, confidence, eligible, core = ROUTE_CORE, CONF_HIGH, True, True
    elif label == LABEL_MEDIUM:
        route, confidence, eligible, core = ROUTE_SECONDARY, CONF_MEDIUM, True, False
    else:
        route, confidence, eligible, core = ROUTE_REJECTED, CONF_REJECTED, False, False
        # Expose UNSTABLE as business label while keeping REJECTED confidence for routing compat
    return {
        "strategy": STRATEGY_NAME,
        "path": "A",
        "label": label,
        "is_eligible": eligible,
        "is_core_regular": core,
        "route": route,
        "confidence": confidence if label != LABEL_UNSTABLE else CONF_REJECTED,
        "path_a_class": label,
        "rejection_reason": reason,
        "purchase_count": purchase_count,
        "norm_mad": fields.get("norm_mad"),
        "cadence_drift": fields.get("cadence_drift"),
        "historical_interval_median": fields.get("historical_interval_median"),
        "historical_interval_mad": fields.get("historical_interval_mad"),
        "recent3_interval_median": fields.get("recent3_interval_median"),
        "pct_extreme_lt10_or_gt180": fields.get("pct_extreme_lt10_or_gt180"),
        "pct_in_refill_band_15_120": fields.get("pct_in_refill_band_15_120"),
        "pct_within_50pct_of_median": fields.get("pct_within_50pct_of_median"),
        "max_over_median": fields.get("max_over_median"),
        "pack_default_days": None,
    }


def classify_path_a(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the approved Phase 17E Path A decision tree.

    Returns a decision dict with ``path_a_class`` in {HIGH, MEDIUM, UNSTABLE}.
    Does not call the production model and does not send WhatsApp.
    """
    purchase_count = _as_int(
        record.get("purchase_count_so_far", record.get("purchase_seq", 1)),
        default=1,
    )
    fields = resolve_path_a_fields(record)
    hist_med = fields["historical_interval_median"]
    norm_mad = fields["norm_mad"]
    drift = fields["cadence_drift"]
    extreme = fields["pct_extreme_lt10_or_gt180"]
    band = fields["pct_in_refill_band_15_120"]
    within50 = fields["pct_within_50pct_of_median"]
    max_ratio = fields["max_over_median"]

    # 1. Depth gate
    if purchase_count < MIN_PURCHASES:
        return _decision(
            label=LABEL_UNSTABLE,
            reason=(
                f"Low-history / not Path A: purchase_count={purchase_count} "
                f"(requires >= {MIN_PURCHASES})."
            ),
            purchase_count=purchase_count,
            fields=fields,
        )

    # 2. Cadence band
    if math.isnan(hist_med) or hist_med < CADENCE_MEDIAN_MIN or hist_med > CADENCE_MEDIAN_MAX:
        return _decision(
            label=LABEL_UNSTABLE,
            reason=f"Cadence median out of refill band [15, 120]: hist_median={hist_med}.",
            purchase_count=purchase_count,
            fields=fields,
        )

    # 3. Required regularity metrics
    if math.isnan(norm_mad) or math.isnan(drift):
        return _decision(
            label=LABEL_UNSTABLE,
            reason="Missing NormMAD or cadence drift; cannot verify Path A regularity.",
            purchase_count=purchase_count,
            fields=fields,
        )

    # 4–6 require diagnostics; if missing after resolve, treat as UNSTABLE
    if math.isnan(extreme) or math.isnan(band) or math.isnan(within50) or math.isnan(max_ratio):
        return _decision(
            label=LABEL_UNSTABLE,
            reason=(
                "Missing Path A diagnostics "
                "(pct_extreme / pct_in_band / pct_within_50 / max_over_median)."
            ),
            purchase_count=purchase_count,
            fields=fields,
        )

    # 4. Extreme gaps
    if extreme >= EXTREME_PCT_MAX:
        return _decision(
            label=LABEL_UNSTABLE,
            reason=(
                f"Extreme interval share too high: pct_extreme={extreme:.1f}% "
                f"(max {EXTREME_PCT_MAX}%)."
            ),
            purchase_count=purchase_count,
            fields=fields,
        )

    # 5. Refill-band support
    if band < BAND_PCT_MIN:
        return _decision(
            label=LABEL_UNSTABLE,
            reason=(
                f"Too few intervals in refill band [15,120]: pct_in_band={band:.1f}% "
                f"(min {BAND_PCT_MIN}%)."
            ),
            purchase_count=purchase_count,
            fields=fields,
        )

    # 6. Contaminated cadence (large outlier + poor concentration)
    if max_ratio >= MAX_OVER_MEDIAN_UNSTABLE and within50 < WITHIN_50_PCT_MIN_FOR_MAX_RATIO:
        return _decision(
            label=LABEL_UNSTABLE,
            reason=(
                f"Contaminated cadence: max/median={max_ratio:.2f} "
                f"with only {within50:.1f}% of intervals within ±50% of median."
            ),
            purchase_count=purchase_count,
            fields=fields,
        )

    # 7. HIGH
    high_a = (
        norm_mad <= HIGH_NORM_MAD_STRICT
        and drift <= HIGH_DRIFT_STRICT
        and max_ratio <= HIGH_MAX_RATIO_STRICT
    )
    high_b = (
        norm_mad <= HIGH_NORM_MAD_ALT
        and drift <= HIGH_DRIFT_ALT
        and max_ratio <= HIGH_MAX_RATIO_ALT
    )
    if high_a or high_b:
        return _decision(
            label=LABEL_HIGH,
            reason=None,
            purchase_count=purchase_count,
            fields=fields,
        )

    # 8. MEDIUM
    if norm_mad <= MEDIUM_NORM_MAD_MAX and drift <= MEDIUM_DRIFT_MAX:
        return _decision(
            label=LABEL_MEDIUM,
            reason=None,
            purchase_count=purchase_count,
            fields=fields,
        )

    # 9. Otherwise UNSTABLE
    return _decision(
        label=LABEL_UNSTABLE,
        reason=(
            f"Fails MEDIUM regularity: NormMAD={norm_mad:.3f} (max {MEDIUM_NORM_MAD_MAX}) "
            f"or Drift={drift:.1f}d (max {MEDIUM_DRIFT_MAX})."
        ),
        purchase_count=purchase_count,
        fields=fields,
    )


def classify_path_a_from_intervals(
    prior_intervals: Sequence[float],
    *,
    purchase_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience: classify from an interval sequence (tests / offline)."""
    ivs = list(prior_intervals)
    diag = compute_path_a_interval_diagnostics(ivs)
    pcount = purchase_count if purchase_count is not None else len(ivs) + 1
    record = {
        "purchase_count_so_far": pcount,
        "historical_interval_median": diag["hist_median"],
        "historical_interval_norm_mad": diag["norm_mad"],
        "norm_mad": diag["norm_mad"],
        "cadence_drift": diag["cadence_drift"],
        "historical_interval_mad": diag["mad"],
        "recent3_interval_median": diag["recent3_median"],
        "pct_extreme_lt10_or_gt180": diag["pct_extreme_lt10_or_gt180"],
        "pct_in_refill_band_15_120": diag["pct_in_refill_band_15_120"],
        "pct_within_50pct_of_median": diag["pct_within_50pct_of_median"],
        "max_over_median": diag["max_over_median"],
        "prior_intervals": ivs,
    }
    return classify_path_a(record)


def evaluate_eligibility(
    record: Dict[str, Any],
    *,
    use_path_a_v17f: Optional[bool] = None,
) -> Dict[str, Any]:
    """Eligibility entrypoint with optional Path A v17F flag.

    Default uses baseline ``hybrid_routing_v17d``. When the feature flag is on
    (or ``use_path_a_v17f=True``), uses the Path A classifier for P>=6 rows.
    Path B is not implemented: P<6 remains UNSTABLE/REJECTED under Path A rules.
    """
    from refillcare.models.hybrid_strategy import evaluate_hybrid_eligibility

    enabled = is_path_a_classifier_enabled() if use_path_a_v17f is None else bool(use_path_a_v17f)
    if not enabled:
        decision = evaluate_hybrid_eligibility(record)
        decision = {
            **decision,
            "strategy": BASELINE_STRATEGY_NAME,
            "path": "A_baseline" if _as_int(record.get("purchase_count_so_far", 1), 1) >= MIN_PURCHASES else "baseline",
            "path_a_class": (
                LABEL_HIGH
                if decision.get("confidence") == CONF_HIGH
                else LABEL_MEDIUM
                if decision.get("confidence") == CONF_MEDIUM
                else LABEL_UNSTABLE
            ),
            "label": (
                LABEL_HIGH
                if decision.get("confidence") == CONF_HIGH
                else LABEL_MEDIUM
                if decision.get("confidence") == CONF_MEDIUM
                else LABEL_UNSTABLE
            ),
        }
        return decision

    return classify_path_a(record)
