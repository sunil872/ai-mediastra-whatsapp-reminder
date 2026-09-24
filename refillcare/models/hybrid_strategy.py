"""Phase 17D approved hybrid prediction strategy (hybrid_routing_v17d).

Eligibility-gated routing:
  eligible + core regular  -> personal historical median (HIGH)
  eligible + secondary     -> improved XGBoost branch (MEDIUM)
  otherwise                -> REJECT (no automated predicted date)

Thresholds are fixed to Phase 17D measured evidence. Reminder stages unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np
import pandas as pd

STRATEGY_NAME = "hybrid_routing_v17d"

# Eligibility / regularity thresholds (PHASE_17D_FINAL_MODEL_STRATEGY)
MIN_PURCHASES = 6
CADENCE_MEDIAN_MIN = 15.0
CADENCE_MEDIAN_MAX = 120.0
CORE_NORM_MAD_MAX = 0.35
CORE_DRIFT_MAX = 7.0
BROAD_NORM_MAD_MAX = 0.50
BROAD_DRIFT_MAX = 10.0
PACK_DEFAULT_DAYS = 30.0  # operational hint for rejected low-history (not an auto prediction)

ROUTE_CORE = "core_personal_median"
ROUTE_SECONDARY = "secondary_experimental_xgb"
ROUTE_REJECTED = "rejected"

CONF_HIGH = "HIGH"
CONF_MEDIUM = "MEDIUM"
CONF_REJECTED = "REJECTED"
CONF_INVALID = "INVALID"


def compute_regularity_from_intervals(
    prior_intervals: Sequence[float],
) -> Dict[str, float]:
    """Leakage-safe NormMAD / recent-3 drift from prior positive intervals only."""
    vals = [float(x) for x in prior_intervals if x is not None and not (isinstance(x, float) and math.isnan(x)) and float(x) > 0]
    if not vals:
        return {
            "mad": float("nan"),
            "norm_mad": float("nan"),
            "recent3_median": float("nan"),
            "cadence_drift": float("nan"),
            "hist_median": float("nan"),
            "n_prior_intervals": 0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    norm_mad = float(mad / med) if med > 0 else float("nan")
    recent3 = float(np.median(arr[-3:]))
    drift = abs(recent3 - med)
    return {
        "mad": mad,
        "norm_mad": norm_mad,
        "recent3_median": recent3,
        "cadence_drift": drift,
        "hist_median": med,
        "n_prior_intervals": len(vals),
    }


def _as_float(val: Any, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _as_int(val: Any, default: int = 1) -> int:
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def resolve_regularity_fields(record: Dict[str, Any]) -> Dict[str, float]:
    """Prefer explicit columns; optionally recompute from prior_intervals.

    Fallback (only when exact MAD/drift columns are absent on legacy feature rows):
    - NormMAD ≈ 0.6745 * std / median
    - Drift ≈ |days_since_previous_purchase − historical_interval_median|
    Exact Phase-17D columns from feature engineering always take precedence.
    """
    norm_mad = _as_float(record.get("historical_interval_norm_mad", record.get("norm_mad")))
    drift = _as_float(record.get("cadence_drift", record.get("recent_cadence_drift")))
    hist_med = _as_float(record.get("historical_interval_median"))
    mad = _as_float(record.get("historical_interval_mad"))
    recent3 = _as_float(record.get("recent3_interval_median", record.get("recent3_median")))

    prior = record.get("prior_intervals")
    if prior is not None and (math.isnan(norm_mad) or math.isnan(drift)):
        computed = compute_regularity_from_intervals(list(prior))
        if math.isnan(norm_mad):
            norm_mad = computed["norm_mad"]
        if math.isnan(drift):
            drift = computed["cadence_drift"]
        if math.isnan(hist_med):
            hist_med = computed["hist_median"]
        if math.isnan(mad):
            mad = computed["mad"]
        if math.isnan(recent3):
            recent3 = computed["recent3_median"]

    # Legacy feature-row fallback (pre-Phase-17D regularity columns)
    if math.isnan(norm_mad) and not math.isnan(hist_med) and hist_med > 0:
        std = _as_float(record.get("historical_interval_std"))
        if not math.isnan(std) and std >= 0:
            # Normal-distribution MAD/std constant; used only when exact MAD unavailable
            mad_approx = 0.67448975 * std
            mad = mad_approx if math.isnan(mad) else mad
            norm_mad = mad_approx / hist_med

    if math.isnan(drift) and not math.isnan(hist_med):
        last_iv = _as_float(record.get("days_since_previous_purchase", record.get("last_purchase_interval")))
        if not math.isnan(last_iv):
            drift = abs(last_iv - hist_med)
            if math.isnan(recent3):
                recent3 = last_iv

    return {
        "norm_mad": norm_mad,
        "cadence_drift": drift,
        "historical_interval_median": hist_med,
        "historical_interval_mad": mad,
        "recent3_interval_median": recent3,
    }


def evaluate_hybrid_eligibility(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Phase 17D eligibility + core/secondary routing decision (no model call)."""
    purchase_count = _as_int(
        record.get("purchase_count_so_far", record.get("purchase_seq", 1)),
        default=1,
    )
    reg = resolve_regularity_fields(record)
    hist_med = reg["historical_interval_median"]
    norm_mad = reg["norm_mad"]
    drift = reg["cadence_drift"]

    if purchase_count < MIN_PURCHASES:
        return {
            "is_eligible": False,
            "is_core_regular": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            "rejection_reason": (
                f"Low-history / insufficient depth: purchase_count={purchase_count} "
                f"(requires >= {MIN_PURCHASES})."
            ),
            "purchase_count": purchase_count,
            **reg,
            "pack_default_days": PACK_DEFAULT_DAYS if purchase_count <= 3 else None,
        }

    if math.isnan(hist_med) or hist_med < CADENCE_MEDIAN_MIN or hist_med > CADENCE_MEDIAN_MAX:
        return {
            "is_eligible": False,
            "is_core_regular": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            "rejection_reason": (
                f"Cadence median out of refill band [15, 120]: hist_median={hist_med}."
            ),
            "purchase_count": purchase_count,
            **reg,
            "pack_default_days": None,
        }

    if math.isnan(norm_mad) or math.isnan(drift):
        return {
            "is_eligible": False,
            "is_core_regular": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            "rejection_reason": "Missing NormMAD or cadence drift; cannot verify regularity.",
            "purchase_count": purchase_count,
            **reg,
            "pack_default_days": None,
        }

    if norm_mad > BROAD_NORM_MAD_MAX or drift > BROAD_DRIFT_MAX:
        return {
            "is_eligible": False,
            "is_core_regular": False,
            "route": ROUTE_REJECTED,
            "confidence": CONF_REJECTED,
            "rejection_reason": (
                f"Irregular cadence: NormMAD={norm_mad:.3f} (max {BROAD_NORM_MAD_MAX}) "
                f"or Drift={drift:.1f}d (max {BROAD_DRIFT_MAX})."
            ),
            "purchase_count": purchase_count,
            **reg,
            "pack_default_days": None,
        }

    is_core = (norm_mad <= CORE_NORM_MAD_MAX) and (drift <= CORE_DRIFT_MAX)
    if is_core:
        return {
            "is_eligible": True,
            "is_core_regular": True,
            "route": ROUTE_CORE,
            "confidence": CONF_HIGH,
            "rejection_reason": None,
            "purchase_count": purchase_count,
            **reg,
            "pack_default_days": None,
        }

    return {
        "is_eligible": True,
        "is_core_regular": False,
        "route": ROUTE_SECONDARY,
        "confidence": CONF_MEDIUM,
        "rejection_reason": None,
        "purchase_count": purchase_count,
        **reg,
        "pack_default_days": None,
    }


def personal_median_prediction(record: Dict[str, Any]) -> float:
    """Core-regular predictor: personal historical median interval."""
    med = _as_float(record.get("historical_interval_median"))
    if math.isnan(med) or med <= 0:
        raise ValueError("historical_interval_median missing or non-positive for core route")
    return float(med)


def build_improved_xgboost_regressor(**overrides):
    """Factory for Phase 17D secondary-branch XGB (quantile median regression)."""
    import xgboost as xgb

    params = dict(
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
    params.update(overrides)
    return xgb.XGBRegressor(**params)
