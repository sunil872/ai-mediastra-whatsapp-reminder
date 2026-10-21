"""Point-in-Time Feature Engineering for RefillCare V1.

Ensures strict point-in-time calculation with zero temporal leakage.
At transaction N, features are calculated strictly using purchases 1 ... N.
Information from N+1 or later is never used.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from refillcare.data.packing import parse_pack_units


POINT_IN_TIME_FEATURE_COLS = [
    # 1. Quantity features
    "latest_units",
    "previous_units",
    "typical_units_median",
    "recent_units_median",
    "quantity_ratio_vs_typical",
    "quantity_ratio_vs_previous",
    "quantity_std",
    # 2. Cadence & temporal features
    "last_interval",
    "prior_interval",
    "historical_median_interval",
    "recent_median_interval",
    "rolling_interval_mean",
    "cadence_norm_mad",
    "cadence_drift",
    # 3. Supply & consumption features
    "estimated_consumption_velocity",
    "estimated_dos",
    "estimated_residual_inventory",
    "estimated_effective_supply",
    "estimated_supply_days",
    "pack_units",
    # 4. Behavioral archetype flags
    "is_stocking_up",
    "is_partial_purchase",
    "is_early_topup",
    "is_post_lapse",
    "is_quantity_anomaly",
    # 5. Customer history metadata
    "purchase_count",
    "customer_tenure_days",
    "purchase_month",
]

# Calendar: many patients buy on a fixed part of the month (salary / pension days).
CALENDAR_FEATURE_COLS = [
    "last_purchase_dom",
    "last_purchase_weekday",
    "usual_purchase_dom",
    "dom_spread_days",
    "dom_anchor_shift_days",
    "dom_fit_gain_days",
]
# Visit level: the patient's other medicines (needs refillcare.features.context.VisitIndex).
VISIT_FEATURE_COLS = [
    "days_since_prev_visit_any_item",
    "visits_last_180d",
    "active_items_180d",
    "other_item_next_due_days",
    "other_items_due_within_30d",
]
# Salt level: store-wide priors per salt + strength, incl. a default dosage per day
# (needs refillcare.features.context.PopulationStats built from the SALT ITEMS LIST enrichment).
SALT_FEATURE_COLS = [
    "salt_median_interval",
    "salt_customer_items",
    "salt_units_per_day",
    "dose_supply_days",
    "personal_vs_salt_interval_ratio",
]
BASE_FEATURE_COLS = list(POINT_IN_TIME_FEATURE_COLS)
POINT_IN_TIME_FEATURE_COLS = BASE_FEATURE_COLS + CALENDAR_FEATURE_COLS + VISIT_FEATURE_COLS + SALT_FEATURE_COLS

DOM_CYCLE_DAYS = 30.44


def _circular_dom(days_of_month: List[int]) -> Tuple[Optional[float], Optional[float]]:
    """Circular mean day-of-month (1..30) and spread in days for recent purchases."""
    if not days_of_month:
        return None, None
    ang = 2 * np.pi * (np.asarray(days_of_month, dtype=float) - 1) / DOM_CYCLE_DAYS
    c, s_ = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
    r = float(np.hypot(c, s_))
    mean_dom = (np.arctan2(s_, c) % (2 * np.pi)) * DOM_CYCLE_DAYS / (2 * np.pi) + 1
    spread = float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-9))))) * DOM_CYCLE_DAYS / (2 * np.pi)
    return round(float(mean_dom), 2), round(spread, 2)


def dom_shift(target_dom: float, current_dom: float) -> float:
    """Signed shortest move (days) from ``current_dom`` to ``target_dom`` on a month cycle."""
    d = (target_dom - current_dom + DOM_CYCLE_DAYS / 2) % DOM_CYCLE_DAYS - DOM_CYCLE_DAYS / 2
    return round(float(d), 2)


def compute_point_in_time_features_single(
    dates: List[date],
    quantities: Optional[List[float]] = None,
    packings: Optional[List[Optional[str]]] = None,
    item_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute point-in-time features for the latest purchase N from history 1...N.

    Args:
        dates: Chronologically sorted list of purchase dates up to current event.
        quantities: List of quantities purchased for each transaction.
        packings: List of packing strings for each transaction.
        item_name: Optional fallback item name for packing parsing.
        context: Optional cross-item context from ``refillcare.features.context.build_context``
            (``visit`` and ``salt_stats``). Missing context gives None (NaN) for those features.

    Returns:
        Dict[str, Any] containing all POINT_IN_TIME_FEATURE_COLS and metadata.
    """
    if not dates:
        raise ValueError("dates list cannot be empty for point-in-time feature extraction")

    N = len(dates)
    last_date = dates[-1]

    # Convert quantities and packings to total units
    units: List[float] = []
    pack_units_val = 10.0
    if quantities:
        for idx, q in enumerate(quantities):
            q_val = float(q) if q and q > 0 else 1.0
            pack_str = packings[idx] if (packings and idx < len(packings) and packings[idx]) else (item_name or "")
            pu = parse_pack_units(pack_str)
            if pu and pu > 0:
                pack_units_val = float(pu)
            u = q_val * pack_units_val
            units.append(u)
    elif item_name:
        pu = parse_pack_units(item_name)
        if pu and pu > 0:
            pack_units_val = float(pu)
        units = [pack_units_val] * N
    else:
        units = [30.0] * N

    latest_u = float(units[-1])
    prev_u = float(units[-2]) if N >= 2 else latest_u

    # 1. Quantity metrics
    typical_u = float(np.median(units)) if units else latest_u
    recent_u_subset = units[-3:] if N >= 3 else units
    recent_u_med = float(np.median(recent_u_subset))
    qty_ratio_typical = latest_u / typical_u if typical_u > 0 else 1.0
    qty_ratio_prev = latest_u / prev_u if prev_u > 0 else 1.0
    qty_std = float(np.std(units)) if N >= 2 else 0.0

    # 2. Cadence metrics
    intervals: List[float] = []
    if N >= 2:
        for i in range(1, N):
            diff = (dates[i] - dates[i - 1]).days
            if diff > 0:
                intervals.append(float(diff))

    last_int = float(intervals[-1]) if intervals else 30.0
    prior_int = float(intervals[-2]) if len(intervals) >= 2 else last_int

    if intervals:
        arr = np.asarray(intervals, dtype=np.float64)
        hist_median = float(np.median(arr))
        recent_arr = arr[-3:] if len(arr) >= 3 else arr
        recent_median = float(np.median(recent_arr))
        roll_mean = float(np.mean(arr[-5:] if len(arr) >= 5 else arr))
        abs_dev = np.abs(arr - hist_median)
        mad = float(np.median(abs_dev))
        norm_mad = float(mad / hist_median) if hist_median > 0 else 1.0
        drift = float(abs(recent_median - hist_median))
    else:
        hist_median = 30.0
        recent_median = 30.0
        roll_mean = 30.0
        norm_mad = 0.0
        drift = 0.0

    # 3. Robust Estimated Consumption Velocity (units/day)
    # Exclude the latest transition if it was an early gap, so early visits don't inflate consumption velocity
    rates: List[float] = []
    if N >= 2 and len(intervals) >= 1:
        end_idx = N - 1 if (last_int <= 20.0 and N >= 3) else N
        for i in range(1, end_idx):
            gap = (dates[i] - dates[i - 1]).days
            if 15 <= gap <= 120 and units[i - 1] > 0:
                rate = units[i - 1] / float(gap)
                if 0.2 <= rate <= 4.0:
                    rates.append(rate)

    # Typical rate anchor: typical units / historical median
    typical_rate = typical_u / hist_median if hist_median > 0 else 1.0
    if rates:
        est_velocity = float(0.5 * float(np.median(rates)) + 0.5 * typical_rate)
    else:
        est_velocity = typical_rate

    # Clamp velocity within plausible clinical range
    est_velocity = float(np.clip(est_velocity, 0.33, 3.0))

    # Estimated Days of Supply from latest purchase alone
    est_dos = latest_u / est_velocity if est_velocity > 0 else latest_u

    # 4. Estimated Residual Home Inventory
    # If the patient returned early (last_interval <= 20 or <= 0.7 * hist_median) and <= 30 days:
    is_early_gap = (last_int <= 20.0) or (hist_median > 0 and last_int < 0.7 * hist_median)
    if is_early_gap and last_int <= 30.0 and N >= 2:
        consumed_units = last_int * est_velocity
        est_residual = float(max(0.0, prev_u - consumed_units))
        # Cap residual at previous purchase units to prevent compounding artifacts
        est_residual = min(est_residual, prev_u, 90.0)
    else:
        est_residual = 0.0

    est_effective_supply = latest_u + est_residual

    # Dimensionally correct estimated supply days: units / (units/day) = days
    est_supply_days = est_effective_supply / est_velocity if est_velocity > 0 else est_effective_supply
    est_supply_days = float(np.clip(est_supply_days, 5.0, 365.0))

    # 5. Behavioral archetype indicators
    is_stocking = 1.0 if (latest_u >= 1.8 * typical_u or latest_u >= 60.0) else 0.0
    is_partial = 1.0 if (latest_u <= 0.6 * typical_u or latest_u <= 15.0) else 0.0
    is_early = 1.0 if (is_early_gap and N >= 2) else 0.0
    is_post_lapse = 1.0 if (last_int >= 1.8 * hist_median or last_int >= 45.0) else 0.0
    is_anomaly = 1.0 if (qty_ratio_typical >= 2.5 or qty_ratio_typical <= 0.35) else 0.0

    # 6. Customer history metadata
    tenure_days = (last_date - dates[0]).days if N >= 2 else 0
    month_val = float(last_date.month)

    feature_dict = {
        "latest_units": round(latest_u, 2),
        "previous_units": round(prev_u, 2),
        "typical_units_median": round(typical_u, 2),
        "recent_units_median": round(recent_u_med, 2),
        "quantity_ratio_vs_typical": round(qty_ratio_typical, 3),
        "quantity_ratio_vs_previous": round(qty_ratio_prev, 3),
        "quantity_std": round(qty_std, 2),
        "last_interval": round(last_int, 1),
        "prior_interval": round(prior_int, 1),
        "historical_median_interval": round(hist_median, 1),
        "recent_median_interval": round(recent_median, 1),
        "rolling_interval_mean": round(roll_mean, 1),
        "cadence_norm_mad": round(norm_mad, 3),
        "cadence_drift": round(drift, 1),
        "estimated_consumption_velocity": round(est_velocity, 4),
        "estimated_dos": round(est_dos, 1),
        "estimated_residual_inventory": round(est_residual, 1),
        "estimated_effective_supply": round(est_effective_supply, 1),
        "estimated_supply_days": round(est_supply_days, 1),
        "pack_units": round(pack_units_val, 1),
        "is_stocking_up": is_stocking,
        "is_partial_purchase": is_partial,
        "is_early_topup": is_early,
        "is_post_lapse": is_post_lapse,
        "is_quantity_anomaly": is_anomaly,
        "purchase_count": float(N),
        "customer_tenure_days": float(tenure_days),
        "purchase_month": month_val,
    }

    # 7. Calendar anchoring (last 6 purchases)
    usual_dom, dom_spread = _circular_dom([d.day for d in dates[-6:]]) if N >= 3 else (None, None)
    pred_by_median = last_date.toordinal() + int(round(hist_median))
    pred_dom = date.fromordinal(pred_by_median).day
    # Does "same day each month" explain recent purchases better than "same interval"?
    # (> 0: calendar-anchored buyer; a strict 30-day buyer drifts across dates and scores <= 0)
    dom_fit_gain = None
    if usual_dom is not None and len(intervals) >= 3:
        recent_iv = np.asarray(intervals[-5:], dtype=float)
        interval_err = float(np.mean(np.abs(recent_iv - np.median(recent_iv))))
        dom_err = float(np.mean([abs(dom_shift(usual_dom, d.day)) for d in dates[-6:]]))
        dom_fit_gain = round(interval_err - dom_err, 2)
    feature_dict.update({
        "dom_fit_gain_days": dom_fit_gain,
        "last_purchase_dom": float(last_date.day),
        "last_purchase_weekday": float(last_date.weekday()),
        "usual_purchase_dom": usual_dom,
        "dom_spread_days": dom_spread,
        "dom_anchor_shift_days": dom_shift(usual_dom, pred_dom) if usual_dom is not None else None,
    })

    # 8. Visit level (other medicines of the same patient)
    visit = (context or {}).get("visit") or {}
    for col in VISIT_FEATURE_COLS:
        feature_dict[col] = visit.get(col)

    # 9. Salt priors + default dosage per day
    st = (context or {}).get("salt_stats") or {}
    salt_med = st.get("median_interval")
    salt_upd = st.get("units_per_day")
    feature_dict.update({
        "salt_median_interval": salt_med,
        "salt_customer_items": st.get("n"),
        "salt_units_per_day": salt_upd,
        "dose_supply_days": round(est_effective_supply / salt_upd, 1) if salt_upd else None,
        "personal_vs_salt_interval_ratio": round(hist_median / salt_med, 3) if salt_med else None,
    })

    return feature_dict
