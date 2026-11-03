"""Deterministic date rules applied around the model (each switchable, each measured by the backtest).

| Rule | Env switch | Default |
|---|---|---|
| High-stability patients with a normal-size purchase use their own median | ``REFILLCARE_RULE_HIGH_STABILITY_MEDIAN`` | off |
| Insulin / device therapy uses the median (model raised insulin MAE) | ``REFILLCARE_RULE_INSULIN_MEDIAN`` | off |
| Day-of-month anchor: snap to the patient's usual purchase day (only when the calendar day explains their history better than a fixed interval) | ``REFILLCARE_RULE_DOM_ANCHOR`` | off |
| Short histories (Path B): ``dos`` (days of supply), ``shrink`` (personal median shrunk toward the salt median) or ``blend`` (average of both) | ``REFILLCARE_RULE_PATH_B_MODE`` | dos |
| Device in-use ceiling (insulin pens, inhalers, drops) for Path A: ``strict`` (30 days per device) or ``observed`` (never below the patient's own median - 35% of insulin patients refill later than 30 days per pen) | ``REFILLCARE_RULE_DEVICE_CAP_MODE`` | strict |

All rules ship **off**; switch one on only after an evaluation run shows it helps
(``scripts/ml_cli.py rules-ab`` compares every rule on the same unseen events).
Quantity archetypes (partial strip, multi-pack, early top-up, stock-up) are never routed to the
median - a half strip must still shorten the interval.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PATH_B_MODES = ("dos", "shrink", "blend")
DEVICE_CAP_MODES = ("strict", "observed")
SHRINK_PRIOR_STRENGTH = 2.0      # salt prior counts as this many intervals
DOM_ANCHOR_MAX_SPREAD_DAYS = 3.0  # usual day must be this consistent
DOM_ANCHOR_MAX_SHIFT_DAYS = 6.0   # never move a prediction further than this
DOM_ANCHOR_MIN_PURCHASES = 4
DOM_ANCHOR_MIN_FIT_GAIN_DAYS = 0.3  # calendar day must explain history better than a fixed interval
DOM_ANCHOR_INTERVAL_RANGE = (20, 40)  # monthly buyers only


def _flag(name: str, default: bool) -> bool:
    from refillcare.stores import store_setting

    return (store_setting(name, "true" if default else "false") or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class EngineRules:
    high_stability_median: bool = False
    insulin_median: bool = False
    dom_anchor: bool = False
    path_b_mode: str = "dos"
    device_cap_mode: str = "strict"

    @classmethod
    def from_env(cls) -> "EngineRules":
        from refillcare.env import ensure_dotenv

        from refillcare.stores import store_setting

        ensure_dotenv()
        mode = (store_setting("REFILLCARE_RULE_PATH_B_MODE", "dos") or "dos").strip().lower()
        cap = (store_setting("REFILLCARE_RULE_DEVICE_CAP_MODE", "strict") or "strict").strip().lower()
        return cls(
            high_stability_median=_flag("REFILLCARE_RULE_HIGH_STABILITY_MEDIAN", False),
            insulin_median=_flag("REFILLCARE_RULE_INSULIN_MEDIAN", False),
            dom_anchor=_flag("REFILLCARE_RULE_DOM_ANCHOR", False),
            path_b_mode=mode if mode in PATH_B_MODES else "dos",
            device_cap_mode=cap if cap in DEVICE_CAP_MODES else "strict",
        )

    @classmethod
    def legacy(cls) -> "EngineRules":
        """Behaviour before these rules existed (for A/B backtests)."""
        return cls(high_stability_median=False, insulin_median=False, dom_anchor=False, path_b_mode="dos", device_cap_mode="strict")

    def describe(self) -> Dict[str, Any]:
        return {
            "high_stability_median": self.high_stability_median,
            "insulin_median": self.insulin_median,
            "dom_anchor": self.dom_anchor,
            "path_b_mode": self.path_b_mode,
            "device_cap_mode": self.device_cap_mode,
        }


def device_ceiling_days(mode: str, cap_days: Optional[float], personal_median: Optional[float]) -> Optional[float]:
    """In-use device ceiling for a Path A prediction.

    ``strict``: the label ceiling (30 days per pen). ``observed``: never below the patient's own
    median interval - a patient who reliably refills a pen every 40 days is reminded at ~40, not 30.
    """
    if cap_days is None:
        return None
    if mode == "observed" and personal_median:
        return max(float(cap_days), float(personal_median))
    return float(cap_days)


def is_normal_quantity(features: Dict[str, Any]) -> bool:
    """No quantity archetype (partial strip, multi-pack, early top-up, stock-up, anomaly)."""
    ratio = float(features.get("quantity_ratio_vs_typical") or 1.0)
    flags = ("is_stocking_up", "is_partial_purchase", "is_early_topup", "is_quantity_anomaly")
    return 0.8 <= ratio <= 1.3 and not any(features.get(f) for f in flags)


def dom_anchor_interval(
    last_date: date,
    interval_days: int,
    features: Dict[str, Any],
    purchase_count: int,
) -> Tuple[int, bool]:
    """Snap a monthly prediction to the patient's usual day of month when that habit is strong."""
    usual = features.get("usual_purchase_dom")
    spread = features.get("dom_spread_days")
    gain = features.get("dom_fit_gain_days")
    lo, hi = DOM_ANCHOR_INTERVAL_RANGE
    if (
        usual is None or spread is None or gain is None or purchase_count < DOM_ANCHOR_MIN_PURCHASES
        or spread > DOM_ANCHOR_MAX_SPREAD_DAYS or gain < DOM_ANCHOR_MIN_FIT_GAIN_DAYS
        or not (lo <= interval_days <= hi)
    ):
        return interval_days, False
    from refillcare.features.point_in_time import dom_shift

    pred = last_date + timedelta(days=int(interval_days))
    shift = dom_shift(float(usual), float(pred.day))
    if abs(shift) < 1 or abs(shift) > DOM_ANCHOR_MAX_SHIFT_DAYS:
        return interval_days, False
    return max(5, int(round(interval_days + shift))), True


def path_b_interval(
    mode: str,
    intervals: List[float],
    dos_days: Optional[float],
    salt_stats: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], str]:
    """Interval for short histories (3-5 purchases). Returns (days, method_label)."""
    usable = [float(x) for x in intervals if 5 <= x <= 180]
    prior = (salt_stats or {}).get("median_interval")
    shrunk: Optional[float] = None
    if usable and prior:
        n = float(len(usable))
        shrunk = (n * float(np.median(usable)) + SHRINK_PRIOR_STRENGTH * float(prior)) / (n + SHRINK_PRIOR_STRENGTH)
    elif usable:
        shrunk = float(np.median(usable))
    dos_ok = dos_days is not None and 10.0 <= float(dos_days) <= 180.0
    if mode == "shrink" and shrunk is not None:
        return shrunk, "shrinkage"
    if mode == "blend" and shrunk is not None and dos_ok:
        return 0.5 * shrunk + 0.5 * float(dos_days), "blend"
    if dos_ok:
        return float(dos_days), "dos"
    return (shrunk, "shrinkage") if shrunk is not None else (None, "none")
