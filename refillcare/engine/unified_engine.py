"""V1 Single-Store Unified Refill Decision Engine.

Orchestrates:
1. Customer-Item Identity: customer_item_key = customerId + itemId
2. Path A / Path B Routing
3. Stability Classification (Phase 17J Policy)
4. Estimated Days of Supply (EDS/DOS) Corroboration
5. Expected Refill Date Calculation & Cycle ID Generation
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from refillcare.engine.decision_types import (
    CHURN_CUTOFF_DAYS,
    PATH_A,
    PATH_B,
    PATH_INELIGIBLE,
    PRED_DAYS_OF_SUPPLY,
    PRED_HISTORICAL_MEDIAN,
    PRED_NONE,
    PRED_PATH_A_HISTORICAL_MEDIAN,
    PRED_PATH_B_DOS_EDS,
    PRED_PATH_B_RECENCY,
    RefillDecision,
    STABILITY_HIGH,
    STABILITY_MEDIUM_RISK,
    STABILITY_MEDIUM_SAFE,
    STABILITY_UNSTABLE,
)
from refillcare.data.packing import parse_pack_units
from refillcare.models.hybrid_strategy import compute_regularity_from_intervals
from refillcare.models.path_a_classifier import (
    CADENCE_MEDIAN_MAX,
    CADENCE_MEDIAN_MIN,
    HIGH_DRIFT_ALT,
    HIGH_DRIFT_STRICT,
    HIGH_MAX_RATIO_ALT,
    HIGH_MAX_RATIO_STRICT,
    HIGH_NORM_MAD_ALT,
    HIGH_NORM_MAD_STRICT,
    MEDIUM_DRIFT_MAX,
    MEDIUM_NORM_MAD_MAX,
    MIN_PURCHASES,
)
from utils.validators import normalize_to_whatsapp_number


def classify_v1_stability(intervals: List[float], purchase_count: int) -> Tuple[str, Dict[str, Any]]:
    """
    Classify regularity & stability according to finalized Phase 17J policy.

    Returns:
        (stability_tier, metrics_dict)
        where stability_tier is one of:
        - HIGH
        - MEDIUM-SAFE
        - MEDIUM-RISK
        - UNSTABLE
    """
    if purchase_count < MIN_PURCHASES or len(intervals) < 2:
        return STABILITY_UNSTABLE, {"cadence_median": 0.0, "cadence_norm_mad": 1.0, "cadence_drift": 99.0}

    reg = compute_regularity_from_intervals(intervals)
    median = reg.get("hist_median", reg.get("cadence_median", 0.0))
    norm_mad = reg.get("norm_mad", reg.get("cadence_norm_mad", 1.0))
    drift = reg.get("cadence_drift", reg.get("drift", 99.0))
    
    # Calculate max ratio if not present
    arr = np.asarray([x for x in intervals if x > 0], dtype=np.float64)
    max_ratio = float(np.max(arr) / median) if len(arr) > 0 and median > 0 else 99.0

    reg["cadence_median"] = median
    reg["cadence_norm_mad"] = norm_mad
    reg["cadence_drift"] = drift
    reg["cadence_max_ratio"] = max_ratio

    # Range safety check (15 - 120 days)
    if median < CADENCE_MEDIAN_MIN or median > CADENCE_MEDIAN_MAX:
        return STABILITY_UNSTABLE, reg

    # Phase 17E/J HIGH criteria
    is_high_1 = (norm_mad <= HIGH_NORM_MAD_STRICT) and (drift <= HIGH_DRIFT_STRICT) and (max_ratio <= HIGH_MAX_RATIO_STRICT)
    is_high_2 = (norm_mad <= HIGH_NORM_MAD_ALT) and (drift <= HIGH_DRIFT_ALT) and (max_ratio <= HIGH_MAX_RATIO_ALT)
    if is_high_1 or is_high_2:
        return STABILITY_HIGH, reg

    # Phase 17J MEDIUM-SAFE criteria
    if norm_mad <= MEDIUM_NORM_MAD_MAX and drift <= MEDIUM_DRIFT_MAX:
        return STABILITY_MEDIUM_SAFE, reg

    # If within bounds but elevated volatility
    if norm_mad <= 0.75 and drift <= 25.0:
        return STABILITY_MEDIUM_RISK, reg

    return STABILITY_UNSTABLE, reg


class UnifiedRefillDecisionEngine:
    """Production decision engine for single-store RefillCare operations."""

    def __init__(self, default_lead_buffer_days: int = 3):
        self.default_lead_buffer_days = default_lead_buffer_days

    @staticmethod
    def compute_dos(
        dates: List[date],
        quantities: Optional[List[float]] = None,
        packings: Optional[List[Optional[str]]] = None,
        item_name: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute (estimated_days_of_supply, latest_units_purchased) for a trajectory.
        """
        if not dates:
            return None, None

        # Build list of total units per purchase
        units: List[float] = []
        if quantities:
            for i, q in enumerate(quantities):
                q_val = float(q) if q and q > 0 else 1.0
                pack_str = packings[i] if (packings and i < len(packings) and packings[i]) else (item_name or "")
                pack_units = parse_pack_units(pack_str)
                unit_val = q_val * float(pack_units) if pack_units and pack_units > 0 else q_val
                units.append(unit_val)
        elif item_name:
            pack_units = parse_pack_units(item_name)
            unit_val = float(pack_units) if pack_units and pack_units > 0 else 30.0
            units = [unit_val] * len(dates)
        else:
            units = [30.0] * len(dates)

        latest_units = units[-1] if units else None

        # Multi-purchase consumption velocity using recent active intervals (<= 120 days)
        if len(dates) >= 2 and len(units) >= 2:
            recent_units_consumed = 0.0
            recent_elapsed_days = 0
            
            # Walk backwards from most recent purchases (up to last 6 intervals)
            max_intervals = min(6, len(dates) - 1)
            for i in range(len(dates) - 1, len(dates) - 1 - max_intervals, -1):
                interval_days = (dates[i] - dates[i - 1]).days
                if 0 < interval_days <= 120:
                    recent_elapsed_days += interval_days
                    recent_units_consumed += units[i - 1]
                else:
                    # Stopped at large historical gap to avoid diluting active consumption velocity
                    break

            if recent_elapsed_days > 0 and recent_units_consumed > 0:
                daily_rate = recent_units_consumed / float(recent_elapsed_days)
                if daily_rate > 0 and latest_units and latest_units > 0:
                    dos = latest_units / daily_rate
                    dos_clamped = max(5.0, min(365.0, dos))
                    return round(float(dos_clamped), 1), round(float(latest_units), 1)

        # Single purchase or fallback to pack/purchased units
        if latest_units and latest_units > 0:
            return round(float(latest_units), 1), round(float(latest_units), 1)

        return None, latest_units

    def evaluate_customer_item_trajectory(
        self,
        customer_id: str,
        customer_name: str,
        mobile_no: Optional[str],
        item_id: str,
        item_name: str,
        dates: List[date],
        quantities: Optional[List[float]] = None,
        packings: Optional[List[Optional[str]]] = None,
        as_of_date: Optional[date] = None,
    ) -> RefillDecision:
        """Evaluate a single customer-item purchase history deterministically."""
        # Filter out future transactions if as_of_date is provided
        if as_of_date:
            dates = [d for d in dates if d <= as_of_date]

        dates = sorted(dates)
        purchase_count = len(dates)
        last_date = dates[-1] if dates else date.today()
        cust_item_key = f"{str(customer_id).strip()}_{str(item_id).strip()}"

        # Clean mobile number
        clean_phone = None
        if mobile_no and str(mobile_no).strip():
            p_norm, _ = normalize_to_whatsapp_number(mobile_no)
            clean_phone = p_norm

        # Compute consecutive intervals
        intervals: List[float] = []
        if len(dates) >= 2:
            intervals = [float((dates[i] - dates[i - 1]).days) for i in range(1, len(dates)) if (dates[i] - dates[i - 1]).days > 0]

        # Compute DOS/EDS
        dos_days, latest_units = self.compute_dos(dates, quantities, packings, item_name)

        # -----------------------------------------------------------------
        # 1. Path A Evaluation (purchase_count >= 6): Cadence-First + Quantity Scaling & DOS Guardrails
        # -----------------------------------------------------------------
        if purchase_count >= MIN_PURCHASES:
            stability_tier, reg = classify_v1_stability(intervals, purchase_count)
            cadence_med = reg.get("cadence_median")
            norm_mad = reg.get("cadence_norm_mad")
            drift = reg.get("cadence_drift")

            if stability_tier in (STABILITY_HIGH, STABILITY_MEDIUM_SAFE):
                # Calculate quantity ratio vs typical purchase
                # Build units history
                units_hist: List[float] = []
                if quantities:
                    for idx, q in enumerate(quantities):
                        q_val = float(q) if q and q > 0 else 1.0
                        pack_str = packings[idx] if (packings and idx < len(packings) and packings[idx]) else (item_name or "")
                        pack_units = parse_pack_units(pack_str)
                        unit_val = q_val * float(pack_units) if pack_units and pack_units > 0 else q_val
                        units_hist.append(unit_val)
                elif item_name:
                    pack_units = parse_pack_units(item_name)
                    unit_val = float(pack_units) if pack_units and pack_units > 0 else 30.0
                    units_hist = [unit_val] * len(dates)
                else:
                    units_hist = [30.0] * len(dates)

                latest_u = units_hist[-1] if units_hist else (latest_units or 30.0)
                prior_u = units_hist[:-1] if len(units_hist) > 1 else units_hist
                recent_prior = prior_u[-5:] if len(prior_u) >= 5 else prior_u
                typical_u = float(np.median(recent_prior)) if recent_prior else latest_u
                qty_ratio = latest_u / typical_u if typical_u > 0 else 1.0
                latest_gap = (dates[-1] - dates[-2]).days if len(dates) >= 2 else 0
                is_lapsed_restart = (latest_gap > 1.5 * cadence_med)

                # Check 1: Bulk Purchase Safety Guardrail (>=90 units and extreme divergence)
                is_bulk = (
                    latest_u >= 90.0
                    and (
                        (dos_days is not None and (dos_days - cadence_med) > 45.0)
                        or (qty_ratio >= 2.5)
                    )
                )
                if is_bulk:
                    return RefillDecision(
                        customer_id=str(customer_id),
                        customer_name=str(customer_name),
                        mobile_no=clean_phone,
                        item_id=str(item_id),
                        item_name=str(item_name),
                        customer_item_key=cust_item_key,
                        path=PATH_A,
                        purchase_count=purchase_count,
                        is_eligible=False,
                        stability_tier=stability_tier,
                        cadence_median=cadence_med,
                        cadence_norm_mad=norm_mad,
                        cadence_drift=drift,
                        dos_days=dos_days,
                        units_purchased=latest_units,
                        prediction_method=PRED_PATH_A_HISTORICAL_MEDIAN,
                        predicted_interval_days=None,
                        last_purchase_date=last_date,
                        expected_refill_date=None,
                        decision_reason=f"BULK_PURCHASE_DIVERGENCE: Historical median={int(round(cadence_med))}d vs Latest Units={int(latest_u)}u (DOS={round(dos_days, 1) if dos_days else '-'}d) indicates bulk purchase. Pharmacist review required.",
                        cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
                    )

                # Quantity-Aware Cadence Scaling & Post-Lapse Reset
                if qty_ratio < 0.8:
                    # Partial purchase (e.g. 1 strip of 10 instead of usual 20 tabs)
                    scaled_interval = round(cadence_med * qty_ratio)
                    predicted_interval = max(5, min(int(scaled_interval), int(latest_u)))
                    reason_rule = f"Partial Purchase Scaled ({int(latest_u)}u vs typical {int(typical_u)}u, ratio {qty_ratio:.0%}: {predicted_interval}d)"
                elif qty_ratio > 1.3 and latest_u < 90.0:
                    # Multi-pack purchase (e.g. 2-month supply = 60 tabs)
                    scaled_interval = round(cadence_med * qty_ratio)
                    predicted_interval = max(15, min(180, int(scaled_interval)))
                    reason_rule = f"Multi-Pack Scaled ({int(latest_u)}u vs typical {int(typical_u)}u, ratio {qty_ratio:.0%}: {predicted_interval}d)"
                elif is_lapsed_restart and latest_u <= 30:
                    # Post-lapse restart with single strip
                    predicted_interval = min(int(round(cadence_med)), int(latest_u))
                    reason_rule = f"Post-Lapse Reset ({latest_gap}d gap > 1.5x cadence {int(cadence_med)}d, capped at {int(latest_u)}u: {predicted_interval}d)"
                else:
                    # Standard recurring cadence
                    predicted_interval = int(round(cadence_med))
                    reason_rule = f"Personal Historical Median ({predicted_interval}d)"

                # Check 2: Divergence Guardrail for MEDIUM-SAFE against scaled interval
                is_divergent = (
                    stability_tier == STABILITY_MEDIUM_SAFE
                    and dos_days is not None
                    and abs(dos_days - predicted_interval) > 30.0
                    and latest_u >= 60.0
                )
                if is_divergent:
                    return RefillDecision(
                        customer_id=str(customer_id),
                        customer_name=str(customer_name),
                        mobile_no=clean_phone,
                        item_id=str(item_id),
                        item_name=str(item_name),
                        customer_item_key=cust_item_key,
                        path=PATH_A,
                        purchase_count=purchase_count,
                        is_eligible=False,
                        stability_tier=stability_tier,
                        cadence_median=cadence_med,
                        cadence_norm_mad=norm_mad,
                        cadence_drift=drift,
                        dos_days=dos_days,
                        units_purchased=latest_units,
                        prediction_method=PRED_PATH_A_HISTORICAL_MEDIAN,
                        predicted_interval_days=None,
                        last_purchase_date=last_date,
                        expected_refill_date=None,
                        decision_reason=f"PREDICTION_DOS_DIVERGENCE: Scaled interval={predicted_interval}d vs DOS={round(dos_days, 1)}d exceeds safety tolerance. Pharmacist review required.",
                        cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
                    )

                exp_date = last_date + timedelta(days=predicted_interval)
                cycle_id = f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}"

                # Check 3: Active Lifecycle Recency / Churn Gate (75 days post-due)
                if as_of_date and (as_of_date - exp_date).days > CHURN_CUTOFF_DAYS:
                    lapsed_days = (as_of_date - exp_date).days
                    return RefillDecision(
                        customer_id=str(customer_id),
                        customer_name=str(customer_name),
                        mobile_no=clean_phone,
                        item_id=str(item_id),
                        item_name=str(item_name),
                        customer_item_key=cust_item_key,
                        path=PATH_A,
                        purchase_count=purchase_count,
                        is_eligible=False,
                        stability_tier=stability_tier,
                        cadence_median=cadence_med,
                        cadence_norm_mad=norm_mad,
                        cadence_drift=drift,
                        dos_days=dos_days,
                        units_purchased=latest_units,
                        prediction_method=PRED_PATH_A_HISTORICAL_MEDIAN,
                        predicted_interval_days=predicted_interval,
                        last_purchase_date=last_date,
                        expected_refill_date=exp_date,
                        decision_reason=f"EXCLUDED_CHURNED_INACTIVE: Expected refill date {exp_date.strftime('%Y-%m-%d')} is {lapsed_days}d in the past (>{CHURN_CUTOFF_DAYS}d churn cutoff). Reactivates upon new purchase.",
                        cycle_id=cycle_id,
                    )

                reason = (
                    f"Path A ({stability_tier}) {reason_rule}. DOS corroborated ({round(dos_days, 1)}d)."
                    if dos_days
                    else f"Path A ({stability_tier}) {reason_rule}."
                )

                return RefillDecision(
                    customer_id=str(customer_id),
                    customer_name=str(customer_name),
                    mobile_no=clean_phone,
                    item_id=str(item_id),
                    item_name=str(item_name),
                    customer_item_key=cust_item_key,
                    path=PATH_A,
                    purchase_count=purchase_count,
                    is_eligible=True,
                    stability_tier=stability_tier,
                    cadence_median=cadence_med,
                    cadence_norm_mad=norm_mad,
                    cadence_drift=drift,
                    dos_days=dos_days,
                    units_purchased=latest_units,
                    prediction_method=PRED_PATH_A_HISTORICAL_MEDIAN,
                    predicted_interval_days=predicted_interval,
                    last_purchase_date=last_date,
                    expected_refill_date=exp_date,
                    decision_reason=reason,
                    cycle_id=cycle_id,
                )

            # Medium-Risk / Unstable Cadence
            return RefillDecision(
                customer_id=str(customer_id),
                customer_name=str(customer_name),
                mobile_no=clean_phone,
                item_id=str(item_id),
                item_name=str(item_name),
                customer_item_key=cust_item_key,
                path=PATH_A,
                purchase_count=purchase_count,
                is_eligible=False,
                stability_tier=stability_tier,
                cadence_median=cadence_med,
                cadence_norm_mad=norm_mad,
                cadence_drift=drift,
                dos_days=dos_days,
                units_purchased=latest_units,
                prediction_method=PRED_NONE,
                predicted_interval_days=None,
                last_purchase_date=last_date,
                expected_refill_date=None,
                decision_reason=f"Path A Ineligible: Stability {stability_tier} (elevated variance/drift). Pharmacist review required.",
                cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
            )

        # -----------------------------------------------------------------
        # 2. Path B Evaluation (purchase_count < 6): Supply/DOS-First
        # -----------------------------------------------------------------
        ref_date = as_of_date or last_date
        recency_days = (ref_date - last_date).days
        is_recent = recency_days <= 180

        m3_count = len(set(d.strftime("%Y-%m") for d in dates if (ref_date - d).days <= 90))
        m6_count = len(set(d.strftime("%Y-%m") for d in dates if (ref_date - d).days <= 180))

        is_path_b_eligible = False
        path_b_sub_reason = "INELIGIBLE"

        if is_recent:
            if m3_count >= 2:
                is_path_b_eligible = True
                path_b_sub_reason = "3M_RECURRING (>=2 distinct months)"
            elif m6_count >= 3:
                is_path_b_eligible = True
                path_b_sub_reason = "6M_RECURRING (>=3 distinct months)"

        if is_path_b_eligible:
            # For eligible Path B, DOS/EDS is the authoritative prediction
            if dos_days is not None and 10.0 <= dos_days <= 180.0:
                predicted_interval = int(round(dos_days))
                exp_date = last_date + timedelta(days=predicted_interval)
                cycle_id = f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}"

                # Active Lifecycle Recency / Churn Gate for Path B (75 days post-due)
                if as_of_date and (as_of_date - exp_date).days > CHURN_CUTOFF_DAYS:
                    lapsed_days = (as_of_date - exp_date).days
                    return RefillDecision(
                        customer_id=str(customer_id),
                        customer_name=str(customer_name),
                        mobile_no=clean_phone,
                        item_id=str(item_id),
                        item_name=str(item_name),
                        customer_item_key=cust_item_key,
                        path=PATH_B,
                        purchase_count=purchase_count,
                        is_eligible=False,
                        stability_tier=STABILITY_MEDIUM_SAFE,
                        cadence_median=None,
                        cadence_norm_mad=None,
                        cadence_drift=None,
                        dos_days=dos_days,
                        units_purchased=latest_units,
                        prediction_method=PRED_PATH_B_DOS_EDS,
                        predicted_interval_days=predicted_interval,
                        last_purchase_date=last_date,
                        expected_refill_date=exp_date,
                        decision_reason=f"EXCLUDED_CHURNED_INACTIVE: Path B expected refill date {exp_date.strftime('%Y-%m-%d')} is {lapsed_days}d in the past (>{CHURN_CUTOFF_DAYS}d churn cutoff). Reactivates upon new purchase.",
                        cycle_id=cycle_id,
                    )

                return RefillDecision(
                    customer_id=str(customer_id),
                    customer_name=str(customer_name),
                    mobile_no=clean_phone,
                    item_id=str(item_id),
                    item_name=str(item_name),
                    customer_item_key=cust_item_key,
                    path=PATH_B,
                    purchase_count=purchase_count,
                    is_eligible=True,
                    stability_tier=STABILITY_MEDIUM_SAFE,
                    cadence_median=None,
                    cadence_norm_mad=None,
                    cadence_drift=None,
                    dos_days=dos_days,
                    units_purchased=latest_units,
                    prediction_method=PRED_PATH_B_DOS_EDS,
                    predicted_interval_days=predicted_interval,
                    last_purchase_date=last_date,
                    expected_refill_date=exp_date,
                    decision_reason=f"Path B Eligible ({path_b_sub_reason}): Authoritative DOS/EDS prediction ({predicted_interval}d).",
                    cycle_id=cycle_id,
                )
            else:
                # Missing or insufficient DOS data
                return RefillDecision(
                    customer_id=str(customer_id),
                    customer_name=str(customer_name),
                    mobile_no=clean_phone,
                    item_id=str(item_id),
                    item_name=str(item_name),
                    customer_item_key=cust_item_key,
                    path=PATH_B,
                    purchase_count=purchase_count,
                    is_eligible=False,
                    stability_tier=STABILITY_MEDIUM_RISK,
                    cadence_median=None,
                    cadence_norm_mad=None,
                    cadence_drift=None,
                    dos_days=dos_days,
                    units_purchased=latest_units,
                    prediction_method=PRED_NONE,
                    predicted_interval_days=None,
                    last_purchase_date=last_date,
                    expected_refill_date=None,
                    decision_reason=f"Path B Ineligible: DOS_DATA_INSUFFICIENT ({path_b_sub_reason} satisfied but pack/quantity data unavailable for authoritative DOS). Pharmacist review required.",
                    cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
                )

        # Ineligible Path B
        reason = "Cold start / insufficient purchase recurrence" if is_recent else f"Stale recency (>180 days: {recency_days}d)"
        return RefillDecision(
            customer_id=str(customer_id),
            customer_name=str(customer_name),
            mobile_no=clean_phone,
            item_id=str(item_id),
            item_name=str(item_name),
            customer_item_key=cust_item_key,
            path=PATH_INELIGIBLE,
            purchase_count=purchase_count,
            is_eligible=False,
            stability_tier=STABILITY_UNSTABLE,
            cadence_median=None,
            cadence_norm_mad=None,
            cadence_drift=None,
            dos_days=dos_days,
            units_purchased=latest_units,
            prediction_method=PRED_NONE,
            predicted_interval_days=None,
            last_purchase_date=last_date,
            expected_refill_date=None,
            decision_reason=f"Ineligible: {reason}",
            cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
        )

    def evaluate_all_customer_items(
        self,
        df: pd.DataFrame,
        as_of_date: Optional[date] = None,
    ) -> Tuple[pd.DataFrame, List[RefillDecision]]:
        """
        Evaluate full transaction dataframe in high-speed grouped batch.

        Returns:
            (decisions_df, decisions_list)
        """
        if df.empty:
            return pd.DataFrame(), []

        # Ensure datetime
        df_clean = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_clean["invoice_date"]):
            df_clean["invoice_date"] = pd.to_datetime(df_clean["invoice_date"], errors="coerce")

        if as_of_date:
            df_clean = df_clean[df_clean["invoice_date"].dt.date <= as_of_date]

        df_clean = df_clean.sort_values("invoice_date").reset_index(drop=True)

        trajectories = {}
        cids = df_clean["customerId"].astype(str).values
        iids = df_clean["itemId"].astype(str).values
        cnames = df_clean["customerName"].values if "customerName" in df_clean.columns else [None] * len(df_clean)
        inames = df_clean["itemName"].values if "itemName" in df_clean.columns else [None] * len(df_clean)
        mobiles = df_clean["MOBILE_NO"].values if "MOBILE_NO" in df_clean.columns else [None] * len(df_clean)
        inv_dates = df_clean["invoice_date"].values
        qtys = df_clean["quantity"].values if "quantity" in df_clean.columns else [None] * len(df_clean)
        packings = df_clean["packing"].values if "packing" in df_clean.columns else [None] * len(df_clean)

        for i in range(len(df_clean)):
            cid = cids[i]
            iid = iids[i]
            key = (cid, iid)
            if key not in trajectories:
                trajectories[key] = {
                    "cid": cid,
                    "iid": iid,
                    "cname": cnames[i],
                    "iname": inames[i],
                    "mobile": mobiles[i],
                    "dates": [],
                    "qtys": [],
                    "packings": [],
                }
            t = trajectories[key]
            if pd.notna(cnames[i]):
                t["cname"] = cnames[i]
            if pd.notna(inames[i]):
                t["iname"] = inames[i]
            if pd.notna(mobiles[i]):
                t["mobile"] = mobiles[i]
            ts = inv_dates[i]
            if pd.notna(ts):
                t["dates"].append(pd.Timestamp(ts).date())
                t["qtys"].append(qtys[i] if pd.notna(qtys[i]) else None)
                t["packings"].append(packings[i] if pd.notna(packings[i]) else None)

        decisions: List[RefillDecision] = []
        records_list: List[Dict[str, Any]] = []

        for key, t in trajectories.items():
            dec = self.evaluate_customer_item_trajectory(
                customer_id=t["cid"],
                customer_name=str(t["cname"]) if pd.notna(t["cname"]) else str(t["cid"]),
                mobile_no=t["mobile"],
                item_id=t["iid"],
                item_name=str(t["iname"]) if pd.notna(t["iname"]) else str(t["iid"]),
                dates=t["dates"],
                quantities=t["qtys"],
                packings=t["packings"],
                as_of_date=as_of_date,
            )
            decisions.append(dec)
            records_list.append(dec.to_dict())

        decisions_df = pd.DataFrame(records_list)
        return decisions_df, decisions

