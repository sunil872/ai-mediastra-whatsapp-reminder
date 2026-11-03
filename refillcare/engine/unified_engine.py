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

    def __init__(
        self,
        default_lead_buffer_days: int = 3,
        model_provider: Optional[Any] = None,
        identity_guard: Optional[bool] = None,
        population_stats: Optional[Any] = None,
        rules: Optional[Any] = None,
    ):
        """
        Args:
            model_provider: refillcare.ml.model_provider.ModelProvider (production / shadow
                models). None keeps the legacy global human model.
            identity_guard: block outreach when the latest purchases of a customer key carry
                different mobile numbers (merged patients). Default: env REFILLCARE_IDENTITY_GUARD (on).
            population_stats: refillcare.features.context.PopulationStats (salt priors). Built
                from the scored data by ``evaluate_all_customer_items`` when not given.
            rules: refillcare.engine.prediction_rules.EngineRules. Default: from env.
        """
        import os

        from refillcare.engine.prediction_rules import EngineRules

        self.default_lead_buffer_days = default_lead_buffer_days
        self.model_provider = model_provider
        if identity_guard is None:
            identity_guard = os.getenv("REFILLCARE_IDENTITY_GUARD", "true").strip().lower() in ("1", "true", "yes", "on")
        self.identity_guard = identity_guard
        self.population_stats = population_stats
        self.rules = rules if rules is not None else EngineRules.from_env()

    # ------------------------------------------------------------------
    # Provenance & safety post-processing
    # ------------------------------------------------------------------
    IDENTITY_WINDOW_PURCHASES = 3

    @staticmethod
    def _distinct_valid_mobiles(mobiles: Optional[List[Any]], window: int) -> List[str]:
        if not mobiles:
            return []
        out: List[str] = []
        for m in list(mobiles)[-window:]:
            if m is None or (isinstance(m, float) and np.isnan(m)):
                continue
            digits = "".join(ch for ch in str(m) if ch.isdigit())
            if len(digits) >= 10:
                d10 = digits[-10:]
                if d10[0] in "6789" and d10 not in out:
                    out.append(d10)
        return out

    def _finalize(self, dec: RefillDecision, mobile_history: Optional[List[Any]] = None) -> RefillDecision:
        if dec.prediction_source is None:
            if not dec.is_eligible:
                dec.prediction_source = "REVIEW" if "review" in str(dec.decision_reason or "").lower() else "NONE"
            elif dec.path == PATH_B:
                dec.prediction_source = "DOS"
                dec.model_version = dec.model_version or "rule:path_b_days_of_supply"

        if self.identity_guard and dec.is_eligible:
            distinct = self._distinct_valid_mobiles(mobile_history, self.IDENTITY_WINDOW_PURCHASES)
            if len(distinct) > 1:
                masked = ", ".join(f"******{d[-4:]}" for d in distinct)
                dec.is_eligible = False
                dec.expected_refill_date = None
                dec.prediction_source = "REVIEW"
                dec.decision_reason = (
                    f"IDENTITY_CONFLICT: last {self.IDENTITY_WINDOW_PURCHASES} purchases of this customer key carry "
                    f"{len(distinct)} different mobile numbers ({masked}); possible merged patients. "
                    f"Verify patient identity before outreach. Original: {dec.decision_reason}"
                )
        return dec

    def _attach_consensus(self, dec: RefillDecision, consensus: Dict[str, Any], shadow: Optional[Dict[str, Any]]) -> RefillDecision:
        dec.model_version = consensus.get("model_version")
        dec.prediction_source = consensus.get("prediction_source")
        dec.raw_ml_prediction_days = consensus.get("raw_ml_prediction_days")
        if shadow is not None:
            dec.shadow_model_version = shadow.get("model_version")
            dec.shadow_predicted_interval_days = shadow.get("predicted_interval_days")
        return dec

    def _consensus_kwargs(self, loaded: Optional[Any]) -> Dict[str, Any]:
        if loaded is None:
            return {}
        if getattr(loaded, "is_baseline", False):
            return {"use_ml": False, "model_version": loaded.version_id}
        return {"model_bundle": loaded.bundle, "model_version": loaded.version_id, "use_ml": True}

    @staticmethod
    def compute_dos(
        dates: List[date],
        quantities: Optional[List[float]] = None,
        packings: Optional[List[Optional[str]]] = None,
        item_name: Optional[str] = None,
        salt_name: Optional[str] = None,
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

        # 0. Check Clinical Device Capacity (Insulin Pens, Inhalers, Eye Drops)
        from refillcare.data.packing import get_device_clinical_capacity
        latest_pack = packings[-1] if packings and len(packings) > 0 else None
        latest_qty = quantities[-1] if quantities and len(quantities) > 0 else 1.0
        try:
            latest_qty = float(latest_qty) if latest_qty is not None and float(latest_qty) > 0 else 1.0
        except (TypeError, ValueError):
            latest_qty = 1.0
        device_cap = get_device_clinical_capacity(item_name, latest_pack, salt_name)
        if device_cap is not None:
            # Clinical stability ceiling is authoritative (e.g., 30.0 days per pen/canister)
            capped_dos = device_cap * latest_qty
            return round(float(capped_dos), 1), round(float(latest_qty), 1)

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
                    # Physiological clamp for oral solids:
                    # Patient non-adherent purchase gaps must not inflate supply estimation.
                    # Standard oral chronic solid dosing is 1 tablet/day (QD) to 2 tablets/day (BID).
                    min_allowed = max(5.0, 0.5 * latest_units)
                    max_allowed = max(10.0, 1.25 * latest_units)
                    dos_clamped = max(min_allowed, min(max_allowed, dos))
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
        transaction_numbers: Optional[List[str]] = None,
        salt_name: Optional[str] = None,
        mobile_history: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefillDecision:
        """Evaluate a single customer-item purchase history deterministically.

        ``mobile_history`` (mobile number per purchase, chronological) enables the identity guard.
        ``context`` = ``refillcare.features.context.build_context(...)`` (visit + salt priors);
        when omitted only the engine's salt priors are used.
        """
        dec = self._evaluate_core(
            customer_id=customer_id,
            customer_name=customer_name,
            mobile_no=mobile_no,
            item_id=item_id,
            item_name=item_name,
            dates=dates,
            quantities=quantities,
            packings=packings,
            as_of_date=as_of_date,
            transaction_numbers=transaction_numbers,
            salt_name=salt_name,
            context=context,
        )
        return self._finalize(dec, mobile_history)

    def _evaluate_core(
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
        transaction_numbers: Optional[List[str]] = None,
        salt_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefillDecision:
        """Evaluate a single customer-item purchase history deterministically.

        Excludes any B2B (SB/) or Unknown transactions if transaction_numbers are provided.
        """
        # Upstream transaction channel filter: Exclude B2B / Inter-Store (SB/) transactions
        if transaction_numbers:
            from refillcare.data.transaction_classifier import classify_transaction
            valid_indices = [
                idx for idx, tn in enumerate(transaction_numbers)
                if classify_transaction(tn)[1]
            ]
            if len(valid_indices) < len(dates):
                dates = [dates[i] for i in valid_indices if i < len(dates)]
                if quantities:
                    quantities = [quantities[i] for i in valid_indices if i < len(quantities)]
                if packings:
                    packings = [packings[i] for i in valid_indices if i < len(packings)]

        # Sort chronologically and drop future transactions while keeping quantities / packings aligned
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        if as_of_date:
            order = [i for i in order if dates[i] <= as_of_date]
        quantities = [quantities[i] if i < len(quantities) else None for i in order] if quantities else quantities
        packings = [packings[i] if i < len(packings) else None for i in order] if packings else packings
        dates = [dates[i] for i in order]
        purchase_count = len(dates)
        cust_item_key = f"{str(customer_id).strip()}_{str(item_id).strip()}"

        # Clean mobile number
        clean_phone = None
        if mobile_no and str(mobile_no).strip():
            p_norm, _ = normalize_to_whatsapp_number(mobile_no)
            clean_phone = p_norm

        if purchase_count == 0:
            return RefillDecision(
                customer_id=str(customer_id),
                customer_name=str(customer_name),
                mobile_no=clean_phone,
                item_id=str(item_id),
                item_name=str(item_name),
                customer_item_key=cust_item_key,
                path=PATH_INELIGIBLE,
                purchase_count=0,
                is_eligible=False,
                stability_tier=STABILITY_UNSTABLE,
                cadence_median=None,
                cadence_norm_mad=None,
                cadence_drift=None,
                dos_days=None,
                units_purchased=None,
                prediction_method=PRED_NONE,
                predicted_interval_days=None,
                last_purchase_date=as_of_date or date.today(),
                expected_refill_date=None,
                decision_reason="No eligible customer transactions found (B2B/Inter-Store or unknown transactions excluded).",
                cycle_id=f"RC-{customer_id}-{item_id}-NONE",
            )

        last_date = dates[-1]

        # Compute consecutive intervals
        intervals: List[float] = []
        if len(dates) >= 2:
            intervals = [float((dates[i] - dates[i - 1]).days) for i in range(1, len(dates)) if (dates[i] - dates[i - 1]).days > 0]

        # Compute DOS/EDS
        dos_days, latest_units = self.compute_dos(dates, quantities, packings, item_name, salt_name)

        # Cross-item context: visit features (if supplied) + salt priors (engine-level)
        ctx: Dict[str, Any] = dict(context or {})
        if "salt_stats" not in ctx and self.population_stats is not None:
            from refillcare.features.context import salt_key

            ctx["salt_stats"] = self.population_stats.lookup(salt_key(salt_name, item_name))

        # Medication therapy classification gate: Exclude generic OTC / FMCG / Acute / Personal Care items
        from refillcare.data.medication_classifier import verify_medication_chronic_status
        latest_pack = packings[-1] if packings and len(packings) > 0 else None
        int_med = float(np.median(intervals)) if intervals else None
        med_class = verify_medication_chronic_status(
            item_name=item_name,
            patient_history_count=purchase_count,
            patient_median_interval=int_med,
            packing=latest_pack,
            salt_name=salt_name,
        )
        if not med_class["is_chronic_eligible"]:
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
                decision_reason=f"Excluded: {med_class['exclusion_reason']} ({med_class['category']})",
                cycle_id=f"RC-{customer_id}-{item_id}-NONCHRONIC",
            )

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

                cad_med_val = float(cadence_med) if cadence_med is not None else 30.0
                latest_u = units_hist[-1] if units_hist else (latest_units or 30.0)
                prior_u = units_hist[:-1] if len(units_hist) > 1 else units_hist
                recent_prior = prior_u[-5:] if len(prior_u) >= 5 else prior_u
                typical_u = float(np.median(recent_prior)) if recent_prior else latest_u
                qty_ratio = latest_u / typical_u if typical_u > 0 else 1.0
                latest_gap = (dates[-1] - dates[-2]).days if len(dates) >= 2 else 0
                is_lapsed_restart = (latest_gap > 1.5 * cad_med_val)

                # Check 1: Bulk Purchase Safety Guardrail (>=90 units and extreme divergence)
                is_bulk = (
                    latest_u >= 90.0
                    and (
                        (dos_days is not None and (dos_days - cad_med_val) > 45.0)
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
                        cadence_median=cad_med_val,
                        cadence_norm_mad=norm_mad,
                        cadence_drift=drift,
                        dos_days=dos_days,
                        units_purchased=latest_units,
                        prediction_method=PRED_PATH_A_HISTORICAL_MEDIAN,
                        predicted_interval_days=None,
                        last_purchase_date=last_date,
                        expected_refill_date=None,
                        decision_reason=f"BULK_PURCHASE_DIVERGENCE: Historical median={int(round(cad_med_val))}d vs Latest Units={int(latest_u)}u (DOS={round(dos_days, 1) if dos_days else '-'}d) indicates bulk purchase. Pharmacist review required.",
                        cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
                    )

                # Multi-Signal Human-Level Consensus & Point-in-Time Features
                from refillcare.features.point_in_time import compute_point_in_time_features_single
                from refillcare.models.supply_hybrid import evaluate_human_hybrid_prediction

                feats = compute_point_in_time_features_single(
                    dates=dates,
                    quantities=quantities,
                    packings=packings,
                    item_name=item_name,
                    context=ctx,
                )

                from refillcare.data.packing import get_device_clinical_capacity

                device_cap_unit = get_device_clinical_capacity(item_name, packings[-1] if packings else None, salt_name)
                device_cap_days = None
                if device_cap_unit is not None:
                    q_last = quantities[-1] if quantities else 1.0
                    try:
                        q_last = float(q_last) if q_last is not None and float(q_last) > 0 else 1.0
                    except (TypeError, ValueError):
                        q_last = 1.0
                    device_cap_days = device_cap_unit * q_last
                    from refillcare.engine.prediction_rules import device_ceiling_days

                    device_cap_days = device_ceiling_days(self.rules.device_cap_mode, device_cap_days, cad_med_val)

                provider = self.model_provider
                prod_kwargs = self._consensus_kwargs(provider.production if provider is not None else None)
                routed_to_median = None
                if prod_kwargs.get("use_ml", True):
                    from refillcare.data.packing import is_insulin_product

                    from refillcare.engine.prediction_rules import is_normal_quantity

                    if self.rules.high_stability_median and stability_tier == STABILITY_HIGH and is_normal_quantity(feats):
                        routed_to_median = "high-stability"
                    elif self.rules.insulin_median and is_insulin_product(item_name, packings[-1] if packings else None, salt_name):
                        routed_to_median = "insulin"
                    if routed_to_median:
                        prod_kwargs = {"use_ml": False, "model_version": prod_kwargs.get("model_version")}
                consensus = evaluate_human_hybrid_prediction(
                    features=feats,
                    stability_tier=stability_tier,
                    cadence_median=cad_med_val,
                    device_cap_days=device_cap_days,
                    **prod_kwargs,
                )
                shadow_consensus = None
                if provider is not None and provider.shadow is not None:
                    shadow_consensus = evaluate_human_hybrid_prediction(
                        features=feats,
                        stability_tier=stability_tier,
                        cadence_median=cad_med_val,
                        device_cap_days=device_cap_days,
                        **self._consensus_kwargs(provider.shadow),
                    )

                predicted_interval = consensus["predicted_interval_days"]
                reason_rule = consensus["consensus_reason"]
                review_required = consensus["review_required"]
                if routed_to_median:
                    reason_rule = f"{reason_rule} [rule: {routed_to_median} -> personal median]"
                if self.rules.dom_anchor and not review_required:
                    from refillcare.engine.prediction_rules import dom_anchor_interval

                    anchored, moved = dom_anchor_interval(last_date, predicted_interval, feats, purchase_count)
                    if moved:
                        reason_rule = f"{reason_rule} [rule: day-of-month anchor {predicted_interval}d -> {anchored}d]"
                        predicted_interval = anchored
                        consensus = {**consensus, "predicted_interval_days": anchored}

                # Check 2: Model-Supply Divergence Guardrail (routes to Pharmacist Review)
                if review_required:
                    return self._attach_consensus(RefillDecision(
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
                        decision_reason=reason_rule,
                        cycle_id=f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}",
                    ), consensus, shadow_consensus)

                exp_date = last_date + timedelta(days=predicted_interval)
                cycle_id = f"RC-{customer_id}-{item_id}-{last_date.strftime('%Y%m%d')}"

                # Check 3: Active Lifecycle Recency / Churn Gate (75 days post-due)
                if as_of_date and (as_of_date - exp_date).days > CHURN_CUTOFF_DAYS:
                    lapsed_days = (as_of_date - exp_date).days
                    return self._attach_consensus(RefillDecision(
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
                    ), consensus, shadow_consensus)

                reason = (
                    f"Path A ({stability_tier}) {reason_rule}. DOS corroborated ({round(dos_days, 1)}d)."
                    if dos_days
                    else f"Path A ({stability_tier}) {reason_rule}."
                )

                return self._attach_consensus(RefillDecision(
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
                ), consensus, shadow_consensus)

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
            # Short history: DOS/EDS, optionally blended with the personal median shrunk toward the
            # salt median (refillcare.engine.prediction_rules.path_b_interval).
            if dos_days is not None and 10.0 <= dos_days <= 180.0:
                from refillcare.engine.prediction_rules import dom_anchor_interval, path_b_interval

                pb_days, pb_method = path_b_interval(self.rules.path_b_mode, intervals, dos_days, ctx.get("salt_stats"))
                predicted_interval = int(round(pb_days if pb_days is not None else dos_days))
                if self.rules.dom_anchor:
                    from refillcare.features.point_in_time import compute_point_in_time_features_single

                    pb_feats = compute_point_in_time_features_single(dates=dates, quantities=quantities, packings=packings, item_name=item_name)
                    predicted_interval, _ = dom_anchor_interval(last_date, predicted_interval, pb_feats, purchase_count)
                path_b_sub_reason = f"{path_b_sub_reason}; method={pb_method}"
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
                    decision_reason=f"Path B Eligible ({path_b_sub_reason}): short-history prediction ({predicted_interval}d).",
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
        only_keys: Optional[set] = None,
    ) -> Tuple[pd.DataFrame, List[RefillDecision]]:
        """
        Evaluate full transaction dataframe in high-speed grouped batch.

        ``df`` must contain every purchase of the patients being scored (visit features look at
        their other medicines). ``only_keys`` = {(customerId, itemId)} limits which customer-items
        are scored (incremental re-scoring after an upload).

        Returns:
            (decisions_df, decisions_list)
        """
        if df.empty:
            return pd.DataFrame(), []

        df_clean = df.copy()

        # Upstream transaction channel filter: Strictly filter out B2B (SB/) or Unknown transactions
        if "invoice_number" in df_clean.columns or "refillcare_eligible" in df_clean.columns:
            from refillcare.data.transaction_classifier import filter_eligible_customer_transactions
            df_clean = filter_eligible_customer_transactions(df_clean, invoice_col="invoice_number")

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean["invoice_date"]):
            df_clean["invoice_date"] = pd.to_datetime(df_clean["invoice_date"], errors="coerce")

        if as_of_date:
            df_clean = df_clean.loc[df_clean["invoice_date"].dt.date <= as_of_date]

        df_clean = pd.DataFrame(df_clean).sort_values("invoice_date").reset_index(drop=True)
        # Patient = name + mobile, therapy = salt group (env switches; no-op on already-keyed rows)
        from refillcare.data.identity import normalise_trajectory_keys

        df_clean = normalise_trajectory_keys(df_clean)

        trajectories = {}
        # Plain numpy arrays for the row loop: pandas 3 keeps text in Arrow arrays, where every
        # element read is a slow Python call (it dominated the per-upload re-scoring time)
        def _col(name: str):
            return df_clean[name].to_numpy(dtype=object, na_value=None) if name in df_clean.columns else [None] * len(df_clean)

        cids = df_clean["customerId"].astype(str).to_numpy(dtype=object)
        iids = df_clean["itemId"].astype(str).to_numpy(dtype=object)
        cnames = _col("customerName")
        inames = _col("itemName")
        mobiles = _col("MOBILE_NO")
        inv_dates = df_clean["invoice_date"].values
        if "quantity" in df_clean.columns:
            # Free strips go home with the patient: billed + free quantity is the supply (customer sales only here)
            from refillcare.data.schema import supply_quantity

            qtys = supply_quantity(df_clean["quantity"], df_clean["freeQuantity"]).values if "freeQuantity" in df_clean.columns else df_clean["quantity"].values
        else:
            qtys = [None] * len(df_clean)
        packings = _col("packing")
        salt_col = next((c for c in ("salt_composition", "SALT") if c in df_clean.columns), None)
        salts = _col(salt_col) if salt_col else [None] * len(df_clean)

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
                    "mobiles": [],
                }
            t = trajectories[key]
            if bool(pd.notna(cnames[i])):
                t["cname"] = cnames[i]
            if bool(pd.notna(inames[i])):
                t["iname"] = inames[i]
            if bool(pd.notna(mobiles[i])):
                t["mobile"] = mobiles[i]
            if bool(pd.notna(salts[i])) and str(salts[i]).strip():
                t["salt"] = salts[i]
            ts = inv_dates[i]
            if bool(pd.notna(ts)):
                t["dates"].append(pd.Timestamp(ts).date())
                t["qtys"].append(qtys[i] if bool(pd.notna(qtys[i])) else None)
                t["packings"].append(packings[i] if bool(pd.notna(packings[i])) else None)
                t["mobiles"].append(mobiles[i] if bool(pd.notna(mobiles[i])) else None)

        from refillcare.features.context import PopulationStats, VisitIndex, build_context

        visit_index = VisitIndex.from_trajectories({k: t["dates"] for k, t in trajectories.items()})
        if self.population_stats is None:
            self.population_stats = PopulationStats.from_events(
                pd.DataFrame({
                    "customerId": cids, "itemId": iids, "date": inv_dates,
                    "quantity": qtys, "packing": packings, "salt": salts, "itemName": inames,
                }),
                as_of=as_of_date,
            )

        decisions: List[RefillDecision] = []
        records_list: List[Dict[str, Any]] = []

        for key, t in trajectories.items():
            if only_keys is not None and (str(t["cid"]).strip(), str(t["iid"]).strip()) not in only_keys:
                continue
            ctx_as_of = min(as_of_date, max(t["dates"])) if (as_of_date and t["dates"]) else (max(t["dates"]) if t["dates"] else as_of_date)
            context = build_context(visit_index, self.population_stats, t["cid"], t["iid"], ctx_as_of, t.get("salt"), t["iname"]) if ctx_as_of else None
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
                salt_name=t.get("salt"),
                mobile_history=t["mobiles"],
                context=context,
            )
            decisions.append(dec)
            records_list.append(dec.to_dict())

        decisions_df = pd.DataFrame(records_list)
        return decisions_df, decisions

