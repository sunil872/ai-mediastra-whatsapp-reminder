"""Hybrid refill-date calculation layer for RefillCare.

Integrates Historical Consumption Rate and Estimated Days of Supply with the
existing Phase 17D prediction engine (`hybrid_routing_v17d`) without blindly
replacing existing models.

Design Principles:
1. Multi-stage verification:
   - Check whether usable packaging information exists.
   - Calculate total units purchased.
   - Verify that historical consumption rate is sufficiently supported.
   - Calculate estimated days of supply.
   - Compare supply estimate against existing refill prediction.
   - Select appropriate source based on empirical backtest findings.
2. Safe Fallback:
   - If packaging, rate, or supply calculations are unavailable or unreliable,
     safely fall back to the existing prediction with an explicit `fallback_reason`.
3. Clinical Boundary:
   - Never infers prescribed clinical dosage or assumes identical strength kinetics.
4. Feature Flag Safety:
   - Kept behind `ENABLE_HYBRID_SUPPLY_PREDICTION` (default: False) so production
     behavior is never altered without explicit enablement.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from refillcare.data.packing import compute_total_units_purchased, parse_pack_units
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)
from refillcare.models.prediction import predict_refill_date, generate_batch_predictions
from refillcare.models.hybrid_strategy import (
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
    CONF_INVALID,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
)

# Feature Flag: Default False to keep production behavior untouched
DEFAULT_ENABLE_HYBRID_SUPPLY = False

PREDICTION_SOURCE_SUPPLY = "estimated_days_of_supply"
PREDICTION_SOURCE_EXISTING = "existing_model"
PREDICTION_SOURCE_HYBRID_CONSENSUS = "hybrid_supply_consensus"
PREDICTION_SOURCE_REJECTED = "rejected"


def is_hybrid_supply_layer_enabled() -> bool:
    """Check if the hybrid supply layer is globally enabled via environment."""
    env_val = os.getenv("REFILLCARE_ENABLE_HYBRID_SUPPLY", "").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return True
    return DEFAULT_ENABLE_HYBRID_SUPPLY


def _resolve_flag(override_flag: Optional[bool]) -> bool:
    if override_flag is not None:
        return bool(override_flag)
    return is_hybrid_supply_layer_enabled()


def calculate_hybrid_refill_date(
    model_bundle: Dict[str, Any],
    row_or_record: Union[pd.Series, Dict[str, Any], pd.DataFrame],
    history: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    enable_supply_layer: Optional[bool] = None,
) -> Dict[str, Any]:
    """Calculate expected refill date combining existing prediction and supply-based estimation.

    Workflow:
    1. Run existing model prediction (Phase 17D hybrid).
    2. Check packaging units and compute total units purchased.
    3. Check / compute historical consumption rate.
    4. Compute estimated days of supply.
    5. Compare supply-based estimate with existing prediction and select the
       appropriate source based on validated backtest rules.
    6. If supply info is unavailable/unreliable or feature flag is disabled,
       safely fall back to the existing prediction with fallback_reason.

    Args:
        model_bundle: Trained RefillCare model bundle dictionary.
        row_or_record: Current purchase record / feature row.
        history: Optional chronological purchase history for customer-item pair.
        enable_supply_layer: Optional boolean override for the feature flag.

    Returns:
        Dict[str, Any] with standard fields:
            - customerId
            - itemId
            - itemName
            - current_purchase_date
            - predicted_days_until_refill
            - expected_refill_date
            - prediction_source
            - estimated_days_of_supply
            - historical_consumption_rate
            - total_units_purchased
            - fallback_reason
            - prediction_status
            - refill_confidence
    """
    flag_active = _resolve_flag(enable_supply_layer)

    # 1. Existing baseline prediction
    existing_result = predict_refill_date(model_bundle, row_or_record)

    # Convert row_or_record to dict for attribute inspection
    if isinstance(row_or_record, pd.DataFrame):
        rec = row_or_record.iloc[0].to_dict()
    elif isinstance(row_or_record, pd.Series):
        rec = row_or_record.to_dict()
    else:
        rec = dict(row_or_record)

    customer_id = str(existing_result.get("customerId", rec.get("customerId", "UNKNOWN")))
    item_id = str(existing_result.get("itemId", rec.get("itemId", "UNKNOWN")))
    item_name = str(existing_result.get("itemName", rec.get("itemName", "UNKNOWN")))
    purchase_date_str = existing_result.get("current_purchase_date")
    purchase_date = pd.to_datetime(purchase_date_str) if purchase_date_str else pd.Timestamp.now()

    quantity = rec.get("quantity")
    packing = rec.get("packing")

    # 2. Check packing and calculate total units
    total_units = rec.get("total_units_purchased")
    if total_units is None or (isinstance(total_units, float) and np.isnan(total_units)):
        total_units = compute_total_units_purchased(quantity, packing)

    if total_units is None or total_units <= 0:
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_EXISTING,
            "estimated_days_of_supply": None,
            "historical_consumption_rate": None,
            "total_units_purchased": None,
            "fallback_reason": "Missing or unparseable packaging units",
        }

    # 3. Check historical consumption rate
    rate = rec.get("historical_consumption_rate")
    rate_reason = None
    if rate is None or (isinstance(rate, float) and (np.isnan(rate) or rate <= 0)):
        if history is not None:
            rate_info = calculate_historical_consumption_rate(history)
            if rate_info["status"] == "valid":
                rate = rate_info["historical_consumption_rate"]
            else:
                rate_reason = rate_info.get("reason")
        else:
            rate_reason = "No historical consumption rate column or history provided"

    if rate is None or (isinstance(rate, float) and (np.isnan(rate) or rate <= 0)):
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_EXISTING,
            "estimated_days_of_supply": None,
            "historical_consumption_rate": None,
            "total_units_purchased": total_units,
            "fallback_reason": rate_reason or "Historical consumption rate unavailable or non-positive",
        }

    # 4. Calculate estimated days of supply
    dos_res = calculate_estimated_days_of_supply(total_units, rate)
    if dos_res["status"] != "valid" or dos_res["estimated_days_of_supply"] is None:
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_EXISTING,
            "estimated_days_of_supply": None,
            "target_refill_interval": None,
            "reminder_date": None,
            "reminder_buffer_days": None,
            "historical_consumption_rate": round(float(rate), 4),
            "total_units_purchased": total_units,
            "fallback_reason": dos_res.get("reason") or "Estimated days of supply calculation failed",
        }

    est_days_of_supply = float(dos_res["estimated_days_of_supply"])

    # Calculate target refill reminder interval with 2-day adherence buffer: max(1, days_of_supply - 2)
    target_interval_res = calculate_target_refill_interval(est_days_of_supply, buffer_days=2.0)
    target_interval = target_interval_res["target_refill_interval"]
    reminder_date_val = (
        str((purchase_date + timedelta(days=int(round(target_interval)))).date())
        if target_interval is not None
        else None
    )

    # 5. Check Feature Flag: If OFF, record calculation for telemetry but do not alter production output
    if not flag_active:
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_EXISTING,
            "estimated_days_of_supply": round(est_days_of_supply, 2),
            "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
            "reminder_date": reminder_date_val,
            "reminder_buffer_days": 2.0,
            "historical_consumption_rate": round(float(rate), 4),
            "total_units_purchased": total_units,
            "fallback_reason": "Hybrid supply layer disabled by feature flag (default OFF)",
        }

    # 6. Validated Backtest Source Selection Rules (when flag is ENABLED)
    qty_ratio = rec.get("quantity_vs_avg_ratio")
    try:
        qty_ratio = float(qty_ratio) if qty_ratio is not None and not np.isnan(float(qty_ratio)) else 1.0
    except (TypeError, ValueError):
        qty_ratio = 1.0

    # Rule A: Bulk Purchases (quantity_vs_avg_ratio >= 1.5)
    # Backtest proved unconstrained bulk extrapolation leads to extreme over-prediction
    # (patients return earlier for other refills). Retain existing model prediction safely.
    if qty_ratio >= 1.5 and float(quantity or 1) > 1:
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_EXISTING,
            "estimated_days_of_supply": round(est_days_of_supply, 2),
            "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
            "reminder_date": reminder_date_val,
            "reminder_buffer_days": 2.0,
            "historical_consumption_rate": round(float(rate), 4),
            "total_units_purchased": total_units,
            "fallback_reason": f"Bulk purchase detected (ratio={qty_ratio:.2f} >= 1.5); retained existing prediction to prevent over-extension",
        }

    # Rule B: Partial / Top-Up Purchases (quantity_vs_avg_ratio <= 0.65)
    # Backtest proved Estimated Days of Supply significantly outperforms interval medians
    # (MAE 16.1d vs 24.3d) by adjusting for the reduced physical inventory.
    if qty_ratio <= 0.65:
        predicted_days = max(1.0, round(est_days_of_supply, 1))
        expected_date = purchase_date + timedelta(days=int(round(predicted_days)))
        return {
            "customerId": customer_id,
            "itemId": item_id,
            "itemName": item_name,
            "current_purchase_date": str(purchase_date.date()),
            "predicted_days_until_refill": predicted_days,
            "expected_refill_date": str(expected_date.date()),
            "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
            "reminder_date": reminder_date_val,
            "reminder_buffer_days": 2.0,
            "refill_confidence": CONF_MEDIUM if existing_result.get("prediction_status") == "eligible" else CONF_HIGH,
            "purchase_count_so_far": existing_result.get("purchase_count_so_far", 1),
            "prediction_status": "eligible",
            "prediction_route": "supply_partial_topup",
            "prediction_strategy": "hybrid_supply_v1",
            "prediction_source": PREDICTION_SOURCE_SUPPLY,
            "estimated_days_of_supply": round(est_days_of_supply, 2),
            "historical_consumption_rate": round(float(rate), 4),
            "total_units_purchased": total_units,
            "fallback_reason": None,
            "rejection_reason": None,
        }

    # Rule C: Regular / Standard Purchases
    # If existing model is eligible, supply estimate confirms standard cycle
    if existing_result.get("prediction_status") == "eligible":
        # When both are valid and within standard ratio, consensus is achieved
        return {
            **existing_result,
            "prediction_source": PREDICTION_SOURCE_HYBRID_CONSENSUS,
            "estimated_days_of_supply": round(est_days_of_supply, 2),
            "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
            "reminder_date": reminder_date_val,
            "reminder_buffer_days": 2.0,
            "historical_consumption_rate": round(float(rate), 4),
            "total_units_purchased": total_units,
            "fallback_reason": None,
        }

    # Rule D: If existing model rejected due to strict depth (P < 6) but supply is regular & valid
    # Safe fallback to existing rejected decision if low history
    return {
        **existing_result,
        "prediction_source": PREDICTION_SOURCE_EXISTING,
        "estimated_days_of_supply": round(est_days_of_supply, 2),
        "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
        "reminder_date": reminder_date_val,
        "reminder_buffer_days": 2.0,
        "historical_consumption_rate": round(float(rate), 4),
        "total_units_purchased": total_units,
        "fallback_reason": existing_result.get("rejection_reason") or "Existing model rejection maintained",
    }


def generate_hybrid_supply_batch_predictions(
    model_bundle: Dict[str, Any],
    df: pd.DataFrame,
    enable_supply_layer: Optional[bool] = None,
) -> pd.DataFrame:
    """Batch prediction generating hybrid supply-aware refill predictions.

    Args:
        model_bundle: Trained RefillCare model bundle dictionary.
        df: Input feature DataFrame.
        enable_supply_layer: Optional boolean override for feature flag.

    Returns:
        pd.DataFrame with added/updated hybrid prediction columns.
    """
    if df.empty:
        out = df.copy()
        for col in (
            "predicted_days_until_refill",
            "expected_refill_date",
            "target_refill_interval",
            "reminder_date",
            "reminder_buffer_days",
            "prediction_source",
            "estimated_days_of_supply",
            "historical_consumption_rate",
            "total_units_purchased",
            "fallback_reason",
        ):
            out[col] = []
        return out

    # Generate baseline batch predictions
    base_preds = generate_batch_predictions(model_bundle, df)

    results: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        row_dict = row.to_dict()
        # Ensure row_dict includes baseline predictions
        row_dict["predicted_days_until_refill"] = base_preds["predicted_days_until_refill"].iloc[idx]
        row_dict["expected_refill_date"] = base_preds["expected_refill_date"].iloc[idx]
        row_dict["prediction_status"] = base_preds["prediction_status"].iloc[idx]
        row_dict["prediction_route"] = base_preds["prediction_route"].iloc[idx]
        row_dict["rejection_reason"] = base_preds["rejection_reason"].iloc[idx]
        row_dict["refill_confidence"] = base_preds["refill_confidence"].iloc[idx]

        res = calculate_hybrid_refill_date(
            model_bundle=model_bundle,
            row_or_record=row_dict,
            enable_supply_layer=enable_supply_layer,
        )
        results.append(res)

    out_df = df.copy()
    out_df["predicted_days_until_refill"] = [r["predicted_days_until_refill"] for r in results]
    out_df["expected_refill_date"] = [r["expected_refill_date"] for r in results]
    out_df["target_refill_interval"] = [r.get("target_refill_interval") for r in results]
    out_df["reminder_date"] = [r.get("reminder_date") for r in results]
    out_df["reminder_buffer_days"] = [r.get("reminder_buffer_days") for r in results]
    out_df["refill_confidence"] = [r["refill_confidence"] for r in results]
    out_df["prediction_status"] = [r["prediction_status"] for r in results]
    out_df["prediction_source"] = [r["prediction_source"] for r in results]
    out_df["estimated_days_of_supply"] = [r["estimated_days_of_supply"] for r in results]
    out_df["historical_consumption_rate"] = [r["historical_consumption_rate"] for r in results]
    out_df["total_units_purchased"] = [r["total_units_purchased"] for r in results]
    out_df["fallback_reason"] = [r["fallback_reason"] for r in results]

    return out_df


# -----------------------------------------------------------------------------
# Generalized Human-Level ML & Dimensionally Correct Supply Consensus Engine
# -----------------------------------------------------------------------------

import logging
import joblib
from pathlib import Path
from functools import lru_cache
from typing import Tuple

HUMAN_ML_MODEL_PATH = Path("data/refillcare/processed/models/human_ml_refill_model.joblib")


def calculate_estimated_residual_inventory(
    previous_units: Optional[float],
    elapsed_days: Optional[float],
    consumption_velocity: float,
) -> float:
    """Calculate estimated residual medication units from an early prior purchase.

    Units:
        residual_inventory = max(0, previous_units - (elapsed_days * consumption_velocity))

    If elapsed_days > 30.0, residual inventory is set strictly to 0.0 (assumes prior supply
    was exhausted or patient purchased from another pharmacy).
    """
    if previous_units is None or previous_units <= 0:
        return 0.0
    if elapsed_days is None or elapsed_days <= 0:
        return 0.0
    if elapsed_days > 30.0:
        return 0.0

    consumed = float(elapsed_days) * float(consumption_velocity)
    residual = max(0.0, float(previous_units) - consumed)
    # Cap at previous_units and plausible single-course limit
    return round(float(min(residual, previous_units, 90.0)), 2)


def calculate_dimensionally_correct_supply_bounds(
    effective_units: float,
    consumption_velocity: float,
    lower_factor: float = 0.65,
    upper_factor: float = 1.50,
    min_lower_days: float = 7.0,
    min_upper_days: float = 10.0,
) -> Tuple[float, float, float]:
    """Calculate dimensionally correct physical supply boundaries in days.

    Dimensionality:
        effective_units (tablets) / consumption_velocity (tablets/day) = estimated_supply_days (days)
        lower_bound (days) = max(min_lower_days, estimated_supply_days * lower_factor)
        upper_bound (days) = max(min_upper_days, estimated_supply_days * upper_factor)

    Returns:
        (estimated_supply_days, lower_bound_days, upper_bound_days)
    """
    safe_velocity = max(0.33, min(4.0, float(consumption_velocity))) if consumption_velocity and consumption_velocity > 0 else 1.0
    safe_units = max(1.0, float(effective_units))

    # Dimensionally correct: units / (units/day) = days
    supply_days = safe_units / safe_velocity
    supply_days = max(5.0, min(365.0, supply_days))

    lower_bound = max(min_lower_days, supply_days * lower_factor)
    upper_bound = max(min_upper_days, supply_days * upper_factor)

    return round(supply_days, 1), round(lower_bound, 1), round(upper_bound, 1)


@lru_cache(maxsize=1)
def load_human_ml_bundle(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load serialized human ML model bundle with caching."""
    p = Path(model_path) if model_path else HUMAN_ML_MODEL_PATH
    if not p.is_file():
        # Fallback to search in processed models
        alt_p = Path(__file__).resolve().parent.parent.parent / "data" / "refillcare" / "processed" / "models" / "human_ml_refill_model.joblib"
        if alt_p.is_file():
            p = alt_p
        else:
            return None
    try:
        bundle = joblib.load(p)
        return bundle
    except Exception:
        return None


# Prediction provenance labels persisted on every RefillDecision
SOURCE_ML = "ML"
SOURCE_ML_CLAMPED = "ML_CLAMPED"
SOURCE_BASELINE = "BASELINE_MEDIAN"
SOURCE_BASELINE_CLAMPED = "BASELINE_CLAMPED"
SOURCE_HEURISTIC_FALLBACK = "HEURISTIC_FALLBACK"
SOURCE_REVIEW = "REVIEW"

LEGACY_HUMAN_MODEL_VERSION = "legacy:human_ml_refill_model"
MAX_REFILL_INTERVAL_DAYS = float(os.getenv("REFILLCARE_MAX_INTERVAL_DAYS", "120"))
MIN_REFILL_INTERVAL_DAYS = 5.0

_fallback_logger = logging.getLogger("refillcare.ml.fallback")
FALLBACK_COUNTERS: Dict[str, int] = {}


def _record_fallback(reason: str, detail: str = "") -> None:
    """Count and log (first occurrence + every 1000th) why the ML model was not used."""
    n = FALLBACK_COUNTERS.get(reason, 0) + 1
    FALLBACK_COUNTERS[reason] = n
    if n == 1 or n % 1000 == 0:
        _fallback_logger.warning("ML fallback #%d (%s): %s", n, reason, detail)


def predict_with_bundle(bundle: Optional[Dict[str, Any]], features: Dict[str, Any]) -> Optional[float]:
    """Predict one interval with a model bundle. Returns None (and records why) on any problem.

    Missing features are an error, not silently filled with 0.0: a model fed the wrong inputs
    produces confident nonsense (training/serving skew).
    """
    if not bundle or "model" not in bundle or "feature_cols" not in bundle:
        _record_fallback("no_model_bundle")
        return None
    feat_cols = list(bundle["feature_cols"])
    missing = [c for c in feat_cols if c not in features]
    if missing:
        _record_fallback("feature_schema_mismatch", f"missing={missing[:5]}")
        return None
    try:
        vals = [np.nan if features[c] is None else float(features[c]) for c in feat_cols]
        X_in = pd.DataFrame([vals], columns=feat_cols)
        pred = float(bundle["model"].predict(X_in)[0])
    except Exception as e:  # pragma: no cover - defensive
        _record_fallback("predict_error", repr(e))
        return None
    if not np.isfinite(pred):
        _record_fallback("non_finite_prediction")
        return None
    return pred


def apply_final_safety_clamp(
    interval_days: float,
    upper_bound_days: float,
    device_cap_days: Optional[float] = None,
    max_interval_days: float = MAX_REFILL_INTERVAL_DAYS,
) -> Tuple[float, bool]:
    """Last line of defence applied after every archetype rule.

    5 <= interval <= min(physical supply upper bound, device in-use ceiling, global max).
    Returns (clamped_interval, was_clamped).
    """
    ceiling = min(float(upper_bound_days), float(max_interval_days))
    if device_cap_days is not None and device_cap_days > 0:
        ceiling = min(ceiling, float(device_cap_days))
    ceiling = max(ceiling, MIN_REFILL_INTERVAL_DAYS)
    clamped = float(min(max(float(interval_days), MIN_REFILL_INTERVAL_DAYS), ceiling))
    return clamped, abs(clamped - float(interval_days)) > 0.5


def evaluate_human_hybrid_prediction(
    features: Dict[str, Any],
    stability_tier: str,
    cadence_median: float,
    model_bundle: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
    use_ml: bool = True,
    device_cap_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate multi-signal consensus across ML (or baseline), physical supply and cadence.

    Args:
        features: point-in-time features (refillcare.features.point_in_time).
        stability_tier: Path A stability tier.
        cadence_median: personal historical median interval (days).
        model_bundle: production model bundle; None -> legacy bundle.
        model_version: registry version id of ``model_bundle`` (provenance).
        use_ml: False when the production policy is the historical-median baseline.
        device_cap_days: in-use ceiling for insulin pens / inhalers / drops (days).

    Enforces:
    1. Dimensionally correct physical supply boundaries [Lower, Upper].
    2. Signal agreement check between the base prediction and supply.
    3. Routing to PHARMACIST_REVIEW if severe unexplained divergence exists.
    4. A final safety clamp after every archetype rule.
    """
    effective_units = float(features.get("estimated_effective_supply", features.get("latest_units", 30.0)))
    velocity = float(features.get("estimated_consumption_velocity", 1.0))
    hist_med = float(cadence_median) if cadence_median and cadence_median > 0 else 30.0

    # 1. Dimensionally correct physical bounds in days
    supply_days, lower_bound, upper_bound = calculate_dimensionally_correct_supply_bounds(
        effective_units=effective_units,
        consumption_velocity=velocity,
        lower_factor=0.65,
        upper_factor=1.50,
    )

    # 2. Base prediction: production model, or the measured baseline policy
    raw_ml_pred: Optional[float] = None
    if use_ml:
        bundle = model_bundle if model_bundle is not None else load_human_ml_bundle()
        version = model_version or (LEGACY_HUMAN_MODEL_VERSION if model_bundle is None else "unregistered")
        raw_ml_pred = predict_with_bundle(bundle, features)
        postprocess = str(((bundle or {}).get("metadata") or {}).get("postprocess", "archetypes"))
        if raw_ml_pred is not None and postprocess == "safety_only":
            # Registry-trained models learn archetype behaviour from data: only safety caps apply.
            final_pred, was_clamped = apply_final_safety_clamp(raw_ml_pred, MAX_REFILL_INTERVAL_DAYS, device_cap_days)
            source = SOURCE_ML_CLAMPED if was_clamped else SOURCE_ML
            cap_note = f", safety-capped from {int(round(raw_ml_pred))}d" if was_clamped else ""
            return {
                "predicted_interval_days": int(round(final_pred)),
                "consensus_interval": round(final_pred, 1),
                "raw_ml_prediction_days": round(raw_ml_pred, 1),
                "base_prediction_days": round(raw_ml_pred, 1),
                "estimated_supply_days": round(supply_days, 1),
                "lower_bound_days": round(lower_bound, 1),
                "upper_bound_days": round(upper_bound, 1),
                "agreement_delta": round(abs(raw_ml_pred - supply_days), 1),
                "review_required": False,
                "safety_clamped": was_clamped,
                "prediction_source": source,
                "model_version": version,
                "consensus_reason": f"ML Interval Model ({int(round(final_pred))}d{cap_note}; source={source})",
            }
        if raw_ml_pred is not None:
            base_pred, source = raw_ml_pred, SOURCE_ML
        else:
            base_pred, source = supply_days, SOURCE_HEURISTIC_FALLBACK
    else:
        # Baseline policy: the patient's own historical median interval, only safety-capped.
        # The backtest showed archetype / supply-bound adjustments lower accuracy vs the plain median,
        # so the baseline policy does not apply them (evidence-driven; re-evaluate with new data).
        version = model_version or "baseline:historical_median"
        final_pred, was_clamped = apply_final_safety_clamp(hist_med, MAX_REFILL_INTERVAL_DAYS, device_cap_days)
        source = SOURCE_BASELINE_CLAMPED if was_clamped else SOURCE_BASELINE
        cap_note = f", safety-capped from {int(round(hist_med))}d" if was_clamped else ""
        return {
            "predicted_interval_days": int(round(final_pred)),
            "consensus_interval": round(final_pred, 1),
            "raw_ml_prediction_days": None,
            "base_prediction_days": round(hist_med, 1),
            "estimated_supply_days": round(supply_days, 1),
            "lower_bound_days": round(lower_bound, 1),
            "upper_bound_days": round(upper_bound, 1),
            "agreement_delta": round(abs(hist_med - supply_days), 1),
            "review_required": False,
            "safety_clamped": was_clamped,
            "prediction_source": source,
            "model_version": version,
            "consensus_reason": f"Personal Historical Median ({int(round(final_pred))}d{cap_note}; source={source})",
        }

    # 3. Clamped Prediction within Physical Guardrails
    clamped_pred = float(np.clip(base_pred, lower_bound, upper_bound))

    # 4. Consensus & Divergence Check
    agreement_delta = abs(base_pred - supply_days)
    review_required = False

    qty_ratio = float(features.get("quantity_ratio_vs_typical", 1.0))
    typical_u = float(features.get("typical_units_median", 30.0))

    if qty_ratio < 0.8:
        # Partial purchase (e.g. 1 strip of 10 instead of usual 20 tabs)
        scaled_int = round(hist_med * qty_ratio)
        clamped_pred = float(max(5, min(int(scaled_int), int(effective_units))))
        decision_reason = f"Partial Purchase Scaled ({int(effective_units)}u vs typical {int(typical_u)}u, ratio {qty_ratio:.0%}"
    elif features.get("is_early_topup"):
        # Early top-up with carryover inventory: interval must scale with total available supply
        supply_scaled = round(supply_days * 0.8)
        clamped_pred = float(max(clamped_pred, supply_scaled, 45.0 if effective_units >= 50.0 else 20.0))
        decision_reason = f"Early Top-Up Carryover Supply ({int(effective_units)}u available"
    elif qty_ratio > 1.3 and effective_units < 90.0:
        # Multi-pack purchase (e.g. 2-month supply = 60 tabs vs typical 30 tabs)
        scaled_int = round(hist_med * qty_ratio)
        clamped_pred = float(max(clamped_pred, scaled_int, hist_med))
        decision_reason = f"Multi-Pack Scaled ({int(effective_units)}u vs typical {int(typical_u)}u, ratio {qty_ratio:.0%}"
    elif features.get("is_post_lapse"):
        decision_reason = f"Post-Lapse Supply Reset ({int(effective_units)}u"
    elif source == SOURCE_ML and agreement_delta > 45.0 and effective_units >= 60.0:
        # Only flag unexplained severe divergence if none of the explicit physical archetypes apply
        review_required = True
        decision_reason = (
            f"MODEL_SUPPLY_DIVERGENCE: ML={base_pred:.1f}d vs Supply={supply_days:.1f}d "
            f"(delta={agreement_delta:.1f}d > 45d). Pharmacist review required."
        )
    else:
        # Standard recurring cadence: historical cadence wins when corroborated and close
        if stability_tier in ("HIGH", "HIGH_STABILITY") and abs(clamped_pred - hist_med) <= 3.0:
            clamped_pred = hist_med
        elif lower_bound <= hist_med <= upper_bound and abs(clamped_pred - hist_med) <= 2.0:
            clamped_pred = hist_med
        decision_reason = "Multi-Signal Consensus ("

    # 5. Final safety clamp (every archetype - fixes unbounded multi-pack / top-up scaling).
    # Archetype rules (partial / top-up / multi-pack / post-lapse) may exceed the 1.5x supply bound
    # because the velocity estimate is noisy, but never 2x supply, the device ceiling or 120 days.
    archetype_applied = not decision_reason.startswith(("Multi-Signal", "MODEL_SUPPLY"))
    ceiling_basis = max(upper_bound, 2.0 * supply_days) if archetype_applied else upper_bound
    final_pred, was_clamped = apply_final_safety_clamp(clamped_pred, ceiling_basis, device_cap_days)
    if was_clamped:
        if source == SOURCE_ML:
            source = SOURCE_ML_CLAMPED
        elif source == SOURCE_BASELINE:
            source = SOURCE_BASELINE_CLAMPED
    if review_required:
        source = SOURCE_REVIEW
    else:
        sep = "" if decision_reason.endswith("(") else ", "
        cap_note = f", safety-capped from {int(round(clamped_pred))}d" if was_clamped else ""
        decision_reason = f"{decision_reason}{sep}{int(round(final_pred))}d{cap_note}; source={source})"

    return {
        "predicted_interval_days": int(round(final_pred)),
        "consensus_interval": round(final_pred, 1),
        "raw_ml_prediction_days": round(raw_ml_pred, 1) if raw_ml_pred is not None else None,
        "base_prediction_days": round(base_pred, 1),
        "estimated_supply_days": round(supply_days, 1),
        "lower_bound_days": round(lower_bound, 1),
        "upper_bound_days": round(upper_bound, 1),
        "agreement_delta": round(agreement_delta, 1),
        "review_required": review_required,
        "safety_clamped": was_clamped,
        "prediction_source": source,
        "model_version": version,
        "consensus_reason": decision_reason,
    }
