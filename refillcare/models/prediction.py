"""Inference and expected refill date calculation for RefillCare.

Implements Phase 17D approved strategy ``hybrid_routing_v17d``:
- Core regular eligible rows -> personal historical median
- Secondary eligible rows -> model_bundle pipeline (improved / quantile XGB when trained)
- Otherwise -> REJECTED (no automated predicted refill date)

Expected date: purchase_date + round(max(1, predicted_days)).
Reminder stages are not modified here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from refillcare.models.hybrid_strategy import (
    STRATEGY_NAME,
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_REJECTED,
    CONF_INVALID,
    ROUTE_CORE,
    ROUTE_SECONDARY,
    ROUTE_REJECTED,
    evaluate_hybrid_eligibility,
    personal_median_prediction,
)
from refillcare.data.packing import compute_total_units_purchased
from refillcare.features.consumption import (
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)


def _row_to_frame(row_or_record: Union[pd.Series, Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(row_or_record, dict):
        return pd.DataFrame([row_or_record])
    if isinstance(row_or_record, pd.Series):
        return pd.DataFrame([row_or_record.to_dict()])
    return row_or_record.copy()


def _purchase_date_from_df(df: pd.DataFrame) -> pd.Timestamp:
    if "invoice_date" in df.columns:
        val = df["invoice_date"].iloc[0]
        if pd.notna(val):
            return pd.to_datetime(val)
    return pd.Timestamp.now()


def _model_predict_days(model_bundle: Dict[str, Any], df: pd.DataFrame) -> float:
    pipeline = model_bundle["pipeline"]
    num_cols = model_bundle.get("features_numeric", [])
    cat_cols = model_bundle.get("features_categorical", [])
    feature_cols = list(num_cols) + list(cat_cols)
    work = df.copy()
    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
    raw = float(pipeline.predict(work[feature_cols])[0])
    return max(1.0, round(raw, 1))


def _model_predict_days_batch(model_bundle: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    pipeline = model_bundle["pipeline"]
    num_cols = model_bundle.get("features_numeric", [])
    cat_cols = model_bundle.get("features_categorical", [])
    feature_cols = list(num_cols) + list(cat_cols)
    work = df.copy()
    for c in feature_cols:
        if c not in work.columns:
            work[c] = np.nan
    raw = np.asarray(pipeline.predict(work[feature_cols]), dtype=np.float64)
    return np.round(np.clip(raw, 1.0, None), 1)


def _finalize_accepted(
    *,
    customer_id: str,
    item_id: str,
    item_name: str,
    purchase_date: pd.Timestamp,
    predicted_days: float,
    purchase_count: int,
    decision: Dict[str, Any],
    supply_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    predicted_days = max(1.0, float(predicted_days))
    days_offset = int(round(predicted_days))
    expected_date = purchase_date + timedelta(days=days_offset)
    res = {
        "customerId": customer_id,
        "itemId": item_id,
        "itemName": item_name,
        "current_purchase_date": str(purchase_date.date()),
        "predicted_days_until_refill": predicted_days,
        "expected_refill_date": str(expected_date.date()),
        "refill_confidence": decision["confidence"],
        "purchase_count_so_far": purchase_count,
        "prediction_status": "eligible",
        "prediction_route": decision["route"],
        "prediction_strategy": STRATEGY_NAME,
        "rejection_reason": None,
        "norm_mad": decision.get("norm_mad"),
        "cadence_drift": decision.get("cadence_drift"),
        "pack_default_days": decision.get("pack_default_days"),
        "total_units_purchased": None,
        "estimated_daily_consumption": None,
        "estimated_days_of_supply": None,
        "reminder_date": None,
        "client_status_label": decision.get("client_status_label"),
        "switch_detected": decision.get("switch_detected", False),
    }
    if supply_info:
        res.update(supply_info)
    return res


def _finalize_rejected(
    *,
    customer_id: str,
    item_id: str,
    item_name: str,
    purchase_date: pd.Timestamp,
    purchase_count: int,
    decision: Dict[str, Any],
    confidence: str = CONF_REJECTED,
    supply_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    res = {
        "customerId": customer_id,
        "itemId": item_id,
        "itemName": item_name,
        "current_purchase_date": str(purchase_date.date()),
        "predicted_days_until_refill": None,
        "expected_refill_date": None,
        "refill_confidence": confidence,
        "purchase_count_so_far": purchase_count,
        "prediction_status": "rejected",
        "prediction_route": ROUTE_REJECTED,
        "prediction_strategy": STRATEGY_NAME,
        "rejection_reason": decision.get("rejection_reason") or "Rejected by Phase 17D hybrid eligibility.",
        "norm_mad": decision.get("norm_mad"),
        "cadence_drift": decision.get("cadence_drift"),
        "pack_default_days": decision.get("pack_default_days"),
        "total_units_purchased": None,
        "estimated_daily_consumption": None,
        "estimated_days_of_supply": None,
        "reminder_date": None,
        "client_status_label": decision.get("client_status_label"),
        "switch_detected": decision.get("switch_detected", False),
    }
    if supply_info:
        res.update(supply_info)
    return res


def predict_refill_date(
    model_bundle: Dict[str, Any],
    row_or_record: Union[pd.Series, Dict[str, Any], pd.DataFrame],
) -> Dict[str, Any]:
    """Generate expected refill prediction using Phase 17D hybrid routing with supply supporting fields.

    Rejected rows return ``predicted_days_until_refill=None`` and
    ``refill_confidence='REJECTED'`` (no automated reminder date).
    """
    df = _row_to_frame(row_or_record)
    purchase_date = _purchase_date_from_df(df)
    rec = df.iloc[0].to_dict()
    customer_id = str(rec.get("customerId", "UNKNOWN"))
    item_id = str(rec.get("itemId", "UNKNOWN"))
    item_name = str(rec.get("itemName", "UNKNOWN"))
    purchase_count = int(rec.get("purchase_count_so_far", rec.get("purchase_seq", 1)) or 1)

    # Compute supply info if packing and consumption rate are available
    supply_info: Dict[str, Any] = {}
    total_units = rec.get("total_units_purchased")
    if total_units is None or (isinstance(total_units, float) and np.isnan(total_units)):
        total_units = compute_total_units_purchased(rec.get("quantity"), rec.get("packing"))

    rate = rec.get("historical_consumption_rate", rec.get("estimated_daily_consumption"))
    if total_units is not None and rate is not None and float(rate) > 0 and float(total_units) > 0:
        dos_res = calculate_estimated_days_of_supply(float(total_units), float(rate))
        if dos_res["status"] == "valid" and dos_res["estimated_days_of_supply"] is not None:
            dos_val = float(dos_res["estimated_days_of_supply"])
            supply_info["total_units_purchased"] = total_units
            supply_info["historical_consumption_rate"] = round(float(rate), 4)
            supply_info["estimated_daily_consumption"] = round(float(rate), 4)
            supply_info["estimated_days_of_supply"] = round(dos_val, 2)
            rem_res = calculate_target_refill_interval(dos_val, buffer_days=2.0)
            if rem_res["status"] == "valid":
                rem_offset = int(round(rem_res["target_refill_interval"]))
                supply_info["reminder_date"] = str((purchase_date + timedelta(days=rem_offset)).date())
                supply_info["target_refill_interval"] = round(rem_res["target_refill_interval"], 2)

    decision = evaluate_hybrid_eligibility(rec)

    if not decision["is_eligible"]:
        return _finalize_rejected(
            customer_id=customer_id,
            item_id=item_id,
            item_name=item_name,
            purchase_date=purchase_date,
            purchase_count=purchase_count,
            decision=decision,
            supply_info=supply_info,
        )

    try:
        if decision["route"] == ROUTE_CORE:
            predicted_days = personal_median_prediction(rec)
        else:
            predicted_days = _model_predict_days(model_bundle, df)
        if predicted_days is None or not np.isfinite(predicted_days) or predicted_days < 0:
            decision = {
                **decision,
                "rejection_reason": f"Invalid routed prediction value: {predicted_days}",
            }
            return _finalize_rejected(
                customer_id=customer_id,
                item_id=item_id,
                item_name=item_name,
                purchase_date=purchase_date,
                purchase_count=purchase_count,
                decision=decision,
                confidence=CONF_INVALID,
                supply_info=supply_info,
            )
        return _finalize_accepted(
            customer_id=customer_id,
            item_id=item_id,
            item_name=item_name,
            purchase_date=purchase_date,
            predicted_days=predicted_days,
            purchase_count=purchase_count,
            decision=decision,
            supply_info=supply_info,
        )
    except Exception as exc:  # noqa: BLE001 — convert to rejection, never crash callers
        decision = {**decision, "rejection_reason": f"Prediction failure: {exc}"}
        return _finalize_rejected(
            customer_id=customer_id,
            item_id=item_id,
            item_name=item_name,
            purchase_date=purchase_date,
            purchase_count=purchase_count,
            decision=decision,
            confidence=CONF_INVALID,
            supply_info=supply_info,
        )


def generate_batch_predictions(
    model_bundle: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Batch hybrid predictions; rejected rows keep null predicted days / expected date."""
    if df.empty:
        out = df.copy()
        for col in (
            "predicted_days_until_refill",
            "expected_refill_date",
            "refill_confidence",
            "prediction_status",
            "prediction_route",
            "prediction_strategy",
            "rejection_reason",
            "estimated_days_of_supply",
            "estimated_daily_consumption",
        ):
            out[col] = []
        return out

    eval_df = df.copy()
    n = len(eval_df)
    pred_days = np.full(n, np.nan, dtype=np.float64)
    conf: List[str] = [CONF_REJECTED] * n
    status: List[str] = ["rejected"] * n
    route: List[str] = [ROUTE_REJECTED] * n
    reasons: List[Optional[str]] = [None] * n
    strategies: List[str] = [STRATEGY_NAME] * n

    decisions = [evaluate_hybrid_eligibility(row.to_dict()) for _, row in eval_df.iterrows()]
    eligible_idx = [i for i, d in enumerate(decisions) if d["is_eligible"]]
    core_idx = [i for i in eligible_idx if decisions[i]["route"] == ROUTE_CORE]
    secondary_idx = [i for i in eligible_idx if decisions[i]["route"] == ROUTE_SECONDARY]

    for i in core_idx:
        try:
            pred_days[i] = max(1.0, round(personal_median_prediction(eval_df.iloc[i].to_dict()), 1))
            conf[i] = CONF_HIGH
            status[i] = "eligible"
            route[i] = ROUTE_CORE
            reasons[i] = None
        except Exception as exc:  # noqa: BLE001
            conf[i] = CONF_INVALID
            status[i] = "rejected"
            route[i] = ROUTE_REJECTED
            reasons[i] = f"Prediction failure: {exc}"

    if secondary_idx:
        sub = eval_df.iloc[secondary_idx]
        try:
            secondary_preds = _model_predict_days_batch(model_bundle, sub)
            for j, i in enumerate(secondary_idx):
                pred_days[i] = float(secondary_preds[j])
                conf[i] = CONF_MEDIUM
                status[i] = "eligible"
                route[i] = ROUTE_SECONDARY
                reasons[i] = None
        except Exception as exc:  # noqa: BLE001
            for i in secondary_idx:
                conf[i] = CONF_INVALID
                status[i] = "rejected"
                route[i] = ROUTE_REJECTED
                reasons[i] = f"Secondary model failure: {exc}"

    for i, d in enumerate(decisions):
        if status[i] == "rejected" and reasons[i] is None:
            reasons[i] = d.get("rejection_reason")

    eval_df["predicted_days_until_refill"] = pred_days
    eval_df["refill_confidence"] = conf
    eval_df["prediction_status"] = status
    eval_df["prediction_route"] = route
    eval_df["prediction_strategy"] = strategies
    eval_df["rejection_reason"] = reasons

    if "invoice_date" in eval_df.columns:
        dates = pd.to_datetime(eval_df["invoice_date"])
        expected: List[Optional[Any]] = []
        for d, days in zip(dates, pred_days):
            if pd.isna(d) or not np.isfinite(days):
                expected.append(None)
            else:
                expected.append((d + timedelta(days=int(round(float(days))))).date())
        eval_df["expected_refill_date"] = expected
    else:
        eval_df["expected_refill_date"] = None

    # Calculate supply and daily consumption if units & rates available
    if "total_units_purchased" not in eval_df.columns and "quantity" in eval_df.columns and "packing" in eval_df.columns:
        eval_df["total_units_purchased"] = [
            compute_total_units_purchased(q, p)
            for q, p in zip(eval_df["quantity"], eval_df["packing"])
        ]

    rates_series = eval_df.get("historical_consumption_rate", eval_df.get("estimated_daily_consumption"))
    units_series = eval_df.get("total_units_purchased")
    if units_series is not None and rates_series is not None:
        dos_vals = []
        edc_vals = []
        for u, r in zip(units_series, rates_series):
            if pd.notna(u) and pd.notna(r) and float(r) > 0 and float(u) > 0:
                dos_c = calculate_estimated_days_of_supply(float(u), float(r))
                if dos_c["status"] == "valid":
                    dos_vals.append(dos_c["estimated_days_of_supply"])
                    edc_vals.append(round(float(r), 4))
                else:
                    dos_vals.append(np.nan)
                    edc_vals.append(np.nan)
            else:
                dos_vals.append(np.nan)
                edc_vals.append(np.nan)
        eval_df["estimated_days_of_supply"] = dos_vals
        eval_df["estimated_daily_consumption"] = edc_vals
        eval_df["historical_consumption_rate"] = edc_vals
    else:
        eval_df["estimated_days_of_supply"] = np.nan
        eval_df["estimated_daily_consumption"] = np.nan

    return eval_df
