"""Path B Refill Predictor Module for Low-History / Recent Recurring Customers.

Implements the approved Path B architecture:
- Evaluates recent purchase cadence over 3-month (>=2 distinct months) and 6-month (>=3 distinct months) windows.
- Calculates authoritative Days of Supply (Pack-Aware) or Recent Inter-Purchase Median.
- Generates expected refill dates and anchored 2-day lead-time reminder dates.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from refillcare.models.path_b_classifier import (
    evaluate_path_b_eligibility,
    ROUTE_3M_RECURRING,
    ROUTE_6M_RECURRING,
    ROUTE_INELIGIBLE,
)
from refillcare.data.packing import parse_pack_units
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    calculate_estimated_days_of_supply,
)


def calculate_path_b_refill_duration(
    history_df: pd.DataFrame,
    last_qty: float = 1.0,
    last_packing: str = "1X10",
    default_days: float = 30.0,
) -> Tuple[float, str]:
    """Determine the authoritative refill duration for a Path B eligible customer.

    Hierarchy:
    1. Pack-aware Days of Supply (if valid velocity and physical pack units).
    2. Recent Inter-Purchase Median (if >= 2 purchases in evaluation window).
    3. Standard 30.0 day adherence baseline.

    Returns:
        Tuple[float, str]: (refill_duration_days, duration_source)
    """
    # 1. Try Days of Supply
    pack_units = parse_pack_units(last_packing) or 1.0
    total_units = last_qty * pack_units
    
    cons_res = calculate_historical_consumption_rate(history_df)
    cons_rate = cons_res.get("historical_consumption_rate") if isinstance(cons_res, dict) else None
    
    if cons_rate is not None and cons_rate > 0:
        dos_res = calculate_estimated_days_of_supply(current_units=total_units, historical_consumption_rate=cons_rate)
        dos = dos_res.get("estimated_days_of_supply") if isinstance(dos_res, dict) else None
        if dos is not None and 5.0 <= dos <= 180.0:
            return round(float(dos), 1), "Path B - Days of Supply (Pack-Aware)"

    # 2. Try Recent Inter-Purchase Median
    if not history_df.empty and len(history_df) >= 2:
        dates = pd.to_datetime(history_df["invoice_date"], errors="coerce").dropna().sort_values().tolist()
        if len(dates) >= 2:
            intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            valid_intervals = [iv for iv in intervals if 10 <= iv <= 180]
            if valid_intervals:
                med_interval = float(np.median(valid_intervals))
                return round(med_interval, 1), "Path B - Recent Historical Median"

    # 3. Standard Baseline
    return default_days, "Path B - Standard 30-Day Refill Baseline"


def predict_path_b_candidate(
    customer_id: str,
    item_id: str,
    history_df: pd.DataFrame,
    prediction_date: Optional[Union[date, str]] = None,
    customer_name: Optional[str] = None,
    item_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    mobile_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate and generate refill prediction snapshot for a single Path B candidate."""
    if prediction_date is None:
        anchor_date = date.today()
    elif isinstance(prediction_date, str):
        anchor_date = pd.to_datetime(prediction_date).date()
    else:
        anchor_date = prediction_date

    if history_df.empty:
        return {
            "is_eligible": False,
            "customerId": str(customer_id),
            "itemId": str(item_id),
            "reason": "Empty purchase history",
        }

    # Extract dates
    dates = history_df["invoice_date"].tolist()
    elig_result = evaluate_path_b_eligibility(
        purchase_dates=dates,
        prediction_date=anchor_date,
        total_purchases=len(history_df),
    )

    last_row = history_df.iloc[-1]
    c_name = customer_name or str(last_row.get("customerName", customer_id))
    i_name = item_name or str(last_row.get("itemName", item_id))
    phone = phone_number or str(last_row.get("MOBILE_NO", ""))
    mob_stat = mobile_status or str(last_row.get("mobile_status", "Missing"))
    last_qty = float(last_row.get("quantity", 1.0))
    last_pack = str(last_row.get("packing", "1X10"))
    last_dt_str = elig_result.get("latest_purchase_date") or str(last_row.get("invoice_date", anchor_date.isoformat()))
    last_dt = pd.to_datetime(last_dt_str).date()

    if not elig_result["is_eligible"]:
        return {
            "snapshot_id": f"snap_{customer_id}_{item_id}_{anchor_date.isoformat()}_{last_dt.isoformat()}",
            "customerId": str(customer_id),
            "itemId": str(item_id),
            "customerName": c_name,
            "itemName": i_name,
            "MOBILE_NO": phone,
            "mobile_status": mob_stat,
            "total_purchases": len(history_df),
            "last_purchase_date": last_dt.isoformat(),
            "prediction_date": anchor_date.isoformat(),
            "is_eligible": False,
            "path": "B",
            "route": elig_result["route"],
            "Reason for Ineligibility": elig_result["reason"],
            "pilot_tier": "Tier C (Cold-Start Review Queue)",
            "status": "ineligible",
        }

    # Calculate authoritative duration
    duration, source_desc = calculate_path_b_refill_duration(
        history_df=history_df,
        last_qty=last_qty,
        last_packing=last_pack,
    )

    exp_refill_dt = last_dt + timedelta(days=int(round(duration)))
    primary_rem_dt = exp_refill_dt - timedelta(days=2)  # 2-day lead buffer

    route_desc = "3-Month Recency (>=2 distinct months)" if elig_result["route"] == ROUTE_3M_RECURRING else "6-Month Recency (>=3 distinct months)"

    return {
        "snapshot_id": f"snap_{customer_id}_{item_id}_{anchor_date.isoformat()}_{last_dt.isoformat()}",
        "customerId": str(customer_id),
        "itemId": str(item_id),
        "customerName": c_name,
        "itemName": i_name,
        "MOBILE_NO": phone,
        "mobile_status": mob_stat,
        "total_purchases": len(history_df),
        "last_purchase_date": last_dt.isoformat(),
        "prediction_date": anchor_date.isoformat(),
        "estimated_days_of_supply": duration,
        "expected_refill_date": exp_refill_dt.isoformat(),
        "reminder_date": primary_rem_dt.isoformat(),
        "is_eligible": True,
        "path": "B",
        "route": elig_result["route"],
        "prediction_source": f"Path B ({route_desc})",
        "pilot_tier": "Tier B (Path B Recurrent Pilot)",
        "status": "pending",
    }
