"""Historical consumption rate and consumption velocity module for RefillCare.

Provides deterministic, leakage-safe computation of historical product consumption
rates (units consumed per day) from chronological purchase records and valid packaging units.

Important Design Principles:
- Requires at least 2 usable purchase occurrences with valid elapsed time (> 0 days).
- Strictly chronological and leakage-safe (uses only past purchases up to the evaluation event).
- Excludes invalid, unparseable, or missing unit calculations.
- Does NOT make clinical dosage assumptions or apply arbitrary dosage buckets.
- Terminology: Strictly called 'historical_consumption_rate' (units/day).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from refillcare.data.packing import parse_pack_units, compute_total_units_purchased


def calculate_historical_consumption_rate(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    date_col: str = "invoice_date",
    units_col: str = "total_units_purchased",
    quantity_col: str = "quantity",
    packing_col: str = "packing",
    min_purchases: int = 2,
) -> Dict[str, Any]:
    """Calculate the historical consumption rate for a customer-item purchase history.

    Formula:
        historical_consumption_rate = total_historical_units_consumed / elapsed_days

    For a sequence of chronological purchases at dates t_1, t_2, ..., t_N with units
    U_1, U_2, ..., U_N, the units consumed over the elapsed period (t_N - t_1)
    are the units purchased prior to the final visit (U_1 + U_2 + ... + U_{N-1}).

    Args:
        history: DataFrame or list of purchase records for a single (customerId, itemId).
        date_col: Name of the purchase date column. Default 'invoice_date'.
        units_col: Name of the total units column. If missing, computed from quantity and packing.
        quantity_col: Name of the quantity column. Default 'quantity'.
        packing_col: Name of the packing column. Default 'packing'.
        min_purchases: Minimum required usable purchases. Default 2.

    Returns:
        Dict[str, Any] containing:
            - 'historical_consumption_rate': Optional[float] (units/day)
            - 'status': 'valid' or 'insufficient_data'
            - 'total_consumed_units': Optional[float]
            - 'total_elapsed_days': Optional[int]
            - 'usable_purchase_count': int
            - 'reason': Optional[str] explaining status if insufficient
    """
    if history is None or (isinstance(history, pd.DataFrame) and history.empty) or (isinstance(history, list) and not history):
        return {
            "historical_consumption_rate": None,
            "estimated_daily_consumption": None,
            "status": "insufficient_data",
            "total_consumed_units": None,
            "total_elapsed_days": None,
            "usable_purchase_count": 0,
            "reason": "History is empty or null.",
        }

    df = pd.DataFrame(history).copy()

    # Ensure date parsing
    if date_col not in df.columns:
        return {
            "historical_consumption_rate": None,
            "estimated_daily_consumption": None,
            "status": "insufficient_data",
            "total_consumed_units": None,
            "total_elapsed_days": None,
            "usable_purchase_count": 0,
            "reason": f"Missing date column '{date_col}'.",
        }

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Compute or validate units
    if units_col not in df.columns:
        if quantity_col in df.columns and packing_col in df.columns:
            df[units_col] = [
                compute_total_units_purchased(q, p)
                for q, p in zip(df[quantity_col], df[packing_col])
            ]
        else:
            return {
                "historical_consumption_rate": None,
                "estimated_daily_consumption": None,
                "status": "insufficient_data",
                "total_consumed_units": None,
                "total_elapsed_days": None,
                "usable_purchase_count": 0,
                "reason": "Cannot calculate units: missing units_col or (quantity + packing).",
            }

    # Filter to usable positive purchases
    valid_mask = pd.to_numeric(df[units_col], errors="coerce").notna() & (df[units_col] > 0)
    usable_df = df[valid_mask].copy()

    usable_count = len(usable_df)
    if usable_count < min_purchases:
        return {
            "historical_consumption_rate": None,
            "estimated_daily_consumption": None,
            "status": "insufficient_data",
            "total_consumed_units": None,
            "total_elapsed_days": None,
            "usable_purchase_count": usable_count,
            "reason": f"Requires at least {min_purchases} usable purchases (found {usable_count}).",
        }

    first_date = usable_df[date_col].iloc[0]
    last_date = usable_df[date_col].iloc[-1]
    elapsed_days = (last_date - first_date).days

    if elapsed_days <= 0:
        return {
            "historical_consumption_rate": None,
            "estimated_daily_consumption": None,
            "status": "insufficient_data",
            "total_consumed_units": None,
            "total_elapsed_days": elapsed_days,
            "usable_purchase_count": usable_count,
            "reason": f"Elapsed time between purchases is {elapsed_days} days (requires > 0).",
        }

    # Units consumed over the elapsed period t_1 to t_N are the units bought at visits 1 to N-1
    consumed_units = float(usable_df[units_col].iloc[:-1].sum())

    if consumed_units <= 0:
        return {
            "historical_consumption_rate": None,
            "estimated_daily_consumption": None,
            "status": "insufficient_data",
            "total_consumed_units": consumed_units,
            "total_elapsed_days": elapsed_days,
            "usable_purchase_count": usable_count,
            "reason": "Total consumed units must be greater than 0.",
        }

    rate = consumed_units / float(elapsed_days)

    return {
        "historical_consumption_rate": round(rate, 4),
        "estimated_daily_consumption": round(rate, 4),
        "status": "valid",
        "total_consumed_units": round(consumed_units, 2),
        "total_elapsed_days": elapsed_days,
        "usable_purchase_count": usable_count,
        "reason": None,
    }


def calculate_estimated_daily_consumption(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    date_col: str = "invoice_date",
    units_col: str = "total_units_purchased",
    quantity_col: str = "quantity",
    packing_col: str = "packing",
    min_purchases: int = 2,
) -> Dict[str, Any]:
    """Calculate the estimated daily consumption from historical purchasing velocity.

    Alias/wrapper for calculate_historical_consumption_rate.
    """
    return calculate_historical_consumption_rate(
        history=history,
        date_col=date_col,
        units_col=units_col,
        quantity_col=quantity_col,
        packing_col=packing_col,
        min_purchases=min_purchases,
    )


def compute_expanding_consumption_rates(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    date_col: str = "invoice_date",
    units_col: str = "total_units_purchased",
    quantity_col: str = "quantity",
    packing_col: str = "packing",
) -> pd.Series:
    """Compute leakage-safe expanding historical consumption rates across a sorted DataFrame.

    For each row i in group (customerId, itemId), computes the historical consumption rate
    using strictly purchases on or before row i (visits 1 to i).

    Args:
        df: Input DataFrame containing chronological transaction records.
        group_cols: Columns defining customer-item history. Default ['customerId', 'itemId'].
        date_col: Name of purchase date column. Default 'invoice_date'.
        units_col: Name of units column. Default 'total_units_purchased'.
        quantity_col: Name of quantity column. Default 'quantity'.
        packing_col: Name of packing column. Default 'packing'.

    Returns:
        pd.Series: float64 Series of historical consumption rates aligned to df.index,
                   with NaN for cold-start (< 2 purchases) or invalid cases.
    """
    if df.empty:
        return pd.Series(dtype=np.float64, index=df.index)

    if group_cols is None:
        group_cols = ["customerId", "itemId"]

    n = len(df)
    rates = np.full(n, np.nan, dtype=np.float64)

    # Ensure dates are datetime
    dates = pd.to_datetime(df[date_col], errors="coerce").values

    # Ensure total units are available
    if units_col in df.columns:
        units = pd.to_numeric(df[units_col], errors="coerce").values
    elif quantity_col in df.columns and packing_col in df.columns:
        parsed_units = [
            compute_total_units_purchased(q, p)
            for q, p in zip(df[quantity_col], df[packing_col])
        ]
        units = np.array([u if u is not None else np.nan for u in parsed_units], dtype=np.float64)
    else:
        return pd.Series(rates, index=df.index, dtype=np.float64)

    # Extract grouping keys
    if len(group_cols) == 2:
        c_vals = df[group_cols[0]].astype(str).values
        i_vals = df[group_cols[1]].astype(str).values
        keys = list(zip(c_vals, i_vals))
    else:
        keys = list(zip(*[df[c].astype(str).values for c in group_cols]))

    # Contiguous expanding history tracker
    current_dates: List[pd.Timestamp] = []
    current_units: List[float] = []
    prev_key = None

    for idx in range(n):
        k = keys[idx]
        if k != prev_key:
            current_dates = []
            current_units = []
            prev_key = k

        d_val = dates[idx]
        u_val = units[idx]

        # Only accumulate valid positive purchases with valid dates
        if pd.notna(d_val) and not np.isnan(u_val) and u_val > 0:
            current_dates.append(pd.Timestamp(d_val))
            current_units.append(float(u_val))

        # Check if at least 2 usable purchases exist so far
        m = len(current_dates)
        if m >= 2:
            first_d = current_dates[0]
            curr_d = current_dates[-1]
            elapsed = (curr_d - first_d).days
            if elapsed > 0:
                # Sum consumed units prior to current purchase
                consumed = sum(current_units[:-1])
                if consumed > 0:
                    rates[idx] = consumed / float(elapsed)

    return pd.Series(rates, index=df.index, dtype=np.float64)


def calculate_estimated_days_of_supply(
    current_units: Optional[Union[int, float]] = None,
    historical_consumption_rate: Optional[Union[int, float]] = None,
    *,
    current_total_units: Optional[Union[int, float]] = None,
    estimated_daily_consumption: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """Calculate the estimated days of supply from current units and historical consumption rate.

    Formula:
        estimated_days_of_supply = current_total_units / estimated_daily_consumption

    Examples:
        - 60 units / 2 units per day  = 30 days
        - 120 units / 2 units per day = 60 days
        - 15 units / 2 units per day  = 7.5 days

    Important Design Principles:
        - Never divides by zero.
        - Returns unavailable when consumption rate is missing, zero, negative, or invalid.
        - Returns unavailable when current_units is missing, negative, or invalid.
        - Preserves decimal precision internally.
        - Does NOT describe the value as prescribed dosage or clinical dosage.
        - Uses the name 'estimated_days_of_supply'.

    Args:
        current_units: The number of units currently available (e.g. from the latest purchase).
        historical_consumption_rate: The historical consumption rate in units/day.
        current_total_units: Optional keyword alias for current_units.
        estimated_daily_consumption: Optional keyword alias for historical_consumption_rate.

    Returns:
        Dict[str, Any] containing:
            - 'estimated_days_of_supply': Optional[float] — the calculated supply duration in days
            - 'status': 'valid' or 'unavailable'
            - 'current_units': the input current_units echoed back
            - 'historical_consumption_rate': the input rate echoed back
            - 'estimated_daily_consumption': the input rate echoed back
            - 'reason': Optional[str] explaining why the result is unavailable
    """
    # Resolve aliases
    eff_units = current_total_units if current_total_units is not None else current_units
    eff_rate = estimated_daily_consumption if estimated_daily_consumption is not None else historical_consumption_rate

    # --- Validate current_units ---
    if eff_units is None:
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": "current_units is missing (None).",
        }

    try:
        current_units_f = float(eff_units)
    except (TypeError, ValueError):
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": f"current_units is not a valid number: {eff_units!r}.",
        }

    if np.isnan(current_units_f) or np.isinf(current_units_f):
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": f"current_units is not finite: {eff_units!r}.",
        }

    if current_units_f < 0:
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": "current_units is negative.",
        }

    # --- Validate historical_consumption_rate ---
    if eff_rate is None:
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": "historical_consumption_rate is missing (None).",
        }

    try:
        rate_f = float(eff_rate)
    except (TypeError, ValueError):
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": f"historical_consumption_rate is not a valid number: {eff_rate!r}.",
        }

    if np.isnan(rate_f) or np.isinf(rate_f):
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": f"historical_consumption_rate is not finite: {eff_rate!r}.",
        }

    if rate_f <= 0:
        return {
            "estimated_days_of_supply": None,
            "status": "unavailable",
            "current_units": eff_units,
            "historical_consumption_rate": eff_rate,
            "estimated_daily_consumption": eff_rate,
            "reason": "historical_consumption_rate must be greater than zero (cannot divide by zero or negative rate).",
        }

    # --- Compute ---
    # Zero units is valid (supply is exhausted → 0 days)
    estimated_days = current_units_f / rate_f

    return {
        "estimated_days_of_supply": round(estimated_days, 4),
        "status": "valid",
        "current_units": eff_units,
        "historical_consumption_rate": eff_rate,
        "estimated_daily_consumption": eff_rate,
        "reason": None,
    }


def calculate_target_refill_interval(
    estimated_days_of_supply: Optional[Union[int, float]],
    buffer_days: float = 2.0,
) -> Dict[str, Any]:
    """Calculate the target refill reminder interval applying the client's adherence buffer.

    Formula:
        target_refill_interval = max(1.0, estimated_days_of_supply - buffer_days)

    Use this only when estimated_days_of_supply is valid (> 0).

    Examples:
        - 30 days supply - 2 days buffer = 28 days target refill interval
        - 60 days supply - 2 days buffer = 58 days target refill interval
        - 7.5 days supply - 2 days buffer = 5.5 days target refill interval
        - 1.5 days supply - 2 days buffer = max(1.0, -0.5) = 1.0 day target interval

    Important Design Principles:
        - Ensures interval is at least 1.0 day (max(1.0, ...)).
        - Returns unavailable when estimated_days_of_supply is missing, negative, or invalid.
        - Preserves decimal precision internally.
        - Keeps expected refill date and reminder target interval distinct.

    Args:
        estimated_days_of_supply: Calculated days of supply.
        buffer_days: Lead time buffer in days for patient adherence reminder. Default 2.0.

    Returns:
        Dict[str, Any] containing:
            - 'target_refill_interval': Optional[float]
            - 'status': 'valid' or 'unavailable'
            - 'estimated_days_of_supply': input supply days echoed back
            - 'buffer_days': buffer value used
            - 'reason': Optional[str] explaining status if unavailable
    """
    if estimated_days_of_supply is None:
        return {
            "target_refill_interval": None,
            "status": "unavailable",
            "estimated_days_of_supply": None,
            "buffer_days": buffer_days,
            "reason": "estimated_days_of_supply is missing (None).",
        }

    try:
        dos_f = float(estimated_days_of_supply)
    except (TypeError, ValueError):
        return {
            "target_refill_interval": None,
            "status": "unavailable",
            "estimated_days_of_supply": estimated_days_of_supply,
            "buffer_days": buffer_days,
            "reason": f"estimated_days_of_supply is not a valid number: {estimated_days_of_supply!r}.",
        }

    if np.isnan(dos_f) or np.isinf(dos_f):
        return {
            "target_refill_interval": None,
            "status": "unavailable",
            "estimated_days_of_supply": estimated_days_of_supply,
            "buffer_days": buffer_days,
            "reason": f"estimated_days_of_supply is not finite: {estimated_days_of_supply!r}.",
        }

    if dos_f <= 0:
        return {
            "target_refill_interval": None,
            "status": "unavailable",
            "estimated_days_of_supply": dos_f,
            "buffer_days": buffer_days,
            "reason": "estimated_days_of_supply must be greater than zero.",
        }

    target_interval = max(1.0, dos_f - float(buffer_days))

    return {
        "target_refill_interval": round(target_interval, 4),
        "status": "valid",
        "estimated_days_of_supply": dos_f,
        "buffer_days": buffer_days,
        "reason": None,
    }
