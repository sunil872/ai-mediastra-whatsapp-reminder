"""RefillCare Path B Recency & Multi-Month Recurrence Classifier.

Path B evaluates customers with low/recent purchase history (<6 purchases) using:
1. Latest-Purchase Recency Filter
2. 3-Month Rolling Evaluation Window: >= 2 distinct calendar months
3. Fallback: 6-Month Rolling Evaluation Window: >= 3 distinct calendar months

If satisfied -> Eligible for refill prediction & reminder scheduling.
If not satisfied -> Ineligible / Clinical Review Queue (Cold-Start).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Sequence, Union, Tuple
import pandas as pd
import numpy as np

# Path B Routing Labels
ROUTE_3M_RECURRING = "3M_RECURRING"
ROUTE_6M_RECURRING = "6M_RECURRING"
ROUTE_INELIGIBLE = "INELIGIBLE_COLD_START"

# Window Thresholds
WINDOW_3M_DAYS = 90
WINDOW_6M_DAYS = 180
MIN_DISTINCT_MONTHS_3M = 2
MIN_DISTINCT_MONTHS_6M = 3
MAX_RECENCY_STALE_DAYS = 180


def _parse_to_date(val: Any) -> Optional[date]:
    """Safely convert string, timestamp, or datetime to date object."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None


def evaluate_path_b_eligibility(
    purchase_dates: Sequence[Any],
    prediction_date: Optional[Union[date, str]] = None,
    total_purchases: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate customer-medication purchase history under the Path B Recency & Multi-Month rules.

    Decision Flow:
    1. Filter purchase dates <= prediction_date (Zero data leakage).
    2. Check Latest-Purchase Recency (latest purchase <= 180 days from cutoff).
    3. Check 3-Month Window [cutoff - 90 days, cutoff]:
       - Count distinct calendar months (YYYY-MM).
       - If distinct_months >= 2 -> ELIGIBLE (3M_RECURRING).
    4. If not satisfied, Check 6-Month Window [cutoff - 180 days, cutoff]:
       - Count distinct calendar months (YYYY-MM).
       - If distinct_months >= 3 -> ELIGIBLE (6M_RECURRING).
    5. Otherwise -> INELIGIBLE (Cold-Start Review Queue).

    Args:
        purchase_dates: Chronological list of purchase dates for (customerId, itemId).
        prediction_date: Cutoff anchor date (defaults to latest purchase date or today).
        total_purchases: Optional total purchase count (if already computed).

    Returns:
        Dict[str, Any] containing eligibility decision, route, distinct month counts, and reasoning.
    """
    valid_dates: List[date] = []
    for d in purchase_dates:
        p_dt = _parse_to_date(d)
        if p_dt is not None:
            valid_dates.append(p_dt)

    if not valid_dates:
        return {
            "is_eligible": False,
            "path": "B",
            "route": ROUTE_INELIGIBLE,
            "distinct_months_3m": 0,
            "distinct_months_6m": 0,
            "days_since_latest": None,
            "latest_purchase_date": None,
            "prediction_date": _parse_to_date(prediction_date) or date.today(),
            "reason": "No valid historical purchase dates available",
        }

    valid_dates.sort()
    
    # Determine prediction anchor cutoff date
    if prediction_date is not None:
        anchor_date = _parse_to_date(prediction_date) or date.today()
    else:
        anchor_date = valid_dates[-1]

    # Strict Zero-Leakage Filter: only purchases <= anchor_date
    history_dates = [d for d in valid_dates if d <= anchor_date]
    if not history_dates:
        return {
            "is_eligible": False,
            "path": "B",
            "route": ROUTE_INELIGIBLE,
            "distinct_months_3m": 0,
            "distinct_months_6m": 0,
            "days_since_latest": None,
            "latest_purchase_date": None,
            "prediction_date": anchor_date,
            "reason": "No purchase history on or before prediction cutoff date",
        }

    latest_date = history_dates[-1]
    days_since_latest = (anchor_date - latest_date).days

    # 1. Latest-Purchase Recency Filter
    if days_since_latest > MAX_RECENCY_STALE_DAYS:
        return {
            "is_eligible": False,
            "path": "B",
            "route": ROUTE_INELIGIBLE,
            "distinct_months_3m": 0,
            "distinct_months_6m": 0,
            "days_since_latest": days_since_latest,
            "latest_purchase_date": latest_date.isoformat(),
            "prediction_date": anchor_date,
            "reason": f"Latest purchase is stale ({days_since_latest} days ago > {MAX_RECENCY_STALE_DAYS} days limit)",
        }

    # 2. 3-Month Window Evaluation (trailing 90 days from cutoff)
    cutoff_3m = anchor_date - timedelta(days=WINDOW_3M_DAYS)
    dates_3m = [d for d in history_dates if d >= cutoff_3m]
    months_3m = set(d.strftime("%Y-%m") for d in dates_3m)
    distinct_months_3m = len(months_3m)

    if distinct_months_3m >= MIN_DISTINCT_MONTHS_3M:
        return {
            "is_eligible": True,
            "path": "B",
            "route": ROUTE_3M_RECURRING,
            "distinct_months_3m": distinct_months_3m,
            "distinct_months_6m": len(set(d.strftime("%Y-%m") for d in history_dates if d >= anchor_date - timedelta(days=WINDOW_6M_DAYS))),
            "qualifying_dates": [d.isoformat() for d in dates_3m],
            "days_since_latest": days_since_latest,
            "latest_purchase_date": latest_date.isoformat(),
            "prediction_date": anchor_date,
            "reason": f"Satisfied 3-month evaluation (>= {MIN_DISTINCT_MONTHS_3M} distinct months: {sorted(list(months_3m))})",
        }

    # 3. 6-Month Window Evaluation (trailing 180 days from cutoff)
    cutoff_6m = anchor_date - timedelta(days=WINDOW_6M_DAYS)
    dates_6m = [d for d in history_dates if d >= cutoff_6m]
    months_6m = set(d.strftime("%Y-%m") for d in dates_6m)
    distinct_months_6m = len(months_6m)

    if distinct_months_6m >= MIN_DISTINCT_MONTHS_6M:
        return {
            "is_eligible": True,
            "path": "B",
            "route": ROUTE_6M_RECURRING,
            "distinct_months_3m": distinct_months_3m,
            "distinct_months_6m": distinct_months_6m,
            "qualifying_dates": [d.isoformat() for d in dates_6m],
            "days_since_latest": days_since_latest,
            "latest_purchase_date": latest_date.isoformat(),
            "prediction_date": anchor_date,
            "reason": f"Satisfied 6-month evaluation (>= {MIN_DISTINCT_MONTHS_6M} distinct months: {sorted(list(months_6m))})",
        }

    # 4. Ineligible / Cold-Start
    return {
        "is_eligible": False,
        "path": "B",
        "route": ROUTE_INELIGIBLE,
        "distinct_months_3m": distinct_months_3m,
        "distinct_months_6m": distinct_months_6m,
        "days_since_latest": days_since_latest,
        "latest_purchase_date": latest_date.isoformat(),
        "prediction_date": anchor_date,
        "reason": f"Insufficient distinct monthly recurrence ({distinct_months_3m} in 3m < {MIN_DISTINCT_MONTHS_3M}, {distinct_months_6m} in 6m < {MIN_DISTINCT_MONTHS_6M})",
    }
