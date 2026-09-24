"""Medication Switch Detection module for RefillCare.

Provides robust, deterministic detection of medication switches (X -> Y)
for customers who transition between formulations, strengths, brands, or
related pharmaceutical products.

Key Principles:
- Preserves existing prediction engine and Phase 17D hybrid routing.
- Differentiates between true medication transitions and ongoing multi-drug concurrent therapy.
- Does NOT blindly copy X's refill interval to Y.
- For newly switched Y:
  1. Uses Y's own purchase quantity and packaging.
  2. Uses X's historical consumption pattern as supporting context.
  3. Uses verified medicine-family/salt relationships where available.
- Progressively transitions reliance to Y's own history as Y purchases accumulate.
- Client-facing label: "Medication Change Detected".
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from refillcare.data.packing import compute_total_units_purchased, parse_pack_units
from refillcare.features.consumption import (
    calculate_estimated_days_of_supply,
    calculate_historical_consumption_rate,
    calculate_target_refill_interval,
)

CLIENT_LABEL_MEDICATION_CHANGE = "Medication Change Detected"

# Regex for stripping common packaging / form suffixes
_FORM_SUFFIX_RE = re.compile(
    r"\b(?:TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|SYRUP|SUSPENSION|INJ|INJECTION|DROPS|GEL|OINTMENT|CREAM|SOLUTION|RESPULES|ROTACAPS|LOTION|SPRAY|POWDER|SACHET|STRIP|BOTTLE|VIAL|AMP|KIT)\b",
    re.IGNORECASE,
)

# Regex for stripping release mechanism suffixes
_RELEASE_SUFFIX_RE = re.compile(
    r"\b(?:SR|ER|XR|CR|PR|MR|DR|XL|TR|FORTE|PLUS|DS|MAX|MINI|RETARD)\b",
    re.IGNORECASE,
)

# Regex for common medicine strength indicators: 500MG, 0.5MG, 50/500, 40 MG, 1GM, 10ML, etc.
_STRENGTH_PATTERN_RE = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*(?:MG|GM|G|MCG|ML|IU|%|M)?\b",
    re.IGNORECASE,
)


def extract_medicine_strength(item_name: Optional[str]) -> Optional[str]:
    """Extract numeric dosage strength string from medication name if present.

    Examples:
        - "GLYCOMET GP 0.5 TAB" -> "0.5"
        - "GLYCOMET GP 1 TAB" -> "1"
        - "TELMA 40MG" -> "40"
        - "JANUMET 50/500" -> "50/500"
        - "PAN 40" -> "40"
    """
    if not item_name or pd.isna(item_name):
        return None
    s = str(item_name).strip().upper()
    # Find all potential numeric strength matches
    matches = _STRENGTH_PATTERN_RE.findall(s)
    if matches:
        # Filter out obvious non-strength integers like pack markers if alone,
        # return the first prominent strength token
        for m in matches:
            clean_m = m.replace(" ", "")
            if clean_m:
                return clean_m
    return None


def extract_brand_stem(item_name: Optional[str]) -> str:
    """Normalize medication name to extract the fundamental brand / product line stem.

    Examples:
        - "GLYCOMET GP 0.5 TAB" -> "glycomet gp"
        - "GLYCOMET GP 1" -> "glycomet gp"
        - "TELMA 40 MG" -> "telma"
        - "TELMA 80 MG TAB" -> "telma"
        - "JANUMET 50/500" -> "janumet"
        - "JANUMET 50/1000" -> "janumet"
        - "AUGMENTIN 625 DUO" -> "augmentin"
    """
    if not item_name or pd.isna(item_name):
        return ""

    s = str(item_name).strip().lower()

    # Remove special punctuation
    s = re.sub(r"[^\w\s\./]", " ", s)

    # Strip release suffixes (SR, ER, FORTE, PLUS, etc.)
    s = _RELEASE_SUFFIX_RE.sub(" ", s)

    # Strip dosage forms (TAB, CAP, SYRUP, etc.)
    s = _FORM_SUFFIX_RE.sub(" ", s)

    # Strip strength tokens and numbers
    s = _STRENGTH_PATTERN_RE.sub(" ", s)

    # Clean redundant whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def classify_medicine_relationship(
    item_x: Union[Dict[str, Any], pd.Series],
    item_y: Union[Dict[str, Any], pd.Series],
    salt_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Classify the relationship between previous medicine X and new medicine Y.

    Checks:
    1. Brand Lineage: matching brand stem (e.g. Glycomet GP 0.5 vs Glycomet GP 1).
    2. Salt Composition: matching active chemical ingredients in SALT master.
    3. Salt Category: matching therapeutic classification (e.g. Anti-Diabetic).
    4. Strength Difference: whether strengths differ (for dosage change context).

    Args:
        item_x: Metadata / record for previous item X.
        item_y: Metadata / record for new item Y.
        salt_df: Optional master salt catalog.

    Returns:
        Dict[str, Any] with relationship classification details.
    """
    name_x = str(item_x.get("itemName") or item_x.get("Item Name") or item_x.get("item_name") or "")
    name_y = str(item_y.get("itemName") or item_y.get("Item Name") or item_y.get("item_name") or "")

    stem_x = extract_brand_stem(name_x)
    stem_y = extract_brand_stem(name_y)

    strength_x = extract_medicine_strength(name_x)
    strength_y = extract_medicine_strength(name_y)

    salt_x = str(item_x.get("salt_composition") or item_x.get("SALT") or "").strip().lower()
    salt_y = str(item_y.get("salt_composition") or item_y.get("SALT") or "").strip().lower()

    cat_x = str(item_x.get("salt_category") or item_x.get("CATEGORY") or "").strip().lower()
    cat_y = str(item_y.get("salt_category") or item_y.get("CATEGORY") or "").strip().lower()

    # Clean missing values
    for val in ("nan", "none", "[00]", ""):
        if salt_x == val:
            salt_x = ""
        if salt_y == val:
            salt_y = ""
        if cat_x == val:
            cat_x = ""
        if cat_y == val:
            cat_y = ""

    is_same_brand = bool(stem_x and stem_y and (stem_x == stem_y or stem_x in stem_y or stem_y in stem_x))
    is_same_salt = bool(salt_x and salt_y and salt_x == salt_y)
    is_same_category = bool(cat_x and cat_y and cat_x == cat_y)

    # Determine primary relationship
    if is_same_brand:
        rel_type = "same_brand_family"
    elif is_same_salt:
        rel_type = "same_salt_composition"
    elif is_same_category:
        rel_type = "same_salt_category"
    else:
        rel_type = "unrelated"

    is_related = rel_type != "unrelated"

    # Dosage change check
    is_dosage_change = False
    if is_related and strength_x and strength_y and strength_x != strength_y:
        is_dosage_change = True

    return {
        "relationship_type": rel_type,
        "is_related": is_related,
        "is_same_brand": is_same_brand,
        "is_same_salt": is_same_salt,
        "is_same_category": is_same_category,
        "is_dosage_change": is_dosage_change,
        "stem_x": stem_x,
        "stem_y": stem_y,
        "strength_x": strength_x,
        "strength_y": strength_y,
    }


def detect_medication_switches_for_customer(
    customer_history_df: pd.DataFrame,
    salt_df: Optional[pd.DataFrame] = None,
    max_gap_days: int = 180,
    min_x_purchases: int = 2,
) -> List[Dict[str, Any]]:
    """Detect medication transitions (X -> Y) across a customer's chronological purchase timeline.

    Rules & Business Logic:
    1. Strict Customer Timeline: Evaluates all items bought by the customer over time.
    2. Multi-Drug Concurrency Protection:
       - If a customer buys Medicine A and Medicine B concurrently across multiple dates,
         they are parallel co-prescriptions (multi-therapy), NOT a switch.
       - A switch requires that Medicine X has ceased (no ongoing purchases of X after Y is established).
    3. Switch Criteria:
       - Customer has established prior history on X (>= min_x_purchases).
       - Y's first purchase occurs around or after X's active period.
       - X is discontinued after Y is introduced.
       - Verified relationship (same brand family, same salt composition, or same category).
    4. Prevents false positive switches between unrelated medications.

    Args:
        customer_history_df: All chronological purchase records for a single customerId.
        salt_df: Optional SALT master DataFrame.
        max_gap_days: Maximum allowable gap in days between X's last purchase and Y's first purchase.
        min_x_purchases: Minimum required prior purchases for X.

    Returns:
        List[Dict[str, Any]]: Detected switch events with full previous and new medicine context.
    """
    if customer_history_df.empty:
        return []

    df = customer_history_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"]).sort_values("invoice_date").reset_index(drop=True)

    if df.empty:
        return []

    customer_id = str(df["customerId"].iloc[0])

    # Identify all distinct items purchased by this customer
    item_groups: Dict[str, pd.DataFrame] = {}
    for item_id, item_df in df.groupby("itemId", sort=False):
        item_groups[str(item_id)] = item_df.sort_values("invoice_date").reset_index(drop=True)

    if len(item_groups) < 2:
        # Customer only ever bought 1 item -> no switch possible
        return []

    # Summarize timeline for each item
    item_summaries: Dict[str, Dict[str, Any]] = {}
    for item_id, item_df in item_groups.items():
        dates = item_df["invoice_date"].tolist()
        first_date = dates[0]
        last_date = dates[-1]
        count = len(item_df)
        item_name = str(item_df["itemName"].iloc[0] if "itemName" in item_df.columns else item_id)
        packing = str(item_df["packing"].iloc[-1] if "packing" in item_df.columns else "")
        salt_comp = str(item_df["salt_composition"].iloc[0] if "salt_composition" in item_df.columns else "")
        salt_cat = str(item_df["salt_category"].iloc[0] if "salt_category" in item_df.columns else "")

        # Calculate historical consumption velocity on this item
        rate_info = calculate_historical_consumption_rate(item_df)

        item_summaries[item_id] = {
            "itemId": item_id,
            "itemName": item_name,
            "packing": packing,
            "salt_composition": salt_comp,
            "salt_category": salt_cat,
            "first_date": first_date,
            "last_date": last_date,
            "purchase_count": count,
            "all_dates": set(d.date() for d in dates),
            "date_list": dates,
            "history_df": item_df,
            "rate_info": rate_info,
        }

    detected_switches: List[Dict[str, Any]] = []

    # Evaluate all pairs (X, Y)
    for x_id, x_info in item_summaries.items():
        if x_info["purchase_count"] < min_x_purchases:
            # X does not have enough established history
            continue

        for y_id, y_info in item_summaries.items():
            if x_id == y_id:
                continue

            # 1. Timeline check: Y's first purchase must be on or after X's first purchase
            y_first = y_info["first_date"]
            x_first = x_info["first_date"]
            x_last = x_info["last_date"]

            if y_first < x_first:
                # Y was started before X -> Y cannot be the switch target from X
                continue

            # Check gap between X's last purchase and Y's first purchase
            gap_days = (y_first - x_last).days
            if gap_days > max_gap_days or gap_days < -30:
                # Too large a gap (> 180 days) or Y started long before X ended
                continue

            # 2. Concurrency Check (Crucial Guardrail against false positive multi-therapy switches):
            # Check if X and Y were purchased together across multiple invoices
            co_dates = x_info["all_dates"].intersection(y_info["all_dates"])
            if len(co_dates) >= 2:
                # Both items are regularly co-prescribed concurrently -> NOT a switch
                continue

            # Check if X continues to be purchased repeatedly long after Y starts
            # (e.g. > 1 purchase of X occurring > 15 days after Y's first purchase)
            x_purchases_after_y = [d for d in x_info["date_list"] if (d - y_first).days > 15]
            if len(x_purchases_after_y) >= 2:
                # Ongoing concurrent multi-drug therapy
                continue

            # 3. Relationship check
            rel = classify_medicine_relationship(x_info, y_info, salt_df)
            if not rel["is_related"]:
                # Do NOT assume every unrelated medicine is a switch
                continue

            # 4. Validated Switch Confirmed!
            # X rate details
            x_rate = x_info["rate_info"].get("historical_consumption_rate")
            x_intervals = x_info["history_df"]["days_since_previous_purchase"].dropna() if "days_since_previous_purchase" in x_info["history_df"].columns else pd.Series(dtype=float)
            x_median_interval = float(x_intervals.median()) if len(x_intervals) > 0 else None

            # New medicine purchase info
            y_df = y_info["history_df"]
            first_y_row = y_df.iloc[0].to_dict()
            y_qty = first_y_row.get("quantity")
            y_packing = first_y_row.get("packing")
            y_units = compute_total_units_purchased(y_qty, y_packing)

            switch_entry = {
                "customerId": customer_id,
                "previous_item_id": x_id,
                "previous_item_name": x_info["itemName"],
                "new_item_id": y_id,
                "new_item_name": y_info["itemName"],
                "switch_date": str(y_first.date()),
                "relationship_type": rel["relationship_type"],
                "is_same_brand": rel["is_same_brand"],
                "is_same_salt": rel["is_same_salt"],
                "is_dosage_change": rel["is_dosage_change"],
                "previous_purchase_count": x_info["purchase_count"],
                "previous_last_purchase_date": str(x_last.date()),
                "previous_historical_consumption_rate": x_rate,
                "previous_median_interval": x_median_interval,
                "new_first_purchase_date": str(y_first.date()),
                "new_purchase_quantity": y_qty,
                "new_packing": y_packing,
                "new_total_units": y_units,
                "client_status_label": CLIENT_LABEL_MEDICATION_CHANGE,
            }
            detected_switches.append(switch_entry)

    return detected_switches


def calculate_switched_medication_refill(
    switch_info: Dict[str, Any],
    current_y_record: Union[Dict[str, Any], pd.Series],
    y_purchase_seq: int = 1,
    y_history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate expected refill date for newly switched medicine Y using progressive history blending.

    Rules:
    - Does NOT blindly copy X's interval to Y.
    - Uses Y's own quantity, packing, and total units purchased.
    - Uses X's historical consumption pattern as supporting context.
    - Progressive history blending:
      * Purchase 1 on Y (cold start): 100% reliance on X's consumption velocity + Y's units.
      * Purchase 2 on Y: 50% reliance on Y's observed interval / consumption + 50% X's context.
      * Purchase 3+ on Y: 85-100% reliance on Y's own observed history.
    - Applies standard 2-day buffer for adherence reminder date.
    - Sets client status label: "Medication Change Detected".

    Args:
        switch_info: Switch metadata dictionary from detect_medication_switches_for_customer.
        current_y_record: Current purchase transaction for item Y.
        y_purchase_seq: Current purchase sequence number on item Y (1 for 1st purchase, 2 for 2nd, etc.).
        y_history_df: Optional history dataframe of item Y purchases up to current event.

    Returns:
        Dict[str, Any] containing prediction and supply calculation results.
    """
    if isinstance(current_y_record, pd.Series):
        rec = current_y_record.to_dict()
    else:
        rec = dict(current_y_record)

    customer_id = str(rec.get("customerId", switch_info.get("customerId", "UNKNOWN")))
    y_item_id = str(rec.get("itemId", switch_info.get("new_item_id", "UNKNOWN")))
    y_item_name = str(rec.get("itemName", switch_info.get("new_item_name", "UNKNOWN")))

    invoice_date_val = rec.get("invoice_date", rec.get("current_purchase_date"))
    purchase_date = pd.to_datetime(invoice_date_val) if invoice_date_val else pd.Timestamp.now()

    # 1. Y's physical units purchased
    y_qty = rec.get("quantity")
    y_packing = rec.get("packing")
    y_total_units = rec.get("total_units_purchased")
    if y_total_units is None or (isinstance(y_total_units, float) and np.isnan(y_total_units)):
        y_total_units = compute_total_units_purchased(y_qty, y_packing)

    x_rate = switch_info.get("previous_historical_consumption_rate")
    try:
        x_rate_f = float(x_rate) if x_rate is not None and not np.isnan(float(x_rate)) else None
    except (TypeError, ValueError):
        x_rate_f = None

    # 2. Progressive Consumption Velocity Blending
    effective_rate: Optional[float] = None
    blending_strategy: str = "switch_cold_start"

    if y_purchase_seq <= 1:
        # First Y purchase -> 100% X consumption velocity
        effective_rate = x_rate_f
        blending_strategy = "switch_p1_x_prior"
    elif y_purchase_seq == 2:
        # Second Y purchase -> blend 50% Y observed rate + 50% X prior rate
        y_rate = None
        if y_history_df is not None and len(y_history_df) >= 2:
            y_rate_info = calculate_historical_consumption_rate(y_history_df)
            if y_rate_info["status"] == "valid":
                y_rate = y_rate_info["historical_consumption_rate"]

        if y_rate is not None and x_rate_f is not None:
            effective_rate = round(0.5 * float(y_rate) + 0.5 * float(x_rate_f), 4)
            blending_strategy = "switch_p2_blend_50_50"
        elif y_rate is not None:
            effective_rate = round(float(y_rate), 4)
            blending_strategy = "switch_p2_y_only"
        else:
            effective_rate = x_rate_f
            blending_strategy = "switch_p2_x_fallback"
    else:
        # Third+ Y purchase -> 85% Y observed rate + 15% X prior rate (or 100% Y rate)
        y_rate = None
        if y_history_df is not None and len(y_history_df) >= 2:
            y_rate_info = calculate_historical_consumption_rate(y_history_df)
            if y_rate_info["status"] == "valid":
                y_rate = y_rate_info["historical_consumption_rate"]

        if y_rate is not None and x_rate_f is not None:
            effective_rate = round(0.85 * float(y_rate) + 0.15 * float(x_rate_f), 4)
            blending_strategy = "switch_p3_y_dominant"
        elif y_rate is not None:
            effective_rate = round(float(y_rate), 4)
            blending_strategy = "switch_p3_y_only"
        else:
            effective_rate = x_rate_f
            blending_strategy = "switch_p3_x_fallback"

    # 3. Calculate Days of Supply for Y
    dos_res = calculate_estimated_days_of_supply(
        current_total_units=y_total_units,
        estimated_daily_consumption=effective_rate,
    )

    if dos_res["status"] == "valid" and dos_res["estimated_days_of_supply"] is not None:
        est_days = float(dos_res["estimated_days_of_supply"])
        rem_res = calculate_target_refill_interval(est_days, buffer_days=2.0)
        target_interval = rem_res["target_refill_interval"]
        exp_date = purchase_date + timedelta(days=int(round(est_days)))
        rem_date = (
            purchase_date + timedelta(days=int(round(target_interval)))
            if target_interval is not None
            else None
        )

        return {
            "customerId": customer_id,
            "itemId": y_item_id,
            "itemName": y_item_name,
            "current_purchase_date": str(purchase_date.date()),
            "predicted_days_until_refill": round(est_days, 1),
            "expected_refill_date": str(exp_date.date()),
            "reminder_date": str(rem_date.date()) if rem_date else None,
            "target_refill_interval": round(target_interval, 2) if target_interval is not None else None,
            "estimated_days_of_supply": round(est_days, 2),
            "estimated_daily_consumption": round(effective_rate, 4) if effective_rate else None,
            "historical_consumption_rate": round(effective_rate, 4) if effective_rate else None,
            "total_units_purchased": y_total_units,
            "purchase_count_so_far": y_purchase_seq,
            "prediction_status": "eligible",
            "prediction_route": "medication_switch",
            "prediction_strategy": blending_strategy,
            "refill_confidence": "HIGH" if y_purchase_seq >= 2 else "MEDIUM",
            "client_status_label": CLIENT_LABEL_MEDICATION_CHANGE,
            "switch_detected": True,
            "previous_item_id": switch_info.get("previous_item_id"),
            "previous_item_name": switch_info.get("previous_item_name"),
            "switch_relationship": switch_info.get("relationship_type"),
        }

    # Safe fallback if units or rate cannot be determined
    return {
        "customerId": customer_id,
        "itemId": y_item_id,
        "itemName": y_item_name,
        "current_purchase_date": str(purchase_date.date()),
        "predicted_days_until_refill": None,
        "expected_refill_date": None,
        "reminder_date": None,
        "target_refill_interval": None,
        "estimated_days_of_supply": None,
        "estimated_daily_consumption": None,
        "historical_consumption_rate": None,
        "total_units_purchased": y_total_units,
        "purchase_count_so_far": y_purchase_seq,
        "prediction_status": "review_required",
        "prediction_route": "medication_switch_fallback",
        "prediction_strategy": blending_strategy,
        "refill_confidence": "REJECTED",
        "client_status_label": CLIENT_LABEL_MEDICATION_CHANGE,
        "switch_detected": True,
        "previous_item_id": switch_info.get("previous_item_id"),
        "previous_item_name": switch_info.get("previous_item_name"),
        "switch_relationship": switch_info.get("relationship_type"),
        "fallback_reason": dos_res.get("reason") or "Insufficient units or rate for switch supply calculation",
    }
