"""Pack size and unit parsing utilities for RefillCare.

Provides robust, deterministic parsing of pharmacy product packaging strings
(e.g., '1X15', '10 TAB', '30 CAP', '100ML') to extract the numeric quantity of
units per package.

Important:
- This module strictly parses the physical pack size (units per container).
- It never guesses ambiguous package formats.
- It does not infer, estimate, or prescribe medical dosages.
"""

from __future__ import annotations

import re
from typing import Optional, Union, Any
import pandas as pd
import numpy as np


# Regex pattern for discrete solid multi-packs: 1X15, 1x15, 1*15, 2X10, 10X10
_SOLID_MULTIPACK_RE = re.compile(r"^(\d+)\s*[Xx\*]\s*(\d+)$")

# Regex pattern for discrete units with type suffix: 10 TAB, 30 CAP, 1 PC, 1 STRIP
_DISCRETE_UNIT_SUFFIX_RE = re.compile(
    r"^(\d+)\s*(?:TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|PC|PCS|PIECE|PIECES|STRIP|STRIPS|VIAL|VIALS|AMP|AMPOULE|AMPOULES|BOTTLE|BOTTLES|T)\.?$",
    re.IGNORECASE,
)

# Regex pattern for discrete units with type prefix: TAB 10, CAP 30, STRIP 1
_DISCRETE_UNIT_PREFIX_RE = re.compile(
    r"^(?:TAB|TABS|TABLET|TABLETS|CAP|CAPS|CAPSULE|CAPSULES|PC|PCS|PIECE|PIECES|STRIP|STRIPS)\s*(\d+)$",
    re.IGNORECASE,
)

# Regex pattern for volumes and weights: 100ML, 200ML, 10ML, 400GM, 15GM, 1L, 1KG
_VOLUME_WEIGHT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:ML|LTR|L|GM|GMS|G|MG|KG)\.?$",
    re.IGNORECASE,
)

# Regex pattern for 1X volume/weight/unit: 1X100ML, 1X50GM, 1X10ML, 1X10TAB
_SINGLE_PACK_PREFIX_RE = re.compile(
    r"^1\s*[Xx\*]\s*(\d+(?:\.\d+)?)\s*(?:ML|LTR|L|GM|GMS|G|MG|KG|PC|PCS|PIECE|TAB|TABS|TABLET|CAP|CAPS|CAPSULE)\.?$",
    re.IGNORECASE,
)

# Regex pattern for pure numeric string: 1, 10, 15, 30
_PLAIN_INTEGER_RE = re.compile(r"^(\d+)$")


def parse_pack_units(packing_str: Optional[Union[str, int, float, Any]]) -> Optional[float]:
    """Parse a packaging specification string into a numeric unit count.

    Extracts the total number of discrete units (tablets, capsules, pieces) or
    standard volume/weight measure (ml, gm) contained in a single package.

    Args:
        packing_str: Raw packing string from inventory/POS data (e.g., '1X15',
                     '10 TAB', '30 CAP', '100ML', '1PC', 15).

    Returns:
        float: The numeric units per pack (e.g., 15.0, 10.0, 30.0, 100.0) if
               the format is recognized and unambiguous.
        None: If the string is null, empty, ambiguous (e.g., '10X4.4GM', '5X3ML'),
              contains unsupported special device codes (e.g., '200MDI', '1KIT'),
              or represents non-positive numbers (e.g., '0', '-5').

    Examples:
        >>> parse_pack_units("1X15")
        15.0
        >>> parse_pack_units("1x15")
        15.0
        >>> parse_pack_units("10 TAB")
        10.0
        >>> parse_pack_units("30 CAP")
        30.0
        >>> parse_pack_units("100ML")
        100.0
        >>> parse_pack_units("10X4.4GM")
        None
    """
    if packing_str is None or pd.isna(packing_str):
        return None

    # Handle numeric input directly
    if isinstance(packing_str, (int, float)) and not isinstance(packing_str, bool):
        val = float(packing_str)
        return val if val > 0 else None

    s = str(packing_str).strip()
    if not s:
        return None

    # Remove leading/trailing typographical artifacts (quotes, backticks, trailing dots)
    s = s.strip("'\"` \t\r\n")
    if not s:
        return None

    # 1. Multi-pack discrete solids (e.g., '1X15', '1x15', '1*15', '2X10')
    m = _SOLID_MULTIPACK_RE.match(s)
    if m:
        factor1, factor2 = int(m.group(1)), int(m.group(2))
        total = factor1 * factor2
        return float(total) if total > 0 else None

    # 2. Discrete units with suffix (e.g., '10 TAB', '30 CAP', '1PC', '1 STRIP')
    m = _DISCRETE_UNIT_SUFFIX_RE.match(s)
    if m:
        val = int(m.group(1))
        return float(val) if val > 0 else None

    # 3. Discrete units with prefix (e.g., 'TAB 10', 'CAP 30')
    m = _DISCRETE_UNIT_PREFIX_RE.match(s)
    if m:
        val = int(m.group(1))
        return float(val) if val > 0 else None

    # 4. Single pack with volume/weight/unit suffix (e.g., '1X100ML', '1X50GM', '1X10TAB')
    m = _SINGLE_PACK_PREFIX_RE.match(s)
    if m:
        val = float(m.group(1))
        return val if val > 0 else None

    # 5. Volume and weight standard measures (e.g., '100ML', '200ML', '400GM', '15GM')
    m = _VOLUME_WEIGHT_RE.match(s)
    if m:
        val = float(m.group(1))
        return val if val > 0 else None

    # 6. Plain integer (e.g., '1', '10', '15', '30')
    m = _PLAIN_INTEGER_RE.match(s)
    if m:
        val = int(m.group(1))
        return float(val) if val > 0 else None

    # If ambiguous, complex multi-dimensional (e.g., '10X4.4GM', '5X3ML'),
    # device format (e.g., '200MDI', '1KIT'), or unrecognizable text -> return None
    return None


def compute_total_units_purchased(
    quantity: Optional[Union[int, float, str, Any]],
    packing: Optional[Union[str, int, float, Any]],
) -> Optional[float]:
    """Calculate the total physical units purchased from quantity and packing string.

    Formula:
        total_units_purchased = quantity * units_per_pack

    Args:
        quantity: Purchase quantity (e.g., 4 strips, 1 bottle, -3 for returns).
        packing: Product packing specification (e.g., '1X15', '10 TAB', '100ML').

    Returns:
        float: Total physical units purchased (e.g., 60.0 for 4 strips of 1X15).
        None: If packing is missing/unparseable/ambiguous, or if quantity is missing/invalid.

    Examples:
        >>> compute_total_units_purchased(4, "1X15")
        60.0
        >>> compute_total_units_purchased(8, "1X15")
        120.0
        >>> compute_total_units_purchased(1, "1X15")
        15.0
        >>> compute_total_units_purchased(4, "10 TAB")
        40.0
        >>> compute_total_units_purchased(-3, "1X15")
        -45.0
        >>> compute_total_units_purchased(4, "UNKNOWN")
        None
        >>> compute_total_units_purchased(None, "1X15")
        None
    """
    if quantity is None or pd.isna(quantity) or isinstance(quantity, bool):
        return None

    try:
        qty_num = float(quantity)
    except (ValueError, TypeError):
        return None

    units_per_pack = parse_pack_units(packing)
    if units_per_pack is None or units_per_pack <= 0:
        return None

    return qty_num * units_per_pack


def enrich_total_units_purchased(
    df: pd.DataFrame,
    quantity_col: str = "quantity",
    packing_col: str = "packing",
    output_units_col: str = "total_units_purchased",
    output_pack_units_col: Optional[str] = "units_per_pack",
) -> pd.DataFrame:
    """Enrich a DataFrame with calculated total units purchased without modifying existing columns.

    Preserves original quantity and packing columns untouched, adding total_units_purchased
    (and optionally units_per_pack) as float64 columns where missing/invalid values are NaN.

    Args:
        df: Input DataFrame containing transactions or purchase histories.
        quantity_col: Name of quantity column. Default 'quantity'.
        packing_col: Name of packing column. Default 'packing'.
        output_units_col: Name of enriched output column. Default 'total_units_purchased'.
        output_pack_units_col: Name of parsed units-per-pack column. Default 'units_per_pack'.

    Returns:
        pd.DataFrame: New DataFrame containing the added units column(s).
    """
    if df.empty:
        out = df.copy()
        out[output_units_col] = pd.Series(dtype=np.float64)
        if output_pack_units_col:
            out[output_pack_units_col] = pd.Series(dtype=np.float64)
        return out

    out = df.copy()

    # Parse packing column
    if packing_col in out.columns:
        parsed_units = [parse_pack_units(p) for p in out[packing_col]]
        pack_units_series = pd.Series(parsed_units, index=out.index, dtype=np.float64)
    else:
        pack_units_series = pd.Series(np.nan, index=out.index, dtype=np.float64)

    # Parse numeric quantities
    if quantity_col in out.columns:
        qty_series = pd.to_numeric(out[quantity_col], errors="coerce")
    else:
        qty_series = pd.Series(np.nan, index=out.index, dtype=np.float64)

    # Calculate total units
    out[output_units_col] = qty_series * pack_units_series

    if output_pack_units_col:
        out[output_pack_units_col] = pack_units_series

    return out

