"""Date parsing and formatting utilities for RefillCare.

Provides deterministic, unambiguous parsing of pharmacy transaction dates
with strict DD-MM-YYYY display formatting and explicit source format detection.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


UI_DATE_FORMAT = "%d-%m-%Y"


def format_date_dd_mm_yyyy(d: Any) -> str:
    """Format any date-like input to standard DD-MM-YYYY string.

    Args:
        d: Date, datetime, Timestamp, or date string.

    Returns:
        str: Date formatted as DD-MM-YYYY (e.g. '01-08-2026'), or '-' if missing/invalid.
    """
    if d is None or pd.isna(d):
        return "-"
    if isinstance(d, (datetime, date, pd.Timestamp)):
        return d.strftime(UI_DATE_FORMAT)
    # If string, attempt parsing
    s = str(d).strip()
    if not s or s.lower() in ("nan", "none", "nat", "-", "null", ""):
        return "-"
    parsed = parse_single_date(s)
    if parsed is not None:
        return parsed.strftime(UI_DATE_FORMAT)
    return str(d)


def parse_single_date(val: Any) -> Optional[datetime]:
    """Deterministically parse a single date value using pharmacy conventions."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "-", "null", ""):
        return None

    # Handle ISO formats: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(s.split(".")[0], fmt)
            except ValueError:
                continue

    # Handle standard Indian pharmacy formats: DD-MM-YYYY or DD/MM/YYYY
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}", s):
        for fmt in (
            "%d-%m-%Y %H:%M:%S", "%d-%m-%Y",
            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y",
            "%d-%b-%Y", "%d-%B-%Y",
            "%d/%b/%Y", "%d/%B/%Y",
        ):
            try:
                return datetime.strptime(s.split(".")[0], fmt)
            except ValueError:
                continue

    # Handle 2-digit year formats: DD-MM-YY or DD/MM/YY
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}$", s):
        for fmt in ("%d-%m-%y", "%d/%m/%y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

    # Fallback to robust parser with dayfirst=True
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(ts):
            return ts.to_pydatetime()
    except Exception:
        pass

    return None


def parse_pharmacy_dates(series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Parse a series of dates deterministically, detecting the format and returning metadata.

    Rules:
    1. If the series already contains datetime/Timestamp objects (e.g. from Excel), preserve directly.
    2. If string, test explicit formats prioritizing standard pharmacy DD-MM-YYYY / DD/MM/YYYY.
    3. Never silently flip days and months (e.g. 01-08-2026 is strictly 1 August 2026, NOT 8 January).
    4. Provide formatted date ranges in DD-MM-YYYY.

    Args:
        series: pd.Series of dates (strings, datetimes, timestamps).

    Returns:
        Tuple[pd.Series, Dict[str, Any]]:
            - parsed_series: pd.Series of pd.Timestamp (NaT for unparseable).
            - metadata: Summary dictionary containing detected format, formatted range, samples, etc.
    """
    if series.empty:
        return pd.Series(dtype="datetime64[ns]"), {
            "source_format": "Empty",
            "date_range_formatted": "N/A (Empty)",
            "min_date_formatted": "-",
            "max_date_formatted": "-",
            "sample_dates_formatted": [],
            "valid_count": 0,
            "invalid_count": 0,
            "is_ambiguous": False,
            "warning": None,
        }

    # Case 1: Already datetime-like (e.g. parsed by openpyxl / pd.read_excel)
    if pd.api.types.is_datetime64_any_dtype(series):
        valid_mask = series.notna()
        valid_s = series[valid_mask]
        valid_count = int(valid_mask.sum())
        invalid_count = len(series) - valid_count

        if valid_count > 0:
            min_dt = valid_s.min()
            max_dt = valid_s.max()
            samples = [d.strftime(UI_DATE_FORMAT) for d in valid_s.iloc[:5]]
            min_str = min_dt.strftime(UI_DATE_FORMAT)
            max_str = max_dt.strftime(UI_DATE_FORMAT)
            date_range = f"{min_str} to {max_str}"
        else:
            min_str = max_str = date_range = "-"
            samples = []

        return series, {
            "source_format": "Excel Native Datetime",
            "date_range_formatted": date_range,
            "min_date_formatted": min_str,
            "max_date_formatted": max_str,
            "sample_dates_formatted": samples,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "is_ambiguous": False,
            "warning": None if invalid_count == 0 else f"{invalid_count} unparseable date values detected.",
        }

    # Case 2: Series contains strings / mixed types
    clean_str_series = series.dropna().astype(str).str.strip()
    non_empty = clean_str_series[~clean_str_series.str.lower().isin(["", "nan", "none", "nat", "null", "-"])]

    if non_empty.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]"), {
            "source_format": "All Null/Empty",
            "date_range_formatted": "N/A (All Empty)",
            "min_date_formatted": "-",
            "max_date_formatted": "-",
            "sample_dates_formatted": [],
            "valid_count": 0,
            "invalid_count": len(series),
            "is_ambiguous": False,
            "warning": "No valid date values found in column.",
        }

    # Test candidate formats against non-empty sample
    sample_vals = non_empty.iloc[:100].tolist()

    detected_format_name = "Mixed / Auto-Detected"
    parsed_timestamps: List[Any] = []

    # Priority 1: DD-MM-YYYY (Hyphen)
    can_dd_mm_yyyy_hyphen = all(re.match(r"^\d{1,2}-\d{1,2}-\d{4}", s) for s in sample_vals)
    # Priority 2: DD/MM/YYYY (Slash)
    can_dd_mm_yyyy_slash = all(re.match(r"^\d{1,2}/\d{1,2}/\d{4}", s) for s in sample_vals)
    # Priority 3: YYYY-MM-DD (ISO Hyphen)
    can_iso_hyphen = all(re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s) for s in sample_vals)
    # Priority 4: YYYY/MM/DD (ISO Slash)
    can_iso_slash = all(re.match(r"^\d{4}/\d{1,2}/\d{1,2}", s) for s in sample_vals)

    if can_dd_mm_yyyy_hyphen:
        detected_format_name = "DD-MM-YYYY (Standard Pharmacy)"
        parsed = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
        if parsed.isna().sum() > len(series) * 0.5:
            # Try with time component
            parsed = pd.to_datetime(series.astype(str).str.split(".").str[0], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    elif can_dd_mm_yyyy_slash:
        detected_format_name = "DD/MM/YYYY (Standard Pharmacy)"
        parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
        if parsed.isna().sum() > len(series) * 0.5:
            parsed = pd.to_datetime(series.astype(str).str.split(".").str[0], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    elif can_iso_hyphen:
        detected_format_name = "YYYY-MM-DD (ISO Format)"
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
        if parsed.isna().sum() > len(series) * 0.5:
            parsed = pd.to_datetime(series.astype(str).str.split(".").str[0], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    elif can_iso_slash:
        detected_format_name = "YYYY/MM/DD (ISO Format)"
        parsed = pd.to_datetime(series, format="%Y/%m/%d", errors="coerce")
    else:
        # Mixed or alphanumeric formats (e.g. 01-Aug-2026) -> apply parse_single_date
        detected_format_name = "DD-MM-YYYY (Robust Auto-Detect)"
        parsed = pd.Series([parse_single_date(v) for v in series], index=series.index)
        parsed = pd.to_datetime(parsed, errors="coerce")

    valid_mask = parsed.notna()
    valid_count = int(valid_mask.sum())
    invalid_count = len(series) - valid_count
    valid_parsed = parsed[valid_mask]

    is_ambiguous = False
    warning = None

    if valid_count > 0:
        min_dt = valid_parsed.min()
        max_dt = valid_parsed.max()
        min_str = min_dt.strftime(UI_DATE_FORMAT)
        max_str = max_dt.strftime(UI_DATE_FORMAT)
        date_range = f"{min_str} to {max_str}"
        samples = [d.strftime(UI_DATE_FORMAT) for d in valid_parsed.iloc[:5]]
    else:
        min_str = max_str = date_range = "-"
        samples = []
        is_ambiguous = True
        warning = "No dates could be reliably parsed from the date column."

    if invalid_count > 0 and warning is None:
        warning = f"{invalid_count} row(s) contained invalid or unparseable date strings."

    return parsed, {
        "source_format": detected_format_name,
        "date_range_formatted": date_range,
        "min_date_formatted": min_str,
        "max_date_formatted": max_str,
        "sample_dates_formatted": samples,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "is_ambiguous": is_ambiguous,
        "warning": warning,
    }
