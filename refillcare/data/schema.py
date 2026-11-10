"""The RefillCare sales schema: the only columns stored from any store's sales file.

Whatever a store uploads (20, 30, 50 columns, any header names), only these fields are kept -
in the master CSV, the processed parquets and (v2) the cloud database:

| Field | Needed for | Required |
|---|---|---|
| invoice_date | every refill interval | yes |
| invoice_number | customer sale (S0/) vs B2B transfer (SB/), de-duplication | expected (synthesised if absent, with a warning) |
| customerName | patient identity, WhatsApp greeting | yes (or customerId) |
| itemCode | catalogue lookup (salt, pack) | expected (item name used if absent) |
| itemName | chronic classification, insulin/inhaler detection, message text | yes (or itemCode) |
| packing | units per pack -> days of supply | expected (filled from the catalogue if absent) |
| quantity | days of supply | yes |
| freeQuantity | free strips extend days of supply | optional |
| daysSupply | days the sale should last (prescribed duration / dose) - the strongest refill signal | optional (stored; used as a model feature once stores send it) |
| MOBILE_NO | WhatsApp, opt-out, name+mobile identity | yes |
| customerId | patient key (party code if the store has one, else the name) | derived |
| itemId | item key (= itemCode) | derived |

Price, tax, batch, expiry and cost columns are dropped at upload: no prediction uses them.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "invoice_date", "invoice_number", "customerName", "itemCode", "itemName", "packing",
    "quantity", "freeQuantity", "MOBILE_NO", "customerId", "itemId", "daysSupply",
)
# Fields a store's file is mapped onto (customerId may come from a party code; itemId is always derived)
MAPPABLE_FIELDS: Tuple[str, ...] = (
    "invoice_date", "invoice_number", "customerName", "customerId", "itemCode", "itemName",
    "packing", "quantity", "freeQuantity", "MOBILE_NO", "daysSupply",
)
REQUIRED_FIELDS: Tuple[str, ...] = ("invoice_date", "quantity", "MOBILE_NO")
# Required, but one may stand in for the other (name <- code / code <- name)
REQUIRED_ONE_OF: Tuple[Tuple[str, str], ...] = (("customerName", "customerId"), ("itemName", "itemCode"))
EXPECTED_FIELDS: Tuple[str, ...] = ("invoice_number", "itemCode", "packing")
OPTIONAL_FIELDS: Tuple[str, ...] = ("freeQuantity", "customerId", "daysSupply")
FIELD_LABELS: Dict[str, str] = {
    "invoice_date": "Invoice / bill date",
    "invoice_number": "Invoice / bill number",
    "customerName": "Patient / customer name",
    "customerId": "Patient / party code (if the store has one)",
    "itemCode": "Item / product code",
    "itemName": "Item / medicine name",
    "packing": "Packing (e.g. 1X15, 3ML)",
    "quantity": "Quantity sold",
    "freeQuantity": "Free quantity",
    "daysSupply": "Days supply / prescribed days (optional)",
    "MOBILE_NO": "Mobile number",
}
# Columns of the old 23-column export that nothing uses any more (dropped from stored data)
LEGACY_UNUSED_COLUMNS: Tuple[str, ...] = (
    "batchNo", "mfgDate", "expiryDate", "rate", "saleRate", "mrp", "invoice_total",
    "discountPercent", "item_discount", "gstAmount", "netAmount", "costPrice",
)
_TEXT_FIELDS = ("invoice_number", "customerName", "customerId", "itemCode", "itemName", "packing", "MOBILE_NO")
_INT_FLOAT = re.compile(r"^\d+\.0+$")


def read_sales_file(source: Union[bytes, str, Path, Any], filename: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV / Excel sales file with every cell as text (mobiles and codes never become floats)."""
    name = (filename or getattr(source, "name", "") or str(source if isinstance(source, (str, Path)) else "")).lower()
    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data, dtype=object)
    else:
        df = pd.read_csv(data, dtype=str, keep_default_na=False, na_values=[""], encoding_errors="ignore")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_text(series: pd.Series) -> pd.Series:
    """Text as the store typed it: 9876543210.0 -> '9876543210', NaN -> None, whitespace trimmed."""
    def _one(v: Any) -> Optional[str]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return None
        return s[: s.index(".")] if _INT_FLOAT.match(s) else s
    return series.map(_one)


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Build the canonical frame from a store file and a {field: store_header} mapping.

    Only canonical columns survive. Derivations: customerId = party code (if mapped) else name;
    customerName = code when the store has no name column; itemCode <-> itemName stand in for each
    other; itemId = itemCode; packing filled from the item catalogue when the store has no pack column.
    """
    out = pd.DataFrame(index=df.index)
    for field in MAPPABLE_FIELDS:
        col = mapping.get(field)
        if col and col in df.columns:
            out[field] = df[col]
    for f in _TEXT_FIELDS:
        if f in out.columns:
            out[f] = clean_text(out[f])
    for f in ("quantity", "freeQuantity", "daysSupply"):
        if f in out.columns:
            out[f] = pd.to_numeric(out[f], errors="coerce")
    if "daysSupply" in out.columns:
        # 1-365 days is a plausible prescription; anything else is a typo / another number
        out["daysSupply"] = out["daysSupply"].where(out["daysSupply"].between(1, 365))

    if "customerId" not in out.columns and "customerName" in out.columns:
        out["customerId"] = out["customerName"]
    if "customerName" not in out.columns and "customerId" in out.columns:
        out["customerName"] = out["customerId"]
    if "itemCode" not in out.columns and "itemName" in out.columns:
        out["itemCode"] = out["itemName"].str.upper()
    if "itemName" not in out.columns and "itemCode" in out.columns:
        out["itemName"] = out["itemCode"].map(_catalog_name_for_code)
    if "itemCode" in out.columns:
        out["itemId"] = out["itemCode"]
    if "packing" not in out.columns and "itemCode" in out.columns:
        out["packing"] = out["itemCode"].map(_catalog_pack_for_code)
    if "freeQuantity" not in out.columns:
        out["freeQuantity"] = np.nan
    if "daysSupply" not in out.columns:
        out["daysSupply"] = np.nan
    if "invoice_number" not in out.columns:
        # No invoice column: deterministic placeholders (channel filtering is not possible -> warned)
        out["invoice_number"] = [f"INV_{i + 1:06d}" for i in range(len(out))]
    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[list(CANONICAL_COLUMNS)].reset_index(drop=True)


def select_canonical_columns(df: pd.DataFrame, keep_extra: Iterable[str] = ()) -> pd.DataFrame:
    """Drop every non-RefillCare column (e.g. from a legacy 23-column history), keeping ``keep_extra``."""
    keep = [c for c in CANONICAL_COLUMNS if c in df.columns] + [c for c in keep_extra if c in df.columns and c not in CANONICAL_COLUMNS]
    return df[keep]


def drop_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the unused price / tax / batch columns of the old export (processed data stays slim)."""
    cols = [c for c in LEGACY_UNUSED_COLUMNS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def supply_quantity(quantity: Any, free_quantity: Any) -> pd.Series:
    """Units the patient took home: billed quantity + free quantity (free strips extend supply)."""
    q = pd.to_numeric(pd.Series(quantity), errors="coerce")
    f = pd.to_numeric(pd.Series(free_quantity), errors="coerce").fillna(0).clip(lower=0)
    return q + f


def _catalog_row(code: Any) -> Optional[Dict[str, Any]]:
    if code is None:
        return None
    from refillcare.data.enrichment import catalog_by_code

    return catalog_by_code().get(str(code).strip())


def _catalog_name_for_code(code: Any) -> Optional[str]:
    row = _catalog_row(code)
    return row.get("Item Name") if row else None


def _catalog_pack_for_code(code: Any) -> Optional[str]:
    row = _catalog_row(code)
    return row.get("PACK") if row else None


def missing_required(mapping: Dict[str, Optional[str]]) -> List[str]:
    """Required fields the mapping leaves unfilled (one-of pairs count as one requirement)."""
    miss = [f for f in REQUIRED_FIELDS if not mapping.get(f)]
    for a, b in REQUIRED_ONE_OF:
        if not mapping.get(a) and not mapping.get(b):
            miss.append(a)
    return miss
