"""SALT Master enrichment module for RefillCare.

Provides functionality to load the item master catalog ('SALT ITEMS LIST.csv')
and enrich transaction records with active ingredient (SALT) composition
and pharmaceutical categorization.

Catalog columns used: Code, Item Name, PACK, SALT, CATEGORY, ITEMCAT (S.No is ignored).
Tax (GST / HSN), stock (CLQTY) and manufacturer columns of the old export are not needed.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

SALT_CATALOG_FILENAME = "SALT ITEMS LIST.csv"
# Older exports, still read when the new file is absent (header after 5 metadata rows)
LEGACY_SALT_CATALOG_FILENAMES = ("SALT WISE ITEMS.csv", "SALT WISE ITEMS.xlsx")
SALT_CATALOG_REQUIRED_COLUMNS = ("Code", "Item Name", "SALT")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def default_salt_catalog_path() -> Path:
    """The item master: ``REFILLCARE_SALT_CATALOG`` if set, else data/refillcare/SALT ITEMS LIST.csv.

    Falls back to a legacy 'SALT WISE ITEMS' export only when the new file does not exist.
    """
    override = os.getenv("REFILLCARE_SALT_CATALOG")
    if override:
        return Path(override)
    folder = _PROJECT_ROOT / "data" / "refillcare"
    preferred = folder / SALT_CATALOG_FILENAME
    if preferred.exists():
        return preferred
    for name in LEGACY_SALT_CATALOG_FILENAMES:
        if (folder / name).exists():
            return folder / name
    return preferred


def _read_catalog_file(path: Path) -> pd.DataFrame:
    """CSV with the header on line 1 (current file) or after metadata rows (legacy export)."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name="Rate List", skiprows=5, engine="openpyxl")
    skip = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("S.No,"):
                skip = i
                break
            if i > 20:
                break
    return pd.read_csv(path, skiprows=skip, dtype=str, encoding="utf-8", encoding_errors="ignore")


# The item-code column may be headed 'Code' (billing export) or 'itemCode' (renamed by the store)
_CATALOG_CODE_ALIASES = {"code", "itemcode", "itemid", "productcode"}


def _normalise_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if "Code" not in df.columns:
        for c in df.columns:
            if "".join(ch for ch in c.lower() if ch.isalnum()) in _CATALOG_CODE_ALIASES:
                df = df.rename(columns={c: "Code"})
                break
    return df


_CATALOG_BY_CODE: Optional[dict] = None


def catalog_by_code() -> dict:
    """{item code: {'Item Name', 'PACK', 'SALT', ...}} from the default catalogue (cached; {} if absent)."""
    global _CATALOG_BY_CODE
    if _CATALOG_BY_CODE is None:
        try:
            m = load_salt_master()
            _CATALOG_BY_CODE = {str(r["Code"]): r for r in m.to_dict(orient="records")}
        except (FileNotFoundError, KeyError, ValueError):
            _CATALOG_BY_CODE = {}
    return _CATALOG_BY_CODE


def load_salt_master(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and normalize the item master catalog (default: :func:`default_salt_catalog_path`).

    - Total master rows: ~36,963
    - Key mapping column: 'Code' (maps to itemCode/itemId)

    Deduplication rule:
    If duplicate 'Code' values exist in the master catalog:
    1. Sort by presence of non-null 'SALT' values first so richer records take precedence.
    2. Retain the first deterministic occurrence per 'Code'.

    Args:
        file_path: Path to the catalog (.csv, or a legacy .xlsx export).

    Returns:
        pd.DataFrame: Cleaned SALT master DataFrame indexed by normalized Code.
    """
    path = Path(file_path) if file_path else default_salt_catalog_path()
    if not path.exists():
        raise FileNotFoundError(f"SALT master file not found at: {path}")

    raw_df = _normalise_catalog_columns(_read_catalog_file(path))

    missing = [c for c in SALT_CATALOG_REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise KeyError(f"SALT master is missing columns {missing}. Found columns: {list(raw_df.columns)}")

    df = raw_df.copy()

    # Normalize Code column as clean string
    df = df[df["Code"].notna()].copy()
    # Convert numeric-like codes (e.g. 1234.0) to clean integer string if applicable
    df["Code"] = (
        df["Code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Filter out empty codes
    df = df[df["Code"] != ""].copy()

    # Clean string columns
    for col in ["SALT", "CATEGORY", "ITEMCAT", "Item Name", "PACK"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan, "[00]": np.nan})

    # Deduplicate on Code: prefer rows with non-null SALT
    df["_has_salt"] = df["SALT"].notna()
    df = df.sort_values(by=["_has_salt"], ascending=False)
    df = df.drop_duplicates(subset=["Code"], keep="first")
    df = df.drop(columns=["_has_salt"])

    return df.reset_index(drop=True)


def enrich_with_salt(
    transactions_df: pd.DataFrame,
    salt_df: pd.DataFrame,
    item_code_col: str = "itemCode",
) -> pd.DataFrame:
    """Enrich transaction records with SALT master fields using Code -> itemCode.

    Rules:
    - Uses left join so that unmatched item codes are safely preserved with NaN values.
    - Unmatched item codes remain unmatched; no synthetic replacement is forced.
    - Preserves exact row count and customer/item identities.

    Args:
        transactions_df: Transactions or purchase events DataFrame.
        salt_df: Normalized SALT master DataFrame.
        item_code_col: Column name in transactions matching SALT 'Code'. Defaults to 'itemCode'.

    Returns:
        pd.DataFrame: Enriched transactions with added salt_* columns.
    """
    if transactions_df.empty:
        return transactions_df.copy()

    if item_code_col not in transactions_df.columns:
        # Fallback to itemId if itemCode is not present
        if "itemId" in transactions_df.columns:
            item_code_col = "itemId"
        else:
            raise KeyError(f"Neither '{item_code_col}' nor 'itemId' found in transactions DataFrame.")

    # Prepare master subset for merging
    salt_subset = salt_df[["Code"]].copy()
    salt_subset["Code"] = salt_subset["Code"].astype(str).str.strip()

    col_mapping = {
        "SALT": "salt_composition",
        "CATEGORY": "salt_category",
        "ITEMCAT": "salt_itemcat",
        "PACK": "salt_pack",
    }

    for orig_col, target_col in col_mapping.items():
        if orig_col in salt_df.columns:
            salt_subset[target_col] = salt_df[orig_col]

    # Ensure transaction matching key is clean string
    tx_df = transactions_df.copy()
    tx_key_series = (
        tx_df[item_code_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    tx_df["_join_key"] = tx_key_series

    # Merge
    merged = tx_df.merge(
        salt_subset,
        left_on="_join_key",
        right_on="Code",
        how="left",
        suffixes=("", "_salt_master"),
    )

    # Clean up temporary join keys
    merged = merged.drop(columns=["_join_key"])
    if "Code" in merged.columns and "Code" not in transactions_df.columns:
        merged = merged.drop(columns=["Code"])

    # Ensure row count preserved
    if len(merged) != len(transactions_df):
        raise ValueError(
            f"Row count mismatch during SALT enrichment: {len(transactions_df)} -> {len(merged)}"
        )

    return merged
