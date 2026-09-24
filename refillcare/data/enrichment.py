"""SALT Master enrichment module for RefillCare.

Provides functionality to load the master catalog 'SALT WISE ITEMS.xlsx'
and enrich transaction records with active ingredient (SALT) composition
and pharmaceutical categorization.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import pandas as pd
import numpy as np


def load_salt_master(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and normalize the SALT WISE ITEMS master catalog.

    Excel details confirmed in Phase 1:
    - Sheet name: 'Rate List'
    - Header starts at Excel row 6 -> skiprows=5
    - Total master rows: ~36,963
    - Key mapping column: 'Code' (maps to itemCode/itemId)

    Deduplication rule:
    If duplicate 'Code' values exist in the master catalog:
    1. Sort by presence of non-null 'SALT' values first so richer records take precedence.
    2. Retain the first deterministic occurrence per 'Code'.

    Args:
        file_path: Path to the 'SALT WISE ITEMS.xlsx' Excel file.

    Returns:
        pd.DataFrame: Cleaned SALT master DataFrame indexed by normalized Code.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SALT master file not found at: {path}")

    # Read Excel skipping 5 metadata header rows
    raw_df = pd.read_excel(
        path,
        sheet_name="Rate List",
        skiprows=5,
        engine="openpyxl",
    )

    # Clean column names (strip whitespace)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # Required Code column check
    if "Code" not in raw_df.columns:
        raise KeyError(f"Expected 'Code' column in SALT master. Found columns: {list(raw_df.columns)}")

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
