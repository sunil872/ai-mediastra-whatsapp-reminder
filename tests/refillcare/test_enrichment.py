"""Unit tests for SALT master enrichment.

Uses synthetic test fixtures only (no real customer data).
"""

import pandas as pd
import numpy as np
import pytest
from refillcare.data.enrichment import (
    load_salt_master,
    enrich_with_salt,
)


@pytest.fixture
def synthetic_salt_excel(tmp_path):
    """Create a temporary synthetic Excel file with 5 metadata header rows."""
    excel_file = tmp_path / "synthetic_salt.xlsx"

    # 5 metadata rows + 1 header row + data rows
    meta_rows = [
        ["Report Title: Master Salt List", "", "", "", "", ""],
        ["Generated: 2026-09-01", "", "", "", "", ""],
        ["Confidential Internal", "", "", "", "", ""],
        ["", "", "", "", "", ""],
        ["", "", "", "", "", ""],
    ]

    header = ["S.No", "CATEGORY", "Code", "Item Name", "PACK", "SALT", "ITEMCAT"]
    data = [
        [1, "CARDIAC", "101", "Medicine Alpha", "10 TAB", "AMLODIPINE-5MG", "TABLETS"],
        [2, "DIABETES", "102", "Medicine Beta", "10 TAB", "METFORMIN-500MG", "TABLETS"],
        [3, "GENERAL", "103", "Medicine Gamma", "100 ML", np.nan, "SYRUP"],
        # Duplicate Code 104: one with NaN SALT, one with valid SALT -> should keep valid SALT
        [4, "DERMA", "104", "Medicine Delta", "20 GM", np.nan, "CREAM"],
        [5, "DERMA", "104", "Medicine Delta Extra", "20 GM", "CLOBETASOL-0.05%", "CREAM"],
    ]

    full_rows = meta_rows + [header] + data
    df = pd.DataFrame(full_rows)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Rate List", index=False, header=False)

    return excel_file


def test_load_salt_master(synthetic_salt_excel):
    """Verify loading SALT master with skiprows=5 and deduplication."""
    master = load_salt_master(synthetic_salt_excel)

    # Master should have 4 unique Codes: 101, 102, 103, 104
    assert len(master) == 4
    assert set(master["Code"]) == {"101", "102", "103", "104"}

    # Duplicate Code 104 must retain the valid SALT
    row_104 = master[master["Code"] == "104"].iloc[0]
    assert row_104["SALT"] == "CLOBETASOL-0.05%"


def test_enrich_with_salt():
    """Verify left-join enrichment on itemCode and preservation of unmatched items."""
    transactions = pd.DataFrame([
        {"customerId": "C1", "itemId": "101", "itemCode": "101", "quantity": 1},
        {"customerId": "C2", "itemId": "102", "itemCode": "102", "quantity": 2},
        {"customerId": "C3", "itemId": "999", "itemCode": "999", "quantity": 3},  # Unmatched
    ])

    salt_master = pd.DataFrame([
        {"Code": "101", "SALT": "AMLODIPINE-5MG", "CATEGORY": "CARDIAC", "ITEMCAT": "TABLETS", "PACK": "10 TAB"},
        {"Code": "102", "SALT": "METFORMIN-500MG", "CATEGORY": "DIABETES", "ITEMCAT": "TABLETS", "PACK": "10 TAB"},
    ])

    enriched = enrich_with_salt(transactions, salt_master, item_code_col="itemCode")

    # Row count must match exactly
    assert len(enriched) == 3

    # Matched items get salt_composition
    assert enriched.loc[0, "salt_composition"] == "AMLODIPINE-5MG"
    assert enriched.loc[0, "salt_category"] == "CARDIAC"
    assert enriched.loc[1, "salt_composition"] == "METFORMIN-500MG"

    # Unmatched item code 999 remains NaN (no crash, not dropped)
    assert pd.isna(enriched.loc[2, "salt_composition"])
    assert pd.isna(enriched.loc[2, "salt_category"])
    assert enriched.loc[2, "customerId"] == "C3"
    assert enriched.loc[2, "itemId"] == "999"
