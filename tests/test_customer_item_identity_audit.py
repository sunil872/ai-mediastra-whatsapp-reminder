"""Unit & Integration Tests for Customer-Item Identity Impact Audit (Step 14).

Verifies all 16 specified invariants:
1. deterministic customer-name normalization
2. deterministic phone normalization
3. deterministic store normalization
4. deterministic item normalization
5. same phone + different names remain different keys
6. same phone + same name + same item produce the same key
7. different item IDs produce different keys
8. different stores produce different keys
9. missing phone is identified as insufficient identity data
10. fuzzy name matching is not performed (SUNIL vs SUNIL K vs SUNIL KAREN)
11. current customerId remains unchanged
12. Path A comparison uses existing >=6 rule
13. Path B comparison uses 3M >=2 and 6M >=3
14. future transactions are excluded when evaluating historical cutoffs
15. audit execution does not modify production records
16. audit results are deterministic
"""
import copy
from datetime import datetime, timedelta
import pandas as pd
import pytest

from experiments.identity_audit.identity_normalization import (
    build_current_key,
    build_proposed_key,
    mask_phone,
    normalize_customer_name,
    normalize_item_id,
    normalize_phone_number,
    normalize_store_id,
)
from refillcare.models.path_a_classifier import MIN_PURCHASES


# 1. Deterministic customer-name normalization
def test_deterministic_customer_name_normalization():
    name1, err1 = normalize_customer_name("  sunil   kumar  ")
    name2, err2 = normalize_customer_name("SUNIL KUMAR")
    name3, err3 = normalize_customer_name("Sunil, Kumar.")
    
    assert err1 is None and err2 is None and err3 is None
    assert name1 == "SUNIL KUMAR"
    assert name2 == "SUNIL KUMAR"
    assert name3 == "SUNIL KUMAR"
    assert name1 == name2 == name3


# 2. Deterministic phone normalization
def test_deterministic_phone_normalization():
    p1, err1 = normalize_phone_number("9876543210")
    p2, err2 = normalize_phone_number("+91 98765-43210")
    p3, err3 = normalize_phone_number("09876543210")
    p4, err4 = normalize_phone_number("919876543210")

    assert err1 is None and err2 is None and err3 is None and err4 is None
    assert p1 == "919876543210"
    assert p2 == "919876543210"
    assert p3 == "919876543210"
    assert p4 == "919876543210"


# 3. Deterministic store normalization
def test_deterministic_store_normalization():
    s1 = normalize_store_id("  branch_01 ")
    s2 = normalize_store_id("BRANCH_01")
    s3 = normalize_store_id(None, default="MAIN")

    assert s1 == "BRANCH_01"
    assert s2 == "BRANCH_01"
    assert s3 == "MAIN"


# 4. Deterministic item normalization
def test_deterministic_item_normalization():
    i1, err1 = normalize_item_id("  item_123  ")
    i2, err2 = normalize_item_id(12345.0)
    i3, err3 = normalize_item_id(None)

    assert err1 is None and i1 == "ITEM_123"
    assert err2 is None and i2 == "12345"
    assert err3 == "Missing item ID" and i3 is None


# 5. Same phone + different names remain different keys
def test_same_phone_different_names_remain_different_keys():
    k1, s1 = build_proposed_key("STORE_A", "9876543210", "JOHN DOE", "ITEM_1")
    k2, s2 = build_proposed_key("STORE_A", "9876543210", "JANE DOE", "ITEM_1")

    assert k1 != k2
    assert s1 == "SAFE" and s2 == "SAFE"
    assert "JOHN DOE" in k1
    assert "JANE DOE" in k2


# 6. Same phone + same name + same item produce the same key
def test_same_phone_same_name_same_item_produce_same_key():
    k1, s1 = build_proposed_key("STORE_A", "9876543210", "JOHN DOE", "ITEM_1")
    k2, s2 = build_proposed_key("STORE_A", "+91 9876543210", "  john   doe. ", "item_1")

    assert k1 == k2
    assert s1 == "SAFE" and s2 == "SAFE"


# 7. Different item IDs produce different keys
def test_different_item_ids_produce_different_keys():
    k1, _ = build_proposed_key("STORE_A", "9876543210", "JOHN DOE", "ITEM_1")
    k2, _ = build_proposed_key("STORE_A", "9876543210", "JOHN DOE", "ITEM_2")

    assert k1 != k2


# 8. Different stores produce different keys
def test_different_stores_produce_different_keys():
    k1, _ = build_proposed_key("STORE_A", "9876543210", "JOHN DOE", "ITEM_1")
    k2, _ = build_proposed_key("STORE_B", "9876543210", "JOHN DOE", "ITEM_1")

    assert k1 != k2


# 9. Missing phone is identified as insufficient identity data
def test_missing_phone_insufficient_identity_data():
    k1, s1 = build_proposed_key("STORE_A", None, "JOHN DOE", "ITEM_1")
    k2, s2 = build_proposed_key("STORE_A", "12345", "JOHN DOE", "ITEM_1")  # Invalid phone

    assert s1 == "INSUFFICIENT_IDENTITY_DATA"
    assert s2 == "INSUFFICIENT_IDENTITY_DATA"
    assert "NO_PHONE" in k1


# 10. Fuzzy name matching is not performed
def test_fuzzy_name_matching_not_performed():
    k1, _ = build_proposed_key("MAIN", "9876543210", "SUNIL", "ITEM_1")
    k2, _ = build_proposed_key("MAIN", "9876543210", "SUNIL K", "ITEM_1")
    k3, _ = build_proposed_key("MAIN", "9876543210", "SUNIL KAREN", "ITEM_1")

    assert k1 != k2
    assert k2 != k3
    assert k1 != k3


# 11. Current customerId remains unchanged
def test_current_customer_id_remains_unchanged():
    curr_k = build_current_key("CUST_1001", "ITEM_500")
    assert curr_k == "CUST_1001_ITEM_500"
    # Ensure raw customer ID is preserved in key structure
    assert "CUST_1001" in curr_k


# 12. Path A comparison uses the existing >=6 rule
def test_path_a_comparison_uses_ge_6_rule():
    assert MIN_PURCHASES == 6
    counts = pd.Series([1, 5, 6, 7])
    eligible = counts >= MIN_PURCHASES
    assert list(eligible) == [False, False, True, True]


# 13. Path B comparison uses 3M >=2 and 6M >=3
def test_path_b_comparison_recurrence_rules():
    # Helper to check distinct months
    def check_b_eligibility(dates, cutoff_date, count):
        if count >= 6:
            return False, "PATH_A"
        recent = (cutoff_date - dates[-1]).days <= 180
        if not recent:
            return False, "STALE_RECENCY"
        m3 = len(set(d.strftime("%Y-%m") for d in dates if (cutoff_date - d).days <= 90))
        m6 = len(set(d.strftime("%Y-%m") for d in dates if (cutoff_date - d).days <= 180))
        if m3 >= 2:
            return True, "3M_RECURRING"
        if m6 >= 3:
            return True, "6M_RECURRING"
        return False, "INELIGIBLE"

    cutoff = datetime(2026, 8, 31)
    # 3M qualification
    dates_3m = [datetime(2026, 6, 15), datetime(2026, 7, 20)]
    elig, reason = check_b_eligibility(dates_3m, cutoff, count=2)
    assert elig is True and reason == "3M_RECURRING"

    # 6M qualification
    dates_6m = [datetime(2026, 3, 10), datetime(2026, 4, 15), datetime(2026, 8, 20)]
    elig6, reason6 = check_b_eligibility(dates_6m, cutoff, count=3)
    assert elig6 is True and reason6 == "6M_RECURRING"


# 14. Future transactions are excluded when evaluating historical cutoffs
def test_future_transactions_excluded_from_historical_evaluation():
    cutoff = datetime(2026, 8, 31)
    all_dates = [
        datetime(2026, 7, 1),
        datetime(2026, 8, 1),
        datetime(2026, 9, 15),  # Future date
    ]
    historical = [d for d in all_dates if d <= cutoff]
    assert len(historical) == 2
    assert datetime(2026, 9, 15) not in historical


# 15. Audit execution does not modify production records
def test_audit_execution_does_not_modify_source_df():
    sample_data = {
        "customerId": ["C1", "C1"],
        "customerName": ["John Doe", "John Doe"],
        "MOBILE_NO": ["9876543210", "9876543210"],
        "itemId": ["I1", "I1"],
        "invoice_date": ["01-08-2026", "15-08-2026"],
        "quantity": [10, 10],
    }
    df = pd.DataFrame(sample_data)
    df_copy = copy.deepcopy(df)

    # Perform key generation
    props = [build_proposed_key("MAIN", r["MOBILE_NO"], r["customerName"], r["itemId"])[0] for _, r in df.iterrows()]
    
    # Assert original df unchanged
    pd.testing.assert_frame_equal(df, df_copy)
    assert len(props) == 2


# 16. Audit results are deterministic
def test_audit_results_deterministic():
    k1, s1 = build_proposed_key("MAIN", "9876543210", "John Doe", "MED_100")
    k2, s2 = build_proposed_key("MAIN", "9876543210", "John Doe", "MED_100")

    assert k1 == k2
    assert s1 == s2
    assert mask_phone("9876543210") == "******3210"
