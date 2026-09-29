"""
Regression and Acceptance Tests for RefillCare V1 — SB/ Inter-Store Transaction Filtering.

Verifies:
1. Customer transaction classification (S0/... -> CUSTOMER_SALE, eligible=True)
2. B2B transaction classification (SB/... -> B2B_INTER_STORE, eligible=False)
3. Case and whitespace normalization
4. Purchase count isolation (5 customer + 1 SB -> purchase_count=5)
5. Path promotion protection (5 customer + 1 SB does NOT become Path A)
6. Latest purchase date protection (SB date cannot replace valid customer date)
7. Historical median cadence protection (SB cannot distort interval calculations)
8. Path B recurrence isolation (SB cannot satisfy 3M >= 2 or 6M >= 3 recurrence)
9. Model datasets zero SB rows (train, validation, test, training_dataset, purchase_history)
10. Reminder generation isolation (SB cannot generate RefillDecision, ReminderCycle, or ReminderStage)
"""

from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pytest

from refillcare.data.transaction_classifier import (
    ChannelClassification,
    ExclusionReason,
    classify_transaction,
    classify_transactions_df,
    filter_eligible_customer_transactions,
    normalize_transaction_number,
)
from refillcare.engine.decision_types import (
    PATH_A,
    PATH_B,
    PATH_INELIGIBLE,
    PRED_PATH_A_HISTORICAL_MEDIAN,
    PRED_NONE,
)
from refillcare.engine.reminder_lifecycle import ReminderLifecycleManager
from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine


class TestTransactionChannelClassification:
    """Test 1 - 3: Classification and Normalization Rules."""

    def test_01_customer_transaction(self):
        """Test 1: S0/102001 -> CUSTOMER_SALE, refillcare_eligible = TRUE."""
        channel, eligible, reason = classify_transaction("S0/102001")
        assert channel == ChannelClassification.CUSTOMER_SALE
        assert eligible is True
        assert reason is None

    def test_02_b2b_transaction(self):
        """Test 2: SB/021001 -> B2B_INTER_STORE, refillcare_eligible = FALSE."""
        channel, eligible, reason = classify_transaction("SB/021001")
        assert channel == ChannelClassification.B2B_INTER_STORE
        assert eligible is False
        assert reason == ExclusionReason.INTER_STORE_TRANSACTION

    def test_03_case_normalization(self):
        """Test 3: Case and whitespace normalization."""
        variants = [
            "SB/021001",
            "sb/021001",
            " SB/021001",
            "SB/021001 ",
            "  sb/021001\t",
            "sB/021001",
        ]
        for v in variants:
            norm = normalize_transaction_number(v)
            assert norm == "SB/021001", f"Failed normalizing '{v}'"
            channel, eligible, reason = classify_transaction(v)
            assert channel == ChannelClassification.B2B_INTER_STORE
            assert eligible is False
            assert reason == ExclusionReason.INTER_STORE_TRANSACTION

    def test_unknown_transaction_safety(self):
        """Unknown transaction formats should be safety-gated and ineligible."""
        channel, eligible, reason = classify_transaction("XX/999999")
        assert channel == ChannelClassification.UNKNOWN
        assert eligible is False
        assert reason == ExclusionReason.UNKNOWN_TRANSACTION_TYPE


class TestCustomerModelingIsolation:
    """Test 4 - 8: Modeling, Cadence, and Path Isolation."""

    def test_04_purchase_count(self):
        """Test 4: 5 customer purchases + 1 SB purchase -> purchase_count = 5."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 1, 31),
            date(2026, 3, 2),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 5, 20),  # SB purchase
        ]
        tx_nums = [
            "S0/102001",
            "S0/102045",
            "S0/102090",
            "S0/102130",
            "S0/102180",
            "SB/021500",  # B2B Inter-Store
        ]

        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_PURCHASE_COUNT",
            customer_name="Isolation Test Customer",
            mobile_no="9876543210",
            item_id="MED_1",
            item_name="TELMISARTAN 40MG",
            dates=dates,
            transaction_numbers=tx_nums,
            as_of_date=date(2026, 6, 1),
        )

        assert decision.purchase_count == 5, f"Expected purchase_count=5, got {decision.purchase_count}"

    def test_05_path_promotion_protection(self):
        """Test 5: 5 customer purchases + 1 SB purchase must NOT become Path A."""
        engine = UnifiedRefillDecisionEngine()
        dates = [
            date(2026, 1, 1),
            date(2026, 1, 31),
            date(2026, 3, 2),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 5, 25),  # SB purchase
        ]
        tx_nums = [
            "S0/102001",
            "S0/102045",
            "S0/102090",
            "S0/102130",
            "S0/102180",
            "SB/021500",
        ]

        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_NO_PROMOTION",
            customer_name="Path Promotion Test",
            mobile_no="9876543210",
            item_id="MED_1",
            item_name="TELMISARTAN 40MG",
            dates=dates,
            transaction_numbers=tx_nums,
            as_of_date=date(2026, 6, 1),
        )

        assert decision.path != PATH_A, "SB transaction illegally promoted customer to Path A"
        assert decision.purchase_count == 5

    def test_06_latest_purchase_protection(self):
        """Test 6: A later SB transaction must not replace the latest valid customer purchase."""
        engine = UnifiedRefillDecisionEngine()
        customer_date = date(2026, 8, 1)
        b2b_date = date(2026, 9, 15)

        dates = [customer_date, b2b_date]
        tx_nums = ["S0/102500", "SB/021900"]

        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_LATEST_DATE",
            customer_name="Latest Date Test",
            mobile_no="9876543210",
            item_id="MED_1",
            item_name="AMLODIPINE 5MG",
            dates=dates,
            transaction_numbers=tx_nums,
            as_of_date=date(2026, 9, 20),
        )

        assert decision.last_purchase_date == customer_date, (
            f"Expected latest purchase {customer_date}, got {decision.last_purchase_date}"
        )

    def test_07_historical_median_protection(self):
        """Test 7: SB transactions must not distort the customer's historical median interval."""
        engine = UnifiedRefillDecisionEngine()
        # 6 customer purchases spaced exactly 30 days apart (Cadence median = 30)
        customer_dates = [
            date(2026, 1, 1),
            date(2026, 1, 31),
            date(2026, 3, 2),
            date(2026, 4, 1),
            date(2026, 5, 1),
            date(2026, 5, 31),
        ]
        # Interleaved SB transaction at day 15
        sb_date = date(2026, 1, 16)

        dates = [
            customer_dates[0],
            sb_date,
            customer_dates[1],
            customer_dates[2],
            customer_dates[3],
            customer_dates[4],
            customer_dates[5],
        ]
        tx_nums = [
            "S0/1001",
            "SB/0099",  # B2B transaction
            "S0/1002",
            "S0/1003",
            "S0/1004",
            "S0/1005",
            "S0/1006",
        ]

        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_MEDIAN_TEST",
            customer_name="Median Test",
            mobile_no="9876543210",
            item_id="MED_1",
            item_name="ROSUVASTATIN 10MG",
            dates=dates,
            transaction_numbers=tx_nums,
            as_of_date=date(2026, 6, 15),
        )

        assert decision.path == PATH_A
        assert decision.purchase_count == 6
        assert decision.cadence_median == 30.0, (
            f"Expected cadence_median=30.0, got {decision.cadence_median}"
        )

    def test_08_path_b_recurrence_isolation(self):
        """Test 8: SB transactions must not satisfy 3M >= 2 or 6M >= 3 distinct months."""
        engine = UnifiedRefillDecisionEngine()
        # Only 1 customer purchase in Month 1 (July 2026), and 1 SB in Month 2 (August 2026)
        dates = [
            date(2026, 7, 10),
            date(2026, 8, 12),  # SB purchase
        ]
        tx_nums = [
            "S0/102001",
            "SB/021001",
        ]

        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_RECURRENCE_TEST",
            customer_name="Recurrence Test",
            mobile_no="9876543210",
            item_id="MED_1",
            item_name="METFORMIN 500MG",
            dates=dates,
            transaction_numbers=tx_nums,
            as_of_date=date(2026, 8, 20),
        )

        # Since only 1 valid customer purchase exists in the last 3 months, 3M count is 1 (< 2)
        assert decision.is_eligible is False, "SB purchase illegally satisfied Path B recurrence requirement"


class TestDatasetAndReminderIsolation:
    """Test 9 - 10: Model Datasets and Reminder Generation Safety."""

    def test_09_model_dataset_zero_sb_rows(self):
        """Test 9: Assert SB transactions in train = 0, val = 0, test = 0."""
        base_dir = Path("c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/data/refillcare/processed")
        splits = {
            "train": base_dir / "train.parquet",
            "validation": base_dir / "validation.parquet",
            "test": base_dir / "test.parquet",
            "training_dataset": base_dir / "training_dataset.parquet",
            "purchase_history": base_dir / "purchase_history.parquet",
        }

        for name, path in splits.items():
            if not path.exists():
                pytest.skip(f"Dataset {path} not found")
            try:
                import pyarrow.parquet as pq
                schema_cols = pq.read_schema(path).names
                candidate_cols = [c for c in ["transaction_number", "Transaction Number", "transaction_id", "inv_no"] if c in schema_cols]
                if candidate_cols:
                    df = pd.read_parquet(path, columns=candidate_cols)
                else:
                    df = pd.DataFrame()
            except Exception:
                df = pd.read_parquet(path)
            for col in ["transaction_number", "Transaction Number", "transaction_id", "inv_no"]:
                if col in df.columns:
                    sb_mask = (
                        df[col]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .str.startswith("SB/")
                    )
                    sb_count = int(sb_mask.sum())
                    assert sb_count == 0, f"Found {sb_count} SB/ rows in {name} ({col})"

    def test_10_reminder_generation_isolation(self):
        """Test 10: An SB transaction must never create RefillDecision, ReminderCycle, or ReminderStage."""
        engine = UnifiedRefillDecisionEngine()
        lifecycle = ReminderLifecycleManager()

        # Customer with ONLY an SB transaction
        decision = engine.evaluate_customer_item_trajectory(
            customer_id="CUST_SB_ONLY",
            customer_name="Inter-Store Business Account",
            mobile_no="9876543210",
            item_id="MED_WHOLESALE",
            item_name="BULK PARACETAMOL 500MG",
            dates=[date(2026, 8, 1)],
            transaction_numbers=["SB/021999"],
            as_of_date=date(2026, 8, 15),
        )

        assert decision.is_eligible is False
        assert decision.expected_refill_date is None

        # Lifecycle generation must refuse or generate None/zero stages for ineligible decisions
        cycle = lifecycle.create_cycle_from_decision(decision)
        assert cycle is None, "SB transaction illegally created a ReminderCycle"
