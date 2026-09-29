"""Unit tests for Med-Sync (Multi-Prescription Synchronization) and Quantile Uncertainty Envelopes."""

from datetime import date, timedelta
import pytest
import numpy as np
import pandas as pd

from refillcare.engine.med_sync import MedSyncEngine, MedSyncBundle, SyncedMedicationItem
from refillcare.engine.decision_types import RefillDecision


def test_med_sync_clustering_and_anchoring():
    """Test clustering multiple prescriptions for a single patient within a 7-day window."""
    engine = MedSyncEngine(sync_window_days=7)

    today = date(2026, 9, 28)
    
    # 3 active prescriptions for Rajesh:
    # 1. Telmisartan due in 10 days (HIGH stability)
    # 2. Metformin due in 12 days (MEDIUM-SAFE stability)
    # 3. Atorvastatin due in 14 days (HIGH stability)
    # 4. Antibiotic / acute eye drop due in 45 days (should be in a separate second cluster)
    decisions = [
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_TELMI_40",
            item_name="Telmisartan 40mg",
            customer_item_key="CUST001_MED_TELMI_40",
            path="PATH_A",
            purchase_count=8,
            is_eligible=True,
            stability_tier="HIGH",
            cadence_median=30.0,
            cadence_norm_mad=0.15,
            cadence_drift=2.0,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=30,
            last_purchase_date=today - timedelta(days=20),
            expected_refill_date=today + timedelta(days=10),
            decision_reason="Path A High Stability",
            cycle_id="CYC_001",
            quantile_p10_date=today + timedelta(days=5),
            quantile_p90_date=today + timedelta(days=15),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_MET_500",
            item_name="Metformin 500mg",
            customer_item_key="CUST001_MED_MET_500",
            path="PATH_A",
            purchase_count=5,
            is_eligible=True,
            stability_tier="MEDIUM-SAFE",
            cadence_median=28.0,
            cadence_norm_mad=0.25,
            cadence_drift=3.0,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=28,
            last_purchase_date=today - timedelta(days=16),
            expected_refill_date=today + timedelta(days=12),
            decision_reason="Path A Medium Safe",
            cycle_id="CYC_002",
            quantile_p10_date=today + timedelta(days=7),
            quantile_p90_date=today + timedelta(days=17),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_ATORVA_10",
            item_name="Atorvastatin 10mg",
            customer_item_key="CUST001_MED_ATORVA_10",
            path="PATH_A",
            purchase_count=6,
            is_eligible=True,
            stability_tier="HIGH",
            cadence_median=30.0,
            cadence_norm_mad=0.10,
            cadence_drift=1.5,
            dos_days=30.0,
            units_purchased=30.0,
            prediction_method="PATH_A_PERSONAL_HISTORICAL_MEDIAN",
            predicted_interval_days=30,
            last_purchase_date=today - timedelta(days=16),
            expected_refill_date=today + timedelta(days=14),
            decision_reason="Path A High Stability",
            cycle_id="CYC_003",
            quantile_p10_date=today + timedelta(days=9),
            quantile_p90_date=today + timedelta(days=19),
        ),
        RefillDecision(
            customer_id="CUST001",
            customer_name="Rajesh Kumar",
            mobile_no="919876543210",
            item_id="MED_EYE_DROP",
            item_name="Refresh Tears Eye Drops",
            customer_item_key="CUST001_MED_EYE_DROP",
            path="PATH_B",
            purchase_count=2,
            is_eligible=True,
            stability_tier="UNSTABLE",
            cadence_median=45.0,
            cadence_norm_mad=0.60,
            cadence_drift=12.0,
            dos_days=45.0,
            units_purchased=1.0,
            prediction_method="PATH_B_DOS_EDS",
            predicted_interval_days=45,
            last_purchase_date=today,
            expected_refill_date=today + timedelta(days=45),
            decision_reason="Path B DOS",
            cycle_id="CYC_004",
            quantile_p10_date=today + timedelta(days=35),
            quantile_p90_date=today + timedelta(days=55),
        ),
    ]

    bundles = engine.cluster_patient_decisions(decisions, sync_window_days=7)

    # Should form 2 bundles: 1 multi-prescription bundle of 3 meds, and 1 single-med bundle
    assert len(bundles) == 2

    b1 = bundles[0]
    assert b1.customer_id == "CUST001"
    assert b1.total_items_count == 3
    assert b1.message_reduction_count == 2
    assert b1.anchor_item_name == "Telmisartan 40mg"
    assert b1.anchor_refill_date == today + timedelta(days=10)
    assert b1.earliest_p10_date == today + timedelta(days=5)
    assert b1.latest_p90_date == today + timedelta(days=19)
    assert "Synchronized Prescription Refill Notice" in b1.bundled_message_text
    assert "Telmisartan 40mg" in b1.bundled_message_text
    assert "Metformin 500mg" in b1.bundled_message_text
    assert "Atorvastatin 10mg" in b1.bundled_message_text

    b2 = bundles[1]
    assert b2.total_items_count == 1
    assert b2.message_reduction_count == 0
    assert b2.anchor_item_name == "Refresh Tears Eye Drops"

    # Impact summary
    impact = engine.summarize_sync_impact(bundles)
    assert impact["total_patients_analyzed"] == 1
    assert impact["total_prescriptions_synced"] == 4
    assert impact["total_dispatches_generated"] == 2
    assert impact["individual_messages_saved"] == 2
    assert impact["message_reduction_rate_pct"] == 50.0


def test_quantile_uncertainty_model_bundle_integrity():
    """Verify serialized chronic multi-quantile model bundle contains P10, P50, P90 regressors."""
    import joblib
    from pathlib import Path

    model_path = Path("data/refillcare/processed/models/chronic_refill_model.joblib")
    if not model_path.exists():
        pytest.skip("Model not yet trained in test environment.")

    bundle = joblib.load(model_path)
    assert "model_p10" in bundle
    assert "model_p50" in bundle
    assert "model_p90" in bundle
    assert "quantile_envelope" in bundle

    features = bundle["feature_cols"]
    sample_x = np.ones((5, len(features)), dtype=np.float32)

    p10 = bundle["model_p10"].predict(sample_x)
    p50 = bundle["model_p50"].predict(sample_x)
    p90 = bundle["model_p90"].predict(sample_x)

    assert len(p10) == 5
    assert len(p50) == 5
    assert len(p90) == 5
