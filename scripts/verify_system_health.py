"""Enterprise System Health & End-to-End Verification Script.

Validates:
1. SQLite Database integrity (enterprise.db, refillcare.db).
2. FastAPI test client and all REST endpoints.
3. Streamlit UI data structures and dual-model joblib bundles.
4. UnifiedRefillDecisionEngine logic and clinical provenance.
5. B2B Wholesale vs Retail transaction filtering.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
import pandas as pd

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def step_banner(title: str):
    print(f"\n{'='*20} {title} {'='*20}")

def verify_databases():
    step_banner("1. Database & Persistence Layer")
    from database.connection import SessionLocal
    from database.models import (
        RefillDecisionModel,
        ReminderCycleModel,
        ReminderStageModel,
        ModelRegistryModel,
    )
    
    session = SessionLocal()
    n_decisions = session.query(RefillDecisionModel).count()
    n_cycles = session.query(ReminderCycleModel).count()
    n_stages = session.query(ReminderStageModel).count()
    n_models = session.query(ModelRegistryModel).count()
    models = session.query(ModelRegistryModel).all()
    session.close()

    print(f"enterprise.db status: OK")
    print(f"  - Decisions: {n_decisions:,}")
    print(f"  - Cycles: {n_cycles:,}")
    print(f"  - Daily Reminder Stages: {n_stages:,}")
    print(f"  - Registered Models: {n_models} ({', '.join(m.version_id for m in models)})")
    
    assert n_decisions > 0, "No decisions found in enterprise.db"
    assert n_stages > 0, "No reminder stages found in enterprise.db"
    assert n_models >= 2, f"Expected at least 2 models, got {n_models}"
    print("Database verification passed successfully.")

def verify_fastapi_endpoints():
    step_banner("2. FastAPI REST Layer")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    # 1. Health
    r = client.get("/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"GET /health -> {r.json()}")

    # 2. KPIs
    r = client.get("/api/v1/analytics/kpi")
    assert r.status_code == 200, f"KPIs failed: {r.text}"
    kpis = r.json()
    print(f"GET /api/v1/analytics/kpi -> monitored={kpis['total_customers_monitored']}, active_model={kpis['active_model_version']}, human_consensus={kpis['human_consensus_active']}")
    assert kpis["total_customers_monitored"] > 0
    assert kpis["human_consensus_active"] is True

    # 3. Consensus Regimen Analytics
    r = client.get("/api/v1/analytics/consensus-regimen")
    assert r.status_code == 200, f"Consensus Regimen Analytics failed: {r.text}"
    consensus = r.json()
    print(f"GET /api/v1/analytics/consensus-regimen -> engine={consensus['engine_version']}, regimens={len(consensus['regimens'])}, archetypes={len(consensus['archetypes'])}")
    assert len(consensus["regimens"]) >= 4
    assert len(consensus["archetypes"]) >= 4

    # 4. Registered Models
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    models = r.json()
    print(f"GET /api/v1/models -> {len(models)} models listed")
    assert any(m["version_id"] == "v1.2.0-human-consensus" for m in models)

    # 5. Review Queue
    r = client.get("/api/reminders/today?target_date=2026-09-24")
    assert r.status_code == 200
    queue = r.json()
    print(f"GET /api/reminders/today -> {len(queue)} items for 2026-09-24")
    if queue:
        item = queue[0]
        print(f"  Sample Item: {item['customer_name']} | Regimen: {item.get('dosage_regimen')} | Archetype: {item.get('archetype')}")
        assert "dosage_regimen" in item
        assert "archetype" in item

    print("FastAPI REST API verification passed successfully.")

def verify_streamlit_components():
    step_banner("3. Streamlit Layer & Model Bundles")
    import app_refillcare as app

    # Test baseline model
    base_bundle = app.load_refill_model()
    assert base_bundle is not None, "Failed to load baseline model bundle"
    assert "pipeline" in base_bundle or "model" in base_bundle
    print("Baseline model bundle loaded successfully.")

    # Test human consensus model
    human_bundle = app.load_human_refill_model()
    assert human_bundle is not None, "Failed to load human consensus model bundle"
    assert "model" in human_bundle
    assert "feature_cols" in human_bundle
    print(f"Human consensus model bundle loaded successfully (feature_cols={len(human_bundle['feature_cols'])}).")

    # Test queue loading
    df_queue = app.load_v1_reminder_queue_df(date(2026, 9, 24))
    print(f"Streamlit reminder queue loaded: {len(df_queue)} rows.")
    assert not df_queue.empty, "Streamlit reminder queue is empty"
    for col in ["Clinical Regimen", "Consensus Archetype", "Decision Reason", "Mobile Number", "Estimated Days of Supply"]:
        assert col in df_queue.columns, f"Missing column {col} in Streamlit reminder queue"
    print("Streamlit UI helpers and data structures passed successfully.")

def verify_unified_decision_engine():
    step_banner("4. Unified Refill Decision Engine")
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine

    engine = UnifiedRefillDecisionEngine(default_lead_buffer_days=3)
    
    # Test DOS computation with standard packaging
    dos, units = engine.compute_dos(
        dates=[date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)],
        quantities=[1.0, 1.0, 1.0],
        packings=["10 TAB", "10 TAB", "10 TAB"],
        item_name="TELMA 40MG 10 TAB"
    )
    print(f"DOS computation: units={units}, dos={dos}")
    assert units == 10.0
    assert dos is not None and dos > 0

    # Test trajectory evaluation
    decision = engine.evaluate_customer_item_trajectory(
        customer_id="999999",
        customer_name="TEST PATIENT",
        mobile_no="9876543210",
        item_id="88888",
        item_name="GLIMESTAR M2 10 TAB",
        dates=[date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)],
        quantities=[1.0, 1.0, 1.0],
        packings=["10 TAB", "10 TAB", "10 TAB"],
        as_of_date=date(2026, 8, 15),
    )
    print(f"Engine Decision: path={decision.path}, stability={decision.stability_tier}, refill_date={decision.expected_refill_date}, reason={decision.decision_reason}")
    assert decision.expected_refill_date is not None
    print("Decision engine verification passed successfully.")

def verify_transaction_filtering():
    step_banner("5. Channel / B2B Wholesale Filter")
    from refillcare.data.transaction_classifier import (
        classify_transaction,
        classify_transactions_df,
        filter_eligible_customer_transactions,
        TRANSACTION_TYPE_CUSTOMER_SALE,
        TRANSACTION_TYPE_B2B_INTER_STORE,
        TRANSACTION_TYPE_UNKNOWN,
    )

    # 1. Direct classification
    retail_type, retail_elig, _ = classify_transaction("S0/2608/00123")
    b2b_type, b2b_elig, b2b_reason = classify_transaction("SB/2608/99999")
    unk_type, unk_elig, _ = classify_transaction("INV-OTHER-456")

    assert retail_type == TRANSACTION_TYPE_CUSTOMER_SALE and retail_elig is True
    assert b2b_type == TRANSACTION_TYPE_B2B_INTER_STORE and b2b_elig is False
    assert unk_type == TRANSACTION_TYPE_UNKNOWN and unk_elig is False

    # 2. DataFrame filtering
    df_test = pd.DataFrame([
        {"TRAN_NO": "S0/2608/001", "CUST_NAME": "Retail Customer", "QTY": 10},
        {"TRAN_NO": "SB/2608/002", "CUST_NAME": "Inter Store Wholesale", "QTY": 500},
        {"TRAN_NO": "OTHER/123", "CUST_NAME": "Unknown Channel", "QTY": 5},
    ])
    df_eligible = filter_eligible_customer_transactions(df_test)
    assert len(df_eligible) == 1
    assert df_eligible.iloc[0]["TRAN_NO"] == "S0/2608/001"
    print(f"B2B vs Retail transaction filter successfully isolated retail transactions ({len(df_eligible)} / {len(df_test)} retained).")

if __name__ == "__main__":
    print("=========================================================")
    print("  REFILLCARE ENTERPRISE SYSTEM HEALTH VERIFICATION")
    print("=========================================================")
    try:
        verify_databases()
        verify_fastapi_endpoints()
        verify_streamlit_components()
        verify_unified_decision_engine()
        verify_transaction_filtering()
        print("\n" + "="*57)
        print("  ALL SYSTEM CHECKS COMPLETED WITH 100% SUCCESS!")
        print("=========================================================\n")
    except Exception as e:
        print(f"\n[ERROR] Health check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
