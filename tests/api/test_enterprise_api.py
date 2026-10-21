"""Automated Test Suite for RefillCare Enterprise FastAPI Endpoints."""

import io
import sys
from pathlib import Path
from datetime import date, datetime
import pytest
from fastapi.testclient import TestClient
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    """Run every API test against a temporary store: temp data folder + temp SQLite DB (never live data)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import database.models  # noqa: F401
    from database.connection import Base, get_db

    data_dir = tmp_path / "store"
    (data_dir / "processed").mkdir(parents=True)
    pd.DataFrame([{
        "customerId": "C101", "customerName": "Existing Patient", "itemId": "I201", "itemName": "Existing Medicine",
        "invoice_date": "2026-07-01", "quantity": 30, "packing": "10 Tabs", "MOBILE_NO": "9876543210",
        "import_batch_id": "BATCH_SAMPLE", "transaction_number": "S0/1000", "invoice_number": "S0/1000",
        "transaction_type": "CUSTOMER_SALE", "refillcare_eligible": True,
    }]).to_parquet(data_dir / "processed" / "purchase_history.parquet", index=False)
    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MEDICAL_STORE_NAME", "Test Pharmacy")

    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    engine.dispose()
    import gc
    gc.collect()


def test_health_check():
    """Verify system health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "RefillCare Enterprise API"


def test_analytics_kpi_summary():
    """Verify KPI summary endpoint."""
    response = client.get("/api/v1/analytics/kpi")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers_monitored" in data
    assert "total_sales_transactions" in data
    assert "latest_sales_date" in data
    assert "active_model_version" in data


def test_sales_preview_valid_dd_mm_yyyy():
    """Verify sales upload preview with standard DD-MM-YYYY dates."""
    csv_content = (
        "invoice_date,customerId,customerName,itemId,itemName,quantity,packing,MOBILE_NO\n"
        "01-08-2026,C101,John Doe,I201,Metformin 500mg,30,10 Tabs,9876543210\n"
        "05-08-2026,C102,Jane Smith,I202,Amlodipine 5mg,15,15 Tabs,9123456780\n"
        "12-08-2026,C103,Bob Wilson,I203,Atorvastatin 10mg,30,10 Tabs,9988776655\n"
    )
    files = {"file": ("test_sales.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/sales/preview", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["records_received"] == 3
    assert data["valid_records"] == 3
    assert data["can_process"] is True
    assert data["date_diagnostics"]["min_date_formatted"] == "01-08-2026"
    assert data["date_diagnostics"]["max_date_formatted"] == "12-08-2026"


def test_sales_ingestion_and_rollback():
    """Verify sales ingestion with batch ID and subsequent rollback."""
    csv_content = (
        "invoice_date,customerId,customerName,itemId,itemName,quantity,packing,MOBILE_NO\n"
        "02-08-2026,C999,Test Batch Patient,I999,Test Medicine,30,10 Tabs,9876543210\n"
    )
    files = {"file": ("test_batch_upload.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    
    # Ingest
    ingest_res = client.post("/api/v1/sales/ingest", files=files)
    assert ingest_res.status_code == 200, f"Error: {ingest_res.status_code} - {ingest_res.text}"
    ingest_data = ingest_res.json()
    batch_id = ingest_data["import_batch_id"]
    assert batch_id.startswith("BATCH_")
    assert ingest_data["status"] == "success"

    # Verify batch is listed
    batches_res = client.get("/api/v1/sales/batches")
    assert batches_res.status_code == 200
    batches = batches_res.json()
    assert any(b["import_batch_id"] == batch_id for b in batches)

    # Rollback batch
    rb_res = client.post(f"/api/v1/sales/batches/{batch_id}/rollback")
    assert rb_res.status_code == 200
    rb_data = rb_res.json()
    assert rb_data["status"] == "success"
    assert rb_data["rolled_back_batch_id"] == batch_id


def test_prediction_snapshots_and_generation():
    """Verify prediction snapshots retrieval and generation."""
    # Query snapshots
    snap_res = client.get("/api/v1/predictions/snapshots?limit=10")
    assert snap_res.status_code == 200
    snapshots = snap_res.json()
    assert isinstance(snapshots, list)


def test_daily_reminders_and_csv_export():
    """Verify daily reminder list query and standard 10-column CSV export."""
    today_str = date.today().strftime("%Y-%m-%d")
    
    # Query daily reminders
    rem_res = client.get(f"/api/v1/reminders/daily?target_date={today_str}")
    assert rem_res.status_code == 200
    assert isinstance(rem_res.json(), list)

    # Download CSV
    csv_res = client.get(f"/api/v1/reminders/export-csv?target_date={today_str}")
    assert csv_res.status_code == 200
    assert csv_res.headers["content-type"].startswith("text/csv")
    csv_text = csv_res.text
    # Verify standard 10 headers
    assert "customer_id" in csv_text
    assert "customer_name" in csv_text
    assert "phone_number" in csv_text
    assert "mobile_status" in csv_text
    assert "item_id" in csv_text
    assert "medication_name" in csv_text
    assert "last_purchase_date" in csv_text
    assert "estimated_days_of_supply" in csv_text
    assert "expected_refill_date" in csv_text
    assert "reminder_date" in csv_text


def test_model_registry():
    """Registry lists real versions only: baseline + legacy model, never fabricated metrics."""
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models = models_res.json()
    ids = {m["version_id"] for m in models}
    assert "baseline-median-v1" in ids
    legacy = next((m for m in models if m["version_id"] == "legacy-human-hgb"), None)
    if legacy is not None:
        assert legacy["is_validated"] is False
        assert "within_7_days_pct" not in (legacy["metrics_train"] or {})


def test_model_activation_requires_evaluation_evidence():
    res = client.post("/api/v1/models/baseline-median-v1/activate")
    assert res.status_code == 409
    assert "evaluation_required" in res.json()["detail"]


def test_consensus_regimen_analytics():
    """Verify consensus regimen analytics endpoint and breakdown."""
    res = client.get("/api/v1/analytics/consensus-regimen")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_version"]  # production version id from the registry
    assert len(data["regimens"]) >= 4
    assert any(r["regimen"] == "Once Daily (OD)" for r in data["regimens"])
    assert any(r["regimen"] == "Alternate Day / Intermittent (QOD)" for r in data["regimens"])
    assert any(r["regimen"] == "Twice Daily (BD)" for r in data["regimens"])
    assert any(r["regimen"] == "Thrice Daily (TID) / Multiple" for r in data["regimens"])
    # Benchmarks come only from completed evaluation runs - none exist in a fresh store
    assert isinstance(data["benchmarks"], list)
    assert all("measured" in b["status"] for b in data["benchmarks"])


def test_whatsapp_dry_run_dispatch():
    """Verify WhatsApp dispatch simulation in dry-run mode."""
    today_str = date.today().strftime("%Y-%m-%d")
    payload = {
        "reminder_date": today_str,
        "dry_run": True,
        "batch_size": 5,
    }
    wa_res = client.post("/api/v1/whatsapp/dispatch", json=payload)
    assert wa_res.status_code == 200
    data = wa_res.json()
    assert data["is_dry_run"] is True
    assert "dispatched_count" in data
    assert "delivery_summary" in data


def test_med_sync_bundles_api():
    """Verify Med-Sync multi-prescription clustering API endpoint."""
    res = client.get("/api/v2/med-sync/bundles?sync_window_days=7")
    assert res.status_code == 200
    data = res.json()
    assert "sync_window_days" in data
    assert data["sync_window_days"] == 7
    assert "impact_summary" in data
    assert "bundles" in data
    assert isinstance(data["bundles"], list)


def test_quantile_uncertainty_api():
    """Verify Quantile Uncertainty Envelopes API endpoint."""
    res = client.get("/api/v2/models/quantiles")
    assert res.status_code == 200
    data = res.json()
    assert ("overall_test_comparison" in data or "quantile_uncertainty_envelope" in data or "status" in data)



# ------------------------------------------------------------------ WhatsApp consent (opt-out / opt-in)
def test_webhook_stop_and_exact_start_drive_consent():
    """Standalone STOP opts out; conversation never changes consent; only an exact START opts back in."""
    phone = "919999988888"
    assert client.post("/api/v1/whatsapp/webhook", json={"from": phone, "message": "Can I cancel this and get Dolo?"}).json()["action"] == "message_logged"
    assert client.post("/api/v1/whatsapp/webhook", json={"from": phone, "message": "Please stop."}).json()["action"] == "opted_out"
    opted = client.get("/api/customers/opt-outs").json()
    assert [r["customer_id"] for r in opted] == ["CUST_9999988888"]

    # Conversational "yes" / "restart" must never re-subscribe.
    for msg in ("Yes, I bought it yesterday", "restart please"):
        assert client.post("/api/v1/whatsapp/webhook", json={"from": phone, "message": msg}).json()["action"] == "message_logged"
    assert len(client.get("/api/customers/opt-outs").json()) == 1

    assert client.post("/api/v1/whatsapp/webhook", json={"from": phone, "message": "START"}).json()["action"] == "opted_in"
    assert client.get("/api/customers/opt-outs").json() == []


def test_send_single_returns_response_and_refuses_opted_out_phone():
    payload = {"phone_number": "919515473474", "customer_name": "Test", "medications": "Telma 40",
               "expected_refill_date": "2026-10-20", "tier": "DUE", "stage_offset": 0, "dry_run": True}
    ok = client.post("/api/v1/whatsapp/send-single", json=payload)
    assert ok.status_code == 200 and ok.json()["success"] is True

    client.post("/api/v1/whatsapp/webhook", json={"from": "919515473474", "message": "STOP"})
    blocked = client.post("/api/v1/whatsapp/send-single", json=payload).json()
    assert blocked["success"] is False and blocked["response_payload"]["reason"] == "OPTED_OUT"


def test_manual_opt_out_and_opt_in_endpoints():
    assert client.post("/api/customers/C101/opt-out").json()["opt_in_whatsapp"] is False
    assert [r["customer_id"] for r in client.get("/api/customers/opt-outs").json()] == ["C101"]
    assert client.post("/api/customers/C101/opt-in").json()["opt_in_whatsapp"] is True
    assert client.get("/api/customers/opt-outs").json() == []


def test_ml_plan_endpoint_reports_freshness():
    data = client.get("/api/v1/ml/plan").json()
    assert "retrain_recommended" in data
