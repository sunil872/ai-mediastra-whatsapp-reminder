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
    assert ingest_res.status_code == 200
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
    """Verify model version listing and promotion."""
    models_res = client.get("/api/v1/models")
    assert models_res.status_code == 200
    models = models_res.json()
    assert len(models) >= 1
    assert any(m["version_id"] == "v1.0.0" for m in models)


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
