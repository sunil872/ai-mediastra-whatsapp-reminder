"""Prediction settings, saved ML reports and the identity cut-over through the FastAPI routes."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import database.models  # noqa: F401
    from api.main import app
    from database.connection import Base, get_db

    data_dir = tmp_path / "store"
    (data_dir / "processed").mkdir(parents=True)
    pd.DataFrame([{
        "customerId": "RAVI", "customerName": "RAVI", "itemId": "T40", "itemCode": "T40", "itemName": "TELMA 40 TAB",
        "invoice_date": pd.Timestamp("2026-07-01"), "quantity": 2, "packing": "1X15", "MOBILE_NO": "9848012345",
        "invoice_number": "S0/1", "refillcare_eligible": True,
    }]).to_parquet(data_dir / "processed" / "purchase_history.parquet", index=False)
    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(data_dir))
    eng = create_engine(f"sqlite:///{(tmp_path / 't.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    S = sessionmaker(bind=eng)

    def _db():
        d = S()
        try:
            yield d
        finally:
            d.close()

    app.dependency_overrides[get_db] = _db
    yield TestClient(app)
    app.dependency_overrides.clear()
    eng.dispose()


def test_settings_reflect_env(client, monkeypatch):
    monkeypatch.setenv("REFILLCARE_RULE_PATH_B_MODE", "blend")
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    s = client.get("/api/v1/ml/settings").json()
    assert s["rules"]["path_b_mode"] == "blend" and s["patient_identity"] == "name_mobile"
    assert len(s["stored_sales_columns"]) == 12 and "daysSupply" in s["stored_sales_columns"] and "REFILLCARE_ITEM_GROUPING" in s["env_switches"]["item_grouping"]


def test_reports_are_saved_and_served(client):
    from refillcare.ml.service import save_report

    assert client.get("/api/v1/ml/reports/rolling").status_code == 404
    save_report("rolling", {"n_folds": 3, "folds": []})
    got = client.get("/api/v1/ml/reports/rolling").json()
    assert got["kind"] == "rolling" and got["report"]["n_folds"] == 3
    assert client.get("/api/v1/ml/reports/other").status_code == 404


def test_rekey_dry_run_and_confirm_guard(client, monkeypatch):
    monkeypatch.setenv("REFILLCARE_PATIENT_IDENTITY", "name_mobile")
    dry = client.get("/api/v1/ml/identity/rekey").json()
    assert dry["applied"] is False and dry["patients_after"] == 1
    assert client.post("/api/v1/ml/identity/rekey", json={"confirm": False}).status_code == 400
