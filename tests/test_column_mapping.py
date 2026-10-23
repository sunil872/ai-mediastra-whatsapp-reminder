"""Column mapping (any store's headers), slim storage (11 RefillCare columns), free quantity in supply."""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from refillcare.data.column_mapping import header_fingerprint, resolve_mapping, save_profile
from refillcare.data.schema import CANONICAL_COLUMNS, LEGACY_UNUSED_COLUMNS, apply_column_mapping, clean_text, read_sales_file


def _store_b(n=3):
    """A different billing export: other header names + price / tax columns we must not store."""
    return pd.DataFrame({
        "Bill No.": [f"S0/{100 + i}" for i in range(n)],
        "Bill Date": [(date(2026, 8, 1) + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(n)],
        "Party Name": ["RAVI KUMAR"] * n,
        "Mobile": ["+91 98480 12345"] * n,
        "Product": ["TELMA 40 TAB"] * n,
        "Prod Code": ["4060"] * n,
        "Pack": ["1X15"] * n,
        "Qty": ["2"] * n,
        "Free Qty": ["0"] * n,
        "MRP": ["120"] * n,
        "GST%": ["12"] * n,
        "Batch": ["B1"] * n,
    })


@pytest.fixture()
def db(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import database.models  # noqa: F401
    from database.connection import Base

    eng = create_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()
    eng.dispose()


def test_known_headers_are_auto_mapped_and_extra_columns_dropped():
    res = resolve_mapping(_store_b(), store_id="s1")
    assert not res.needs_confirmation
    assert res.mapping["invoice_number"] == "Bill No." and res.mapping["MOBILE_NO"] == "Mobile"
    assert res.mapping["itemCode"] == "Prod Code" and res.mapping["freeQuantity"] == "Free Qty"
    assert set(res.ignored_columns) == {"MRP", "GST%", "Batch"}
    out = apply_column_mapping(_store_b(), res.mapping)
    assert list(out.columns) == list(CANONICAL_COLUMNS)  # only the 11 RefillCare columns survive
    assert out["itemId"].tolist() == ["4060"] * 3 and out["customerId"].tolist() == ["RAVI KUMAR"] * 3


def test_short_headers_need_confirmation_with_value_based_suggestions():
    df = pd.DataFrame({"Dt": ["01-08-2026", "02-08-2026"], "Nm": ["RAVI", "SITA"], "Ph": ["9848012345", "9701112222"],
                       "Medicine": ["TELMA 40", "DOLO 650"], "Units": ["2", "1"], "S.No": ["1", "2"], "Expiry Date": ["01-2027", "02-2027"]})
    res = resolve_mapping(df, store_id="s1")
    assert res.needs_confirmation and set(res.missing_required) == {"MOBILE_NO", "customerName"}
    by_field = {f["field"]: f for f in res.to_dict()["fields"]}
    assert by_field["MOBILE_NO"]["column"] == "Ph" and by_field["MOBILE_NO"]["source"] == "suggestion"
    assert by_field["invoice_date"]["column"] == "Dt"  # expiry date is not taken as the bill date
    fixed = resolve_mapping(df, store_id="s1", user_mapping={**res.mapping, "MOBILE_NO": "Ph", "customerName": "Nm"})
    assert not fixed.needs_confirmation


def test_saved_profile_is_reused_for_the_same_layout(db):
    df = pd.DataFrame({"Dt": ["01-08-2026"], "Nm": ["RAVI"], "Ph": ["9848012345"], "Medicine": ["TELMA 40"], "Units": ["2"]})
    mapping = {"invoice_date": "Dt", "customerName": "Nm", "MOBILE_NO": "Ph", "itemName": "Medicine", "quantity": "Units"}
    save_profile(db, list(df.columns), mapping, store_id="s1", confirmed_by="tester")
    again = resolve_mapping(df[["Units", "Medicine", "Ph", "Nm", "Dt"]], db=db, store_id="s1")  # column order may change
    assert not again.needs_confirmation and all(again.matches[f].source == "saved" for f in mapping)
    other_store = resolve_mapping(df, db=db, store_id="s2")
    assert other_store.needs_confirmation  # profiles are per store
    assert header_fingerprint(["A", "b"]) == header_fingerprint(["B", "a"])


def test_saved_profile_rejects_incomplete_or_duplicate_mappings(db):
    with pytest.raises(ValueError, match="Required fields"):
        save_profile(db, ["Dt", "Qty"], {"invoice_date": "Dt", "quantity": "Qty"}, store_id="s1")
    with pytest.raises(ValueError, match="two fields"):
        save_profile(db, ["Dt", "N", "P", "Q"], {"invoice_date": "Dt", "customerName": "N", "itemName": "N", "MOBILE_NO": "P", "quantity": "Q"}, store_id="s1")


def test_free_quantity_is_optional():
    df = _store_b().drop(columns=["Free Qty"])
    res = resolve_mapping(df, store_id="s1")
    assert not res.needs_confirmation and "freeQuantity" not in res.mapping
    assert any("free-quantity" in w for w in res.warnings())
    out = apply_column_mapping(df, res.mapping)
    assert "freeQuantity" in out.columns and out["freeQuantity"].isna().all()


def test_excel_style_numbers_do_not_corrupt_mobiles_or_codes():
    out = clean_text(pd.Series([9876543210.0, "4060.0", " 98480 12345 ", None]))
    assert out.tolist()[:3] == ["9876543210", "4060", "98480 12345"] and pd.isna(out.iloc[3])
    df = read_sales_file(b"Mobile,Qty\n9876543210,2\n,1\n", "x.csv")
    assert df["Mobile"].tolist()[0] == "9876543210"


def test_free_quantity_extends_days_of_supply():
    from refillcare.engine.unified_engine import UnifiedRefillDecisionEngine

    def rows(free_on_last):
        # 1 strip of 15 every 15 days (1 tab/day); the latest bill may carry a free strip
        return pd.DataFrame([{"customerId": "P", "customerName": "P", "MOBILE_NO": "9848012345", "itemId": "T40", "itemName": "TELMA 40 TAB",
                              "invoice_date": pd.Timestamp(date(2026, 1, 1) + timedelta(days=15 * k)), "quantity": 1,
                              "freeQuantity": free_on_last if k == 2 else 0, "packing": "1X15", "refillcare_eligible": True} for k in range(3)])
    eng = UnifiedRefillDecisionEngine(identity_guard=False)
    _, paid = eng.evaluate_all_customer_items(rows(0), as_of_date=date(2026, 2, 1))
    _, free = eng.evaluate_all_customer_items(rows(1), as_of_date=date(2026, 2, 1))
    assert paid[0].dos_days == 15 and free[0].dos_days == 30  # 1 paid + 1 free strip = 30 days at 1 tab/day


def test_upload_stores_only_refillcare_columns(tmp_path):
    from refillcare.data.sales_ingestion import process_monthly_sales_data

    data_dir = tmp_path / "store"
    res = process_monthly_sales_data(_store_b(), existing_history_df=pd.DataFrame(), source_filename="b.csv", data_dir=data_dir)
    assert res["status"] == "success"
    master = pd.read_csv(data_dir / "customer_data_fields.csv", dtype=str)
    assert list(master.columns) == list(CANONICAL_COLUMNS) and len(master) == 3
    assert set(res["metrics"]["ignored_columns"]) == {"MRP", "GST%", "Batch"}
    hist = pd.read_parquet(data_dir / "processed" / "purchase_history.parquet")
    assert not ({"MRP", "GST%", "Batch"} | set(LEGACY_UNUSED_COLUMNS)) & set(hist.columns)


def test_legacy_wide_history_is_slimmed_on_next_upload(tmp_path):
    from refillcare.data.sales_ingestion import process_monthly_sales_data

    legacy = apply_column_mapping(_store_b(), resolve_mapping(_store_b(), store_id="s").mapping)
    for c in LEGACY_UNUSED_COLUMNS:
        legacy[c] = "1"
    legacy["invoice_date"] = pd.to_datetime(legacy["invoice_date"], dayfirst=True)
    new = _store_b(1)
    new["Bill Date"] = ["10/08/2026"]
    new["Bill No."] = ["S0/999"]
    res = process_monthly_sales_data(new, existing_history_df=legacy, source_filename="n.csv", data_dir=tmp_path / "s")
    assert res["status"] == "success"
    assert not set(LEGACY_UNUSED_COLUMNS) & set(res["updated_history_df"].columns)


def test_missing_required_column_blocks_processing():
    from refillcare.data.sales_ingestion import process_monthly_sales_data

    df = _store_b().drop(columns=["Mobile"])
    res = process_monthly_sales_data(df, existing_history_df=pd.DataFrame(), data_dir=Path("unused"))
    assert res["status"] == "error" and "Mobile" in res["message"]
    assert res["column_mapping"]["needs_confirmation"]


def test_catalog_accepts_itemcode_header(tmp_path):
    from refillcare.data.enrichment import load_salt_master

    p = tmp_path / "cat.csv"
    p.write_text("S.No,CATEGORY,itemCode,Item Name,PACK,SALT,ITEMCAT\n1,PHARMA,4060,TELMA 40MG TAB,1X30,TELMISARTAN-40Mg [00],PHARMA\n", encoding="utf-8")
    m = load_salt_master(p)
    assert m.loc[0, "Code"] == "4060" and m.loc[0, "SALT"].startswith("TELMISARTAN")


def test_slim_script_keeps_raw_backup_and_rows(tmp_path, monkeypatch):
    import importlib.util

    data_dir = tmp_path / "store"
    (data_dir / "processed").mkdir(parents=True)
    wide = apply_column_mapping(_store_b(), resolve_mapping(_store_b(), store_id="s").mapping)
    for c in LEGACY_UNUSED_COLUMNS:
        wide[c] = "9"
    wide.to_csv(data_dir / "customer_data_fields.csv", index=False)
    wide.to_parquet(data_dir / "processed" / "purchase_history.parquet", index=False)
    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(data_dir))
    spec = importlib.util.spec_from_file_location("slim", Path(__file__).resolve().parent.parent / "scripts" / "slim_sales_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from refillcare.store import StoreConfig

    store = StoreConfig.from_env()
    assert mod.slim_master(store, apply=False)["status"] == "dry run"
    assert len(pd.read_csv(store.master_csv, nrows=0).columns) == len(CANONICAL_COLUMNS) + len(LEGACY_UNUSED_COLUMNS)
    r = mod.slim_master(store, apply=True)
    assert r["status"] == "slimmed" and r["rows"] == 3
    assert list(pd.read_csv(store.master_csv, dtype=str).columns) == list(CANONICAL_COLUMNS)
    raw = list(store.raw_backup_dir.glob("customer_data_fields_raw_*.csv"))
    assert len(raw) == 1 and "mrp" in pd.read_csv(raw[0], nrows=0).columns
    assert mod.slim_parquet(store.history_parquet, apply=True)["status"] == "slimmed"
    assert mod.slim_master(store, apply=True)["status"] == "already slim"  # idempotent, raw backup kept once


def test_api_confirm_flow(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import database.models  # noqa: F401
    from api.main import app
    from database.connection import Base, get_db

    data_dir = tmp_path / "store"
    (data_dir / "processed").mkdir(parents=True)
    monkeypatch.setenv("REFILLCARE_DATA_DIR", str(data_dir))
    eng = create_engine(f"sqlite:///{(tmp_path / 'api.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    S = sessionmaker(bind=eng)

    def _db():
        d = S()
        try:
            yield d
        finally:
            d.close()

    app.dependency_overrides[get_db] = _db
    try:
        client = TestClient(app)
        csv = "Dt,Nm,Ph,Medicine,Units,Rate\n01-08-2026,RAVI,9848012345,TELMA 40,2,120\n"
        files = lambda: {"file": ("s.csv", io.BytesIO(csv.encode()), "text/csv")}  # noqa: E731
        det = client.post("/api/v1/sales/column-mapping/detect", files=files()).json()
        assert det["needs_confirmation"]
        pre = client.post("/api/v1/sales/preview", files=files()).json()
        assert pre["can_process"] is False and pre["column_mapping"]["needs_confirmation"]
        assert client.post("/api/v1/sales/ingest", files=files()).status_code == 400

        mapping = {"invoice_date": "Dt", "customerName": "Nm", "MOBILE_NO": "Ph", "itemName": "Medicine", "quantity": "Units"}
        saved = client.post("/api/v1/sales/column-mapping", json={"headers": det["headers"], "mapping": mapping})
        assert saved.status_code == 200
        pre2 = client.post("/api/v1/sales/preview", files=files()).json()
        assert pre2["can_process"] and pre2["valid_records"] == 1
        assert pre2["column_mapping"]["ignored_columns"] == ["Rate"]
        profiles = client.get("/api/v1/sales/column-mapping/profiles").json()
        assert len(profiles) == 1 and profiles[0]["mapping"]["MOBILE_NO"] == "Ph"
        override = client.post("/api/v1/sales/preview", files=files(), data={"column_mapping": json.dumps(mapping)}).json()
        assert override["can_process"]
    finally:
        app.dependency_overrides.clear()
        eng.dispose()
