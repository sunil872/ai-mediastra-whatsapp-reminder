"""Rebuild the store's original sales history (batch LEGACY_HISTORY) from the master CSV, line by line.

    .venv\\Scripts\\python scripts\\rebuild_legacy_history.py --dry-run     # report only, no changes
    .venv\\Scripts\\python scripts\\rebuild_legacy_history.py               # rebuild + re-score everyone

Why (9 Nov 2026): the first load kept ONE line per bill + date + medicine and dropped the strips of the
other lines (e.g. S0/26135 NODOSIS 1 + 3 + 1 strips was stored as 1). Now every line is stored exactly as
billed - same validation as daily uploads, identical lines kept (line_key counts repeats) - and the engine
adds them up per bill (``refillcare.data.cleaning.merge_bill_lines``). B2B (SB/) lines are stored as not
eligible (counted, never reminded); returns are netted like uploads.

Only lines dated BEFORE the first later upload are rebuilt; later upload batches are not touched.
Back up the database first (SQLite: the script refuses to run without --i-have-a-backup).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402
from sqlalchemy import delete, func, select  # noqa: E402

from database.connection import SessionLocal  # noqa: E402
from database.models import ImportBatchModel, SalesTransactionModel as Line  # noqa: E402
from refillcare.data import history_store as hs  # noqa: E402
from refillcare.data.cleaning import merge_bill_lines  # noqa: E402
from refillcare.data.incremental_ingest import b2b_lines_to_store, get_population_stats, rescore_customers  # noqa: E402
from refillcare.data.sales_ingestion import _enrich_with_catalog_salt, validate_monthly_sales_data  # noqa: E402

LEGACY = "LEGACY_HISTORY"


def normalise_upload_batches(db) -> dict:
    """Later upload batches: same cleaned patient ids as the history + matching line keys (re-upload safe)."""
    from refillcare.data.cleaning import clean_customer_identifier

    changed_rows = 0
    batch_ids = [b for (b,) in db.execute(select(ImportBatchModel.import_batch_id).where(
        ImportBatchModel.import_batch_id != LEGACY, ImportBatchModel.is_active.is_(True)))]
    for bid in batch_ids:
        rows = db.query(Line).filter(Line.import_batch_id == bid).order_by(Line.id).all()
        if not rows:
            continue
        for r in rows:
            def _c(v):  # same rule as upload validation: name-style ids only
                return (clean_customer_identifier(v) or v) if v and not re.search(r"[\d_]", str(v)) else v

            cid, cname = _c(r.customer_id), _c(r.customer_name)
            if cid != r.customer_id or cname != r.customer_name:
                r.customer_id, r.customer_name = cid, cname
                changed_rows += 1
        frame = pd.DataFrame({
            "invoice_number": [r.invoice_number for r in rows], "itemCode": [r.item_code or r.item_id for r in rows],
            "itemId": [r.item_id for r in rows], "invoice_date": [r.invoice_date for r in rows],
            "customerName": [r.customer_name or r.customer_id for r in rows], "customerId": [r.customer_id for r in rows],
            "quantity": [r.quantity for r in rows], "freeQuantity": [r.free_quantity for r in rows],
        })
        for r, k in zip(rows, hs.compute_line_keys(frame)):
            r.line_key = k
    db.flush()
    return {"upload_batches": len(batch_ids), "rows_with_cleaned_patient_id": changed_rows}


def retire_orphan_cycles(db) -> int:
    """Active reminder cycles whose patient-medicine key no longer exists in the history (old punctuated ids)."""
    from datetime import datetime

    from database.models import ReminderCycleModel, ReminderStageModel
    from refillcare.data.identity import normalise_trajectory_keys

    keyed = normalise_trajectory_keys(hs.load_all(db))
    live = set(keyed["customerId"].astype(str).str.strip() + "_" + keyed["itemId"].astype(str).str.strip())
    now = datetime.utcnow()
    retired = 0
    for cyc in db.query(ReminderCycleModel).filter(ReminderCycleModel.is_active.is_(True)).all():
        if cyc.customer_item_key in live:
            continue
        cyc.is_active, cyc.superseded_at = False, now
        for st in db.query(ReminderStageModel).filter(ReminderStageModel.cycle_id == cyc.cycle_id,
                                                      ReminderStageModel.status.in_(("PENDING", "DUE", "APPROVED"))).all():
            st.status, st.failure_reason, st.updated_at = "CANCELLED", "Patient id cleaned (history rebuild 09-11-2026)", now
        retired += 1
    db.commit()
    return retired


def _first_upload_date(db):
    return db.execute(select(func.min(ImportBatchModel.min_sales_date)).where(
        ImportBatchModel.import_batch_id != LEGACY, ImportBatchModel.is_active.is_(True),
        ImportBatchModel.min_sales_date.isnot(None))).scalar()


def redate_unsent_stages() -> dict:
    """One-time after a data correction: active cycles whose decision's expected refill date moved (strips now
    added up) get their UNSENT stages moved to the new date. Sent / failed / cancelled stages are untouched.
    The daily scheduler deliberately keeps an existing cycle as it is, so this is done here once."""
    from datetime import datetime, timedelta

    from database.models import RefillDecisionModel, ReminderCycleModel, ReminderStageModel
    from refillcare.engine.persistence import RefillPersistenceManager, get_stage_message_template

    pm = RefillPersistenceManager()
    now = datetime.utcnow()
    out = {"cycles_redated": 0, "stages_redated": 0, "stages_expired": 0}
    with SessionLocal() as db:
        rows = (db.query(ReminderCycleModel, RefillDecisionModel)
                .join(RefillDecisionModel, ReminderCycleModel.decision_id == RefillDecisionModel.decision_id)
                .filter(ReminderCycleModel.is_active.is_(True), RefillDecisionModel.is_eligible.is_(True),
                        RefillDecisionModel.expected_refill_date.isnot(None),
                        ReminderCycleModel.expected_refill_date != RefillDecisionModel.expected_refill_date).all())
        for cyc, dec in rows:
            new_exp = dec.expected_refill_date
            stages = db.query(ReminderStageModel).filter(
                ReminderStageModel.cycle_id == cyc.cycle_id,
                ReminderStageModel.status.in_(("PENDING", "DUE", "APPROVED"))).all()
            for st in stages:
                st.target_send_date = new_exp + timedelta(days=int(st.stage_offset))
                st.expected_refill_date = new_exp
                st.message_text = get_stage_message_template(
                    stage_offset=int(st.stage_offset), customer_name=dec.customer_name, store_name=pm.store_name,
                    medicine_name=dec.item_name, expected_refill_date=new_exp, store_contact=pm.store_contact)
                st.updated_at = now
                out["stages_redated"] += 1
            cyc.expected_refill_date, cyc.updated_at = new_exp, now
            out["cycles_redated"] += 1
        db.commit()
        _, latest = hs.date_bounds(db)
        out["stages_expired"] = pm.expire_past_due_stages(db, today=(latest + timedelta(days=1)) if latest else None)
        db.commit()
    return out


def rescore_everyone(chunk: int = 800) -> dict:
    """Re-score every patient in batches, each with a fresh session (bounded memory)."""
    with SessionLocal() as db:
        get_population_stats(db, force_refresh=True)
        customers = sorted(hs.load_all(db)["customerId"].dropna().astype(str).unique().tolist())
    hs.clear_cache()
    print(f"Re-scoring {len(customers):,} customers in batches of {chunk} ...", flush=True)
    tot = {"customer_items_rescored": 0, "cycles_created": 0, "cycles_superseded": 0, "stages_created": 0, "stages_superseded": 0}
    for i in range(0, len(customers), chunk):
        with SessionLocal() as db:
            res = rescore_customers(db, customers[i:i + chunk])
        for k in ("customer_items_rescored", "cycles_created", "cycles_superseded", "stages_created", "stages_superseded"):
            tot[k] += int(res.get(k) or 0)
        tot["evaluation_as_of_date"] = res.get("evaluation_as_of_date")
        print(f"  {min(i + chunk, len(customers)):,}/{len(customers):,} customers", flush=True)
    with SessionLocal() as db:
        tot["retired_cycles_of_old_ids"] = retire_orphan_cycles(db)
    hs.clear_cache()
    return tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/refillcare/customer_data_fields.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-have-a-backup", action="store_true")
    ap.add_argument("--no-rescore", action="store_true")
    ap.add_argument("--rescore-only", action="store_true", help="skip the rebuild; only re-score every patient")
    ap.add_argument("--redate-only", action="store_true", help="only move unsent reminders to the corrected dates")
    args = ap.parse_args()

    t0 = time.time()
    if args.redate_only:
        print(json.dumps(redate_unsent_stages(), indent=2))
        return 0
    if args.rescore_only:
        print(json.dumps(rescore_everyone(), indent=2))
        print(json.dumps(redate_unsent_stages(), indent=2))
        print(f"Done in {time.time() - t0:,.0f} s")
        return 0
    raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    with SessionLocal() as db:
        cutoff = _first_upload_date(db)
        if cutoff is None:
            print("No later upload batch found; refusing to guess the cut-off date.")
            return 1
        # The original export (dd-mm-yy). Rows appended later in dd-mm-yyyy are upload copies or old test
        # rows (CUST_1, NEW_CUST_99, ...) and are not part of the original history.
        d = pd.to_datetime(raw["invoice_date"], format="%d-%m-%y", errors="coerce")
        legacy_raw = raw[d.notna() & (d < pd.Timestamp(cutoff))].copy()
        print(f"Rows not in the original export format (skipped): {int(d.isna().sum()):,}")
        legacy_raw["invoice_date"] = d[legacy_raw.index].dt.strftime("%Y-%m-%d")  # unambiguous for validation
        print(f"CSV rows {len(raw):,}; original history (before {cutoff:%d-%m-%Y}): {len(legacy_raw):,}")

        valid, review, metrics = validate_monthly_sales_data(legacy_raw)
        if metrics.get("error"):
            print("Validation error:", metrics["error"])
            return 1
        b2b = b2b_lines_to_store(review)
        q = pd.to_numeric(review.get("quantity"), errors="coerce")
        returns = review[(q < 0) & review["customerId"].notna()] if not review.empty else review
        valid = _enrich_with_catalog_salt(valid)

        old_lines = int(db.execute(select(func.count(Line.id)).where(Line.import_batch_id == LEGACY)).scalar() or 0)
        old_keys = {(c, i) for c, i in db.execute(select(Line.customer_id, Line.item_id).where(Line.import_batch_id == LEGACY).distinct())}
        new_keys = set(zip(valid["customerId"].astype(str), valid["itemId"].astype(str)))
        merged = merge_bill_lines(valid)
        ex = merged[(merged["invoice_number"] == "S0/26135") & (merged["itemId"].astype(str) == "8468")]
        report = {
            "customer_sale_lines": len(valid), "b2b_lines": len(b2b), "returns": len(returns),
            "other_review_lines": int(len(review) - len(b2b) - len(returns)),
            "purchases_after_adding_bill_lines": len(merged), "old_stored_lines": old_lines,
            "patient_medicine_keys_old": len(old_keys), "keys_new": len(new_keys),
            "keys_in_both": len(old_keys & new_keys),
            "example_S0_26135_NODOSIS_strips": float(ex["quantity"].sum()) if not ex.empty else None,
        }
        print(json.dumps(report, indent=2))
        if args.dry_run:
            print("Dry run: nothing changed.")
            return 0
        if not args.i_have_a_backup:
            print("Refusing to change the database without --i-have-a-backup.")
            return 1

        from refillcare.data.incremental_ingest import store_write_lock

        with store_write_lock(db):
            db.execute(delete(Line).where(Line.import_batch_id == LEGACY))
            ins = hs.append_lines(db, valid, LEGACY)
            ins_b2b = hs.append_lines(db, b2b, LEGACY) if not b2b.empty else {"inserted": 0}
            netting = hs.net_returns(db, returns) if not returns.empty else {}
            norm = normalise_upload_batches(db)
            batch = db.get(ImportBatchModel, LEGACY)
            batch.source_filename = f"{Path(args.csv).name} (rebuilt line by line 09-11-2026)"
            batch.record_count = int(len(legacy_raw))
            batch.records_inserted = int(ins["inserted"] + ins_b2b["inserted"])
            batch.records_skipped = int(len(legacy_raw) - batch.records_inserted)
            batch.records_requiring_review = int(len(review) - len(b2b))
            batch.data_warnings = {"rebuilt": "09-11-2026", "returns_netted": netting}
            db.commit()
        hs.clear_cache()
        print(f"Stored {ins['inserted']:,} customer lines + {ins_b2b['inserted']:,} B2B lines; returns {netting}; uploads {norm}")

        if args.no_rescore:
            return 0
    print(json.dumps(rescore_everyone(), indent=2))
    print(json.dumps(redate_unsent_stages(), indent=2))
    print(f"Done in {time.time() - t0:,.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
