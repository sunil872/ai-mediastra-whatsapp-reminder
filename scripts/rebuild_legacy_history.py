"""Rebuild the store's original sales history (batch LEGACY_HISTORY) from the master CSV, line by line.

    .venv\\Scripts\\python scripts\\rebuild_legacy_history.py --dry-run     # report only, no changes
    .venv\\Scripts\\python scripts\\rebuild_legacy_history.py               # rebuild + re-score everyone

Why (9 Nov 2026): the first load kept ONE line per bill + date + medicine and dropped the strips of the
other lines (e.g. S0/26135 NODOSIS 1 + 3 + 1 strips was stored as 1). Now every line is stored exactly as
billed - same validation as daily uploads, identical lines kept (line_key counts repeats) - and the engine
adds them up per bill (``refillcare.data.cleaning.merge_bill_lines``). B2B (SB/) lines are stored as not
eligible (counted, never reminded); returns are netted like uploads.

Only lines dated BEFORE the first later upload are rebuilt; later upload batches are not touched.

    --store-returns  (12 Nov 2026) keep the original history's return lines as rows (CUSTOMER_RETURN) WITHOUT
                     changing any sale: the netting is replayed in memory on the CSV's original quantities with
                     the same rule, checked to reproduce every stored sale exactly, and only then the return
                     lines are saved with the sale each one was subtracted from (or RETURN_WAITING).
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
from refillcare.data.incremental_ingest import b2b_lines_to_store, get_population_stats, rescore_customers, revert_returned_sales  # noqa: E402
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


def replay_returns(db, valid: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Link every original-history return line to the sale it was subtracted from. Read-only.

    1. Replay ``history_store.net_returns`` in memory on the CSV's original quantities (same order, window and
       correction rule as the rebuild) for the patient-medicines that have returns.
    2. Patient-medicines where the replay differs from the database (their returns were applied in an earlier,
       different sequence) are linked from the database state instead: each reduced sale gets the return(s)
       that took its strips (a late full return carries its return date in the sale's reason).
    3. Strict check on EVERY sale: original strips - stored strips == strips its linked returns took off.
    """
    window, correction = hs.return_window_days(), hs.return_correction_days()
    rk = pd.DataFrame({"cid": returns["customerId"].astype(str).values, "iid": returns["itemId"].astype(str).values,
                       "d": pd.to_datetime(returns["invoice_date"], errors="coerce").values,
                       "q": pd.to_numeric(returns["quantity"], errors="coerce").abs().values}, index=returns.index)
    ret_groups = set(zip(rk["cid"], rk["iid"]))

    v = pd.DataFrame({"key": hs.compute_line_keys(valid), "cid": valid["customerId"].astype(str).values,
                      "iid": valid["itemId"].astype(str).values,
                      "d": pd.to_datetime(valid["invoice_date"], errors="coerce").values,
                      "q": pd.to_numeric(valid["quantity"], errors="coerce").fillna(0).astype(float).values})
    v = v.drop_duplicates("key")
    dbf = pd.DataFrame(db.execute(select(Line.line_key, Line.id, Line.quantity, Line.refillcare_eligible, Line.exclusion_reason)
                                  .where(Line.import_batch_id == LEGACY, Line.transaction_type != hs.RETURN_TYPE)).all(),
                       columns=["key", "id", "dq", "de", "dr"])
    v = v.merge(dbf, on="key", how="left")
    del dbf
    missing = int(v["id"].isna().sum())
    v = v[v["id"].notna()]
    v["id"] = v["id"].astype(int)
    v["dq"] = v["dq"].astype(float)
    v["de"] = v["de"].fillna(True).astype(bool)
    v["dr"] = v["dr"].where(v["dr"].notna(), None)

    # Sales of patient-medicines without any return must be untouched
    in_g = pd.Series([g in ret_groups for g in zip(v["cid"], v["iid"])], index=v.index)
    plain = v[~in_g]
    plain_bad = int(((plain["q"] - plain["dq"]).abs() > 1e-6).sum() + (~plain["de"]).sum() - (~plain["de"] & plain["dr"].astype(str).str.startswith("INTER")).sum())
    g = v[in_g]
    sales: dict = {}
    orig: dict = {}
    dbs: dict = {}
    state: dict = {}
    for cid, iid, d, q, sid, dq, de, dr in g[["cid", "iid", "d", "q", "id", "dq", "de", "dr"]].itertuples(index=False):
        sales.setdefault((cid, iid), []).append((pd.Timestamp(d), sid))
        orig[sid], dbs[sid], state[sid] = q, (dq, de, dr if isinstance(dr, str) and dr else None), [q, True, None]
    del v, g, plain

    applied: dict = {}
    for idx, cid, iid, d, qty in rk.itertuples():
        if pd.isna(d) or not qty or qty <= 0:
            continue
        lo = d - pd.Timedelta(days=window)
        best = None
        for sd, sid in sales.get((cid, iid), []):
            st = state[sid]
            if lo <= sd <= d and st[0] > 0 and st[1] and (best is None or (sd, sid) > best):
                best = (sd, sid)
        if best is None:
            continue
        sd, sid = best
        st = state[sid]
        before = st[0]
        st[0] = max(0.0, before - qty)
        applied[idx] = (sid, round(before - st[0], 3))
        if st[0] <= 0:
            lag = int((d.normalize() - sd.normalize()).days)
            st[1] = False
            st[2] = f"{hs.RETURN_FULL_LATE}:{d.date().isoformat()}" if lag > correction else hs.RETURN_FULL_CORRECTION
    group_of = {sid: k for k, lst in sales.items() for _, sid in lst}
    bad_groups = set()
    for sid, (q, e, r) in state.items():
        dq, de, dr = dbs[sid]
        if abs(dq - q) > 1e-6 or de != e or (dr or None) != r:
            bad_groups.add(group_of[sid])
    replay_mismatch_groups = len(bad_groups)

    # 2. relink the differing patient-medicines from the database state
    applied = {i: a for i, a in applied.items() if (rk.at[i, "cid"], rk.at[i, "iid"]) not in bad_groups}
    inferred = 0
    win = pd.Timedelta(days=window)
    for grp in bad_groups:
        rets = sorted([i for i in rk.index[(rk["cid"] == grp[0]) & (rk["iid"] == grp[1])]
                       if pd.notna(rk.at[i, "d"]) and rk.at[i, "q"] > 0], key=lambda i: (rk.at[i, "d"], i))
        used: set = set()
        for sd, sid in sorted(sales.get(grp, [])):
            need = round(orig[sid] - dbs[sid][0], 3)
            if need <= 0:
                continue
            reason = dbs[sid][2] or ""
            late_d = pd.Timestamp(reason.split(":", 1)[1]) if reason.startswith(hs.RETURN_FULL_LATE + ":") else None
            cands = [i for i in rets if i not in used and sd <= rk.at[i, "d"] <= sd + win]
            cands.sort(key=lambda i: (0 if late_d is not None and rk.at[i, "d"] == late_d else 1, rk.at[i, "d"], i))
            for i in cands:
                if need <= 0:
                    break
                take = round(min(float(rk.at[i, "q"]), need), 3)
                applied[i] = (sid, take)
                used.add(i)
                need = round(need - take, 3)
                inferred += 1

    # 3. strict check on every sale of a patient-medicine with returns (the rest checked above)
    took: dict = {}
    for sid, q in applied.values():
        took[sid] = round(took.get(sid, 0.0) + q, 3)
    off = [sid for sid in orig if abs(round(orig[sid] - dbs[sid][0], 3) - took.get(sid, 0.0)) > 1e-6]
    return {"applied": applied, "sales_missing_in_db": missing, "sales_checked": len(orig),
            "sales_without_returns_changed": plain_bad, "returns": int(len(returns)), "returns_applied": len(applied),
            "returns_waiting": int(len(returns) - len(applied)),
            "patient_medicines_replay_differs": replay_mismatch_groups, "returns_linked_from_db_state": inferred,
            "sales_not_explained": len(off), "not_explained_examples": off[:5],
            "mismatches": len(off) + plain_bad}


def store_legacy_returns(db, returns: pd.DataFrame, applied: dict) -> dict:
    """Save the original history's return lines as rows (caller has verified the replay)."""
    ret = returns.copy()
    ret["transaction_type"] = hs.RETURN_TYPE
    ret["refillcare_eligible"] = False
    ret["exclusion_reason"] = [hs._return_status(applied.get(i)) for i in ret.index]
    keep = [c for c in hs.COLUMN_MAP if c in ret.columns]
    from refillcare.data.incremental_ingest import store_write_lock

    with store_write_lock(db):
        res = hs.append_lines(db, ret[keep], LEGACY)
        batch = db.get(ImportBatchModel, LEGACY)
        w = dict(batch.data_warnings or {}) if not isinstance(batch.data_warnings, str) else json.loads(batch.data_warnings)
        w["returns_stored"] = True
        w["returns_stored_on"] = "12-11-2026 (from the master CSV, sales unchanged)"
        batch.data_warnings = w
        db.commit()
    hs.clear_cache()
    return res


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
        from refillcare.engine.rekey import retire_stale_decisions

        tot["stale_decisions"] = retire_stale_decisions(db)
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
    ap.add_argument("--store-returns", action="store_true",
                    help="only keep the original history's return lines as rows (sales unchanged; verified replay)")
    ap.add_argument("--retire-stale-only", action="store_true",
                    help="only delete decisions that no longer stand (old patient keys, rolled back / returned purchases)")
    args = ap.parse_args()

    t0 = time.time()
    if args.retire_stale_only:
        from refillcare.engine.rekey import retire_stale_decisions

        with SessionLocal() as db:
            print(json.dumps(retire_stale_decisions(db), indent=2))
        return 0
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
        if args.store_returns:
            del raw, legacy_raw, merged
            if __import__("os").getenv("REPLAY_ORDER") == "date":
                returns = returns.assign(_d=pd.to_datetime(returns["invoice_date"], errors="coerce")).sort_values("_d", kind="stable").drop(columns="_d")
            rp = replay_returns(db, valid, returns)
            print(json.dumps({k: v for k, v in rp.items() if k != "applied"}, indent=2, default=str))
            ok = rp["mismatches"] == 0 and rp["sales_missing_in_db"] == 0
            if not ok:
                print("Replay does NOT reproduce the stored sales: nothing saved.")
                return 1
            if args.dry_run:
                print("Dry run: replay verified, nothing saved.")
                return 0
            if not args.i_have_a_backup:
                print("Refusing to change the database without --i-have-a-backup.")
                return 1
            res = store_legacy_returns(db, returns, rp["applied"])
            print(f"Saved {res['inserted']:,} return lines ({res['skipped']:,} already saved); sales unchanged. "
                  f"{time.time() - t0:,.0f} s")
            return 0
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
            summary = {k: v for k, v in netting.items() if k != "returned_sales"}  # (timestamps are not JSON)
            batch.data_warnings = {"rebuilt": "11-11-2026 (returns: 90-day window, late full return = review)",
                                   "returns_netted": summary}
            db.commit()
        hs.clear_cache()
        # same follow-up as an upload: corrections bring the earlier schedule back, late full returns pause
        # the medicine (decision of the returned purchase invalidated; the re-score keeps it paused)
        reverted = revert_returned_sales(db, netting) if netting.get("returned_sales") else {}
        hs.clear_cache()
        print(f"Stored {ins['inserted']:,} customer lines + {ins_b2b['inserted']:,} B2B lines; "
              f"returns {summary}; follow-up {reverted}; uploads {norm}")

        if args.no_rescore:
            return 0
    print(json.dumps(rescore_everyone(), indent=2))
    print(json.dumps(redate_unsent_stages(), indent=2))
    print(f"Done in {time.time() - t0:,.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
