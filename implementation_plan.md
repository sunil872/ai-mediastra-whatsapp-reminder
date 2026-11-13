# RefillCare — Implementation Plan (executed to date, and what is next)

Single store (NGOs Colony, `store-1`, sales Dec 2020 – 15 Oct 2026) in v1; multi-store through the cloud-hosted
Mediastra web application in v2 (built by the backend team).
How to set up, run and use the system: [README.md](README.md). What is still needed for live mode (store + backend team):
[future_implementation/GO_LIVE_CHECKLIST.md](future_implementation/GO_LIVE_CHECKLIST.md). Architecture for the backend team:
[future_implementation/PLAN_BACKEND_DEVOPS_TENANCY.md](future_implementation/PLAN_BACKEND_DEVOPS_TENANCY.md). Mediastra full-stack
integration (NestJS / Angular): [docs/integration/](docs/integration/). Client-facing deployment options:
[docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md](docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md).

*Last updated: 13 Nov 2026.*

Status legend: **Done** = built and verified · **Configured** = built, switched by a setting · **Backend team** = owned by Mediastra · ⏳ = next

---

## 0. Current state (13 Nov 2026)

| Area | State |
|---|---|
| Store | NGOs Colony, `store-1`, single store on local SQLite (`data/refillcare/processed/enterprise.db`) |
| Sales data | 24-12-2020 to 15-10-2026: 917,110 lines received = 910,927 sales lines (908,366 patient + 2,561 B2B) + 6,047 return lines saved + 136 not saved (83 invalid rows, 53 returns of the 1-15 Oct 2026 upload kept as counts) |
| Returns | 4,906 applied to their sale (4,885 original history + 21 October); 1,162 waiting for a sale; 32 older unmatched; 3,733 sales fully returned (2,350 billing corrections, 1,383 late) |
| Pending upload | 16-31 Oct 2026 (`dummy/October actual dataset 16-10-2026 to 31-10-2026.xlsx`): its batch was rolled back; upload it again |
| Production model | `hgb-20260518-20261103071906` (gate passed 3 Nov 2026); retrain after the October upload |
| Surfaces | Streamlit (6 pharmacist tabs), FastAPI (`/docs`), web UI (`/`), all on the same shared services |
| Tests | 907 passed, 1 skipped (PostgreSQL test needs a server) |
| Repository | Cleaned 13 Nov 2026 (section 6.7): folder 16.8 GB → 3.3 GB, nothing needed lost |

---

## 1. Data and ingestion

| # | Item | Status | Result |
|---|---|---|---|
| 1.1 | Sales upload (CSV / Excel), date parsing (day-first, ambiguity blocked), de-duplication, append to the store database (no file backups since 8.4), batch rollback | Done | Same code path for Streamlit, FastAPI and the web UI (`incremental_ingest.ingest_sales_batch`) |
| 1.2 | Channel filter: customer sales (`S0/`) kept, B2B transfers (`SB/`) and unknown excluded | Done | 907,080 S0 vs 2,571 SB rows |
| 1.3 | Chronic / acute medication classifier (Supabase rules + local cache, salt catalogue fallback) | Done | |
| 1.4 | **Column mapping for any store**: alias config (≈200 names) + value checks, confirm screen only when needed, mapping saved per store and file layout, list / delete | Done | Real 23-column file and a differently named export map with no questions |
| 1.5 | **Only 11 fields stored** (date, invoice, name, item code, item name, packing, quantity, free quantity, mobile, customer id, item id); price / tax / batch / expiry columns dropped at upload | Done | Master 132.3 → 78.5 MB; history parquet 33.1 → 17.1 MB; predictions identical on 44,601 decisions |
| 1.6 | Raw full export kept as a separate backup for training (`data/refillcare/raw_backup/`, git-ignored) | Done | |
| 1.7 | Uploads read as text (mobiles / codes never become floats); item catalogue accepts `Code` or `itemCode` | Done | |
| 1.8 | Item catalogue switched to `SALT ITEMS LIST.csv` (7 columns: S.No, CATEGORY, Code, Item Name, PACK, SALT, ITEMCAT) | Done | 36,963 items, 22,168 with a salt |

## 2. Prediction engine

| # | Item | Status | Result |
|---|---|---|---|
| 2.1 | Path A (≥ 6 purchases, stable): registry ML model or personal median, safety caps | Done | |
| 2.2 | Path B (3–5 purchases): days of supply from the patient's own pace | Done | |
| 2.3 | Pharmacist review for unstable, bulk, identity-conflict and too-short histories | Done | Coverage 58% of refills |
| 2.4 | 5 quantity archetypes (multi-pack, partial strip, early top-up, post-lapse, consensus bounds) | Done | |
| 2.5 | Features: calendar (usual day of month, fit gain), visits (other medicines, last visit), salt priors (typical interval, units per day, dose supply) — point-in-time, same code for train / backtest / serve | Done | |
| 2.6 | Free quantity counted in days of supply (training = serving) | Done | No effect on today's data (0 S0 rows with free quantity) |
| 2.7 | Prediction rules, each switchable and measured: high-stability median, insulin median, day-of-month anchor, Path B `shrink` / `blend`, device cap `observed` | Configured | Only `blend` helps: +0.46 pp within ±7, MAE 8.15 → 7.87 d on the test month; mixed (±1 pp) over 3 months but lower MAE every month |
| 2.8 | **Patient identity = name + mobile** (`REFILLCARE_PATIENT_IDENTITY=name_mobile`) | Configured | 14,071 names → 35,040 patients; 2,663 names had > 1 mobile (62% of rows); 505 identity-blocked medicines unblocked |
| 2.9 | **Therapy = salt group** (`REFILLCARE_ITEM_GROUPING=salt`): brand / strength switches continue one cycle | Configured | 73k rows regrouped; +0.7 pp within ±7 on top of name + mobile |
| 2.10 | Identity cut-over: retire old-key reminders, re-score, never re-send a sent stage, DB backup first (ML tab, API, script) | Done | |

## 3. ML lifecycle

| # | Item | Status | Result |
|---|---|---|---|
| 3.1 | Model registry (production / shadow / candidate / retired) in `enterprise.db` | Done | |
| 3.2 | Dynamic windows from the latest sale (test 30 d, validation 30 d, 45-day censor buffer) | Done | Test window now 18 Jul – 16 Aug 2026 |
| 3.3 | Training fixes: label-horizon bias guard (90 d), log-ratio target vs personal median, early stopping | Done | |
| 3.4 | Promotion gate incl. per-segment MAE check (insulin regression blocked) | Done | |
| 3.5 | Retrain cycle (train → select on validation → gate on test), freshness experiment | Done | |
| 3.6 | Rule A/B and rolling 3-month test, saved reports shown in both UIs | Done | Model beats baseline engine every month: +3.0 / +4.1 / +1.3 pp (58.7% vs 55.6% pooled, 5,065 refills) |
| 3.7 | Live outcome tracking of sent reminders (on time / early / late / no refill) | Done | |
| 3.8 | **Production retrain on the real database (3 Nov 2026)** with `scripts/retrain_production.py` (online DB backup → retrain cycle → gate → promote) in the project's Python 3.13 / scikit-learn 1.7.2 | Done | `hgb-20260518-20261103071906` promoted, gate passed; old model retired. Test 18 Jun – 17 Jul 2026: **56.7%** within ±7 d vs baseline 53.6%; MAE **10.4** vs 11.2 d. Retrain again after the 16 Oct – today upload |

Production model (real database, 3 Nov 2026):

| Model | Within ±7 d | MAE |
|---|---|---|
| Baseline engine | 53.6% | 11.2 d |
| `hgb-20260518-20261103071906` (production) | 56.7% | 10.4 d |

Earlier sandbox results (test month 18 Jul – 16 Aug 2026, 1,740 refills):

| Model | Within ±7 d | MAE |
|---|---|---|
| Raw personal median | 58.0% | 11.1 d |
| Baseline engine | 59.5% | 8.7 d |
| Previous model `hgb-20260717` | 61.4% | 8.1 d |
| New model `hgb-20260617-20261021065538` | 62.2% | 8.15 d |
| New model + `blend` | 62.8% (with device cap `observed`) | 7.85 d |

## 4. Reminders and WhatsApp

| # | Item | Status |
|---|---|---|
| 4.1 | 7-stage lifecycle (-7, -3, -1, 0, +2, +5, +45 d), repurchase supersedes pending stages, past-due stages expire | Done |
| 4.2 | Med-Sync: one message per patient for several medicines | Done |
| 4.3 | Production send flow (7 Nov 2026): one **Send WhatsApp Messages (N)** button for today's reminders → confirmation dialog with counts (to send, medicines, skipped: no valid mobile / opted out / already sent) → **Yes** = pharmacist approval + live send of all. Dry-run and the Batch Outreach section removed from the UI. Still enforced: stale-sales guard (override needs a reason), STOP opt-out, valid mobile, idempotent (no double sends). API parity: `POST /api/v1/whatsapp/dispatch` with `dry_run=false, approve_before_send=true` (batch_size up to 5,000) | Done |
| 4.4 | Opt-out: STOP / UNSUBSCRIBE via webhook, exact START to opt in, pharmacist **Stop / Resume messages** by searching the patient's name or mobile (no customer ID needed; `GET /api/customers/search`, `POST /api/customers/whatsapp-messages`); every send path skips opted-out phones (last-10-digit match) | Done |
| 4.5 | Standalone text / image campaign apps (`whatsapp_campaigns/`) | Done |
| 4.6 | **Messages Sent** view for pharmacists: one day (India time), Sent / Failed counts, time · patient · mobile · plain failure reason; dry-run test rows and technical columns hidden. Full audit data unchanged in `whatsapp_delivery_logs` (needed for no-double-send, billing, outcomes). API: `GET /api/v1/whatsapp/delivery-log` | Done |
| 4.7 | **Pharmacist "Add to reminders"** (Patients Needing Review, 9 Nov 2026): list of patients NOT getting reminders whose medicine is chronic or not yet classified by the medication rules (Supabase + built-in; known non-chronic never shown), for ONE day (default today) with search by customer name / mobile; tick one row → add with editable refill gap and next due date, tick many (or all) → bulk add with each patient's own gap (`POST /api/v1/reminders/pharmacist-add-bulk`). Stored in `pharmacist_reminder_overrides` and applied inside `save_refill_decisions` / `sync_reminder_cycles`, so every re-scoring keeps it (date moves after a repurchase; retired automatically when the engine starts reminding the patient itself). Passes the send-time chronic filter, never in the control group, undo via Remove. API: `GET /api/v1/reminders/review-candidates`, `POST /api/v1/reminders/pharmacist-add`, `POST /api/v1/reminders/pharmacist-remove`, `GET /api/v1/reminders/pharmacist-added` | Done |
| 4.8 | **Store-dynamic app + Pharmacy Overview (9 Nov 2026)**: Streamlit opens the store from `?store_id=` (Mediastra link; `REFILLCARE_ALLOWED_STORES` in production), every DB cache keyed by store (`store_cache_data`), legacy parquet/CSV used only for the original store, new store with no uploads shows empty numbers. Overview = this store, this month only (`refillcare/engine/overview_kpis.py`, also in `GET` operations KPI): customers monitored, patients getting reminders, reminder messages (up to 7 per patient-medicine), due today, patients needing review today. Removed next-month horizon and hard-coded numbers (909,657 raw base, 2,264 B2B, 98.1 %, model-calibration and dosage-regimen panels). Review list query 25-65 s → ~2 s | Done |
| 4.9 | **One send rule + full surface sync (9 Nov 2026)**: reminders are sent only for medicines the Medication Rules mark chronic, or that a pharmacist added (`persistence.medicine_can_be_sent`, applied once in the shared reminder queue, so Streamlit lists, overview counts, API and WhatsApp send agree). Engine-scheduled 'not classified' medicines (e.g. supplements bought regularly) are not sent and appear in Patients Needing Review; adding one replaces the engine schedule. New API: `GET /api/v1/overview`, `GET /api/v1/whatsapp/send-summary`, `POST /api/v1/whatsapp/send-today` (Yes-send parity with Streamlit, same bundles via `_outreach_bundles`), `GET /api/v1/reminders/review-other` (shared `refillcare/engine/review_lists.py`); API accepts `?store_id=`. Web UI (`frontend/`) rebuilt to match Streamlit. Rule for every future change: `.agents/rules/every_change_syncs_all_surfaces.md`, `.claude/CLAUDE.md` | Done |
| 4.10 | **Bill lines added up (9 Nov 2026)**: one bill + date + medicine = ONE purchase and the strips of ALL its lines are added (different batches and the same batch scanned again, e.g. S0/26135 NODOSIS 1 + 3 + 1 = 5 strips) - `cleaning.merge_bill_lines`, used by the decision engine and purchase history; training/evaluation already sum per day. Validated on the store CSV: adding lines predicts the next visit at 32 d (real 31 d) vs 16 d when identical lines are dropped (23,641 purchases). Original history rebuilt line by line (`scripts/rebuild_legacy_history.py`; it had kept one line per bill and stored 5,855 returns as purchases), B2B (SB/) lines stored as not eligible in history and uploads, uploads clean name-style patient ids like the original load. Transaction counts shared by Streamlit and API (`history_store.channel_counts`; removed hard-coded 909,657 / 2,264 from the API) | Done |
| 4.11 | **Med-Sync bundles by send date + clean stage labels (10 Nov 2026, implemented by Sunil without Claude)**: `MedSyncEngine.cluster_patient_decisions` now groups a patient's reminders by the day the message is SENT (`target_send_date`, fallback expected refill date) instead of each medicine's own refill date, still separately per tier (DUE / FOLLOW-UP / LAPSED never mixed). Example SURYA NARAYANA 09-11-2026: 4 medicines at stages -7/-3/-1/0 -> 1 WhatsApp message instead of 4. Stage column shows DUE / FOLLOW / LAPSED (Streamlit, API `stage_display`, web UI); `target_send_date` added to `MedSyncBundleSchema` and `WhatsAppCandidateItem`. Tests: `tests/refillcare/test_med_sync.py` (same send date, follow-up bundling, tier isolation). **Result reported by Sunil: ~25% fewer WhatsApp messages.** | Done |
| 4.12 | **Store-visit scenario fixes (10 Nov 2026)**, found by automated scenario tests on a real patient (CHANDRA MOHAN, partial refill): **A** Med-Sync bundle id now includes the send day (`..._S<yyyymmdd>`); it carried only the refill date, so after the first message every later stage (-1, 0, +2, +5) was skipped as 'already delivered'. **B** patient = name + mobile by default (`REFILLCARE_PATIENT_IDENTITY=name_mobile`): 46% of sales lines were under names billed with 5+ phones; another person with the same name could cancel a patient's reminders and switch them to his phone. Engine `only_keys` accepts billing ids or patient keys; Customers Monitored counts patients (name + mobile, cached per data version). **C** a return-only upload re-scores and reverts the returned purchase like a rollback (earlier schedule resumes). **D** `REFILLCARE_RULE_QUANTITY_SCALING` (on): fewer strips than 2/3 of usual -> interval scaled down (within 7 days 37.4% -> 47.1% on 20,702 purchases; buying more is not scaled: 40.8% -> 29.4% if it were). **E** separate bills of the same medicine on the same day = one purchase (engine merges per patient + medicine + day, after patient keys). Tests: `tests/test_store_visit_scenarios.py`; real-data suite (18 scenarios) kept outside the repo. Live store re-scored | Done |
| 4.13 | **Return rule (11 Nov 2026)**: a return is subtracted from the sale it follows within `REFILLCARE_RETURN_WINDOW_DAYS` (90, was 14; reminder-medicine returns: median 10 d, 75% within 29 d, 90% within 56 d). A FULL return within `REFILLCARE_RETURN_CORRECTION_DAYS` (7) is a billing correction (earlier schedule resumes); later, the sale is marked `RETURNED_LATE:<date>`, that medicine's reminders pause (incl. +45) and the patient is listed in Patients Needing Review ('Returned all strips ... therapy stopped or changed?'); Add to reminders or a new purchase resumes them (`pharmacist_overrides.apply_return_stops`, applied in save / sync before pharmacist additions). Partial returns reduce the strips. No new tables. Original history re-netted with the rule (`scripts/rebuild_legacy_history.py`) and all patients re-scored | Done |
| 4.14 | **One current decision per patient-medicine (11 Nov 2026)**: after the name -> name + mobile switch, 298,654 old keys' decisions (e.g. `A SUDHEER KUMAR_1678` next to `A SUDHEER KUMAR_8555914289_1678`) stayed in `refill_decisions`; every 'latest decision per key' reader showed them: Patients Needing Review listed patients twice (229 rows / 187 patients instead of 126 / 105), Add to reminders could create a second reminder under the old key, Stop / Resume search showed people twice, 4,158 open outcomes would have become false NO_REFILL. `rekey.retire_stale_decisions` deletes decisions on keys that no longer exist or newer than the key's latest real purchase (cycles keep history, open outcomes -> SUPERSEDED, excluded from live accuracy); run by the identity re-key, after every patient re-score and by `scripts/rebuild_legacy_history.py --retire-stale-only` (live store: 319,218 removed). Stop / Resume search: one row per phone | Done |
| 4.15 | **One-click upload (12 Nov 2026)**: upload -> automatic file summary, check results and Data Preview (first 10 lines) -> ONE button 'Save & Update Reminders' (Streamlit + web UI) that validates, stores the lines, re-scores the patients and updates their reminders (`incremental_ingest.ingest_sales_batch`, which always did all of this). Removed the separate 'Validate File Hygiene' (only re-displayed the automatic check) and both 'Generate Updated Predictions' buttons (legacy in-memory engine, nothing saved). 'Duplicate Records' renamed 'Repeated Bill Lines' (kept and added up per bill) | Done |
| 4.16 | **Every uploaded line accounted for (12 Nov 2026)**: return lines (negative quantity) are saved as their own rows (`transaction_type` CUSTOMER_RETURN, never a purchase - every loader leaves them out) with their status: `RETURN_APPLIED:<sale id>:<qty>` or `RETURN_WAITING`. Received = saved sales + saved returns + invalid rows. The same return uploaded twice is no longer subtracted twice; a return uploaded before its sale is applied when the sale arrives; rolling back a return upload gives the strips back (and the sale's reminders); rolling back a sale puts its returns back to waiting; widening `REFILLCARE_RETURN_WINDOW_DAYS` + 'Re-check waiting returns' (Streamlit Upload tab, web UI, `POST /api/v1/sales/returns/recheck`) applies them without re-uploading. Waiting list: `GET /api/v1/sales/returns/waiting`. Rows not saved are listed with a plain reason (upload result in Streamlit / web UI with CSV download, `skipped_rows` in `POST /api/v1/sales/ingest`). One wording for the 'Sales lines stored' breakdown everywhere (`history_store.return_breakdown_text`, API `breakdown_summary/detail/audit`). Original history (24-12-2020 to 30-09-2026): its 6,047 return lines were saved from the master CSV with `scripts/rebuild_legacy_history.py --store-returns` WITHOUT changing any sale (netting replayed on the original quantities; 39 patient-medicines whose returns had been applied in an earlier sequence linked from the stored state; check: every sale's original - stored strips = strips its linked returns took off, 0 unexplained): 4,885 linked to their sale, 1,162 waiting. Only the 1-15 Oct 2026 upload (file not on disk) keeps its 53 returns as counts. No new tables | Done |
| 4.17 | **Tab icons (13 Nov 2026)**: the Streamlit tabs show Material icons (dashboard, upload file, calendar, layers, warning, chat; event-upcoming and insights for the hidden tabs) - the same symbols as the web UI sidebar. No API change. | Done |

## 5. Interfaces

| # | Item | Status |
|---|---|---|
| 5.1 | Streamlit operations app: 6 pharmacist tabs with Material icons (Pharmacy Overview, Upload Sales Data, Patient Reminders, Combined Prescriptions (Med-Sync), Patients Needing Review, WhatsApp Messages). Upcoming Refills and AI Accuracy & Settings hidden 8 Nov 2026 via `TAB_VISIBILITY` (ML lifecycle via API / scripts / worker) | Done |
| 5.2 | FastAPI + web app with the same features (parity rule: shared services, no inter-dependency) | Done |
| 5.3 | Write protection (`REFILLCARE_API_KEY`, local-only when unset) | Done |
| 5.4 | Cache cleared after every mutation in Streamlit | Done |

## 6. Repository hygiene (3 Nov 2026)

| # | Item | Status |
|---|---|---|
| 6.1 | Removed: prototype CLI scripts `run_train.py`, `run_prediction.py`, `run_reminder.py`, `run_message.py` (replaced by `scripts/ml_cli.py`, `scripts/retrain_production.py` and the apps); `scratch/`, `file_to_markdown_converter/`, `conversation_history/`, duplicate root `PLAN_BACKEND_DEVOPS.md` (kept in `docs/`). Archived locally to `Documents\refillcare_archive_20261103.zip` | Done |
| 6.2 | Retired model `hgb-20260617-20261016062117.joblib` and unused `refill_model_v2.0.0.pkl` deleted 4 Nov (registry row kept, `artifact_uri` cleared; rollback now skips versions without an artifact). Untracked: identity-audit reports (may contain patient identifiers) | Done |
| 6.3 | **Repo audit (4 Nov 2026)** — removed from git (history keeps them): legacy `model/`, `preprocessing/`, Path B benchmark (`refillcare/models/path_b_*`), one-off identity audit `experiments/`, dead modules `refillcare/features/pipeline.py`, `reminder/send_message.py`, `reminder/message_logs.py`; one-off scripts `audit_channel_impact`, `backtest_*`, `run_offline_validation`, `build_refillcare_notebooks`, outdated `scripts/README.md`; `notebooks/`; 48 phase reports / old plans in `docs/` (kept: deployment options, WhatsApp template, Postman collection); 7 September phase-report JSONs in `data/refillcare/processed/`. 25 tests of the removed modules went with them (suite: 860 passed, 1 skipped). Every removal checked with an import-reachability scan from the Streamlit, API, worker, campaign and script entry points | Done |
| 6.4 | `.gitignore`: all `*.joblib` / `*.pkl` / `*.parquet` / Office lock files / store DBs ignored; exceptions only for the 4 legacy models still loaded by code (`refill_model.joblib`, `human_ml_refill_model.joblib`, `chronic_refill_model.joblib`, `refill_model.pkl`); `.env.example`, `.agents/rules/` and `logs/.gitkeep` stay versioned | Done |
| 6.5 | Production model files are **not** in git (they come from the registry / S3 artifact store) — the backend team must provide them to deployed containers (future_implementation/GO_LIVE_CHECKLIST C1). Untrack the 4 legacy models once deploys fetch artifacts from S3 | Backend team |
| 6.6 | Patient data in git history (old commits) and the tracked sales file in `dummy/` — purge before the repository is shared outside the team (future_implementation/GO_LIVE_CHECKLIST A8) | ⏳ |
| 6.7 | **Storage cleanup (13 Nov 2026)**, each step checked before and after: 48 old Git LFS snapshots of `enterprise.db` / the master CSV removed with `git lfs prune` (13 GB; the repo no longer uses LFS); 5 master-CSV copies in `data/refillcare/backups/` and the September copy (`backup_20260921_120454`: database + parquets) removed after a line-by-line comparison - the 8 sales whose price / tax / batch columns existed only there were saved to `raw_backup/customer_data_fields_raw_extra_rows_from_old_backups.csv` (the sales themselves were already in the master CSV and database); 50 test store folders (`stores/t-*`, `demo-store`, `benchmark_store`), an empty root `enterprise.db`, September exports, an old log and all `__pycache__`. Kept on purpose: `processed/refillcare.db` (opened by the API), `raw_backup/` (only full 23-column copy), `stores/default` (registered store). Verified: 18 kept files byte-identical (checksums), database integrity ok with the same row counts, 907 tests passed, API serves the same numbers. Folder 16.8 → 3.3 GB | Done |
| 6.8 | **Documentation reorganised (13 Nov 2026)**: the two full-stack integration guides moved from the root to `docs/integration/`; README rewritten as the full setup / operations reference (new-machine setup, screens, upload and returns, prediction, reminders, model lifecycle, complete API list, configuration, data files, layout, scripts, tests, troubleshooting); `.env.example` completed (refill template, Supabase, gate settings); unused `frontend/favicon.png` removed | Done |

---

## 7. Decisions waiting for you

| Decision | Recommendation |
|---|---|
| Upload the 16-31 Oct 2026 file again (its batch was rolled back), then the November sales, then retrain (`.venv\Scripts\python scripts\retrain_production.py`) | Yes. The last retrain was 3 Nov (3.8); the next one learns the latest months |
| Turn on `REFILLCARE_RULE_PATH_B_MODE=blend` | Yes. Lower error every month, fixes the insulin gate failure |
| Turn on `REFILLCARE_ITEM_GROUPING=salt`, then run the identity cut-over (`scripts/rekey_reminders.py`, dry run first) | Yes. Brand / strength switches continue one cycle (+0.7 pp). Patient = name + mobile is already the default (4.12) |
| 'Re-check waiting returns' would apply 34 of the 1,162 waiting original-history returns (23 old sales, 2021-2025, become fully returned) | Optional; small effect on current reminders. Leave them, or press Re-check once after the October upload |

## 8. Multi-Store Enterprise Platform & Cloud Readiness (v2 executed)

The cloud-hosted Mediastra web application handles tenant / store logins and orchestrates RefillCare via `X-Store-Id`.

| # | Item | Status | Result |
|---|---|---|---|
| 8.1 | **Multi-store isolation on single AWS RDS instance**: each store gets schema `store_<id>` in PostgreSQL; SQLite uses isolated `enterprise_<id>.db` | Done | Tested in `tests/test_enterprise_platform.py` and `tests/test_postgres_tenancy.py` |
| 8.2 | **Incremental sales upload**: SHA-256 duplicate rejection, append new rows only, re-score only affected patients | Done | Measured on the real 827k-line history (`scripts/benchmark_incremental_upload.py`): 1-day file (486 rows) **7.4 s** incl. re-scoring 483 medicines; same day again **0.4 s** (0 added); back-dated day **5.2 s**. Old full-rewrite path on the same day: 121 s (23 s save + 98 s re-score) |
| 8.3 | **Data integrity & validation**: credit notes (`CN/`, `SR/`, `CR/`, etc.) and returns netted against sales; future (> tomorrow) & typo dates rejected | Done | Typos like 2027 never expire active reminders |
| 8.4 | **No backup copies**: eliminated 5 automatic CSV backups (~669 MB); RDS automated snapshots / SQLite online backup used instead | Done | Batch rollback deletes the batch's lines, reverts its reminders and re-scores only the medicines it touched: **8.1 s** for a 1-day batch (was ~10 min before scoping + faster visit features) |
| 8.5 | **Stale-data guard**: live WhatsApp sends blocked when sales data is stale (> max lag days) or uploads failed to sync | Done | Prevents reminding patients who already purchased; requires written override reason if bypassed |
| 8.6 | **Durable background jobs & worker**: `JobModel` table survives restarts, atomic job claiming (runs once across distributed workers) | Done | Monthly retrain, daily reminder expiry, failed-sync retries run in dedicated worker |
| 8.7 | **S3 model artifact store**: checksum verification, automated pruning keeping newest 3 models per store, one-click rollback | Done | Models decoupled from container filesystem |
| 8.8 | **API & Security hardening**: `X-API-Key` required on all `/api` routes (503 fail-closed in production); webhook signature check; IP rate limiting; phone numbers masked in all logs | Done | Covered by `tests/test_api_security.py` |
| 8.9 | **Database storage slimming**: dropped 4 duplicate/low-cardinality indexes (`customer_id`, `invoice_date`, `refillcare_eligible`, `transaction_type`), compacted `line_key` to 24 hex chars (96 bits) | Done | Store DB for 827k lines: 472 MB -> **366 MB**; migration v3 drops the indexes on existing databases. Line keys must not change again once a store has data (re-uploads would duplicate) |
| 8.10 | **Full test suite**: 907 passed, 1 skipped on 13 Nov 2026 (PostgreSQL test needs a server; was 885 on 3 Nov) | Done | Early stopping uses the newest 10% of events (`fit(X_val=...)`, scikit-learn >= 1.6) with a logged fallback for older versions |
| 8.11 | **Streamlit aligned with FastAPI**: incremental upload, shared rollback, data-status panel + sync retry, stale-data guard on live sends, model rollback; no login / store picker (Mediastra handles tenant > store logins) | Done | Works on `REFILLCARE_STORE_ID`; keep it on a private network |
| 8.12 | **Scoring speed-ups**: engine row loop reads plain arrays (pandas 3 Arrow strings were ~10 s per upload), rows loaded as column tuples, MIN/MAX date queries split, pure-Python medians in visit features | Done | 1-day upload 63 s -> 7.4 s |
| 8.13 | **Streamlit start-up**: prediction overview read from the stored engine decisions (one query, as FastAPI) instead of re-predicting every history on each load; history cache kept per view; parquet fast path only while it matches the database's legacy rows; one Supabase rules fetch shared by the classifier and the rules screen | Done | First load 251 s -> 35-47 s on the real 835k-line copy (overview 223 s -> 0.2 s); eligible list 0 -> 3,957 (the old on-the-fly path lacked the interval columns) |
| 8.14 | **Audit fixes (3 Nov)**: model-fallback warning (Streamlit banner + `/health` `degraded`); a retrain never reuses a model that cannot load; one-command retrain `scripts/retrain_production.py` (backup + gate); Supabase rules fetched once and loaded at API start-up (lock: no request classifies without rules); KPI totals from SQL; `scripts/db_maintenance.py` (WAL / VACUUM) | Done | `/reminders/daily` cold 12.5 -> 6.8 s, `/analytics/kpi` 18.4 -> 7.7 s, both 0.9 s warm. Go-live tasks for the store and the backend team: [future_implementation/GO_LIVE_CHECKLIST.md](future_implementation/GO_LIVE_CHECKLIST.md) |
| 8.15 | **Python environment guard**: the model must be loaded by the scikit-learn it was trained with (1.7.2, needs Python ≥ 3.10). The fallback warning now names the exact cause (trained vs running versions); Streamlit shows a red banner when started from another Python (e.g. a system 3.9); `run_streamlit.bat` / `run_api.bat` always use `.venv`; VS Code terminals activate `.venv` | Done | Root cause of the 3 Nov "ML model warning": Streamlit started from the system Python 3.9 |
| 8.16 | **UI deprecations**: `use_container_width` → `width="stretch"` in Streamlit and the campaign apps | Done | No Streamlit deprecation warnings |
| 8.17 | **Production review (8 Nov 2026)**: windows 30/30/90 kept (86% of refills fall within 90 days; a 60-day buffer would drop 21% as false lapses). Evaluations now use every qualifying event instead of a 3,000 sample (`ML_MAX_EVAL_EVENTS`, default all; gate CI about ±2.3 → ±1.7 pp). Scheduler bug fixed: a finished-but-not-promoted retrain was re-queued every hour; now once per store per month (3 retries on failure). Reminder schedule reviewed (avg 9 msgs / patient / 30 days, 4 near-identical DUE messages); **business decision: keep the 7 stages (−7, −3, −1, 0, +2, +5, +45), no monthly cap** | Done |
| 8.18 | **Accuracy review (8 Nov 2026, team of agents)**: fair benchmark on the same 44 features and windows - XGBoost (3 setups), neural networks (2 MLPs), RandomForest / ExtraTrees and blends do **not** beat production HGB (best +0.8 pp on one window; 3-window rolling re-test: pooled +0.1 to +0.4 pp, CI includes 0 - production kept); realistic ceiling about 60-63% within ±7 (oracle). Short-history rule: 8 alternatives tested, current days-of-supply rule kept. Implemented: **C1-C3** business metrics in every evaluation (within ±14, reminder before purchase, reminder 0-7 days before purchase, purchase kinds: lapse / early top-up / regular, groups: 12+ purchases, ML-served, short history) and a measured accuracy panel on the dashboard (typed-in fallback numbers removed); **C6** verified: all code matches phones on the last 10 digits; **C7** optional `daysSupply` upload field stored end to end; **C8** no-reminder control group `REFILLCARE_HOLDOUT_PCT` (default 0 = off since 11 Nov 2026; e.g. 10 to measure uplift) skipped by every live send path + `holdout_uplift` in the outcome summary | Done |

---

## 9. Next Steps

Store / operations (this team): set up the office laptop (README section 3), upload 16-31 Oct and November,
retrain, decide the switches (section 7), pilot sign-off and dry run (GO_LIVE_CHECKLIST part A).
RefillCare team (code for multi-tenant live): GO_LIVE_CHECKLIST part B = the flaws and work plan in
[future_implementation/PRODUCTION_READINESS_SCENARIO_1.md](future_implementation/PRODUCTION_READINESS_SCENARIO_1.md)
(tenant isolation, no local files / legacy models, async upload, per-tenant WhatsApp, IST time zone, job queue).
Backend team (v2 cloud): everything in GO_LIVE_CHECKLIST part C — RDS / Docker / S3, network and secrets,
multi-store onboarding and migration of `store-1`, daily data push, WhatsApp webhook, monitoring, DPDP.

| # | Item | Why |
|---|---|---|
| 9.1 | Backend team: `docker compose up -d` with PostgreSQL / RDS and run `tests/test_postgres_tenancy.py` against it | Validation of multi-schema Postgres migrations (the one skipped test) |
| 9.2 | Mediastra SSO / JWT token integration | Transition from shared `X-API-Key` to granular Mediastra user tokens |
| 9.3 | Real Mediastra patient ID linkage | Replaces derived name + mobile identity when Mediastra customer code is available |
| 9.4 | Dose capture at point of sale (e.g. 1-0-1) | Exact days of supply calculation for 1st-time / short histories |
