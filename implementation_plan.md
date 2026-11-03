# RefillCare — Implementation Plan (executed to date, and what is next)

Single store (NGOs Colony, `store-1`, sales Dec 2020 – 15 Oct 2026) in v1; multi-store through the cloud-hosted
Mediastra web application in v2 (built by the backend team).
How to run and use the system: [README.md](README.md). What is still needed for live mode (store + backend team):
[GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md). Architecture for the backend team:
[docs/PLAN_BACKEND_DEVOPS_TENANCY.md](docs/PLAN_BACKEND_DEVOPS_TENANCY.md). Client-facing deployment options:
[docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md](docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md).

*Last updated: 3 Nov 2026.*

Status legend: [Done] Completed & Verified · [Configured] Built & Parameterized · ⏳ next

---

## 1. Data and ingestion

| # | Item | Status | Result |
|---|---|---|---|
| 1.1 | Sales upload (CSV / Excel), date parsing (day-first, ambiguity blocked), de-duplication, append to master with pre-upload backup, batch rollback | Done | Same code path for Streamlit and FastAPI |
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
| 4.3 | Pharmacist approval before live sends; dry run by default; idempotent (no double sends) | Done |
| 4.4 | Opt-out: STOP / UNSUBSCRIBE via webhook, exact START to opt in, pharmacist opt-out / opt-in; every send path skips opted-out phones (last-10-digit match) | Done |
| 4.5 | Standalone text / image campaign apps (`whatsapp_campaigns/`) | Done |

## 5. Interfaces

| # | Item | Status |
|---|---|---|
| 5.1 | Streamlit operations app — 8 tabs incl. column mapping, saved mappings, ML settings, rule A/B, 3-month test, identity cut-over | Done |
| 5.2 | FastAPI + web app with the same features (parity rule: shared services, no inter-dependency) | Done |
| 5.3 | Write protection (`REFILLCARE_API_KEY`, local-only when unset) | Done |
| 5.4 | Cache cleared after every mutation in Streamlit | Done |

## 6. Repository hygiene (3 Nov 2026)

| # | Item | Status |
|---|---|---|
| 6.1 | Removed: prototype CLI scripts `run_train.py`, `run_prediction.py`, `run_reminder.py`, `run_message.py` (replaced by `scripts/ml_cli.py`, `scripts/retrain_production.py` and the apps); `scratch/`, `file_to_markdown_converter/`, `conversation_history/`, duplicate root `PLAN_BACKEND_DEVOPS.md` (kept in `docs/`). Archived locally to `Documents\refillcare_archive_20261103.zip` | Done |
| 6.2 | Untracked (kept on disk): retired model `hgb-20260617-20261016062117.joblib`, unused `refill_model_v2.0.0.pkl`, identity-audit reports (may contain patient identifiers) | Done |
| 6.3 | Kept on purpose: `model/` + `preprocessing/` (legacy Path B benchmark, tested), `experiments/identity_audit/*.py` (imported by `tests/test_customer_item_identity_audit.py`), `docs/` phase reports and `notebooks/` (project history) | Done |
| 6.4 | `.gitignore`: all `*.joblib` / `*.pkl` / `*.parquet` / Office lock files / store DBs ignored; exceptions only for the 4 legacy models still loaded by code (`refill_model.joblib`, `human_ml_refill_model.joblib`, `chronic_refill_model.joblib`, `refill_model.pkl`); `.env.example`, `.agents/rules/` and `logs/.gitkeep` stay versioned | Done |
| 6.5 | Production model files are **not** in git (they come from the registry / S3 artifact store) — the backend team must provide them to deployed containers (GO_LIVE_CHECKLIST B1). Untrack the 4 legacy models once deploys fetch artifacts from S3 | Backend team |
| 6.6 | Patient data in git history (old commits) — purge before the repository is shared outside the team (GO_LIVE_CHECKLIST A9) | ⏳ |

---

## 7. Decisions waiting for you

| Decision | Recommendation |
|---|---|
| Turn on `REFILLCARE_RULE_PATH_B_MODE=blend` | Yes — lower error every month, fixes the insulin gate failure |
| Turn on `REFILLCARE_PATIENT_IDENTITY=name_mobile` and `REFILLCARE_ITEM_GROUPING=salt`, then run the identity cut-over | Yes — correct patient per phone; dry run first |
| Upload the missing sales 16 Oct – today, then retrain again (`.venv\Scripts\python scripts\retrain_production.py`) | Yes — first retrain done 3 Nov (3.8); the next one learns the latest months and, after the switches above, measures the final setup |

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
| 8.10 | **Full test suite**: 885 passed, 1 skipped (PostgreSQL test needs a server) | Done | Early stopping uses the newest 10% of events (`fit(X_val=...)`, scikit-learn >= 1.6) with a logged fallback for older versions |
| 8.11 | **Streamlit aligned with FastAPI**: incremental upload, shared rollback, data-status panel + sync retry, stale-data guard on live sends, model rollback; no login / store picker (Mediastra handles tenant > store logins) | Done | Works on `REFILLCARE_STORE_ID`; keep it on a private network |
| 8.12 | **Scoring speed-ups**: engine row loop reads plain arrays (pandas 3 Arrow strings were ~10 s per upload), rows loaded as column tuples, MIN/MAX date queries split, pure-Python medians in visit features | Done | 1-day upload 63 s -> 7.4 s |
| 8.13 | **Streamlit start-up**: prediction overview read from the stored engine decisions (one query, as FastAPI) instead of re-predicting every history on each load; history cache kept per view; parquet fast path only while it matches the database's legacy rows; one Supabase rules fetch shared by the classifier and the rules screen | Done | First load 251 s -> 35-47 s on the real 835k-line copy (overview 223 s -> 0.2 s); eligible list 0 -> 3,957 (the old on-the-fly path lacked the interval columns) |
| 8.14 | **Audit fixes (3 Nov)**: model-fallback warning (Streamlit banner + `/health` `degraded`); a retrain never reuses a model that cannot load; one-command retrain `scripts/retrain_production.py` (backup + gate); Supabase rules fetched once and loaded at API start-up (lock: no request classifies without rules); KPI totals from SQL; `scripts/db_maintenance.py` (WAL / VACUUM) | Done | `/reminders/daily` cold 12.5 -> 6.8 s, `/analytics/kpi` 18.4 -> 7.7 s, both 0.9 s warm. Go-live tasks for the store and the backend team: [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md) |
| 8.15 | **Python environment guard**: the model must be loaded by the scikit-learn it was trained with (1.7.2, needs Python ≥ 3.10). The fallback warning now names the exact cause (trained vs running versions); Streamlit shows a red banner when started from another Python (e.g. a system 3.9); `run_streamlit.bat` / `run_api.bat` always use `.venv`; VS Code terminals activate `.venv` | Done | Root cause of the 3 Nov "ML model warning": Streamlit started from the system Python 3.9 |
| 8.16 | **UI deprecations**: `use_container_width` → `width="stretch"` in Streamlit and the campaign apps | Done | No Streamlit deprecation warnings |

---

## 9. Next Steps

Store / operations (this team): upload 16 Oct – today, retrain, decide the switches (section 7), pilot sign-off
and dry run — GO_LIVE_CHECKLIST part A.
Backend team (v2 cloud): everything in GO_LIVE_CHECKLIST part B — RDS / Docker / S3, network and secrets,
multi-store onboarding and migration of `store-1`, daily data push, WhatsApp webhook, monitoring, DPDP.

| # | Item | Why |
|---|---|---|
| 9.1 | Backend team: `docker compose up -d` with PostgreSQL / RDS and run `tests/test_postgres_tenancy.py` against it | Validation of multi-schema Postgres migrations (the one skipped test) |
| 9.2 | Mediastra SSO / JWT token integration | Transition from shared `X-API-Key` to granular Mediastra user tokens |
| 9.3 | Real Mediastra patient ID linkage | Replaces derived name + mobile identity when Mediastra customer code is available |
| 9.4 | Dose capture at point of sale (e.g. 1-0-1) | Exact days of supply calculation for 1st-time / short histories |
