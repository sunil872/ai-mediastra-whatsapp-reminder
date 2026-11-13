# RefillCare: Refill Prediction & WhatsApp Reminders for Pharmacies

**MediAstra RefillCare** predicts when each chronic-medicine patient will run out of their medicine and sends a
WhatsApp refill reminder before that happens. It is the AI module behind the *Refill Reminder* feature of the
Mediastra pharmacy web application.

| | |
|---|---|
| **Store in production (v1)** | NGOs Colony, `store-1` (single store, local SQLite) |
| **Sales data loaded** | 24-12-2020 to 15-10-2026 (916,974 rows: 910,927 sales lines + 6,047 return lines) |
| **Serving model** | `hgb-20260518-20261103071906` (HistGradientBoosting, promoted 3 Nov 2026) |
| **Multi-store (v2)** | Through the Mediastra cloud application (NestJS / Angular, AWS); see [docs/integration/](docs/integration/) |
| **Status and history of every change** | [implementation_plan.md](implementation_plan.md) |

*README last updated: 13 Nov 2026.*

---

## Contents

1. [What the system does](#1-what-the-system-does)
2. [Architecture](#2-architecture)
3. [Setup on a new machine](#3-setup-on-a-new-machine)
4. [Running the applications](#4-running-the-applications)
5. [Pharmacist screens](#5-pharmacist-screens)
6. [Uploading sales data](#6-uploading-sales-data)
7. [How a refill date is predicted](#7-how-a-refill-date-is-predicted)
8. [Reminders and WhatsApp](#8-reminders-and-whatsapp)
9. [Model lifecycle](#9-model-lifecycle)
10. [REST API](#10-rest-api)
11. [Configuration (`.env`)](#11-configuration-env)
12. [Data files](#12-data-files)
13. [Repository layout](#13-repository-layout)
14. [Scripts](#14-scripts)
15. [Tests and quality gates](#15-tests-and-quality-gates)
16. [Data protection](#16-data-protection)
17. [Troubleshooting](#17-troubleshooting)
18. [Documentation index](#18-documentation-index)

---

## 1. What the system does

1. **Upload sales** (CSV / Excel, any column layout). Columns are mapped automatically. Only the 11 fields
   needed for reminders are stored. Every line is accounted for: sales and return lines are saved, and invalid
   rows are listed with a reason.
2. **Clean.** B2B transfers (`SB/…`), credit notes and acute / OTC items (painkillers, syrups, creams) are
   excluded from reminders. Lines of one bill and medicine are added up into one purchase, and returns are
   subtracted from the sale they belong to.
3. **Predict** each patient-medicine's next refill date with the ML model (long, stable histories) or days of
   supply (short histories). Unclear cases go to the pharmacist for review.
4. **Remind.** Each refill cycle has 7 WhatsApp stages (-7, -3, -1, 0, +2, +5, +45 days). A patient's medicines
   due on the same send day are bundled into one message (Med-Sync). A repurchase cancels the pending stages.
5. **Learn.** The train / validation / test windows move forward with the data every month. A new model is
   promoted only when it beats the current one on unseen data, overall and in every segment.

Who uses what:

| User | Surface |
|---|---|
| Pharmacist | Mediastra web app (v2), or today the Streamlit app / built-in web UI opened with `?store_id=` |
| Mediastra backend (NestJS) | RefillCare REST API (`X-API-Key`, `X-Store-Id`) |
| RefillCare team | Streamlit, scripts, `/docs` |

---

## 2. Architecture

```text
             Sales file (CSV / Excel)                    Mediastra web app (v2)
                       |                                          |
          +------------+-------------+                            |  X-API-Key + X-Store-Id
          |                          |                            v
   Streamlit app              Web UI (frontend/)  ---->  FastAPI  (api/main.py, api/ml_routes.py)
   (app_refillcare.py)        served by FastAPI                   |
          |                                                       |
          +----------------------+--------------------------------+
                                 |   same shared services (no logic duplicated per surface)
                                 v
     refillcare/  data (mapping, ingest, returns)  ->  engine (Path A / B, reminders, Med-Sync)
                  ml (registry, training, gate)     ->  whatsapp (client, consent, dispatcher)
                                 |
                                 v
     enterprise.db (SQLite; PostgreSQL schema per store in the cloud)  +  model files (local disk / S3)
                                 |
                  refillcare.jobs.worker: monthly retrain, daily stage expiry, sync retries
```

**Rules every change follows** ([.agents/rules/](.agents/rules/), [.claude/CLAUDE.md](.claude/CLAUDE.md)):
- **Surface parity.** Shared logic lives in `refillcare/`. Streamlit, the API (`/docs`) and the web UI call the
  same function, so they always show the same numbers and actions.
- **No login or store picker** in the pharmacist app. Mediastra handles logins; the store comes from `?store_id=`
  (Streamlit / web UI) or `X-Store-Id` (API).
- **Safety.** Tests never send WhatsApp messages or write to Supabase. Live sends need an explicit pharmacist
  confirmation.

---

## 3. Setup on a new machine

Requirements: **Windows 10/11 (or macOS / Linux), Python 3.13** (minimum 3.10, because scikit-learn 1.7.2 does
not install on older versions), Git, about 4 GB free disk and 8 GB RAM (16 GB for a retrain).

### 3.1 Get the code

```bash
git clone https://github.com/sunil872/ai-mediastra-whatsapp-reminder.git
cd ai-mediastra-whatsapp-reminder
```

(Or unzip a copy of the project folder. Do not copy `.venv/`; create it fresh as below.)

### 3.2 Create the Python environment

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate             # macOS / Linux: source .venv/bin/activate
pip install -r requirements-dev.txt   # app + pytest / httpx / pip-audit (production image: requirements.txt only)
```

> **Always run from `.venv`.** A model only loads in the scikit-learn version it was trained with (1.7.2). If
> the app is started from another Python (for example a system Python 3.9), Streamlit shows a red banner and
> predictions fall back to the baseline engine. `run_streamlit.bat` / `run_api.bat` always use `.venv`, and VS
> Code terminals activate it automatically ([.vscode/settings.json](.vscode/settings.json)).

### 3.3 Configuration

```bash
copy .env.example .env             # macOS / Linux: cp .env.example .env
```

Fill in your own keys (Xinno WhatsApp, Supabase, store name). When moving from another machine, copy the existing
`.env` instead. **Never commit `.env`.**

### 3.4 Data (not in Git: patient data)

Copy these from the old machine into the same paths (stop Streamlit and the API first):

| Path | Required | Notes |
|---|---|---|
| `data/refillcare/processed/enterprise.db` **and** its `enterprise.db-wal` / `enterprise.db-shm` files | **yes** | The master database. The `-wal` file holds recent writes; copy all three together, or first run `python scripts/db_maintenance.py` (WAL checkpoint) so the `.db` alone is complete |
| `data/refillcare/processed/models/hgb-20260518-20261103071906.joblib` | **yes** | Production model; not in Git |
| `data/refillcare/processed/refillcare.db` | yes | Legacy reminder storage still opened by the API (`reminder/storage.py`) |
| `data/refillcare/customer_data_fields.csv`, `processed/purchase_history.parquet`, `processed/clean_transactions.parquet` | yes | Original history (fast path for counts, rebuild scripts) |
| `data/refillcare/raw_backup/` | recommended | Only full 23-column copy of the original export (price, tax, batch) |
| `data/refillcare/processed/{train,validation,test,training_dataset}.parquet` | optional | Only for offline training experiments |
| `data/refillcare/stores/default/` | yes | Registered store folder (small) |

`data/refillcare/SALT ITEMS LIST.csv` and the 4 legacy model files are in Git.

### 3.5 Verify

```bash
.venv\Scripts\python -m pytest -q        # expected: 907 passed, 1 skipped
run_api.bat                              # then open http://127.0.0.1:8000/health and http://127.0.0.1:8000/
run_streamlit.bat                        # then open http://localhost:8501
```

`/health` must report `healthy` (not `degraded`) and name the production model. The Pharmacy Overview must show sales up
to 15-10-2026.

---

## 4. Running the applications

| Application | Command | URL |
|---|---|---|
| Streamlit pharmacist / operations app | `run_streamlit.bat` (or `.venv\Scripts\streamlit run app_refillcare.py`) | http://localhost:8501 (`?store_id=store-1`) |
| FastAPI + web UI | `run_api.bat` (or `.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`) | http://127.0.0.1:8000 · API docs `/docs` |
| Background worker | `.venv\Scripts\python -m refillcare.jobs.worker` | Monthly retrain, daily stage expiry, sync retries |
| Docker Compose (full stack) | `docker compose up -d` | API, Streamlit, worker and PostgreSQL containers |
| Standalone WhatsApp campaigns | `streamlit run whatsapp_campaigns/app.py` (text) · `whatsapp_campaigns/app_image_campaign.py` (image) | Ad-hoc broadcasts from a customer CSV; separate from RefillCare ([whatsapp_campaigns/README.md](whatsapp_campaigns/README.md)) |

Health probes: `/livez` (process up), `/readyz` (database reachable), `/health` (`healthy` / `degraded`, serving
model, data freshness).

---

## 5. Pharmacist screens

Streamlit and the web UI show the same tabs, numbers and actions (tab icons are the same Material symbols).

| Tab | What it is for |
|---|---|
| **Pharmacy Overview** | This store, this month: customers monitored, patients getting reminders, reminder messages, due today, patients needing review today; sales lines stored with the received / saved / not-saved breakdown; measured model accuracy |
| **Upload Sales Data** | Upload a file → automatic summary, checks and preview → one button **Save & Update Reminders**. Column mapping, upload history with rollback, rows not saved (with reason and CSV download), returns waiting for their sale with **Re-check waiting returns** |
| **Patient Reminders** | Reminders by day / rolling 30 days / month, with stage (DUE / FOLLOW / LAPSED) |
| **Combined Prescriptions (Med-Sync)** | One message per patient per send day for several medicines |
| **Patients Needing Review** | Patients the engine will not remind automatically (unstable history, full late return, not-classified medicine). **Add to reminders** (single or bulk, editable gap / due date) and **Remove** |
| **WhatsApp Messages** | **Send WhatsApp Messages (N)** for today with a confirmation dialog (to send / skipped: no valid mobile, opted out, already sent), Messages Sent log for one day, Stop / Resume messages by patient name or mobile |

Hidden tabs (`TAB_VISIBILITY` in `app_refillcare.py`): *Upcoming Refills* and *AI Accuracy & Settings*. The ML
lifecycle remains available through `/api/v1/ml/*`, `scripts/retrain_production.py` and the worker.

---

## 6. Uploading sales data

**Any store's file works.** Headers are matched to the RefillCare fields by name (about 200 aliases in
[config/column_aliases.json](config/column_aliases.json); add a store's names there, no code change needed) and by
what the values look like (dates, 10-digit mobiles, `S0/` invoices, `1X15` packs, numbers). Unclear headers get a
confirm screen with the best guess pre-selected. The confirmed mapping is saved per store and file layout.

**Only these 11 fields are stored**; price, tax, batch, expiry and cost columns are dropped at upload:

| Field | Why it is needed | Required |
|---|---|---|
| `invoice_date` | refill intervals | yes |
| `quantity` | days of supply (negative = return) | yes |
| `MOBILE_NO` | WhatsApp, opt-out, patient identity | yes |
| `customerName` (or a party code) | patient identity, greeting | yes |
| `itemName` (or `itemCode`) | chronic classification, message text | yes |
| `invoice_number` | customer sale (`S0/`) vs B2B (`SB/`) | expected |
| `itemCode` | catalogue lookup (salt, pack) | expected |
| `packing` | units per pack | expected (catalogue fallback) |
| `freeQuantity` | free strips extend days of supply | optional |
| `customerId`, `itemId` | patient / item keys | derived |

**What happens on Save & Update Reminders** (`refillcare/data/incremental_ingest.py`, same for Streamlit, web UI
and `POST /api/v1/sales/ingest`):

1. The same file twice is refused (SHA-256); only lines not stored yet are appended (a partial day followed by the
   full day adds only the missing lines).
2. **Returns** (negative quantity) are saved as their own rows (`CUSTOMER_RETURN`, never a purchase) and subtracted
   from the sale they follow within `REFILLCARE_RETURN_WINDOW_DAYS` (90). A full return within 7 days is a billing
   correction; later, the medicine's reminders pause and the patient appears in *Patients Needing Review*. A return
   uploaded before its sale waits and is applied when the sale arrives; the same return uploaded twice is never
   subtracted twice.
3. Only the patients in the file are re-scored and their reminder cycles updated.
4. **Every line is accounted for:** received = sales saved + returns saved + rows not saved (each with a reason).

**Rollback** of any batch removes its lines, gives returned strips back, reverts its reminders and re-scores only
the medicines it touched (about 8 s for a 1-day batch). Measured on the real history: 1-day upload about 7 s,
same day again 0.4 s (`python scripts/benchmark_incremental_upload.py`).

---

## 7. How a refill date is predicted

| Patient history | Method |
|---|---|
| **6 or more purchases, stable** (Path A) | Registry ML model (learns the deviation from the patient's own median) with safety caps (insulin pen / inhaler ceiling, 120-day maximum) |
| **3 to 5 purchases** (Path B) | Days of supply from units bought and the patient's own pace |
| Unstable, too little history, bulk or odd purchase, identity conflict | Pharmacist review; no automatic reminder |

Features are point-in-time and identical in training, backtest and serving: purchase intervals and stability,
quantity vs usual (partial / multi-pack / top-up), calendar (usual day of month), visits (other medicines, days
since last visit) and salt priors from the item catalogue.

**Patient = name + mobile** (`REFILLCARE_PATIENT_IDENTITY=name_mobile`, default): the same name with a different
phone is a different patient. **Bill lines are added up:** one bill + date + medicine = one purchase.
Fewer strips than 2/3 of usual shortens the interval (`REFILLCARE_RULE_QUANTITY_SCALING`, on).

**Accuracy** (share of refills predicted within ±7 days, on unseen data):

| Evaluation | Model | Baseline engine |
|---|---|---|
| Production model, test 18 Jun to 17 Jul 2026 | **56.7%**, MAE 10.4 d | 53.6%, MAE 11.2 d |
| 3 back-to-back test months (5,065 refills) | 58.7% | 55.6% |

XGBoost, neural networks, RandomForest and blends were benchmarked on the same features and did not beat the
production model (implementation_plan 8.18).

---

## 8. Reminders and WhatsApp

- **7 stages per refill cycle:** -7, -3, -1, 0 (DUE), +2, +5 (FOLLOW), +45 (LAPSED). Past-due stages expire; a
  repurchase supersedes pending stages.
- **Send rule:** only medicines the Medication Rules mark chronic, or that a pharmacist added
  (`persistence.medicine_can_be_sent`), are sent. The same rule drives lists, counts, API and sending.
- **Med-Sync:** one message per patient per send day and tier (about 25% fewer messages).
- **Safeguards on every send path:** pharmacist confirmation, STOP / UNSUBSCRIBE opt-out (last-10-digit match),
  valid mobile, no double sends (idempotent), stale-sales guard (live sends blocked when the latest sale is older
  than `REFILLCARE_MAX_DATA_LAG_DAYS`; an override needs a written reason).
- **Delivery log:** every attempt is stored in `whatsapp_delivery_logs` (audit, billing, outcome tracking).
- Template: `refillcare_medicine_reminder` ([docs/WHATSAPP_TEMPLATE_REFILLCARE.md](docs/WHATSAPP_TEMPLATE_REFILLCARE.md)).

---

## 9. Model lifecycle

- **Windows move with the data:** test = last 30 days before (last sale − 90 days), validation = the 30 days
  before that, train = everything earlier.
- **Promotion gate:** a candidate is promoted only if, on the untouched test window, it beats production and the
  baseline overall and does not regress in any segment (therapy such as insulin, history depth) on within ±7 days
  and MAE, and does not raise early or late reminders.
- **Registry:** every version is stored in `enterprise.db` (production / shadow / candidate / retired) with its
  train window, metrics, evaluation reports and library versions.
- **Fail-safe serving:** if the production model cannot load, predictions fall back to the baseline engine;
  Streamlit shows a red warning with the cause and `/health` reports `degraded`.

```bash
.venv\Scripts\python scripts\retrain_production.py   # DB backup → retrain → gate → promote (needs about 4 GB free RAM)
python scripts/ml_cli.py plan                        # current windows; is a retrain due?
python scripts/ml_cli.py retrain [--promote]         # train → select on validation → gate on test
python scripts/ml_cli.py rules-ab                    # which prediction rules help (read-only)
python scripts/ml_cli.py rolling --folds 3           # is the gain real across several months?
python scripts/ml_cli.py list | gate --version <id> | promote --version <id>
```

---

## 10. REST API

Interactive reference: `http://127.0.0.1:8000/docs` (OpenAPI at `/openapi.json`). Every `/api` route needs
`X-API-Key` = `REFILLCARE_API_KEY` (when unset in development, only the local machine is accepted; in production
the API refuses to serve) and the store via `X-Store-Id` or `?store_id=`.

| Area | Endpoints |
|---|---|
| Overview & analytics | `GET /api/v1/overview` · `GET /api/v1/analytics/kpi` · `GET /api/v1/analytics/transaction-types` (received / saved / not-saved breakdown) · `GET /api/v1/analytics/consensus-regimen` |
| Sales upload | `POST /api/v1/sales/column-mapping/detect` · `POST /api/v1/sales/column-mapping` · `GET/DELETE /api/v1/sales/column-mapping/profiles` · `POST /api/v1/sales/preview` · `POST /api/v1/sales/ingest` (returns `skipped_rows`) · `GET /api/v1/sales/data-status` · `GET /api/v1/sales/batches` · `POST /api/v1/sales/batches/{id}/rollback` · `POST /api/v1/sales/batches/{id}/retry-sync` |
| Returns | `GET /api/v1/sales/returns/waiting` · `POST /api/v1/sales/returns/recheck` |
| Reminders | `GET /api/v1/reminders/daily` · `GET /api/v1/reminders/months` · `GET /api/v1/reminders/export-csv` · `GET /api/v1/reminders/export-json` · `GET /api/v2/med-sync/bundles` · `GET /api/reminders/today` · `POST /api/reminders/{id}/approve\|reject\|dispatch` |
| Patients needing review | `GET /api/v1/review/queue\|summary` · `GET /api/v1/reminders/review-candidates` · `GET /api/v1/reminders/review-other` · `POST /api/v1/reminders/pharmacist-add` · `POST /api/v1/reminders/pharmacist-add-bulk` · `POST /api/v1/reminders/pharmacist-remove` · `GET /api/v1/reminders/pharmacist-added` |
| WhatsApp | `GET /api/v1/whatsapp/send-summary` · `POST /api/v1/whatsapp/send-today` · `GET /api/v1/whatsapp/candidates` · `POST /api/v1/whatsapp/preview\|send-single\|dispatch` · `GET /api/v1/whatsapp/delivery-log` · `POST /api/v1/whatsapp/webhook` (STOP / START) |
| Patients & consent | `GET /api/customers/search` · `POST /api/customers/whatsapp-messages` (stop / resume) · `GET /api/customers/opt-outs` · `POST /api/customers/{id}/opt-out\|opt-in` · `GET /api/customers/{id}/refill-history` · `GET /api/refill-decisions[/{id}]` |
| Medication rules | `GET/POST /api/v1/medications/rules` · `DELETE /api/v1/medications/rules/{id}` · `POST /api/v1/medications/classify` |
| ML lifecycle | `GET /api/v1/ml/plan\|models\|evaluations\|settings\|outcomes/summary` · `POST /api/v1/ml/retrain\|train\|evaluations\|rules-ab\|rolling\|models/rollback\|outcomes/refresh` · `GET /api/v1/ml/models/{id}/gate` · `POST /api/v1/ml/models/{id}/promote\|status` · `GET /api/v1/ml/reports/{kind}` · `GET/POST /api/v1/ml/identity/rekey` |
| Stores & jobs | `GET/POST /api/v1/stores` · `GET /api/v1/stores/current` · `GET /api/v1/jobs[/{id}]` |
| Health | `GET /livez` · `GET /readyz` · `GET /health` |

The webhook has no API key (the provider must reach it); it is verified with `WHATSAPP_WEBHOOK_SECRET` (signature
or token, required when `REFILLCARE_ENV=production`). The full NestJS / Angular integration contract is in
[docs/integration/FULLSTACK_INTEGRATION_HANDOVER.md](docs/integration/FULLSTACK_INTEGRATION_HANDOVER.md).

---

## 11. Configuration (`.env`)

[.env.example](.env.example) documents every setting with its default. The main groups:

| Variables | Purpose |
|---|---|
| `XINNO_API_URL`, `XINNO_API_KEY`, `XINNO_WABA_NUMBER` | WhatsApp provider |
| `WHATSAPP_REFILL_TEMPLATE_NAME`, `WHATSAPP_TEMPLATE_*`, `XINNO_IMAGE_*`, `CLOUDINARY_*` | Approved templates (RefillCare and campaigns), image hosting |
| `MEDICAL_STORE_NAME`, `MEDICAL_STORE_CONTACT` | Store name / phone in messages |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` | Medication rules (chronic / acute); local cache when unavailable |
| `REFILLCARE_STORE_ID`, `REFILLCARE_ALLOWED_STORES`, `REFILLCARE_DATA_DIR`, `REFILLCARE_SALT_CATALOG` | Default store, stores the app may open, data folder, item catalogue |
| `REFILLCARE_API_KEY`, `CORS_ALLOW_ORIGINS`, `WHATSAPP_WEBHOOK_SECRET`, `REFILLCARE_RATE_LIMIT_PER_MIN` | API protection |
| `REFILLCARE_ENV`, `DATABASE_URL`, `REFILLCARE_MODEL_BUCKET` | Production mode (fails closed, JSON logs, no `/docs`), PostgreSQL / RDS, S3 models |
| `REFILLCARE_LIVE_REQUIRES_APPROVAL`, `REFILLCARE_STALE_DATA_GUARD`, `REFILLCARE_MAX_DATA_LAG_DAYS`, `REFILLCARE_HOLDOUT_PCT` | Send safety, control group |
| `REFILLCARE_RETURN_WINDOW_DAYS`, `REFILLCARE_RETURN_CORRECTION_DAYS` | Return handling |
| `REFILLCARE_RULE_*`, `REFILLCARE_PATIENT_IDENTITY`, `REFILLCARE_ITEM_GROUPING` | Prediction rules and keys (after changing identity / grouping run `python scripts/rekey_reminders.py --apply`) |
| `ML_*` | Window sizes, gate tolerances, training target, auto-retrain |

---

## 12. Data files

All under `data/refillcare/` and git-ignored (they contain patient data), except the item catalogue and 4 legacy
model files.

| Path | Content |
|---|---|
| `processed/enterprise.db` (+ `-wal`, `-shm`) | **The master:** sales and return lines, upload batches, engine decisions, reminder cycles / stages, pharmacist additions, opt-outs, delivery logs, model registry |
| `processed/models/` | Model files. Production `hgb-20260518-20261103071906.joblib` (not in Git); legacy `human_ml_refill_model.joblib` (fallback), `refill_model.*`, `chronic_refill_model.joblib` |
| `processed/refillcare.db` | Legacy reminder storage still opened by the API |
| `customer_data_fields.csv` | Original history (11 columns), source for `scripts/rebuild_legacy_history.py` |
| `processed/purchase_history.parquet`, `processed/clean_transactions.parquet` | Legacy history fast path (used while it matches the database) |
| `processed/{train,validation,test,training_dataset}.parquet` | Offline training datasets (optional) |
| `raw_backup/` | Full 23-column original export, plus `customer_data_fields_raw_extra_rows_from_old_backups.csv` (8 rows recovered from old backups) |
| `SALT ITEMS LIST.csv` | Item master: Code, Item Name, PACK, SALT, CATEGORY, ITEMCAT (36,963 items) |
| `stores/<id>/` | Data folder of each additional registered store (`default`) |
| `processed/enterprise_backup_*.db` | Online backups made by `retrain_production.py` / `db_maintenance.py --vacuum`; delete once verified |

---

## 13. Repository layout

```text
app_refillcare.py            Streamlit pharmacist / operations app
api/                         FastAPI: main.py (routes, web UI hosting), ml_routes.py, services.py, schemas.py, security.py
frontend/                    Web UI served by FastAPI: index.html, js/app.js, js/api.js, css/styles.css, assets/
refillcare/
  data/                      schema (11 fields), column_mapping, incremental_ingest (uploads, rollback, returns),
                             history_store (database history, returns, channel counts), identity, enrichment (catalogue),
                             medication_classifier, cleaning, sales_ingestion (legacy CSV path)
  engine/                    unified_engine (Path A / B), prediction_rules, persistence (cycles & stages),
                             reminder_lifecycle, med_sync, review_lists, overview_kpis, pharmacist_overrides, rekey, post_ingest
  features/                  point_in_time features, context (visits, salt priors)
  ml/                        training, evaluation, promotion (gate), registry, service, model_provider, artifact_store (local / S3)
  models/                    supply_hybrid (consensus & safety caps), path_a_classifier, legacy training / prediction
  jobs/                      durable job queue and worker
  whatsapp/                  client, dispatcher, consent (opt-out), template formatter
  tenancy.py, stores.py, store.py   store id → database / schema / data folder
reminder/                    scheduler, dispatch, storage (legacy, still used by ingestion and the API)
database/                    SQLAlchemy models, control models (stores), connection, additive migrations
services/, utils/            Xinno / Cloudinary clients, phone validation, bulk-send helpers
config/                      config.py, column_aliases.json
whatsapp_campaigns/          standalone text / image broadcast apps (independent of RefillCare)
scripts/                     operations and maintenance scripts (section 14)
tests/                       pytest suite
assets/                      brand logos / favicon (Streamlit, API fallback)
docs/                        deployment options, WhatsApp template, Postman collection
docs/integration/            Mediastra full-stack integration (NestJS / Angular handover and the platform guide)
future_implementation/       go-live checklist, multi-tenant AWS plan, production readiness, ML / market audit
data/                        local data (git-ignored, section 12)
.agents/rules/, .claude/     engineering rules for every change (surface parity, no push, audits)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, .devcontainer/   containers, CI, dev container
run_streamlit.bat, run_api.bat                                               Windows launchers (always .venv)
```

---

## 14. Scripts

Run from the project root with `.venv\Scripts\python scripts\<name>.py`. Scripts that change data have a dry run
and need an explicit flag to write.

| Script | Purpose |
|---|---|
| `retrain_production.py` | One-command production retrain: RAM check, DB backup, retrain, gate, promote (`--no-promote` to measure only) |
| `ml_cli.py` | Model lifecycle: plan, train, evaluate, retrain, gate, promote, rules-ab, rolling, outcomes |
| `db_maintenance.py` | SQLite WAL checkpoint; `--vacuum` compacts after a backup |
| `rebuild_legacy_history.py` | Rebuild / re-net the original history from `customer_data_fields.csv`; `--retire-stale-only`; `--store-returns` (save original return lines without changing sales). Needs `--i-have-a-backup` to write |
| `import_history_to_db.py` | Onboard a store: load an existing history (CSV / Excel / parquet) into its database |
| `rekey_reminders.py` | Identity / salt-grouping cut-over (dry run by default) |
| `purge_and_sync_reminders.py` | Re-apply the medication classifier (supersedes reminders for non-chronic items) |
| `sync_enterprise_db_chronic.py` | Mark non-chronic items (syrups, drops, topicals, OTC) not eligible in decisions, cycles and stages of `enterprise.db` |
| `benchmark_incremental_upload.py` | Reproducible timing of upload, re-upload, back-dated day and rollback (temporary database) |
| `run_sales_pipeline.py` | Ingest a sales file and regenerate predictions from the command line (legacy path) |
| `slim_sales_data.py` | One-time: keep only the 11 fields (raw copy to `raw_backup/` first) |
| `train_chronic_specialized_model.py`, `train_human_ml_model.py` | Offline benchmark / legacy model training |
| `seed_supabase_rules.py` | Seed medication rules into Supabase (writes to Supabase; run deliberately) |
| `verify_system_health.py`, `verify_column_aliases.py`, `verify_image_bulk_dry_run.py` | Checks |
| `execute_single_live_test.py`, `execute_bulk_live_test.py` | **Real WhatsApp sends** for controlled live tests; never run in development |

---

## 15. Tests and quality gates

```bash
.venv\Scripts\python -m pytest -q                       # full suite: 907 passed, 1 skipped (PostgreSQL test needs a server)
.venv\Scripts\python -m pytest tests/test_store_visit_scenarios.py -q   # returns, rollback, store-visit scenarios
.venv\Scripts\python -m pytest tests/api -q             # API contract
.venv\Scripts\python -m pytest tests/test_api_security.py -q            # API keys, rate limits, webhook signatures
```

Tests use temporary folders and isolated databases; they never touch live data and never send WhatsApp messages.
They create temporary `data/refillcare/stores/t-*` folders, which can be deleted afterwards.

**Definition of done for every change** ([.claude/CLAUDE.md](.claude/CLAUDE.md)): shared logic → Streamlit →
FastAPI (`/docs`) → web UI → docs updated in the same change; full `pytest -q` green; Streamlit AppTest with
0 exceptions; uvicorn checked on `/openapi.json`, `/` and the changed routes.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push / pull request: the full suite, the
PostgreSQL schema-per-store test against a Postgres 16 service, `pip-audit`, and a Docker image build. The test
steps are currently non-blocking (`continue-on-error: true`), so check the test step's result in the CI log.

---

## 16. Data protection

- Patient data (sales files, database, parquets, backups, exports, logs) and `.env` are git-ignored; never commit
  them. Only the item catalogue and 4 legacy model files are versioned.
- Phone numbers are masked in every log line and in identity-conflict messages.
- Older commits contain patient data, and `dummy/` holds a real sales file: purge the history before the
  repository leaves the team (future_implementation/GO_LIVE_CHECKLIST A8).
- Production: `REFILLCARE_ENV=production` refuses to serve without an API key and webhook secret.

---

## 17. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Red "ML model" banner, `/health` = `degraded` | App started outside `.venv` or the model file is missing. Use `run_streamlit.bat` / `run_api.bat`; check `data/refillcare/processed/models/hgb-20260518-20261103071906.joblib` |
| Numbers older than expected after copying to a new machine | `enterprise.db-wal` was not copied. Copy it with the `.db`, or run `scripts/db_maintenance.py` on the old machine first |
| `database is locked` | Another Streamlit / API / script holds a write; wait or stop it. Do not copy the database while apps run |
| Live send blocked: stale data | Latest sale older than `REFILLCARE_MAX_DATA_LAG_DAYS`; upload the missing sales (or override with a reason) |
| Medication rules show "local cache" | Supabase unreachable or keys missing; reminders still use built-in rules, retried every `SUPABASE_RULES_RETRY_SECONDS` |
| `MemoryError` in a retrain | Needs about 4 GB free RAM; close other apps and run `retrain_production.py` again |
| Port 8000 / 8501 in use | Stop the old server (`Get-NetTCPConnection -LocalPort 8000`) before starting a new one |

---

## 18. Documentation index

| Document | For |
|---|---|
| [implementation_plan.md](implementation_plan.md) | Everything built, measured results, open decisions, next steps |
| [future_implementation/GO_LIVE_CHECKLIST.md](future_implementation/GO_LIVE_CHECKLIST.md) | What is still needed for live mode (A store / ops, B RefillCare code, C backend team) |
| [future_implementation/PLAN_BACKEND_DEVOPS_TENANCY.md](future_implementation/PLAN_BACKEND_DEVOPS_TENANCY.md) | Multi-tenant architecture and AWS deployment |
| [future_implementation/PRODUCTION_READINESS_SCENARIO_1.md](future_implementation/PRODUCTION_READINESS_SCENARIO_1.md) | Every code change from one local store to multi-tenant AWS |
| [future_implementation/PROJECT_AUDIT_ML_AND_MARKET.md](future_implementation/PROJECT_AUDIT_ML_AND_MARKET.md) | Model accuracy review, deep-learning decision, competitor comparison |
| [docs/integration/FULLSTACK_INTEGRATION_HANDOVER.md](docs/integration/FULLSTACK_INTEGRATION_HANDOVER.md) | API contracts and integration for the NestJS / Angular team |
| [docs/integration/MEDIASTRA_FULLSTACK_GUIDE_FOR_REFILLCARE.md](docs/integration/MEDIASTRA_FULLSTACK_GUIDE_FOR_REFILLCARE.md) | The Mediastra platform (from the full-stack team) and where RefillCare fits |
| [docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md](docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md) | Client-facing deployment options |
| [docs/WHATSAPP_TEMPLATE_REFILLCARE.md](docs/WHATSAPP_TEMPLATE_REFILLCARE.md) | Approved WhatsApp template |
| [whatsapp_campaigns/README.md](whatsapp_campaigns/README.md) | Standalone campaign apps |
