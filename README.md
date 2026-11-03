# RefillCare — Refill Prediction & WhatsApp Reminders for Pharmacies

RefillCare predicts when each chronic-medicine patient will run out, and sends a WhatsApp refill reminder
before they do. v1 runs for one store (NGOs Colony, `store-1`; sales data Dec 2020 – 15 Oct 2026). Multi-store (v2)
comes through the Mediastra web application, built by the backend team.

| Document | For |
|---|---|
| [implementation_plan.md](implementation_plan.md) | what is built, measured results, next steps |
| [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md) | everything still needed for live mode — part A store / operations, part B backend team |
| [docs/PLAN_BACKEND_DEVOPS_TENANCY.md](docs/PLAN_BACKEND_DEVOPS_TENANCY.md) | multi-tenant architecture and AWS deployment (backend team) |

What it does, end to end:

1. **Upload sales** in any column layout (CSV / Excel). Columns are mapped automatically to the RefillCare
   fields; only the 11 fields needed for reminders are stored.
2. **Clean** — B2B transfers (`SB/…`) and acute / OTC items (painkillers, syrups, creams) are excluded.
3. **Predict** each patient-medicine's next refill date (ML model or personal history, see below).
4. **Remind** — 7 WhatsApp stages per refill cycle (-7, -3, -1, 0, +2, +5, +45 days), several medicines of one
   patient bundled into one message (Med-Sync); a repurchase cancels pending reminders.
5. **Learn** — every month the train / validation / test windows move forward; a new model is promoted only
   when it beats the current one on unseen data.

---

## 1. Quick start

**Python 3.13** (minimum 3.10 — scikit-learn 1.7.2 does not install on older versions).

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # tests / linting: pip install -r requirements-dev.txt
copy .env.example .env            # fill in your own keys (never commit .env)
```

> **Always run from `.venv`.** The ML model only loads in the scikit-learn it was trained with. If Streamlit is
> started from another Python (e.g. a system Python 3.9 on PATH), the app shows a red banner and falls back to the
> baseline engine. Use `run_streamlit.bat` / `run_api.bat` (Windows) or activate `.venv` first; VS Code terminals
> activate it automatically.

| Interface | Command | Open |
|---|---|---|
| Streamlit operations app | `run_streamlit.bat` (or `.venv\Scripts\streamlit run app_refillcare.py`) | http://localhost:8501 |
| FastAPI + web app | `run_api.bat` (or `.venv\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8000`) | http://127.0.0.1:8000 (API docs: `/docs`) |
| Background job worker | `python -m refillcare.jobs.worker` | Runs scheduled retrains, reminder expiry, sync retries |
| Docker Compose (Full Stack) | `docker compose up -d` | API, Streamlit, Worker & PostgreSQL container |
| Standalone WhatsApp campaigns | `streamlit run whatsapp_campaigns/app.py` (text) · `whatsapp_campaigns/app_image_campaign.py` (image) | ad-hoc broadcasts from a customer CSV, dry-run by default |

Streamlit and the web app use the same services, so every number and action is the same in both.

| Streamlit tab | Web app tab | What it is for |
|---|---|---|
| Operations Dashboard | Operations Dashboard | KPIs, customer sales vs B2B, data freshness |
| Data Ingestion & Sync | Data Ingestion & Sync | Upload sales, column mapping, saved mappings, batch rollback |
| Forecast Analytics | Forecast Analytics | Eligible / review patients and predicted refill dates |
| Clinical Dispatch Queue | Clinical Dispatch Queue | Reminders by date, rolling 30 days or month |
| Med-Sync Consolidation | Med-Sync Consolidation | One message per patient for several medicines |
| Exception & Review Queue | Exception & Review Queue | Patients the engine will not remind automatically |
| WhatsApp Gateway | WhatsApp Gateway | Preview, test send, batch send, opt-outs |
| Model Governance | Model Governance | Train/val/test windows, registry, gate, retrain, prediction settings, rule A/B, 3-month test, identity cut-over |
| – | Medication Classifier | Chronic / acute rules (Supabase, cached locally) |

---

## 2. Uploading sales data

**Any store's file works.** Headers are matched to the RefillCare fields by name (≈200 aliases in
[config/column_aliases.json](config/column_aliases.json) — add a store's names there, no code change) **and** by
what the values look like (dates, 10-digit mobiles, `S0/` invoices, `1X15` packs, numbers).

- Known headers → mapped automatically. Unclear headers → a confirm screen with the best guess pre-selected.
- The confirmed mapping is saved **per store and file layout**; next month's file with the same columns is
  mapped instantly. Saved mappings can be listed and deleted in the upload tab.
- **Only these 11 fields are stored** — every other column (price, tax, batch, expiry, cost) is dropped at upload:

| Field | Why it is needed | Required |
|---|---|---|
| `invoice_date` | refill intervals | yes |
| `quantity` | days of supply | yes |
| `MOBILE_NO` | WhatsApp, opt-out, patient identity | yes |
| `customerName` (or a party code) | patient identity, greeting | yes |
| `itemName` (or `itemCode`) | chronic classification, message text | yes |
| `invoice_number` | customer sale (S0/) vs B2B (SB/) | expected — without it every row counts as a customer sale |
| `itemCode` | catalogue lookup (salt, pack) | expected — item name used if absent |
| `packing` | units per pack | expected — taken from the catalogue if absent |
| `freeQuantity` | free strips extend days of supply | optional |
| `customerId`, `itemId` | patient / item keys | derived (party code or name; item code) |

Each upload appends only the lines not stored yet to the store's database (same file twice is refused;
a partial day followed by the full day adds only the missing lines), re-scores only the medicines in the
file, syncs reminder cycles and re-plans the ML windows. Any batch can be rolled back. Measured on the real
827k-line history: a 1-day upload takes ~7 s, the same day again 0.4 s, a rollback ~8 s
(`python scripts/benchmark_incremental_upload.py`).

**Files** (all git-ignored — they contain patient data):

| Path | Content |
|---|---|
| `data/refillcare/processed/enterprise.db` | **the master**: sales lines, upload batches, engine decisions, reminder cycles / stages, opt-outs, model registry (SQLite here, PostgreSQL / RDS in the cloud) |
| `data/refillcare/processed/models/` | trained model files (production model is not in git — it comes from the registry / S3) |
| `data/refillcare/processed/*.parquet`, `customer_data_fields.csv`, `backups/` | legacy pre-database history and its backups (read-only fast path / older CSV path) |
| `data/refillcare/raw_backup/` | the original full export, kept for re-training |
| `data/refillcare/processed/enterprise_backup_*.db` | online backups taken by `retrain_production.py` / `db_maintenance.py --vacuum` — delete once verified |
| `data/refillcare/SALT ITEMS LIST.csv` | item master: Code / itemCode, Item Name, PACK, SALT, CATEGORY, ITEMCAT |

A one-time `python scripts/slim_sales_data.py --apply` converts an old 23-column master to the 11 columns
(dry run without `--apply`; the raw file is copied to `raw_backup/` first).

---

## 3. How a refill date is predicted

| Patient history | Method |
|---|---|
| **≥ 6 purchases, stable** (Path A) | Registry model (HistGradientBoosting, learns the deviation from the patient's own median) or the personal median, with safety caps (insulin pen / inhaler in-use ceiling, 120-day max) |
| **3–5 purchases** (Path B) | Days of supply from units bought and the patient's own pace (`dos`), optionally blended with the median shrunk toward the salt's typical interval (`blend`) |
| Unstable / too little history / bulk or odd purchase | Pharmacist review — no automatic reminder |

Features (identical in training, backtest and serving; point-in-time, no future data):
purchase intervals and their stability, quantity vs usual, partial / multi-pack / top-up flags,
**calendar** (usual day of month and how well it explains the history), **visits** (other medicines, days
since the last visit), and **salt priors** from the item catalogue (store-wide typical interval and units per day).

**Patient and therapy keys** (switches in `.env`, off by default):
- `REFILLCARE_PATIENT_IDENTITY=name_mobile` — same name + different mobile = different patient
  (the store data has 14,071 names but 30,730 mobiles).
- `REFILLCARE_ITEM_GROUPING=salt` — brand or strength switches of the same molecule continue one refill cycle
  (only when bought one after the other; combinations stay separate).
- After switching either one, run the **identity cut-over** once (ML tab, or `python scripts/rekey_reminders.py --apply`):
  it backs up the database, retires old-key reminders, re-scores everyone and never re-sends a stage already sent.

**Prediction rules** (`REFILLCARE_RULE_*`, all off by default) are measured before use with the **rule A/B**
(ML tab, `ml_cli.py rules-ab`). Current measurements on 1,740 unseen refills (18 Jul – 16 Aug 2026):

| Setting | Within ±7 days | Avg error |
|---|---|---|
| Baseline engine (personal median) | 59.5% | 8.7 d |
| New model, no rules | 62.2% | 8.15 d |
| New model + `PATH_B_MODE=blend` | 62.6% | 7.87 d |

Over 3 back-to-back test months (5,065 refills) the model beats the baseline engine every month
(58.7% vs 55.6% pooled); `blend` lowers the average error every month but is ±1 pp on within-±7 depending on the month.

**Production (real database, 3 Nov 2026):** `hgb-20260518-20261103071906` — test 18 Jun – 17 Jul 2026:
56.7% within ±7 days (baseline 53.6%), average error 10.4 d (baseline 11.2 d).

---

## 4. Model lifecycle

- **Windows move with the data:** test = last 30 days before (last sale − 45 days), validation = the 30 days
  before that, train = everything earlier. A new month of sales shifts all three.
- **Promotion gate:** a candidate is promoted only if, on the untouched test window, it beats production and the
  baseline overall **and** does not regress in any reported segment (e.g. therapy such as insulin, history depth) on within-±7 and MAE.
- **Models are versioned** in `enterprise.db` (production / shadow / candidate / retired) with their train
  window, metrics and evaluation reports.

- **Fail-safe serving:** if the production model cannot load (missing file, checksum, other scikit-learn version)
  predictions fall back to the baseline engine; Streamlit shows a red warning with the cause and `/health` reports
  `degraded`.

```bash
python scripts/retrain_production.py          # after an upload: DB backup → retrain → gate → promote (≥ 4 GB RAM free)
python scripts/ml_cli.py plan                 # current windows, is a retrain due?
python scripts/ml_cli.py retrain [--promote]  # train → pick on validation → gate on test
python scripts/ml_cli.py rules-ab             # which prediction rules help? (read-only)
python scripts/ml_cli.py rolling --folds 3    # is the gain real across several months?
python scripts/ml_cli.py list | gate --version <id> | promote --version <id>
```

---

## 5. API (FastAPI)

Write endpoints need header `X-API-Key` = `REFILLCARE_API_KEY`; when it is unset they only work from the server
machine. Full, interactive list: `/docs`.

| Area | Endpoints |
|---|---|
| Sales upload | `POST /api/v1/sales/column-mapping/detect` · `POST /api/v1/sales/column-mapping` · `GET/DELETE /api/v1/sales/column-mapping/profiles` · `POST /api/v1/sales/preview` · `POST /api/v1/sales/ingest` (both accept an optional `column_mapping` JSON form field) · `GET /api/v1/sales/batches` · `POST /api/v1/sales/batches/{id}/rollback` |
| Reminders | `GET /api/v1/reminders/daily` · `GET /api/v1/reminders/months` · `GET /api/v1/reminders/export-csv\|export-json` · `POST /api/reminders/{id}/approve\|reject\|dispatch` · `GET /api/v2/med-sync/bundles` |
| Review & decisions | `GET /api/v1/review/queue\|summary` · `GET /api/refill-decisions` · `GET /api/customers/{id}/refill-history` |
| WhatsApp | `GET /api/v1/whatsapp/candidates` · `POST /api/v1/whatsapp/preview\|send-single\|dispatch` · `POST /api/v1/whatsapp/webhook` (STOP / START) · `GET /api/customers/opt-outs` · `POST /api/customers/{id}/opt-out\|opt-in` |
| ML lifecycle | `GET /api/v1/ml/plan\|models\|evaluations\|health\|settings` · `POST /api/v1/ml/retrain\|train\|evaluations\|rules-ab\|rolling` · `GET /api/v1/ml/reports/{rules_ab\|rolling}` · `GET/POST /api/v1/ml/identity/rekey` · `POST /api/v1/ml/models/{id}/promote\|status` |
| Medication rules | `GET/POST /api/v1/medications/rules` · `DELETE /api/v1/medications/rules/{id}` · `POST /api/v1/medications/classify` |
| Analytics | `GET /health` · `GET /api/v1/analytics/kpi\|transaction-types\|consensus-regimen` |

Every send path skips opted-out patients. The webhook has no write key (the provider must reach it); it is
verified with `WHATSAPP_WEBHOOK_SECRET` (signature or token, required when `REFILLCARE_ENV=production`).
`/health` also reports the serving model and data freshness.

---

## 6. Configuration (`.env`)

See [.env.example](.env.example) for every switch. The main ones:

| Variable | Purpose |
|---|---|
| `XINNO_*`, `WHATSAPP_*_TEMPLATE_*` | WhatsApp provider and approved templates |
| `MEDICAL_STORE_NAME`, `MEDICAL_STORE_CONTACT` | store name / phone in messages |
| `REFILLCARE_DATA_DIR`, `REFILLCARE_STORE_ID`, `REFILLCARE_SALT_CATALOG` | store data folder, store id (mapping profiles), item catalogue |
| `REFILLCARE_API_KEY` | write protection for the API |
| `REFILLCARE_ENV` | `production` fails closed without API key / webhook secret, JSON logs, no `/docs` |
| `DATABASE_URL` | PostgreSQL / RDS (unset = local SQLite `enterprise.db`) |
| `WHATSAPP_WEBHOOK_SECRET`, `REFILLCARE_STALE_DATA_GUARD`, `REFILLCARE_WARM_RULES` | webhook verification, block live sends on stale data, preload medication rules at API start |
| `REFILLCARE_RULE_*`, `REFILLCARE_PATIENT_IDENTITY`, `REFILLCARE_ITEM_GROUPING` | prediction rules and keys (shown read-only in the ML tab) |
| `ML_*` | window sizes, gate tolerances, training target, auto-retrain |
| `SUPABASE_*` | medication-rule tables (optional; local cache otherwise) |

---

## 7. Project layout

```text
app_refillcare.py           Streamlit operations app
api/                        FastAPI app (main.py routes, ml_routes.py, services.py, schemas.py, security.py)
frontend/                   Web app served by FastAPI (index.html, js/app.js, js/api.js, css/)
refillcare/
  data/                     schema.py (11 stored fields), column_mapping.py, sales_ingestion.py, incremental_ingest.py,
                            history_store.py (DB history), identity.py, enrichment.py (item catalogue),
                            medication_classifier.py, cleaning, dates, packing
  engine/                   unified_engine.py (Path A/B), prediction_rules.py, persistence.py (cycles & stages),
                            post_ingest.py, rekey.py, med_sync.py
  features/                 point_in_time.py, context.py (visits, salt priors)
  ml/                       training.py, evaluation.py (backtests), promotion.py (gate), registry.py, service.py,
                            model_provider.py (serving + fallback), artifact_store.py (local / S3)
  models/                   supply_hybrid.py (consensus & safety caps)
  jobs/                     durable job queue + worker (retrain, reminder expiry, sync retries)
  whatsapp/                 client, dispatcher, consent (opt-out), template formatter
  tenancy.py, stores.py     store id → database / schema (multi-store), logging_setup.py
reminder/                   scheduler, dispatch, storage used by ingestion and the pilot tools
database/                   SQLAlchemy models, control_models.py (stores), connection, additive migrations
services/, utils/           Xinno / Cloudinary clients, phone validation, bulk-send helpers
whatsapp_campaigns/         standalone text / image broadcast apps
model/, preprocessing/      legacy Path B predictor + feature table (kept as a tested benchmark)
experiments/identity_audit/ identity-normalisation audit code (used by its test; reports are git-ignored)
config/                     config.py, column_aliases.json
assets/, frontend/assets/   logos / favicon (Streamlit, web app)
scripts/                    see below
tests/                      pytest suite
docs/, notebooks/           deployment options (client), WhatsApp template, backend/tenancy plan, phase reports (history)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml   container + CI (operated by the backend team)
run_streamlit.bat, run_api.bat                              Windows launchers that always use .venv
```

| Script | Purpose |
|---|---|
| `retrain_production.py` | one-command production retrain: RAM check, DB backup, retrain, gate, promote (`--no-promote` to measure only) |
| `db_maintenance.py` | SQLite WAL checkpoint; `--vacuum` compacts after a backup |
| `ml_cli.py` | model lifecycle: plan, train, evaluate, retrain, gate, promote, rules-ab, rolling, outcomes |
| `run_sales_pipeline.py` | ingest a sales file and regenerate predictions from the command line |
| `slim_sales_data.py` | one-time: keep only the 11 fields (raw backup first) |
| `import_history_to_db.py` | onboard a store: load an existing history (CSV / Excel / parquet) into its database |
| `benchmark_incremental_upload.py` | reproducible timing of 1-day upload, re-upload, back-dated day and rollback (temporary database) |
| `rekey_reminders.py` | identity / salt-grouping cut-over (dry run by default) |
| `train_chronic_specialized_model.py` | offline quantile benchmark shown in the review tab |
| `seed_supabase_rules.py` | seed medication rules into Supabase |
| `purge_and_sync_reminders.py` | re-apply the medication classifier to `enterprise.db` (supersedes reminders for non-chronic / OTC items) |
| `verify_system_health.py`, `verify_column_aliases.py`, `verify_image_bulk_dry_run.py` | checks |
| `execute_single_live_test.py`, `execute_bulk_live_test.py` | controlled WhatsApp sends (dry run unless told otherwise) |

---

## 8. Tests

```bash
.venv\Scripts\python -m pytest -q          # full suite: 885 passed, 1 skipped (PostgreSQL test needs a server)
pytest tests/test_enterprise_platform.py -q # multi-store, incremental upload, stale data guard
pytest tests/test_api_security.py -q        # API keys, rate-limiting, webhook signatures
pytest tests/test_column_mapping.py -q     # column mapping & 11-field storage
```

Tests never touch live data: they use temporary folders and isolated databases, and never send real WhatsApp messages.

## 9. Data protection

Patient data (sales files, backups, parquets, `enterprise.db`, exports, logs, audit reports), trained model files
and `.env` are git-ignored. Do not commit them. Only 4 legacy model files that code still loads are versioned.
Phone numbers are masked in logs and in identity-conflict messages. Older commits still contain patient data —
purge the history before the repository leaves the team (GO_LIVE_CHECKLIST A9).
