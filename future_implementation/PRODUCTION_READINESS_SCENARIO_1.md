# RefillCare — Production Readiness, Scenario 1: from one local store to multi-tenant live

**Scenario 1.** RefillCare was built and tested on one machine for one store (NGOs Colony, `store-1`) using
local files (SQLite `enterprise.db`, parquet / CSV, joblib models under `data/refillcare/processed/`). It must
now run inside the existing Mediastra cloud product (QA: `qa.mediastracloud.com`, then Live) where **each tenant
has several stores**, data lives in **AWS RDS PostgreSQL + S3**, and RefillCare runs as **Docker containers**
pushed into the existing Mediastra deployment.

This document lists **every flaw found in the code** that blocks or weakens that move, with the change needed,
the owner and the order of work. It was produced by a review team (AI/ML engineer, backend/DevOps engineer,
application/data engineer, tech lead) reading the code on 4 Nov 2026; every item has a file reference.

Companion documents in this folder:
- [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md) — the tick-list to go live (store, RefillCare team, backend team).
- [PLAN_BACKEND_DEVOPS_TENANCY.md](PLAN_BACKEND_DEVOPS_TENANCY.md) — the target architecture and AWS / CI-CD plan.
- [requirements.txt](requirements.txt) — production dependencies for the API / worker image.

Severity: **Blocker** = must be fixed before any second store or any live send on AWS · **High** = before Live
(QA may run with it) · **Medium** = within the first month of Live · **Low** = clean-up.
Owner: **RC** = RefillCare team (this repo: ML / application code) · **BE** = Mediastra backend / DevOps team.

---

## 0. Decisions taken (4 Nov 2026)

| # | Question | Decision | Consequence |
|---|---|---|---|
| D1 | How do sales reach RefillCare? | **File upload is kept.** Store staff upload the CSV / Excel in the **Mediastra UI**; Mediastra's backend forwards it to the RefillCare API | Upload must become asynchronous: file → S3 → `202 Accepted` → worker ingests (section 5). Column mapping per store stays |
| D2 | Which UI do pharmacists use? | **Mediastra UI calls the RefillCare API** (server-to-server). Streamlit becomes an internal ops / ML tool only | The browser never holds a RefillCare key; Mediastra's backend calls RefillCare with a service token carrying tenant + store + user (section 1) |
| D3 | Data isolation in RDS | **One PostgreSQL schema per store** (what the code already does) plus a control plane mapping tenant → stores | Schema names must include the tenant (or be a surrogate key) — today `store_<id>` collides across tenants (section 1) |
| D4 | WhatsApp sender | **One WhatsApp (Xinno / WABA) account per tenant**; all stores of the tenant send from it | Credentials and templates move from global env vars to per-tenant secrets; the webhook is routed by WABA number to the tenant; STOP applies to all stores of that tenant (section 6) |

Open decisions (need an answer before the item they block):

| # | Question | Blocks | Recommendation |
|---|---|---|---|
| O1 | Are Mediastra store ids unique globally or only within a tenant? | T3 | Use a RefillCare `store_key` (surrogate, globally unique) either way |
| O2 | Does Mediastra issue a JWT (with `tenant_id`, `store_ids`, role) that RefillCare can verify (JWKS)? | T1 | Yes — otherwise per-tenant API keys (hashed) as an interim |
| O3 | Medication rules: one shared rule set for all tenants, or per-tenant overrides? | M15, A10 | Shared base set + tenant overrides, in Postgres |
| O4 | Item / salt catalogue: does each tenant have its own item codes? | A9 | Yes — catalogue per tenant from Mediastra's item master (`Items` = 35,782 in the QA screenshot) |
| O5 | Pooled (cross-store) ML model for small stores? | M13 | Not now (consent / product decision) — baseline engine for stores without enough history |
| O6 | Do the standalone campaign apps (`whatsapp_campaigns/`) ship to production? | W6 | No — not part of the refill product |

---

## 1. Target architecture (one paragraph)

```
Mediastra Web (Angular)  ──►  Mediastra backend  ──(service JWT: tenant_id, store_id, user, role)──►  RefillCare API (FastAPI, N replicas)
                                     │ upload file                                                    │  reads / writes
                                     ▼                                                                ▼
                               S3  s3://<bucket>/{tenant}/{store}/uploads/…           RDS PostgreSQL: schema refillcare_control
                                                                                       (tenants, stores, whatsapp accounts, jobs, consent)
RefillCare worker (jobs: ingest, sync, expire, dispatch)  ◄── job queue (Postgres) ──  + one schema per store  s_<store_key>
RefillCare ML worker (train / evaluate, ≥ 6 GB, own queue)                             (sales, decisions, cycles, stages, model registry)
Scheduler (one ECS scheduled task)                                                     S3  s3://<bucket>/{tenant}/{store}/models/{version}.joblib
Xinno (one WABA per tenant)  ──webhook──►  /api/v1/whatsapp/webhook/{tenant_key}  → tenant-level consent (STOP / START)
Streamlit ops tool (internal only, SSO, read-only by default)
```

No container keeps state on its local disk. Every request, job and log line carries `tenant_id` + `store_id`.

---

## 2. Hard-coded local paths — what each one becomes

The example from the request plus every other local-file dependency found:

| File : line | Constant / path | Today | In production |
|---|---|---|---|
| `app_refillcare.py:72` | `DEFAULT_MODEL_PATH = data/refillcare/processed/models/refill_model.joblib` | Legacy XGBoost model of store-1; dashboard shows "Telemetry Offline" without it (`:2158-2161`) | **Delete.** Model status comes from the registry (`ModelProvider.from_registry(db).describe()`) |
| `app_refillcare.py:73` | `DEFAULT_HUMAN_MODEL_PATH = …/human_ml_refill_model.joblib` | Legacy model | **Delete** |
| `app_refillcare.py:74` | `DEFAULT_TEST_DATA_PATH = processed/test.parquet` | store-1 test split | **Delete** — evaluation reads the store schema via `ml.evaluation` |
| `app_refillcare.py:75` | `DEFAULT_HISTORY_DATA_PATH = processed/purchase_history.parquet` | store-1 history | **Delete** — `history_store.load_store_history(store_id)` (Postgres) |
| `app_refillcare.py:76` | `DEFAULT_TRAIN_DATA_PATH = processed/training_dataset.parquet` | store-1 training set | **Delete** — training matrix is built from Postgres in the ML worker |
| `config/config.py:31-68` | `DATA_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, parquet paths, `LOGS_DIR`, `mkdir` at import | Local folders under the project root; creates dirs at import (fails on a read-only root FS); `MODELS_DIR` ignores `REFILLCARE_DATA_DIR` | Keep only `REFILLCARE_CACHE_DIR` (default `/tmp/refillcare`, created lazily). No mkdir at import. Logs to stdout |
| `config/config.py:31-36`, `database/connection.py:30` | SQLite fallback when `DATABASE_URL` is unset | Silent SQLite `enterprise.db` | **Refuse to start** in production without a PostgreSQL `DATABASE_URL` |
| `refillcare/store.py:34-74` | `StoreConfig.data_dir`, `master_csv`, `history_parquet`, `clean_transactions_parquet`, `raw_backup_dir`, `legacy_sqlite`; `mkdir` per request (`:77`) | Per-store local folder `/data/refillcare/stores/<id>` | Replace by `StoreConfig.s3_prefix = s3://<bucket>/{tenant}/{store}/`; drop the file properties |
| `refillcare/ml/artifact_store.py:51-57,72-85,103-105` | Models saved to local `MODELS_DIR` when `REFILLCARE_MODEL_BUCKET` is unset; global `MODELS_DIR` fallback for the default store | Other replicas cannot load the model → silent baseline | Bucket **mandatory** in production; key `{tenant}/{store}/models/{version}.joblib`; download cache `/tmp/refillcare/models/{tenant}/{store}/` |
| `refillcare/ml/registry.py:58,261-281` | `MODELS_DIR / human_ml_refill_model.joblib` bootstrap | Registers store-1's legacy model as **production in every new store** | **Delete** the bootstrap; new store → `baseline-median-v1` |
| `refillcare/models/supply_hybrid.py:377,437-451` | `HUMAN_ML_MODEL_PATH` (cwd-relative), `lru_cache(maxsize=1)` | Served when a store has no production model | **Delete** from serving |
| `refillcare/data/sales_ingestion.py:310-452,628-663,791-803` | `customer_data_fields.csv`, `backups/customer_data_fields_pre_*.csv`, `purchase_history.parquet`, `clean_transactions.parquet` | Legacy CSV / parquet master (still reachable from Streamlit rollback `app_refillcare.py:1226`) | **Delete** (Postgres `sales_transactions` is the master) |
| `refillcare/data/pipeline.py:60-61,145-160` | master CSV → parquet | Legacy | Delete (or one-off migration tool outside the image) |
| `refillcare/data/history_store.py:159-167,312-321` | parquet fast path + parquet auto-import on every ingest | Legacy | Remove after the one-time import of `store-1` |
| `refillcare/data/enrichment.py:18-41`, `medication_classifier.py:358-398` | `data/refillcare/SALT ITEMS LIST.csv` (baked into the image) | store-1's item codes used for every tenant; a bad file silently caches an empty catalogue | Postgres table `item_catalogue(tenant_id, code, name, pack, salt, category)` loaded from Mediastra's item master; fail loudly if missing |
| `database/fetch_sales.py:31-38` | `PURCHASE_HISTORY_PATH` parquet first; raw `SELECT * FROM sales_transactions` | Reads the `public` schema on Postgres (raw SQL bypasses `schema_translate_map`) | Delete (only used by a legacy script) |
| `reminder/storage.py:19,35-51` (via `api/services.py:93`, `app_refillcare.py:2164`) | `processed/refillcare.db` SQLite created per request | Local SQLite on every container, even on Postgres; read by `services.py:311,398,833,1333` | **Delete**; port the 4 readers to the store schema |
| `refillcare/whatsapp/client.py:80-86`, `services/xinno_*.py:100,132,136`, `whatsapp_campaigns/*` | `logs/*.log` file handlers (`xinno_image_template.py:136` adds a handler per call) | Local log files, contain send details | stdout JSON logs only (CloudWatch) |
| `refillcare/ml/service.py` reports, `api/services.py:1484` | `processed/ml_reports/*.json`, `processed/*.json` | Local report files | Store schema table `ml_reports` (or S3 `{tenant}/{store}/reports/`) |
| `config/column_aliases.json`, `refillcare/data/column_mapping.py:39-63` | Global alias file | Shared default (fine) | Keep as shared default + per-tenant overrides in Postgres; mapping profiles are already per store in the DB |
| `app_refillcare.py:79-90` | `find_artifact_path` searches cwd, `..`, `../..` | Fragile | Assets resolved relative to the package only |
| `.dockerignore:34-37` | `!data/refillcare/processed/models/*.joblib`, `!…/*.json`, `!SALT ITEMS LIST.csv` | **store-1 patient-derived models and data baked into the image** | Remove the re-include lines; image contains code only |

---

## 3. Findings by area

### 3.1 Tenancy, authentication, authorisation

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| T1 | Blocker | BE+RC | `api/security.py:37-57` | One shared `REFILLCARE_API_KEY` for every tenant; `X-Actor` is free text, so the audit trail is spoofable | Verify Mediastra's service JWT (JWKS; claims `tenant_id`, `store_ids`/`store_id`, `sub`, `role`, `aud=refillcare`). Interim: hashed per-tenant API keys in the control plane. Actor = JWT `sub` |
| T2 | Blocker | RC | `api/main.py:158-162`, `refillcare/tenancy.py:49-57` | Store taken from `X-Store-Id` **or `?store=`**; `X-Tenant-Id` stored in a context var that nothing reads — any key holder can read / write any store of any tenant | Store and tenant come from the verified token; reject a store not owned by the token's tenant; remove the `?store=` fallback |
| T3 | Blocker | RC | `refillcare/tenancy.py:34-44`, `database/connection.py:78`, `artifact_store.py:68-70`, `database/control_models.py` | Schema `store_<id>` ignores the tenant; `a-b` and `a_b` collide; `StoreModel.store_id` is a global primary key; `tenant_id` is nullable free text | Control plane: `tenants(tenant_id PK)`, `stores(store_key PK, tenant_id FK NOT NULL, mediastra_store_id, UNIQUE(tenant_id, mediastra_store_id))`. Schema `s_<store_key>` (≤ 63 chars); S3 prefix `{tenant}/{store_key}/`; refuse collisions |
| T4 | Blocker | RC | `frontend/js/api.js:10-20`, CORS `api/main.py:128-134` | The static web app keeps the master key in `localStorage` | Do not deploy `frontend/` to tenants (D2). Remove `X-API-Key` from the CORS allow-list; browsers never call RefillCare directly |
| T5 | High | RC | `api/main.py:418-435`, `refillcare/stores.py:55-81` | `GET /api/v1/stores` lists all tenants' stores; `POST /api/v1/stores` can change a store's `tenant_id` (takeover) | Platform-admin role only; `tenant_id` immutable; list filtered by token tenant |
| T6 | Medium | RC | `refillcare/tenancy.py:52`, `refillcare/stores.py:105-110` | No store context → silently `REFILLCARE_STORE_ID`; unknown stores auto-registered outside production | No default store anywhere except local dev; workers and UI raise without a context |
| T7 | Medium | RC | `refillcare/stores.py:16` | Store settings cached 60 s per process — a disabled store keeps being served | Cache ≤ 10 s or invalidate via `updated_at` check |

### 3.2 Database (PostgreSQL / RDS)

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| D-1 | Blocker | RC | `database/migrations.py:52-63` | `drop_prototype_tenant_ml_tables` **drops** `ml_model_versions`, `ml_evaluation_runs`, `refill_decision_outcomes` on every store init if they have a `tenant_id` column — adding tenant columns later would wipe the model registry. `_v3` also drops indexes (the "additive only" claim is false) | Remove before go-live; migrations versioned and reviewed (Alembic or the existing version table), never destructive at runtime |
| D-2 | High | RC+BE | `database/connection.py:108-129` | Each process creates / migrates a store schema on first request under one in-process lock; no Postgres advisory lock; replicas race; app DB user needs CREATE at runtime; all stores wait while one migrates | Migrations run as a **deploy job** (one ECS task) per schema under `pg_advisory_lock(hash(schema))`; request path only checks the version (control plane `stores.schema_version`); app user has DML rights only |
| D-3 | High | RC | `api/main.py:310-411`, `incremental_ingest.py:162-206` | Upload endpoints are `async def` but run blocking pandas / DB work on the event loop; up to 2 M rows inserted in one transaction; advisory lock released at the first commit so re-scoring is not serialised across containers | Handlers `def` (thread pool) and the work moved to the worker (section 5); batch inserts in chunks; store lock held for ingest + sync |
| D-4 | Medium | RC | `database/fetch_sales.py:38`, `scripts/db_maintenance.py`, `scripts/retrain_production.py`, `refillcare/engine/rekey.py:36-46` | SQLite-only code: raw SQL without schema, `sqlite3` backups, WAL / VACUUM | Delete or gate to SQLite; Postgres uses RDS snapshots / PITR |
| D-5 | Medium | RC | `database/models.py` (all `DateTime` columns) | Naive `datetime.utcnow()` timestamps; `JSON` not `JSONB` | `DateTime(timezone=True)` (`timestamptz`), aware UTC; `JSONB` on Postgres |
| D-6 | Medium | BE | — | RDS PITR restores the whole instance, not one store | Runbook: per-schema `pg_dump` / restore for one store; tested once |

### 3.3 Stateless containers and memory

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| S1 | Blocker | RC | `artifact_store.py:56-57,72-85` | S3 optional → models on the container disk; API replicas fall back to baseline silently | Refuse to start in production without `REFILLCARE_MODEL_BUCKET`; register / promote only `s3://` URIs |
| S2 | Blocker | RC+BE | `.dockerignore:34-37` | Store-1 models, reports and item list baked into the image (PHI-derived) | Remove re-includes; rebuild; scan the image |
| S3 | High | RC | `refillcare/data/history_store.py:41,142-210` | Unbounded in-memory cache of every store's full history (0.5–1 GB per store, per uvicorn worker) — OOM with many stores | LRU bounded by bytes, keyed by store; API should query only what a request needs (decisions table), not full history |
| S4 | High | RC | `model_provider.py:21,45-59`, `supply_hybrid.py:437,467` | `_CACHE` never evicts; `LOAD_ERRORS` keyed only by version id; `FALLBACK_COUNTERS` per process; one `lru_cache` model not keyed by store | Key by `(tenant, store, version)`; LRU on `_CACHE`; per-store metrics |
| S5 | Medium | RC | `api/main.py:110-121` | Thread at import fetches Supabase rules in every worker / replica | Keep, but bounded and per-tenant once rules move to Postgres (M15) |
| S6 | Medium | RC | `config/config.py:34,40,50,68`, `refillcare/store.py:77` | mkdir at import / per request | Lazy, under `/tmp` |

### 3.4 Machine learning

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| M1 | Blocker | RC | `refillcare/ml/registry.py:261-281` | `ensure_bootstrap_versions` registers `legacy-human-hgb` (store-1's model, no checksum) as **production** in every new store → tenant B predicted by store-1 patients' model; also `joblib.load` without checksum (`:269`) | Delete the legacy bootstrap; a new store starts on `baseline-median-v1` |
| M2 | Blocker | RC | `refillcare/engine/unified_engine.py:186-192`, `supply_hybrid.py:570` | When a store has no production model, `use_ml` stays True and the global legacy model is loaded | `ModelProvider.from_registry` always returns baseline when there is no production row; remove `load_human_ml_bundle` from serving |
| M3 | High | RC | `api/services.py:363-372`, `refillcare/models/prediction.py:74-75` | `POST /api/v1/predictions/generate` passes the registry bundle to the legacy predictor (expects `pipeline`) — every secondary row fails; raises when production is baseline | Route through `UnifiedRefillDecisionEngine` + `ModelProvider` (as upload re-scoring already does) |
| M4 | High | RC | `app_refillcare.py:108-160,2145-2160,2657-2662,2779-2783,4649,4741` | Streamlit loads legacy models, "Generate Updated Predictions" re-scores any store with store-1's XGBoost model, and runs `retrain_cycle` / `rolling_evaluation` (≈ 4 GB) inside the UI container | Remove legacy loaders and buttons; ML actions = `enqueue("retrain" / "rolling")` |
| M5 | High | RC | `ml/service.py:24-44`, `training.py:64-97`, `history_store.py:142-150` | Training pulls the full history into pandas; per-event Python loop; no lookback bound | Dedicated ML worker (≥ 6 GB, no history cache), column-pruned SQL, bounded lookback (e.g. 5 years) |
| M6 | High | RC | `ml/service.py:125`, `jobs/queue.py:29-33`, `jobs/worker.py:259-267` | Retrain recommended whenever production is not validated → re-enqueued **every hour** after it finishes; each run repeats two full backtests; every worker loads every store's history hourly | Dedupe against done / failed jobs in the same month; back-off after "gate failed"; single scheduler |
| M7 | Medium | RC | `api/main.py:409` vs `jobs/worker.py:267` | Dedupe keys `retrain` vs `retrain:YYYY-MM` — both can queue | One key format + DB unique index (J1) |
| M8 | Medium | RC | `artifact_store.py:127-149`, `registry.py:222-237` | Pruning keeps newest 3 + production / shadow; rollback picks an older retired production whose file was pruned → baseline | Always keep the last N versions that were ever production |
| M9 | Medium | RC | `artifact_store.py:98-100` | Fixed `.download` temp name — concurrent workers corrupt the download | Unique temp file + `os.replace` |
| M10 | Medium | RC | `model_provider.py:241-246`, `registry.py:41-48` | scikit-learn mismatch only warns and still unpickles; Python / numpy version not compared; `code_commit` shells out to git (absent in the image) | Refuse to load on major.minor mismatch of scikit-learn / numpy; `GIT_SHA` from env at build |
| M11 | Medium | BE | — | Pickles from S3 = code execution if the bucket is writable by the API | Worker role may `PutObject`; API role read-only; bucket versioning + SSE-KMS; Object Lock optional |
| M12 | Medium | RC | `prediction_rules.py:59-61` | Three rule switches only global (env); Path B / device cap per store | All rules overridable per store (store `settings`) |
| M13 | Medium | RC | `training.py:165`, `ml/service.py:54,88`, `promotion.py` | ≥ 500 events, ≥ 180 days + 90-day buffer, ≥ 200 gate events — small stores never get a model and today fall into M1 / M2 | Explicit policy: baseline + pharmacist review until thresholds are met; shown in the UI |
| M14 | Low | RC | `scripts/ml_cli.py`, `scripts/retrain_production.py`, `scripts/train_*.py` | No `--tenant/--store`; fixed local output paths; `train_chronic_specialized_model.py` overwrites `refill_model.joblib` | Required store argument; scripts enqueue jobs; offline trainers out of the image |
| M15 | High | RC+BE | `refillcare/data/medication_classifier.py:411-560` | Supabase rule table is global (no tenant / store); edits by one tenant change all; replicas disagree up to 5 min; outage silently falls back; a rule change does not re-score | Rules in Postgres control plane (shared base + tenant overrides, versioned); rules version stamped on decisions and models; degraded source blocks live sends |

### 3.5 Sales ingestion (decision D1: file upload kept)

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| I1 | Blocker | RC+BE | `api/main.py:310-411` | Upload parsed and re-scored synchronously inside the HTTP request (minutes for big files; times out behind the ALB) | `POST /api/v1/sales/uploads` stores the file in S3 `{tenant}/{store}/uploads/{batch_id}`, returns `202 {batch_id}`; worker job `ingest_batch` parses, inserts, re-scores; `GET /api/v1/sales/batches/{id}` gives progress |
| I2 | High | RC | `incremental_ingest.py:124` | Idempotency only by file checksum; same rows in a different layout pass the checksum (line keys still dedupe rows, but a batch is created) | `Idempotency-Key` header from Mediastra + unique index on `(store, file_sha256)`; keep line-key dedupe |
| I3 | High | RC | `refillcare/data/dates.py:87-89`, `cleaning.py:104`, `column_mapping.py:107` | Day-first dates assumed for every store; an MM/DD export is silently misread | Date format saved in the store's mapping profile and validated on preview |
| I4 | Medium | RC | `refillcare/engine/post_ingest.py:20-34` | Unparseable latest-sale date silently becomes today | Raise |
| I5 | Medium | RC | `incremental_ingest.py:337-342` | `data_status` falls back to parquet | DB only |
| I6 | Medium | BE | — | Upload limits | ALB / API body limit and `MAX_UPLOAD_MB` aligned; virus / type check on S3 upload |

### 3.6 WhatsApp (decision D4: one sender per tenant)

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| W1 | Blocker | RC | `api/main.py:158-160`, `api/services.py:1903-1914`, `refillcare/whatsapp/consent.py:62-70` | In production the webhook gets **400** (no `X-Store-Id`) → STOP is dropped (legal). Even with a store, STOP applies to one store only; the response echoes the full phone | Route `/api/v1/whatsapp/webhook/{tenant_key}` (or by receiving WABA number) to the tenant; **tenant-level consent table** `whatsapp_consent(tenant_id, phone_key, status, at, source)` checked on every send; mask phones in responses |
| W2 | Blocker | RC | `whatsapp/dispatcher.py:160-245,290-345`, `engine/persistence.py:660-720` | Idempotency is check-then-send: two replicas / a double click both send; a crash after Xinno 200 before commit re-sends (FAILED retried 3×) | Claim rows atomically (`UPDATE … SET status='SENDING', claim=… WHERE status='APPROVED' … RETURNING` / `FOR UPDATE SKIP LOCKED`), unique live-log row per stage, reconcile SENDING rows, idempotency key to Xinno if supported |
| W3 | Blocker | RC+BE | `whatsapp/client.py:31-33,120-135`, `services/xinno_whatsapp.py:233-250`, `xinno_image_template.py:214-219`, `dispatcher.py:85`, `persistence.py:95` | Global `XINNO_API_KEY`, `XINNO_WABA_NUMBER`, template; default store name "PHARMA HUBB" and a hard-coded contact | `tenant_whatsapp_accounts(tenant_id, waba_number, api_key_secret_arn, template_name, language, image_template)`; store name / contact from the store registry; **no defaults** — send fails if missing |
| W4 | High | RC | `whatsapp/dispatcher.py:183-195` | Month mode / no date sends future-dated APPROVED stages early (a −7 day reminder weeks ahead) | Live sends only for `target_send_date == today (store time zone)` |
| W5 | High | RC | — (no code) | No send window — live sends possible at 2 a.m.; no scheduled dispatch job (`jobs/queue.py:18` has none) | Per-store send window (default 09:00–20:00 IST); `dispatch_due` job per store after the morning upload; per-store `live_send_enabled` kill switch |
| W6 | High | RC | `whatsapp_campaigns/*`, `utils/bulk_send.py`, `utils/image_campaign.py` | Hard `dry_run=False`, no opt-out check, phone-list CSV uploads | Not shipped to production (O6) |
| W7 | Medium | RC | `security.py:78` | Webhook token accepted as `?token=` (lands in ALB logs); one global secret; no Meta GET verification | Header only; per-tenant secret |
| W8 | Medium | RC | `client.py:31`, `xinno_whatsapp.py:52`, `config.py:61` | Three different default Xinno URLs | One setting |
| W9 | Medium | RC | `services/cloudinary_image.py:34,47-49` | Global Cloudinary folder `pharma_hubb/refill_campaign`, public URLs | Per-tenant folder or S3 + CloudFront signed URLs (only if image templates are used) |

### 3.7 Time zone and dates

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| Z1 | High | RC | `date.today()` in `persistence.py:103,582`, `incremental_ingest.py:265,323,344`, `jobs/worker.py:93,149,365,421`, `post_ingest.py:34,168`, `unified_engine.py:380`, `api/services.py:119`, `app_refillcare.py` | Containers run in UTC: between 00:00 and 05:30 IST "today" is yesterday — send queue, stale-data guard and expiry shift | `store_today(store)` helper using `ZoneInfo(store.timezone or "Asia/Kolkata")` everywhere; `TZ` not relied on |
| Z2 | Medium | RC | ~40 call sites of `datetime.utcnow()` / naive `datetime.now()` (listed by the review: `persistence.py`, `dispatcher.py`, `consent.py`, `column_mapping.py`, `incremental_ingest.py`, `rekey.py`, `jobs/queue.py`, `sales_ingestion.py`, `decision_types.py`, `reminder/dispatch.py`) | Deprecated, naive, mixed | Aware UTC (`datetime.now(timezone.utc)`) + `timestamptz` (with D-5) |
| Z3 | Low | RC | `app_refillcare.py` ~800 | "Reminder Date" = expected − 2 days; stages are −7/−3/−1/0 | Show the next stage date |

### 3.8 Background jobs and scheduling

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| J1 | High | RC | `jobs/queue.py:29-33,202-210` | Dedupe = read then insert, no unique constraint → duplicate jobs | Partial unique index `(store_key, dedupe_key) WHERE status IN ('queued','running')`; claim with `FOR UPDATE SKIP LOCKED` |
| J2 | High | RC | `jobs/queue.py:20,43-46,193` | A job "running" > 180 min is re-queued with no heartbeat / attempts check → OOM-killed retrain loops forever; slow one runs twice; SIGTERM leaves "running" 3 h | `heartbeat_at` column, `max_attempts`, per-kind timeout, graceful shutdown re-queues |
| J3 | High | RC+BE | `jobs/worker.py:296-300,447-473` | One FIFO loop, one job at a time: a 30-min retrain blocks every store's sync / expiry | Two queues / services: `default` (ingest, sync, expire, dispatch) and `ml` (train, evaluate); `JobModel.tenant_id`, `priority` |
| J4 | High | BE | `jobs/worker.py` scheduler | Scheduler runs in every worker unless `REFILLCARE_DISABLE_SCHEDULER` is set | One leader (ECS scheduled task, or `pg_try_advisory_lock` leader) |
| J5 | Medium | RC | `jobs/worker.py:259-262,435` | Hourly pass loads full history of every store | Plan from control-plane metadata (latest sale date, last retrain) |

### 3.9 Streamlit (decision D2: internal tool only)

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| U1 | Blocker | BE | `app_refillcare.py:1994` | No login, no tenant / store picker, runs on `REFILLCARE_STORE_ID`; anyone who reaches it can send live / roll back | Not exposed to tenants. If deployed: private network + SSO, store picker limited to the operator's tenants, live send disabled |
| U2 | High | RC | `app_refillcare.py:285,836,1550,1568,1594,1690,1822,1908` | `st.cache_data` functions without a store id argument | Store id (and tenant) in every cached function's arguments |
| U3 | High | RC | `app_refillcare.py:303,318` | Hard-coded counts `2264` and `909657` (store-1 legacy totals) shown as metrics | Compute from the store schema (breaks the "no hard-coded metrics" rule) |
| U4 | Blocker | RC | `app_refillcare.py:4081-4106` | Store name defaults to "PHARMA HUBB", contact hard-coded; operator can type any name into patient messages | From the store registry, read-only |
| U5 | Medium | RC | `app_refillcare.py:181-189,238-251,332,2183,2370-2376,3325,3991` | Parquet / CSV fallbacks, benchmark JSON and hard-coded MAE strings shown as current figures | Remove; registry / evaluation tables only |
| U6 | Medium | RC | `app_refillcare.py:2620-2630` | Upload + re-score synchronous in the Streamlit process | Same async upload API as I1 |
| U7 | Low | RC | `app_refillcare.py:2215,2648,4488` | `st.cache_*.clear()` wipes every session's cache | Clear only the store's keys (version argument) |

### 3.10 Docker, CI/CD, configuration

| ID | Sev | Owner | File : line | Flaw | Required change |
|---|---|---|---|---|---|
| C1 | High | BE | `.github/workflows/ci.yml:41,46` | Tests and Postgres test `continue-on-error: true`; `pip-audit` cannot fail; docker job `if: always()` — CI can never go red; no ECR push, no SHA tags, no QA → Live promotion | Blocking tests; full suite on a Postgres service container; build once, tag by commit SHA, push to ECR, deploy to QA, manual approval to Live; GitHub OIDC (no static AWS keys) |
| C2 | Medium | BE | `Dockerfile` | One stage with Streamlit, matplotlib, xgboost, supabase, cloudinary for every role; `COPY . .` copies legacy folders and scripts; `REFILLCARE_ENV=production` hard-coded; HEALTHCHECK on :8000 fails for worker / Streamlit; no graceful-shutdown timeout | Multi-stage; API / worker image from [requirements.txt](requirements.txt); separate ops-UI image; explicit COPY list; env from the task definition; per-role health checks; `--timeout-graceful-shutdown` |
| C3 | Medium | BE | `Dockerfile:45` | `--forwarded-allow-ips "*"` (client IP spoofable; rate limiter keys on it) | ALB subnet CIDR only |
| C4 | High | RC | `config/config.py`, `api/main.py` startup | Production mode does not refuse to start — it answers 503 per request when keys are missing; SQLite / missing bucket not checked | Start-up validation in production: Postgres URL, bucket, JWT / key config, webhook secret, catalogue present — else exit non-zero |
| C5 | High | BE | `api/main.py:150`, `security.py:85-108` | Rate limit per (key prefix, client IP) in process memory: all Mediastra traffic shares one bucket, and each replica has its own | Limit per tenant at the gateway (ALB / API Gateway / WAF), or Redis |
| C6 | Medium | RC | `api/main.py:236` | `/health` is public and shows model, rules status and auth mode of the default store | `/livez` and `/readyz` public; `/health` authenticated, per store |
| C7 | Low | RC | `requirements.txt` | Streamlit, xgboost, matplotlib, cloudinary in the production set | Split: [requirements.txt](requirements.txt) (API / worker) + ops-UI extras |

### 3.11 Security, privacy, observability

| ID | Sev | Owner | Flaw | Required change |
|---|---|---|---|---|
| P1 | High | BE | Phones and names in plain text across many tables (`database/models.py:83,107,182,229,251,271`); no retention / erasure / export (DPDP Act 2023); no access audit | Retention job per store; erasure + export endpoints by phone; access log; RDS / S3 encryption (KMS) |
| P2 | Medium | RC | Logs have no `tenant_id`; masking covers only Indian 10-digit mobiles | `tenant_id`, `store_id`, `request_id`, `job_id` in every JSON log line; generic phone masking |
| P3 | Medium | BE | No metrics / tracing | CloudWatch metrics per store: sends, failures, opt-outs, model degraded, data lag, job failures; alarms (see checklist) |
| P4 | High | BE | Keys used in development are in the developer's local `.env` | New production keys in Secrets Manager; rotate the development ones |
| P5 | Medium | RC | Older git commits contain patient files | Purge history before the repository leaves the team |

### 3.12 Legacy code to remove before production (image must not contain it)

| Area | Files |
|---|---|
| Legacy data path | `refillcare/data/pipeline.py`; file functions in `sales_ingestion.py` (`append_sales_records_to_master_csv`, `remove_records_from_master_csv`, `prune_old_backups`, parquet writes in `process_monthly_sales_data` / `rollback_monthly_sales_import`, `ingest_monthly_sales_pipeline`); `database/fetch_sales.py`; parquet paths in `config/config.py`; file properties of `StoreConfig` |
| Legacy SQLite storage | `reminder/storage.py` (after porting its 4 readers), `reminder/send_message.py`, `reminder/message_logs.py`; `reminder/dispatch.py` / `scheduler.py` (only pilot / evaluation modules use them) |
| Legacy models | Streamlit `load_refill_model` / `load_human_refill_model` / "Generate Updated Predictions"; `supply_hybrid.load_human_ml_bundle`; registry legacy bootstrap; `model/`, `preprocessing/` (tested benchmark only — keep in repo, exclude from image); `refillcare/features/pipeline.py:45-47`, `refillcare/models/training.py:278-303`, `refillcare/evaluation/pilot_preflight.py:175` |
| Campaign tooling | `whatsapp_campaigns/*`, `utils/bulk_send.py`, `utils/image_campaign.py`, `utils/audit.py`, `services/xinno_image_template.py`, `services/cloudinary_image.py` (check `services/xinno_whatsapp.py`: still imported by `persistence.py:44`) |
| Scripts (keep in repo, not in image) | `run_sales_pipeline.py`, `slim_sales_data.py`, `purge_and_sync_reminders.py`, `sync_enterprise_db_chronic.py` (mass UPDATE, no `--store`, no dry run), `seed_supabase_rules.py` (deletes global rules), `execute_*_live_test.py` (real sends), `verify_*`, `db_maintenance.py`, `train_*`, `backtest_*`, `run_offline_validation.py`, `build_refillcare_notebooks.py`. Keep: `import_history_to_db.py` (one-off migration; add S3 input), `ml_cli.py` (with `--store`) |

---

## 4. Work plan

### Phase 1 — RefillCare team: blockers in this repo (QA cannot onboard a second store before this)

| Step | Items | Done when |
|---|---|---|
| 1.1 | Control plane + identity: tenants, stores (`store_key`), tenant in every key | T2, T3, T5, T6 · unit tests: tenant A token cannot reach tenant B store; schema / S3 prefix from `store_key` |
| 1.2 | Token verification (JWT or hashed per-tenant keys) | T1 · actor from token |
| 1.3 | Remove legacy model paths and bootstrap; baseline for stores without a model | M1, M2, M3, M4 (model parts), section 2 rows 1–5, 9–10 · test: new store serves `baseline-median-v1` |
| 1.4 | Production start-up validation; S3 mandatory; no SQLite; no local files | C4, S1, S2, S6, section 2 · container runs with a read-only root FS and `/tmp` only |
| 1.5 | Remove destructive migration; versioned migrations runnable as a job | D-1, D-2 (code part) |
| 1.6 | Async upload via S3 + worker ingest | I1, I2, D-3, U6 |
| 1.7 | WhatsApp per tenant: accounts table, webhook routing, tenant consent, atomic claim, send window, today-only live sends | W1–W5, U4 |
| 1.8 | Time zone | Z1 (+ Z2 with D-5) |
| 1.9 | Jobs: unique dedupe, heartbeat, attempts, two queues, retrain back-off | J1, J2, J3 (code), M6, M7 |
| 1.10 | Remove legacy code from the image | 3.12, U3, U5, W6 |
| 1.11 | Postgres test suite green | every test also run against Postgres in CI (C1) |

### Phase 2 — Backend team: platform (in parallel with phase 1)

RDS + schemas + migration job (D-2), S3 + IAM roles (M11), ECR / ECS services (API, worker, ML worker, scheduler, optional ops UI), CI/CD with QA → Live (C1, C2, C3), secrets (P4), gateway rate limits (C5), JWT issuance from Mediastra (T1), Mediastra UI screens calling the API (D2), upload flow (I1, I6), per-tenant Xinno accounts and webhook URLs (W3), monitoring and alarms (P3), DPDP (P1). Details: [PLAN_BACKEND_DEVOPS_TENANCY.md](PLAN_BACKEND_DEVOPS_TENANCY.md).

### Phase 3 — QA (`qa.mediastracloud.com`)

1. Migrate `store-1` into tenant "NGOs Colony" (sales, decisions, cycles, stages, **opt-outs**, sent history, mappings; retrain on AWS).
2. Onboard a second test store under the same tenant and one store under a second tenant.
3. Run the acceptance tests below; one week of dry runs.

### Phase 4 — Live

Go / no-go in [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md); first month with pharmacist approval before every live send.

---

## 5. Acceptance tests for Scenario 1 (must all pass on QA)

| # | Test | Expected |
|---|---|---|
| AT1 | Token for tenant A calls any endpoint with a store of tenant B | 403 |
| AT2 | `?store=` query parameter, missing token, browser call with no service token | 401 / 400 — never the default store |
| AT3 | Brand-new store (no history) | Serves `baseline-median-v1`; no legacy / store-1 model ever loaded; no file written outside `/tmp` |
| AT4 | Upload a 1-day file for store X | `202`, batch completes in the worker, only store X's schema and S3 prefix change; same file again → rejected as duplicate |
| AT5 | Retrain store X while store Y uploads | Store Y's ingest / sync / expiry continue (separate queue); exactly one retrain job for X |
| AT6 | Promote a model in one API replica | All replicas serve the new version within the provider TTL; `/health` not degraded |
| AT7 | Kill the API container mid-dispatch, restart | No reminder sent twice (SENDING reconciliation) |
| AT8 | STOP from a patient of store X | Opt-out recorded at tenant level; no store of that tenant sends to that phone again |
| AT9 | Dispatch called at 01:00 IST and at 23:00 IST | No live send outside the window; "today" = IST date |
| AT10 | Container with read-only root FS, no `data/` folder | API, worker, ML worker start and pass `/readyz` |
| AT11 | Image scan | No `.joblib`, `.parquet`, `.csv`, `.db`, `.env` in the image |
| AT12 | CI with a failing test | Pipeline red, no image pushed |
