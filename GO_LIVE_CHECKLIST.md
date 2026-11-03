# RefillCare — Go-Live Checklist

Everything that must be done before RefillCare sends live WhatsApp reminders from the Mediastra
cloud application. Two owners:

- **Store / Ops (RefillCare owner)** — data, model and clinical sign-off for the current store (NGOs Colony, `store-1`).
- **Backend team (Mediastra)** — AWS (RDS, S3, containers), multi-tenant / multi-store, security, monitoring.

The RefillCare code is already multi-store aware (one schema per store, `X-Store-Id` header, per-store
locks, jobs, models). What is listed for the backend team is the infrastructure and integration around it.

Status today (3 Nov 2026): single store on SQLite; 885 tests pass; Streamlit and FastAPI verified on a
copy of the real database (835,361 sales lines, Dec 2020 – 15 Oct 2026).

---

## Part A — Store / Ops (before live)

| # | Task | How | Done when |
|---|---|---|---|
| A1 | Upload the missing sales **16 Oct – today** | Streamlit "Upload" tab or `POST /api/v1/sales/ingest` (only new lines are added) | `GET /api/v1/sales/data-status` shows `"ok_to_send": true` (latest sale ≤ 2 days old) |
| A2 | **Retrain the model** in the app's Python (the current production model was pickled by an older scikit-learn and cannot load — the app is on the baseline engine and shows a red "ML model warning") | Close other apps (≥ 4 GB RAM free), stop Streamlit/API, then `.venv\Scripts\python scripts\retrain_production.py` (takes a DB backup first; promotes only if the gate approves) | Red warning gone in Streamlit; `/health` → `"status": "healthy"`, `serving_model.degraded: false`. **Done 3 Nov 2026**: `hgb-20260518-20261103071906` promoted (gate passed; test 18 Jun – 17 Jul: 56.7% within ±7 d vs baseline 53.6%, MAE 10.4 vs 11.2 d). Retrain again after A1 so the model learns from the latest months |
| A3 | Decide the prediction switches (measured on the sandbox; recommended **yes**) | Set in `.env`: `REFILLCARE_RULE_PATH_B_MODE=blend`, `REFILLCARE_PATIENT_IDENTITY=name_mobile`, `REFILLCARE_ITEM_GROUPING=salt`; then run the identity cut-over (ML tab → dry run first) and retrain (A2) | Cut-over report reviewed; no double sends |
| A4 | Clinical sign-off | Pharmacist reviews the eligible list, review queue and classification rules (Supabase) for one week of dry runs | Signed off |
| A5 | Dry-run a full week | Keep `REFILLCARE_LIVE_REQUIRES_APPROVAL=true`; approve/reject in the review queue; dispatch in dry-run mode | Messages, Med-Sync bundles and opt-out handling look right |
| A6 | Housekeeping (stop the app first) | `.venv\Scripts\python scripts\db_maintenance.py` (folds the 364 MB WAL file back); optional `--vacuum` (takes a backup first) | DB file compacted |
| A7 | Delete old local files after a verified backup | `data/refillcare/processed/enterprise_backup_*.db`, `data/refillcare/raw_backup/` (127 MB), legacy folders `model/`, `preprocessing/`, `notebooks/`, `scratch/`, `experiments/`, `run_*.py`. **Keep** `refillcare.db` for now — still referenced by `database/migrations.py`, `reminder/storage.py`, `refillcare/store.py` | Repo and disk clean |
| A8 | Review and commit the changes (local) | `git diff`, then commit; never commit `.env`, `*.db`, patient CSV/Excel | Clean `git status` |
| A9 | Decide on patient data in git history | Older commits contain patient files (PHI); purging rewrites history (`git filter-repo`) — needs everyone to re-clone | Decision recorded |
| A10 | Hand the backend team: the store's history file(s), the Supabase rules project, the Xinno templates, and this checklist | — | Backend team confirms receipt |

---

## Part B — Backend team

### B1. Infrastructure (AWS)

| # | Task | Notes |
|---|---|---|
| B1.1 | **RDS PostgreSQL** (one instance, one schema per store `store_<id>`) | Set `DATABASE_URL=postgresql+psycopg://…`; `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` per container. Schemas and migrations are created automatically on first use of a store (`init_store`) |
| B1.2 | RDS **automated backups + point-in-time recovery** (7–35 days), Multi-AZ for production | RefillCare keeps **no** file backups of its own on AWS |
| B1.3 | **S3 bucket for models** (private, versioning on, SSE encryption) | `REFILLCARE_MODEL_BUCKET`, `REFILLCARE_MODEL_PREFIX`, `REFILLCARE_MODEL_KEEP=3`; IAM role for the containers with `s3:GetObject/PutObject/DeleteObject/ListBucket` on that prefix only. Artifacts are SHA-256 checked before loading |
| B1.4 | Build the image from `Dockerfile`; run **three services** from it | API: default `CMD` (uvicorn, 2 workers); worker: `python -m refillcare.jobs.worker`; Streamlit: `streamlit run app_refillcare.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`. `docker-compose.yml` is the local reference |
| B1.5 | **Exactly one worker** with the scheduler (or set `REFILLCARE_DISABLE_SCHEDULER=true` on all but one) | Jobs are claimed atomically, so extra workers are safe for job execution; the schedule (daily expiry, monthly retrain, sync retries) needs one scheduler |
| B1.6 | Python version and libraries exactly as `requirements.txt` (Python 3.13, scikit-learn 1.7.2) | A model must be loaded by the scikit-learn it was trained with. **Any scikit-learn upgrade = retrain every store's model** (`scripts/retrain_production.py` or `POST /retrain`) |
| B1.7 | Container memory: API ≥ 2 GB, worker ≥ 4 GB (training), Streamlit ≥ 2 GB | Measured: history of 835k lines ≈ 0.5–1 GB in memory; training peaks ≈ 3–4 GB |
| B1.8 | Run the **PostgreSQL test** and the compose stack once before live | `REFILLCARE_PG_TEST_URL=postgresql+psycopg://… pytest tests/test_postgres_tenancy.py`; `docker compose up -d`; `pytest -q` in the image |

### B2. Network and security

| # | Task | Notes |
|---|---|---|
| B2.1 | API and Streamlit in **private subnets**, reachable only through the ALB / Mediastra backend | Streamlit has **no login by design** (Mediastra handles logins): it must never be public. Put it behind Mediastra's auth (e.g. ALB OIDC / internal network / VPN) |
| B2.2 | TLS on the ALB; HTTP → HTTPS redirect | |
| B2.3 | Restrict forwarded headers | The image runs uvicorn with `--forwarded-allow-ips "*"`; the rate limiter keys on the client IP, so the security group must only allow traffic from the ALB (or set `--forwarded-allow-ips` to the ALB subnet CIDR) |
| B2.4 | Secrets in **AWS Secrets Manager / SSM**, injected as env vars | `REFILLCARE_API_KEY`, `WHATSAPP_WEBHOOK_SECRET`, `XINNO_API_KEY`, `SUPABASE_ANON_KEY` (reads), `SUPABASE_SERVICE_ROLE_KEY` (rule edits only), `CLOUDINARY_*`, `DATABASE_URL`. Never bake `.env` into the image (`.dockerignore` already excludes it) |
| B2.5 | Production mode | `REFILLCARE_ENV=production` → API refuses to start serving without `REFILLCARE_API_KEY` (503) and rejects unsigned webhooks; docs off (`REFILLCARE_ENABLE_DOCS=false`); JSON logs |
| B2.6 | CORS | `CORS_ALLOW_ORIGINS` = the Mediastra web origin(s) only |
| B2.7 | Rotate keys | The Xinno / Supabase / Cloudinary keys in the developer's local `.env` were used during development — issue fresh production keys |

### B3. Multi-tenant / multi-store integration (Mediastra ↔ RefillCare)

| # | Task | Notes |
|---|---|---|
| B3.1 | **Register every store** before its first upload | `POST /api/v1/stores` `{store_id, tenant_id, store_name, store_contact, settings}`; `store_id` = 1–40 chars `[a-z0-9_-]`. Requests for unknown stores are rejected |
| B3.2 | **Mediastra backend sets `X-Store-Id`** from the logged-in user's tenant → store mapping; the browser must never choose it | RefillCare trusts the header (one shared `X-API-Key`). The tenant → store authorisation check lives in Mediastra |
| B3.3 | Per-user tokens (later) | Replace the shared API key with Mediastra JWT / SSO validation (planned item 9.2) |
| B3.4 | Per-store settings | Store name/contact for messages and per-store overrides (identity mode, rules) go in the store's `settings` |
| B3.5 | Per-store WhatsApp sender | Today one Xinno account / WABA number (`XINNO_WABA_NUMBER`) and template set serve the deployment. If each store sends from its own number, the credentials must become per-store settings (code change needed) |
| B3.6 | **Migrate the current store** (`store-1`, SQLite) into its RDS schema | Sales: `DATABASE_URL=<RDS> python scripts/import_history_to_db.py --store <id> --file <history> --name "NGOs Colony"`, then re-upload the 2026 batch files (or export `sales_transactions` from the SQLite file). **Also carry over the operational state**: opt-outs (STOP list — legally required), sent / approved reminder stages and WhatsApp delivery logs (so nobody is messaged twice), saved column mappings, and the model registry (or retrain on AWS) |
| B3.7 | The store id must stay the same everywhere (`store-1` today unless renamed during migration) | Store id = schema name = S3 prefix = model registry scope |

### B4. Data flow

| # | Task | Notes |
|---|---|---|
| B4.1 | **Daily sales push** from Mediastra POS to `POST /api/v1/sales/ingest` per store (yesterday's sales, or a rolling few days) | Re-sending the same file is rejected by checksum; overlapping files only add missing lines. Re-scoring is limited to the patients in the upload (≈ 7 s per day of data) |
| B4.2 | Push **before** the morning send window | Live sends are blocked when the latest sale is older than `REFILLCARE_MAX_DATA_LAG_DAYS` (2) or an upload failed to sync (`GET /api/v1/sales/data-status` → `ok_to_send`). Alert on `ok_to_send=false` |
| B4.3 | Retry failed syncs | `POST /api/v1/sales/batches/{id}/retry-sync` (the worker also retries) |
| B4.4 | Upload limits | `MAX_UPLOAD_MB`, `REFILLCARE_MAX_UPLOAD_ROWS`; dates are day-first; future dates and typo years are rejected |
| B4.5 | Wrong upload | `POST /api/v1/sales/batches/{id}/rollback` (removes its lines, reverts its reminders, re-scores the touched patients) |

### B5. WhatsApp

| # | Task | Notes |
|---|---|---|
| B5.1 | Configure the Xinno **webhook URL** → `https://<api>/api/v1/whatsapp/webhook` with `WHATSAPP_WEBHOOK_SECRET` (HMAC `X-Hub-Signature-256` or `X-Webhook-Token`) | Delivery status + STOP / START opt-out handling depend on it |
| B5.2 | Approved templates live in Xinno for every language used | `WHATSAPP_TEMPLATE_NAME`, `WHATSAPP_TEMPLATE_LANGUAGE`, `XINNO_IMAGE_TEMPLATE_NAME`, `XINNO_IMAGE_URL` |
| B5.3 | Keep `REFILLCARE_LIVE_REQUIRES_APPROVAL=true` for the first month | Pharmacist approves before live dispatch |
| B5.4 | Send window / rate | Dispatch in the morning after the daily upload; respect Xinno rate limits |

### B6. Monitoring and operations

| # | Task | Notes |
|---|---|---|
| B6.1 | Health checks | `/livez` (container), `/readyz` (DB), `/health` (details) |
| B6.2 | **Alarms** | `/health` `status != healthy` — includes `serving_model.degraded` (model failed to load → baseline); `data-status.ok_to_send=false` at send time; failed jobs (`GET /api/v1/jobs`); 5xx rate; RDS CPU / storage |
| B6.3 | Logs → CloudWatch (JSON, `REFILLCARE_LOG_FORMAT=json`) | Phone numbers are masked in every log line; set log retention (e.g. 90 days) |
| B6.4 | Monthly retrain per store | Worker schedule (`ML_RETRAIN_AFTER_DAYS`); keep `ML_AUTO_PROMOTE=false` until trusted — promote from the ML tab / `POST /models/{id}/promote` after reviewing the gate; rollback: `POST /models/rollback` |
| B6.5 | New store with little history | Needs a few months of sales before the model is meaningful; until then the baseline engine + pharmacist review are used (automatic) |
| B6.6 | Runbook | Wrong upload → batch rollback; bad model → model rollback; outage of Supabase → built-in rules keep working (classification falls back to the local catalogue) |

### B7. Compliance

| # | Task | Notes |
|---|---|---|
| B7.1 | Patient data (names, mobiles, medicines) is health data — India **DPDP Act 2023** | Consent for WhatsApp reminders, opt-out honoured (built in), data retention policy, access logging |
| B7.2 | Encryption at rest (RDS, S3) and in transit (TLS) | |
| B7.3 | Least-privilege DB user for the app; separate admin user for migrations | |

---

## Production environment variables (minimum)

```
REFILLCARE_ENV=production
DATABASE_URL=postgresql+psycopg://<user>:<pass>@<rds-host>:5432/refillcare
REFILLCARE_API_KEY=<secret>
WHATSAPP_WEBHOOK_SECRET=<secret>
CORS_ALLOW_ORIGINS=https://<mediastra-web-origin>
REFILLCARE_MODEL_BUCKET=<bucket>            REFILLCARE_MODEL_PREFIX=refillcare/models   REFILLCARE_MODEL_KEEP=3
REFILLCARE_LOG_FORMAT=json                  REFILLCARE_ENABLE_DOCS=false
REFILLCARE_LIVE_REQUIRES_APPROVAL=true      REFILLCARE_STALE_DATA_GUARD=true            REFILLCARE_MAX_DATA_LAG_DAYS=2
ML_AUTO_PROMOTE=false
XINNO_API_URL=…  XINNO_API_KEY=<secret>  XINNO_WABA_NUMBER=…  WHATSAPP_TEMPLATE_NAME=…  WHATSAPP_TEMPLATE_LANGUAGE=…
SUPABASE_URL=…  SUPABASE_ANON_KEY=<secret>  SUPABASE_SERVICE_ROLE_KEY=<secret, rule edits only>
REFILLCARE_STORE_ID=<default store for Streamlit>   # Streamlit serves one store per deployment
# Store decisions (A3), if approved:
REFILLCARE_RULE_PATH_B_MODE=blend  REFILLCARE_PATIENT_IDENTITY=name_mobile  REFILLCARE_ITEM_GROUPING=salt
```

Full list with explanations: [.env.example](.env.example).

---

## Go / no-go

Live sending may start when **all** are true:

1. `/health` → `healthy`, `serving_model.degraded: false` (A2)
2. `data-status` → `ok_to_send: true` on the day (A1, B4)
3. Pharmacist sign-off and one week of dry runs (A4, A5)
4. Webhook signed and opt-outs verified end-to-end (B5.1)
5. API key, production mode, private network for Streamlit (B2)
6. RDS backups on, alarms on (B1.2, B6.2)
7. Current store migrated **with** its opt-out list and sent-message history (B3.6)
