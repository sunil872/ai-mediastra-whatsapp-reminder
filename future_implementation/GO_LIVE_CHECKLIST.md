# RefillCare — Go-Live Checklist (multi-tenant, AWS)

Everything that must be done before RefillCare sends live WhatsApp reminders from the Mediastra cloud
application (QA `qa.mediastracloud.com`, then Live), where **each tenant has several stores**.

Three owners:
- **Part A — Store / Ops** (RefillCare owner): data, model and clinical sign-off for the first store (NGOs Colony, today `store-1`).
- **Part B — RefillCare team** (this repo): code changes listed in [PRODUCTION_READINESS_SCENARIO_1.md](PRODUCTION_READINESS_SCENARIO_1.md) (IDs such as T1, M1 refer to it).
- **Part C — Backend / DevOps team** (Mediastra): AWS, CI/CD, Mediastra UI and integration. Architecture: [PLAN_BACKEND_DEVOPS_TENANCY.md](PLAN_BACKEND_DEVOPS_TENANCY.md).

Decisions (4 Nov 2026): sales arrive as **file upload** through the Mediastra UI · pharmacists use the
**Mediastra UI, which calls the RefillCare API** · **one PostgreSQL schema per store** · **one WhatsApp sender per tenant**.

Status today (4 Nov 2026): one store on local SQLite; 885 tests pass; production model
`hgb-20260518-20261103071906` (56.7% within ±7 d vs baseline 53.6%). The code is store-aware but **not
tenant-safe yet** — Part B is required before a second store or any live send on AWS.

---

## Part A — Store / Ops

| # | Task | How | Done when |
|---|---|---|---|
| A1 | Upload the missing sales **16 Oct – today** (local) | Streamlit upload tab (run with `run_streamlit.bat`) | Data status `ok_to_send: true` |
| A2 | Retrain after A1 | `.venv\Scripts\python scripts\retrain_production.py` (backup + gate) | `/health` healthy, `serving_model.degraded: false`. First retrain done 3 Nov |
| A3 | Decide the prediction switches | `REFILLCARE_RULE_PATH_B_MODE=blend`, `REFILLCARE_PATIENT_IDENTITY=name_mobile`, `REFILLCARE_ITEM_GROUPING=salt` → identity cut-over (dry run first) → retrain. On AWS these become the store's `settings` | Decision recorded |
| A4 | Clinical sign-off | Pharmacist reviews eligible list, review queue and medication rules over one week of dry runs | Signed off |
| A5 | Dry-run week on **QA** after migration (C3.6) | Approval required; dispatch in dry run | Messages, Med-Sync bundles, opt-outs correct |
| A6 | Hand over to the backend team | The store's full sales history file(s), Supabase rules export, Xinno templates, local `enterprise.db` copy (for opt-outs / sent history) — through a secure channel, never git / email | Receipt confirmed |
| A7 | Local housekeeping after a verified migration | Delete `data/refillcare/processed/enterprise_backup_*.db`, `raw_backup/`; keep the last full backup until Live is stable | Disk clean |
| A8 | Patient data in git history | Purge (`git filter-repo`) before the repo leaves the team; everyone re-clones | Decision recorded (P5) |

---

## Part B — RefillCare team (code; details and file references in PRODUCTION_READINESS_SCENARIO_1.md)

| # | Task | IDs | Done when |
|---|---|---|---|
| B1 | **Tenant control plane**: `tenants`, `stores(store_key, tenant_id NOT NULL, mediastra_store_id)`; schema `s_<store_key>`; S3 prefix `{tenant}/{store_key}/`; tenant on jobs, logs, caches | T3, T5, T6, J3, P2 | AT1, AT2 |
| B2 | **Verified identity**: Mediastra service JWT (or hashed per-tenant keys); store must belong to the token's tenant; remove `?store=`; actor from the token | T1, T2, T4 | AT1, AT2 |
| B3 | **No legacy / store-1 models**: delete legacy bootstrap and `load_human_ml_bundle` serving; baseline for stores without a model; fix `/predictions/generate` | M1, M2, M3, M4 | AT3 |
| B4 | **Stateless**: S3 mandatory, no SQLite / CSV / parquet / log files, no mkdir at import, `.dockerignore` re-includes removed; production start-up validation | S1, S2, S6, C4, section 2 | AT10, AT11 |
| B5 | **Safe migrations**: remove `drop_prototype_tenant_ml_tables`; migrations as a deploy job with advisory lock; request path only checks the version | D-1, D-2 | Migration job idempotent on 2 replicas |
| B6 | **Async upload**: S3 → `202` → worker `ingest_batch`; idempotency key; date format per mapping profile | I1–I5, D-3, U6 | AT4 |
| B7 | **WhatsApp per tenant**: accounts table + secret refs; webhook per tenant; tenant-level consent; atomic send claim; today-only live sends; send window; `dispatch_due` job; kill switch | W1–W5, W7, W8, U4 | AT7, AT8, AT9 |
| B8 | **Time zone**: `store_today()` (IST default) everywhere; aware UTC timestamps | Z1, Z2, D-5 | AT9 |
| B9 | **Jobs**: unique dedupe index, heartbeat, attempts, timeouts, `default` + `ml` queues, retrain back-off | J1, J2, J3, J5, M6, M7 | AT5 |
| B10 | **ML hardening**: bounded caches keyed by store, unique temp downloads, keep ever-production artifacts, refuse library mismatch, `GIT_SHA`, per-store rule switches | S3, S4, M8–M10, M12, M13 | AT6 |
| B11 | **Rules and catalogue per tenant** in Postgres (shared base + overrides; version stamped on decisions / models); item catalogue from Mediastra's item master | M15, A9 (O3, O4) | Rule change by tenant A invisible to tenant B |
| B12 | **Streamlit as internal tool**: store-keyed caches, no hard-coded metrics / store name, ML actions enqueued, live send off | U2–U7 | Review passed |
| B13 | **Remove legacy code from the image** | 3.12, W6 | AT11 |
| B14 | **Postgres test suite green in CI** | C1 | AT12 |

---

## Part C — Backend / DevOps team

### C1. Infrastructure (AWS)

| # | Task | Notes |
|---|---|---|
| C1.1 | **RDS PostgreSQL** (Multi-AZ for Live), database `refillcare` (not Mediastra's own database) | Control plane in schema `refillcare_control`; one schema per store `s_<store_key>`. App user = DML only; migration user owns schemas |
| C1.2 | Automated backups + PITR (7–35 days); per-schema `pg_dump` runbook for restoring **one** store | PITR restores the whole instance (D-6) |
| C1.3 | **S3 bucket** (private, versioning, SSE-KMS, block public access) | Prefixes `{tenant}/{store_key}/uploads/`, `/models/`, `/reports/`. IAM: API role = Get / List; worker + ML worker = Get / Put / Delete; lifecycle rule on `uploads/` |
| C1.4 | **ECR + ECS (Fargate)** services from one code base: `api` (N replicas), `worker` (default queue), `ml-worker` (ml queue, ≥ 6 GB), `scheduler` (ECS scheduled task, exactly one), `migrate` (one-off task per deploy), optional `ops-ui` (Streamlit, internal) | Read-only root FS; `/tmp` writable; `TZ` not relied on (code uses store time zone) |
| C1.5 | Python 3.13 and [requirements.txt](requirements.txt) pins | Any scikit-learn / numpy change = retrain every store before deploying it |
| C1.6 | Sizing | API ≥ 2 GB (history cache bounded per B10), worker ≥ 2 GB, ml-worker ≥ 6 GB (835k-line store peaks ≈ 4 GB) |

### C2. Network and security

| # | Task | Notes |
|---|---|---|
| C2.1 | RefillCare API in private subnets; **only Mediastra's backend calls it** (D2) | No browser → RefillCare traffic; CORS not needed for tenants |
| C2.2 | Mediastra backend issues a **service JWT** per call: `tenant_id`, `store_id`, `sub` (user), `role`, `aud=refillcare`, short expiry; publish JWKS | Interim: per-tenant API keys (hashed in the control plane) |
| C2.3 | TLS everywhere; `--forwarded-allow-ips` = ALB subnet CIDR | C3 |
| C2.4 | Secrets Manager: `DATABASE_URL`, JWT / JWKS config, per-tenant Xinno API keys (referenced from `tenant_whatsapp_accounts`), webhook secrets, Supabase keys (until rules move to Postgres) | Rotate every key used in development (P4) |
| C2.5 | Rate limits **per tenant** at the gateway / WAF | In-process limiter is per replica (C5) |
| C2.6 | Streamlit ops UI (if deployed): internal network + SSO, operators only, live send disabled | U1 |
| C2.7 | `/livez`, `/readyz` for the load balancer; `/health` authenticated | C6 |

### C3. Mediastra integration

| # | Task | Notes |
|---|---|---|
| C3.1 | Tenant / store onboarding from Mediastra admin → `POST /api/v1/tenants`, `POST /api/v1/stores` (platform-admin token) | Creates control-plane rows; the migrate job creates the schema |
| C3.2 | **Refill screens in the Mediastra UI** (dashboard tab like Sales / Purchase / Analytics): due reminders, review queue, approve / reject, dispatch, opt-outs, upload, model status | All through the Mediastra backend → RefillCare API with the service JWT |
| C3.3 | **Sales upload** (D1): file chosen in Mediastra UI → Mediastra backend → `POST /api/v1/sales/uploads` (multipart or pre-signed S3) with `Idempotency-Key` → poll `GET /api/v1/sales/batches/{id}` | Column-mapping confirm screen when a new layout is detected |
| C3.4 | Daily upload **before** the morning send window | Live sends blocked when data is stale (`ok_to_send=false`) |
| C3.5 | Per-tenant WhatsApp account: WABA number, template names / language, API key secret; webhook URL `https://<api>/api/v1/whatsapp/webhook/{tenant_key}` with the tenant's secret | D4 |
| C3.6 | **Migrate `store-1`** into tenant "NGOs Colony": sales (`scripts/import_history_to_db.py --tenant … --store …`), decisions, cycles, stages, **opt-outs (legal)**, sent / delivery history, mapping profiles; retrain on AWS | Store key fixed once (schema = S3 prefix = registry scope) |

### C4. CI/CD

| # | Task | Notes |
|---|---|---|
| C4.1 | Blocking tests (remove `continue-on-error`), full suite also on a Postgres service container, `pip-audit` blocking | C1 |
| C4.2 | Build once per commit, tag by SHA, push to ECR, deploy to **QA** automatically, **Live** after manual approval | Same image in QA and Live |
| C4.3 | Deploy order: `migrate` task (wait for exit 0) → API → workers | Rollback = previous image tag (migrations backward compatible) |
| C4.4 | GitHub OIDC role, no static AWS keys; image scan (no data files, AT11) | |

### C5. Monitoring and operations

| # | Task | Notes |
|---|---|---|
| C5.1 | JSON logs to CloudWatch with `tenant_id`, `store_id`, `request_id`, `job_id`; retention 90 days | Phones masked |
| C5.2 | Alarms per store: model degraded, `ok_to_send=false` at send time, failed / stuck jobs, send failures, opt-out spikes, 5xx, RDS CPU / storage / connections | |
| C5.3 | Monthly retrain per store on the ml queue; `ML_AUTO_PROMOTE=false` until trusted | Rollback: `POST /api/v1/ml/models/rollback` |
| C5.4 | Runbook: wrong upload → batch rollback; bad model → model rollback; Xinno down → sends stay APPROVED and retry; restore one store → per-schema restore | |

### C6. Compliance (DPDP Act 2023)

| # | Task |
|---|---|
| C6.1 | Consent text for WhatsApp reminders agreed per tenant; STOP honoured across the tenant (B7) |
| C6.2 | Retention policy + purge job; patient data export / erasure by phone (P1) |
| C6.3 | Encryption at rest (RDS, S3 with KMS) and in transit; access logging |

---

## Production environment (per container; secrets from Secrets Manager)

```
REFILLCARE_ENV=production
DATABASE_URL=postgresql+psycopg://<app_user>:<secret>@<rds-host>:5432/refillcare
REFILLCARE_MODEL_BUCKET=<bucket>            # mandatory in production (B4)
REFILLCARE_CACHE_DIR=/tmp/refillcare
REFILLCARE_JWT_JWKS_URL=https://<mediastra>/.well-known/jwks.json   REFILLCARE_JWT_AUDIENCE=refillcare   # new (B2)
REFILLCARE_LOG_FORMAT=json                  REFILLCARE_ENABLE_DOCS=false
REFILLCARE_LIVE_REQUIRES_APPROVAL=true      REFILLCARE_STALE_DATA_GUARD=true    REFILLCARE_MAX_DATA_LAG_DAYS=2
REFILLCARE_DEFAULT_TIMEZONE=Asia/Kolkata    # new (B8); per-store override in store settings
REFILLCARE_WORKER_QUEUE=default|ml          # new (B9); one value per service
ML_AUTO_PROMOTE=false
SUPABASE_URL / SUPABASE_ANON_KEY            # until rules move to Postgres (B11)
# Not global any more (per tenant / store in the control plane): Xinno key, WABA number, templates, store name / contact, rule switches
```

Variables marked "new" do not exist in the code yet — they are part of Part B.

---

## Go / no-go for Live

Live sending may start for a tenant when **all** are true:

1. Part B done and acceptance tests AT1–AT12 ([PRODUCTION_READINESS_SCENARIO_1.md](PRODUCTION_READINESS_SCENARIO_1.md) §5) passed on QA.
2. `/health` for the store: healthy, `serving_model.degraded: false` (baseline is acceptable for a new store).
3. Data status `ok_to_send: true` on the day.
4. Pharmacist sign-off and one week of dry runs on QA (A4, A5).
5. Tenant's webhook verified end-to-end: STOP from a test phone blocks every store of the tenant.
6. JWT / per-tenant keys, private network, production mode, rotated secrets (C2).
7. RDS backups + per-store restore runbook tested, alarms on (C1.2, C5.2).
8. `store-1` migrated **with** its opt-out list and sent-message history (C3.6).
