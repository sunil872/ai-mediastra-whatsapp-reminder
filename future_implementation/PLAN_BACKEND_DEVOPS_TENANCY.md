# RefillCare — Multi-Tenant Architecture & AWS DevOps Plan

**For:** Mediastra backend team, DevOps / cloud team, RefillCare team.
**Scope:** running RefillCare inside the Mediastra cloud product (QA `qa.mediastracloud.com`, then Live) for
many tenants, each with several stores, on AWS RDS PostgreSQL + S3 + Docker (ECS).
*Rewritten 4 Nov 2026 after a code review; the earlier version had wrong endpoint names, wrong schema names,
an ALB path that the app does not serve, an iframe to an unauthenticated Streamlit, and a store-only (not tenant)
security model.*

Related: [PRODUCTION_READINESS_SCENARIO_1.md](PRODUCTION_READINESS_SCENARIO_1.md) (every code flaw, IDs used
below) · [GO_LIVE_CHECKLIST.md](GO_LIVE_CHECKLIST.md) (tick-list) · [requirements.txt](requirements.txt).

---

## 1. Decisions

| # | Decision |
|---|---|
| D1 | Sales arrive as a **file upload** (CSV / Excel) made in the Mediastra UI and forwarded by Mediastra's backend |
| D2 | Pharmacists use the **Mediastra UI**; Mediastra's backend calls the RefillCare API server-to-server. RefillCare's own `frontend/` and Streamlit are **not** exposed to tenants |
| D3 | **One PostgreSQL schema per store**, plus a control-plane schema for tenants / stores / WhatsApp accounts / consent / jobs |
| D4 | **One WhatsApp (Xinno / WABA) sender per tenant**; all its stores send from it; STOP applies tenant-wide |

---

## 2. Architecture

```
                         ┌──────────────────────────── Mediastra (existing) ────────────────────────────┐
 Pharmacist browser ───► │ Mediastra Web (Angular)  ──►  Mediastra backend (auth, tenant → store map)  │
                         └────────────────────────────────────────┬──────────────────────────────────────┘
                                                                  │ HTTPS, private
                                                                  │ Authorization: Bearer <service JWT: tenant_id, store_id, sub, role>
                                                                  ▼
            ┌──────────────────────── RefillCare (ECS, private subnets) ─────────────────────────┐
            │ api (FastAPI, N replicas)      worker (queue=default)      ml-worker (queue=ml)    │
            │ scheduler (ECS scheduled task, 1)   migrate (one-off per deploy)   ops-ui (opt.)   │
            └───────────┬───────────────────────────────┬───────────────────────────┬────────────┘
                        ▼                               ▼                           ▼
      RDS PostgreSQL  database "refillcare"     S3 bucket (SSE-KMS, versioned)      Xinno WhatsApp
        refillcare_control  (tenants, stores,     {tenant}/{store_key}/uploads/      one WABA per tenant
          whatsapp accounts, consent, jobs)       {tenant}/{store_key}/models/        ──webhook──► api
        s_<store_key>  (one schema per store:     {tenant}/{store_key}/reports/          /api/v1/whatsapp/webhook/{tenant_key}
          sales, decisions, cycles, stages,
          mappings, model registry)
```

Rules: no container keeps state on local disk (only `/tmp` caches); every request, job, cache key and log
line carries `tenant_id` and `store_key`; no default store in production.

---

## 3. Tenancy model (D3)

### 3.1 Control plane — schema `refillcare_control` (new, part of B1)

| Table | Key columns |
|---|---|
| `tenants` | `tenant_id` PK (Mediastra tenant id), `name`, `status`, `timezone` (default `Asia/Kolkata`), `created_at` |
| `stores` | `store_key` PK (RefillCare surrogate, `[a-z0-9]{1,24}`), `tenant_id` FK NOT NULL, `mediastra_store_id`, UNIQUE(`tenant_id`, `mediastra_store_id`), `name`, `contact`, `timezone`, `settings` JSONB (rule switches, identity mode, send window), `schema_version`, `status` |
| `tenant_whatsapp_accounts` | `tenant_id` FK, `waba_number` UNIQUE, `api_key_secret_arn`, `webhook_secret_arn`, `template_name`, `template_language`, `image_template` |
| `whatsapp_consent` | (`tenant_id`, `phone_key`) PK, `status` (opted_in / opted_out), `source`, `changed_at` |
| `jobs` | today's `JobModel` + `tenant_id`, `queue`, `priority`, `heartbeat_at`, `max_attempts`; partial unique index (`store_key`, `dedupe_key`) where active |
| `medication_rules`, `item_catalogue` | shared base rows (`tenant_id` NULL) + tenant overrides (O3, O4) |

Today's `refillcare_stores` (`database/control_models.py`) becomes `stores`; `store_id` is no longer a global
free-text key.

### 3.2 Store schemas

- Name: `s_<store_key>` (≤ 63 chars, no collisions — today `store_<id>` maps `a-b` and `a_b` to the same schema and ignores the tenant).
- Content: the existing per-store tables (`sales_transactions`, upload batches, `refill_decisions`, reminder cycles / stages, mapping profiles, `ml_model_versions`, `ml_evaluation_runs`, outcomes). Opt-outs move to the tenant-level `whatsapp_consent`.
- Selection at runtime: SQLAlchemy `schema_translate_map` from the verified `store_key` (already implemented in `database/connection.py`). Raw SQL must be schema-qualified (D-4).

### 3.3 Migrations

- A **`migrate` ECS task** runs on every deploy: for the control plane and then every active store schema, under `pg_advisory_lock(hashtext(schema))`, records `stores.schema_version`. The deploy waits for exit code 0.
- The API / workers only **check** `schema_version` (no DDL at request time; app DB user has no CREATE right).
- Migrations are additive and backward compatible with the previous image (rollback = previous tag). The destructive `drop_prototype_tenant_ml_tables` is removed (D-1).
- New store: control-plane row → `migrate --store <key>` (or the onboarding endpoint enqueues it).

---

## 4. Security and the Mediastra ↔ RefillCare contract (D2)

### 4.1 Authentication

- Mediastra's backend signs a **short-lived service JWT** for each call: `iss=mediastra`, `aud=refillcare`, `tenant_id`, `store_id` (Mediastra id), `sub` (user), `role` (`pharmacist`, `store_admin`, `platform_admin`), `exp` ≤ 5 min. RefillCare validates with the JWKS URL.
- RefillCare maps (`tenant_id`, `store_id`) → `store_key` and **rejects** a store not owned by that tenant (403). The `X-Store-Id` header and `?store=` parameter are removed for production.
- Actor in every audit row = JWT `sub` (replaces the free-text `X-Actor`).
- Interim (if JWT is late): per-tenant API keys, stored hashed in `tenants`, plus `X-Store-Id` checked against the tenant.
- Browsers never call RefillCare; CORS stays closed for tenant origins. RefillCare's `frontend/` is not deployed.

### 4.2 Roles

| Role | Can |
|---|---|
| `pharmacist` | read reminders / review queue / decisions; approve, reject, dispatch; opt-out / opt-in |
| `store_admin` | + upload sales, column mapping, rollback, store settings |
| `platform_admin` | + tenants / stores onboarding, ML promote / rollback across stores, medication-rule base set |

### 4.3 Endpoints Mediastra uses (existing paths; ★ = to be added in Part B)

| Area | Endpoints |
|---|---|
| Onboarding | ★`POST /api/v1/tenants` · `POST /api/v1/stores` (platform admin; `tenant_id` immutable) · `GET /api/v1/stores/current` |
| Sales upload (D1) | `POST /api/v1/sales/column-mapping/detect` · `POST /api/v1/sales/column-mapping` · `GET/DELETE /api/v1/sales/column-mapping/profiles` · `POST /api/v1/sales/preview` · ★`POST /api/v1/sales/uploads` (async, `Idempotency-Key`, returns `202 {batch_id}`; replaces the synchronous `POST /api/v1/sales/ingest`) · `GET /api/v1/sales/batches` · ★`GET /api/v1/sales/batches/{id}` · `POST /api/v1/sales/batches/{id}/rollback` · `POST /api/v1/sales/batches/{id}/retry-sync` · `GET /api/v1/sales/data-status` |
| Reminders | `GET /api/v1/reminders/daily` · `GET /api/reminders/today` · `GET /api/v1/reminders/months` · `GET /api/reminders/{id}` · `POST /api/reminders/{id}/approve` · `/reject` · `/dispatch` · `GET /api/v2/med-sync/bundles` · `GET /api/v1/reminders/export-csv` / `export-json` |
| Review | `GET /api/v1/review/queue` · `GET /api/v1/review/summary` · `GET /api/refill-decisions` · `GET /api/customers/{id}/refill-history` |
| WhatsApp | `GET /api/v1/whatsapp/candidates` · `POST /api/v1/whatsapp/preview` · `/send-single` · `/dispatch` (live only for today's stages inside the send window) · `GET /api/customers/opt-outs` · `POST /api/customers/{id}/opt-out` · `/opt-in` |
| Webhook (from Xinno) | ★`POST /api/v1/whatsapp/webhook/{tenant_key}` (signature with the tenant's secret; replaces the global `POST /api/v1/whatsapp/webhook`, which returns 400 in production today) |
| ML (platform admin / ML team) | prefix `/api/v1/ml`: `GET /plan` · `/models` · `/models/{id}/gate` · `/evaluations` · `/settings` · `/reports/{kind}` · `/outcomes/summary` · `POST /retrain` · `/rules-ab` · `/rolling` (enqueue jobs) · `POST /models/{id}/promote` · `/models/rollback` · `GET /api/v1/jobs` · `/api/v1/jobs/{id}` |
| Analytics | `GET /api/v1/analytics/kpi` · `/transaction-types` · `/consensus-regimen` |
| Health | `GET /livez` (`{"status":"alive"}`) · `GET /readyz` (DB) — public to the load balancer · `GET /health` (authenticated, per store) |

The app serves `/api/...` and `/api/v1/...` at the root (no `root_path`). Route the RefillCare **host**
(e.g. `refillcare.internal`) to the API target group; do **not** use an ALB path prefix such as
`/api/refillcare/*` (ALB cannot strip it).

---

## 5. Data flows

### 5.1 Sales upload (D1)

1. Store admin picks the file in the Mediastra UI → Mediastra backend → `POST /api/v1/sales/column-mapping/detect` (first time a layout is seen: confirm screen) → `POST /api/v1/sales/uploads` with `Idempotency-Key`.
2. API stores the file at `s3://<bucket>/{tenant}/{store_key}/uploads/{batch_id}.<ext>`, creates the batch (`queued`), returns `202`.
3. `worker` (queue `default`) runs `ingest_batch` under the store's advisory lock: parse → only new lines (line keys) → re-score affected medicines → sync reminder cycles.
4. Mediastra polls `GET /api/v1/sales/batches/{id}` (`queued` → `running` → `done` / `failed`); same file again → `duplicate`.
5. Upload before the morning send window; live sends are blocked when `data-status.ok_to_send=false`.

### 5.2 Daily reminders

`scheduler` (once per hour) enqueues per store: `expire_stages`, `dispatch_due` (inside the store's send window,
today in the store's time zone, only approved stages when approval is required), and the monthly `retrain`
(queue `ml`, back-off after a failed gate). No scheduler runs inside API or worker containers.

### 5.3 WhatsApp (D4)

- Credentials, templates and webhook secret from `tenant_whatsapp_accounts` (secrets via ARN); store name / contact from `stores` — no global `XINNO_*` / `MEDICAL_STORE_*` in production.
- Send = atomic claim (`APPROVED → SENDING` with a claim token) → Xinno call → `SENT` + unique live-log row; a reconciler resolves stale `SENDING` rows; never re-send after an ambiguous timeout without checking delivery status.
- Webhook `/api/v1/whatsapp/webhook/{tenant_key}`: delivery status + STOP / START → `whatsapp_consent` (tenant-wide); every send checks consent.

### 5.4 Models

Training only in `ml-worker`; artifacts `s3://<bucket>/{tenant}/{store_key}/models/{version}.joblib` with
SHA-256, library versions and `GIT_SHA` in the registry row. API replicas read the production version id
(≤ 60 s TTL per store), download to `/tmp/refillcare/models/...` and verify before loading. A store without a
validated model is served by `baseline-median-v1` — never by another store's or a legacy model.

---

## 6. AWS infrastructure

| Component | Setting |
|---|---|
| VPC | API, workers, RDS in private subnets; ALB internal (Mediastra backend in the same VPC or via PrivateLink / peering) |
| RDS | PostgreSQL 16+, Multi-AZ (Live), automated backups + PITR, performance insights; parameter `max_connections` sized for replicas × pool (`DB_POOL_SIZE` / `DB_MAX_OVERFLOW`); RDS Proxy recommended |
| DB users | `refillcare_migrator` (owns schemas, used only by `migrate`), `refillcare_app` (DML on control + store schemas) — no `GRANT ALL`, no `CREATEDB` |
| S3 | private, versioning, SSE-KMS, block public access, lifecycle on `uploads/` (e.g. 90 days) |
| IAM | `api` role: S3 Get / List; `worker` / `ml-worker`: Get / Put / Delete on the bucket prefix; Secrets Manager read of the needed ARNs only |
| ECS (Fargate) | `api` 2+ tasks (1 vCPU / 2–4 GB), `worker` 1–2 (2 GB), `ml-worker` 1 (2 vCPU / 6–8 GB), `scheduler` scheduled task, `migrate` one-off; read-only root FS, `/tmp` volume |
| Secrets | Secrets Manager → task env (`DATABASE_URL`, JWKS config) and per-tenant Xinno secrets read at runtime by ARN |
| Edge | WAF + per-tenant rate limits at the gateway; TLS; uvicorn `--forwarded-allow-ips` = ALB CIDR |
| Logs / metrics | CloudWatch JSON logs (90 days) with `tenant_id` / `store_key` / `request_id` / `job_id`; metrics and alarms per store (checklist C5.2) |

Images (from one repo, multi-stage Dockerfile): `refillcare-core` (API / worker / ml-worker / scheduler / migrate —
[requirements.txt](requirements.txt)), `refillcare-ops-ui` (Streamlit, optional, internal). Code only — no
`data/`, models, catalogue or `.env` in any image (`.dockerignore` re-includes removed).

---

## 7. CI/CD (GitHub Actions)

```
on: push / pull_request
jobs:
  test:      python 3.13; pip install -r requirements-dev.txt; pytest -q            (blocking)
  test-pg:   services: postgres:16 → REFILLCARE_PG_TEST_URL=… pytest -q             (blocking, full suite on Postgres)
  audit:     pip-audit -r future_implementation/requirements.txt                    (blocking)
  build:     needs [test, test-pg, audit]; docker build (multi-stage); image scan (no *.joblib/*.parquet/*.csv/*.db/.env)
             tag = git SHA; push to ECR (GitHub OIDC role — no static AWS keys)
  deploy-qa: on main: run ECS task "migrate" (wait, exit 0) → update api, worker, ml-worker, scheduler → smoke tests
  deploy-live: manual approval; same image tag; same order
```

Rollback: redeploy the previous SHA (migrations are backward compatible). Never deploy `:latest`.

---

## 8. Rollout

| Step | What | Exit criterion |
|---|---|---|
| 1 | RefillCare Part B (code) + backend platform (infra, JWT, CI/CD) in parallel | Acceptance tests AT1–AT12 green on QA |
| 2 | Migrate `store-1` into tenant "NGOs Colony" on QA (sales, decisions, cycles, stages, opt-outs, sent history, mappings); retrain on AWS | Same eligible-list counts as local ±1% |
| 3 | Add a second store (same tenant) and a store of a second tenant on QA | Isolation tests pass; STOP tenant-wide |
| 4 | One week of dry runs on QA with the pharmacist | Sign-off |
| 5 | Live for tenant 1 with approval before every send; then tenant by tenant | Go / no-go list in the checklist |

---

## 9. Post-deploy smoke test (each environment)

1. `GET /livez`, `GET /readyz` → 200.
2. Authenticated `GET /health` for one store → `healthy`, `serving_model.degraded=false` (or baseline for a new store).
3. Token of tenant A + store of tenant B → 403.
4. Upload a 1-day test file to a test store → `202` → batch `done`; again → `duplicate`.
5. `POST /api/v1/whatsapp/dispatch` in dry run → candidates listed, nothing sent.
6. Signed test webhook STOP for a test phone → consent `opted_out` for the tenant.
7. `GET /api/v1/jobs` → no failed / stuck jobs.
