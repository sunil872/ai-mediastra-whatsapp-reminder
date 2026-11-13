# Mediastra platform guide for the RefillCare team

Audience: the AI/ML engineers who built [ai-mediastra-whatsapp-reminder](https://github.com/sunil872/ai-mediastra-whatsapp-reminder) (branch `master`) and will add it to this application with Cursor or Claude Code.

This file describes the Mediastra pharmacy billing app as it exists in this repository on 12 Nov 2026. It also says exactly where a refill-reminder screen belongs, and which parts of `FULLSTACK_INTEGRATION_HANDOVER` must not be copied.

No application code was changed to produce this document.

---

## 1. How to use this file

1. Read sections 3, 4, and 8 before writing any Angular or NestJS code. Those sections correct the handover.
2. Keep the Python service as the clinical engine. Mediastra does not reimplement prediction, med-sync, or the 7-stage lifecycle.
3. The browser talks only to the NestJS API, with the user's JWT. NestJS talks to RefillCare with `X-API-Key`. The Angular app must not call port 8000 and must not hold the RefillCare API key.
4. When prompting Cursor, point it at this file and at the files listed in section 12. Tell it to follow the Payment Alerts screen pattern, and to leave prediction code in the Python repo.

---

## 2. The two systems

| | Mediastra (this repo) | RefillCare (Python repo) |
|---|---|---|
| Job | Pharmacy billing ERP: masters, sales, purchase, stock, print, users, permissions, existing WhatsApp sends | Predict chronic refill dates and run the reminder lifecycle |
| Language | TypeScript | Python 3.13 |
| API | NestJS 10, Express, port 3000 | FastAPI 0.141, Uvicorn, port 8000 |
| UI | Angular 19 (Fuse admin template) | Streamlit today. The new screen is Angular inside Mediastra |
| Data access | Prisma 5.22 | SQLAlchemy 2 + psycopg 3 |
| Database | One PostgreSQL 15 database, schema `public`, `tenantId` + `storeId` on rows. Store-mode installs use SQLite | Same PostgreSQL server is allowed. RefillCare tables stay in `refillcare_control` and `s_<store_key>`. A second SQLite file is only for the standalone Python app |
| ML | None in this repo | scikit-learn 1.7, HistGradientBoosting, pandas, joblib. Optional XGBoost for offline benchmarks |
| WhatsApp already shipping | Xinno gateway, tenant settings in `WhatsAppSettings` | Xinno gateway, its own sender and consent table |

RefillCare source that matters:

- `api/main.py` — HTTP routes
- `api/ml_routes.py` — prefix `/api/v1/ml`
- `api/security.py` — `X-API-Key`, `X-Store-Id`, optional `X-Actor`
- `api/schemas.py`, `api/services.py`
- `refillcare/` — clinical engine (`unified_engine.py`, `reminder_lifecycle.py`, `med_sync.py`, `template_formatter.py` as named in the handover)

Clinical behaviour already built in Python (do not rebuild it in Nest or Angular):

- Chronic vs acute classification. Acute drugs get no reminder.
- Path A: 3 or more purchases, ML cadence model.
- Path B: 1–2 purchases, days-of-supply from pack size, quantity, free quantity, and estimated daily dose.
- Exception queue for anomalies, paediatric flags, and patients more than 180 days past the expected refill.
- Stages at −7, −3, −1, 0, +2, +5, +45 days.
- Repurchase cancels the remaining stages.
- Med-sync bundles medicines due within ±4 days into one message.
- STOP / START opt-out is tenant-wide.

---

## 3. Facts that override the handover sample

The handover's sample Nest controller and Angular component will not work in this app. Use this table.

| Handover sample | What this repository actually does |
|---|---|
| Menu file `frontend/src/app/core/navigation/navigation.data.ts` | That file does not exist. The sidebar is `GET /navigation/user/:userId`. The seed is `backend/prisma/seeds/postgres/navigation-structure-data.json`. `frontend/src/app/mock-api/common/navigation/data.ts` is an unused Fuse mock. |
| Route `/utilities/mini-apps/refill-reminders` | There is no `mini-apps` URL segment. Mini Apps is a menu group. Its pages are root routes such as `/payment-alerts` and `/forecast`. |
| Angular calls `/api/v1/reminders/daily` with `X-Store-Id` | The browser calls Nest only. Global prefix `app.setGlobalPrefix('api')` is commented out in `backend/src/main.ts`. Nest routes are `/payment-alerts`, `/sales`, `/auth`, and so on. A few controllers hardcode an `api/` prefix themselves (`api/whatsapp`). |
| Header `X-Store-Id: NEHA STORE` from the browser | The Angular interceptor sends `Authorization: Bearer <jwt>`, `tenant-id: <uuid>`, and `store-id: <uuid>`. Header name is `store-id`, lowercase, and the value is the Store UUID from `sessionStorage.active_store_id`. |
| `X-API-Key` in the Angular app | That key stays on the Nest server (`REFILLCARE_API_KEY`). `api/security.py` requires it on every `/api/*` route because responses contain patient names and phone numbers. |
| Approve URL `/api/reminders/{id}/approve` mixed with `/api/v1/...` | Both families exist on the Python app. See section 9. Nest should proxy one chosen family and hide the split from Angular. |
| RefillCare container is already beside the Nest task in ECS | It is not deployed from this repo. This repo's workflows deploy only the Nest image and the Angular static build. |
| Frontend served by `http-server` in production | That is only `frontend/Dockerfile` for local Compose. Production and current QA upload the Angular build to S3 and serve it with CloudFront. |

Do not copy the sample API key from the handover into git, into Angular, or into a committed `.env`.

---

## 4. Where the new screen goes

### 4.1 Menu today

Utilities (`moduleKey: utilities` in the navigation seed) has two features:

1. **Settings** — print settings, transaction settings, report settings, hidden template builders. Routes under `/utilities/...` and `/transaction-settings`.
2. **Mini Apps** — feature key `payment-alerts` (the key is historical; the visible name is Mini Apps).

Mini Apps pages in the seed:

| Order | Label | Page key | Route |
|---|---|---|---|
| 1 | Payment Alerts | `payment-alerts.page` | `/payment-alerts` |
| 2 | Vendor Comparison | `vendor-comparison.page` | `/vendor-pricing/comparison` |
| 3 | Purchase Order Summary | `purchase-order-summary.page` | `/purchase-order-summary` |
| 4 | Forecast | `forecast.page` | `/forecast` |
| 5 | Marketplace | `Marketplace.utlities` | `/marketplace` |

Mini Tenant Admin and Mini Store Admin do not see that full tree. `flattenMiniAppAdminNavigation` in `backend/src/navigation/navigation.service.ts` lifts only these four page keys, in this order: `forecast.page`, `payment-alerts.page`, `vendor-comparison.page`, `purchase-order-summary.page`. Marketplace is omitted. Their home is `/mini-tenant-dashboard`, whose four cards are hardcoded in `frontend/src/app/modules/platform/mini-tenant-dashboard/mini-tenant-dashboard.component.ts`.

**Product request:** add refill reminders as the 5th mini app.

**What that means in this repo:**

- It becomes the 5th card on the mini-tenant dashboard (there are 4 cards today).
- It becomes a new page under the Mini Apps feature in the navigation seed.
- Its page key must be added to `targetPageKeys` in `flattenMiniAppAdminNavigation`, or mini-app admins will not see it.
- On the full sidebar it will sit next to Marketplace, so the Mini Apps group will show six entries. Marketplace is already the fifth menu row. Do not replace Marketplace and do not put the screen under `/utilities/print-settings`.

**Route to create:** `/refill-reminders`. Same style as `/payment-alerts` and `/forecast`. Suggested page key: `refill-reminders.page`.

Changing the JSON file does not update a running database. Navigation rows are seeded (`npm run seed:navigation` in `backend/`). Existing environments need that seed, or an equivalent data update, before the menu item appears. `MenuPermissionGuard` wraps authenticated routes in `frontend/src/app/app.routes.ts`, so a route with no matching navigation permission will not open for a normal user.

### 4.2 Screens the Angular app should own

Map these to RefillCare. Do not rebuild the ML.

| Screen | User action | Python routes to proxy |
|---|---|---|
| Today's queue | List due stages, approve, reject, send one message | `GET /api/v1/reminders/daily`, `POST /api/reminders/{id}/approve`, `POST /api/reminders/{id}/reject`, `POST /api/v1/whatsapp/send-single` |
| Preview | Show the exact WhatsApp text before send | `POST /api/v1/whatsapp/preview` |
| Exception queue | Pharmacist reviews ineligible or risky rows | `GET /api/v1/review/queue`, `GET /api/v1/review/summary` |
| Upload / sync status | Only if sales still arrive as a file. Prefer a Nest export from live bills (section 7) | `POST /api/v1/sales/ingest`, `GET /api/v1/jobs/{job_id}`, `GET /api/v1/sales/data-status` |
| Send log | What was dispatched | `GET /api/v1/whatsapp/delivery-log` |

KPI and model-admin routes (`/api/v1/analytics/*`, `/api/v1/ml/*`) can wait. They are not required for the first mini-app screen.

---

## 5. Repository map

```
pharma-billing-saas/
  backend/                 NestJS API, Prisma schema, seeds, Docker
  frontend/                Angular 19 app, optional Electron shell
  .github/workflows/       CI and AWS deploy
  docs/                    this file
```

Backend modules live in `backend/src/<module>`. There are about 90 feature folders (sales, purchase, item, customers, stores, auth, payment-alerts, einvoice, and others). A new refill proxy belongs in a new folder `backend/src/refill-reminders/` registered from `backend/src/app.module.ts`. There is no refill module today. The word "refill" in purchase code means form prefill, not this feature.

Frontend features live in `frontend/src/app/modules/`. Authenticated URLs are assembled in `frontend/src/app/routing/authenticated-feature-children.routes.ts` from:

- `dashboard-integrations.routes.ts`
- `masters.routes.ts`
- `transactions.routes.ts`
- `reports.routes.ts`
- `utilities.routes.ts` — also mounts `/forecast`
- `transaction-settings.routes.ts` — mounts `/payment-alerts`

---

## 6. Runtime architecture

```
Browser (Angular 19)
  Authorization: Bearer <jwt>
  tenant-id: <tenant uuid>
  store-id:  <store uuid>
        |
        | HTTPS
        v
NestJS 10 on ECS (port 3000)
  JwtAuthGuard (global)
  UserTenantInterceptor (store access check, AsyncLocalStorage)
  Prisma 5.22  ----------------->  PostgreSQL 15, schema public
        |
        | server-side HTTP only
        | X-API-Key, X-Store-Id: <refillcare store key>, X-Actor: <mediastra user id>
        v
RefillCare FastAPI (port 8000)
  schemas refillcare_control and s_<store_key>
  Xinno WhatsApp (whatsapp.xinno.in)
```

Local ports: Angular `ng serve` on 4200, Nest on 3000, RefillCare on 8000 (`/docs` is its OpenAPI page, `/livez` and `/readyz` are health).

### 6.1 Environments

| | UI | API |
|---|---|---|
| Production (`main`) | `https://app.mediastracloud.com` | `https://api.mediastracloud.com` |
| QA (`qa` branch, workflows `*_qa_yg`) | `https://qa.mediastracloud.com` | `https://qa-api.mediastracloud.com` |
| Local | `http://localhost:4200` | `http://localhost:3000` |

Region `ap-south-1`. GitHub Actions assumes IAM role `arn:aws:iam::535276128727:role/github-actions-prod` for production. Backend image goes to ECR, then ECS. A one-off Fargate task runs `npx prisma migrate deploy` before the service update. Frontend `ng build` is synced to S3 and CloudFront is invalidated. Legacy EC2 SSH workflows (`cd_backend_qa.yml`, `cd_frontend_qa.yml`) are manual leftovers.

`frontend/src/environments/environment.ts` points `apiUrl` at `http://localhost:3000`. `environment.prod.ts` in git currently points at the QA API. The production workflow rewrites `apiUrl` to `PROD_API_URL` (`https://api.mediastracloud.com`) during the build. Feature code must use `environment.apiUrl`, never a hardcoded host.

CORS allowed origins are listed in `backend/src/main.ts`. Local `http://localhost:4200` is allowed. Production origins in that list are `http://app.mediastracloud.com` while the live site is HTTPS. Add `https://app.mediastracloud.com` when the refill UI is deployed, or browser calls from production will fail.

### 6.2 Two database modes

`PrismaRouterService` (`backend/src/prisma-router.service.ts`):

- Cloud (`APP_MODE` unset or not `store`): one PostgreSQL database, `DATABASE_URL`.
- Store / offline (`APP_MODE=store`): SQLite, `STORE_DB_URL`.

RefillCare's Postgres schemas apply to cloud mode. The offline SQLite build does not have `refillcare_control`. The first integration target is cloud.

Prisma manages only models in `backend/prisma/schema.prisma` (schema `public`). It has no `@@schema` for RefillCare. Creating refill tables inside `public` without adding them to `schema.prisma` will make `prisma migrate` treat them as drift. Keep RefillCare tables in non-public schemas, as the Python service already does.

---

## 7. Multi-tenant model and the sales feed

One database. Every business row carries `tenantId` and, for store data, `storeId`. Both are UUIDs.

| Model | Table | What it is |
|---|---|---|
| `Tenant` | tenant | Pharmacy group. `tenantName`, `companyName` |
| `OrganizationMaster` | organization | Middle level between tenant and store |
| `Store` | `Store` | One shop. `storeName`, `code` (unique), `contactPhone`, `companyName` |
| `User` | `User` | Login. Phone is unique. Tenant and store are not columns on User |
| `UsersOnboardingRoles` | | Assigns user + role + tenant + organization + store |
| `Role` | | `role` (display name), `roleKey`, `level` (0 = super admin, 1 = tenant admin, 2 = store admin, 3+ lower) |

A user can belong to several stores. The active store is `sessionStorage.active_store_id`. Nest checks that the user may access that store (`validateUserStoreAccess` in the tenant interceptor). Super admin can switch tenant. Everyone else is pinned to their own tenant.

Level rule (`LevelFilterService`): a user may administer users and roles whose level number is strictly greater than their own (lower privilege). Level 0 bypasses that filter. This rule is for user and role admin. It does not filter sales rows. Store access is the check that matters for refill data.

### 7.1 Pharmacy sales (the 11 clinical fields)

Live bills are `SalesInvoices` plus `SalesInvoiceItems`. `InvoiceItem` is subscription billing, not a pharmacy bill. Do not read it.

RefillCare's upload fields, and the Prisma source for each:

| RefillCare field | Mediastra source | Notes |
|---|---|---|
| `invoice_date` | `SalesInvoices.invoiceDate` | `timestamp` |
| `invoice_number` | `SalesInvoices.invoiceNumber` | Also `invoicePrefix`, `invoiceSuffix`, `invoiceSequence`, `financialYear`. Retail vs wholesale is not the literal prefix `S0/` vs `SB/` until product confirms the rule. Candidates: `invoicePrefix`, `billingMode`, `Customers.customerType` |
| `quantity` | `SalesInvoiceItems.quantity` | `Decimal(10,2)` |
| `freeQuantity` | `SalesInvoiceItems.freeQuantity` | default 0 |
| `customerId` | `SalesInvoices.customerId` | UUID, nullable |
| `customerName` | `Customers.customerName` | Bill can also point at `FamilyMember` via `familyMemberId`. Patient name may be the family member, not the account holder |
| `MOBILE_NO` | `Customers.phone1` | Required, varchar(12), unique per tenant. `phone2` is optional. Family member phone is `FamilyMember.number` |
| `itemId` | `SalesInvoiceItems.itemId` | UUID |
| `itemCode` | `Item.itemCode` | **Integer**, unique per tenant, not a string code |
| `itemName` | `Item.itemName` | Not stored on the invoice line. Join `Item` |
| `packing` | `Item.packing` | Free text such as `1X10`. Not a numeric pack size column |

There is no daily-dose column on `SalesInvoiceItems`. Path B dosage has to come from RefillCare's own classifier or from `Generic.dosageFrequency` (text, via `ItemGeneric`). Do not invent a dose from `quantity`.

Ignore prices, GST, margins, and `totalAmount` when building the clinical extract. The handover already drops them. Filter `SalesInvoices.isActive = true` and always filter `tenantId` + `storeId`.

Store identity bridge (required before any call):

| Mediastra | RefillCare |
|---|---|
| `Store.id` UUID in header `store-id` | `X-Store-Id` store key, today a name such as `NEHA STORE` |
| `Store.code`, `Store.storeName` | Candidate keys. They are not the same string as the UUID |

Nest must translate UUID → RefillCare store key. The Angular app keeps sending the UUID it already has.

Message template fields the Python formatter fills from store data:

- `{{2}}` and `{{6}}` pharmacy name → `Store.storeName` or `Store.companyName`
- `{{5}}` phone → `Store.contactPhone`
- `{{1}}` patient → customer or family member, per the decision in section 13

---

## 8. Auth flow the new UI must keep

1. User signs in. `AuthService` stores the access token in `localStorage.accessToken`.
2. `frontend/src/app/core/auth/auth.interceptor.ts` adds `Authorization`, `tenant-id`, and `store-id` on every call except public auth URLs.
3. Nest `JwtStrategy` validates the bearer token. JWT claims include `userId`, `tenantId`, `storeId`, `organizationId`, `isSuperAdmin`, `minRoleLevel`. Roles are not inside the token. They are loaded from `UsersOnboardingRoles`.
4. `UserTenantInterceptor` resolves the requested store: header `store-id`, then body `storeId`, then query `storeId`, then the JWT store. It rejects a store the user cannot access.
5. `JwtAuthGuard` is global. `@Public()` opts out. `PermissionGuard` runs only when a route has `@RequirePermission()`.

A refill controller without the global guard would expose patient phone numbers. Do not mark it `@Public()`.

Angular services use `HttpClient` and `environment.apiUrl`. Example shape, matching Payment Alerts analytics:

```ts
this.http.get<ReminderRow[]>(`${environment.apiUrl}/refill-reminders/daily`)
```

The interceptor attaches the three headers. The service does not set `X-Store-Id` or `X-API-Key`.

Token refresh on 401 is `POST /auth/sign-in-with-token`. Do not add a second auth client.

---

## 9. RefillCare HTTP surface (from `api/main.py`, branch `master`)

Base URL inside the VPC: `http://<refillcare-host>:8000`.

Auth (`api/security.py`):

- `X-API-Key` required on `/api/*`
- `X-Store-Id` selects the store schema
- `X-Actor` optional, max 64 chars. Pass the Mediastra `userId` so the Python audit trail shows who approved a send
- Webhook: `X-Hub-Signature-256` or `X-Webhook-Token`

Health (no clinical data): `GET /livez`, `GET /readyz`, `GET /health`.

Routes the first Nest proxy needs:

```
GET  /api/v1/reminders/daily
GET  /api/v1/reminders/months
GET  /api/v1/review/queue
GET  /api/v1/review/summary
POST /api/reminders/{reminder_id}/approve
POST /api/reminders/{reminder_id}/reject
POST /api/reminders/{reminder_id}/dispatch
POST /api/v1/whatsapp/preview
POST /api/v1/whatsapp/send-single
GET  /api/v1/whatsapp/delivery-log
POST /api/v1/whatsapp/webhook
```

Also present, for a later slice: sales ingest and jobs (`/api/v1/sales/*`, `/api/v1/jobs/*`), predictions (`/api/v1/predictions/*`), med-sync (`/api/v2/med-sync/bundles`), medication rules (`/api/v1/medications/*`), ML lifecycle (`/api/v1/ml/*` in `ml_routes.py`), opt-out (`/api/customers/{id}/opt-out`).

The handover's `POST /api/v1/sales/uploads` is not the route in `main.py`. Ingest is `POST /api/v1/sales/ingest`. Confirm the multipart field name in `api/main.py` before wiring an upload screen.

Nest proxy sketch (implement in `backend/src/refill-reminders/`, do not expose the key):

- `RefillRemindersModule` imports `HttpModule` from `@nestjs/axios` (it is not global).
- Register the module in `app.module.ts`.
- Controller prefix `refill-reminders`, guarded like every other business controller.
- Service reads `REFILLCARE_API_URL` and `REFILLCARE_API_KEY` from config.
- Map `request.user` store UUID to the RefillCare store key, then set `X-Store-Id` and `X-Actor`.

`ForecastModule` (`backend/src/forecast`) is a poor template: it is not imported in `app.module.ts`, so `POST /forecast/upload` is not live. Copy the Payment Alerts module registration instead.

---

## 10. WhatsApp already in Mediastra

Three senders exist. RefillCare is a fourth, and it must stay on its own template.

| Code | Registered | Gateway | Used for |
|---|---|---|---|
| `backend/src/payment-alerts/whatsapp` | Yes. Controller prefix `api/whatsapp` | Xinno HTTP, and a `whatsapp-web.js` session for QR / bulk | Supplier payment settlements and payment reminders. UI `/payment-alerts` |
| `backend/src/whatsapp-settings` | Yes | Does not send. CRUD for the tenant row | Settings screen `/payment-alerts/whatsapp-settings` |
| `backend/src/transaction-sender-alerts` | Yes | Xinno HTTP only | "Send invoice PDF" on sales and purchase lists. No menu page of its own |
| `backend/src/whatsapp` | **No.** Not imported in `app.module.ts` | Meta Graph `graph.facebook.com` | Dead code (`hello_world` test) |

`WhatsAppSettings` (Prisma, one row per tenant): `apiUrl` default `https://whatsapp.xinno.in/REST/directApi/message`, `wabaNumber`, `apiKey`, `templateName` default `payment_settl2`, `templateNames` JSON, `languageCode`, `isActive`.

Payment Alerts falls back to env (`WHATSAPP_API_ENDPOINT`, `WHATSAPP_WABA_NUMBER`, `WHATSAPP_API_KEY`, template names) when the tenant row is missing. Transaction sender does not fall back for credentials. It uses `templateNames.transactionSender` or `transaction_alert2`.

Refill reminders are patient clinical messages, not payment messages and not invoice PDFs. Do not call `POST api/whatsapp/send-bulk` for them, and do not reuse `payment_settl2`. Either:

- let RefillCare send, using its own Xinno credentials and the 6-placeholder template in the handover, or
- teach the Nest Xinno client that same template and pass it the rendered body.

Pick one sender. Two senders will double-message and split the STOP list. RefillCare already stores consent in `whatsapp_consent` and cancels a cycle on repurchase. That logic should remain the sender of record unless product moves consent into `public`.

Payment Alerts UI to copy for layout only:

- Shell: `frontend/src/app/modules/administration/payment-alerts/`
- Routes: `payment-alerts.routes.ts` (standalone `loadComponent` children)
- Header: `components/header/header.component.ts`
- Dashboard table: `pages/dashboard-page/`
- HTTP: `services/payment-alerts-analytics.service.ts` (`${environment.apiUrl}/payment-alerts/...`)

Page chrome uses Angular Material (`mat-table`, `mat-button`) and Tailwind. DevExtreme is for enterprise reports. Syncfusion is in `package.json` and is not used by feature screens. New refill pages should be standalone components with Material and Tailwind, same as Payment Alerts. NgRx is global (`app.config.ts`) and is not used by Payment Alerts or Forecast. Do not add a new NgRx slice for this screen.

---

## 11. Frontend pattern to follow

Closest working mini app: Payment Alerts.

1. Add a route in `frontend/src/app/routing/` (new `refill-reminders.routes.ts`, or a child of `utilities.routes.ts` with `path: 'refill-reminders'`). Lazy `loadComponent` or `loadChildren`.
2. Feature folder `frontend/src/app/modules/utilities/refill-reminders/` (or `modules/administration/refill-reminders/` if you want it beside payment-alerts). Contents: `pages/`, `services/refill-reminders.service.ts`, `models/`.
3. Service methods call `${environment.apiUrl}/refill-reminders/...` only.
4. Add the page under Mini Apps in `navigation-structure-data.json`, and add the route key to the permissions block near `"/payment-alerts"` (that block lists actions such as Export and View Details).
5. Add the page key to `targetPageKeys` in `navigation.service.ts`.
6. Add a fifth `featureCards` entry on `mini-tenant-dashboard.component.ts`. The current card numbers (sent today, accuracy, and so on) are placeholders, not live queries. A new card can ship with static text the same way, then be wired to `GET /api/v1/analytics/kpi` later.
7. Workspace tab label: `frontend/src/app/layout/common/workspace-tabs/workspace-tab-url.util.ts` maps the first URL segment to a title (`payment-alerts`, `forecast`). Add `refill-reminders`.

Forecast (`frontend/src/app/modules/platform/forecast`) is an iframe to an external tool. Refill reminders need a real Material table, because staff must approve and send. Do not iframe Streamlit for the production screen. Streamlit can stay as the Python team's local console.

Role strings for the limited menu: `mini-tenant-admin`, `mini-store-admin` (`backend/prisma/seeds/postgres/role.seed.ts` and `navigation.service.ts`).

---

## 12. Files a Cursor session should open

Read:

- `docs/MEDIASTRA_FULLSTACK_GUIDE_FOR_REFILLCARE.md` (this file)
- `backend/prisma/seeds/postgres/navigation-structure-data.json` (Mini Apps block, about line 884)
- `backend/src/navigation/navigation.service.ts` (`targetPageKeys`)
- `frontend/src/app/modules/platform/mini-tenant-dashboard/mini-tenant-dashboard.component.ts`
- `frontend/src/app/modules/administration/payment-alerts/payment-alerts.routes.ts`
- `frontend/src/app/core/auth/auth.interceptor.ts`
- `backend/src/app.module.ts`
- `backend/src/main.ts` (CORS, no global prefix)
- RefillCare `api/main.py` and `api/security.py`

Leave untouched unless the task says otherwise:

- `backend/prisma/schema.prisma` (no refill tables in `public`)
- Payment Alerts send path (`payment-alerts/whatsapp`)
- Transaction sender WhatsApp
- Sales billing calculation

Suggested Nest routes the UI binds to:

```
GET  /refill-reminders/daily
POST /refill-reminders/:id/approve
POST /refill-reminders/:id/reject
POST /refill-reminders/:id/send
GET  /refill-reminders/review-queue
```

Nest translates those to the Python paths in section 9.

---

## 13. Decisions still open

These are not answered by the code. Do not guess them inside a pull request.

1. **Fifth app.** Confirm refill reminders is a new mini-tenant dashboard card plus a new Mini Apps page, and that Marketplace stays. Section 4 assumes that.
2. **Who sends WhatsApp.** RefillCare's Xinno client (consent and repurchase cancellation stay in Python) or Mediastra's `WhatsAppSettings` row (one WABA number per tenant, shared with payment alerts).
3. **Sales feed.** Nightly or per-invoice push from `SalesInvoices` into `POST /api/v1/sales/ingest`, or pharmacists keep uploading Excel. Live bills already exist, so a file upload duplicates data.
4. **Store key.** Which Mediastra field is `X-Store-Id`: `Store.code`, `Store.storeName`, or a new mapping table.
5. **Patient identity.** `Customers.customerName` + `phone1`, or `FamilyMember.name` + `FamilyMember.number` when `familyMemberId` is set.
6. **Retail vs wholesale.** Which column drops `SB/` style B2B bills from reminders.
7. **Approval.** Pharmacist must approve each stage in the new UI, or RefillCare auto-dispatches and the UI is a log.
8. **CORS.** Add `https://app.mediastracloud.com` to `main.ts` before production use.
9. **Secrets and volume.** ECS task env for `REFILLCARE_API_URL`, `REFILLCARE_API_KEY`, and the disk or EFS path RefillCare uses for models. This repo's ECS workflow does not mount `/data/refillcare`. Container disk on the current Nest task is ephemeral.
10. **Offline store mode.** Is refill reminders cloud-only for the first release? SQLite store mode has no RefillCare schema.

---

## 14. Technology list (this repo)

**Application**

- TypeScript 5.6 (frontend) and 5.9 (backend), Node 20
- NestJS 10: Express, CQRS, Schedule, Swagger (only when `SWAGGER_ENABLED=true`, path `/api`), Socket.IO, Passport JWT, bcrypt, class-validator, Helmet
- Prisma 5.22, PostgreSQL 15 (`pg_trgm`, `fuzzystrmatch`), SQLite for store mode, `mssql` only for EasySol migration scripts
- Angular 19.2, Angular Material 19, Tailwind 3, RxJS, Transloco, NgRx (masters and sales slices, not mini apps)
- Electron 30 optional desktop shell
- PDF: pdfkit and Puppeteer on the server, jsPDF in the browser
- Excel: `xlsx` / `xlsx-js-style`
- Payments: Razorpay
- E-invoice: Masters India (`api.mastersindia.co` / sandbox)
- Marketplaces: Pharmarack, Zennx
- Login city lookup: ip-api.com
- Email: Nodemailer, Gmail OAuth via `googleapis` in `email-settings`

**Cloud**

- AWS `ap-south-1`: ECR, ECS, S3, CloudFront, IAM OIDC for GitHub Actions
- S3 application storage (`backend/src/common/aws/services/s3.service.ts`) is behind `USE_S3_STORAGE=true` and is used by migration backup upload, not by logos or prescriptions. Those use `./uploads` served at `/uploads/`
- RDS restore-from-S3 exists for EasySol migration (`rds.service.ts`). The production database host is not declared in this repo; it arrives as `DATABASE_URL` on the ECS task

**Tests and tooling**

- Backend: Jest, Supertest, ESLint, Prettier
- Frontend: Karma/Jasmine, Playwright, Angular ESLint
- Husky on the root `package.json`

---

## 15. End-to-end flow after integration

1. Staff signs in. Interceptor stores tenant UUID and active store UUID.
2. Staff opens Mini Apps → Refill Reminders (`/refill-reminders`). `MenuPermissionGuard` allows it because the navigation seed contains the page.
3. Angular `GET /refill-reminders/daily`.
4. Nest checks the JWT and the store, maps the UUID to the RefillCare store key, and calls `GET /api/v1/reminders/daily` with `X-API-Key`, `X-Store-Id`, and `X-Actor`.
5. The table shows patient, medicine, stage, and status from Python. Angular does not compute the date.
6. Approve calls Nest, Nest calls `POST /api/reminders/{id}/approve`.
7. Send calls Nest, Nest calls `POST /api/v1/whatsapp/send-single`. Python applies opt-out, med-sync, and the Xinno template.
8. A new sale in Mediastra is pushed into RefillCare (decision 3). Python cancels pending stages when the patient has already bought.

That is the whole join between the two codebases.
