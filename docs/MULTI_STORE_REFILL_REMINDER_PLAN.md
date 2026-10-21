# RefillCare Refill Reminder: Running It Per Store (20+ Store Pilot)

**Short answer: yes, this is possible.** Each store's server runs its own copy of RefillCare (FastAPI + database + model). The Mediastra application calls that local API. Patient data never leaves the store, and stores never share data.

Most of this already works for a single store. This document covers how it runs, what already exists, and what still needs building before the pilot.

---

## 1. Requirements (as agreed)

| # | Requirement |
|---|---|
| 1 | RefillCare runs **on each store's server**. Patient data must not leave the store. |
| 2 | Mediastra handles tenant / store / org logins. A tenant can have many stores. **Not built here.** |
| 3 | Mediastra's live app ("Refill Reminder") calls this repo's **FastAPI** with an API key. |
| 4 | Onboarding: the store uploads **at least 3 years** of sales with the mandatory columns. |
| 5 | After that, the pharmacist uploads **daily sales** through the app's upload option. |
| 6 | A **separate model per store**. |
| 7 | Every send needs **pharmacist approval**. |
| 8 | Each store has its **own WhatsApp (Xinno) account and number**. |
| 9 | **Medicine rules** live in each store's local database. |

---

## 2. How it works

```
 ┌──────────── Mediastra live application (tenant / store logins) ────────────┐
 │  Store A user                 Store B user                 Store N user    │
 └──────┬────────────────────────────┬─────────────────────────────┬──────────┘
        │ X-API-Key (A)              │ X-API-Key (B)               │ X-API-Key (N)
        ▼                            ▼                             ▼
 ┌─ Store A server ─────┐   ┌─ Store B server ─────┐      ┌─ Store N server ─────┐
 │ RefillCare FastAPI   │   │ RefillCare FastAPI   │ ...  │ RefillCare FastAPI   │
 │ purchase history     │   │ purchase history     │      │ purchase history     │
 │ enterprise.db        │   │ enterprise.db        │      │ enterprise.db        │
 │ model registry + .joblib │ model registry + .joblib │  │ model registry + .joblib │
 │ medicine rules (local)│  │ medicine rules (local)│     │ medicine rules (local)│
 │ Xinno account A      │   │ Xinno account B      │      │ Xinno account N      │
 └──────────────────────┘   └──────────────────────┘      └──────────────────────┘
```

- **One install per store.** Each install has its own `.env`: database, data folder, store name and contact, Xinno keys, and API key.
- **No data mixing.** Each install only ever sees its own folder and database, so stores can't merge or leak data.
- **Labels only.** `tenant_id`, `store_id` and `org_id` are recorded in each install's config for labelling and support. RefillCare doesn't do any login work.

---

## 3. What already exists in this repo

| Capability | Where |
|---|---|
| Upload sales (Excel/CSV), validate, de-duplicate, append to history | `refillcare/data/sales_ingestion.py` (`process_monthly_sales_data`) |
| Predict "as of the day after the last sale" | `refillcare/engine/post_ingest.py` |
| After each upload: re-score only the affected patients, cancel reminders for patients who already bought, expire past-due stages, record outcomes | `refillcare/engine/post_ingest.py`, `refillcare/engine/persistence.py` |
| 7-stage reminder lifecycle (−7, −3, −1, 0, +2, +5, +45 days) and Med-Sync bundling | `refillcare/engine/` |
| Pharmacist approval before live send; dry run; no double sends | `refillcare/engine/persistence.py`, `refillcare/whatsapp/dispatcher.py` |
| Train / validation / test windows taken from the store's own latest sale | `refillcare/ml/service.py` (`plan_time_splits`) |
| Retrain → pick best on validation → promotion gate on test | `refillcare/ml/service.py` (`retrain_cycle`); CLI `scripts/ml_cli.py retrain --promote` |
| Model registry per database; falls back to the simple median baseline when a model isn't good enough | `refillcare/ml/registry.py`, `refillcare/ml/model_provider.py` |
| API-key protection for write actions | `api/security.py` (`X-API-Key` = `REFILLCARE_API_KEY`) |
| Per-install settings via env | `refillcare/store.py` (`REFILLCARE_DATA_DIR`, store name/contact), `DATABASE_URL`, Xinno keys |

---

## 4. Data flow

### 4.1 Onboarding a store (once)
1. Install RefillCare on the store server with that store's `.env`.
2. The pharmacist uploads **3+ years** of sales (e.g. Sep 2023 – Sep 2026).
3. RefillCare checks the mandatory columns and the 3-year minimum, then builds the store's purchase history and database.
4. Run a retrain on that store's data:
   - **Train** window: everything before validation. Fits the model.
   - **Validation** window: the 30 days before test. Picks the best model.
   - **Test** window: the last 30 days that can be scored. Promotion gate only.
   - **Last 45 days:** not scored, because those patients may not have come back yet.

   If the new model passes the gate it is promoted; otherwise the store keeps the simple median baseline, the safe default for small stores.
5. Predictions are generated as of the day after the last sale (Oct 1 for data to Sep 30). Each patient gets a next-refill date and reminder stages; "October" is just a view of those dates.

### 4.2 Every day
1. The pharmacist uploads **yesterday's sales**. Re-uploading the last 7 days is safer; duplicates are removed automatically.
2. RefillCare appends the new rows to the stored history. It never needs the 3 years again.
3. For patients who bought, it cancels pending reminders and predicts the next refill. Unsent stages that are now past due expire, and outcomes (on time / late / still open) are recorded.
4. Today's reminder list is built. The pharmacist approves it, and messages go out from the store's own WhatsApp number.

**Example:** data up to Sep 30 → Oct 1 reminders are approved and sent → on Oct 2 the pharmacist uploads Oct 1 sales → patients who bought on Oct 1 are taken out of the queue → Oct 2's reminders are built from the updated data, and so on.

### 4.3 Monthly (or once 30+ days of new data exist)
- Run `retrain --promote`. The windows move forward automatically with the new data, and the model changes only if the gate approves.
- In the NGOs Colony test, retraining more often did not improve accuracy: fresher data changed it by −0.3 points. Monthly is enough.

---

## 5. Answers to the upload questions

| Question | Answer |
|---|---|
| Do we upload 3 years + new days every time? | **No.** Upload 3+ years **once**, then only new days. History is kept on the store server. |
| How does the model "know" the past if we only upload new days? | The **model file** (~1 MB) holds learned patterns, not patient data. The **purchase history** on the store server holds every patient's past purchases. Each prediction uses both. |
| Do we upload parquet / model files plus new sales? | **No.** Pharmacists only upload sales files. Parquet, database and model files are created and kept on the server. **Back them up nightly.** |

---

## 6. Sizing (measured on the NGOs Colony repo data)

| Item | Size |
|---|---|
| Raw sales CSV | 126 MB |
| Purchase history parquet (827,955 rows, 14,071 patients, Dec 2020 – Sep 2026) | 32 MB |
| `enterprise.db` | 317 MB |
| One trained model | ~1 MB |
| Rows per year | ~130–160k (3 years ≈ 420k rows) |

**RAM:** daily operation is light, but training is memory-heavy. On a laptop with 0.8 GB free, training with a 150k-event sample ran out of memory; a 50k-event sample worked. Plan on **4 GB+ free RAM** per store server for retrains, or keep the 50k sample.

**Disk:** under 1 GB per store, plus backups.

---

## 7. Gaps to build before the pilot

| # | Gap | Why it matters | Effort |
|---|---|---|---|
| 1 | **Mandatory-column contract + 3-year minimum** on the first upload | No explicit check exists today | Small |
| 2 | **Stale-data guard**: warn or block approvals when the daily upload is missing | Without it, a patient who already bought can get a reminder | Small |
| 3 | **Medicine rules in the local database**: today rules come from Supabase (cloud) with a local fallback | Data must stay on the store server | Medium |
| 4 | **Real customer code** from Mediastra: today `customerId` is the patient's name | Different patients with the same name are merged | Small (column mapping) |
| 5 | **Per-store install package**: `.env` template, start-up service, nightly backup of history, database and models | 20+ servers need to be set up identically and kept up to date | Medium |
| 6 | **Store labels** (`TENANT_ID`, `STORE_ID`, `ORG_ID`) in each install's config, shown in `/health` and logs | Support and audit across many installs | Small |
| 7 | **Scheduled jobs**: monthly `retrain --promote`, daily stage expiry | Keeps every store's model and queue current without manual steps | Small |

---

## 8. Pilot plan
1. **Agree the upload contract:** mandatory columns, date format, customer code, and how a "day" of sales is defined.
2. **Build gaps 1–4**, and test them in a sandbox with the NGOs Colony data.
3. **Package the per-store install (gap 5)** and install it on **2–3 pilot stores** first.
4. For each pilot store:
   - onboard with 3+ years of data;
   - run a retrain;
   - check the gate result;
   - run **dry-run** reminders for 1–2 weeks;
   - compare predicted refill dates with actual purchases (outcome summary).
5. **Go live** with pharmacist approval, then roll out to the remaining stores.

---

## 9. Risks and points to confirm
- **Opt-out:** the plan currently has no STOP option. WhatsApp Business policy generally expects opt-out requests to be honoured, so keep at least the pharmacist's manual opt-out and confirm with Xinno.
- **Missed uploads:** reminders are only as current as the last upload (gap 2).
- **Small stores:** may never pass the gate and will run on the median baseline. This is safe, but accuracy will be lower.
- **Server hardware:** confirm free RAM before scheduling retrains.
- **Backups:** losing a server without a backup means uploading 3 years of history again.
