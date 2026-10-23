# RefillCare — Implementation Plan (executed to date, and what is next)

Single store (NGOs Colony) in v1; multi-store through the cloud-hosted Mediastra web application in v2.
How to run and use the system: [README.md](README.md). Client-facing deployment options:
[docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md](docs/DEPLOYMENT_OPTIONS_LOCAL_VS_CLOUD.md).

Status legend: ✅ done and tested · 🟡 built, switched off until you decide · ⏳ next

---

## 1. Data and ingestion

| # | Item | Status | Result |
|---|---|---|---|
| 1.1 | Sales upload (CSV / Excel), date parsing (day-first, ambiguity blocked), de-duplication, append to master with pre-upload backup, batch rollback | ✅ | Same code path for Streamlit and FastAPI |
| 1.2 | Channel filter: customer sales (`S0/`) kept, B2B transfers (`SB/`) and unknown excluded | ✅ | 907,080 S0 vs 2,571 SB rows |
| 1.3 | Chronic / acute medication classifier (Supabase rules + local cache, salt catalogue fallback) | ✅ | |
| 1.4 | **Column mapping for any store**: alias config (≈200 names) + value checks, confirm screen only when needed, mapping saved per store and file layout, list / delete | ✅ | Real 23-column file and a differently named export map with no questions |
| 1.5 | **Only 11 fields stored** (date, invoice, name, item code, item name, packing, quantity, free quantity, mobile, customer id, item id); price / tax / batch / expiry columns dropped at upload | ✅ | Master 132.3 → 78.5 MB; history parquet 33.1 → 17.1 MB; predictions identical on 44,601 decisions |
| 1.6 | Raw full export kept as a separate backup for training (`data/refillcare/raw_backup/`, git-ignored) | ✅ | |
| 1.7 | Uploads read as text (mobiles / codes never become floats); item catalogue accepts `Code` or `itemCode` | ✅ | |
| 1.8 | Item catalogue switched to `SALT ITEMS LIST.csv` (7 columns: S.No, CATEGORY, Code, Item Name, PACK, SALT, ITEMCAT) | ✅ | 36,963 items, 22,168 with a salt |

## 2. Prediction engine

| # | Item | Status | Result |
|---|---|---|---|
| 2.1 | Path A (≥ 6 purchases, stable): registry ML model or personal median, safety caps | ✅ | |
| 2.2 | Path B (3–5 purchases): days of supply from the patient's own pace | ✅ | |
| 2.3 | Pharmacist review for unstable, bulk, identity-conflict and too-short histories | ✅ | Coverage 58% of refills |
| 2.4 | 5 quantity archetypes (multi-pack, partial strip, early top-up, post-lapse, consensus bounds) | ✅ | |
| 2.5 | Features: calendar (usual day of month, fit gain), visits (other medicines, last visit), salt priors (typical interval, units per day, dose supply) — point-in-time, same code for train / backtest / serve | ✅ | |
| 2.6 | Free quantity counted in days of supply (training = serving) | ✅ | No effect on today's data (0 S0 rows with free quantity) |
| 2.7 | Prediction rules, each switchable and measured: high-stability median, insulin median, day-of-month anchor, Path B `shrink` / `blend`, device cap `observed` | 🟡 | Only `blend` helps: +0.46 pp within ±7, MAE 8.15 → 7.87 d on the test month; mixed (±1 pp) over 3 months but lower MAE every month |
| 2.8 | **Patient identity = name + mobile** (`REFILLCARE_PATIENT_IDENTITY=name_mobile`) | 🟡 | 14,071 names → 35,040 patients; 2,663 names had > 1 mobile (62% of rows); 505 identity-blocked medicines unblocked |
| 2.9 | **Therapy = salt group** (`REFILLCARE_ITEM_GROUPING=salt`): brand / strength switches continue one cycle | 🟡 | 73k rows regrouped; +0.7 pp within ±7 on top of name + mobile |
| 2.10 | Identity cut-over: retire old-key reminders, re-score, never re-send a sent stage, DB backup first (ML tab, API, script) | ✅ | |

## 3. ML lifecycle

| # | Item | Status | Result |
|---|---|---|---|
| 3.1 | Model registry (production / shadow / candidate / retired) in `enterprise.db` | ✅ | |
| 3.2 | Dynamic windows from the latest sale (test 30 d, validation 30 d, 45-day censor buffer) | ✅ | Test window now 18 Jul – 16 Aug 2026 |
| 3.3 | Training fixes: label-horizon bias guard (90 d), log-ratio target vs personal median, early stopping | ✅ | |
| 3.4 | Promotion gate incl. per-segment MAE check (insulin regression blocked) | ✅ | |
| 3.5 | Retrain cycle (train → select on validation → gate on test), freshness experiment | ✅ | |
| 3.6 | Rule A/B and rolling 3-month test, saved reports shown in both UIs | ✅ | Model beats baseline engine every month: +3.0 / +4.1 / +1.3 pp (58.7% vs 55.6% pooled, 5,065 refills) |
| 3.7 | Live outcome tracking of sent reminders (on time / early / late / no refill) | ✅ | |

Sandbox results (test month 18 Jul – 16 Aug 2026, 1,740 refills):

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
| 4.1 | 7-stage lifecycle (-7, -3, -1, 0, +2, +5, +45 d), repurchase supersedes pending stages, past-due stages expire | ✅ |
| 4.2 | Med-Sync: one message per patient for several medicines | ✅ |
| 4.3 | Pharmacist approval before live sends; dry run by default; idempotent (no double sends) | ✅ |
| 4.4 | Opt-out: STOP / UNSUBSCRIBE via webhook, exact START to opt in, pharmacist opt-out / opt-in; every send path skips opted-out phones (last-10-digit match) | ✅ |
| 4.5 | Standalone text / image campaign apps (`whatsapp_campaigns/`) | ✅ |

## 5. Interfaces

| # | Item | Status |
|---|---|---|
| 5.1 | Streamlit operations app — 8 tabs incl. column mapping, saved mappings, ML settings, rule A/B, 3-month test, identity cut-over | ✅ |
| 5.2 | FastAPI + web app with the same features (parity rule: shared services, no inter-dependency) | ✅ |
| 5.3 | Write protection (`REFILLCARE_API_KEY`, local-only when unset) | ✅ |
| 5.4 | Cache cleared after every mutation in Streamlit | ✅ |

## 6. Repository hygiene (this round)

- One README and this one plan; phase reports, notebooks and the legacy CLI / model / preprocessing code are
  removed (see the cleanup list in the hand-over; history stays in git).
- Patient data, backups, `enterprise.db`, generated ML reports and newly trained model files are git-ignored.

---

## 7. Decisions waiting for you

| Decision | Recommendation |
|---|---|
| Turn on `REFILLCARE_RULE_PATH_B_MODE=blend` | Yes — lower error every month, fixes the insulin gate failure |
| Turn on `REFILLCARE_PATIENT_IDENTITY=name_mobile` and `REFILLCARE_ITEM_GROUPING=salt`, then run the identity cut-over | Yes — correct patient per phone; dry run first |
| Retrain and promote on the real database (needs a machine with ≥ 4 GB free RAM; backup first) | After the two switches above, so the gate measures the final setup |

## 8. Next (v2 — multi-store on the Mediastra cloud)

The Mediastra web application (cloud) handles tenant / store / org logins and calls this API; each store has
its own data, model and WhatsApp account.

| # | Item | Why |
|---|---|---|
| 8.1 | Tenant / store scoping of every table and data folder (the mapping profiles already carry `store_id`) | No data mixing between stores |
| 8.2 | 3-year minimum and mandatory-field check on a store's first upload | Enough history to train and evaluate |
| 8.3 | Stale-data guard: warn / block approvals when the daily upload is missing | Avoid reminding a patient who already bought |
| 8.4 | Medicine rules per store in the store database (today: Supabase + local cache) | Store-specific rules |
| 8.5 | Scheduled jobs per store: monthly `retrain --promote`, daily stage expiry | No manual steps across 20+ stores |
| 8.6 | Provider signature check on the WhatsApp webhook | Before exposing the API publicly |
| 8.7 | Real customer code from Mediastra when available (replaces name + mobile) | Strongest patient identity |
| 8.8 | Dose capture at billing (e.g. 1-0-1) | Exact days of supply for short histories |

Deferred (enterprise level, after v2): refill-date ranges instead of one date, a return / churn model,
automated tuning (Optuna / CatBoost), and raising coverage above 58% (changes pharmacist workload).
