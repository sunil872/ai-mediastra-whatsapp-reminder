# Mediastra RefillCare — Implementation Plan & Architecture

> **Platform:** Clinical AI Refill Prediction, Med-Sync & WhatsApp Outreach  
> **Status:** Production-Ready & Verified  
> **Data Horizon:** Dynamic Continuous-Observation (Rolling 30 Days + Complete Month)

---

## 1. System Overview

RefillCare automates chronic medication refill forecasting and patient communication for retail pharmacy chains. It replaces naive 30-day static reminders with individualized clinical predictions, bundles multiple prescriptions into a single monthly delivery/pickup appointment, and sends timely WhatsApp notifications.

```
[ Raw ERP / POS Sales ] ──▶ [ Ingestion & B2B Gate ] ──▶ [ Chronic Classifier ]
                                                                 │
                                                                 ▼
[ Multi-Channel Delivery ] ◀── [ Lifecycle & Repurchase ] ◀── [ Dual-Path AI Engine ]
  • Streamlit Dashboard           Auto-Superseding               • Path A (Cadence + Quantiles)
  • FastAPI & Web SPA             State Machine                  • Path B (Days-of-Supply)
  • WhatsApp Gateway                                             • Med-Sync Bundler
```

---

## 2. Key Implemented Modules

### Module 1: Ingestion & Continuous-Observation Pipeline
- **Raw Transaction Cleanse:** Ingests master POS sales (`909,651` records), isolates B2B/wholesale transfers (`SB/...`), removes internal duplicates (`-17,222`), leaving `892,429` clean retail customer transactions.
- **Dynamic As-Of Anchor ($T_{\text{eval}} = T_{\max} + 1\text{d}$):**
  - Evaluates from the day following the latest transaction date.
  - Supports arbitrary upload batches (1-day, 7-day, 13-day, or full month) without skipping days or misaligning forecast horizons.
- **Rolling 30-Day Outreach Horizon:**
  - Evaluates outreach over the active window $[T_{\text{eval}}, T_{\text{eval}} + 29\text{d}]$ alongside standard calendar month views.
- **Master Dataset Append & Safety Backups:**
  - Auto-appends validated records to `data/refillcare/customer_data_fields.csv` with timestamped pre-image backups (`data/refillcare/backups/`).
  - Supports 1-click batch rollback to safely restore history.

### Module 2: Clinical Chronic Medication Classifier
- **Active Rule Integration:** Classifies items via Supabase rule tables with local cached fallbacks.
- **Formulation Safety Gate:** Strictly excludes acute items (Paracetamol, Dolo, pain killers, cold syrups, eye/ear drops, topical creams) so only true maintenance medications generate refill cycles.

### Module 3: Dual-Path Prediction & 5 Behavioral Archetypes
- **Path A ($\ge 6$ purchases):** Longitudinal cadence evaluated via Median Absolute Deviation (MAD) + 3-Head Quantile XGBoost Regressors ($P_{10}$ early warning, $P_{50}$ median refill date, $P_{90}$ lapsed therapy bound).
- **Path B ($< 6$ purchases):** Days-of-Supply (DOS) calculated from units purchased, pack sizes, and daily consumption frequency.
- **5 Clinical Behavioral Archetypes:**
  1. *Multi-Pack Scaled:* Scales supply days when buying 60/90 units.
  2. *Early Top-Up Carryover ($R_{inv}$):* Credits remaining pills forward if refilling early.
  3. *Partial Strip Clamping:* Clamps short 10-tablet purchases to 10–15 days to avoid delayed outreach.
  4. *Post-Lapse Reset:* Re-anchors baseline cadence after therapy breaks ($>75$ days).
  5. *Physical Consensus Bounds:* Bounds predictions within $[0.65, 1.50] \times \text{DOS}$.

### Module 4: Med-Sync (Multi-Prescription Synchronization)
- **Patient Appointment Bundling:** Combines multiple active prescriptions for the same patient (matched by Name + Phone) into a single unified appointment reminder.
- **Configurable Sync Window:** Flexible 0 to 14-day window (default 0-day exact same-day or 8-day range).
- **3-Tier Lifecycle Isolation:**
  - *Tier 1 (Due/Advance):* Stages -7d, -3d, -1d, 0d.
  - *Tier 2 (Follow-up):* Stages +2d, +5d.
  - *Tier 3 (Lapsed Re-engagement):* Stage +45d.
- **Noise Reduction:** Cuts WhatsApp message volume by over 50%, eliminating notification fatigue.

### Module 5: Lifecycle State Machine & Repurchase Auto-Reset
- **7-Stage Engagement:** `-7d, -3d, -1d, 0d (Due Today), +2d, +5d, +45d (Lapsed)`.
- **Early Walk-In / Repurchase Auto-Reset:**
  - When a patient buys their medicine before or during the reminder cycle, the system automatically marks existing pending stages as `STATUS_SUPERSEDED`.
  - Instantly resets and schedules the next refill cycle without sending obsolete reminders.

### Module 6: Interfaces & Enterprise Delivery
- **Streamlit Operations Dashboard (`app_refillcare.py`):**
  - Multi-Horizon Selector: **Specific Date**, **Rolling 30 Days (Active Horizon)**, and **Complete Month**.
  - Real-time KPIs with raw vs cleaned audit metrics.
  - Technical Diagnostics toggle to hide/show complex quantile details for retail staff.
- **FastAPI Backend (`api/`):**
  - REST endpoints for daily reminders, rolling date ranges, Med-Sync bundles, KPIs, approvals, and sales ingestion.
  - 100% data and count parity with Streamlit.
- **WhatsApp & Xinno Gateway:**
  - 6-variable dynamic template (`refillcare_medicine_reminder`) with custom pharmacy store name, phone, live preview card, test dispatch, and dry-run safety modes.

---

## 3. Data Flow & Transaction Lifecycle

| Step | Action | Output / Storage |
| :--- | :--- | :--- |
| **1. Upload** | Pharmacist uploads monthly or weekly sales spreadsheet | Validated records + batch ID |
| **2. Cleanse** | Isolate B2B (`SB/...`), purge duplicate rows, format dates | `clean_transactions.parquet` |
| **3. Ingest** | Pre-image backup created, new rows appended to master | `customer_data_fields.csv` & `purchase_history.parquet` |
| **4. Score** | Unified decision engine scores eligible patient items | `RefillDecisionModel` in `enterprise.db` |
| **5. Sync** | Cycles created, repurchased items marked `SUPERSEDED` | `ReminderCycleModel` & `ReminderStageModel` |
| **6. Bundle** | Med-Sync bundles same-patient medications | Consolidated appointment queue |
| **7. Outreach** | Pharmacist reviews & dispatches via WhatsApp Gateway | Delivery logs & CSV/JSON export |

---

## 4. Verification & QA Status

- **Unit Tests:** `pytest tests/refillcare/test_medication_classifier.py` — **11/11 Passed (100%)**.
- **Dynamic Ingestion Verification:**
  - 1-day slice (`01-10-2026`) $\to$ Horizon: `02-10-2026` to `31-10-2026` ✅
  - 13-day slice (`13-10-2026`) $\to$ Horizon: `14-10-2026` to `12-11-2026` ✅
  - Month-end (`30-09-2026`) $\to$ Horizon: `01-10-2026` to `30-10-2026` ✅
- **Enterprise DB Querying:**
  - Target day (`13-10-2026`): 415 scheduled reminders ✅
  - Complete October 2026: 11,192 scheduled reminders ✅
  - Active Rolling 30 Days: 7,491 scheduled reminders ✅
- **Repurchase State Machine:** 440 historical superseded records, 22,828 active stages cleanly tracked ✅
