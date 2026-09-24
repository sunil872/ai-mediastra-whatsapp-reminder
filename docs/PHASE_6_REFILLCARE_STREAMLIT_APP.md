# RefillCare — Phase 6: Streamlit Application Documentation

**Status:** Completed  
**Application Entrypoint:** [`app_refillcare.py`](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/app_refillcare.py)  
**Test Suite:** [`tests/refillcare/test_app_refillcare.py`](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/tests/refillcare/test_app_refillcare.py) (11 tests, 50 passed in full suite)

---

## 1. Overview & Purpose

**RefillCare UI (`app_refillcare.py`)** is an operational Streamlit web dashboard built specifically for pharmacy operators and care coordinators. It bridges the gap between machine learning intelligence and daily pharmacy workflows without exposing technical code or machine learning jargon.

### Core Objectives:
1. **Pharmacy-Centric Presentation:** Uses standard pharmacy terms like *Expected Refill Date*, *Last Purchase Date*, and *History Quality* instead of ML terms like "forecasting", "features", or "target".
2. **Reuses Phases 4 & 5 Logic:** Directly uses the Phase 4 XGBoost model pipeline (`refillcare.models.prediction.generate_batch_predictions`) and Phase 5 deterministic scheduler (`reminder.scheduler.RefillReminderScheduler`).
3. **Multi-Stage Queue Management:** Displays the 6-stage reminder cycle (`-7d`, `-3d`, `-1d`, `0d`, `+2d`, `+5d`) with dynamic, client-ready message copy.
4. **Transparent Cold-Start Separation:** Distinctly isolates eligible repeat buyers from single-purchase (cold-start) records.
5. **Decoupled Architecture:** Read-only scheduling inspection; no WhatsApp messages are sent and frozen applications (`app.py`, `app_image_campaign.py`) remain completely untouched.

---

## 2. Architecture & Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│             Processed Data Artifacts (Parquet)              │
│       - test.parquet (Recent May-June 2026 Evaluation)      │
│       - purchase_history.parquet (Complete Timelines)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Phase 4 Trained Model: refill_model.joblib         │
│          (XGBoost Pipeline with Numeric & Categorical)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          app_refillcare.prepare_prediction_overview         │
│   - Filters cold-start (< 2 purchases)                      │
│   - Runs generate_batch_predictions()                       │
│   - Formats clean pharmacy columns & history quality        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│      Phase 5 Scheduler: RefillReminderScheduler             │
│   - Auto-generates 6 reminder stages (-7, -3, -1, 0, +2, +5)│
│   - Enforces automatic cycle resets upon repeat purchases   │
│   - Deduplicates reminder keys                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Web Application                   │
│   - Executive Operations Summary Metrics                    │
│   - Tab 1: Refill Predictions & CSV Export                  │
│   - Tab 2: Reminder Schedule Queue & CSV Export             │
│   - Tab 3: Patient & Medication Deep Dive                   │
│   - Tab 4: Cold-Start / Ineligible Records                  │
│   - Tab 5: System Health & Artifact Inspector               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. User Interface Features

### 3.1 Operations Summary
At the top of the application, five high-level metrics summarize current pharmacy adherence:
- **Total Histories:** Distinct customer + medicine histories in the selected window.
- **Eligible for Reminders:** Multi-purchase histories with valid predicted refill cycles.
- **Cold-Start (Single Purchase):** First-time buyers excluded from automated reminders.
- **Predictions Generated:** Total active refill estimates.
- **Active Reminders:** Total active scheduled touchpoints across the 6 stages.

---

### 3.2 Tab 1: Refill Predictions
Displays upcoming medication replenishment estimates:
- **Columns:** Customer Name, Medicine, Delivery Phone, Last Purchase Date, Purchase Count, Predicted Interval (Days), Expected Refill Date, History Quality.
- **Search & Filters:** Real-time search by patient name, medication name, and history quality tier (`high_history`, `medium_history`, `low_history`).
- **CSV Export:** Instant one-click download of filtered predictions.

---

### 3.3 Tab 2: Reminder Schedule Queue
Displays the scheduled multi-stage touchpoint queue:
- **Stages:** Exactly six touchpoints per cycle:
  - `-7 days`: Early advance notice (*"due in about 7 days, around DD-MM-YYYY"*)
  - `-3 days`: Planning reminder (*"due in about 3 days, around DD-MM-YYYY"*)
  - `-1 day`: Eve-of-refill alert (*"due tomorrow, DD-MM-YYYY"*)
  - `0 days`: Due-date reminder (*"due today, DD-MM-YYYY"*)
  - `+2 days`: Grace period follow-up (*"expected refill date was DD-MM-YYYY..."*)
  - `+5 days`: Final adherence check (*"follow-up regarding expected refill..."*)
- **Filters:** By reminder stage, status (`scheduled`, `cancelled`, `completed`), customer, and medication.
- **CSV Export:** Downloadable dispatch queue for operational audit.

---

### 3.4 Tab 3: Patient & Medication Deep Dive
Allows pharmacy staff to inspect an individual patient's full purchasing history:
- **Patient & Medicine Selectors:** Interactive dropdowns filtered to active patients.
- **Summary Metrics:** Total lifetime purchases, historical median interval, expected refill date, and history quality rating.
- **Timeline Table:** Chronological record of all prior billing events, dates, quantities, and interval days.
- **Upcoming Touchpoints:** Real-time view of the scheduled touchpoints for that specific medicine.

---

### 3.5 Tab 4: Ineligible / Cold-Start Transparency
Provides visibility into customer records that were **not** scheduled for automated reminders:
- Clearly lists why each record was omitted (e.g. *"Cold-start history: purchase count is 1 (< 2)"*).
- Prevents clinical confusion by showing that first-time buyers are not lost, but safely held until repeat patterns emerge.

---

### 3.6 Tab 5: System Health & Artifact Info
Displays artifact locations, model architecture details (XGBoost pipeline), and validation metrics (~15.2 days MAE) for auditing and operational review.

---

## 4. Performance & Caching

The application uses Streamlit's native caching mechanisms:
- `@st.cache_resource`: For the heavy XGBoost model bundle (`refill_model.joblib`), ensuring it is loaded into memory only once.
- `@st.cache_data`: For loading datasets (`test.parquet`, `purchase_history.parquet`), eliminating disk I/O on every user filter or interaction.
- Prediction generation over 11,000 records executes in **under 0.5 seconds**.

---

## 5. How to Run & Verify

To launch the Streamlit dashboard locally:

```bash
streamlit run app_refillcare.py
```

The application will launch on `http://localhost:8501` (or next available port).

---

## 6. MVP Limitations & Clinical Notes

1. **Estimate Nature:** Expected refill dates are operational estimates based on empirical interval regularity; they do not diagnose medical conditions or guarantee patient behavior.
2. **Read-Only Scheduling:** WhatsApp message sending is intentionally excluded in Phase 6 to ensure pharmacy operators can review predictions safely before any outbound messaging is enabled.
3. **Preservation of Existing Apps:** Existing WhatsApp campaign files (`app.py`, `app_image_campaign.py`) remain completely unchanged and operational.
