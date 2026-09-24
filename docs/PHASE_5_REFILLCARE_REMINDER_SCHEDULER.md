# RefillCare — Phase 5: Reminder Eligibility & Scheduling Engine

**Status:** Completed  
**Component:** `reminder/scheduler.py`  
**Test Suite:** `tests/refillcare/test_scheduler.py` (11 unit tests, 39 total passed in full suite)

---

## 1. Purpose & Scope

The **RefillCare Reminder Eligibility & Scheduling Engine** forms the business logic layer that converts ML regression predictions (`predicted_days_until_next_purchase` from the Phase 4 XGBoost model) into actionable, deterministic, and duplicate-free reminder schedules.

### Core Objectives:
1. **Refill Identity Protection:** Maintain strict patient + medication cycle isolation (`customerId + itemId`).
2. **Deterministic Multi-Stage Reminders:** Automatically calculate 6 standard refill touchpoints (`-7d`, `-3d`, `-1d`, `0d`, `+2d`, `+5d`).
3. **Automated Refill Cycle Reset:** Instantly cancel pending/un-sent reminders from prior cycles when a customer records a new purchase for that medication.
4. **Idempotency & Duplicate Prevention:** Guarantee zero duplicate reminder generation when scheduling jobs are rerun.
5. **Cold-Start & Quality Filtering:** Exclude single-purchase timelines from automated reminder generation until sufficient history is established.

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│             Canonical Purchase Event History                │
│       (customerId, itemId, invoice_date, purchase_count)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           Phase 4 XGBoost Inference Engine                  │
│       (predicted_days_until_next_purchase, ~15.2d MAE)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│       Phase 5: Eligibility & Prediction Validation          │
│   - Check purchase_count >= 2 (exclude cold-start)          │
│   - Validate non-negative, finite interval prediction       │
│   - Grade history quality (high / medium / low)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Expected Refill Date Calculation                 │
│      expected_refill_date = invoice_date + predicted_days   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Refill Cycle Reset & Cancellation               │
│   Cancel prior scheduled reminders for (customerId, itemId) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             6-Stage Deterministic Schedule                  │
│        [-7 days, -3 days, -1 day, 0 days, +2 days, +5 days] │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Daily Due Reminder Queue                    │
│     (Filtered by reminder_date == target_date & scheduled)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Refill Identity

In pharmacy operations, **`customerId + itemId`** uniquely identifies a medication replenishment timeline:
- **`customerId`:** Unique customer identifier in the billing system.
- **`itemId`:** Unique medication catalog code (`itemId == itemCode`).
- **`MOBILE_NO` is NOT Identity:** Multiple family members routinely share a single primary phone number. Merging timelines by phone number would cause cross-patient medication confusion. `MOBILE_NO` is strictly treated as a contact delivery channel.
- **Multiple Medications Isolation:** If customer `C001` takes `TELMISARTAN` (`M001`) and `METFORMIN` (`M002`), each medication maintains an independent refill reminder schedule and separate cycle reset triggers.

---

## 4. Expected Refill Date Calculation

The expected refill date is computed deterministically from the latest purchase date:
$$\text{expected\_refill\_date} = \text{latest\_purchase\_date} + \text{round}(\text{predicted\_interval\_days})$$

### Validation Rules:
- **Finite Check:** Predictions that are `NaN`, `None`, or `infinite` are rejected with status `invalid_prediction`.
- **Non-Negative Check:** Predictions $<0$ days are rejected with status `invalid_prediction`.
- **Zero-Day Handling:** Zero-day predictions are explicitly supported ($\text{expected\_refill\_date} = \text{latest\_purchase\_date}$) and flagged for clinical workflow review.
- **Date Formatting:** Accepts both ISO (`YYYY-MM-DD`) and Indian standard formats (`DD/MM/YYYY`, `DD-MM-YYYY`).

---

## 5. Refill Eligibility & History Quality Grading

| Conceptual Status | Condition | Action |
| :--- | :--- | :--- |
| **`eligible`** | `purchase_count >= 2` AND finite non-negative prediction | Schedules full 6-stage reminder cycle |
| **`ineligible`** | `purchase_count < 2` (cold-start / single purchase) | Excluded from scheduling with explicit reason |
| **`invalid_prediction`** | `predicted_days < 0` OR `NaN` OR unparseable date | Excluded from scheduling with explicit error log |

### History Quality Indicator (Rule-Based):
- **`high_history`:** $\ge 5$ purchases, or $\ge 3$ purchases with recurring adherence history.
- **`medium_history`:** $3-4$ purchases, or $2$ purchases with recurring adherence history.
- **`low_history`:** Exactly $2$ purchases.
- **`ineligible`:** $<2$ purchases.

> **Note:** History quality grades are heuristic operational indicators reflecting historical data depth, not Bayesian statistical probabilities.

---

## 6. The Six Reminder Stages & Dynamic Copy

The engine generates exactly six scheduled touchpoints relative to `expected_refill_date`:

| Stage | Offset | Purpose | Dynamic Message Template |
| :---: | :---: | :--- | :--- |
| **-7d** | $-7\text{ days}$ | Early advance notice | *"Your regular medicine refill may be due in about 7 days, around DD-MM-YYYY."* |
| **-3d** | $-3\text{ days}$ | Planning reminder | *"Your regular medicine refill may be due in about 3 days, around DD-MM-YYYY."* |
| **-1d** | $-1\text{ day}$ | Eve-of-refill alert | *"Your regular medicine refill may be due tomorrow, DD-MM-YYYY."* |
| **0d** | $0\text{ days}$ | Due-date reminder | *"Your regular medicine refill may be due today, DD-MM-YYYY."* |
| **+2d** | $+2\text{ days}$ | Post-due grace follow-up | *"Your expected refill date was DD-MM-YYYY. If you still need your medicine, please contact us."* |
| **+5d** | $+5\text{ days}$ | Final adherence check | *"This is a follow-up regarding your expected medicine refill on DD-MM-YYYY. Please contact us if you still need your medicine."* |

*(Dates are automatically formatted as `DD-MM-YYYY` in all client-facing message copies).*

---

## 7. Reminder Record Schema

Each scheduled reminder instance is modeled by the `ReminderRecord` data structure:

```python
@dataclass
class ReminderRecord:
    reminder_id: str             # Deterministic key: {customerId}___{itemId}___{expected_refill_date}___{stage}d
    customerId: str              # Unique customer billing identity
    itemId: str                  # Unique medicine catalog identity
    customerName: str            # Customer display name
    MOBILE_NO: str                  # Delivery phone number
    itemName: str                # Commercial medicine name
    latest_purchase_date: str    # YYYY-MM-DD
    predicted_interval_days: float # Predicted refill cycle length
    expected_refill_date: str    # YYYY-MM-DD
    reminder_stage: int          # -7, -3, -1, 0, 2, 5
    reminder_date: str           # YYYY-MM-DD
    message: str                 # Exact dynamic copy
    status: str = "scheduled"    # "scheduled", "cancelled", "completed"
    history_quality: str = "medium_history"
    created_at: str = ...        # ISO timestamp
```

---

## 8. Refill Cycle Reset on New Purchase

When a customer makes a new purchase of medication `M001`:
1. All pending (`status == "scheduled"`) reminders from the **previous** cycle for `(customerId, itemId)` are automatically updated to `status = "cancelled"`.
2. Old post-refill reminders (e.g. `+2d`, `+5d`) from the prior cycle are immediately suppressed.
3. A new cycle is generated using the new purchase date as `latest_purchase_date` and a refreshed XGBoost prediction.

---

## 9. Duplicate Protection & Idempotency

The scheduler guarantees deterministic and idempotent behavior:
- **Reminder Key:** `reminder_id = f"{customerId}___{itemId}___{expected_refill_date}___{stage:+d}d"`
- Re-running the scheduler over identical batch data detects existing keys and preserves existing status without generating duplicate reminder entries.

---

## 10. Daily Due Reminder Selection

The module provides an efficient filter method for daily operational jobs:

```python
from reminder.scheduler import RefillReminderScheduler

scheduler = RefillReminderScheduler()
# ... load or register active schedules ...

due_today = scheduler.get_due_reminders("2026-09-02")
# Returns only records where reminder_date == "2026-09-02" and status == "scheduled"
```

---

## 11. In-Memory Demonstration Walkthrough

Executing `python -m reminder.scheduler` runs a fully self-contained demonstration:

```text
=== RefillCare Phase 5 Scheduler Demonstration ===
Customer ID:           CUST_DEMO_001
Medicine ID:           ITEM_TELMI_40
Latest Purchase Date:  2026-08-10 (10-08-2026)
Predicted Interval:    30.0 days
Expected Refill Date:  2026-09-09 (09-09-2026)

Generated 6 Reminder Stages:
-------------------------------------------------------------------------------------
Stage -7d | Date: 2026-09-02 | Status: scheduled | Copy:
  "Your regular medicine refill may be due in about 7 days, around 09-09-2026."
Stage -3d | Date: 2026-09-06 | Status: scheduled | Copy:
  "Your regular medicine refill may be due in about 3 days, around 09-09-2026."
Stage -1d | Date: 2026-09-08 | Status: scheduled | Copy:
  "Your regular medicine refill may be due tomorrow, 09-09-2026."
Stage +0d | Date: 2026-09-09 | Status: scheduled | Copy:
  "Your regular medicine refill may be due today, 09-09-2026."
Stage +2d | Date: 2026-09-11 | Status: scheduled | Copy:
  "Your expected refill date was 09-09-2026. If you still need your medicine, please contact us."
Stage +5d | Date: 2026-09-14 | Status: scheduled | Copy:
  "This is a follow-up regarding your expected medicine refill on 09-09-2026. Please contact us if you still need your medicine."
-------------------------------------------------------------------------------------

Due Reminders on 2026-09-02: 1 found (Stage -7d)

--- Simulating New Purchase on 2026-09-08 (Cycle Reset) ---
Cancelled prior reminders: 6
New Expected Refill Date:  2026-10-08 (08-10-2026)

All Reminders in Registry for Patient + Medicine:
  Exp: 2026-09-09 | Stage: -7d | RemDate: 2026-09-02 | Status: cancelled
  Exp: 2026-09-09 | Stage: -3d | RemDate: 2026-09-06 | Status: cancelled
  Exp: 2026-09-09 | Stage: -1d | RemDate: 2026-09-08 | Status: cancelled
  Exp: 2026-09-09 | Stage: +0d | RemDate: 2026-09-09 | Status: cancelled
  Exp: 2026-09-09 | Stage: +2d | RemDate: 2026-09-11 | Status: cancelled
  Exp: 2026-09-09 | Stage: +5d | RemDate: 2026-09-14 | Status: cancelled
  Exp: 2026-10-08 | Stage: -7d | RemDate: 2026-10-01 | Status: scheduled
  Exp: 2026-10-08 | Stage: -3d | RemDate: 2026-10-05 | Status: scheduled
  Exp: 2026-10-08 | Stage: -1d | RemDate: 2026-10-07 | Status: scheduled
  Exp: 2026-10-08 | Stage: +0d | RemDate: 2026-10-08 | Status: scheduled
  Exp: 2026-10-08 | Stage: +2d | RemDate: 2026-10-10 | Status: scheduled
  Exp: 2026-10-08 | Stage: +5d | RemDate: 2026-10-13 | Status: scheduled
```

---

## 12. Why WhatsApp Sending is Isolated from Phase 5

Phase 5 strictly focuses on **scheduling logic and queue management**. WhatsApp transmission is intentionally decoupled because:
1. **Safety & Auditability:** Scheduling records must be reviewable by pharmacy staff before dispatch.
2. **Rate-Limiting & Provider Agnosticism:** The scheduling layer remains independent of downstream WhatsApp API providers (Xinno, Twilio, Meta Cloud API, etc.).
3. **Preservation of Frozen Applications:** Existing WhatsApp broadcast campaigns (`app.py`, `app_image_campaign.py`) remain completely untouched.

---

## 13. Current MVP Limitations & Next Steps

1. **Single-Medicine Dispatch:** Phase 5 models reminders per medication (`customerId + itemId`). If a patient has multiple due reminders on the same day, future phases can implement multi-medication batch digest grouping.
2. **Manual Staff Override:** Phase 6 Streamlit UI will allow pharmacy operators to view pending reminder queues and manually pause or dismiss reminders.
