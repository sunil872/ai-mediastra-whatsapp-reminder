# Phase 14 — RefillCare Controlled Pilot Operations & Real Outcome Collection Report

**System:** Medication Refill Reminder System — RefillCare  
**Phase:** 14 — Controlled Pilot Operations & Real Outcome Collection  
**Status:** **CONTROLLED PILOT OPERATIONS READY**  
**Date:** September 2026  
**Document Version:** 1.0.0  

---

## 1. Executive Summary & Objective

The objective of Phase 14 is to operationalize the controlled pilot workflow, establish pre-send safety validation safeguards, and provide an auditable framework for collecting real operational pilot outcomes.

> [!IMPORTANT]
> **Critical Safety & Live-Send Boundary:**
> - Entering Phase 14 does **NOT** authorize real WhatsApp message transmission.
> - **Zero** automatic sending, **zero** background cron workers, and **zero** automatic retries.
> - Default operational state remains **Dry-Run Simulation**.
> - Live WhatsApp dispatch is strictly isolated behind explicit, multi-step manual operator authorization and pre-send safety validation.
> - The approved template `refillcare_medicine_reminder` (6 body variables, Utility category, en) remains completely frozen.

---

## 2. Pilot Operating Model

RefillCare's controlled pilot operating model couples machine learning prediction with human pharmacist supervision:

```
[POS Historical Transactions]
             │
             ▼
[XGBoost Interval Prediction (Frozen Model)]
             │
             ▼
[Tier A Cohort Prioritization] (>=5 purchases or recurring >=3)
             │
             ▼
[Daily Pilot Queue Generation]
             │
             ▼
[Operator Clinical Review & Feedback] ──▶ [Structured Rejection] (Suppressed)
             │
             ▼ (Explicit Approval)
[Pre-Send Safety Check Engine] (9 Assertions)
             │
             ▼ (Pass)
[Controlled Dispatch] (Dry-Run by Default; Live only with explicit user authorization)
             │
             ▼
[Persistent SQLite Audit Logging] (Idempotent Lifecycle)
             │
             ▼
[POS Repurchase Matching] ((customerId, itemId) Identity)
             │
             ▼
[Pilot Outcome Evaluation & Health Surveillance]
```

---

## 3. Daily Pilot Workflow

The daily operational workflow is structured into 6 sequential phases:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. Data Refresh │ ──▶ │ 2. Health Check │ ──▶ │ 3. Queue Review │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────┴────────┐
│ 6. Outcomes Log │ ◀── │ 5. Audit Verify │ ◀── │ 4. Safe Dispatch│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Data Refresh:** Ingest latest POS pharmacy sales and check data freshness.
2. **Health Check:** Review automated surveillance alerts (e.g., stale POS data, elevated failure rate).
3. **Queue Review:** Inspect Tier A candidates due for reminder dispatch on the operational date.
4. **Safe Dispatch:** Run pre-send safety checks; execute Dry-Run simulation or authorized live dispatch.
5. **Audit Verify:** Confirm dispatch state transitions in persistent SQLite storage.
6. **Outcomes Log:** Track subsequent pharmacy repurchases and compute empirical error metrics.

---

## 4. Candidate Review & Prioritization

The daily queue strictly prioritizes candidates based on risk-calibrated history tiers:

- **Tier A (Strong Pilot — Primary Review Queue):** Patients with $\ge 5$ lifetime purchases or $\ge 3$ recurring purchases (`is_recurring_history = 1`). Historical MAE = **9.64 days**, $\pm 7\text{d}$ accuracy = **47.27%**.
- **Tier B (Review Required — Secondary Queue):** Patients with 3–4 purchases requiring explicit pharmacist interval verification.
- **Tier C (Suppressed / Cold-Start):** Excluded from active reminder queues to protect patient trust.

---

## 5. Structured Operator Review & Rejection

Operators record explicit decisions for each queue item:
- **`APPROVE`:** Authorizes candidate for safety validation and dispatch.
- **`REJECT`:** Suppresses candidate with a mandatory structured rejection code.
- **`DEFER`:** Keeps candidate in pending queue for follow-up review.

### Standardized Structured Rejection Reasons:
1. `customer_already_purchased`
2. `prediction_appears_incorrect`
3. `customer_not_appropriate`
4. `invalid_contact`
5. `duplicate`
6. `operator_decision`
7. `other`

---

## 6. Pre-Send Safety Check Engine

Before any reminder can be dispatched, it must pass 9 automated assertions in [refillcare/evaluation/pilot_operations.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/refillcare/evaluation/pilot_operations.py#L180-L280):

| Assertion | Safety Check | Failure Code |
| :--- | :--- | :--- |
| **1. Identity Validation** | `customerId` and `itemId` present and non-empty | `FAIL_IDENTITY_MISSING` |
| **2. Phone Format** | Digits length $\ge 10$; valid format | `FAIL_PHONE_INVALID` |
| **3. Reminder ID** | Valid string identifier present | `FAIL_REMINDER_ID_MISSING` |
| **4. Duplicate Prevention** | SQLite audit check confirms reminder not already `accepted` | `FAIL_DUPLICATE_AUDIT` |
| **5. Cycle Status** | Reminder is not marked as `cancelled` | `FAIL_CYCLE_CANCELLED` |
| **6. Cycle-Reset Check** | Customer has not repurchased on/after reminder date | `FAIL_ALREADY_PURCHASED` |
| **7. Template Name** | Exact match with `refillcare_medicine_reminder` | `FAIL_TEMPLATE_MISMATCH` |
| **8. Parameter Count** | Payload contains exactly 6 body parameters | `FAIL_PARAM_COUNT` |
| **9. Privacy Sanitization** | UI and logs display masked phone (`***1234`) with zero credentials | `PASS_SAFETY` |

If any assertion fails, dispatch is **blocked immediately**.

---

## 7. Dispatch Controls & Dry-Run Safeguards

- **Default State:** `Dry-Run Simulation (No real WhatsApp messages sent)` is checked by default.
- **Live Transmission Requirements:**
  1. Live sending toggle enabled by operator.
  2. Multi-point pre-send safety validation passes.
  3. Pharmacist checks explicit acknowledgment: *"I have reviewed the recipient list and authorize dispatching these reminders via WhatsApp."*
  4. Separate instruction from business governance authorizing live dispatch.

---

## 8. Cycle Reset & Transaction Invalidation

When a patient makes a new pharmacy purchase:
1. All pending/future reminders for that exact `(customerId, itemId)` are cancelled.
2. The ML engine generates a new `expected_refill_date` from the latest invoice date.
3. A fresh 6-stage reminder cycle is scheduled.
4. **Isolation Guarantee:** Family members sharing the same phone number or patients on different medications are never affected by a single cycle reset.

---

## 9. Real Outcome Collection

As the pilot progresses, RefillCare collects verified outcomes by joining the persistent SQLite audit table against daily POS transactions:
- `reminder_id`, `customerId`, `itemId`, `pilot_tier`
- `reminder_date`, `expected_refill_date`, `reminder_stage`
- `actual_refill_date`, `days_to_refill`, `prediction_error_days`
- `timing_category` (`PRE-REFILL`, `SAME-DAY`, `POST-REFILL FOLLOW-UP`, `PENDING OBSERVATION`)
- `observation_status` (`Observed Refill`, `Pending Observation`, `No Observed Refill`)

---

## 10. Observation Window & Right-Censoring

```
Reminder Dispatched ──▶ [Active 30-Day Window: PENDING OBSERVATION] ──▶ [30d Elapsed: CLOSED WINDOW]
```

- **Active Window:** Dispatches with elapsed time $\le 30\text{ days}$ and no repurchase are tracked as `Pending Observation (Window Open)`.
- **Closed Window:** Dispatches with elapsed time $> 30\text{ days}$ and no repurchase are tracked as `No Observed Refill (Window Closed)`.

---

## 11. Daily Operational Metrics

The operational dashboard in [app_refillcare.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/app_refillcare.py) tracks:
- **Queue Metrics:** Total Candidates, Tier A Candidates, Reviewed, Approved, Rejected.
- **Dispatch Metrics:** Dispatched, Dry-Run Accepted, Live Accepted, Failed, Duplicates Blocked.
- **Adherence & Error Metrics:** Observed Refills, Pending Observation, Post-Reminder Refill Rate (%), Empirical Refill MAE (Days), $\pm 7\text{d}$ Accuracy (%).

---

## 12. Operator Clinical Feedback

Pharmacists record operational feedback to guide continuous improvement:
- `prediction_looked_correct`
- `prediction_looked_too_early`
- `prediction_looked_too_late`
- `customer_already_purchased`
- `customer_should_not_receive_reminder`
- `contact_issue`
- `medicine_issue`
- `pos_data_issue`
- `other`

---

## 13. Automated Pilot Health Checks (10 Warnings)

RefillCare continuously evaluates 10 operational safety assertions:
1. `HEALTH_01_NO_CANDIDATES`: Empty candidate queue on pilot date.
2. `HEALTH_02_HIGH_REJECTION_RATE`: Operator rejection rate $> 30\%$.
3. `HEALTH_03_HIGH_DISPATCH_FAILURE_RATE`: Provider dispatch failure rate $> 10\%$.
4. `HEALTH_04_DUPLICATE_ATTEMPTS`: Detection of duplicate dispatch requests.
5. `HEALTH_05_STALE_REMINDERS`: Detected purchases occurring before reminder dispatch.
6. `HEALTH_06_LARGE_PREDICTION_ERRORS`: Observations with absolute error $\ge 14$ days.
7. `HEALTH_07_EXCESSIVE_LATE_FOLLOWUPS`: $+2\text{d}$ and $+5\text{d}$ stages representing $> 40\%$ of dispatches.
8. `HEALTH_08_STALE_POS_DATA`: Ingestion latency $> 3$ days between latest transaction and evaluation date.
9. `HEALTH_09_AUDIT_DISCREPANCIES`: Scheduler vs SQLite audit mismatch.
10. `HEALTH_10_IDENTITY_COLLISION`: Shared phone number with mismatched customer identities.

---

## 14. Privacy & Information Security

- **Masked Phone Compliance:** Phone numbers are strictly masked (`***1234`) across all UI tables, audit logs, and exports.
- **Zero Credentials:** API keys, secrets, and auth tokens are never logged or exported.
- **Audit Persistence:** Data is stored locally in SQLite (`data/refillcare/processed/refillcare.db`).

---

## 15. Model Freeze Policy

The Phase 4 trained XGBoost model bundle (`refill_model.joblib`) remains **strictly frozen**:
- No hyperparameter tuning
- No feature modifications
- No automated retraining during initial operations

---

## 16. Reminder Schedule Freeze Policy

The deterministic 6-stage reminder cycle remains **strictly frozen**:
- `[-7, -3, -1, 0, +2, +5]` relative days
- Template text and variable mappings remain untouched

---

## 17. Test Strategy & Validation Results

The Phase 14 operational test suite ([tests/refillcare/test_pilot_operations.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/tests/refillcare/test_pilot_operations.py)) validates all 9 operational scenarios:
- **Scenario A:** Approved recurring customer $\rightarrow$ observed refill.
- **Scenario B:** Operator rejection $\rightarrow$ dispatch suppressed.
- **Scenario C:** Pre-reminder purchase $\rightarrow$ cycle reset.
- **Scenario D:** Shared phone $\rightarrow$ independent customer outcomes.
- **Scenario E:** Accepted reminder $\rightarrow$ pending observation.
- **Scenario F:** Pre-reminder purchase $\rightarrow$ stale reminder detection.
- **Scenario G:** Duplicate send $\rightarrow$ blocked by SQLite audit.
- **Scenario H:** Failed send $\rightarrow$ manual retry only.
- **Scenario I:** Multi-medicine patient $\rightarrow$ independent cycles.

---

## 18. Real Pilot vs. Test Fixture Data

> [!CAUTION]
> **Data Integrity Distinction:**
> - **Test Fixture Results:** Used solely within unit test suites to verify matching logic.
> - **Real Pilot Results:** As live messaging has not yet been executed in production, empirical pilot evidence is currently:
>   $$\text{Real Pilot Outcomes Status} = \text{NOT YET AVAILABLE}$$
> - The system explicitly avoids synthesizing artificial pilot outcome statistics.

---

## 19. Live-Send Authorization Boundary

Live messaging is protected by an explicit governance boundary:
- Real sending is **NOT** triggered by test runs, application startup, or phase progression.
- Live WhatsApp API requests occur only when an operator explicitly disables Dry-Run, checks authorization, and clicks dispatch during an active authorized session.

---

## 20. Operational Runbook

Pharmacists follow the 12-step daily operating procedure:

```
STEP 1:  Launch RefillCare (Streamlit UI).
STEP 2:  Click '🔄 Refresh Data' to pull latest POS transactions.
STEP 3:  Inspect 'Operations & Audit Summary' and Health Alerts banner.
STEP 4:  Open 'Controlled WhatsApp Dispatch' tab and set Target Due Date.
STEP 5:  Review due Tier A candidates and inspect 'Pre-Send Safety' status.
STEP 6:  Approve clinically appropriate regimens; reject inappropriate ones.
STEP 7:  Record structured rejection reasons where applicable.
STEP 8:  Confirm Dry-Run simulation toggle (or Live mode if authorized).
STEP 9:  Check explicit authorization confirmation checkbox.
STEP 10: Click Dispatch and inspect batch outcome table.
STEP 11: In 'Pilot Outcomes & Analytics', review matched purchases and MAE.
STEP 12: Export Daily Pilot Summary (CSV) for governance logging.
```

---

## 21. Known Limitations

1. **POS Ingestion Latency:** Sales data uploaded with delay may cause stale reminder dispatches. *(Mitigated by UI manual data refresh trigger).*
2. **External / Unaffiliated Purchases:** Refills occurring at competitor pharmacies cannot be observed in local POS data. *(Mitigated by right-censoring).*

---

## 22. Next Steps & Recommendation

**RefillCare Phase 14 is COMPLETE and FULLY VALIDATED.**

- **Controlled Pilot Operations:** **READY**
- **Real Pilot Data:** **NOT YET AVAILABLE** *(Awaiting initiation of real pilot operations)*
- **Model & Schedule:** **FROZEN**
- **Safety Verification:** **PASS** *(128 tests passing, 0 live requests, 0 automatic sending)*
- **Pilot Governance Decision:** **INSUFFICIENT DATA** *(Maintain Tier A restriction under Dry-Run simulation until real pilot dispatches occur)*
