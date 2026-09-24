# Phase 12 — RefillCare 30-Day Controlled Pilot Execution & Monitoring Report

**System:** Medication Refill Reminder System — RefillCare  
**Phase:** 12 — 30-Day Controlled Pilot Execution & Operational Monitoring  
**Status:** **PILOT MONITORING READY**  
**Date:** September 2026  
**Document Version:** 1.0.0  

---

## 1. Executive Summary & Objective

The objective of Phase 12 is to establish, implement, and rigorously validate the operational monitoring framework required for conducting a **30-day controlled pilot** of RefillCare.

Following Phase 11's classification of RefillCare as **Controlled Pilot Ready**, this phase provides the operational evidence instrumentation to measure the true day-to-day effectiveness, empirical prediction accuracy, notification fatigue, and audit consistency of the system.

> [!IMPORTANT]
> **Safety & Governance Mandate:**
> - The term "30-day pilot" denotes an **operator-controlled evaluation period**, NOT an automated daily broadcasting job.
> - **Zero** automatic sending, **zero** background cron jobs/workers, and **zero** automatic retries.
> - Real WhatsApp dispatch requires explicit human operator authorization per batch.
> - The approved template `refillcare_medicine_reminder` (6 body variables, Utility category) remains strictly frozen.

---

## 2. Pilot Configuration & Safeguards

The 30-day controlled pilot operates under standardized operational parameters defined in [refillcare/evaluation/pilot_monitoring.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/refillcare/evaluation/pilot_monitoring.py#L20-L38):

| Parameter | Configuration | Operational Rule |
| :--- | :--- | :--- |
| **Pilot Name** | `RefillCare 30-Day Controlled Pilot` | Standard run identifier |
| **Duration** | `30 Days` | Bounded operational observation window |
| **Mode** | `controlled_manual` | Human-in-the-loop approval mandatory |
| **Default Dispatch State** | `Dry-Run Simulation` | Safe default preventing accidental live sending |
| **Auto-Send Enabled** | `FALSE (Disabled)` | No background or headless dispatch permitted |
| **Auto-Retry Enabled** | `FALSE (Disabled)` | Failures require manual root-cause inspection |
| **Reminder Stages** | `[-7, -3, -1, 0, +2, +5]` | 6 deterministic relative touchpoints |
| **Max Reminders Per Cycle** | `3` | Caps excessive touchpoint fatigue |
| **Observation Window** | `30 Days` | Window to monitor post-reminder purchase events |

---

## 3. Target Candidate Population & Pilot Tiers

Candidate selection applies Phase 11's 3-tier risk-calibrated categorization:

1. **Tier A (Strong Pilot — Primary Target):**
   - Criteria: $\ge 5$ lifetime purchases, or $\ge 3$ purchases with recurring consistency (`is_recurring_history = 1`), or `high_history`.
   - Historical Metric Basis: MAE = **9.64 days**, $\pm 7$ days accuracy = **47.27%**.
   - Queue Status: Primary cohort for daily reminder review and operator dispatch.
2. **Tier B (Review Required — Secondary Cohort):**
   - Criteria: 3–4 purchases with moderate irregularity.
   - Historical Metric Basis: MAE = 14.8 days.
   - Queue Status: Pharmacist review of purchase interval before authorizing dispatch.
3. **Tier C (Suppressed / Cold-Start):**
   - Criteria: 1–2 purchases or invalid intervals.
   - Queue Status: Completely excluded from dispatch queues to safeguard patient trust.

---

## 4. Operational End-to-End Funnel

The RefillCare pilot funnel tracks every transaction and reminder lifecycle transition:

```
[1. Total Candidates] (Historical Purchase Histories)
         │
         ▼
[2. Eligible Regimens] (Tier A & Tier B Filtered)
         │
         ▼
[3. Reminders Due Today] (Reminder Date <= Evaluation Date)
         │
         ▼
[4. Operator Reviewed] (Pharmacist Inspection in App UI)
         │
         ▼
[5. Explicitly Approved] (Manual Confirmation Checkbox)
         │
         ▼
[6. Dispatch Attempted] (Logged to SQLite as 'sending')
         │
         ▼
[7. Provider Accepted] (Xinno WhatsApp API HTTP 200 / Dry-Run)
         │
         ▼
[8. Observed Refill] (Subsequent Purchase strictly > Reminder Date)
```

### Funnel Computation Metrics:
- **Eligible Regimens:** Regimens meeting clinical and history criteria.
- **Due Reminders:** Reminders scheduled on or before the current operational date.
- **Duplicates Intercepted:** Prior accepted reminders blocked from retransmission.
- **Provider Accepted (Live vs Dry-Run):** Explicit separation in SQLite audit.
- **Observed Refill:** Matched subsequent purchase within observation window.

---

## 5. Post-Reminder Refill Measurement & Non-Causal Attribution

### Precise Matching Rule
A refill is recorded if and only if:
1. **Strict Identity Match:** Transaction matches exact `customerId` AND `itemId`.
2. **Temporal Order:** Transaction invoice date occurs strictly *after* the reminder dispatch date ($t_{\text{refill}} > t_{\text{reminder}}$).
3. **Observation Bounded:** $1 \le (t_{\text{refill}} - t_{\text{reminder}}) \le 30\text{ days}$.

```
Reminder Dispatched (t_0) ───▶ [1 to 30 Days Window] ───▶ Repurchase (t_refill)
```

### Non-Causal Framing
> [!NOTE]
> **Scientific Integrity Notice:** A subsequent customer repurchase after receiving a reminder demonstrates **post-reminder refill association**, NOT causal proof that the reminder induced the purchase. RefillCare metrics explicitly use the term **"Post-Reminder Refill Rate"** and refrain from causal claims.

$$\text{Post-Reminder Refill Rate} = \frac{\text{Observed Refills}}{\text{Observed Refills} + \text{No Refill (Closed Window)}} \times 100\%$$

---

## 6. Right-Censoring Management

During the 30-day pilot, recent reminder dispatches will not yet have had sufficient elapsed calendar time for a customer to repurchase.

RefillCare avoids falsely labeling open-window reminders as failures through 3 distinct states:

| Classification | Condition | Reporting Impact |
| :--- | :--- | :--- |
| **Observed Refill** | Subsequent purchase recorded within 30 days | Counted as successful refill outcome |
| **Pending Observation (Window Open)** | No purchase yet, but $(t_{\text{now}} - t_{\text{reminder}}) \le 30\text{ days}$ | Excluded from denominator to prevent bias |
| **No Observed Refill (Window Closed)** | No purchase and $(t_{\text{now}} - t_{\text{reminder}}) > 30\text{ days}$ | Counted as non-refill outcome |

---

## 7. Empirical Accuracy Evaluation on Pilot Refills

When an actual subsequent purchase occurs during the pilot, RefillCare compares `expected_refill_date` against `actual_refill_date`:

- $\text{Prediction Error (Days)} = \text{actual\_refill\_date} - \text{expected\_refill\_date}$
- $\text{Absolute Error (Days)} = |\text{prediction\_error\_days}|$
- **MAE:** Mean of absolute error days across evaluated pilot repurchases.
- **Median Absolute Error:** Robust median error.
- **Accuracy Windows:** $\%$ within $\pm 1$ day, $\pm 3$ days, and $\pm 7$ days.

---

## 8. False-Positive & Notification Fatigue Surveillance

RefillCare monitors potential patient annoyance through dedicated surveillance metrics:

1. **Multi-Stage Exposure:** Tracks total dispatches per patient ($N_{\text{dispatches}} / N_{\text{patients}}$).
2. **Follow-Up Escalation:** Quantifies dispatches at overdue stages (`+2d`, `+5d`).
3. **High-Frequency Patients:** Flags patients receiving $\ge 3$ reminders across cycles.
4. **Early Purchase Interception:** Automatically identifies purchases occurring before stage `0d`.

---

## 9. Shared-Phone Isolation Validation

In retail pharmacy environments, family members frequently share a single mobile number. RefillCare strictly validates:

$$\text{Identity} = (\text{customerId}, \text{itemId})$$
$$\text{Delivery Destination} = \text{MOBILE_NO}$$

- Patient A purchasing Medicine X on Phone P and Patient B purchasing Medicine Y on Phone P maintain **completely isolated reminder schedules, audit records, and refill matching**.
- Transactions for Patient A never falsely satisfy or cancel Patient B's cycle.

---

## 10. Cycle-Reset Validation

When a patient makes a new pharmacy purchase:
1. All pending/future reminders for that `(customerId, itemId)` are marked `cancelled`.
2. The ML engine generates a new `expected_refill_date`.
3. A fresh 6-stage reminder cycle is initialized.
4. Cancelled reminders are strictly excluded from daily due queues and dispatch attempt counts.

---

## 11. Audit Reconciliation & State Integrity

The monitoring engine performs real-time reconciliation between the in-memory scheduler queue and persistent SQLite storage ([reminder/storage.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/reminder/storage.py)):

- **In-Sync Records:** Reminders properly committed in SQLite `reminder_audit`.
- **Missing DB Records:** Flags any unpersisted schedule items.
- **Stale In-Flight Records:** Detects reminders stuck in `sending` status.
- **Audit Health Indicator:** Reports `HEALTHY` or `ATTENTION_REQUIRED`.

---

## 12. Operator Daily Workflow & Dashboard

The pharmacy operator follows a standard 12-step daily operating procedure (SOP):

```
 1. Launch RefillCare (Streamlit UI).
 2. Inspect 'Operations & Audit Summary' header metrics.
 3. Open 'Pilot Monitoring & Analytics' tab to review funnel and audit health.
 4. Open 'Controlled WhatsApp Dispatch' tab and select Target Due Date.
 5. Review due recipient list, masked phones, and preview variables.
 6. Pharmacist verifies clinical appropriateness for Tier A / Tier B patients.
 7. Confirm Dry-Run simulation toggle (or Live mode if approved).
 8. Check explicit authorization checkbox.
 9. Click Dispatch.
10. Inspect batch outcome table and verify SQLite persistence.
11. In 'Audit & Delivery History', inspect any failed records for manual retry.
12. Export Daily Pilot Summary Report (CSV) for governance logging.
```

---

## 13. Pilot Decision Framework

At the conclusion of the 30-day evaluation period, stakeholders utilize the evidence-based decision framework:

| Outcome | Trigger Conditions | Recommended Action |
| :--- | :--- | :--- |
| **Outcome A: Expand** | - Tier A MAE $\le 10$ days and $\pm 7$d accuracy $\ge 45\%$<br>- Zero duplicate messages or safety violations<br>- Dispatch failure rate $< 2\%$<br>- High patient & pharmacist satisfaction | Expand pilot candidate volume to wider Tier A population. |
| **Outcome B: Restrict / Refine** | - Tier A performs well, but Tier B shows elevated error ($> 15$ days)<br>- Some follow-up stages (+5d) exhibit low conversion<br>- Minor operational friction | Restrict pilot to strict Tier A only; remove `+5d` stage. |
| **Outcome C: Pause / Stop** | - Systemic cycle-reset failures or duplicate messaging<br>- Unacceptable failure rate ($> 10\%$)<br>- Patient complaints regarding incorrect refill timing | Pause system immediately for re-engineering and model recalibration. |

---

## 14. Remaining Risks & Mitigations

1. **Phone Number Portability / Invalidation:**
   - *Risk:* WhatsApp message fails due to inactive number.
   - *Mitigation:* Explicit provider error categorization (`invalid_phone`, `provider_rejected`) logged to SQLite with manual retry support.
2. **Off-Platform Purchases:**
   - *Risk:* Patient purchases medicine at another pharmacy chain.
   - *Mitigation:* Handled cleanly via right-censoring; system does not falsely diagnose non-adherence.
3. **Data Freshness / Latency:**
   - *Risk:* POS transactions uploaded with multi-day delay.
   - *Mitigation:* Daily data refresh button in RefillCare UI ensures POS transactions sync before dispatch.

---

## 15. Final Recommendation

**RefillCare Phase 12 is COMPLETE and VALIDATED.**

The pilot monitoring engine, interactive analytics dashboard, post-reminder matching logic, and SQLite reconciliation have passed all unit, integration, and safety tests.

**Recommendation:** Proceed to initiate the 30-Day Controlled Pilot under **Tier A restricted mode** with default **Dry-Run simulation**.
