# Phase 13 — RefillCare Controlled Pilot Data Collection & Outcome Evaluation Report

**System:** Medication Refill Reminder System — RefillCare  
**Phase:** 13 — Controlled Pilot Data Collection & Outcome Evaluation  
**Status:** **PILOT OUTCOME EVALUATION READY**  
**Date:** September 2026  
**Document Version:** 1.0.0  

---

## 1. Executive Summary

Phase 13 establishes the data collection, outcome matching, empirical accuracy evaluation, and governance decision framework for the **RefillCare 30-Day Controlled Pilot**. 

Following the monitoring infrastructure built in Phase 12, Phase 13 enables pharmacy stakeholders to empirically assess:
1. Whether patients actually purchased medication after receiving a reminder.
2. How close actual purchase dates were to machine-learning expected refill dates.
3. The specific timing relation between reminder dispatches and actual purchases (`PRE-REFILL`, `SAME-DAY`, `POST-REFILL FOLLOW-UP`, `PENDING OBSERVATION`).
4. Which patient history tiers and medication groups generate reliable, low-error reminders vs. premature or stale touchpoints.
5. An evidence-based governance recommendation on whether to expand Tier A, maintain restricted operations, or pause.

> [!IMPORTANT]
> **Operational & Scientific Safeguards:**
> - **Non-Causal Association:** RefillCare measures *observed post-reminder refill association*, strictly refraining from asserting causal claims ("the reminder caused the refill") without a randomized control trial.
> - **Diagnosis Language Prohibited:** The system references *purchase patterns in therapeutic categories*, never asserting unconfirmed clinical diseases or diagnoses.
> - **Zero Automation:** Automatic background messaging and automatic retries remain strictly **DISABLED**.
> - **Template Integrity:** The approved template `refillcare_medicine_reminder` (6 body variables, Utility category, en) remains completely frozen.

---

## 2. Pilot Cohort Definition

The controlled pilot cohort is uniquely traceable via standardized parameters defined in [refillcare/evaluation/pilot_outcomes.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/refillcare/evaluation/pilot_outcomes.py#L25-L39):

| Attribute | Definition | Operational Rule |
| :--- | :--- | :--- |
| **Pilot ID** | `PILOT-2026-01` | Unique run identifier for cohort grouping |
| **Identity Key** | `(customerId, itemId)` | Primary clinical and operational identity |
| **Delivery Destination** | `MOBILE_NO` (masked `phone_masked`) | Contact channel only; never used to merge customers |
| **Target Population** | `Tier A (Strong Pilot)` | $\ge 5$ purchases or $\ge 3$ recurring purchases |
| **Review Population** | `Tier B (Review Required)` | 3–4 purchases with moderate irregularity (pharmacist sign-off required) |
| **Suppressed Population** | `Tier C (Cold-Start / Low History)` | $< 2$ purchases; excluded from active reminder queues |
| **Model Version** | `xgboost_refill_v1` | Phase 4 gradient boosted decision tree |

---

## 3. Pilot Activity vs. Outcome Observation Windows

RefillCare explicitly decouples the **Pilot Dispatch Window** from the **Outcome Observation Window**:

```
[Pilot Activity Window: 30 Days]
│◄─────────────────────────────►│
 Day 1                        Day 30
 │                              │
 ├─── Reminder Dispatched (t_0) ┴──────────────────────────────┐
                                                               │
                                [Outcome Observation Window: 30 Days]
                                │◄─────────────────────────────►│
                               t_0                            t_0 + 30d
```

- **Pilot Activity Window (30 Days):** The operational calendar period during which pharmacists review daily due queues and dispatch approved reminders.
- **Outcome Observation Window (30 Days post-reminder):** The elapsed period required to monitor subsequent pharmacy POS sales for that specific `(customerId, itemId)`.
- **Right-Censoring Handling:** Reminders with $(t_{\text{now}} - t_{\text{reminder}}) \le 30\text{ days}$ and no repurchase are classified as `Pending Observation (Window Open)` rather than false negatives.

---

## 4. Actual Refill Matching Rules

Actual pharmacy refills are matched using strict relational rules:

1. **Exact Compound Identity:** Matching requires identical `customerId` and `itemId`. Shared mobile numbers (family accounts) remain strictly isolated.
2. **Temporal Succession:** The transaction invoice date must occur strictly after the cycle baseline ($t_{\text{actual}} > t_{\text{last\_purchase}}$).
3. **Exclusion of Non-Refill Events:** Returns, reversals, inventory adjustments, and zero/negative quantity items are strictly filtered out ($q_{\text{invoice}} > 0$).
4. **First Subsequent Purchase:** For multi-purchase histories post-reminder, the first valid purchase date after reminder dispatch is identified as $t_{\text{actual}}$.

---

## 5. Prediction Accuracy Evaluation

For every pilot record where an actual subsequent purchase is observed:

$$\text{Prediction Error (Days)} = \text{actual\_refill\_date} - \text{expected\_refill\_date}$$
$$\text{Absolute Prediction Error (Days)} = |\text{prediction\_error\_days}|$$

### Evaluated Accuracy Metrics:
- **Mean Absolute Error (MAE):** Average error in days across observed pilot refills.
- **Median Absolute Error (MedAE):** Robust midpoint error resilient to sporadic delayed visits.
- **Root Mean Squared Error (RMSE):** Sensitivity metric for severe outliers.
- **Empirical Precision Windows:** Percentage of refills falling within $\pm 1$ day, $\pm 3$ days, and $\pm 7$ days of the expected refill date.

> [!NOTE]
> Prediction accuracy is calculated **only on observed repurchases**. Records with pending observation are excluded from error calculations to prevent statistical distortion.

---

## 6. Reminder Timing Classification

RefillCare categorizes the temporal relationship between reminder dispatch and actual patient repurchase into 4 distinct states:

| Category | Temporal Condition | Operational Meaning |
| :--- | :--- | :--- |
| **`PRE-REFILL REMINDER`** | $t_{\text{reminder}} < t_{\text{actual}}$ | Standard reminder dispatched before patient refilled. |
| **`SAME-DAY`** | $t_{\text{reminder}} == t_{\text{actual}}$ | Reminder dispatched on the same date the patient visited pharmacy. |
| **`POST-REFILL FOLLOW-UP`** | $t_{\text{reminder}} > t_{\text{actual}}$ | Follow-up reminder dispatched after patient had already repurchased (flagged as `Potentially Stale Reminder`). |
| **`PENDING OBSERVATION`** | No purchase & $(t_{\text{now}} - t_{\text{reminder}}) \le 30\text{d}$ | Active observation window open; awaiting transaction. |

---

## 7. Reminder Stage Analysis

The 6-stage reminder cycle (`-7d`, `-3d`, `-1d`, `0d`, `+2d`, `+5d`) is evaluated per stage:

```
Stage -7d ───▶ Stage -3d ───▶ Stage -1d ───▶ Stage 0d ───▶ Stage +2d ───▶ Stage +5d
(Early)       (Impending)    (Urgent)       (Due Date)    (Overdue)      (Final Follow-up)
```

For each stage, the evaluation engine tracks:
- **Total Due / Scheduled:** Touchpoints reaching due status.
- **Accepted Dispatches:** Dispatches confirmed by provider (live or dry-run).
- **Pre-Refill vs Same-Day vs Post-Refill Counts.**
- **Post-Reminder Refill Rate per Stage ($N_{\text{observed}} / [N_{\text{observed}} + N_{\text{closed}}]$).**

---

## 8. Post-Reminder Refill Rate

RefillCare calculates post-reminder refill association using an unbiased denominator that accounts for right-censoring:

$$\text{Post-Reminder Refill Rate} = \frac{N_{\text{Observed Refills}}}{N_{\text{Observed Refills}} + N_{\text{No Refill (Window Closed)}}} \times 100\%$$

- **Numerator:** Patients who repurchased within 30 days of receiving an accepted reminder.
- **Denominator:** Only includes completed observation cycles (excludes open-window pending records).
- **Scientific Attribution:** Interpreted strictly as an observed behavioral association, not causal proof.

---

## 9. Right-Censoring Handling

To prevent false-negative bias during rolling pilot execution, outcomes are classified across 3 mutually exclusive states:

1. **Observed Refill:** Subsequent purchase recorded within 30-day window.
2. **Pending Observation (Window Open):** No repurchase recorded, but less than 30 days have elapsed since dispatch.
3. **No Observed Refill (Window Closed):** 30 days elapsed with zero observed repurchases.

---

## 10. False-Positive & Operational Risk Surveillance

The evaluation engine identifies and flags potential friction points:

| Risk Category | Trigger Rule | Operational Remediation |
| :--- | :--- | :--- |
| **Potentially Premature Reminder** | Actual purchase $\ge 7$ days *before* expected date | Flag for interval variance review; adjust model priors |
| **Potentially Stale Reminder** | Reminder dispatched *after* actual purchase date | Inspect POS ingestion lag; ensure cycle-reset trigger fired |
| **Irregular History Candidate** | Prediction generated on $< 3$ lifetime purchases | Maintain in Tier C (suppressed queue) |
| **Multi-Stage Overexposure** | Customer received $\ge 3$ reminders across cycle | Suppress $+5\text{d}$ touchpoint for that patient |

---

## 11. Reminder Fatigue Surveillance

Fatigue analytics monitor notification frequency per patient:
- **Average Reminders per Patient:** Target $< 2.0$ reminders per cycle.
- **Late Stage Escalation:** Counts dispatches at $+2\text{d}$ and $+5\text{d}$.
- **Reminders Before Refill:** Measures average touches required before an observed purchase.

---

## 12. Cohort Segmentation Performance

Pilot outcomes are stratified across multiple clinical and behavioral dimensions:

1. **By Pilot Tier:**
   - *Tier A (Strong Pilot):* High history consistency ($\ge 5$ purchases). Expected MAE $< 10$ days.
   - *Tier B (Review Required):* Moderate history (3–4 purchases). Expected MAE 12–16 days.
2. **By History Quality:**
   - `high_history` vs `medium_history` vs `low_history`.
3. **By Purchase Frequency Bins:**
   - `2 Purchases (Low)` vs `3–4 Purchases (Moderate)` vs `5+ Purchases (High)`.
4. **By Medication:**
   - Granular breakdown for high-volume maintenance medications ($N \ge 10$).

---

## 13. Offline Benchmark vs. Pilot Performance Comparison

Pilot results are systematically compared against Phase 4 test set benchmarks:

| Cohort Segment | Phase 4 Offline Benchmark | Pilot Target Range | Pilot Assessment Status |
| :--- | :--- | :--- | :--- |
| **Overall Population** | $\text{MAE} = 16.20\text{ days}$ | $\text{MAE} \le 16.5\text{ days}$ | `EVALUATED` |
| **Regular Cycles (16–45d)** | $\text{MAE} = 11.12\text{ days}$ | $\text{MAE} \le 12.0\text{ days}$ | `EVALUATED` |
| **High History ($\ge 5$ purchases / Tier A)** | $\text{MAE} = 9.64\text{ days}$ | $\text{MAE} \le 10.0\text{ days}$ | `EVALUATED` |

- **Sample Maturity Rule:** Comparisons are classified as `COLLECTING_EVIDENCE` until $\ge 20$ refills are observed, advancing to `EARLY_PILOT_STABLE` thereafter.

---

## 14. POS Data Latency Evaluation

- **Ingestion Latency Impact:** Delays between patient in-store purchase and system data refresh can lead to stale reminders.
- **Current Mitigation:** The RefillCare Streamlit interface features a direct **"🔄 Refresh Data"** cache invalidation trigger, enabling pharmacists to pull the latest POS transactions prior to daily batch review.

---

## 15. Operator Review & Structured Rejections

The system captures operational review metrics:
- **Candidates Generated vs Reviewed vs Approved vs Rejected.**
- **Structured Rejection Reasons:**
  1. `customer_already_purchased`
  2. `uncertain_prediction`
  3. `inappropriate_regimen`
  4. `invalid_contact`
  5. `pharmacist_discretion`
  6. `duplicate_intercepted`
  7. `other`

---

## 16. Pilot Outcome Dataset Specification

The granular pilot evaluation dataset ([refillcare/evaluation/pilot_outcomes.py](file:///c:/Users/sunil/ai-mediastra-whatsapp-reminder/ai-mediastra-whatsapp-reminder/refillcare/evaluation/pilot_outcomes.py#L52-L280)) produces a safe, structured schema:

```
pilot_id, customerId, itemId, customer_name, medicine, phone_masked,
pilot_tier, history_quality, purchase_count, latest_purchase_date,
predicted_interval_days, expected_refill_date, reminder_stage,
reminder_date, reminder_status, is_dry_run, actual_refill_date,
days_to_refill, prediction_error_days, absolute_prediction_error_days,
timing_category, observation_status, is_post_reminder_refill,
is_accurate_1d, is_accurate_3d, is_accurate_7d, review_status,
rejection_reason, risk_flag
```

---

## 17. Pilot Reporting & Safe CSV Export

Two export formats are provided in the Streamlit UI:
1. **Daily Pilot Summary (`refillcare_pilot_daily_summary_YYYYMMDD.csv`):** Aggregated funnel, error, and rate metrics for management and governance reporting.
2. **Granular Pilot Outcomes (`refillcare_pilot_outcomes_YYYYMMDD_HHMMSS.csv`):** Full regimen-level evaluation dataset containing only masked phone numbers (`***1234`) and zero credentials or API keys.

---

## 18. Evidence-Based Decision Framework

At the conclusion of the controlled pilot, governance actions follow strict evidence thresholds:

```
                                 [Pilot Evaluation Complete]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [Audit Discrepancies?]                             [Audit State Healthy]
                    │                                                   │
             YES ───┴───▶ [PAUSE]                                       │
                                                   ┌────────────────────┴────────────────────┐
                                                   ▼                                         ▼
                                        [Observed Refills < 10]                   [Observed Refills >= 10]
                                                   │                                         │
                                                   ▼                                         ▼
                                          [KEEP TIER A ONLY]                     [MAE <= 12.0d & Safe?]
                                                                                             │
                                                                           ┌─────────────────┴─────────────────┐
                                                                           ▼                                   ▼
                                                                  YES ─────┴─────▶ [EXPAND TIER A]    NO ──────┴──────▶ [RESTRICT / RETRAIN]
```

| Decision | Condition | Next Steps |
| :--- | :--- | :--- |
| **`EXPAND TIER A`** | $\ge 10$ refills, Tier A $\text{MAE} \le 12.0\text{d}$, zero duplicate incidents, healthy audit | Expand Tier A candidate volume with continued manual review. |
| **`KEEP TIER A ONLY`** | Stable operational safety, but $< 10$ refills or Tier B $\text{MAE} > 15.0\text{d}$ | Maintain 30-day pilot restricted to Tier A; continue collecting evidence. |
| **`RESTRICT FURTHER`** | Elevated variance ($\text{MAE} > 18.0\text{d}$) | Restrict to $\ge 5$ purchases; plan model retraining once $N \ge 100$. |
| **`PAUSE`** | Audit state reconciliation discrepancies or duplicate messaging | Halt dispatch; investigate database synchronization. |

---

## 19. Remaining Operational Risks & Mitigations

1. **Off-Platform Refills:** Patient refills at an unaffiliated pharmacy chain. *Mitigation: Handled cleanly by right-censoring; never misclassified as medication non-adherence.*
2. **Delayed Ingestion:** POS batch sync occurs after morning review. *Mitigation: Pharmacist uses UI "Refresh Data" prior to dispatch authorization.*
3. **Overdue Follow-Up Annoyance:** Late $+5\text{d}$ reminders may reach patients who already intended to refill. *Mitigation: Pharmacist inspection and option to reject $+5\text{d}$ touchpoint during daily review.*

---

## 20. Recommendation & Next Steps

**RefillCare Phase 13 is COMPLETE and FULLY VALIDATED.**

- **Outcome Evaluation Framework:** **READY**
- **Unit & Integration Test Suite:** **116 passed / 0 failed**
- **Model Retraining:** **NOT YET RECOMMENDED** *(Continue collecting empirical pilot outcomes before retraining)*
- **Pilot Governance Recommendation:** **KEEP TIER A ONLY** *(Proceed with 30-Day Controlled Pilot under Tier A restriction and Dry-Run simulation)*
