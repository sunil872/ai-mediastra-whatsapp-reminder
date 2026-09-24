# Phase 17D — Reminder-System Business Metrics

**Date:** 2026-09-17  
**Status:** Offline evaluation only — **0 WhatsApp messages sent**; production code/model/reminder logic untouched  
**Script:** `scripts/phase17d_business_metrics.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_business_metrics_results.json`  
**Strategies:** Best Phase 17D predictors from `PHASE_17D_HYBRID_COMPARISON`

---

## 1. Purpose

Translate Phase 17D prediction quality into **reminder-operations language**:

| Business question | Metric |
|---|---|
| How far is the predicted refill date from truth? | Predicted refill date error (MAE / MedAE) |
| Would staged reminders (−3d / −7d) land near the true refill? | ±3d / ±7d accuracy |
| How many refill events would get an automated reminder opportunity? | Reminder opportunity coverage |
| Who is intentionally left out? | Customers / events excluded by eligibility |
| Would we nudge patients too soon? | Potential early reminders (`pred < actual`) |
| Would we nudge after they already refilled? | Potential late reminders (`pred > actual`) |
| When does the date call fail operationally? | `|error| > 7` or `> 14`, or no prediction emitted |

**Important:** Early/late are **counterfactual date-bias flags** from holdout truth. No messages were sent.

---

## 2. Evaluation setup

| Control | Value |
|---|---|
| Test set | `test.parquet`, `target > 0` (**7,488** events) |
| Dates | **2026-07-01 → 2026-08-30** |
| Unique customers | **1,034** |
| Unique customer–item pairs | **4,611** |
| WhatsApp sends | **0** |

### Strategies evaluated

| Strategy | Role |
|---|---|
| Current production XGBoost | Live `reg:squarederror` model (read-only) |
| Personal historical median | Strong simple baseline |
| Best personal cadence | Tiered cadence heuristic (30d pack on low history) |
| Best experimental XGBoost | In-memory `quantileerror` q50 |
| Hybrid routing | Eligibility-gated: core → median; secondary → experimental XGB |

---

## 3. Metric definitions

| Term | Definition |
|---|---|
| **Predicted refill date error** | `\|predicted_days − actual_days\|` (same as calendar-date offset error) |
| **Potential early reminder** | `predicted < actual` → predicted refill date is **before** true next purchase (risk of premature nudges) |
| **Potential late reminder** | `predicted > actual` → predicted refill date is **after** true next purchase (risk of post-purchase / missed-window nudges) |
| **Prediction failure (±7)** | Among emitted predictions, `\|error\| > 7` (misses the ±7d reminder window) |
| **Prediction failure (±14)** | Among emitted predictions, `\|error\| > 14` (severe miss) |
| **No-emit failure** | Eligibility reject — **no automated reminder opportunity** (hybrid only) |
| **Reminder opportunity coverage** | % of test events with an emitted prediction |

Signed-error convention: `pred − actual`. Negative ⇒ early; positive ⇒ late.

---

## 4. Customer segments

| Segment | Definition | Events | % | Unique customers |
|---|---|---:|---:|---:|
| **Eligible recurring** | Hybrid-eligible: `P≥6`, NormMAD≤0.50, Drift≤10, hist median ∈[15,120], `is_recurring_history=1` | 2,533 | 33.8% | 580 |
| **Low-history** | `purchase_count_so_far ≤ 3` | 1,699 | 22.7% | 558 |
| **Regular** | Class 1: `P≥6`, NormMAD≤0.35, Drift≤7 | 2,789 | 37.3% | 534 |
| **Irregular** | Class 2-like: `P≥6` and (NormMAD>0.35 **or** Drift>7) | 2,366 | 31.6% | 570 |

Segments can overlap (e.g. regular ∩ eligible recurring).

---

## 5. Overall reminder-system scorecard

### 5.1 Coverage & exclusion

| Strategy | Opportunity coverage (events) | Events excluded | Customers with opportunity | Customers excluded |
|---|---:|---:|---:|---:|
| Current XGBoost | 100% (7,488) | 0 | 1,034 (100%) | 0 |
| Personal hist median | 100% | 0 | 1,034 | 0 |
| Best personal cadence | 100% | 0 | 1,034 | 0 |
| Best experimental XGB | 100% | 0 | 1,034 | 0 |
| **Hybrid routing** | **33.8% (2,533)** | **4,955 (66.2%)** | **580 (56.1%)** | **454 (43.9%)** |

Hybrid also excludes **61.5%** of customer–item pairs (2,837 / 4,611).

### 5.2 Date accuracy & early/late risk (on emitted predictions)

| Strategy | Date error MAE | MedAE | ±3d | ±7d | Early % | Late % | Mean days early | Mean days late | Fail >7d | Fail >14d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid** | **7.77** | **5.0** | **38.7%** | **62.6%** | 34.2% | 59.9% | 6.4 | 9.3 | **37.4%** | **16.9%** |
| Best experimental XGB | 13.05 | 7.4 | 26.6% | 48.3% | 27.2% | 72.8% | 6.0 | 15.7 | 51.7% | 30.8% |
| Best personal cadence | 14.08 | 7.0 | 31.9% | 50.4% | 30.6% | 63.1% | 7.5 | 18.7 | 49.6% | 30.2% |
| Personal hist median | 26.72 | 7.0 | 32.0% | 51.0% | 31.3% | 62.6% | 7.6 | 38.9 | 49.0% | 30.8% |
| Current XGBoost | 43.19 | 19.6 | 9.9% | 23.5% | **8.9%** | **91.1%** | 5.9 | **46.8** | **76.5%** | **59.1%** |

### 5.3 How to read early vs late

- **Current XGBoost is systematically late** (91% late; mean **+47 days** when late) → predicted refill dates are too far out; −7/−3 stage reminders fire after many patients already returned.
- **Median / cadence / hybrid** are more balanced (~30–35% early, ~60% late) with much smaller late magnitude on eligible traffic.
- **Experimental XGB** still leans late (73%) but with smaller late magnitude than production XGB (mean late **16d** vs **47d**).

### 5.4 “Ops miss” vs intentional exclusion

| Strategy | No-emit (excluded) | Emitted but outside ±7 | Notes |
|---|---:|---:|---|
| Current XGBoost | 0 | 5,726 (76.5% of universe) | High **bad-reminder** risk if always-on |
| Best experimental XGB | 0 | 3,872 (51.7%) | Better, still ~half outside ±7 |
| Best personal cadence | 0 | 3,714 (49.6%) | Similar always-on miss rate |
| Personal hist median | 0 | 3,672 (49.0%) | Same |
| Hybrid | 4,955 (66.2%) | 948 (12.7% of universe) | Most “misses” are **chosen non-sends**, not bad sends |

For reminder product design, **hybrid’s no-emit is a feature** (protect patients from spam). Do not equate exclusion with a failed WhatsApp send.

---

## 6. Segment deep-dive

### 6.1 Eligible recurring customers (n=2,533 events, 580 customers)

Primary automated-reminder audience.

| Strategy | Coverage in segment | Date MAE | ±3d | ±7d | Early % | Late % | Fail >7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Best experimental XGB** | 100% | **7.39** | 35.8% | **62.5%** | 35.7% | 64.3% | **37.5%** |
| Hybrid | 100% | 7.77 | **38.7%** | **62.6%** | 34.2% | 59.9% | 37.4% |
| Personal median / cadence | 100% | 8.04 | ~38% | ~62% | ~35% | ~58% | ~38% |
| Current XGBoost | 100% | 14.33 | 14.4% | 35.5% | 13.2% | 86.8% | 64.5% |

**Takeaway:** Even on the best audience, production XGB is ~2× worse on date error and mostly late. Experimental XGB / hybrid are operationally usable (~63% within ±7d).

---

### 6.2 Low-history customers (n=1,699 events, 558 customers)

| Strategy | Coverage | Date MAE | ±3d | ±7d | Early % | Late % | Fail >7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Best personal cadence** (30d pack) | 100% | **14.96** | 14.1% | 24.7% | 18.0% | 79.0% | 75.3% |
| Best experimental XGB | 100% | 25.34 | 9.2% | 19.9% | 11.5% | 88.5% | 80.1% |
| Personal hist median | 100% | 70.29 | 16.4% | 28.3% | 22.5% | 75.3% | 71.8% |
| Current XGBoost | 100% | **116.85** | 0.0% | 0.2% | 0.1% | **99.9%** | **99.8%** |
| **Hybrid** | **0%** | — | — | — | — | — | — (all excluded) |

**Takeaway:** Hybrid correctly **excludes 100%** of low-history events. Always-on production XGB is essentially a prediction failure here (almost all late by ~117 days). Cadence’s fixed 30d pack is the least-bad always-on option, but still not reminder-grade (±7d only ~25%).

---

### 6.3 Regular customers (n=2,789 events, 534 customers)

| Strategy | Coverage | Date MAE | ±3d | ±7d | Early % | Late % | Fail >7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| Personal median / cadence | 100% | **6.14** | **~51%** | **~73.5%** | ~38% | ~51% | **~26%** |
| Best experimental XGB | 100% | **6.00** | 43.3% | 71.5% | 38.3% | 61.6% | 28.5% |
| Hybrid (scored 1,819 / 65%) | 65.2% | 7.39 | 43.8% | 66.7% | 35.9% | 55.9% | 33.3% |
| Current XGBoost | 100% | 11.68 | 18.7% | 44.8% | 15.2% | 84.8% | 55.3% |

**Takeaway:** Regular patients are the **gold reminder cohort**. Personal median wins ±3d/±7d; experimental XGB matches MAE. Hybrid covers ~65% of regulars (others fail the 15–120d cadence band or Candidate-5 outer gate) with solid emitted quality.

---

### 6.4 Irregular customers (n=2,366 events, 570 customers)

| Strategy | Coverage | Date MAE | ±3d | ±7d | Early % | Late % | Fail >7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best experimental XGB | 100% | 11.49 | 21.8% | 44.4% | 27.1% | 72.9% | 55.6% |
| Personal median / cadence | 100% | ~16.5 | ~24% | ~44% | ~32% | ~65% | ~56% |
| Hybrid (scored 714 / 30%) | **30.2%** | **8.74** | 25.9% | **52.0%** | 29.7% | 70.3% | **48.0%** |
| Current XGBoost | 100% | 25.26 | 9.1% | 20.9% | 9.7% | 90.3% | 79.1% |

**Takeaway:** Irregulars are high spam risk if always reminded. Hybrid only admits the better-behaved ~30% (Candidate-5 secondary) and improves ±7d among those admitted. The remaining ~70% should stay on pack defaults / manual review.

---

## 7. Reminder-product interpretation

### What “good” looks like for RefillCare WhatsApp stages
Stages are anchored on predicted refill date (`−7, −3, −1, 0, +2, +5` in scheduler). Business usefulness requires:

1. Enough **opportunity coverage** among chronic recurrers  
2. High **±7d** (so −7/−3 stages still near the true window)  
3. Controlled **late bias** (production XGB’s +47d late mean breaks the stage ladder)  
4. Explicit **exclusion** of low-history / chaotic timelines (avoid false reminders)

### Strategy verdicts (reminder lens)

| Strategy | Reminder verdict |
|---|---|
| **Hybrid routing** | **Best precision product:** ~34% event coverage / ~56% of customers get ≥1 opportunity; emitted MAE 7.8d; ±7d 63%; excludes low-history entirely |
| **Best experimental XGB** | **Best always-on learner:** use if pharmacy demands near-full coverage; still ~52% outside ±7 overall — pair with soft eligibility in practice |
| **Best personal cadence** | **Best always-on heuristic:** protects low-history via 30d pack; overall ±7d ~50% |
| Personal hist median | Strong on regulars; unsafe as always-on due to cold-start / long-gap RMSE |
| Current production XGB | **Not reminder-ready as always-on:** 91% late, 76% outside ±7, catastrophic on low-history |

---

## 8. Artifact & messaging safety

| Check | Result |
|---|---|
| WhatsApp messages sent | **0** |
| Production model modified | **No** (`refill_model.joblib` mtime unchanged) |
| Reminder scheduler / dispatch modified | **No** |
| Production code modified | **No** |

Outputs only:

- `docs/PHASE_17D_BUSINESS_METRICS.md` (this file)
- `data/refillcare/processed/phase17d_business_metrics_results.json`
- `scripts/phase17d_business_metrics.py`

---

## 9. Recommendation (non-binding)

1. **Pilot automated WhatsApp only on hybrid-eligible recurring patients** (~34% of events / ~56% of customers in this holdout).  
2. Prefer **hybrid** or **experimental quantile XGB on the eligible set** for predicted refill dates.  
3. **Do not** auto-remind low-history customers with the current production XGB.  
4. Treat irregular non-eligible patients as **excluded by design**, not as model failures.  
5. Keep production WhatsApp dispatch frozen until an explicit authorized rollout adopts these gates.

**No messages were sent in this phase.**
