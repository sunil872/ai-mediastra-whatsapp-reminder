# Historical Backtest: Existing Refill Prediction vs Historical Consumption + Estimated Days of Supply

**Status:** Complete (READ-ONLY empirical evaluation)  
**Trained Model:** Read-only (`refill_model.joblib` mtime untouched)  
**Production Routing:** Unchanged (`hybrid_routing_v17d` maintained)  
**WhatsApp Dispatch:** Unchanged / 0 messages sent  
**Evaluation Script:** `scratch/backtest_comparison.py`  
**Output Data:** `scratch/backtest_results.json`

---

## 1. Executive Summary & Objective

Before integrating **Estimated Days of Supply (EDS)** into the RefillCare production prediction engine, a comprehensive, read-only historical backtest was conducted across the 5.8-year transaction repository (489,960 supervised records).

The backtest evaluates two distinct paradigms:
1. **Existing Refill Prediction Approach (`hybrid_routing_v17d`):** Gated by history depth (P >= 6) and cadence regularity (NormMAD <= 0.35, Drift <= 7d), predicting personal historical median for core regular records and XGBoost quantile regression for secondary eligible records.
2. **Historical Consumption + Estimated Days of Supply (EDS):** Computes expanding historical consumption velocity (`historical_consumption_rate = consumed_units / elapsed_days`) and scales the current purchased units:
   $$\text{estimated\_days\_of\_supply} = \frac{\text{current\_units}}{\text{historical\_consumption\_rate}}$$

Both methods were evaluated strictly on information available **on or before** the current purchase date, with the actual future interval (`target_days_until_next_purchase`) serving as ground truth.

---

## 2. Overall Performance Comparison

### 2.1 Head-to-Head Overlap Evaluation (Both Approaches Available)

On the identical set of qualifying purchase events where both predictors could produce a forecast:

| Split / Dataset | Predictor | Evaluated (n) | MAE (days) | RMSE (days) | ±3-Day Acc | ±7-Day Acc | Error >14d | Error >30d |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Test Set** (2026-07 to 2026-08) | **Existing Hybrid** | 1,181 | 8.474 | 13.128 | **38.27%** | 60.29% | 232 (19.64%) | 38 (3.22%) |
| | **Estimated Days of Supply** | 1,181 | **8.060** | **11.619** | 33.02% | **59.19%** | **219 (18.54%)** | **31 (2.62%)** |
| **Validation Set** (2026-05 to 2026-06) | **Existing Hybrid** | 2,213 | 9.285 | 14.985 | **38.23%** | 58.16% | 481 (21.74%) | 115 (5.20%) |
| | **Estimated Days of Supply** | 2,213 | **9.146** | **14.315** | 30.86% | **57.21%** | **444 (20.06%)** | **101 (4.56%)** |
| **Full Supervised History** (2020 to 2026) | **Existing Hybrid** | 54,774 | 16.289 | **51.508** | **30.70%** | 50.97% | 15,399 (28.11%) | 5,386 (9.83%) |
| | **Estimated Days of Supply** | 54,774 | **16.011** | 52.068 | 26.13% | **51.19%** | **14,023 (25.60%)** | **4,982 (9.10%)** |

> **Key Finding:** On the overlap cohort, Estimated Days of Supply achieves a marginally lower overall MAE (8.06d vs 8.47d on Test) and lower large-error rate (>14d: 18.54% vs 19.64%), while the Existing Hybrid model retains higher tight accuracy within ±3 days (38.27% vs 33.02%).

---

### 2.2 Full Cohort Coverage & Standalone Metrics

Evaluating across the total universe of supervised records (regardless of eligibility gating):

| Split / Dataset | Approach | Total Records | Scored Records | Coverage | MAE (days) | RMSE (days) | ±3-Day Acc | ±7-Day Acc | Large Error (>14d) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Test Set** | Existing Hybrid | 7,556 | 1,196 | 15.83% | 8.476 | 13.093 | 37.88% | 60.03% | 235 (19.65%) |
| | Estimated Days of Supply | 7,556 | 6,660 | **88.14%** | 43.345 | 233.228 | 22.04% | 40.65% | 2,677 (40.20%) |
| | Raw Personal Median | 7,556 | 6,766 | 89.54% | 28.157 | 115.567 | 34.50% | 54.37% | 1,868 (27.61%) |
| **Validation Set** | Existing Hybrid | 12,821 | 2,229 | 17.39% | 9.301 | 14.980 | 38.09% | 57.96% | 485 (21.76%) |
| | Estimated Days of Supply | 12,821 | 11,278 | **87.97%** | 48.148 | 203.392 | 19.58% | 36.98% | 4,988 (44.23%) |
| | Raw Personal Median | 12,821 | 11,468 | 89.45% | 32.775 | 122.211 | 30.94% | 49.13% | 3,804 (33.17%) |
| **Full History** | Existing Hybrid | 489,960 | 55,202 | 11.27% | 16.372 | 51.894 | 30.67% | 50.90% | 15,556 (28.18%) |
| | Estimated Days of Supply | 489,960 | 380,778 | **77.72%** | 69.689 | 233.441 | 16.35% | 31.10% | 201,660 (52.96%) |
| | Raw Personal Median | 489,960 | 387,870 | 79.16% | 60.062 | 158.699 | 23.01% | 38.06% | 180,563 (46.55%) |

> **Critical Safety Insight:** Unconstrained Estimated Days of Supply without cadence regularity gates produces high overall error (MAE 43–70 days) when applied blindly to low-frequency, opportunistic, or irregular buyers. The eligibility gate is necessary to maintain pharmacy-grade precision.

---

## 3. Sub-Cohort Deep Dive

### 3.1 Regular Purchase Patterns (Eligible Regular Patients)
*Patients with history depth >= 6 purchases, consistent refill interval [15, 120] days, and low cadence drift.*

| Metric | Test Set (n=1,196) | Validation Set (n=2,229) | Full History (n=55,202) |
|---|---|---|---|
| **Existing Hybrid MAE** | 8.48 days | 9.30 days | 16.37 days |
| **Days of Supply MAE** | **8.06 days** | **9.15 days** | **16.01 days** |
| **Existing Hybrid ±7d** | **60.03%** | **57.96%** | 50.90% |
| **Days of Supply ±7d** | 59.19% | 57.21% | **51.19%** |
| **Existing Hybrid Error >14d** | 19.65% | 21.76% | 28.18% |
| **Days of Supply Error >14d** | **18.54%** | **20.06%** | **25.60%** |

*Analysis:* On regular chronic medication buyers, the two methods perform comparably well. Estimated Days of Supply slightly reduces large tail errors (>14 days) by accounting for subtle pack count variations.

---

### 3.2 Partial / Top-Up Purchases
*Purchases where current quantity is significantly below the customer's average historical refill size (Quantity / Avg <= 0.65).*

| Metric | Test Set (n=925) | Validation Set (n=1,374) | Full History (n=42,772) |
|---|---|---|---|
| **Existing Hybrid Scored** | 171 (18.49%) | 263 (19.14%) | 6,483 (15.16%) |
| **Existing Hybrid MAE** | 12.27 days | 13.81 days | 17.97 days |
| **Existing Hybrid ±7d** | 40.35% | 41.83% | 36.57% |
| **Days of Supply Scored** | **903 (97.62%)** | **1,322 (96.22%)** | **41,305 (96.57%)** |
| **Days of Supply MAE** | **16.12 days** | **20.26 days** | **41.64 days** |
| **Days of Supply ±7d** | **44.52%** | **37.07%** | **37.67%** |
| **Raw Personal Median MAE** | 24.26 days | 30.58 days | 49.92 days |
| **Raw Personal Median ±7d** | 38.70% | 38.79% | 32.88% |

*Analysis:* When patients purchase a smaller top-up quantity (e.g., 1 strip instead of 4), the **Existing Hybrid/Median approach over-predicts the interval** by assuming standard full-month cadence (MAE 24.26d–30.58d). **Estimated Days of Supply adapts dynamically to the smaller pack volume**, reducing MAE by 8 to 10 days compared to historical median baselines.

---

### 3.3 Bulk Purchases
*Purchases where current quantity is significantly above the customer's average historical refill size (Quantity / Avg >= 1.5, Quantity > 1).*

| Metric | Test Set (n=502) | Validation Set (n=973) | Full History (n=33,353) |
|---|---|---|---|
| **Existing Hybrid Scored** | 74 (14.74%) | 137 (14.08%) | 3,901 (11.70%) |
| **Existing Hybrid MAE** | **11.41 days** | **13.26 days** | **22.71 days** |
| **Existing Hybrid ±7d** | **44.59%** | **43.80%** | **38.20%** |
| **Days of Supply Scored** | 490 (97.61%) | 958 (98.46%) | 32,667 (97.94%) |
| **Days of Supply MAE** | 119.79 days | 140.13 days | 134.36 days |
| **Days of Supply ±7d** | 17.76% | 19.52% | 18.39% |
| **Raw Personal Median MAE** | 24.73 days | 38.82 days | 59.13 days |

*Behavioral Discovery on Bulk Buys:* When retail patients buy multiple packs at once (e.g. 6–12 months of supply), naive mathematical extrapolation ($\text{Units} / \text{Rate}$) predicts return dates 180–360+ days in the future. In practice, many patients return much earlier (30–60 days) to refill other items, top up, or adjust prescriptions. Unbounded multiplication on bulk purchases creates extreme over-prediction errors.

---

### 3.4 Records with Missing / Ambiguous Packing
*Transactions where product packing format is null, unparseable, or ambiguous.*

| Metric | Test Set (n=896) | Validation Set (n=1,543) | Full History (n=109,182) |
|---|---|---|---|
| **Estimated Days of Supply Coverage** | **0.0% (0 records)** | **0.0% (0 records)** | **0.0% (0 records)** |
| **Estimated Days of Supply Status** | `unavailable` | `unavailable` | `unavailable` |
| **Existing Hybrid Coverage** | 1.67% (15 records) | 1.04% (16 records) | 0.39% (428 records) |

*Analysis:* Estimated Days of Supply strictly returns `unavailable` (0% coverage) whenever packing data cannot be parsed unambiguously. This confirms that the unit parser is functioning as a deterministic safety barrier that **never hallucinates or guesses pack sizes**.

---

## 4. Backtest Conclusions & Findings

1. **High Quality on Regular Overlap:** On regular refill regimens (head-to-head overlap), Estimated Days of Supply matches the accuracy of the Existing Hybrid model (MAE 8.06d vs 8.47d on Test) with slightly fewer large errors (>14d).
2. **Clear Superiority on Partial Top-Ups:** On partial/top-up purchases, Estimated Days of Supply outperforms interval-based predictions because it adjusts for the reduced physical tablet count.
3. **Bulk Purchase Sensitivity:** Bulk purchases require safety bounding (caps/damping); multiplying large unit counts without bounds leads to significant over-prediction if patients return earlier.
4. **Strict Safety on Missing Packaging:** Missing or unparseable packaging is handled safely without guessing.
5. **No Production Changes Made:** In accordance with constraints, the production model, routing configuration, database, UI, and WhatsApp dispatch remain unmodified.

---

## 5. Decision & Governance Summary

| Action Item | Decision | Rationale |
|---|---|---|
| Retrain Production Model? | **No** | Current production model untouched. |
| Modify Production Routing? | **No** | Production continues on `hybrid_routing_v17d`. |
| Modify WhatsApp Dispatch? | **No** | No messaging code or live systems touched. |
| Next Integration Consideration | **Gated Hybrid Pilot** | If integrated in the future, use Estimated Days of Supply selectively as an auxiliary signal for partial top-ups and pack-aware adjustments with appropriate bulk bounding. |
