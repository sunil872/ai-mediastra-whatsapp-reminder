# Phase 17D: RefillCare Purchase-History Depth & Reliability Analysis

**Evaluation Date:** 2026-09-17  
**Dataset Version:** 5.8-Year Pharmacy Dataset (`data/refillcare/customer_data_fields.csv`)  
**Evaluation Scope:** Test Holdout Partition (`data/refillcare/processed/test.parquet`, $N=7,556$, `2026-07-01` to `2026-08-31`) and Validation Partition (`data/refillcare/processed/validation.parquet`, $N=12,821$, `2026-05-01` to `2026-06-30`).  
**Mode:** Read-Only Empirical Audit (No modifications to production code, datasets, or models).

---

## 1. Executive Summary

This study examines the relationship between **purchase-history depth** (the number of prior transactions recorded for a patient-medicine pair, `purchase_count_so_far`) and the **accuracy & clinical reliability** of next-purchase refill predictions.

```
========================================================================================================================
PURCHASE-HISTORY DEPTH BENCHMARK SUMMARY (Test Holdout: N=7,556)
========================================================================================================================
Depth Tier        Depth (P)    Test Count    Share (%)  | Base MAE  XGB MAE | Base MedAE  XGB MedAE | Base ±7d   XGB ±7d | Clinical Reliability
------------------------------------------------------------------------------------------------------------------------
Cold Start        1 Purchase      790        10.5%      |   16.94    159.96 |    17.00     166.10 |   18.99%     0.00% | ❌ UNRELIABLE
Single Interval   2 Purchases     517         6.8%      |  127.12     93.96 |    13.00      69.30 |   38.49%     0.00% | ❌ UNRELIABLE
Sparse History    3 Purchases     423         5.6%      |  101.93     67.60 |    15.50      47.21 |   30.26%     0.71% | ❌ UNRELIABLE
Early Transition  4 Purchases     339         4.5%      |   40.76     54.20 |    12.00      40.18 |   39.23%     2.36% | ⚠️ MARGINAL
Early Transition  5 Purchases     304         4.0%      |   37.02     49.13 |    13.50      38.92 |   33.88%     2.96% | ⚠️ MARGINAL
Established       6–10 Purchases 1,059       14.0%      |   20.15     33.13 |     8.00      23.08 |   49.10%    11.90% | 🟡 MODERATE
Deep Chronic      >10 Purchases  4,124       54.6%      |    8.58     14.14 |     5.00       9.48 |   62.83%    39.35% | ✅ HIGHLY RELIABLE
========================================================================================================================
```

### Core Conclusion on Prediction Feasibility:
- **Sufficient History Threshold:** **$\ge 6$ prior purchases** (and ideally **$>10$ purchases**) provides sufficient historical recurrence for reliable, automated WhatsApp refill reminders.
- **Deep Chronic Cohort ($>10$ purchases, 54.6% of test customers):** The Historical Median Baseline achieves **8.58 days MAE**, **5.00 days MedAE**, and **62.83% accuracy within $\pm 7$ days** (81.81% within $\pm 14$ days).
- **Combined Reliable Cohort ($\ge 6$ purchases, 68.6% of test customers):** Baseline achieves **10.95 days MAE**, **5.00 days MedAE**, and **60.02% accuracy within $\pm 7$ days**.
- **Sparse / Cold-Start Cohort ($\le 3$ purchases, 22.9% of test customers):** Historical variance and lack of cadence make individual regression predictions unreliable; these patients require fixed category-level reminders rather than automated personal timeline ML.

---

## 2. Granular Depth Performance Matrix (Test Holdout: $N=7,556$)

The table below details every discrete prior purchase count from $P=1$ to $P>10$:

| Prior Purchases ($P$) | Sample Count ($N$) | Cohort Share (%) | Observed Target Mean | Observed Target Median | Baseline MAE (days) | XGBoost MAE (days) | Baseline MedAE (days) | XGBoost MedAE (days) | Baseline $\pm 1$d (%) | XGBoost $\pm 1$d (%) | Baseline $\pm 3$d (%) | XGBoost $\pm 3$d (%) | Baseline $\pm 7$d (%) | XGBoost $\pm 7$d (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1 Purchase** | **790** | 10.46% | 17.04d | 14.0d | **16.94d** | 159.96d | **17.00d** | 166.10d | **4.56%** | 0.00% | **9.49%** | 0.00% | **18.99%** | 0.00% |
| **2 Purchases** | **517** | 6.84% | 18.49d | 16.0d | 127.12d | **93.96d** | **13.00d** | 69.30d | **13.54%** | 0.00% | **23.98%** | 0.00% | **38.49%** | 0.00% |
| **3 Purchases** | **423** | 5.60% | 18.75d | 17.0d | 101.93d | **67.60d** | **15.50d** | 47.21d | **9.22%** | 0.00% | **17.97%** | 0.00% | **30.26%** | 0.71% |
| **4 Purchases** | **339** | 4.49% | 18.98d | 17.0d | **40.76d** | 54.20d | **12.00d** | 40.18d | **13.57%** | 0.29% | **23.60%** | 0.88% | **39.23%** | 2.36% |
| **5 Purchases** | **304** | 4.02% | 18.88d | 18.0d | **37.02d** | 49.13d | **13.50d** | 38.92d | **7.89%** | 0.33% | **19.08%** | 0.99% | **33.88%** | 2.96% |
| **6–10 Purchases**| **1,059** | 14.02% | 20.21d | 18.0d | **20.15d** | 33.13d | **8.00d** | 23.08d | **14.26%** | 0.76% | **28.42%** | 3.49% | **49.10%** | 11.90% |
| **>10 Purchases** | **4,124** | **54.58%** | 18.80d | 17.0d | **8.58d** | 14.14d | **5.00d** | 9.48d | **21.90%** | 5.67% | **40.98%** | 17.00% | **62.83%** | 39.35% |

---

## 3. Validation Partition Cross-Check ($N=12,821$)

Evaluating the identical depth cohorts on the Validation partition (`2026-05-01` to `2026-06-30`) confirms the robustness and stability of the depth pattern:

| Prior Purchases ($P$) | Val Count ($N$) | Share (%) | Observed Target Mean | Baseline MAE (days) | XGBoost MAE (days) | Baseline MedAE (days) | XGBoost MedAE (days) | Baseline $\pm 3$d (%) | XGBoost $\pm 3$d (%) | Baseline $\pm 7$d (%) | XGBoost $\pm 7$d (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1 Purchase** | **1,353** | 10.55% | 30.70d | **19.29d** | 151.76d | **18.00d** | 155.78d | **13.16%** | 0.00% | **23.06%** | 0.00% |
| **2 Purchases** | **934** | 7.28% | 30.72d | 149.30d | **93.45d** | **22.00d** | 73.74d | **15.95%** | 0.64% | **26.98%** | 0.96% |
| **3 Purchases** | **768** | 5.99% | 31.40d | 106.96d | **65.73d** | **25.50d** | 50.63d | **13.93%** | 0.91% | **23.57%** | 3.26% |
| **4 Purchases** | **595** | 4.64% | 31.65d | **42.03d** | 49.43d | **14.00d** | 35.03d | **23.03%** | 1.85% | **37.82%** | 5.04% |
| **5 Purchases** | **484** | 3.78% | 29.90d | **37.21d** | 41.61d | **11.00d** | 28.08d | **23.35%** | 2.48% | **39.26%** | 7.44% |
| **6–10 Purchases**| **2,016** | 15.72% | 30.10d | **21.01d** | 30.46d | **9.50d** | 22.18d | **24.36%** | 5.11% | **43.15%** | 14.43% |
| **>10 Purchases** | **6,671** | **52.03%** | 25.73d | **10.32d** | 14.57d | **5.00d** | 9.60d | **38.27%** | 16.79% | **58.72%** | 38.41% |

---

## 4. Deep-Dive Tier-by-Tier Analysis

```
       ==========================================================================
                     ACCURACY VS HISTORY DEPTH TRAJECTORY
       ==========================================================================
         ±7-Day Accuracy (%)
          100% |
               |
           75% |                                                     62.8% [>10] (Base)
               |                                            49.1% [6-10]
           50% |                           39.2%    33.9%            39.4% [>10] (XGB)
               |                  38.5%    [P=4]    [P=5]
           25% |         19.0%    [P=2]   
               |         [P=1]            30.3%
               |                          [P=3]                     11.9% [6-10] (XGB)
            0% |---------0.0%------0.0%----0.7%-----2.4%-----3.0%-----------------
                         P=1       P=2     P=3      P=4     P=5     P=6-10   P>10
       ==========================================================================
```

### Tier 1: Cold Start ($P = 1$, 10.5% of test set)
- **Data State:** Zero prior intervals. Features like `historical_interval_median` and `days_since_previous_purchase` are 100% `NaN`.
- **XGBoost Behavior:** Fails completely (**MAE 159.96d**, 0% $\pm 7$d) because trees assign imputed `NaN`s to high-value training leaves derived from historical multi-year returnees.
- **Baseline Behavior:** Predicts the general population median (31.0 days), achieving an MAE of **16.94 days** against the true test mean of 17.04 days.
- **Operational Verdict:** ❌ **Not suitable for individualized automated ML prediction.**

### Tier 2: Single Prior Interval ($P = 2$, 6.8% of test set)
- **Data State:** Exactly 1 historical interval. If a patient visited once in 2021 and once in 2025, that single interval is ~1,400 days.
- **XGBoost Behavior:** High error (**MAE 93.96d**, MedAE 69.30d).
- **Baseline Behavior:** Heavily influenced by extreme outliers (**MAE 127.12d**, RMSE 344.95d), though its **MedAE drops sharply to 13.00 days** and achieves **38.49% within $\pm 7$ days** for regular patients.
- **Operational Verdict:** ❌ **High variance; unrepresentative outlier risk.**

### Tier 3: Sparse History ($P = 3$, 5.6% of test set)
- **Data State:** Exactly 2 historical intervals.
- **Performance:** Baseline MAE is **101.93 days** (MedAE 15.50d; 30.26% $\pm 7$d); XGBoost MAE is **67.60 days** (MedAE 47.21d).
- **Operational Verdict:** ❌ **Insufficient sample size for individual cadence stability.**

### Tier 4: Early Transition ($P = 4$ and $P = 5$, 8.5% of test set)
- **Data State:** 3 to 4 prior intervals.
- **Performance:** Baseline MAE stabilizes significantly to **40.76 days ($P=4$) and 37.02 days ($P=5$)**, with MedAE reaching **12.00–13.50 days** and $\pm 7$-day accuracy reaching **34%–39%**.
- **Operational Verdict:** ⚠️ **Emerging reliability; suitable for gentle, wide-window reminder notices (-7d).**

### Tier 5: Established History ($P = 6\text{–}10$, 14.0% of test set)
- **Data State:** 5 to 9 prior intervals.
- **Performance:** Baseline MAE drops to **20.15 days**, MedAE drops to **8.00 days**, and $\pm 7$-day accuracy reaches **49.10%** (67.42% within $\pm 14$ days).
- **Operational Verdict:** 🟡 **Moderately reliable; suitable for standard multi-stage reminder schedules.**

### Tier 6: Deep Chronic History ($P > 10$, 54.6% of test set)
- **Data State:** $\ge 10$ prior intervals (representing 4,124 test events).
- **Performance:**
  - **Baseline:** **MAE 8.58 days**, **MedAE 5.00 days**, **40.98% within $\pm 3$ days**, **62.83% within $\pm 7$ days**, **81.81% within $\pm 14$ days**.
  - **XGBoost:** **MAE 14.14 days**, **MedAE 9.48 days**, **39.35% within $\pm 7$ days**, **64.40% within $\pm 14$ days**.
- **Operational Verdict:** ✅ **Production-grade; highly reliable for automated reminder dispatch.**

---

## 5. Cumulative History Depth Thresholds

Aggregating performance above specific cutoff thresholds clarifies where to place production eligibility boundaries:

| Eligibility Cutoff | Covered Test Events ($N$) | Share of Total (%) | Baseline MAE (days) | XGBoost MAE (days) | Baseline MedAE (days) | Baseline $\pm 3$d (%) | Baseline $\pm 7$d (%) | Baseline $\pm 14$d (%) | Operational Suitability |
|---|---|---|---|---|---|---|---|---|---|
| **All Events ($P \ge 1$)** | **7,556** | 100.0% | 27.00d | 43.71d | 7.00d | 31.82% | 50.61% | 70.33% | Whole population baseline |
| **Repeat Buyers ($P \ge 2$)** | **6,766** | 89.5% | 28.17d | 30.13d | 6.00d | 34.42% | 54.30% | 74.11% | Filters pure cold start |
| **History $\ge 3$ Purchases** | **6,249** | 82.7% | 20.61d | 25.26d | 6.00d | 35.29% | 55.51% | 75.82% | Filters single interval volatility |
| **History $\ge 4$ Purchases** | **5,826** | 77.1% | 14.71d | 22.18d | 5.50d | 36.54% | 57.35% | 77.96% | Early cadence stability |
| **History $\ge 6$ Purchases** | **5,183** | **68.6%** | **10.95d** | **18.02d** | **5.00d** | **38.41%** | **60.02%** | **78.87%** | **RECOMMENDED MINIMUM THRESHOLD** |
| **History $> 10$ Purchases** | **4,124** | **54.6%** | **8.58d** | **14.14d** | **5.00d** | **40.98%** | **62.83%** | **81.81%** | **PREMIUM CHRONIC TIER** |

---

## 6. Business & Clinical Operational Recommendations (FOR LATER IMPLEMENTATION)

*(Note: In accordance with audit requirements, no production modifications have been made. These recommendations are documented for subsequent architectural phases).*

### 1. Clinical Tiering Strategy for Automated WhatsApp Reminders:
- **Tier A (Deep Chronic, $P > 10$ — 54.6% volume):**  
  Deploy full 6-stage reminder cycle (`-7d, -3d, -1d, 0d, +3d, +7d`). Baseline delivers **8.58d MAE / 62.8% $\pm 7$d accuracy**.
- **Tier B (Established Chronic, $P = 6\text{–}10$ — 14.0% volume):**  
  Deploy standard 4-stage reminder cycle (`-7d, -3d, 0d, +3d`). Baseline delivers **20.15d MAE / 49.1% $\pm 7$d accuracy**.
- **Tier C (Early Transition, $P = 4\text{–}5$ — 8.5% volume):**  
  Deploy conservative 2-stage reminder notice (`-3d, 0d`) with low-urgency notification copy.
- **Tier D (Sparse / Cold-Start, $P \le 3$ — 22.9% volume):**  
  Do **NOT** attempt individual timeline predictions. Instead, utilize standard **Fixed 30-Day Pack Defaults** or pharmacy-level seasonal wellness promotions.

### 2. Predictor Selection by Depth:
- **For $P \ge 6$:** Utilize the **Personal Historical Median**, which is the optimal point estimator for stationary chronic renewals.
- **For $P < 6$:** Fall back to **Therapeutic Category / SALT-level Medians** (e.g. 30 days for antihypertensives/oral hypoglycemics) rather than unconstrained tree regression models.

---
