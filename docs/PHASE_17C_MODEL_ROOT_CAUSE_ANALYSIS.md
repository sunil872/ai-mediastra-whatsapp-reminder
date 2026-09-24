# Phase 17C: RefillCare Model Performance Root-Cause Analysis

**Audit & Analysis Date:** 2026-09-17  
**Scope:** Root-cause investigation into why the serialized XGBoost regression model (`refill_model.joblib`) performs worse than the simple Historical Median Baseline on the new 5.8-year pharmacy dataset.  
**Mode:** Read-Only Analytical Investigation (No code changes, no dataset alterations, no model retraining).

---

## Executive Summary

When evaluated on the prospective holdout partitions of the new ~5.8-year dataset (`data/refillcare/customer_data_fields.csv`), the serialized **XGBoost Regressor** exhibits an aggregate **Test MAE of 43.71 days** (23.41% within $\pm 7$ days), whereas the simple **Historical Median Baseline** achieves **27.00 days MAE** (50.61% within $\pm 7$ days).

```
========================================================================================
OVERALL BENCHMARK COMPARISON (Test Holdout: 2026-07-01 to 2026-08-31, N=7,556)
========================================================================================
Metric                  Historical Baseline       XGBoost Regressor        Difference
----------------------------------------------------------------------------------------
MAE (Mean Absolute Error)     27.00 days               43.71 days             +16.71 days (XGB Worse)
Median Absolute Error (MedAE)  7.00 days               19.84 days             +12.84 days (XGB Worse)
Accuracy within ±3 Days       31.82%                    9.85%                 -21.97%     (XGB Worse)
Accuracy within ±7 Days       50.61%                   23.41%                 -27.20%     (XGB Worse)
RMSE (Root Mean Sq Error)    109.54 days               70.31 days             -39.23 days (XGB Better!)
========================================================================================
```

### The Central Paradox & Root-Cause Summary
1. **Loss Function Objective Mismatch (`reg:squarederror` vs MAE/$\pm 7$d accuracy):**  
   XGBoost optimizes **Mean Squared Error (MSE)**, mathematically predicting the **conditional mean** $E[Y|X]$. Because 5.8 years of historical data includes extreme long-tail repurchase intervals (up to 2,060 days), the training set target distribution is heavily right-skewed (**Mean: 96.04 days vs Median: 31.00 days**). Consequently, XGBoost systematically over-predicts by +40 to +45 days (predicting an average of 61.47 days on the test set, where the true test target mean is only 18.80 days).
2. **Severe Temporal Right-Censoring Asymmetry in Validation/Test Sets:**  
   The entire dataset terminates on `2026-08-31`. In the 2-month test partition (`2026-07-01` to `2026-08-31`), prospective next purchases after Aug 31 are unobserved (censored as final visits). Thus, the test set *only* contains rapid, acute, or high-frequency repurchasers (true target maximum: 60.0 days; true mean: 18.80 days).
3. **Cold-Start / Sparse History Failure (1st & 2nd purchases):**  
   For patients with only 1 prior purchase (10.5% of test data), XGBoost predicts an average of **159.96 days** (resulting in an MAE of **159.96 days** and 0.0% $\pm 7$d accuracy), whereas the Baseline fallback predicts **31.00 days** (MAE of **16.94 days**).
4. **Deep Chronic Histories (>10 Purchases):**  
   For established chronic patients (54.6% of test data), XGBoost error drops significantly to **14.14 days MAE** (39.35% $\pm 7$d), but the simple personal median remains superior at **8.58 days MAE** (62.83% $\pm 7$d).

---

## 1. Verified Datasets, Row Counts & Date Ranges

The modeling pipeline operates on the following chronological partitions:

| Partition Name | File Path | Row Count | % of Supervised | Invoice Date Min | Invoice Date Max | Observed Target Max |
|---|---|---|---|---|---|---|
| **Raw Dataset** | `data/refillcare/customer_data_fields.csv` | 895,557 | — | 2020-12-24 | 2026-08-31 | 2,060 days |
| **Clean Transactions** | `data/refillcare/processed/clean_transactions.parquet` | 878,676 | — | 2020-12-24 | 2026-08-31 | 2,060 days |
| **Purchase History** | `data/refillcare/processed/purchase_history.parquet` | 817,804 | — | 2020-12-24 | 2026-08-31 | 2,060 days |
| **Full Supervised Set** | `data/refillcare/processed/training_dataset.parquet` | 489,960 | 100.0% | 2020-12-24 | 2026-08-31 | 2,060 days |
| **Train Partition** | `data/refillcare/processed/train.parquet` | **469,583** | **95.84%** | **2020-12-24** | **2026-04-30** | **2,060 days** |
| **Validation Partition** | `data/refillcare/processed/validation.parquet` | **12,821** | **2.62%** | **2026-05-01** | **2026-06-30** | **121 days** |
| **Test Holdout Partition** | `data/refillcare/processed/test.parquet` | **7,556** | **1.54%** | **2026-07-01** | **2026-08-31** | **60 days** |

---

## 2. Target Distribution & Percentile Breakdown

The target variable is `target_days_until_next_purchase` ($t_{i+1} - t_i$). Because the training dataset covers 5.35 years while validation and test partitions are narrow prospective windows at the end of the timeline, the target distributions diverge fundamentally:

```
========================================================================================
TARGET DISTRIBUTION COMPARISON (DAYS UNTIL NEXT PURCHASE)
========================================================================================
Statistic / Quantile      Train Partition (N=469,583)   Val Partition (N=12,821)   Test Partition (N=7,556)
----------------------------------------------------------------------------------------
Mean                            96.04 days                    28.08 days                 18.80 days
Standard Deviation (Std)       193.46 days                    20.07 days                 11.88 days
Median (P50)                    31.00 days                    25.00 days                 17.00 days
P01                              1.00 days                     1.00 days                  1.00 days
P05                              4.00 days                     5.00 days                  2.00 days
P10                              7.00 days                     8.00 days                  5.00 days
P25                             15.00 days                    13.00 days                 10.00 days
P75                             76.00 days                    35.00 days                 28.00 days
P90                            239.00 days                    57.00 days                 34.00 days
P95                            447.00 days                    69.00 days                 40.00 days
P99                          1,056.18 days                    96.00 days                 51.00 days
Maximum (P100)               2,060.00 days                   121.00 days                 60.00 days
Zero-Day Targets (0 days)        3,973 (0.85%)                    54 (0.42%)                 68 (0.90%)
========================================================================================
```

### Key Distribution Insights:
- **Massive Skewness in Training Set:** Over 5.35 years, 25% of all historical training intervals exceed 76 days, 10% exceed 239 days, and 1% exceed 1,056 days.
- **Window Truncation in Val/Test:** In the 2-month test partition (`2026-07-01` to `2026-08-31`), no interval can mathematically exceed 61 days because the dataset ends on August 31. This creates an artificial distribution shift where the true test target mean is only **18.80 days**.

---

## 3. Detailed Performance Breakdown by Cohort & History Depth

Evaluating both models on the test set holdout ($N=7,556$) reveals how history depth governs accuracy:

| History Depth Bucket | Sample Count | % of Test Set | True Target Mean | True Target Median | XGBoost MAE | Baseline MAE | XGBoost $\pm 7$d (%) | Baseline $\pm 7$d (%) | Operational Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| **1 Purchase (Cold Start)** | 790 | 10.5% | 17.04d | 14.00d | **159.96d** | **16.94d** | 0.00% | 18.99% | XGBoost suffers catastrophic failure due to missing interval features and tree mean bias; Baseline default (31d) stays close to true mean. |
| **2 Purchases (1 Interval)** | 517 | 6.8% | 18.49d | 16.00d | **93.96d** | **127.12d** | 0.00% | 38.49% | Single prior intervals in baseline often contain multi-year gaps; XGBoost still over-predicts. |
| **3–5 Purchases** | 1,066 | 14.1% | 18.86d | 17.00d | **58.07d** | **63.97d** | 1.88% | 34.15% | Moderate history; both models impacted by early transition noise, but Baseline achieves 34% $\pm 7$d. |
| **6–10 Purchases** | 1,059 | 14.0% | 20.21d | 18.00d | **33.13d** | **20.15d** | 11.90% | 49.10% | Transition to recurring stability; Baseline beats XGBoost by 13 days MAE. |
| **>10 Purchases (Deep Chronic)** | **4,124** | **54.6%** | **18.80d** | **17.00d** | **14.14d** | **8.58d** | **39.35%** | **62.83%** | Established chronic patients; Baseline achieves **8.58 days MAE** and **62.83% accuracy within $\pm 7$ days**. |

---

## 4. Performance Breakdown by Actual Target Refill Interval

| Target Interval Category | Sample Count | % of Test Set | True Target Mean | XGBoost Pred Mean | XGBoost MAE | Baseline MAE | XGBoost $\pm 7$d (%) | Baseline $\pm 7$d (%) |
|---|---|---|---|---|---|---|---|---|
| **Same-Day (0 days)** | 68 | 0.9% | 0.00d | 100.78d | 100.78d | 56.62d | 10.29% | 17.65% |
| **Short (1–14 days)** | 3,221 | 42.6% | 8.33d | 60.31d | 52.04d | 31.65d | 17.45% | 48.74% |
| **Regular Refill (15–45 days)** | **4,075** | **53.9%** | **25.87d** | **60.62d** | **36.14d** | **21.56d** | **28.83%** | **54.55%** |
| **Medium Refill (46–90 days)** | 192 | 2.5% | 51.06d | 84.98d | 44.27d | 54.11d | 13.02% | 9.90% |

### Key Interval Findings:
- In the core **15–45 day regular chronic window** (representing 53.9% of all test events), true target mean is **25.87 days**.
- XGBoost predicts an average of **60.62 days** (over-predicting by +34.75 days).
- The Baseline predicts closer to the median interval, achieving **21.56 days MAE** and **54.55% accuracy within $\pm 7$ days** on this segment.

---

## 5. Major Root Causes of XGBoost Underperformance

### Root Cause 1: Loss Function Disparity (`reg:squarederror` vs MAE / $\pm 7$d Window)
- **Mathematical Reality:** An XGBoost model configured with `objective="reg:squarederror"` optimizes $E[(Y - \hat{Y})^2]$. The theoretical Bayes optimal predictor for squared error loss is the **conditional mean** $E[Y|X]$.
- In contrast, the optimal point estimator for **Mean Absolute Error (MAE)** is the **conditional median** $\text{Med}(Y|X)$.
- Because the 5.8-year training set has a high positive skew (Mean 96.04d vs Median 31.00d), minimizing squared error pulls tree leaf outputs aggressively toward the long tail.
- **Evidence:** On the Test set, XGBoost actually achieves a **lower RMSE than the Baseline** (70.31 days vs 109.54 days), confirming that XGBoost successfully minimized squared loss! However, under MAE and $\pm 7$-day accuracy evaluation, predicting the mean inflates the linear error on the 75% of patients who refill in $\le 30$ days.

### Root Cause 2: Severe Temporal Right-Censoring / Truncation Boundary
- Training examples cover up to 5.35 years (`2020-12-24` to `2026-04-30`), where long-gap re-purchases (e.g., episodic antibiotics, topical ointments, or multi-year churned returnees) are fully observed up to 2,060 days.
- In contrast, the prospective Validation (`2026-05` to `2026-06`) and Test (`2026-07` to `2026-08`) sets are narrow 2-month slices at the end of the historical data timeline.
- Any patient with a true 90-day, 180-day, or 365-day refill cycle who purchased in July 2026 did not return before August 31, 2026. Under the supervised learning formulation, their final purchase has no observed next date and was strictly excluded from the test set.
- Consequently, the test holdout consists exclusively of **fast / high-frequency repurchasers (true target mean: 18.80 days)**. An ML model trained on 5.35 years of unconstrained long tails inevitably over-predicts when evaluated only on short-cycle survivors.

### Root Cause 3: Cold-Start Imputation Artifacts on 1st & 2nd Purchases
- For patients on their 1st purchase ($N=790$ in test), historical interval features (`historical_interval_median`, `historical_interval_mean`, `historical_interval_std`, `days_since_previous_purchase`) are 100% `NaN`.
- In the training set, 1st-time buyers who eventually made a 2nd purchase took an average of $>160$ days to return.
- As a result, the trained XGBoost tree branches assign 1st-time test patients to leaves predicting **~160 days**, producing a **159.96-day MAE**.
- The Historical Baseline, by contrast, falls back to the dataset median of **31.00 days**, resulting in an MAE of **16.94 days**.

### Root Cause 4: Stationary Renewal Property of Chronic Prescription Refills
- For established chronic patients (>10 purchases, representing 54.6% of test volume), refill dynamics behave as a stationary renewal process governed by prescription pack size (e.g., 30 tablets every 30 days).
- A patient's expanding empirical median is an almost unbiased, minimum-variance estimator for stationary renewal intervals.
- Adding complex tree splits based on calendar month, day of week, quantity ratios, or packaging categories adds variance without providing predictive signal beyond the patient's individual median cadence.

---

## 6. Investigation of the Legacy 5.10-Day MAE / 82.2% $\pm 7$d Benchmark

In Phase 15 and Phase 16 documentation (`docs/PHASE_15_REFILLCARE_CONTROLLED_PILOT_EXECUTION.md`, `docs/PHASE_16_REFILLCARE_CONTROLLED_MICRO_PILOT.md`), the following metrics were documented:
> `XGBoost Prediction Model: FROZEN (MAE: 5.10 days, Within ±3d: 57.0%, ±7d: 82.2%)`

### Forensic Investigation Findings:
1. **Not a Full-Dataset Test Holdout Metric:**  
   In the original 1-year Phase 4 model training report (`docs/PHASE_4_REFILLCARE_MODEL_TRAINING.md`), the aggregate test set metrics were:
   - **Aggregate Test MAE:** **16.20 days** (within $\pm 7$d: **32.27%**)
   - **Regular Cycles (16–45d) MAE:** **11.12 days** (within $\pm 7$d: **42.54%**)
   - **Deep History (>5 Purchases) MAE:** **9.64 days** (within $\pm 7$d: **47.27%**)
2. **Selective Sub-Cohort / Operational Aspiration:**  
   The `5.10-day MAE / 82.2% ±7d` metric was established during Phase 13/15 as an **ideal operational target / high-confidence Tier A chronic subset benchmark** (patients with $\ge 5$ prior purchases, strict 28–32 day refill histories, and filtering out acute/one-off transactions).
3. **Difference in Dataset Time Span:**  
   In the legacy 1-year dataset extract, historical time was limited to 12 months, preventing massive multi-year intervals (500–2,000 days) from inflating the training target mean (1-year train mean was ~30–45 days vs 96.04 days in the 5.8-year dataset).
4. **Conclusion:**  
   The `5.10-day MAE / 82.2% ±7d` figure was **never an end-to-end regression metric on all transactions**. It reflects a highly selective chronic sub-cohort assumption from early pilot planning.

---

## 7. Stale / Old Artifacts Found

| File / Location | Nature of Staleness | Details |
|---|---|---|
| `docs/PHASE_15_REFILLCARE_CONTROLLED_PILOT_EXECUTION.md` | Benchmark Metric Reference | Retains the `5.10-day MAE / 82.2% ±7d` aspirational chronic benchmark in table headers. |
| `docs/PHASE_16_REFILLCARE_CONTROLLED_MICRO_PILOT.md` | Benchmark Metric Reference | Retains the `5.10-day MAE / 82.2% ±7d` aspirational chronic benchmark in preflight checklist. |
| `notebooks/RefillCare_Phases_1_to_3_Walkthrough.ipynb` | Cached Output Display | Cell 8 displays 157,750 intervals; Cell 11 displays legacy Phase 3 baseline validation metrics (Count: 22,621 / MAE 19.26d). |
| `RefillCare_Phases_1_to_3_Walkthrough.ipynb` (Root) | Redundant File | Unexecuted duplicate (0 cell outputs). |

---

## 8. Summary of Findings & Next-Step Architectural Options (DO NOT IMPLEMENT NOW)

### Summary Table

| Evaluation Aspect | Current XGBoost (`refill_model.joblib`) | Historical Median Baseline | Root Cause of Difference |
|---|---|---|---|
| **Overall Test MAE** | 43.71 days | **27.00 days** | Squared error objective pulls predictions toward 5.8-year training mean (96d). |
| **Overall Test $\pm 7$d Accuracy** | 23.41% | **50.61%** | Baseline predicts median, aligning with the peak of the chronic distribution. |
| **Cold-Start (1 Purchase) MAE** | 159.96 days | **16.94 days** | Tree leaf assigned to long-return training examples vs Baseline fallback to 31d. |
| **Deep Chronic (>10 Purchases) MAE** | 14.14 days | **8.58 days** | Individual median is optimal point estimator for stationary renewal processes. |
| **Overall Test RMSE** | **70.31 days** | 109.54 days | XGBoost successfully minimizes squared error, but squared error is misaligned with MAE. |

---

### Architectural Options for Future Refinement (Recommendations Only — No Changes Implemented):

1. **Objective Function Alignment (MAE / Quantile Regression):**
   - Retrain XGBoost / LightGBM with `objective="reg:absoluteerror"` (L1 loss) or Quantile Regression (`reg:quantileerror`, $\alpha=0.50$ for median, and $\alpha=0.25, 0.75$ for prediction uncertainty bounds).
2. **Hybrid / Fallback Clinical Routing Architecture:**
   - **For Established Chronic Patients (>5 purchases):** Route prediction through the **Historical Median Baseline** (or an ensemble heavily weighted on personal median cadence), which delivers **8.58–10.95 days MAE** and **60–63% $\pm 7$d accuracy**.
   - **For Cold-Start / Early Patients (1–2 purchases):** Route to Category/SALT-level median defaults (e.g. 30 days for chronic cardiac/diabetic drugs) rather than unconstrained tree regression.
3. **Training Target Capping / Outlier Truncation:**
   - Cap training interval targets at a clinical chronic threshold (e.g., 90 or 120 days) to prevent 500–2,000 day episodic re-purchases from distorting chronic reminder intervals.
4. **Temporal Split Alignment with Horizon Matching:**
   - Define training targets with an explicit prospective reminder window (e.g., *“Will patient refill within 45 days?”*) to eliminate right-censoring distortion at the boundary of historical data.

---
