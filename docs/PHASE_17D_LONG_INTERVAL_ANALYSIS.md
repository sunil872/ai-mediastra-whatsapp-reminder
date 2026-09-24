# Phase 17D — Long Purchase Interval Analysis

**Date:** 2026-09-17  
**Status:** Analysis only — production data/model **not** modified  
**Source:** `training_dataset.parquet` supervised rows with observed next purchase  
**Scorer:** production `refill_model.joblib` (read-only)  
**Script:** `scripts/phase17d_long_interval_analysis.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_long_interval_results.json`

---

## 1. Scope

| Item | Value |
|---|---|
| Supervised rows analyzed | **489,960** |
| Data span | **2020-12-24 → 2026-08-31** |
| Target | `target_days_until_next_purchase` (observed next purchase only) |
| Production artifacts changed | **No** (`model_mtime_unchanged: true`) |

### Bucket definitions

`0–14`, `15–30`, `31–45`, `46–60`, `61–90`, `91–120`, `121–180`, `>180` days.

### Definitions used

| Concept | Definition |
|---|---|
| **History depth** | `purchase_count_so_far` at the prediction event |
| **Recurring (flag)** | `is_recurring_history == 1` (Phase 3 rule: ≥3 purchases and median interval in 15–120d) |
| **Recurring same bucket** | Same `(customerId, itemId)` has **≥2** observed targets in that interval bucket |
| **Isolated in bucket** | Only one observed target for that pair in that bucket |
| **Prediction error** | Production model: `pred − actual` (signed) and MAE/MedAE |
| **Consistent with personal cadence** | Target within **0.5×–2×** of personal `historical_interval_median` |

---

## 2. Bucket summary

| Bucket | Count | % of supervised | Target median | MAE (prod model) | MedAE | Mean signed error (pred−actual) | Baseline MAE (hist median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–14 | 117,229 | 23.93% | 8 | 63.35 | 31.33 | **+63.29** | 24.86 |
| 15–30 | 126,404 | 25.80% | 22 | 48.11 | 23.15 | +47.55 | 20.10 |
| 31–45 | 75,749 | 15.46% | 35 | 45.51 | 21.16 | +43.32 | 22.97 |
| 46–60 | 30,519 | 6.23% | 53 | 56.30 | 31.56 | +48.87 | 40.12 |
| 61–90 | 35,822 | 7.31% | 72 | 57.35 | 39.66 | +42.17 | 56.24 |
| 91–120 | 20,479 | 4.18% | 104 | 58.26 | 50.12 | +27.69 | 80.69 |
| 121–180 | 23,474 | 4.79% | 145 | 57.85 | 53.68 | −1.69 | 114.02 |
| **>180** | **60,284** | **12.30%** | **362** | **320.43** | **205.15** | **−316.49** | 419.72 |

### Share view

- **Chronic-like 15–45d:** **41.3%** of supervised events  
- **Long intervals >90d:** **21.3%** of supervised events  
- **Ultra-long >180d alone:** **12.3%** (large enough to dominate overall MAE if not filtered)

---

## 3. History depth by bucket

| Bucket | Mean depth | Median depth | % ≤2 purchases | % 3–5 | % >5 |
|---|---:|---:|---:|---:|---:|
| 0–14 | 25.07 | 8 | 26.7% | 16.3% | 57.1% |
| 15–30 | 14.03 | 8 | 22.5% | 18.3% | 59.3% |
| 31–45 | 11.62 | 7 | 24.3% | 19.7% | 55.9% |
| 46–60 | 8.50 | 5 | 32.4% | 22.8% | 44.8% |
| 61–90 | 7.16 | 4 | 37.1% | 24.6% | 38.3% |
| 91–120 | 5.94 | 3 | 42.9% | 24.8% | 32.3% |
| 121–180 | 5.01 | 3 | 48.4% | 24.8% | 26.8% |
| **>180** | **3.33** | **2** | **64.3%** | 20.7% | **15.0%** |

**Pattern:** As intervals lengthen, histories get **shallower**. Long gaps are disproportionately cold-start / low-repeat pairs, not deep chronic refillers.

Compare:

- Long (>90d) mean depth **4.22**
- Short/chronic (15–45d) mean depth **13.13**

---

## 4. Recurring vs isolated

| Bucket | `% is_recurring_history` | `% recurring same bucket` | `% isolated in bucket` | `% target ≈ personal hist median (0.5–2×)` | Median (target / hist_median) |
|---|---:|---:|---:|---:|---:|
| 0–14 | 33.6% | 81.4% | 18.7% | 53.0% | 0.56 |
| 15–30 | 63.5% | 85.4% | 14.6% | 78.8% | 0.93 |
| 31–45 | 68.0% | 78.7% | 21.3% | 77.7% | 1.10 |
| 46–60 | 57.8% | 54.6% | 45.4% | 63.5% | 1.42 |
| 61–90 | 52.1% | 54.3% | 45.7% | 49.7% | 1.67 |
| 91–120 | 43.7% | 37.5% | 62.5% | 41.4% | 1.90 |
| 121–180 | 36.4% | 38.4% | 61.6% | 36.6% | 2.26 |
| **>180** | **22.5%** | 47.6% | **52.4%** | **23.6%** | **4.36** |

**Interpretation:**

- **15–45d** looks like genuine recurring refill cadence (high recurring flag, high same-bucket repeat, target ≈ personal median).
- **91–180d** mixes some genuine low-frequency cycles with many **isolated** gaps.
- **>180d** is rarely consistent with personal historical median (ratio ~4.4×) → mostly **breaks / returns after long absence**, not steady refill behavior.

Overall for intervals **>90d**:

- Recurring same-bucket: **43.5%**
- `is_recurring_history` flag: **29.8%**

---

## 5. Prediction error behavior

### Short / medium intervals (0–90d)
Production `reg:squarederror` model **over-predicts** (positive signed error +42 to +63 days). Predictions have a heavy right tail (pred means 70–115d while actuals are much smaller).

### Long intervals
| Bucket | Pred mean | Actual mean | Signed error | Meaning |
|---|---:|---:|---:|---|
| 91–120 | 132 | 104 | +28 | mild over-predict |
| 121–180 | 145 | 147 | ~0 | roughly centered |
| **>180** | **170** | **486** | **−316** | severe **under-predict** |

So ultra-long gaps are **not** being treated as refill cycles by the model (predictions stay ~half-year-ish while many actuals are 1–several years). That is expected if the true process is churn/return rather than chronic cadence.

Baseline personal median also fails badly on >180 (MAE **420**), confirming these are not well described by prior refill medians.

---

## 6. Window position / censoring effects

| Bucket | Median days from invoice → dataset end | % invoices in last 180d of dataset |
|---|---:|---:|
| 0–14 | 1066 | 8.9% |
| 15–30 | 927 | 9.4% |
| … | … | … |
| 121–180 | 1074 | 1.9% |
| **>180** | **1301** | **0.0%** |

### What this means

1. **True right-censoring** (no next purchase yet) removes rows from the supervised set entirely (last purchase → target NaN). Those rows are **not** in this analysis.
2. For rows that **do** have a target, both purchases were observed — so the gap is a real elapsed time, not an imputed censored value.
3. **Selection / right-truncation bias** still exists: an interval of length \(L\) can only be observed if the first purchase occurs at least \(L\) days before dataset end. That is why **>180d events never appear in the last 180 days of the window** (`0.0%`). Ultra-long intervals are over-sampled from earlier calendar years and under-sampled near the end.

So long intervals are **not primarily censoring artifacts in the label**, but the **observable long-interval sample is window-biased**.

---

## 7. Heuristic classification of long intervals (>90d)

Applied only to >90d events (n ≈ 104,237):

| Class | Share | Meaning |
|---|---:|---|
| likely_irregular_or_cold_start_gap | **39.5%** | Shallow history + isolated long gap |
| likely_irregular_break_from_personal_cadence | **25.3%** | Had a shorter personal median, then a much longer gap |
| mixed_or_unclear | 24.8% | Does not cleanly match other rules |
| likely_genuine_low_frequency_refill | **10.0%** | Repeated same long bucket + depth≥3 + consistent with hist median |
| possible_edge_of_window_selection_effect | 0.3% | Rare residual edge cases |

---

## 8. Conclusions

### Do long intervals represent genuine refill behavior?
**Sometimes, but mostly no.**

- Only about **~10%** of >90d events look like **genuine low-frequency refill** (repeated long cadence aligned with personal history).
- A substantial minority (**~37–48%** depending on bucket) of 91–180d gaps **do** recur in the same bucket for the same customer–item and may include quarterly / 3–4 month pack behavior — but consistency with personal median drops as length grows.

### Irregular purchasing?
**Yes — dominant for ultra-long gaps.**

- >180d: median depth **2**, **64%** have ≤2 purchases, target is typically **~4×** personal historical median.
- Heuristics label **~65%** of >90d events as irregular/cold-start or cadence-break.

### Data artifacts?
**Partially.**

- Duplicate/same-day mechanics are already filtered by `target > 0` in modeling; 0–14d still contains many true short revisits (deep histories, high same-bucket recurrence).
- No evidence that >180d labels are fabricated; they are measured gaps. Artifact risk is more about **including churn/return gaps in a refill-prediction training objective**.

### Censoring effects?
**Indirect, via observability — not via fake long labels.**

- Censoring removes unfinished spells (no target).
- Right-truncation shapes **which** long intervals can be seen (hence 0% of >180d events in the last 180 dataset days).

---

## 9. Implications for RefillCare

1. **Do not treat >180d (and arguably >120d) as core refill-reminder training signal.** They inflate error and pull squared-error models into bad compromise fits.
2. **Operational reminder eligibility should prefer:**
   - depth ≥3 (ideally >5)
   - recurring / regular cadence (15–90d personal median)
   - exclude isolated ultra-long gaps
3. This aligns with Phase 17D objective findings: MAE/quantile losses help, but **cohort filtering** is equally important because ~12% of supervised rows are >180d absences, not monthly refill cycles.

---

## 10. Artifact safety

| Check | Result |
|---|---|
| Production model modified | **No** |
| Production data modified | **No** |
| `refill_model.joblib` mtime unchanged | **Yes** |

Outputs written by this analysis only:

- `docs/PHASE_17D_LONG_INTERVAL_ANALYSIS.md`
- `data/refillcare/processed/phase17d_long_interval_results.json`
- `scripts/phase17d_long_interval_analysis.py`
