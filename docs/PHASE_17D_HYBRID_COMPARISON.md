# Phase 17D — Hybrid Prediction Comparison (Read-Only Offline)

**Date:** 2026-09-17  
**Status:** Experimental only — production code, model artifacts, reminder logic, and WhatsApp **not** modified  
**Script:** `scripts/phase17d_hybrid_comparison.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_hybrid_comparison_results.json`

---

## 1. Purpose

Compare five prediction approaches on the **same untouched temporal test set**, using only thresholds and winners already supported by Phase 17D experiments:

| # | Strategy | Source |
|---|---|---|
| 1 | Current production XGBoost | `refill_model.joblib` (`reg:squarederror`) |
| 2 | Personal historical median | Phase 4 / cadence baseline |
| 3 | Best personal-cadence strategy | Tiered heuristic from `PHASE_17D_PERSONAL_CADENCE_EXPERIMENT` |
| 4 | Best experimental XGBoost | `reg:quantileerror` + `quantile_alpha=0.5` (`PHASE_17D_XGB_OBJECTIVE_EXPERIMENT`) — **in-memory only** |
| 5 | Rule-based hybrid routing | Regularity Candidate 5/6 + history-depth + cadence band gates |

---

## 2. Evaluation controls

| Control | Value |
|---|---|
| Test partition | `test.parquet`, `target > 0` (**7,488** rows) |
| Dates | **2026-07-01 → 2026-08-30** |
| Train (for baseline fallback + experimental XGB only) | `train.parquet`, `target > 0` (**465,610**) |
| Production model | Loaded **read-only**; mtime unchanged |
| Experimental XGB | Trained in-memory; **not** written to disk |

---

## 3. Hybrid routing rules (Phase 17D thresholds only)

```
ELIGIBLE (Candidate-5 outer gate + depth + cadence band):
  purchase_count_so_far >= 6
  AND NormMAD <= 0.50
  AND |Recent3 − HistMedian| <= 10 days
  AND historical_interval_median ∈ [15, 120]

THEN:
  IF NormMAD <= 0.35 AND Drift <= 7:
      → personal historical median          # Class 1 / Candidate 6 core
  ELSE:
      → experimental XGB (quantile q50)     # Candidate-5 secondary

ELSE:
  → REJECT (ineligible for automated prediction)
```

| Threshold | Phase 17D support |
|---|---|
| `P >= 6` | History-depth recommended minimum |
| `NormMAD ≤ 0.35`, `Drift ≤ 7` | Regularity Class 1 / Candidate 6 |
| `NormMAD ≤ 0.50`, `Drift ≤ 10` | Regularity Candidate 5 (balanced coverage) |
| Hist median `[15, 120]` | `is_recurring_history` / long-interval implications |
| Tiered cadence (strategy 3) | Personal-cadence experiment recommendation |
| Quantile q50 XGB (strategy 4) | Best objective in XGB objective experiment |

NormMAD / recent-3 drift are computed **leakage-safe** from expanding prior intervals in `purchase_history.parquet` (read-only).

### Best personal-cadence tiers (strategy 3)

| Depth | Predictor |
|---|---|
| `P > 10` | Full historical median |
| `P = 6–10` | Recent-5 median |
| `P = 4–5` | Recent-3 median |
| `P ≤ 3` | Fixed **30-day** pack default |

---

## 4. Overall results (scored predictions)

Always-on strategies score **100%** of the test universe. Hybrid scores only eligible rows and **rejects** the rest.

| Strategy | Coverage | Rejected | MAE ↓ | MedAE ↓ | RMSE ↓ | ±1d ↑ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **hybrid_routing** *(eligible only)* | **33.83%** | **66.17%** | **7.77** | **5.00** | **11.60** | **18.79%** | **38.73%** | **62.57%** |
| best_experimental_xgboost | 100% | 0% | 13.05 | 7.39 | 20.36 | 10.26% | 26.56% | 48.29% |
| best_personal_cadence | 100% | 0% | 14.08 | 7.00 | 34.10 | 16.89% | 31.90% | 50.40% |
| personal_historical_median | 100% | 0% | 26.72 | 7.00 | 109.43 | 16.95% | 32.02% | 50.96% |
| current_xgboost | 100% | 0% | 43.19 | 19.61 | 69.62 | 3.26% | 9.92% | 23.53% |

### Ranking notes

- **Best accuracy on emitted predictions:** hybrid (by refusing hard cases).
- **Best always-on accuracy:** experimental quantile XGB (MAE **13.05**).
- **Best always-on ±7d / MedAE among always-on:** personal median / tiered cadence (~**50–51%** ±7d, MedAE **7**); cadence wins on MAE by replacing cold-start spikes with a 30d pack default.
- **Current production XGB is last** on every overall accuracy metric.

---

## 5. Coverage & rejection (hybrid)

| Metric | Value |
|---|---:|
| Eval universe | 7,488 |
| Predictions emitted | **2,533** |
| Coverage | **33.83%** |
| Rejected / ineligible | **4,955 (66.17%)** |

### Hybrid route mix (among eligible)

| Route | Count | Share of eligible |
|---|---:|---:|
| Core → personal median | 1,819 | 71.8% |
| Secondary → experimental XGB | 714 | 28.2% |

### Reject reason stack (counts; rows may fail multiple gates — reported as first-pass buckets)

| Reason bucket | Count |
|---|---:|
| Depth `< 6` | 2,333 |
| Hist median outside `[15, 120]` (among depth≥6) | 1,562 |
| Broad regularity fail (among depth≥6 & cadence OK) | 1,060 |

Coverage (~34%) is slightly below Candidate-6’s ~37% in the regularity report because this experiment also requires hist median ∈ `[15, 120]` and uses `target > 0` only.

---

## 6. Apples-to-apples: same hybrid-eligible subset (n=2,533)

| Strategy | MAE | MedAE | RMSE | ±1d | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|
| **best_experimental_xgboost** | **7.39** | **4.85** | **10.68** | 14.61% | 35.81% | **62.53%** |
| hybrid_routing | 7.77 | 5.00 | 11.60 | **18.79%** | **38.73%** | **62.57%** |
| personal_historical_median | 8.04 | 5.00 | 12.11 | 18.83% | 38.02% | 61.90% |
| best_personal_cadence | 8.04 | 5.00 | 12.12 | 19.31% | 38.45% | 61.82% |
| current_xgboost | 14.33 | 10.39 | 20.47 | 4.90% | 14.41% | 35.49% |

On the eligible cohort alone, **experimental XGB edges hybrid on MAE**, while hybrid matches ±7d and improves ±1d/±3d by routing core regulars to the personal median. Current production XGB remains far behind even on this easy cohort.

---

## 7. History-depth cohorts

### Always-on strategies (Test, target>0)

#### Depth ≤2 (n=1,282)

| Strategy | MAE | MedAE | ±1d | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| best_personal_cadence | **15.38** | **16.00** | 5.9% | 12.4% | **22.8%** |
| best_experimental_xgboost | 27.05 | 23.93 | 2.0% | 7.5% | 15.7% |
| personal_historical_median | 60.46 | 17.00 | 8.6% | 15.8% | 27.5% |
| current_xgboost | 133.15 | 138.98 | 0.0% | 0.0% | 0.0% |

#### Depth 3–5 (n=1,051)

| Strategy | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|
| best_experimental_xgboost | **18.16** | **11.38** | **16.3%** | **35.1%** |
| best_personal_cadence | 28.02 | 12.00 | 22.6% | 36.4% |
| personal_historical_median | 63.15 | 14.00 | 20.1% | 34.4% |
| current_xgboost | 57.44 | 42.12 | 0.6% | 1.9% |

#### Depth 6–10 (n=1,051)

| Strategy | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|
| best_experimental_xgboost | **12.04** | **7.44** | **25.9%** | **47.3%** |
| personal_historical_median | 20.06 | 7.50 | 28.6% | 49.5% |
| best_personal_cadence | 20.12 | 8.00 | 29.5% | 49.2% |
| current_xgboost | 32.80 | 23.01 | 3.5% | 12.0% |
| hybrid *(scored n=473)* | **10.27** | **7.00** | **29.6%** | **51.6%** |

#### Depth >10 (n=4,104)

| Strategy | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|
| best_experimental_xgboost | **7.62** | **4.90** | 35.3% | 62.1% |
| personal_historical_median / cadence | 8.56 | 5.00 | **41.0%** | **62.9%** |
| current_xgboost | 14.10 | 9.46 | 17.1% | 39.4% |
| hybrid *(scored n=2,060)* | **7.20** | **4.59** | **40.8%** | **65.1%** |

---

## 8. Regularity cohorts

Aligned with `PHASE_17D_REGULARITY_ANALYSIS` class definitions:

| Class | Definition | n |
|---|---|---:|
| Class 1 core regular | `P≥6`, NormMAD≤0.35, Drift≤7 | 2,789 |
| Class 2 high-volume variable | `P≥6`, not Class 1 | 2,366 |
| Class 3 emerging | `P=4–5` | 634 |
| Class 4 sparse | `P≤3` | 1,699 |

### Class 1 (high-confidence regular)

| Strategy | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|
| personal_historical_median | **6.14** | **3.00** | **51.0%** | **73.7%** |
| best_personal_cadence | 6.14 | 3.00 | 51.3% | 73.5% |
| best_experimental_xgboost | 6.00 | 3.69 | 43.3% | 71.5% |
| current_xgboost | 11.68 | 7.98 | 18.7% | 44.8% |
| hybrid *(scored n=1,819)* | 7.39 | 4.00 | 43.8% | 66.7% |

### Class 2 (variable chronic)

| Strategy | MAE | MedAE | ±7d |
|---|---:|---:|---:|
| best_experimental_xgboost | **11.49** | **8.12** | **44.4%** |
| personal_historical_median | 16.51 | 9.00 | 44.2% |
| best_personal_cadence | 16.54 | 9.00 | 44.3% |
| current_xgboost | 25.26 | 17.60 | 20.9% |
| hybrid *(scored n=714)* | **8.74** | **6.62** | **52.0%** |

### Class 3 / 4

Hybrid **rejects all** Class 3–4 rows (coverage 0% by design). Always-on winners:

- Class 3: experimental XGB MAE **16.9**; cadence **37.5**; current XGB **51.3**
- Class 4: cadence (30d pack) MAE **15.0**; experimental XGB **25.3**; current XGB **116.9**

### `is_recurring_history == 1` (n=4,283)

| Strategy | MAE | MedAE | ±7d |
|---|---:|---:|---:|
| best_experimental_xgboost | **10.19** | **6.91** | **50.4%** |
| best_personal_cadence | 12.35 | 7.50 | 49.5% |
| personal_historical_median | 12.66 | 8.00 | 48.7% |
| current_xgboost | 24.49 | 16.61 | 23.9% |
| hybrid *(all 2,533 eligible are recurring)* | **7.77** | **5.00** | **62.6%** |

---

## 9. Interpretation

1. **Current production XGBoost is not competitive** as an always-on refill predictor on this holdout (MAE **43.2**, ±7d **23.5%**).
2. **Personal median** remains strong on deep / Class-1 regulars (Class-1 MAE **6.14**, ±7d **73.7%**) but is wrecked by cold-start / long-gap tails (overall MAE **26.7**, RMSE **109**).
3. **Tiered personal cadence** is the best simple always-on heuristic (overall MAE **14.1**) by forcing a 30d pack default on `P≤3` instead of trusting sparse history.
4. **Best experimental XGB (quantile q50)** is the best always-on learner (MAE **13.05**, RMSE **20.4**) and slightly best on the hybrid-eligible subset (MAE **7.39**).
5. **Hybrid routing** trades coverage for reliability:
   - Emits predictions for **~34%** of test events
   - Rejects **~66%** (depth / irregular / non-refill cadence)
   - On emitted rows: MAE **7.77**, MedAE **5**, ±7d **62.6%**
   - This matches the Phase 17D operational thesis: *do not auto-remind irregular / shallow / ultra-long-gap patients*

---

## 10. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** |
| `phase4_model_report.json` mtime unchanged | **Yes** |
| Production code modified | **No** |
| Reminder logic modified | **No** |
| WhatsApp / dispatch modified | **No** |
| Experimental model saved | **No** (in-memory only) |

Outputs written by this experiment only:

- `docs/PHASE_17D_HYBRID_COMPARISON.md` (this file)
- `data/refillcare/processed/phase17d_hybrid_comparison_results.json`
- `scripts/phase17d_hybrid_comparison.py`

---

## 11. Recommendation (non-binding)

For a future explicit production redesign (not done here):

1. Prefer **hybrid eligibility** (Candidate-5 + `P≥6` + hist median 15–120d) over always-on production XGB.
2. Within eligible traffic, either:
   - **Hybrid as defined** (core → personal median; secondary → quantile XGB), or
   - **Always use experimental quantile XGB** on the eligible set (slightly better MAE on this holdout).
3. Keep rejected patients on **pack defaults / manual review**, not unconstrained ML.
4. Do **not** replace `refill_model.joblib` until an authorized retrain explicitly adopts quantile (and optionally hybrid gates).

**No production change was made in this phase.**
