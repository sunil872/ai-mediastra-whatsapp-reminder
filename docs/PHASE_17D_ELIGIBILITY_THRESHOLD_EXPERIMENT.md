# Phase 17D — Eligibility Threshold Experiment (Read-Only)

**Date:** 2026-09-17  
**Status:** Offline experiment only — production code, model, data, reminder logic, WhatsApp **unchanged**  
**Script:** `scripts/phase17d_eligibility_threshold_experiment.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_eligibility_threshold_results.json`  
**Base strategy:** `hybrid_routing_v17d` (only the history-depth gate is varied)

---

## 1. Purpose

Measure the **coverage vs accuracy tradeoff** of the hybrid depth gate.

Sweep:

`P >= 3`, `P >= 4`, `P >= 5`, `P >= 6`, `P >= 7`, `P >= 10`

**Held fixed** (same as Phase 17D hybrid):

| Rule | Value |
|---|---|
| Cadence band | hist median ∈ `[15, 120]` |
| Broad regularity (eligible) | NormMAD ≤ 0.50 and Drift ≤ 10 |
| Core / HIGH | NormMAD ≤ 0.35 and Drift ≤ 7 → personal median |
| Secondary / MEDIUM | else eligible → production XGBoost (read-only, **not** retrained) |
| Reject | fails any gate |

---

## 2. Controls

| Control | Value |
|---|---|
| Test set | Same as Hybrid Evaluation: `test.parquet`, `target > 0` |
| Rows | **7,488** |
| Dates | **2026-07-01 → 2026-08-30** |
| Production model | `refill_model.joblib` read-only; mtime unchanged |
| WhatsApp | **0** |
| Current production depth | **P ≥ 6** |

Metrics (MAE / MedAE / ±3d / ±7d) are on **accepted predictions only**.

**Regular vs irregular** below use core regularity rules among rows that meet the **swept depth gate** (`regular_at_depth_gate` / `irregular_at_depth_gate`).

---

## 3. Overall results by depth threshold

| Depth gate | Accepted | Coverage | Rejected | MAE ↓ | MedAE ↓ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| P ≥ 3 | 3,018 | 40.30% | 4,470 | 12.29 | 7.00 | 32.01% | 51.49% |
| P ≥ 4 | 2,848 | 38.03% | 4,640 | 11.56 | 6.50 | 32.87% | 52.84% |
| P ≥ 5 | 2,679 | 35.78% | 4,809 | 10.93 | 6.00 | 33.63% | 54.01% |
| **P ≥ 6** *(current)* | **2,533** | **33.83%** | **4,955** | **10.39** | **6.00** | **34.43%** | **55.11%** |
| P ≥ 7 | 2,430 | 32.45% | 5,058 | 10.25 | 6.00 | 34.77% | 55.80% |
| P ≥ 10 | 2,145 | 28.65% | 5,343 | **9.49** | **5.65** | **36.22%** | **57.11%** |

### Tradeoff pattern

- Looser depth → **more coverage**, **worse** MAE / ±7d  
- Stricter depth → **less coverage**, **better** MAE / ±7d  
- Moving **3 → 6** gains **~1.9d MAE** and **+3.6pp ±7d** while losing **~6.5pp** coverage  
- Moving **6 → 10** gains **~0.9d MAE** and **+2.0pp ±7d** while losing **~5.2pp** coverage  

---

## 4. HIGH vs MEDIUM by threshold

### HIGH (core → personal median)

| Depth gate | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| P ≥ 3 | 2,193 | 8.70 | 5.00 | 40.54% | 62.47% |
| P ≥ 4 | 2,065 | 8.22 | 4.50 | 41.60% | 64.02% |
| P ≥ 5 | 1,921 | 7.59 | 4.00 | 42.89% | 65.80% |
| **P ≥ 6** | **1,819** | **7.39** | **4.00** | **43.76%** | **66.74%** |
| P ≥ 7 | 1,740 | 7.25 | 4.00 | 44.25% | 67.59% |
| P ≥ 10 | 1,522 | **6.90** | **4.00** | **46.12%** | **68.73%** |

### MEDIUM (secondary → production XGB)

| Depth gate | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| P ≥ 3 | 825 | 21.84 | 16.35 | 9.33% | 22.30% |
| P ≥ 4 | 783 | 20.37 | 15.69 | 9.83% | 23.37% |
| P ≥ 5 | 758 | 19.41 | 15.29 | 10.16% | 24.14% |
| **P ≥ 6** | **714** | **18.05** | **14.40** | **10.64%** | **25.49%** |
| P ≥ 7 | 690 | 17.79 | 14.24 | 10.87% | 26.09% |
| P ≥ 10 | 623 | **15.80** | **13.18** | **12.04%** | **28.73%** |

HIGH stays reminder-usable across all gates. MEDIUM remains weak under the unretrained production XGB (consistent with Hybrid Evaluation).

---

## 5. Regular vs irregular (at depth gate)

### Regular (NormMAD≤0.35 & Drift≤7, among rows with `P ≥ gate`)

| Depth gate | Cohort | Accepted | Coverage in cohort | MAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|
| P ≥ 3 | 3,349 | 2,193 | 65.5% | 8.70 | 40.5% | 62.5% |
| P ≥ 4 | 3,145 | 2,065 | 65.7% | 8.22 | 41.6% | 64.0% |
| P ≥ 5 | 2,932 | 1,921 | 65.5% | 7.59 | 42.9% | 65.8% |
| **P ≥ 6** | **2,789** | **1,819** | **65.2%** | **7.39** | **43.8%** | **66.7%** |
| P ≥ 7 | 2,671 | 1,740 | 65.1% | 7.25 | 44.3% | 67.6% |
| P ≥ 10 | 2,352 | 1,522 | 64.7% | 6.90 | 46.1% | 68.7% |

Accepted regular rows ≡ HIGH counts (core route).

### Irregular (P≥gate and not core-regular)

| Depth gate | Cohort | Accepted | Coverage in cohort | MAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|
| P ≥ 3 | 2,857 | 825 | 28.9% | 21.84 | 9.3% | 22.3% |
| P ≥ 4 | 2,644 | 783 | 29.6% | 20.37 | 9.8% | 23.4% |
| P ≥ 5 | 2,524 | 758 | 30.0% | 19.41 | 10.2% | 24.1% |
| **P ≥ 6** | **2,366** | **714** | **30.2%** | **18.05** | **10.6%** | **25.5%** |
| P ≥ 7 | 2,238 | 690 | 30.8% | 17.79 | 10.9% | 26.1% |
| P ≥ 10 | 1,946 | 623 | 32.0% | 15.80 | 12.0% | 28.7% |

Accepted irregular rows ≡ MEDIUM counts.

---

## 6. Interpretation

1. **Monotone tradeoff:** Every step up in `P` improves MAE / ±7d and reduces coverage.
2. **P ≥ 6 remains a balanced operating point** (current production): 33.8% coverage, MAE 10.4, ±7d 55.1%, HIGH ±7d 66.7%.
3. **P ≥ 10** is the best-accuracy gate if coverage can shrink to ~29% (MAE 9.5, HIGH ±7d 68.7%).
4. **P ≥ 3–5** buy coverage (up to 40%) but pull overall MAE toward 11–12d and dilute HIGH quality.
5. **MEDIUM quality is the limiter** at every threshold while production squared-error XGB remains the secondary predictor; depth alone cannot fix MEDIUM.

---

## 7. Non-binding recommendation

| Goal | Suggested depth gate |
|---|---|
| Balanced pilot (current) | **Keep P ≥ 6** |
| Max precision / smaller pilot | Consider **P ≥ 7** or **P ≥ 10** |
| Max coverage | **P ≥ 3–5** only if MEDIUM branch is improved (e.g. authorized quantile retrain) first |

**No production threshold change was made in this experiment.**

---

## 8. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** |
| Production code modified | **No** |
| Data / reminder logic / WhatsApp modified | **No** |
| WhatsApp messages sent | **0** |

Outputs only:

- `docs/PHASE_17D_ELIGIBILITY_THRESHOLD_EXPERIMENT.md` (this file)
- `data/refillcare/processed/phase17d_eligibility_threshold_results.json`
- `scripts/phase17d_eligibility_threshold_experiment.py`
