# Phase 17D — XGBoost Objective Experiment (Offline)

**Date:** 2026-09-17  
**Status:** Experimental only — production model **not** replaced  
**XGBoost version:** 3.4.0  
**Script:** `scripts/phase17d_xgb_objective_experiment.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_xgb_objective_results.json`

---

## 1. Purpose

Compare the current Phase 4 default objective (`reg:squarederror`) against MAE-oriented / robust regression objectives supported by the installed XGBoost build, under **identical** controls:

| Control | Value |
|---|---|
| Features | Same Phase 4 numeric + categorical set (24 raw → 70 after OHE) |
| Training data | `train.parquet`, `target > 0` (**465,610** rows) |
| Validation | `2026-05-01` → `2026-06-30` (**12,767** rows) |
| Test | `2026-07-01` → `2026-08-30` (**7,488** rows) |
| Preprocessing | **Single** fitted `ColumnTransformer` (median impute + OHE `min_frequency=50`), reused for all objectives |
| Hyperparameters | `n_estimators=150`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `tree_method=hist` |

Only the **objective** (and quantile alpha where required) changed.

---

## 2. Objectives tested

| Label | XGBoost `objective` | Notes |
|---|---|---|
| `reg:squarederror` | `reg:squarederror` | **Current production default** |
| `reg:absoluteerror` | `reg:absoluteerror` | Direct MAE / L1 loss |
| `reg:pseudohubererror` | `reg:pseudohubererror` | Robust / Huber-like |
| `reg:quantileerror_q50` | `reg:quantileerror` (`quantile_alpha=0.5`) | Median regression |

---

## 3. Overall results

### 3.1 Validation

| Objective | MAE ↓ | MedAE ↓ | RMSE ↓ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|
| **reg:quantileerror_q50** | **13.99** | **8.01** | **21.25** | **25.11%** | **46.04%** |
| reg:absoluteerror | 18.33 | 10.37 | 28.10 | 18.33% | 38.09% |
| reg:pseudohubererror | 28.94 | 15.87 | 50.03 | 14.58% | 28.60% |
| reg:squarederror *(current)* | 42.93 | 19.69 | 68.44 | 9.53% | 22.25% |

### 3.2 Test

| Objective | MAE ↓ | MedAE ↓ | RMSE ↓ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|
| **reg:quantileerror_q50** | **13.05** | **7.39** | **20.36** | **26.56%** | **48.29%** |
| reg:absoluteerror | 18.77 | 10.39 | 29.32 | 18.94% | 38.46% |
| reg:pseudohubererror | 27.78 | 14.72 | 48.63 | 15.61% | 30.49% |
| reg:squarederror *(current)* | 43.29 | 19.61 | 69.58 | 9.67% | 23.14% |

### Ranking (by Validation MAE)

1. `reg:quantileerror_q50`
2. `reg:absoluteerror`
3. `reg:pseudohubererror`
4. `reg:squarederror` (current production)

---

## 4. Prediction distribution (Test)

Actual test targets: mean **18.97**, median **17.0**, max **60**.

| Objective | Pred mean | Pred median | Pred p95 | Pred max | Notes |
|---|---:|---:|---:|---:|---|
| reg:squarederror | 61.25 | 38.71 | 181.52 | 332.48 | Strong positive bias / long right tail |
| reg:absoluteerror | 35.66 | 30.27 | 83.91 | 172.22 | Closer, still somewhat high |
| reg:pseudohubererror | 40.31 | 30.47 | 129.61 | 474.11 | Wide tail; ~7.4% clipped to 1d |
| **reg:quantileerror_q50** | **28.75** | **27.34** | **64.93** | **142.85** | Closest to actual median behavior |

`reg:squarederror` systematically over-predicts on this long-history dataset (pred mean 61 vs actual ~19), which inflates MAE even when MedAE is “only” ~20 days.

---

## 5. Errors by history depth (Test)

### 2 purchases

| Objective | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| reg:squarederror | 1282 | 132.81 | 138.03 | 0.0% | 0.0% |
| reg:absoluteerror | 1282 | 43.69 | 41.13 | 2.2% | 4.8% |
| reg:pseudohubererror | 1282 | 33.24 | 28.49 | 5.3% | 13.1% |
| **reg:quantileerror_q50** | 1282 | **27.05** | **23.93** | **7.5%** | **15.7%** |

### 3–5 purchases

| Objective | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| reg:squarederror | 1051 | 57.31 | 41.29 | 0.9% | 1.6% |
| reg:absoluteerror | 1051 | 26.57 | 16.87 | 8.1% | 21.2% |
| reg:pseudohubererror | 1051 | 29.96 | 19.86 | 11.6% | 24.7% |
| **reg:quantileerror_q50** | 1051 | **18.16** | **11.38** | **16.3%** | **35.1%** |

### >5 purchases (deep chronic)

| Objective | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| reg:squarederror | 5155 | 18.17 | 11.55 | 13.9% | 33.3% |
| reg:absoluteerror | 5155 | 10.98 | 6.93 | 25.3% | 50.4% |
| reg:pseudohubererror | 5155 | 25.98 | 11.36 | 19.0% | 36.0% |
| **reg:quantileerror_q50** | 5155 | **8.52** | **5.33** | **33.4%** | **59.1%** |

---

## 6. Interpretation

1. **Current `reg:squarederror` is not competitive** on the 5.8-year dataset under these fixed controls. It loses on MAE, MedAE, ±3d, ±7d, and over-predicts badly (test pred mean 61 vs actual ~19).
2. **MAE-oriented objectives help a lot.** `reg:absoluteerror` cuts Test MAE from **43.3 → 18.8** days and lifts ±7d from **23% → 38%**.
3. **Best experimental objective:** `reg:quantileerror` with `quantile_alpha=0.5` (median regression):
   - Test MAE **13.05** days
   - Test MedAE **7.39** days
   - Test ±7d **48.29%**
   - Deep history (>5 purchases): MAE **8.52**, ±7d **59.1%**
4. `reg:pseudohubererror` is intermediate overall, but weaker than absolute/quantile on deep-history MAE and shows more extreme prediction tails.

---

## 7. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** |
| `phase4_model_report.json` mtime unchanged | **Yes** |
| Production model replaced | **No** |

This experiment saved **only**:

- `docs/PHASE_17D_XGB_OBJECTIVE_EXPERIMENT.md` (this file)
- `data/refillcare/processed/phase17d_xgb_objective_results.json` (raw metrics)
- `scripts/phase17d_xgb_objective_experiment.py` (reproducible runner)

---

## 8. Recommendation (non-binding)

For a future Phase 4 retrain (separate, explicit change):

1. Prefer **`reg:quantileerror` + `quantile_alpha=0.5`**, or at minimum **`reg:absoluteerror`**, over `reg:squarederror`.
2. Re-validate on the same temporal holdouts before replacing `refill_model.joblib`.
3. Keep deep-history (≥5 purchases) as the primary pharmacy reminder cohort — that is where accuracy becomes operationally useful (~8.5d MAE / ~59% within ±7d under quantile loss).

**No production change was made in this phase.**
