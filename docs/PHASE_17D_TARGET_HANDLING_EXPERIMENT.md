# Phase 17D — Target Handling Experiment (Offline)

**Date:** 2026-09-17  
**Status:** Experimental only — production model/data **not** modified  
**Evidence basis:** `docs/PHASE_17D_LONG_INTERVAL_ANALYSIS.md`  
**Script:** `scripts/phase17d_target_handling_experiment.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_target_handling_results.json`

---

## 1. Purpose

Test **evidence-supported** ways to handle the heavy-tailed refill target, without changing features, splits, or the production objective.

Long-interval analysis showed:

- **>180d** = **12.3%** of supervised rows, mostly irregular / cold-start / return-after-absence (not genuine refill)
- **>120d** is arguably non-core for reminder training
- Raw ultra-long labels pull `reg:squarederror` into over-long predictions on short/medium holdouts

---

## 2. Controls (identical across strategies)

| Control | Value |
|---|---|
| Features | Same Phase 4 numeric + categorical set (24 → 70 after OHE) |
| Training data | `train.parquet`, `target > 0` (**465,610** rows) |
| Validation | `2026-05-01` → `2026-06-30` (**12,767** rows) |
| Test | `2026-07-01` → `2026-08-30` (**7,488** rows) |
| Preprocessing | **Single** fitted `ColumnTransformer` reused for all strategies |
| Model | XGBoost `reg:squarederror`, `n_estimators=150`, `max_depth=6`, `lr=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42` |
| Evaluation | Always vs **original** (uncapped / untransformed) holdout targets |

Only the **training target encoding** changed.

### Train label shape (raw)

| Stat | Value |
|---|---:|
| Mean / median | 96.9 / 31.0 days |
| P95 / max | 450 / 2060 days |
| % >120d | 18.0% |
| % >180d | 13.0% |

### Holdout note

Val max actual ≈ **121d**, Test max actual = **60d**. No `>180d` actuals appear in these temporal holdouts (right-truncation / recent-window effect). Gains below are therefore measured on **operationally relevant short–medium refill windows**, which is the intended ReminderCare use case.

---

## 3. Strategies tested

| Strategy | Training target | Evidence |
|---|---|---|
| **unchanged** | \(y\) | Control (current Phase 4 / production) |
| **cap_180** | \(\min(y, 180)\) | >180d mostly irregular/churn; dominates MAE |
| **cap_120** | \(\min(y, 120)\) | Analysis: >120d arguably non-core refill signal |
| **log1p** | \(\log(1+y)\); predict \(\mathrm{expm1}(\hat{z})\) | Robust transform for heavy-tailed day counts |

Caps changed **60,284** (`cap_180`) / **83,753** (`cap_120`) train labels. `log1p` remaps all train labels onto a compressed scale.

---

## 4. Overall results

### 4.1 Validation

| Strategy | MAE ↓ | MedAE ↓ | RMSE ↓ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|
| **log1p** | **16.13** | **9.13** | **24.59** | **21.55%** | **42.37%** |
| cap_120 | 17.76 | 11.88 | 24.51 | 15.14% | 33.35% |
| cap_180 | 22.11 | 13.85 | 31.34 | 13.40% | 29.70% |
| unchanged *(current)* | 42.93 | 19.69 | 68.44 | 9.53% | 22.25% |

### 4.2 Test

| Strategy | MAE ↓ | MedAE ↓ | RMSE ↓ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|
| **log1p** | **15.67** | **8.46** | **24.62** | **23.80%** | **44.55%** |
| cap_120 | 19.06 | 12.62 | 26.26 | 14.42% | 32.09% |
| cap_180 | 23.80 | 14.63 | 33.80 | 12.73% | 29.13% |
| unchanged *(current)* | 43.29 | 19.61 | 69.58 | 9.67% | 23.14% |

### Ranking (Validation MAE)

1. `log1p`
2. `cap_120`
3. `cap_180`
4. `unchanged` (current production target)

**vs unchanged (Test):** `log1p` cuts MAE **43.3 → 15.7** (−64%) and lifts ±7d **23% → 45%**.

---

## 5. Prediction distribution (Test)

Actual test targets: mean **18.97**, median **17**, max **60**.

| Strategy | Pred mean | Pred median | Pred max | Bias pattern |
|---|---:|---:|---:|---|
| unchanged | 61.25 | 38.71 | 332.5 | Strong over-predict / long tail |
| cap_180 | 41.42 | 34.97 | 140.7 | Still high |
| cap_120 | 36.48 | 33.51 | 110.0 | Better, still high |
| **log1p** | **31.62** | **26.67** | **172.5** | Closest to actual median |

---

## 6. Cohort results (Test)

### 6.1 By history depth

#### 2 purchases

| Strategy | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| unchanged | 1282 | 132.81 | 138.03 | 0.0% | 0.0% |
| cap_180 | 1282 | 55.33 | 56.62 | 0.2% | 0.7% |
| cap_120 | 1282 | 40.54 | 41.17 | 0.4% | 2.2% |
| **log1p** | 1282 | **37.12** | **35.37** | **3.3%** | **6.5%** |

#### 3–5 purchases

| Strategy | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| unchanged | 1051 | 57.31 | 41.29 | 0.9% | 1.6% |
| cap_180 | 1051 | 33.48 | 25.78 | 2.8% | 8.1% |
| cap_120 | 1051 | 26.40 | 21.22 | 4.4% | 11.6% |
| **log1p** | 1051 | **21.73** | **13.42** | **11.7%** | **28.2%** |

#### >5 purchases (deep chronic)

| Strategy | Count | MAE | MedAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|
| unchanged | 5155 | 18.17 | 11.55 | 13.9% | 33.3% |
| cap_180 | 5155 | 13.99 | 9.13 | 17.9% | 40.5% |
| cap_120 | 5155 | 12.22 | 8.28 | 20.0% | 43.7% |
| **log1p** | 5155 | **9.10** | **5.57** | **31.4%** | **57.4%** |

### 6.2 By actual interval band (Test)

| Band | Count | unchanged MAE | cap_180 | cap_120 | **log1p** |
|---|---:|---:|---:|---:|---:|
| 0–14d | 3221 | 52.29 | 29.64 | 24.40 | **19.48** |
| 15–45d | 4075 | 36.16 | 19.36 | 15.01 | **12.52** |
| 46–90d | 192 | 43.61 | 20.26 | 15.23 | **18.45** |

`cap_120` is slightly better than `log1p` on the thin **46–90d** test band (15.2 vs 18.5 MAE); `log1p` wins overall and on the dominant 0–45d / deep-history cohorts.

### 6.3 Validation interval band 91–180d (n=190)

Holdout still has a small longish band on validation:

| Strategy | MAE | ±7d |
|---|---:|---:|
| unchanged | 55.96 | 6.3% |
| **cap_180** | **36.39** | **9.0%** |
| cap_120 | 43.80 | 0.5% |
| log1p | 49.48 | 3.2% |

Hard caps preserve more mass near longer labels; `log1p` / `cap_120` trade some accuracy on rare long holdout gaps for large gains on short/medium refill.

---

## 7. Interpretation

1. **Unchanged raw targets are a major accuracy bottleneck** under `reg:squarederror`. Training mean ≈97d / P95=450d teaches the model to over-predict on recent holdouts (pred mean 61 vs actual ~19).
2. **Validated upper caps help a lot.** Both `cap_180` and `cap_120` cut Test MAE roughly in half vs unchanged, consistent with long-interval evidence that ultra-long gaps are not core refill signal.
3. **`cap_120` beats `cap_180`** on overall MAE / ±7d — aligning with the stronger “exclude >120d from core training” reading of the analysis.
4. **Best strategy in this experiment: `log1p`.** It compresses the heavy tail without hard truncating genuine mid-length cycles, yielding:
   - Test MAE **15.67**
   - Test MedAE **8.46**
   - Test ±7d **44.55%**
   - Deep history MAE **9.10**, ±7d **57.4%**
5. Target handling alone (with production squared-error objective) nearly matches the earlier **absolute-error** objective experiment and approaches **quantile q50**, confirming label scale is as important as loss choice.

---

## 8. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** |
| `phase4_model_report.json` mtime unchanged | **Yes** |
| Production model replaced | **No** |

Outputs written by this experiment only:

- `docs/PHASE_17D_TARGET_HANDLING_EXPERIMENT.md` (this file)
- `data/refillcare/processed/phase17d_target_handling_results.json`
- `scripts/phase17d_target_handling_experiment.py`

---

## 9. Recommendation (non-binding)

For a future explicit retrain (not done here):

1. Prefer **`log1p` target transform** (with `expm1` at inference), or at minimum **`cap_120`**, over raw uncapped days under squared error.
2. Still consider combining with MAE/quantile objectives from `PHASE_17D_XGB_OBJECTIVE_EXPERIMENT.md` in a later factorial experiment.
3. Keep deep-history (≥5 purchases) as the primary reminder cohort — `log1p` reaches ~**9d MAE / ~57% within ±7d** there under squared error alone.

**No production change was made in this phase.**
