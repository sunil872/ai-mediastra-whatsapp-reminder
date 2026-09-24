# Phase 17D — Hybrid Strategy Evaluation (Read-Only)

**Date:** 2026-09-17  
**Status:** Evaluation only — no retrain, no `refill_model.joblib` overwrite, no production code changes, **0 WhatsApp messages**  
**Script:** `scripts/phase17d_hybrid_evaluation.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_hybrid_evaluation_results.json`  
**Implemented strategy under test:** `hybrid_routing_v17d`

---

## 1. Purpose

Compare four predictors on the **exact same untouched temporal test set**:

1. Current production XGBoost (`refill_model.joblib`, read-only)
2. Historical median baseline
3. Personal cadence strategy (tiered heuristic)
4. `hybrid_routing_v17d` (implemented prediction layer)

---

## 2. Controls

| Control | Value |
|---|---|
| Test set | `test.parquet`, `target > 0` |
| Test rows | **7,488** |
| Dates | **2026-07-01 → 2026-08-30** |
| Production model | Loaded read-only; **mtime unchanged** |
| Hybrid HIGH branch | Personal `historical_interval_median` |
| Hybrid MEDIUM branch | Existing production XGBoost pipeline (**not** retrained to quantile) |
| WhatsApp | **0 sends** |

Metrics for gated strategies are computed on **accepted (emitted) predictions only**. Rejected rows contribute to rejection counts, not MAE.

---

## 3. Overall comparison

| Strategy | Test rows | Accepted | Rejected | MAE ↓ | MedAE ↓ | RMSE ↓ | ±1d ↑ | ±3d ↑ | ±7d ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current XGBoost | 7,488 | 7,488 (100%) | 0 | 43.19 | 19.61 | 69.62 | 3.26% | 9.92% | 23.53% |
| Historical median | 7,488 | 7,488 (100%) | 0 | 26.72 | 7.00 | 109.43 | 16.95% | 32.02% | 50.96% |
| Personal cadence | 7,488 | 7,488 (100%) | 0 | 14.08 | 7.00 | 34.10 | 16.89% | 31.90% | 50.40% |
| **hybrid_routing_v17d** | 7,488 | **2,533 (33.83%)** | **4,955 (66.17%)** | **10.39** | **6.00** | **16.38** | **17.37%** | **34.43%** | **55.19%** |

### Ranking (accepted-prediction MAE)

1. `hybrid_routing_v17d` (10.39) — selective coverage  
2. Personal cadence (14.08) — always-on  
3. Historical median (26.72) — always-on  
4. Current XGBoost (43.19) — always-on  

---

## 4. `hybrid_routing_v17d` detail

### 4.1 Confidence / route mix

| Label | Count | Share of test |
|---|---:|---:|
| **HIGH** (core → personal median) | **1,819** | 24.3% |
| **MEDIUM** (secondary → production XGB) | **714** | 9.5% |
| **REJECTED** | **4,955** | 66.2% |

Route counts match confidence: `core_personal_median=1819`, `secondary_experimental_xgb=714`, `rejected=4955`.

### 4.2 HIGH vs MEDIUM accuracy

| Confidence | Count | MAE | MedAE | RMSE | ±1d | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HIGH** | 1,819 | **7.39** | **4.00** | **11.51** | **22.81%** | **43.76%** | **66.74%** |
| **MEDIUM** | 714 | 18.05 | 14.40 | 24.78 | 3.50% | 10.64% | 25.77% |

**Interpretation:** HIGH (personal median on core regulars) is reminder-grade. MEDIUM is weaker here because the secondary branch still uses the **existing squared-error production model** (no quantile retrain in this evaluation).

### 4.3 By purchase-history depth

| Depth | Cohort rows | Accepted | Rejected | MAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|
| ≤2 purchases | 1,282 | 0 | 1,282 (100%) | — | — | — |
| 3–5 purchases | 1,051 | 0 | 1,051 (100%) | — | — | — |
| 6–10 purchases | 1,051 | 473 (45.0%) | 578 | 15.57 | 24.52% | 44.19% |
| >10 purchases | 4,104 | 2,060 (50.2%) | 2,044 | **9.20** | **36.70%** | **57.72%** |

Low-history cohorts are fully rejected by design (`P < 6`).

### 4.4 Regular vs irregular histories

Definitions (Class-1 style):

- **Regular:** `P≥6`, NormMAD≤0.35, Drift≤7  
- **Irregular:** `P≥6` and (NormMAD>0.35 or Drift>7)

| Segment | Cohort rows | Accepted | Rejected | MAE | ±3d | ±7d |
|---|---:|---:|---:|---:|---:|---:|
| Regular (Class 1) | 2,789 | 1,819 (65.2%) | 970 | **7.39** | **43.76%** | **66.74%** |
| Irregular (≥6) | 2,366 | 714 (30.2%) | 1,652 | 18.05 | 10.64% | 25.77% |

Regular accepted rows are exactly the HIGH cohort. Irregular accepted rows are the MEDIUM cohort (Candidate-5 outer gate but not core).

---

## 5. Evidence-based conclusions

1. **`hybrid_routing_v17d` is the best precision strategy** on this holdout among the four (accepted MAE **10.39**, ±7d **55.2%**), at **33.8%** coverage.
2. **HIGH confidence** is the operational sweet spot: MAE **7.39**, ±7d **66.7%** on 1,819 events.
3. **MEDIUM** underperforms HIGH while production XGB remains the secondary model (MAE **18.05**, ±7d **25.8%**). A future authorized quantile retrain of the secondary branch is expected to improve MEDIUM (prior offline experiment: quantile MAE ~7.4 on the eligible subset).
4. **Always-on current XGBoost remains last** (MAE 43.2, ±7d 23.5%).
5. **Personal cadence** is the best always-on heuristic (MAE 14.1) but cannot match hybrid precision without eligibility gates.

---

## 6. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** |
| Production model retrained | **No** |
| Production code modified by this evaluation | **No** |
| WhatsApp messages sent | **0** |

Outputs written by this evaluation only:

- `docs/PHASE_17D_HYBRID_EVALUATION.md` (this file)
- `data/refillcare/processed/phase17d_hybrid_evaluation_results.json`
- `scripts/phase17d_hybrid_evaluation.py`
