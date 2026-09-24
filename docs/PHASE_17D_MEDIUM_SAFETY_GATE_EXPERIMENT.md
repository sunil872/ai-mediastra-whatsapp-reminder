# Phase 17D — MEDIUM Prediction-Safety Gate Experiment (Read-Only)

**Date:** 2026-09-18  
**Status:** Offline experiment only — production code, model, data, reminder logic, WhatsApp **unchanged**  
**Script:** `scripts/phase17d_medium_safety_gate_experiment.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_medium_safety_gate_results.json`  
**Scope:** `hybrid_routing_v17d` **MEDIUM** branch only

---

## 1. Purpose

Test lightweight **prediction-safety gates** that reduce catastrophic MEDIUM errors without retraining or changing the production XGBoost artifact.

Compare five strategies on the same MEDIUM cohort:


| ID    | Strategy                                                                                      |
| ----- | --------------------------------------------------------------------------------------------- |
| **A** | Current XGBoost (no gate)                                                                     |
| **B** | Reject when predicted interval **> 60** days                                                  |
| **C** | Reject when prediction **> 2 ×** historical median                                            |
| **D** | Use personal historical median when prediction **> 2 ×** historical median *(still accepted)* |
| **E** | Combine **B + C** (reject if either gate fires)                                               |


---



## 2. Controls


| Control     | Value                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------- |
| Test set    | Same temporal holdout: `test.parquet`, `target > 0`                                       |
| Universe    | **7,488** rows                                                                            |
| Dates       | **2026-07-01 → 2026-08-30**                                                               |
| MEDIUM rows | **714** (hybrid secondary: P≥6, hist median ∈ [15,120], NormMAD≤0.50, Drift≤10, not core) |
| Predictor   | Production `refill_model.joblib` (`reg:squarederror`), read-only                          |
| Model mtime | **unchanged** (`1789645154.9395924`)                                                      |
| WhatsApp    | **0**                                                                                     |
| Retrain     | **None**                                                                                  |


**Metric scope:** MAE / MedAE / ±3d / ±7d are on **accepted** predictions only.  
**Coverage:** `accepted / 714` MEDIUM rows (also shown vs full test).  
**Catastrophic:** abs error **> 30 days** (same definition as MEDIUM error analysis).

Baseline MEDIUM error mass (strategy A):


| Mass                   | Value                                             |
| ---------------------- | ------------------------------------------------- |
| Total absolute error   | **12,886.32**                                     |
| Catastrophic abs error | **5,179.51** (106 rows, 40.2% of total abs error) |


---



## 3. Gate trigger counts


| Flag                   | Count | % of MEDIUM |
| ---------------------- | ----- | ----------- |
| Pred > 60              | 54    | 7.56%       |
| Pred > 2 × hist median | 84    | 11.76%      |
| Either (union for E)   | 114   | 15.97%      |
| Both gates             | 24    | 3.36%       |
| Only pred > 60         | 30    | 4.20%       |
| Only pred > 2× hist    | 60    | 8.40%       |


---



## 4. Results by strategy


| Strategy                                  | Accepted | Rejected | Coverage (MEDIUM) | Coverage (test) | MAE ↓     | MedAE ↓   | ±3d ↑      | ±7d ↑      |
| ----------------------------------------- | -------- | -------- | ----------------- | --------------- | --------- | --------- | ---------- | ---------- |
| **A** Current XGB                         | 714      | 0        | 100.0%            | 9.54%           | 18.05     | 14.40     | 10.64%     | 25.49%     |
| **B** Reject pred >60                     | 660      | 54       | 92.44%            | 8.81%           | 14.62     | 13.29     | 11.52%     | 27.58%     |
| **C** Reject pred >2× hist                | 630      | 84       | 88.24%            | 8.41%           | 15.24     | 13.17     | 12.06%     | 28.89%     |
| **D** Fallback to hist median if pred >2× | 714      | 0        | **100.0%**        | 9.54%           | 14.58     | **12.00** | **13.31%** | **30.81%** |
| **E** Combine B+C                         | 600      | 114      | 84.03%            | 8.01%           | **13.46** | 12.30     | 12.67%     | 30.33%     |




### Error mass and catastrophic contribution


| Strategy | Total abs error | vs A      | Catastrophic count | Catastrophic abs error | vs A cat  | Cat share of accepted abs error |
| -------- | --------------- | --------- | ------------------ | ---------------------- | --------- | ------------------------------- |
| **A**    | 12,886.32       | 100%      | 106                | 5,179.51               | 100%      | 40.2%                           |
| **B**    | 9,649.52        | **74.9%** | 57                 | 2,050.33               | **39.6%** | 21.3%                           |
| **C**    | 9,603.76        | 74.5%     | 63                 | 2,720.63               | 52.5%     | 28.3%                           |
| **D**    | 10,410.26       | 80.8%     | 65                 | 2,795.13               | 54.0%     | 26.9%                           |
| **E**    | **8,074.94**    | **62.7%** | **38**             | **1,299.43**           | **25.1%** | **16.1%**                       |


---



## 5. Interpretation



### A — baseline

Current MEDIUM XGB is late-biased (mean bias +17.1d). Catastrophic cases are only **14.9%** of rows but **40%** of absolute error.

### B — reject pred >60

- Cheap: rejects only **7.6%** of MEDIUM.
- Removes most extreme long predictions → total abs error **−25%**, catastrophic abs error **−60%**.
- MAE improves **18.05 → 14.62**; ±7d **25.5% → 27.6%**.
- Best **reject-only** gate for cutting catastrophic mass per rejected row.



### C — reject pred >2× hist median

- Rejects **11.8%** (more than B).
- Similar total-abs reduction to B (~**−25%**), but keeps more catastrophic residual (**52.5%** of baseline cat mass vs B’s **39.6%**).
- Slightly better ±7d than B (**28.9%**) because it catches inflated preds even when pred ≤60.



### D — fallback to personal median (no coverage loss)

- Same accepted count as A (**714**).
- Replaces 84 inflated XGB preds with hist median → MAE **14.58**, best ±7d among all (**30.8%**), best MedAE (**12.0**).
- Still retains more catastrophic error mass than B/E (fallback can still miss when hist median itself is far from actual).
- Best option if **coverage must stay 100%** of MEDIUM.



### E — B + C combined reject

- Rejects **16.0%** of MEDIUM (114 rows).
- Strongest safety: lowest MAE (**13.46**), lowest total abs error (**62.7%** of A), lowest catastrophic abs error (**25.1%** of A).
- ±7d **30.3%** — nearly matches D, with less residual catastrophic mass.
- Best if the product goal is **minimize bad MEDIUM reminders** and some extra reject is acceptable.

---



## 6. Ranking (by use case)


| Goal                                           | Prefer                       |
| ---------------------------------------------- | ---------------------------- |
| Lowest MAE / least catastrophic residual       | **E**                        |
| Keep full MEDIUM coverage                      | **D**                        |
| Smallest reject footprint with big cat cut     | **B**                        |
| Catch inflated-vs-personal-cadence misses ≤60d | **C** (or include via **E**) |


Relative to A:


| Strategy | Δ MAE     | Δ ±7d       | Coverage kept | Cat abs error remaining |
| -------- | --------- | ----------- | ------------- | ----------------------- |
| B        | −3.43     | +2.1 pp     | 92.4%         | 39.6% of A              |
| C        | −2.81     | +3.4 pp     | 88.2%         | 52.5% of A              |
| D        | −3.47     | **+5.3 pp** | **100%**      | 54.0% of A              |
| E        | **−4.59** | +4.8 pp     | 84.0%         | **25.1% of A**          |


---



## 7. Conclusion (analysis only — no production change)

Safety gates on MEDIUM predictions **materially reduce error mass** without retraining:

1. **Reject pred >60 (B)** is a high-leverage, low-coverage-cost cut of catastrophic tails.
2. **2× hist-median (C)** catches additional inflation cases B misses.
3. **Combined reject (E)** gives the strongest safety profile.
4. **Hist-median fallback (D)** improves accuracy **without rejecting anyone**, and wins on ±7d / MedAE if coverage is sacred.

These remain **offline** recommendations. Production routing, `refill_model.joblib`, reminder logic, and WhatsApp were **not** modified.

---



## 8. Artifact safety


| Check                                      | Result                         |
| ------------------------------------------ | ------------------------------ |
| `refill_model.joblib` mtime unchanged      | **Yes** (`1789645154.9395924`) |
| `phase4_model_report.json` mtime unchanged | **Yes**                        |
| WhatsApp messages sent                     | **0**                          |
| Production code modified                   | **No**                         |
| Retrain                                    | **No**                         |


