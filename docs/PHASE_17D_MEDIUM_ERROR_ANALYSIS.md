# Phase 17D — MEDIUM-Branch Error Analysis (Read-Only)

**Date:** 2026-09-17  
**Status:** Offline analysis only — production code, model, data, reminder logic, WhatsApp **unchanged**  
**Script:** `scripts/phase17d_medium_error_analysis.py`  
**Raw JSON:** `data/refillcare/processed/phase17d_medium_error_analysis_results.json`  
**Scope:** `hybrid_routing_v17d` **MEDIUM** branch only (production XGBoost)

---

## 1. Purpose

Quantify **where and why** MEDIUM (secondary) predictions fail under the current hybrid rules, using the same untouched temporal test set and the current production `reg:squarederror` XGBoost model (no retrain).

MEDIUM = eligible but **not** core HIGH:

| Gate | Value |
|---|---|
| Depth | P ≥ 6 |
| Cadence | hist median ∈ `[15, 120]` |
| Broad regularity | NormMAD ≤ 0.50 and Drift ≤ 10 |
| Core / HIGH | NormMAD ≤ 0.35 **and** Drift ≤ 7 → personal median |
| **MEDIUM** | eligible but fails core → **production XGBoost** |

---

## 2. Controls

| Control | Value |
|---|---|
| Test set | Same temporal holdout: `test.parquet`, `target > 0` |
| Universe | **7,488** rows |
| Dates | **2026-07-01 → 2026-08-30** |
| MEDIUM rows | **714** (9.5% of test; 28.2% of hybrid-accepted 2,533) |
| Predictor | `refill_model.joblib` read-only (`reg:squarederror`) |
| Model mtime | **unchanged** (`1789645154.9395924`) |
| WhatsApp | **0** |
| Retrain | **None** |

Bias = `predicted − actual` (positive = late reminder / over-prediction).

---

## 3. Overall MEDIUM performance

| Metric | Value |
|---|---:|
| N | 714 |
| MAE | **18.05** |
| MedAE | 14.40 |
| RMSE | 24.78 |
| Mean bias | **+17.08** |
| Median bias | +14.24 |
| Late (pred > actual) | **90.48%** |
| Early (pred < actual) | 9.52% |
| ±3d | 10.64% |
| ±7d | **25.49%** |
| Mean pred / actual | 37.33 / 20.25 |
| Median pred / actual | 33.35 / 19.00 |
| Pred p95 / actual p95 | 71.52 / 38.00 |

**Headline:** MEDIUM errors are dominated by **systematic late bias**. The model predicts ~17 days longer than actual on average; only ~1 in 4 land within ±7 days.

---

## 4. Absolute error buckets

| Abs error (days) | Count | % | MAE | Mean bias | % late | Mean pred | Mean actual | Mean depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–3 | 76 | 10.6% | 1.51 | +0.43 | 61.8% | 29.9 | 29.4 | 39.8 |
| 4–7 | 106 | 14.9% | 4.91 | +2.94 | 79.3% | 29.5 | 26.5 | 36.0 |
| 8–14 | 165 | 23.1% | 10.57 | +9.24 | 92.7% | 29.9 | 20.6 | 35.8 |
| **15–30** | **261** | **36.6%** | **20.41** | **+19.72** | **98.1%** | 36.1 | 16.4 | 25.9 |
| **>30** | **106** | **14.9%** | **48.86** | **+48.86** | **100%** | 65.2 | 16.3 | 15.6 |

### Takeaways

- **51.4%** of MEDIUM rows have abs error **>14d** (15–30 + >30).
- Those rows contribute **81.5%** of total absolute error mass.
- Worst bucket (>30d) is 100% late, mean pred **65** vs actual **16**, and shallower history (mean depth **15.6**).
- Best bucket (0–3d) has deeper history (mean depth **~40**) and pred ≈ actual ≈ **30d**.

---

## 5. By purchase-history depth

| Depth | Count | % | MAE | ±7d | Mean bias | Mean pred | Mean actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | 24 | 3.4% | 25.40 | 8.3% | +25.0 | 43.2 | 18.2 |
| **7–10** | **93** | **13.0%** | **34.07** | **6.5%** | **+33.6** | **55.6** | 22.0 |
| 11–20 | 211 | 29.6% | 19.77 | 19.4% | +19.3 | 38.7 | 19.4 |
| 21–50 | 268 | 37.5% | 14.00 | 33.2% | +12.8 | 33.8 | 21.0 |
| >50 | 118 | 16.5% | **10.04** | **37.3%** | +8.2 | 27.3 | 19.1 |

### Takeaways

- Error **falls sharply with depth**: MAE 34 → 10 from depth 7–10 to >50.
- Shallow-eligible (**P 6–10**, n=117) = **16.4%** of MEDIUM but **29.3%** of total abs-error mass; ±7d only **6.8%**.
- Deep history (>50) is still late-biased (+8d) but far more usable (±7d 37%).

---

## 6. By historical median interval

| Hist median | Count | % | MAE | ±7d | Mean bias | Mean pred | Mean actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| <20 | 223 | 31.2% | 14.29 | 25.6% | +13.2 | 28.6 | 15.4 |
| 20–29 | 291 | 40.8% | 16.47 | 27.5% | +15.3 | 35.2 | 19.9 |
| 30–44 | 169 | 23.7% | 19.90 | 26.0% | +19.2 | 45.0 | 25.8 |
| **45–59** | **22** | **3.1%** | **41.52** | **4.6%** | **+41.5** | **69.7** | 28.2 |
| **60–90** | **6** | **0.8%** | **68.49** | **0%** | **+68.5** | **101.2** | 32.7 |
| **91–120** | **3** | **0.4%** | **73.05** | **0%** | **+73.1** | **94.1** | 21.0 |

### Takeaways

- Vast majority of MEDIUM sits in hist median **<45d** (95.7%).
- Long personal cadence (**≥45d**, n=31) is rare but catastrophic: MAE **41–73**, 100% late, pred often **2–3×** actual.
- Even short-cadence buckets remain systematically late (pred mean well above actual and often above hist median).

---

## 7. By NormMAD

| NormMAD band | Count | % | MAE | ±7d | Mean bias | Mean drift |
|---|---:|---:|---:|---:|---:|---:|
| ≤0.35 *(drift-driven MEDIUM)* | 151 | 21.2% | 15.66 | **38.4%** | +14.7 | 8.66 |
| **0.35–0.50** | **563** | **78.9%** | **18.69** | **22.0%** | **+17.7** | 4.83 |

MEDIUM is mostly **NormMAD-elevated** (not merely drift-elevated). Drift-only MEDIUM is meaningfully better (±7d almost 2× higher).

---

## 8. By Drift

| Drift band | Count | % | MAE | ±7d | Mean bias | Mean NormMAD |
|---|---:|---:|---:|---:|---:|---:|
| ≤7 *(NormMAD-driven MEDIUM)* | 420 | 58.8% | **19.29** | 21.9% | +18.3 | 0.42 |
| 7–10 | 294 | 41.2% | 16.27 | **30.6%** | +15.3 | 0.33 |

Confirming §7: rows that fail core **only** on NormMAD (Drift still ≤7) are the larger, worse slice.

---

## 9. By predicted interval

| Pred bucket | Count | % | MAE | ±7d | Mean bias | Mean actual |
|---|---:|---:|---:|---:|---:|---:|
| 1–14 | 1 | 0.1% | 4.14 | 100% | −4.1 | 18.0 |
| 15–30 | 257 | 36.0% | 10.37 | **34.6%** | +8.8 | 15.6 |
| 31–45 | 317 | 44.4% | 15.03 | 28.1% | +14.3 | 21.9 |
| **46–60** | **85** | **11.9%** | **26.07** | **3.5%** | **+25.7** | 25.4 |
| **61–90** | **34** | **4.8%** | **48.37** | **0%** | **+48.4** | 24.7 |
| **>90** | **20** | **2.8%** | **79.61** | **0%** | **+79.6** | 25.3 |

### Takeaways

- Predictions **>45d** (n=139, 19.5%) drive most catastrophic errors; >60d (n=54) alone is **25.1%** of total abs-error mass with **0%** ±7d.
- Even “reasonable” pred 15–30 still has mean bias **+8.8** (actual mean only 15.6).

---

## 10. By actual interval

| Actual bucket | Count | % | MAE | ±7d | Mean bias | Mean pred |
|---|---:|---:|---:|---:|---:|---:|
| **0–14** | **256** | **35.9%** | **24.30** | **1.2%** | **+24.3** | **33.3** |
| 15–30 | 312 | 43.7% | 15.78 | 31.7% | +15.5 | 37.8 |
| 31–45 | 136 | 19.1% | **11.30** | **58.1%** | +7.8 | 42.2 |
| 46–60 | 10 | 1.4% | 20.77 | 10.0% | +8.8 | 60.4 |

### Takeaways

- Short actuals (**≤14d**) are the hardest: model still predicts ~33d → near-total late failure (±7d **1.2%**).
- Pattern `actual ≤14 & pred >30` (n=125, 17.5%) alone is **33.2%** of total abs-error mass.
- When actual is already 31–45d, MEDIUM is comparatively usable (±7d **58%**).

---

## 11. By prediction bias

| Bias bucket | Count | % | MAE | Mean pred | Mean actual |
|---|---:|---:|---:|---:|---:|
| early >14d | 5 | 0.7% | 18.16 | 27.8 | 46.0 |
| early 8–14d | 12 | 1.7% | 9.14 | 28.9 | 38.0 |
| early 1–7d | 51 | 7.1% | 2.86 | 29.3 | 32.2 |
| late 1–7d | 131 | 18.4% | 3.74 | 29.7 | 26.0 |
| late 8–14d | 153 | 21.4% | 10.68 | 29.9 | 19.3 |
| **late 15–30d** | **256** | **35.9%** | **20.46** | 36.3 | 15.8 |
| **late >30d** | **106** | **14.9%** | **48.86** | 65.2 | 16.3 |

**90.5% late** vs **9.5% early**. Early large misses are almost nonexistent (n=5 for early >14d).

---

## 12. Largest error patterns (ranked by share of total abs error)

| Rank | Pattern | N | % of MEDIUM | MAE | Mean bias | Share of total \|error\| | ±7d |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Abs error >14d | 367 | 51.4% | 28.63 | +28.1 | **81.5%** | 0% |
| 2 | Late bias >14d | 362 | 50.7% | 28.77 | +28.8 | **80.8%** | 0% |
| 3 | NormMAD-driven MEDIUM (Drift ≤7) | 420 | 58.8% | 19.29 | +18.3 | **62.9%** | 21.9% |
| 4 | Late / abs error >30d | 106 | 14.9% | 48.86 | +48.9 | **40.2%** | 0% |
| 5 | Actual ≤14d but pred >30d | 125 | 17.5% | 34.21 | +34.2 | **33.2%** | 0% |
| 6 | Shallow depth 6–10 | 117 | 16.4% | 32.29 | +31.8 | **29.3%** | 6.8% |
| 7 | Pred > 2× hist median | 84 | 11.8% | 39.08 | +38.9 | **25.5%** | 0% |
| 8 | Pred >60d | 54 | 7.6% | 59.94 | +59.9 | **25.1%** | 0% |
| 9 | Both NormMAD & Drift above core | 143 | 20.0% | 16.91 | +15.9 | 18.8% | 22.4% |
| 10 | Drift-driven MEDIUM (NormMAD ≤0.35) | 151 | 21.2% | 15.66 | +14.7 | 18.4% | **38.4%** |
| 11 | Actual 15–30 but pred >60d | 19 | 2.7% | 65.19 | +65.2 | 9.6% | 0% |
| 12 | Early bias >14d | 5 | 0.7% | 18.16 | −18.2 | 0.7% | 0% |

---

## 13. Synthesis — primary failure modes

1. **Systematic over-prediction (late bias)**  
   ~90% of MEDIUM rows are late; mean bias **+17d**. Early errors are negligible.

2. **Error mass concentrated in large late misses**  
   Abs error >14d is half of rows but **>80%** of total absolute error.

3. **Short actuals + long predictions**  
   When the next refill is ≤14d, production XGB still often predicts >30d → near-zero ±7d accuracy and ~1/3 of error mass.

4. **Long predicted intervals are unreliable**  
   Pred >60d (only 7.6% of MEDIUM) carries **25%** of error mass and **0%** ±7d.

5. **Shallow eligible history amplifies failure**  
   Depth 6–10: MAE **32**, ±7d **~7%**, ~29% of error mass despite being 16% of rows.

6. **NormMAD-driven MEDIUM is worse than drift-driven**  
   Most MEDIUM volume and error come from NormMAD in (0.35, 0.50] with Drift still ≤7; drift-only MEDIUM is notably better (±7d 38% vs 22%).

7. **Long personal hist median (≥45d) is rare but toxic**  
   n=31, MAE 41–73, 100% late — model extrapolates far beyond actual refill timing.

---

## 14. Implications (analysis only — no production change)

These findings explain why MEDIUM MAE (~18) and ±7d (~25%) lag HIGH (personal median) so badly under the current squared-error production model:

- The branch is not “noisy around truth”; it is **directionally late**.
- The worst cases share a signature: **shallow or elevated-NormMAD history + inflated XGB interval**, especially when actual refill is short.

Offline options already evidenced elsewhere in Phase 17D (not applied here):

- Quantile / MAE objectives reduce global over-prediction.
- Tighter MEDIUM gates (e.g. depth, pred-cap vs hist median, reject pred >60) would cut error mass at coverage cost.
- Prefer personal median / reject over XGB when pred ≫ hist median.

**This document does not change production routing, model, or messaging.**

---

## 15. Artifact safety

| Check | Result |
|---|---|
| `refill_model.joblib` mtime unchanged | **Yes** (`1789645154.9395924`) |
| `phase4_model_report.json` mtime unchanged | **Yes** |
| WhatsApp messages sent | **0** |
| Production code / reminder logic modified | **No** |
| Retrain | **No** |
