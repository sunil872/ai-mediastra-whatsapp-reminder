# Phase 17D — Final RefillCare Prediction Architecture (Evidence-Based)

**Date:** 2026-09-17  
**Status:** Recommendation only — **production code / model / reminder logic / WhatsApp not modified**  
**Basis:** All Phase 17D experiment reports listed in §11  
**Eval holdout:** Untouched temporal test set (`test.parquet`, typically `target > 0`, **N=7,488** unless a cited report used full N=7,556)

---

## Executive recommendation

**Selected architecture: eligibility-gated hybrid routing**

| Layer | Choice | Why (measured) |
|---|---|---|
| Who gets an automated predicted date | Hybrid eligibility (Candidate-5 + depth + cadence band) | Business metrics: ±7d **62.6%** on emitted rows vs **23.5%** for current XGB always-on |
| Core regulars | Personal historical median | Class-1 ±7d **~73.7%**; beats production XGB and matches/beats experimental XGB on ±3d |
| Secondary eligible | Improved XGBoost (`reg:quantileerror`, α=0.5) | Best always-on learner; best MAE on eligible subset (**7.39d**) |
| Everyone else | Reject automated timeline; pack / manual | Low-history current-XGB MAE **~117d**, ~100% late |

Current production `reg:squarederror` XGBoost is **not** the selected predictor for reminder dates.

---

## Head-to-head (untouched test, target>0, N=7,488)

Sources: `PHASE_17D_HYBRID_COMPARISON`, `PHASE_17D_BUSINESS_METRICS`, `PHASE_17D_XGB_OBJECTIVE_EXPERIMENT`.

| Strategy | Coverage | Date MAE | MedAE | ±3d | ±7d | Late bias | Reminder fitness |
|---|---:|---:|---:|---:|---:|---|---|
| **Current XGBoost** | 100% | 43.2 | 19.6 | 9.9% | 23.5% | **91% late** (mean +47d when late) | Not reminder-ready |
| **Personal cadence** (tiered) | 100% | 14.1 | 7.0 | 31.9% | 50.4% | ~63% late | Best always-on heuristic |
| **Improved XGBoost** (quantile q50) | 100% | **13.1** | 7.4 | 26.6% | 48.3% | ~73% late | Best always-on ML |
| **Hybrid routing** | **33.8%** | **7.8** | **5.0** | **38.7%** | **62.6%** | ~60% late (mean +9d) | **Best precision product** |

On the **same hybrid-eligible subset (n=2,533)**:

| Strategy | MAE | ±7d |
|---|---:|---:|
| Improved XGB | **7.39** | 62.5% |
| Hybrid | 7.77 | **62.6%** |
| Personal median / cadence | 8.04 | ~62% |
| Current XGB | 14.33 | 35.5% |

**Decision rule used here:** Prefer hybrid as the **system architecture** (eligibility + routing). Use improved XGB only inside the secondary eligible branch (and optionally as an A/B against “always quantile on eligible”). Do **not** promote current production XGB.

---

## 1. Selected prediction strategy

**Name:** `hybrid_routing_v17d`

```
IF eligible:
  IF core_regular:
      predicted_days = personal_historical_median
  ELSE:
      predicted_days = improved_xgboost_quantile_q50
ELSE:
      REJECT (no automated predicted refill date)
```

**Not selected as primary always-on predictors:**

- Current production XGBoost — loses on MAE, ±3d, ±7d, and is systematically late (`PHASE_17D_BASELINE`, business metrics).
- Always-on personal cadence — strong heuristic overall (MAE 14.1) but unsafe on irregulars/long gaps without gates; better as **fallback pack logic for rejected low-history**, not as sole chronic predictor.
- Always-on improved XGB — best full-coverage ML, but ~52% of all test events still outside ±7d; reminder spam risk without eligibility.

---

## 2. Eligibility rule

Emit an automated predicted refill date **only if all** hold:

| Gate | Threshold | Evidence |
|---|---|---|
| History depth | `purchase_count_so_far >= 6` | History-depth analysis: recommended minimum; deep chronic usable |
| Cadence band | `historical_interval_median ∈ [15, 120]` | Long-interval + `is_recurring_history` definition; ultra-long gaps mostly churn |
| Broad regularity | `NormMAD <= 0.50` **and** `\|Recent3 − HistMed\| <= 10` | Regularity Candidate 5 (balanced coverage) |

**Measured coverage on test (target>0):**

- Events: **2,533 / 7,488 (33.8%)**
- Customers with ≥1 opportunity: **580 / 1,034 (56.1%)**
- Customer–item pairs: **1,774 / 4,611 (38.5%)**
- Excluded events: **4,955 (66.2%)**
- Excluded customers: **454 (43.9%)**

---

## 3. Regularity rule

Compute leakage-safe expanding stats from prior intervals only:

- `MAD = median(|Δt − median(Δt)|)`
- `NormMAD = MAD / median(Δt)` (median > 0)
- `Drift = |median(last 3 intervals) − historical_interval_median|`

**Routing inside eligible:**

| Tier | Rule | Predictor |
|---|---|---|
| **Core regular** | `NormMAD <= 0.35` **and** `Drift <= 7` | Personal historical median |
| **Secondary chronic** | Eligible but not core | Improved XGB (quantile q50) |

Evidence:

- Class 1 (core): personal median MAE **6.14**, ±7d **73.7%** (`PHASE_17D_REGULARITY_ANALYSIS` / hybrid cohorts).
- Candidate 6 = core thresholds; Candidate 5 = outer eligibility (`PHASE_17D_REGULARITY_THRESHOLD_SELECTION`).
- Hybrid route mix on test: **1,819 core median** + **714 secondary XGB**.

---

## 4. Low-history handling

| Depth | Automated predicted date? | Handling |
|---|---|---|
| `P <= 3` | **No** (reject) | Fixed **30-day pack default** / category pack / manual pharmacy review — **not** unconstrained ML |
| `P = 4–5` | **No** under hybrid eligibility (`P>=6`) | Same non-ML defaults; cadence experiment’s recent-3 is optional offline heuristic only |
| `P >= 6` | Only if regularity + cadence gates pass | Hybrid as above |

Evidence:

- Low-history (`P<=3`, n=1,699): current XGB MAE **116.9**, ±7d **0.2%,** ~100% late.
- Hybrid excludes **100%** of low-history events by design (`PHASE_17D_BUSINESS_METRICS`).
- Cadence tiered 30d pack is the least-bad always-on option for `P<=3` (MAE **15.0**) but still only ~25% ±7d — **not** WhatsApp multi-stage grade.

---

## 5. Model objective

For the **improved XGBoost** branch (and any future retrain of the ML component):

| Setting | Selected value | Evidence |
|---|---|---|
| Objective | **`reg:quantileerror`** | Best of squared / absolute / pseudohuber / quantile on identical controls |
| Quantile | **`quantile_alpha = 0.5`** | Median regression; Test MAE **13.05**, ±7d **48.3%** always-on |
| Rejected default | `reg:squarederror` (current production) | Test MAE **43.3**, ±7d **23.1%**; strong over-predict |

**Secondary evidence (not required for v1):** under fixed squared error, `log1p` target transform cut Test MAE to **15.7** (`PHASE_17D_TARGET_HANDLING_EXPERIMENT`). Quantile already outperforms that on the same holdout. A future factorial (quantile × log1p / cap_120) was **not** measured — do not claim it without a new experiment.

**Training hyperparameters to keep identical to Phase 17D controls:**  
`n_estimators=150`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, same Phase 4 feature set + OHE `min_frequency=50`.

---

## 6. Training-window decision

| Option | Status | Decision |
|---|---|---|
| Full Phase 4 window (~5.8y, train through 2026-04-30) | Used by all completed Phase 17D ML experiments | **Retain for now** |
| Latest 3-year window | Experiment **did not complete** (OOM / interrupted; no valid metrics doc) | **No evidence to switch** |

**Decision:** Keep the **current 5.8-year temporal train partition** until a successful, reported 3y-vs-5.8y experiment exists. Do not shorten the window based on intuition.

Optional future training hygiene (supported by long-interval analysis, not yet re-trained end-to-end): down-weight or cap labels `>120` / `>180` days — those rows are mostly irregular/churn, not refill cadence.

---

## 7. Expected-date calculation

Keep the existing RefillCare formula (scheduler / prediction modules) — Phase 17D did not find evidence against it:

```
predicted_days = max(1.0, predicted_interval_days)   # clip non-positive
expected_refill_date = latest_purchase_date + round(predicted_days)
```

Where `predicted_interval_days` comes from the hybrid router (§1).

Reminder stages remain anchored on `expected_refill_date` (current stage offsets in scheduler). Phase 17D recommends **not changing stage math** until predicted dates are trustworthy on the eligible cohort.

---

## 8. Confidence / rejection logic

| Outcome | Condition | Confidence label | Reminder action (recommended) |
|---|---|---|---|
| **HIGH** | Eligible **and** core regular (NormMAD≤0.35, Drift≤7) | `HIGH` | Full multi-stage cycle |
| **MEDIUM** | Eligible **and** secondary (Candidate-5 but not core) | `MEDIUM` | Conservative 3-stage (−7, 0, +7) |
| **REJECT** | Fails eligibility | `REJECTED` / ineligible | No automated predicted-date WhatsApp; pack default or manual |
| **INVALID** | NaN / non-finite / negative interval after routing | `INVALID` | Same as reject |

This replaces the weaker production heuristic (`P>=3` + `is_recurring` + `std<=10`) with **measured** NormMAD/Drift gates.

Rejection is intentional: business metrics treat no-emit as **patient protection**, not a model crash.

---

## 9. Metrics achieved on the untouched test set

### 9.1 Selected hybrid (emitted predictions only)

| Metric | Value |
|---|---:|
| Events scored | 2,533 |
| Opportunity coverage | **33.83%** |
| Customers excluded | **454 (43.91%)** |
| Predicted refill date MAE | **7.77 days** |
| MedAE | **5.00 days** |
| RMSE | **11.60 days** |
| ±1d | 18.79% |
| ±3d | **38.73%** |
| ±7d | **62.57%** |
| Potential early | 34.15% (mean 6.4d when early) |
| Potential late | 59.93% (mean 9.3d when late) |
| Fail \|error\|>7 | 37.43% |
| Fail \|error\|>14 | 16.94% |

### 9.2 Reference always-on benchmarks (same test)

| Strategy | MAE | ±3d | ±7d |
|---|---:|---:|---:|
| Current XGBoost | 43.19 | 9.92% | 23.53% |
| Personal cadence | 14.08 | 31.90% | 50.40% |
| Improved XGB (quantile q50) | 13.05 | 26.56% | 48.29% |

### 9.3 Segment snapshot (hybrid)

| Segment | Coverage in segment | MAE (emitted) | ±7d |
|---|---:|---:|---:|
| Eligible recurring | 100% | 7.77 | 62.6% |
| Regular (Class 1) | 65.2% scored | 7.39 | 66.7% |
| Irregular | 30.2% scored | 8.74 | 52.0% |
| Low-history | **0%** (all excluded) | — | — |

---

## 10. Known limitations

1. **Coverage vs precision tradeoff:** Hybrid covers only ~34% of events / ~56% of customers. Pharmacy must accept pack defaults for the rest.
2. **Holdout has few ultra-long actuals:** Test max actual ≈ 60d (target>0). Long-interval conclusions come from the full supervised history; recent-window right-truncation hides >180d labels in the holdout.
3. **Training-window A/B incomplete:** No measured proof that 3y beats 5.8y (or vice versa).
4. **Quantile × log1p / cap not factorial-tested:** Objective and target-handling wins were measured separately.
5. **Hybrid vs always-quantile on eligible:** On n=2,533 eligible rows, quantile MAE (7.39) slightly beats hybrid (7.77). Hybrid wins on ±1/±3 and product clarity (median for clockwork patients). An A/B at pilot time is warranted.
6. **NormMAD/Drift require interval history:** Must be computed leakage-safe at inference; not currently first-class production features.
7. **Improved XGB exists only in-memory in experiments:** Production `refill_model.joblib` is still squared-error until an authorized retrain.
8. **No live outcome lift measured:** Metrics are offline date-error / early-late proxies. True refill-adherence or WhatsApp conversion was not measured.
9. **Personal median cold-start / long-gap RMSE remains large if used always-on** — gates are mandatory.
10. **Category-level pack defaults** for rejects are recommended operationally but were not fully re-benchmarked as a separate Phase 17D model.

---

## 11. Evidence index

| Report | What it contributed |
|---|---|
| `PHASE_17D_BASELINE.md` | Current XGB vs hist median baseline |
| `PHASE_17D_HISTORY_DEPTH_ANALYSIS.md` | `P>=6` minimum / deep chronic tiers |
| `PHASE_17D_PERSONAL_CADENCE_EXPERIMENT.md` | Hist median / recent-5 / 30d pack tiers |
| `PHASE_17D_REGULARITY_ANALYSIS.md` | NormMAD, Drift, Class 1–4 |
| `PHASE_17D_REGULARITY_THRESHOLD_SELECTION.md` | Candidate 5/6 eligibility |
| `PHASE_17D_LONG_INTERVAL_ANALYSIS.md` | >120/>180d mostly non-refill; cadence band |
| `PHASE_17D_XGB_OBJECTIVE_EXPERIMENT.md` | Quantile q50 objective winner |
| `PHASE_17D_TARGET_HANDLING_EXPERIMENT.md` | log1p/cap under squared error |
| `PHASE_17D_HYBRID_COMPARISON.md` | Strategy bake-off + routing |
| `PHASE_17D_BUSINESS_METRICS.md` | Coverage, early/late, reminder lens |

---

## 12. Implementation stance (explicit)

| Action | Now |
|---|---|
| Modify production code | **No** |
| Replace `refill_model.joblib` | **No** |
| Change WhatsApp dispatch | **No** |
| Change reminder scheduler | **No** |

This document is the **architecture contract** for a future authorized Phase 18+ implementation and pilot.

**Selected one-liner:**  
*Eligibility-gated hybrid: core regulars → personal median; secondary eligible → quantile XGBoost; low-history/irregular → reject; expected date = purchase date + round(clip(pred, ≥1)); keep 5.8y train until window A/B completes.*
