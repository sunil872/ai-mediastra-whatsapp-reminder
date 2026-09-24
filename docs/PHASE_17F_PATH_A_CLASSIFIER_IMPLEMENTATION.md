# Phase 17F — Path A Classifier Implementation (Non-Production)

**Date:** 2026-09-18  
**Status:** Implemented behind feature flag — **default OFF**. Production prediction still uses `hybrid_routing_v17d`.  
**Scope:** Path A only (`purchase_count >= 6`). Path B **not** implemented.  
**WhatsApp / reminder stages / production model:** **unchanged**.

---

## 1. Purpose

Implement the approved Phase 17E Path A decision tree as a **feature-flagged, non-production classifier**, without replacing the current hybrid baseline.

---

## 2. Files changed

| File | Change |
|---|---|
| `refillcare/models/path_a_classifier.py` | **New** — Path A v17F classifier + flag + diagnostics helpers |
| `refillcare/features/engineering.py` | Add leakage-safe diagnostics: `pct_extreme_lt10_or_gt180`, `pct_in_refill_band_15_120`, `pct_within_50pct_of_median`, `max_over_median` |
| `refillcare/models/__init__.py` | Export Path A classifier APIs (baseline hybrid exports retained) |
| `tests/refillcare/test_phase17f_path_a_classifier.py` | **New** — canonical + safety regression tests |
| `scripts/phase17f_path_a_cohort_counts.py` | **New** — offline current vs proposed cohort counts |
| `docs/PHASE_17F_PATH_A_CLASSIFIER_IMPLEMENTATION.md` | This document |

**Not modified:** `prediction.py` default routing, WhatsApp, reminder scheduler stages, `refill_model.joblib`, Path B.

---

## 3. Classifier behavior

### Strategy names

| Strategy | Role |
|---|---|
| `hybrid_routing_v17d` | **Baseline / production** (unchanged) |
| `path_a_classifier_v17f` | Non-production Path A classifier |

### Feature flag

```text
USE_PATH_A_CLASSIFIER_V17F  (default False)
ENV: REFILLCARE_USE_PATH_A_V17F=1|true|yes|on
```

- `classify_path_a(...)` / `classify_path_a_from_intervals(...)` always available for offline / tests.
- `evaluate_eligibility(...)` uses baseline hybrid unless flag is on or `use_path_a_v17f=True`.
- `predict_refill_date` continues to call `evaluate_hybrid_eligibility` (baseline). Flag does **not** auto-switch live prediction in this phase.

### Decision tree (Path A)

```
1. purchase_count < 6                         → UNSTABLE
2. hist_median ∉ [15, 120]                    → UNSTABLE
3. NormMAD or Drift missing                   → UNSTABLE
4. pct(interval <10 OR >180) >= 35%           → UNSTABLE
5. pct(interval in [15,120]) < 55%            → UNSTABLE
6. max/median >= 3.5 AND pct within ±50% < 65 → UNSTABLE
7. HIGH if:
     (NormMAD <= 0.30 AND Drift <= 7 AND max/median <= 2.2)
     OR
     (NormMAD <= 0.35 AND Drift <= 5 AND max/median <= 2.0)
8. MEDIUM if NormMAD <= 0.55 AND Drift <= 15
     (after steps 1–6 pass)
9. Otherwise                                  → UNSTABLE
```

`UNSTABLE` maps to routing `REJECTED` / confidence `REJECTED` for compatibility; business label is exposed as `path_a_class` / `label`.

### Diagnostic features (engineering)

Computed leakage-safe from expanding prior intervals at each purchase event:

- `pct_extreme_lt10_or_gt180`
- `pct_in_refill_band_15_120`
- `pct_within_50pct_of_median`
- `max_over_median`

These are **eligibility diagnostics**, not added to production `NUMERIC_FEATURES` / model training inputs.

---

## 4. Regression tests

| Case | Expected |
|---|---|
| `30,30,31,29,30,32` | **HIGH** |
| `30,60,45,30,90,30,60,30,90` | **MEDIUM** (baseline hybrid rejects) |
| `5,180,12,240,3,150` | **UNSTABLE** |
| Contaminated `[30]×7 + [90]` (low NormMAD/Drift, max/median > 2.2) | **not HIGH** |
| Missing NormMAD/Drift | **UNSTABLE** |
| Missing diagnostics without intervals | **UNSTABLE** |
| `purchase_count < 6` | **UNSTABLE** (no Path B) |
| Feature flag default | **OFF** |
| Engineering emits diagnostic columns | pass |

---

## 5. Test results

```text
python -m pytest tests/refillcare -q
167 passed
```

(Previously 155; +12 Phase 17F tests.)

---

## 6. Current vs proposed cohort counts

Same temporal test holdout (`target > 0`, N = **7,488**). Path A slice: `purchase_count >= 6`, N = **5,155**.

| Class | Baseline `hybrid_routing_v17d` | Path A `path_a_classifier_v17f` | Δ |
|---|---:|---:|---:|
| HIGH | 1,819 | 669 | −1,150 |
| MEDIUM | 714 | 1,540 | +826 |
| UNSTABLE | 2,622 | 2,946 | +324 |
| Accepted (H+M) | 2,533 | 2,209 | −324 |

Raw JSON: `data/refillcare/processed/phase17f_path_a_cohort_counts.json`

Interpretation matches Phase 17E: HIGH becomes stricter/purer; MEDIUM absorbs genuine variable recurrers; overall accepted coverage drops modestly.

---

## 7. Safety confirmations

| Check | Result |
|---|---|
| Production model `refill_model.joblib` overwritten / refit | **No** (mtime unchanged: `1789645154.9395924`) |
| WhatsApp messages sent | **0** |
| Reminder stages modified | **No** |
| Live sending enabled | **No** |
| Path B implemented | **No** |
| `hybrid_routing_v17d` replaced | **No** (still baseline) |
| Path A feature flag default | **OFF** |

---

## 8. Stop point

Phase 17F stops after implementation + tests + cohort counts.

**Not done in this phase (deferred):** hybrid evaluation of HIGH=median / MEDIUM=safety-gated XGB under the new Path A labels.
