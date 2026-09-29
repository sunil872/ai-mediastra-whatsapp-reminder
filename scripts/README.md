# RefillCare Engineering & Operational Scripts Directory

This directory contains utility, training, validation, benchmark, and operational scripts for the RefillCare enterprise medication reminder platform.

---

## 1. Core Model Training & Artifact Generation

| Script | Purpose | Output Artifact |
|---|---|---|
| [`train_human_ml_model.py`](file:///scripts/train_human_ml_model.py) | Trains the generalized Path A Human Consensus ML model on 330k clinical transitions with daily dosage recognition and inventory carryover. | `data/refillcare/processed/models/human_ml_refill_model.joblib` |
| [`stream_build_phase17a_dataset.py`](file:///scripts/stream_build_phase17a_dataset.py) | Low-memory streaming generator for longitudinal transition features from canonical history. | `data/refillcare/processed/training_dataset.parquet` |

---

## 2. Benchmarks, Gap Analysis & Offline Holdout Validation

| Script | Purpose | Benchmark Metrics |
|---|---|---|
| [`run_offline_validation.py`](file:///scripts/run_offline_validation.py) | 4-way evaluation on August 2026 holdout: (1) Median Baseline, (2) DOS Only, (3) Hybrid Heuristic, (4) Human Consensus ML. | Generates `data/refillcare/processed/four_way_benchmark_results.json` |
| [`detailed_august_gap_analysis.py`](file:///scripts/detailed_august_gap_analysis.py) | Granular error distribution, early/late refills, dosage breakdowns, and exact day error quantiles for August 2026. | Console report & summary metrics |
| [`backtest_human_scenarios.py`](file:///scripts/backtest_human_scenarios.py) | Evaluates the 5 key behavioral archetypes (Multi-pack, Early Top-up, Partial Purchase, Post-lapse, Normal cadence). | Generates `data/refillcare/processed/human_backtest_results.json` |
| [`backtest_consumption_dos_vs_hybrid.py`](file:///scripts/backtest_consumption_dos_vs_hybrid.py) | Longitudinal comparison of pure Days-of-Supply vs Hybrid cadence modeling. | Evaluation log |

---

## 3. Transaction Channel Classification & Wholesale Auditing

| Script | Purpose | Description |
|---|---|---|
| [`audit_channel_impact.py`](file:///scripts/audit_channel_impact.py) | Verifies complete upstream isolation of B2B inter-store wholesale transfers (`SB/...`) from patient retail sales (`S0/...`). | Generates `data/refillcare/processed/channel_filter_audit.json` |
| [`verify_column_aliases.py`](file:///scripts/verify_column_aliases.py) | Validates normalization rules across heterogeneous pharmacy CSV and XLSX headers. | Verification report |

---

## 4. Live Operational Dispatch & Testing

| Script | Purpose | Safety Mode |
|---|---|---|
| [`execute_single_live_test.py`](file:///scripts/execute_single_live_test.py) | Dispatches a controlled, single-recipient template message via the Xinno WhatsApp Gateway. | Supports DRY-RUN / LIVE |
| [`execute_bulk_live_test.py`](file:///scripts/execute_bulk_live_test.py) | Dispatches batch-gated reminder stages for today's operational queue. | Default: DRY-RUN |
| [`verify_image_bulk_dry_run.py`](file:///scripts/verify_image_bulk_dry_run.py) | Dry-run verification of media/image delivery payload formatting. | DRY-RUN |

---

## 5. Architectural Research & Phase 17 Progression

- `phase17b_validate.py` – `phase17k_path_a_eligibility_architecture_audit.py`: Historical research scripts documenting the transition from unweighted medians to MAD stability tiering, DOS bounds, and Consensus ML.
- `build_refillcare_notebooks.py`: Builds reproducible Jupyter demonstration notebooks.
