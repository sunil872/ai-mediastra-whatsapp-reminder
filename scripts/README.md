# RefillCare Enterprise Engineering & Operational Scripts

This directory contains production utilities, model training pipelines, clinical validations, and live operational dispatch tools for the RefillCare platform.

---

## 1. Core Model Training & Artifact Generation

| Script | Purpose | Output Artifact |
|---|---|---|
| [`train_chronic_specialized_model.py`](file:///scripts/train_chronic_specialized_model.py) | Trains the 3-head Quantile Gradient Boosted model ($P_{10}, P_{50}, P_{90}$) with Median-loss optimization and acute medication purging. | `data/refillcare/processed/models/chronic_refill_model.joblib`<br>`data/refillcare/processed/chronic_model_benchmark_report.json` |
| [`train_human_ml_model.py`](file:///scripts/train_human_ml_model.py) | Trains the Path A Human Consensus ML model on clinical transitions with daily dosage recognition and inventory carryover. | `data/refillcare/processed/models/human_ml_refill_model.joblib` |

---

## 2. Clinical Benchmarks & Longitudinal Validation

| Script | Purpose | Benchmark Metrics |
|---|---|---|
| [`run_offline_validation.py`](file:///scripts/run_offline_validation.py) | Multi-strategy evaluation on holdout test partition comparing Historical Median, Days-of-Supply, and Gradient Boosted models. | Benchmark summary & evaluation logs |
| [`backtest_human_scenarios.py`](file:///scripts/backtest_human_scenarios.py) | Backtests the 5 key behavioral archetypes (Multi-pack, Early Top-up, Partial Purchase, Post-lapse, Normal cadence). | Generates `data/refillcare/processed/human_backtest_results.json` |
| [`backtest_consumption_dos_vs_hybrid.py`](file:///scripts/backtest_consumption_dos_vs_hybrid.py) | Longitudinal comparison of pure Days-of-Supply vs Hybrid cadence modeling. | Evaluation log |

---

## 3. Data Cleansing & Ingestion Auditing

| Script | Purpose | Description |
|---|---|---|
| [`audit_channel_impact.py`](file:///scripts/audit_channel_impact.py) | Verifies complete upstream isolation of B2B inter-store wholesale transfers (`SB/...`) from patient retail sales (`S0/...`). | Generates `data/refillcare/processed/channel_filter_audit.json` |
| [`verify_column_aliases.py`](file:///scripts/verify_column_aliases.py) | Validates normalization rules across heterogeneous pharmacy CSV and XLSX headers. | Verification report |

---

## 4. Live Operational Dispatch & WhatsApp Testing

| Script | Purpose | Safety Mode |
|---|---|---|
| [`execute_single_live_test.py`](file:///scripts/execute_single_live_test.py) | Dispatches a controlled, single-recipient template message via the Xinno WhatsApp Gateway. | Supports DRY-RUN / LIVE |
| [`execute_bulk_live_test.py`](file:///scripts/execute_bulk_live_test.py) | Dispatches batch-gated reminder stages for today's operational queue. | Default: DRY-RUN |
| [`verify_image_bulk_dry_run.py`](file:///scripts/verify_image_bulk_dry_run.py) | Dry-run verification of promotional media/image campaign payload formatting. | DRY-RUN |

---

## 5. System Health & Notebook Automation

| Script | Purpose | Description |
|---|---|---|
| [`verify_system_health.py`](file:///scripts/verify_system_health.py) | End-to-end verification of database tables, model loading, pipeline execution, and sample predictions. | Health status verification |
| [`build_refillcare_notebooks.py`](file:///scripts/build_refillcare_notebooks.py) | Generates reproducible Jupyter demonstration notebooks in `notebooks/`. | Notebooks builder |
