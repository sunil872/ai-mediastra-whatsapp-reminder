"""Data processing package for RefillCare.

Provides data cleaning, aggregation, history creation, SALT enrichment,
validation, and end-to-end pipeline execution.
"""

from refillcare.data.cleaning import (
    load_raw_transactions,
    clean_transactions,
    aggregate_invoice_items,
)
from refillcare.data.enrichment import (
    load_salt_master,
    enrich_with_salt,
)
from refillcare.data.history import (
    create_purchase_history,
    compute_interval_statistics,
)
from refillcare.data.validation import (
    validate_dataset,
    generate_validation_report,
)
from refillcare.data.pipeline import (
    run_refillcare_data_pipeline,
)
from refillcare.data.packing import (
    parse_pack_units,
    compute_total_units_purchased,
    enrich_total_units_purchased,
)
from refillcare.data.monthly_ingestion import (
    validate_monthly_sales_data,
    ingest_monthly_sales_pipeline,
    normalize_sales_dataframe,
)
from refillcare.data.transaction_classifier import (
    classify_transaction,
    classify_transactions_df,
    filter_eligible_customer_transactions,
    get_transaction_channel_summary,
    ChannelClassification,
    ExclusionReason,
    TRANSACTION_TYPE_CUSTOMER_SALE,
    TRANSACTION_TYPE_B2B_INTER_STORE,
    TRANSACTION_TYPE_UNKNOWN,
    EXCLUSION_REASON_INTER_STORE,
    EXCLUSION_REASON_UNKNOWN_TYPE,
)

__all__ = [
    "load_raw_transactions",
    "clean_transactions",
    "aggregate_invoice_items",
    "load_salt_master",
    "enrich_with_salt",
    "create_purchase_history",
    "compute_interval_statistics",
    "validate_dataset",
    "generate_validation_report",
    "run_refillcare_data_pipeline",
    "parse_pack_units",
    "compute_total_units_purchased",
    "enrich_total_units_purchased",
    "validate_monthly_sales_data",
    "ingest_monthly_sales_pipeline",
    "normalize_sales_dataframe",
    "classify_transaction",
    "classify_transactions_df",
    "filter_eligible_customer_transactions",
    "get_transaction_channel_summary",
    "TRANSACTION_TYPE_CUSTOMER_SALE",
    "TRANSACTION_TYPE_B2B_INTER_STORE",
    "TRANSACTION_TYPE_UNKNOWN",
    "EXCLUSION_REASON_INTER_STORE",
    "EXCLUSION_REASON_UNKNOWN_TYPE",
]
