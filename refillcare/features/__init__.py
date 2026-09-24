"""Feature engineering and dataset preparation package for RefillCare."""

from refillcare.features.engineering import (
    build_feature_dataset,
    split_dataset_temporally,
    extract_feature_target_matrices,
)
from refillcare.features.consumption import (
    calculate_historical_consumption_rate,
    compute_expanding_consumption_rates,
    calculate_estimated_days_of_supply,
    calculate_target_refill_interval,
)
from refillcare.features.medication_switch import (
    extract_brand_stem,
    extract_medicine_strength,
    classify_medicine_relationship,
    detect_medication_switches_for_customer,
    calculate_switched_medication_refill,
    CLIENT_LABEL_MEDICATION_CHANGE,
)

__all__ = [
    "build_feature_dataset",
    "split_dataset_temporally",
    "extract_feature_target_matrices",
    "calculate_historical_consumption_rate",
    "compute_expanding_consumption_rates",
    "calculate_estimated_days_of_supply",
    "calculate_target_refill_interval",
    "extract_brand_stem",
    "extract_medicine_strength",
    "classify_medicine_relationship",
    "detect_medication_switches_for_customer",
    "calculate_switched_medication_refill",
    "CLIENT_LABEL_MEDICATION_CHANGE",
]
