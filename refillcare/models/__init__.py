"""RefillCare Machine Learning Model Training, Evaluation, and Prediction Package."""

from refillcare.models.evaluation import (
    evaluate_regression_model,
    analyze_performance_by_interval_group,
    analyze_performance_by_history_length,
    extract_feature_importances,
)
from refillcare.models.training import (
    train_and_evaluate_all_models,
    build_model_pipeline,
    save_model_bundle,
    load_model_bundle,
)
from refillcare.models.prediction import (
    predict_refill_date,
    generate_batch_predictions,
)
from refillcare.models.hybrid_strategy import (
    STRATEGY_NAME as HYBRID_STRATEGY_NAME,
    evaluate_hybrid_eligibility,
    build_improved_xgboost_regressor,
)
from refillcare.models.path_a_classifier import (
    STRATEGY_NAME as PATH_A_STRATEGY_NAME,
    USE_PATH_A_CLASSIFIER_V17F,
    classify_path_a,
    classify_path_a_from_intervals,
    evaluate_eligibility as evaluate_path_a_or_baseline_eligibility,
    is_path_a_classifier_enabled,
)
from refillcare.models.supply_hybrid import (
    calculate_hybrid_refill_date,
    generate_hybrid_supply_batch_predictions,
    is_hybrid_supply_layer_enabled,
    PREDICTION_SOURCE_SUPPLY,
    PREDICTION_SOURCE_EXISTING,
    PREDICTION_SOURCE_HYBRID_CONSENSUS,
    PREDICTION_SOURCE_REJECTED,
)

__all__ = [
    "evaluate_regression_model",
    "analyze_performance_by_interval_group",
    "analyze_performance_by_history_length",
    "extract_feature_importances",
    "train_and_evaluate_all_models",
    "build_model_pipeline",
    "save_model_bundle",
    "load_model_bundle",
    "predict_refill_date",
    "generate_batch_predictions",
    "HYBRID_STRATEGY_NAME",
    "evaluate_hybrid_eligibility",
    "build_improved_xgboost_regressor",
    "PATH_A_STRATEGY_NAME",
    "USE_PATH_A_CLASSIFIER_V17F",
    "classify_path_a",
    "classify_path_a_from_intervals",
    "evaluate_path_a_or_baseline_eligibility",
    "is_path_a_classifier_enabled",
    "calculate_hybrid_refill_date",
    "generate_hybrid_supply_batch_predictions",
    "is_hybrid_supply_layer_enabled",
    "PREDICTION_SOURCE_SUPPLY",
    "PREDICTION_SOURCE_EXISTING",
    "PREDICTION_SOURCE_HYBRID_CONSENSUS",
    "PREDICTION_SOURCE_REJECTED",
]
