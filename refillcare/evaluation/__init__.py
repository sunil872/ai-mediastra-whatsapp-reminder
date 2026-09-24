"""Evaluation package for RefillCare baseline and predictive models."""

from refillcare.evaluation.baseline import (
    evaluate_baseline_predictions,
    compute_prediction_metrics,
    HistoricalMedianBaseline,
)
from refillcare.evaluation.micro_pilot import (
    MicroPilotCandidate,
    MicroPilotReport,
    MicroPilotOutcomeRecord,
    MicroPilotStopCondition,
    prepare_micro_pilot_batch,
    execute_micro_pilot_batch,
    check_micro_pilot_stop_conditions,
    evaluate_micro_pilot_outcomes,
    generate_micro_pilot_summary_report,
    ALLOWED_MICRO_PILOT_BATCH_SIZES,
    DEFAULT_MICRO_PILOT_BATCH_SIZE,
)

from refillcare.evaluation.holdout_validation import run_historical_holdout_validation

__all__ = [
    "evaluate_baseline_predictions",
    "compute_prediction_metrics",
    "HistoricalMedianBaseline",
    "MicroPilotCandidate",
    "MicroPilotReport",
    "MicroPilotOutcomeRecord",
    "MicroPilotStopCondition",
    "prepare_micro_pilot_batch",
    "execute_micro_pilot_batch",
    "check_micro_pilot_stop_conditions",
    "evaluate_micro_pilot_outcomes",
    "generate_micro_pilot_summary_report",
    "run_historical_holdout_validation",
    "ALLOWED_MICRO_PILOT_BATCH_SIZES",
    "DEFAULT_MICRO_PILOT_BATCH_SIZE",
]
