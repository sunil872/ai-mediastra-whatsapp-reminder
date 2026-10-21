"""RefillCare ML lifecycle: model registry, evaluation, promotion gates and outcomes.

Principle: no accuracy number is shown to users unless an evaluation run measured it on held-out
future data and stored it in ``ml_evaluation_runs``.
"""
