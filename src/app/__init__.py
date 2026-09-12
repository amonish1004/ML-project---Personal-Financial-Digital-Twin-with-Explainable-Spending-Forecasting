"""
Personal Financial Digital Twin — Application Package
"""

from src.app.schema import (
    EXACT_14_FEATURE_ORDER,
    FinancialState,
    validate_financial_state,
    derive_feature_vector,
)
from src.app.inference import (
    get_model,
    predict_spending,
)
from src.app.simulator import (
    simulate_scenario,
    ALLOWED_PRIMARY_INPUTS,
    DERIVED_FEATURES,
)
from src.app.explainer import (
    get_explainer,
    explain_prediction,
)

__all__ = [
    "EXACT_14_FEATURE_ORDER",
    "FinancialState",
    "validate_financial_state",
    "derive_feature_vector",
    "get_model",
    "predict_spending",
    "simulate_scenario",
    "ALLOWED_PRIMARY_INPUTS",
    "DERIVED_FEATURES",
    "get_explainer",
    "explain_prediction",
]
