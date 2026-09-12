"""
Personal Financial Digital Twin — What-If Scenario Simulator Engine

Provides counterfactual scenario analysis for individual financial states.

Mathematical & Architectural Principles:
1. Re-uses the single prediction gateway `predict_spending()` from `src.app.inference`.
2. Ensures baseline state remains completely unmutated.
3. Enforces automatic derivation of `spending_t`, `spending_3m_mean`, and `spending_3m_std` (ddof=0).
4. Prohibits direct manual overrides of derived features.
5. Evaluates model attribution counterfactuals (non-causal interpretation).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional, Set
import copy

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.schema import (
    FinancialState,
    validate_financial_state,
    derive_feature_vector,
)
from src.app.inference import predict_spending

# Supported primary / user-editable input fields
ALLOWED_PRIMARY_INPUTS: Set[str] = {
    "ending_balance_t",
    "income_credit_t",
    "debit_count_t",
    "spending_hh_t",
    "spending_st_t",
    "spending_in_t",
    "spending_lo_t",
    "spending_io_t",
    "spending_other_t",
    "spending_t_minus_1",
    "spending_t_minus_2",
}

# Derived features prohibited from direct override
DERIVED_FEATURES: Set[str] = {
    "spending_t",
    "spending_3m_mean",
    "spending_3m_std",
}


def simulate_scenario(
    baseline_state: Union[FinancialState, Dict[str, Any]],
    scenario_changes: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Simulates a counterfactual what-if scenario against a baseline financial state.

    Process:
    1. Validates baseline financial state without modifying original inputs.
    2. Validates scenario modifications (rejects unknown or derived feature overrides).
    3. Creates a scenario copy and applies modifications to primary inputs.
    4. Automatically recalculates derived features via schema validation/derivation logic.
    5. Obtains baseline and scenario spending forecasts using predict_spending gateway.
    6. Computes absolute and percentage prediction differences safely.

    Parameters
    ----------
    baseline_state : FinancialState or Dict[str, Any]
        Initial valid financial state profile.
    scenario_changes : Dict[str, Any]
        Dictionary of allowed primary input overrides (e.g. {"spending_hh_t": 700.0}).
    model_path : Path or str, optional
        Path to model artifact (defaults to models/catboost_model.joblib).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - baseline_state: Dict representation of validated baseline
        - scenario_state: Dict representation of scenario state with recalculated derived features
        - baseline_prediction: Baseline spending forecast (CZK)
        - scenario_prediction: Scenario spending forecast (CZK)
        - absolute_difference: scenario_prediction - baseline_prediction (CZK)
        - percentage_difference: Percentage change float, or None if baseline is zero/near-zero
    """
    # 1. Validate baseline state
    validated_baseline = validate_financial_state(baseline_state)

    if not isinstance(scenario_changes, dict):
        raise TypeError(f"scenario_changes must be a dictionary, got {type(scenario_changes).__name__}")

    # 2. Validate scenario change keys
    for key in scenario_changes.keys():
        if key in DERIVED_FEATURES:
            raise ValueError(
                f"Direct override of derived feature '{key}' is prohibited. "
                f"Derived features are calculated automatically from category spendings and lags."
            )
        if key not in ALLOWED_PRIMARY_INPUTS:
            raise ValueError(
                f"Invalid or unsupported scenario field '{key}'. "
                f"Allowed primary inputs are: {sorted(list(ALLOWED_PRIMARY_INPUTS))}"
            )

    # 3. Construct scenario state cleanly without mutating baseline
    base_dict = validated_baseline.to_dict()
    scenario_dict = copy.deepcopy(base_dict)
    scenario_dict.update(scenario_changes)

    # 4. Validate updated scenario state (recomputes types, non-negativity, and integer rules)
    validated_scenario = validate_financial_state(scenario_dict)

    # 5. Execute predictions through single authoritative inference gateway
    baseline_pred = predict_spending(validated_baseline, model_path=model_path)
    scenario_pred = predict_spending(validated_scenario, model_path=model_path)

    # 6. Calculate differences safely
    abs_diff = round(scenario_pred - baseline_pred, 2)

    if abs(baseline_pred) < 1e-9:
        pct_diff = None
    else:
        pct_diff = round(((scenario_pred - baseline_pred) / baseline_pred) * 100.0, 2)

    return {
        "baseline_state": validated_baseline.to_dict(),
        "scenario_state": validated_scenario.to_dict(),
        "baseline_prediction": baseline_pred,
        "scenario_prediction": scenario_pred,
        "absolute_difference": abs_diff,
        "percentage_difference": pct_diff,
    }
