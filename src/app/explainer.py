"""
Personal Financial Digital Twin — Individual Real-Time SHAP Explanation Service

Provides single-instance local feature attributions using analytical TreeSHAP.

Key Principles:
1. Re-uses validate_financial_state and derive_feature_vector from src.app.schema.
2. Re-uses predict_spending() and get_model() from src.app.inference.
3. Caches TreeExplainer via singleton pattern (_EXPLAINER_CACHE) for fast execution.
4. Preserves EXACT_14_FEATURE_ORDER and exact feature derivation contracts.
5. Evaluates model attribution decomposition (non-causal interpretation).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import shap

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.schema import (
    EXACT_14_FEATURE_ORDER,
    FinancialState,
    validate_financial_state,
    derive_feature_vector,
)
from src.app.inference import get_model, predict_spending

# Global TreeExplainer cache (Singleton Pattern)
_EXPLAINER_CACHE: Optional[shap.TreeExplainer] = None


def get_explainer(model_path: Optional[Union[str, Path]] = None) -> shap.TreeExplainer:
    """
    Loads CatBoost model and returns a cached singleton TreeExplainer instance.

    CatBoost models provide exact tree structures for analytical TreeSHAP calculation.

    Returns
    -------
    shap.TreeExplainer
        Cached TreeExplainer instance.
    """
    global _EXPLAINER_CACHE
    if _EXPLAINER_CACHE is not None and model_path is None:
        return _EXPLAINER_CACHE

    model = get_model(model_path)
    explainer = shap.TreeExplainer(model)

    if model_path is None or Path(model_path) == PROJECT_ROOT / "models" / "catboost_model.joblib":
        _EXPLAINER_CACHE = explainer

    return explainer


def explain_prediction(
    state: Union[FinancialState, Dict[str, Any], pd.DataFrame],
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Computes real-time TreeSHAP feature attributions for a single financial state instance.

    Process:
    1. Validates input state structure and derives exact 14-feature vector X.
    2. Obtains authoritative point forecast via predict_spending() gateway.
    3. Accesses frozen model and cached TreeExplainer singleton.
    4. Computes single-instance Shapley values and base value E[f(X)].
    5. Formats structured feature-level attributions following EXACT_14_FEATURE_ORDER.
    6. Verifies exact SHAP additivity: base_value + sum(shap_values) == raw_prediction.

    Parameters
    ----------
    state : FinancialState, Dict, or pd.DataFrame
        Input financial state.
    model_path : Path or str, optional
        Path to model artifact (defaults to models/catboost_model.joblib).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - prediction: Forecast spending from predict_spending() (CZK, 2 decimals)
        - raw_prediction: Unrounded raw CatBoost model forecast (CZK)
        - base_value: Expected target value E[f(X)] (CZK)
        - features: List of 14 dicts, each with "feature", "value", and "shap_value"
        - raw_shap_values: List of unrounded float SHAP values
        - reconstructed_prediction: base_value + sum(shap_values) (CZK)
        - additivity_delta: abs(reconstructed_prediction - raw_prediction)
    """
    # 1. Validate state and derive exact 14-feature vector
    if isinstance(state, pd.DataFrame):
        if state.shape[0] != 1:
            raise ValueError(f"Explainer gateway expects single instance (1 row), got shape {state.shape}")
        if list(state.columns) == EXACT_14_FEATURE_ORDER:
            X = state[EXACT_14_FEATURE_ORDER]
        else:
            row_dict = state.iloc[0].to_dict()
            valid_state = validate_financial_state(row_dict)
            X = derive_feature_vector(valid_state)
    else:
        valid_state = validate_financial_state(state)
        X = derive_feature_vector(valid_state)

    # Assert column order contract
    assert X.columns.tolist() == EXACT_14_FEATURE_ORDER, "X columns mismatch EXACT_14_FEATURE_ORDER"

    # 2. Obtain prediction from single authoritative prediction gateway
    prediction = predict_spending(X, model_path=model_path)

    # 3. Access model & cached explainer singleton
    model = get_model(model_path)
    explainer = get_explainer(model_path)

    raw_pred = float(model.predict(X)[0])

    # 4. Compute single-instance TreeSHAP explanation
    explanation_obj = explainer(X)

    # Extract base value E[f(X)]
    if hasattr(explanation_obj, "base_values") and explanation_obj.base_values is not None:
        bv = explanation_obj.base_values
        base_value = float(bv[0]) if isinstance(bv, (np.ndarray, list)) else float(bv)
    else:
        base_value = float(explainer.expected_value)

    # Extract SHAP values array for single instance
    if isinstance(explanation_obj.values, np.ndarray):
        shap_row = explanation_obj.values[0]
    else:
        shap_row = np.array(explanation_obj.values)[0]

    # 5. Format feature contributions in EXACT_14_FEATURE_ORDER
    features_list = []
    total_shap_sum = 0.0

    for idx, feature_name in enumerate(EXACT_14_FEATURE_ORDER):
        val = float(X[feature_name].iloc[0])
        s_val = float(shap_row[idx])
        total_shap_sum += s_val

        # Preserve integer representation for counts where applicable
        formatted_val = int(val) if feature_name == "debit_count_t" and val.is_integer() else round(val, 2)

        features_list.append({
            "feature": feature_name,
            "value": formatted_val,
            "shap_value": round(s_val, 4),
        })

    reconstructed_pred = base_value + total_shap_sum
    additivity_delta = abs(reconstructed_pred - raw_pred)

    return {
        "prediction": prediction,
        "raw_prediction": round(raw_pred, 4),
        "base_value": round(base_value, 4),
        "features": features_list,
        "raw_shap_values": [float(s) for s in shap_row],
        "reconstructed_prediction": round(reconstructed_pred, 4),
        "additivity_delta": round(additivity_delta, 6),
    }
