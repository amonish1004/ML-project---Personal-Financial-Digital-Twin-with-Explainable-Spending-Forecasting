import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

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

# Global model cache (Singleton Pattern)
_MODEL_CACHE: Optional[CatBoostRegressor] = None


def get_model(model_path: Optional[Union[str, Path]] = None) -> CatBoostRegressor:
    """
    Loads and caches the frozen CatBoost model artifact (models/catboost_model.joblib).
    
    Verifies that the model artifact matches EXACT_14_FEATURE_ORDER upon loading.
    
    Returns
    -------
    CatBoostRegressor
        Loaded frozen model.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and model_path is None:
        return _MODEL_CACHE

    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "catboost_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model artifact not found at: {model_path}")

    model = joblib.load(model_path)

    # Model feature schema assertion
    if hasattr(model, "feature_names_") and model.feature_names_ is not None:
        if list(model.feature_names_) != EXACT_14_FEATURE_ORDER:
            raise ValueError(
                f"Model feature contract mismatch!\n"
                f"Expected: {EXACT_14_FEATURE_ORDER}\n"
                f"Model has: {list(model.feature_names_)}"
            )

    if model_path == PROJECT_ROOT / "models" / "catboost_model.joblib":
        _MODEL_CACHE = model

    return model


def predict_spending(
    state: Union[FinancialState, Dict[str, Any], pd.DataFrame],
    model_path: Optional[Union[str, Path]] = None
) -> float:
    """
    Primary Reusable Prediction Gateway Function.
    
    SINGLE SOURCE OF TRUTH FOR MODEL PREDICTION.
    
    Process:
    1. Validates input state structure, data types, and finiteness.
    2. Derives 14-feature vector X with exact ordering and ddof=0 mathematics.
    3. Loads frozen CatBoost model.
    4. Executes model inference.
    
    Parameters
    ----------
    state : FinancialState, Dict, or pd.DataFrame
        Input financial state.
    model_path : Path or str, optional
        Path to model artifact (defaults to models/catboost_model.joblib).
        
    Returns
    -------
    float
        Predicted next-month spending in CZK (rounded to 2 decimal places).
    """
    # Handle DataFrame input if passed directly
    if isinstance(state, pd.DataFrame):
        if state.shape[0] != 1:
            raise ValueError(f"Inference gateway expects single instance (1 row), got shape {state.shape}")
        if list(state.columns) == EXACT_14_FEATURE_ORDER:
            X = state[EXACT_14_FEATURE_ORDER]
        else:
            # Convert row to dict and validate/derive
            row_dict = state.iloc[0].to_dict()
            valid_state = validate_financial_state(row_dict)
            X = derive_feature_vector(valid_state)
    else:
        # Validate input financial state
        valid_state = validate_financial_state(state)
        # Derive exact 14-feature vector
        X = derive_feature_vector(valid_state)

    # Safety check on feature order before passing to CatBoost
    assert X.columns.tolist() == EXACT_14_FEATURE_ORDER, "X columns do not match EXACT_14_FEATURE_ORDER"

    # Access frozen CatBoost model
    model = get_model(model_path)

    # Perform inference
    pred = model.predict(X)
    pred_val = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)

    return round(pred_val, 2)
