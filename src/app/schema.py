import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd

# Authoritative Feature Order Contract expected by models/catboost_model.joblib
EXACT_14_FEATURE_ORDER: List[str] = [
    "spending_t",
    "spending_t_minus_1",
    "spending_t_minus_2",
    "spending_3m_mean",
    "spending_3m_std",
    "debit_count_t",
    "income_credit_t",
    "ending_balance_t",
    "spending_hh_t",
    "spending_st_t",
    "spending_in_t",
    "spending_lo_t",
    "spending_io_t",
    "spending_other_t",
]


@dataclass
class FinancialState:
    """
    Structured internal financial state representation for an individual account profile.
    
    Primary State Inputs:
    - ending_balance_t: Liquid account balance at month t (CZK)
    - income_credit_t: Cash inflows / deposits at month t (CZK)
    - debit_count_t: Count of debit transactions at month t (integer >= 0)
    
    Category Outflow Inputs:
    - spending_hh_t: Household debit subtotal (CZK)
    - spending_st_t: Statement/fee debit subtotal (CZK)
    - spending_in_t: Insurance debit subtotal (CZK)
    - spending_lo_t: Loan repayment debit subtotal (CZK)
    - spending_io_t: Interest outflow debit subtotal (CZK)
    - spending_other_t: Uncategorized debit subtotal (CZK)
    
    Historical Lag Inputs:
    - spending_t_minus_1: Total debit spending at month t-1 (CZK)
    - spending_t_minus_2: Total debit spending at month t-2 (CZK)
    """
    ending_balance_t: float
    income_credit_t: float
    debit_count_t: int
    spending_hh_t: float = 0.0
    spending_st_t: float = 0.0
    spending_in_t: float = 0.0
    spending_lo_t: float = 0.0
    spending_io_t: float = 0.0
    spending_other_t: float = 0.0
    spending_t_minus_1: float = 0.0
    spending_t_minus_2: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialState":

        """Constructs FinancialState safely from a dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Converts FinancialState to dictionary."""
        return asdict(self)


def validate_financial_state(state: Union[FinancialState, Dict[str, Any]]) -> FinancialState:
    """
    Validates numeric integrity, types, finiteness, and basic domain constraints.
    
    Raises
    ------
    TypeError: If fields are non-numeric.
    ValueError: If required fields are missing, NaN, infinite, or physically invalid.
    """
    if isinstance(state, dict):
        state = FinancialState.from_dict(state)
    elif not isinstance(state, FinancialState):
        raise TypeError(f"Expected FinancialState instance or dict, got {type(state).__name__}")

    state_dict = state.to_dict()

    # 1. Required field completeness check
    required_fields = ["ending_balance_t", "income_credit_t", "debit_count_t"]
    for req in required_fields:
        if state_dict.get(req) is None:
            raise ValueError(f"Required financial state input '{req}' is missing.")

    # 2. Type & Numeric Finiteness Check
    for key, val in state_dict.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(f"Field '{key}' must be numeric (int/float), got {type(val).__name__} = {val}")

        val_float = float(val)
        if math.isnan(val_float):
            raise ValueError(f"Field '{key}' contains NaN (Not a Number).")
        if math.isinf(val_float):
            raise ValueError(f"Field '{key}' contains infinite value ({val}).")

    # 3. Domain & Integer Constraints
    debit_count = state.debit_count_t
    if isinstance(debit_count, float) and not debit_count.is_integer():
        raise ValueError(f"Field 'debit_count_t' must be a whole integer, got {debit_count}")
    if debit_count < 0:
        raise ValueError(f"Field 'debit_count_t' cannot be negative, got {debit_count}")

    if state.income_credit_t < 0:
        raise ValueError(f"Field 'income_credit_t' cannot be negative, got {state.income_credit_t}")

    category_fields = [
        "spending_hh_t", "spending_st_t", "spending_in_t",
        "spending_lo_t", "spending_io_t", "spending_other_t",
        "spending_t_minus_1", "spending_t_minus_2"
    ]
    for cat in category_fields:
        if state_dict[cat] < 0:
            raise ValueError(f"Field '{cat}' cannot be negative, got {state_dict[cat]}")

    return state


def derive_feature_vector(state: FinancialState) -> pd.DataFrame:
    """
    Derives the complete, mathematically consistent 14-feature vector X.
    
    Verified Feature Mathematics:
    1. spending_t = spending_hh_t + spending_st_t + spending_in_t + spending_lo_t + spending_io_t + spending_other_t
    2. spending_3m_mean = (spending_t + spending_t_minus_1 + spending_t_minus_2) / 3.0
    3. spending_3m_std  = np.std([spending_t, spending_t_minus_1, spending_t_minus_2], ddof=0)
    
    Returns
    -------
    pd.DataFrame of shape (1, 14) matching EXACT_14_FEATURE_ORDER.
    """
    # 1. Derive spending_t
    spending_t = float(
        state.spending_hh_t +
        state.spending_st_t +
        state.spending_in_t +
        state.spending_lo_t +
        state.spending_io_t +
        state.spending_other_t
    )

    s_tm1 = float(state.spending_t_minus_1)
    s_tm2 = float(state.spending_t_minus_2)

    # 2. Derive 3-month rolling mean
    spending_3m_mean = float((spending_t + s_tm1 + s_tm2) / 3.0)

    # 3. Derive 3-month rolling std (EXPLICIT ddof=0 matching training feature engineering)
    spending_3m_std = float(np.std([spending_t, s_tm1, s_tm2], ddof=0))

    # Assemble feature mapping
    feature_dict = {
        "spending_t": spending_t,
        "spending_t_minus_1": s_tm1,
        "spending_t_minus_2": s_tm2,
        "spending_3m_mean": spending_3m_mean,
        "spending_3m_std": spending_3m_std,
        "debit_count_t": int(state.debit_count_t),
        "income_credit_t": float(state.income_credit_t),
        "ending_balance_t": float(state.ending_balance_t),
        "spending_hh_t": float(state.spending_hh_t),
        "spending_st_t": float(state.spending_st_t),
        "spending_in_t": float(state.spending_in_t),
        "spending_lo_t": float(state.spending_lo_t),
        "spending_io_t": float(state.spending_io_t),
        "spending_other_t": float(state.spending_other_t),
    }

    # Construct DataFrame with explicit column order
    df_vector = pd.DataFrame([feature_dict])[EXACT_14_FEATURE_ORDER]

    # Verify column order contract
    assert df_vector.columns.tolist() == EXACT_14_FEATURE_ORDER, "Derived feature vector column mismatch"

    return df_vector
