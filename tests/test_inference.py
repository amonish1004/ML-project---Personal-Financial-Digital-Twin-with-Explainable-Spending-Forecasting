import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import (
    EXACT_14_FEATURE_ORDER,
    FinancialState,
    validate_financial_state,
    derive_feature_vector,
    get_model,
    predict_spending,
)


def test_1_model_loading():
    print("--- Test 1: Model Loading ---")
    model = get_model()
    assert model is not None, "Model failed to load"
    assert hasattr(model, "predict"), "Loaded object is not a valid model"
    print("  [PASS] models/catboost_model.joblib loaded successfully.")


def test_2_feature_contract():
    print("--- Test 2: Feature Contract Verification ---")
    state = FinancialState(ending_balance_t=10000.0, income_credit_t=15000.0, debit_count_t=5)
    X = derive_feature_vector(state)
    assert X.shape == (1, 14), f"Expected shape (1, 14), got {X.shape}"
    assert X.columns.tolist() == EXACT_14_FEATURE_ORDER, "Column names or order mismatch"
    print("  [PASS] Feature vector contains exactly 14 features in exact order.")


def test_3_feature_derivation():
    print("--- Test 3: Feature Derivation & ddof=0 Verification ---")
    # Deterministic Test Inputs:
    # spending_hh_t = 1000, spending_st_t = 200, spending_in_t = 300, lo=0, io=0, other=500
    # spending_t = 1000 + 200 + 300 + 0 + 0 + 500 = 2000.0
    # spending_t_minus_1 = 1500.0
    # spending_t_minus_2 = 1000.0
    # Values array: [2000.0, 1500.0, 1000.0]
    # spending_3m_mean = (2000 + 1500 + 1000) / 3 = 1500.0
    # spending_3m_std with ddof=0: sqrt(((2000-1500)^2 + (1500-1500)^2 + (1000-1500)^2)/3)
    #                             = sqrt((250000 + 0 + 250000)/3) = sqrt(500000/3) = 408.24829046...
    # (Note: ddof=1 would give sqrt(500000/2) = 500.0)

    state = FinancialState(
        ending_balance_t=12000.0,
        income_credit_t=20000.0,
        debit_count_t=8,
        spending_hh_t=1000.0,
        spending_st_t=200.0,
        spending_in_t=300.0,
        spending_lo_t=0.0,
        spending_io_t=0.0,
        spending_other_t=500.0,
        spending_t_minus_1=1500.0,
        spending_t_minus_2=1000.0,
    )

    X = derive_feature_vector(state)

    derived_s_t = X["spending_t"].iloc[0]
    derived_mean = X["spending_3m_mean"].iloc[0]
    derived_std = X["spending_3m_std"].iloc[0]

    assert derived_s_t == 2000.0, f"Expected spending_t = 2000.0, got {derived_s_t}"
    assert derived_mean == 1500.0, f"Expected 3m_mean = 1500.0, got {derived_mean}"

    expected_std_ddof0 = np.std([2000.0, 1500.0, 1000.0], ddof=0)
    expected_std_ddof1 = np.std([2000.0, 1500.0, 1000.0], ddof=1)

    assert abs(derived_std - expected_std_ddof0) < 1e-6, f"ddof=0 mismatch: got {derived_std}, expected {expected_std_ddof0}"
    assert abs(derived_std - expected_std_ddof1) > 1.0, "Derived std incorrectly matched ddof=1!"

    print(f"  [PASS] Feature mathematics verified. spending_t={derived_s_t}, mean={derived_mean}, std(ddof=0)={derived_std:.4f}")


def test_4_prediction():
    print("--- Test 4: Prediction Execution & Finiteness ---")
    state = FinancialState(ending_balance_t=15000.0, income_credit_t=25000.0, debit_count_t=10)
    pred = predict_spending(state)
    assert isinstance(pred, float), f"Prediction output must be float, got {type(pred)}"
    assert not math.isnan(pred), "Prediction is NaN"
    assert not math.isinf(pred), "Prediction is Infinite"
    print(f"  [PASS] Prediction succeeded. Output = {pred:.2f} CZK")


def test_5_invalid_input_rejection():
    print("--- Test 5: Invalid Input Rejection ---")
    
    # NaN check
    try:
        predict_spending({"ending_balance_t": float("nan"), "income_credit_t": 10000.0, "debit_count_t": 5})
        assert False, "Failed to reject NaN"
    except ValueError:
        pass

    # Infinity check
    try:
        predict_spending({"ending_balance_t": float("inf"), "income_credit_t": 10000.0, "debit_count_t": 5})
        assert False, "Failed to reject Infinity"
    except ValueError:
        pass

    # Non-numeric check
    try:
        predict_spending({"ending_balance_t": "invalid_string", "income_credit_t": 10000.0, "debit_count_t": 5})
        assert False, "Failed to reject string"
    except TypeError:
        pass

    # Negative count check
    try:
        predict_spending({"ending_balance_t": 10000.0, "income_credit_t": 10000.0, "debit_count_t": -3})
        assert False, "Failed to reject negative debit count"
    except ValueError:
        pass

    print("  [PASS] Malformed and out-of-bounds inputs properly rejected with explicit exceptions.")


def test_6_repeatability():
    print("--- Test 6: Repeatability & Determinism ---")
    state = FinancialState(
        ending_balance_t=8500.0,
        income_credit_t=14000.0,
        debit_count_t=6,
        spending_hh_t=3000.0,
        spending_t_minus_1=2800.0,
        spending_t_minus_2=3100.0,
    )

    pred1 = predict_spending(state)
    pred2 = predict_spending(state)
    pred3 = predict_spending(state)

    assert pred1 == pred2 == pred3, f"Inconsistent predictions across calls: {pred1}, {pred2}, {pred3}"
    print(f"  [PASS] 100% deterministic output across repeated calls ({pred1:.2f} CZK).")


def test_7_direct_model_consistency_check():
    print("--- Test 7: Direct CatBoost Prediction Consistency Check ---")
    dataset_csv = PROJECT_ROOT / "data" / "processed" / "supervised_spending_dataset.csv"
    assert dataset_csv.exists(), f"Dataset CSV missing at: {dataset_csv}"

    df = pd.read_csv(dataset_csv, nrows=5)
    sample_row = df.iloc[0]

    # Reconstruct FinancialState from raw dataset row
    state = FinancialState(
        ending_balance_t=float(sample_row["ending_balance_t"]),
        income_credit_t=float(sample_row["income_credit_t"]),
        debit_count_t=int(sample_row["debit_count_t"]),
        spending_hh_t=float(sample_row["spending_hh_t"]),
        spending_st_t=float(sample_row["spending_st_t"]),
        spending_in_t=float(sample_row["spending_in_t"]),
        spending_lo_t=float(sample_row["spending_lo_t"]),
        spending_io_t=float(sample_row["spending_io_t"]),
        spending_other_t=float(sample_row["spending_other_t"]),
        spending_t_minus_1=float(sample_row["spending_t_minus_1"]),
        spending_t_minus_2=float(sample_row["spending_t_minus_2"]),
    )

    # 1. Prediction via reusable inference gateway
    gateway_pred = predict_spending(state)

    # 2. Direct prediction via model.predict(raw_row[EXACT_14_FEATURE_ORDER])
    model = get_model()
    raw_X = pd.DataFrame([sample_row[EXACT_14_FEATURE_ORDER].to_dict()])[EXACT_14_FEATURE_ORDER]
    direct_pred = float(model.predict(raw_X)[0])
    direct_pred_rounded = round(direct_pred, 2)

    diff = abs(gateway_pred - direct_pred_rounded)
    assert diff < 1e-3, f"Inference gateway ({gateway_pred}) does not match direct CatBoost prediction ({direct_pred_rounded})"

    print(f"  [PASS] Inference Gateway ({gateway_pred:.2f} CZK) == Direct CatBoost ({direct_pred_rounded:.2f} CZK). Difference = {diff:.6f}")


def run_all_tests():
    print("=" * 80)
    print("PERSONAL FINANCIAL DIGITAL TWIN — STEP 2 INFERENCE LAYER TEST SUITE")
    print("=" * 80 + "\n")

    test_1_model_loading()
    test_2_feature_contract()
    test_3_feature_derivation()
    test_4_prediction()
    test_5_invalid_input_rejection()
    test_6_repeatability()
    test_7_direct_model_consistency_check()

    print("\n" + "=" * 80)
    print("ALL 7 STEP 2 INFERENCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
