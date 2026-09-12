import sys
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import (
    FinancialState,
    derive_feature_vector,
    predict_spending,
    simulate_scenario,
)


def get_sample_baseline() -> FinancialState:
    """Returns a valid baseline FinancialState for testing."""
    return FinancialState(
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


def test_1_baseline_prediction():
    print("--- Test 1: Baseline Simulation ---")
    baseline = get_sample_baseline()
    res = simulate_scenario(baseline, {})
    assert "baseline_prediction" in res, "Missing baseline_prediction key"
    assert isinstance(res["baseline_prediction"], float), "baseline_prediction must be float"
    assert not math.isnan(res["baseline_prediction"]), "baseline_prediction is NaN"
    print(f"  [PASS] Baseline simulated successfully: {res['baseline_prediction']:.2f} CZK")


def test_2_scenario_prediction():
    print("--- Test 2: Scenario Prediction ---")
    baseline = get_sample_baseline()
    res = simulate_scenario(baseline, {"spending_hh_t": 1500.0})
    assert "scenario_prediction" in res, "Missing scenario_prediction key"
    assert isinstance(res["scenario_prediction"], float), "scenario_prediction must be float"
    assert not math.isnan(res["scenario_prediction"]), "scenario_prediction is NaN"
    print(f"  [PASS] Scenario simulated successfully: {res['scenario_prediction']:.2f} CZK")


def test_3_difference_calculation():
    print("--- Test 3: Absolute Difference Calculation ---")
    baseline = get_sample_baseline()
    res = simulate_scenario(baseline, {"spending_hh_t": 1500.0})
    expected_diff = round(res["scenario_prediction"] - res["baseline_prediction"], 2)
    assert abs(res["absolute_difference"] - expected_diff) < 1e-5, (
        f"Expected absolute_difference {expected_diff}, got {res['absolute_difference']}"
    )
    print(f"  [PASS] Absolute difference verified: {res['absolute_difference']:.2f} CZK")


def test_4_percentage_calculation():
    print("--- Test 4: Percentage Difference Calculation ---")
    baseline = get_sample_baseline()
    res = simulate_scenario(baseline, {"spending_hh_t": 1500.0})
    base_pred = res["baseline_prediction"]
    scen_pred = res["scenario_prediction"]
    expected_pct = round(((scen_pred - base_pred) / base_pred) * 100.0, 2)
    assert abs(res["percentage_difference"] - expected_pct) < 1e-5, (
        f"Expected percentage_difference {expected_pct}%, got {res['percentage_difference']}%"
    )
    print(f"  [PASS] Percentage difference verified: {res['percentage_difference']:.2f}%")


def test_5_zero_baseline_safety():
    print("--- Test 5: Zero Baseline Safety ---")
    baseline = get_sample_baseline()
    with patch("src.app.simulator.predict_spending", side_effect=[0.0, 500.0]):
        res = simulate_scenario(baseline, {"spending_hh_t": 1500.0})
        assert res["percentage_difference"] is None, (
            f"Expected None for zero baseline, got {res['percentage_difference']}"
        )
        assert res["absolute_difference"] == 500.0
    print("  [PASS] Zero baseline prediction handled safely without NaN or Inf (percentage_difference = None).")


def test_6_derived_spending_t():
    print("--- Test 6: Derived spending_t Recalculation ---")
    baseline = get_sample_baseline()
    # Initial categories sum: 1000 + 200 + 300 + 0 + 0 + 500 = 2000.0
    res = simulate_scenario(baseline, {"spending_hh_t": 1200.0})  # +200 increase
    # New sum should be 2200.0
    scen_state = res["scenario_state"]
    X_scen = derive_feature_vector(FinancialState.from_dict(scen_state))
    derived_s_t = X_scen["spending_t"].iloc[0]
    assert derived_s_t == 2200.0, f"Expected spending_t = 2200.0, got {derived_s_t}"
    print(f"  [PASS] Category change correctly derived spending_t = {derived_s_t:.2f} CZK")


def test_7_derived_3m_mean():
    print("--- Test 7: Derived 3-Month Mean Recalculation ---")
    baseline = get_sample_baseline()
    # Baseline spending_t = 2000.0, lag1 = 1500.0, lag2 = 1000.0 -> mean = 1500.0
    res = simulate_scenario(baseline, {"spending_hh_t": 1200.0})  # spending_t becomes 2200.0
    scen_state = res["scenario_state"]
    X_scen = derive_feature_vector(FinancialState.from_dict(scen_state))
    derived_mean = X_scen["spending_3m_mean"].iloc[0]
    expected_mean = (2200.0 + 1500.0 + 1000.0) / 3.0
    assert abs(derived_mean - expected_mean) < 1e-6, f"Expected mean {expected_mean}, got {derived_mean}"
    print(f"  [PASS] Scenario spending correctly derived 3m_mean = {derived_mean:.2f} CZK")


def test_8_derived_3m_std_ddof0():
    print("--- Test 8: Derived 3-Month Std ddof=0 Verification ---")
    baseline = get_sample_baseline()
    # baseline spending_t = 2000.0, lag1 = 1500.0, lag2 = 1000.0
    # Values: [2000, 1500, 1000]
    res = simulate_scenario(baseline, {})
    scen_state = res["scenario_state"]
    X_scen = derive_feature_vector(FinancialState.from_dict(scen_state))
    derived_std = X_scen["spending_3m_std"].iloc[0]
    expected_std_ddof0 = np.std([2000.0, 1500.0, 1000.0], ddof=0)
    expected_std_ddof1 = np.std([2000.0, 1500.0, 1000.0], ddof=1)
    assert abs(derived_std - expected_std_ddof0) < 1e-6, (
        f"ddof=0 mismatch: got {derived_std}, expected {expected_std_ddof0}"
    )
    assert abs(derived_std - expected_std_ddof1) > 1.0, "Derived std matched ddof=1 instead of ddof=0!"
    print(f"  [PASS] Derived 3m_std strictly follows ddof=0 population formula ({derived_std:.4f}).")


def test_9_no_mutation():
    print("--- Test 9: Baseline State Immutability ---")
    baseline = get_sample_baseline()
    original_dict = baseline.to_dict()
    _ = simulate_scenario(baseline, {"spending_hh_t": 5000.0, "income_credit_t": 50000.0})
    current_dict = baseline.to_dict()
    assert original_dict == current_dict, "Baseline state object was mutated during simulation!"
    print("  [PASS] Baseline state remained 100% unmutated after simulation execution.")


def test_10_multiple_scenarios():
    print("--- Test 10: Multiple Independent Scenarios ---")
    baseline = get_sample_baseline()
    res1 = simulate_scenario(baseline, {"spending_hh_t": 3000.0})
    res2 = simulate_scenario(baseline, {"spending_hh_t": 100.0})

    assert res1["scenario_state"]["spending_hh_t"] == 3000.0
    assert res2["scenario_state"]["spending_hh_t"] == 100.0
    assert res1["scenario_prediction"] != res2["scenario_prediction"]
    assert baseline.spending_hh_t == 1000.0
    print("  [PASS] Multiple scenarios executed independently against single baseline without contamination.")


def test_11_invalid_scenario_field():
    print("--- Test 11: Invalid Scenario Field Rejection ---")
    baseline = get_sample_baseline()
    try:
        simulate_scenario(baseline, {"unknown_field_x": 500.0})
        assert False, "Failed to reject unknown scenario field"
    except ValueError as e:
        assert "Invalid or unsupported scenario field" in str(e)
    print("  [PASS] Unknown scenario field rejected with explicit ValueError.")


def test_12_derived_feature_override():
    print("--- Test 12: Derived Feature Override Prevention ---")
    baseline = get_sample_baseline()
    for derived_field in ["spending_t", "spending_3m_mean", "spending_3m_std"]:
        try:
            simulate_scenario(baseline, {derived_field: 2500.0})
            assert False, f"Failed to reject direct override to {derived_field}"
        except ValueError as e:
            assert "Direct override of derived feature" in str(e)
    print("  [PASS] Direct override attempts for spending_t, spending_3m_mean, and spending_3m_std rejected.")


def test_13_inference_gateway_usage():
    print("--- Test 13: Inference Gateway Re-use Verification ---")
    baseline = get_sample_baseline()
    with patch("src.app.simulator.predict_spending", return_value=1234.56) as mock_predict:
        res = simulate_scenario(baseline, {"spending_hh_t": 1500.0})
        assert mock_predict.call_count == 2, (
            f"Expected predict_spending to be called twice (baseline & scenario), got {mock_predict.call_count}"
        )
        assert res["baseline_prediction"] == 1234.56
        assert res["scenario_prediction"] == 1234.56
    print("  [PASS] Simulator verified to route predictions exclusively through predict_spending() gateway.")


def run_all_tests():
    print("=" * 80)
    print("PERSONAL FINANCIAL DIGITAL TWIN — STEP 3 WHAT-IF SIMULATOR TEST SUITE")
    print("=" * 80 + "\n")

    test_1_baseline_prediction()
    test_2_scenario_prediction()
    test_3_difference_calculation()
    test_4_percentage_calculation()
    test_5_zero_baseline_safety()
    test_6_derived_spending_t()
    test_7_derived_3m_mean()
    test_8_derived_3m_std_ddof0()
    test_9_no_mutation()
    test_10_multiple_scenarios()
    test_11_invalid_scenario_field()
    test_12_derived_feature_override()
    test_13_inference_gateway_usage()

    print("\n" + "=" * 80)
    print("ALL 13 STEP 3 WHAT-IF SIMULATOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
