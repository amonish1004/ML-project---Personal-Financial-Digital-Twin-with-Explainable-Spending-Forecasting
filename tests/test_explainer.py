import sys
import math
import json
import time
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
    derive_feature_vector,
    predict_spending,
    get_model,
    get_explainer,
    explain_prediction,
)


def get_sample_state() -> FinancialState:
    """Returns a valid deterministic baseline FinancialState for testing."""
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


def test_1_explainer_initialization():
    print("--- Test 1: TreeExplainer Initialization ---")
    explainer = get_explainer()
    assert explainer is not None, "Explainer failed to initialize"
    assert hasattr(explainer, "__call__") or hasattr(explainer, "shap_values"), "Invalid explainer object"
    print("  [PASS] shap.TreeExplainer initialized successfully from frozen CatBoost model.")


def test_2_single_instance_explanation():
    print("--- Test 2: Single-Instance Explanation Execution ---")
    state = get_sample_state()
    exp = explain_prediction(state)
    assert isinstance(exp, dict), f"Expected dict result, got {type(exp)}"
    assert "prediction" in exp and "base_value" in exp and "features" in exp
    print(f"  [PASS] Single-instance explanation executed: prediction = {exp['prediction']} CZK, base_value = {exp['base_value']} CZK")


def test_3_exact_14_features_returned():
    print("--- Test 3: Feature Contract & Sequence Verification ---")
    state = get_sample_state()
    exp = explain_prediction(state)
    features = exp["features"]
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"
    returned_names = [f["feature"] for f in features]
    assert returned_names == EXACT_14_FEATURE_ORDER, f"Feature sequence mismatch!\nExpected: {EXACT_14_FEATURE_ORDER}\nGot: {returned_names}"
    print("  [PASS] Explanation returned exactly 14 features in EXACT_14_FEATURE_ORDER.")


def test_4_prediction_gateway_consistency():
    print("--- Test 4: Inference Gateway Consistency ---")
    state = get_sample_state()
    gateway_pred = predict_spending(state)
    exp = explain_prediction(state)
    assert exp["prediction"] == gateway_pred, (
        f"Explainer prediction ({exp['prediction']}) does not match gateway prediction ({gateway_pred})"
    )
    print(f"  [PASS] Explainer prediction ({exp['prediction']} CZK) matches predict_spending() gateway ({gateway_pred} CZK).")


def test_5_finiteness_and_types():
    print("--- Test 5: Numerical Finiteness & Type Integrity ---")
    state = get_sample_state()
    exp = explain_prediction(state)

    for field in ["prediction", "raw_prediction", "base_value", "reconstructed_prediction", "additivity_delta"]:
        val = exp[field]
        assert isinstance(val, (int, float)), f"Field '{field}' is non-numeric: {type(val)}"
        assert not math.isnan(val), f"Field '{field}' is NaN"
        assert not math.isinf(val), f"Field '{field}' is Infinite"

    for feat in exp["features"]:
        for k in ["value", "shap_value"]:
            v = feat[k]
            assert isinstance(v, (int, float, np.number)), f"Feature '{feat['feature']}' field '{k}' is non-numeric"
            assert not math.isnan(float(v)), f"Feature '{feat['feature']}' field '{k}' is NaN"
            assert not math.isinf(float(v)), f"Feature '{feat['feature']}' field '{k}' is Infinite"

    print("  [PASS] All output numerical fields are finite, valid numbers.")


def test_6_primary_feature_values_match():
    print("--- Test 6: Primary Input Feature Values Fidelity ---")
    state = get_sample_state()
    exp = explain_prediction(state)
    feat_map = {f["feature"]: f["value"] for f in exp["features"]}

    assert feat_map["ending_balance_t"] == 12000.0
    assert feat_map["income_credit_t"] == 20000.0
    assert feat_map["debit_count_t"] == 8
    assert feat_map["spending_hh_t"] == 1000.0
    assert feat_map["spending_st_t"] == 200.0
    assert feat_map["spending_in_t"] == 300.0
    assert feat_map["spending_lo_t"] == 0.0
    assert feat_map["spending_io_t"] == 0.0
    assert feat_map["spending_other_t"] == 500.0
    assert feat_map["spending_t_minus_1"] == 1500.0
    assert feat_map["spending_t_minus_2"] == 1000.0

    print("  [PASS] Primary feature values in explanation match input state exactly.")


def test_7_derived_spending_t():
    print("--- Test 7: Derived spending_t Recalculation in Explanation ---")
    state = get_sample_state()
    # 1000 + 200 + 300 + 0 + 0 + 500 = 2000.0
    exp = explain_prediction(state)
    feat_map = {f["feature"]: f["value"] for f in exp["features"]}
    assert feat_map["spending_t"] == 2000.0, f"Expected spending_t = 2000.0, got {feat_map['spending_t']}"
    print(f"  [PASS] Derived spending_t correctly calculated as {feat_map['spending_t']} CZK.")


def test_8_derived_3m_std_ddof0():
    print("--- Test 8: Derived 3-Month Std ddof=0 Verification in Explanation ---")
    state = get_sample_state()
    # spending_t = 2000, lag1 = 1500, lag2 = 1000
    exp = explain_prediction(state)
    feat_map = {f["feature"]: f["value"] for f in exp["features"]}
    expected_std_ddof0 = round(float(np.std([2000.0, 1500.0, 1000.0], ddof=0)), 2)
    expected_std_ddof1 = round(float(np.std([2000.0, 1500.0, 1000.0], ddof=1)), 2)

    assert feat_map["spending_3m_std"] == expected_std_ddof0, (
        f"Expected spending_3m_std {expected_std_ddof0}, got {feat_map['spending_3m_std']}"
    )
    assert feat_map["spending_3m_std"] != expected_std_ddof1
    print(f"  [PASS] Derived spending_3m_std in explanation uses ddof=0 ({feat_map['spending_3m_std']}).")


def test_9_shap_additivity():
    print("--- Test 9: TreeSHAP Additivity / Efficiency Verification ---")
    state = get_sample_state()
    exp = explain_prediction(state)
    bv = exp["base_value"]
    raw_shap_sum = sum(exp["raw_shap_values"])
    reconstructed_raw = bv + raw_shap_sum
    raw_pred = exp["raw_prediction"]

    delta = abs(reconstructed_raw - raw_pred)
    assert delta < 1e-4, f"Additivity delta too large: |{reconstructed_raw} - {raw_pred}| = {delta}"
    print(f"  [PASS] Full-precision TreeSHAP additivity verified: base ({bv:.4f}) + sum(raw_SHAP) ({raw_shap_sum:.4f}) == raw ({raw_pred:.4f}). Delta = {delta:.6f}")



def test_10_invalid_input_rejection():
    print("--- Test 10: Invalid Input Rejection ---")
    try:
        explain_prediction({"ending_balance_t": float("nan"), "income_credit_t": 10000.0, "debit_count_t": 5})
        assert False, "Failed to reject NaN"
    except ValueError:
        pass

    try:
        explain_prediction({"ending_balance_t": 10000.0, "income_credit_t": 10000.0, "debit_count_t": -1})
        assert False, "Failed to reject negative debit count"
    except ValueError:
        pass

    print("  [PASS] Invalid and out-of-bounds inputs properly rejected before SHAP calculation.")


def test_11_explainer_singleton_caching():
    print("--- Test 11: Singleton Explainer Caching & Measured Execution Time ---")
    state = get_sample_state()

    # First call (includes explainer initialization)
    t0 = time.perf_counter()
    exp1 = explain_prediction(state)
    t1 = time.perf_counter()
    init_time_ms = (t1 - t0) * 1000.0

    # Second call (uses cached singleton)
    t2 = time.perf_counter()
    exp2 = explain_prediction(state)
    t3 = time.perf_counter()
    cached_time_ms = (t3 - t2) * 1000.0

    explainer1 = get_explainer()
    explainer2 = get_explainer()
    assert explainer1 is explainer2, "get_explainer() did not return cached singleton instance!"

    print(f"  [PASS] Singleton explainer cached cleanly.")
    print(f"         Initial call latency : {init_time_ms:.2f} ms")
    print(f"         Cached call latency  : {cached_time_ms:.2f} ms")


def test_12_repeatability_determinism():
    print("--- Test 12: Repeatability & Determinism ---")
    state = get_sample_state()
    exp1 = explain_prediction(state)
    exp2 = explain_prediction(state)

    assert exp1["prediction"] == exp2["prediction"]
    assert exp1["base_value"] == exp2["base_value"]
    for f1, f2 in zip(exp1["features"], exp2["features"]):
        assert f1["feature"] == f2["feature"]
        assert f1["value"] == f2["value"]
        assert f1["shap_value"] == f2["shap_value"]

    print("  [PASS] 100% deterministic output across repeated calls.")


def test_13_frozen_model_preservation():
    print("--- Test 13: Frozen CatBoost Model Preservation ---")
    model = get_model()
    state = get_sample_state()
    X = derive_feature_vector(state)
    pred_before = float(model.predict(X)[0])

    _ = explain_prediction(state)

    pred_after = float(model.predict(X)[0])
    assert pred_before == pred_after, "Model state or weights changed after explain_prediction() call!"
    print("  [PASS] Frozen CatBoost model weights and prediction output remained completely unmutated.")


def test_14_serializable_structure():
    print("--- Test 14: JSON Serialization Compatibility ---")
    state = get_sample_state()
    exp = explain_prediction(state)
    json_str = json.dumps(exp, indent=2)
    assert len(json_str) > 0
    reloaded = json.loads(json_str)
    assert reloaded["prediction"] == exp["prediction"]
    print("  [PASS] Explanation dictionary is 100% JSON-serializable for REST API and frontend consumption.")


def run_all_tests():
    print("=" * 80)
    print("PERSONAL FINANCIAL DIGITAL TWIN — STEP 4 SHAP EXPLAINER TEST SUITE")
    print("=" * 80 + "\n")

    test_1_explainer_initialization()
    test_2_single_instance_explanation()
    test_3_exact_14_features_returned()
    test_4_prediction_gateway_consistency()
    test_5_finiteness_and_types()
    test_6_primary_feature_values_match()
    test_7_derived_spending_t()
    test_8_derived_3m_std_ddof0()
    test_9_shap_additivity()
    test_10_invalid_input_rejection()
    test_11_explainer_singleton_caching()
    test_12_repeatability_determinism()
    test_13_frozen_model_preservation()
    test_14_serializable_structure()

    print("\n" + "=" * 80)
    print("ALL 14 STEP 4 SHAP EXPLAINER TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
