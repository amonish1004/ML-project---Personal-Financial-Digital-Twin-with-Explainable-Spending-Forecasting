"""
Personal Financial Digital Twin — End-to-End System Testing & Validation (Dimension 13)

Test Suite validating cross-layer system behavior from user-facing presentation layer
and REST API gateway down to application services (D7/D8/D9) and frozen CatBoost model.

Coverage:
1. System Health & Artifact Availability (GET /health).
2. Authoritative Schema & Feature Contract Metadata (GET /api/schema).
3. End-to-End Baseline Spending Forecast Workflow (HTTP -> API -> D7 -> CatBoost).
4. End-to-End Counterfactual Scenario Simulation Workflow (HTTP -> API -> D8 -> Re-derivation).
5. End-to-End TreeSHAP Explanation & Mathematical Additivity (HTTP -> API -> D9 -> TreeExplainer).
6. Frontend UI Layout, Asset Serving & API Contract Integrity (GET /, /static/*).
7. System Error Handling & Boundary Guardrails (Input validation, zero-division, override rejection).
8. Frozen Model & Feature Derivation Mathematical Fidelity (ddof=0 std, 14-feature sequence).
"""

import sys
import numpy as np
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.api.server import app
from src.app.schema import EXACT_14_FEATURE_ORDER, FinancialState, derive_feature_vector
from src.app.inference import predict_spending, get_model
from src.app.simulator import simulate_scenario, ALLOWED_PRIMARY_INPUTS, DERIVED_FEATURES
from src.app.explainer import explain_prediction

client = TestClient(app)

# Authoritative Test Baseline Profile
TEST_BASELINE_STATE = {
    "ending_balance_t": 18500.0,
    "income_credit_t": 32000.0,
    "debit_count_t": 15,
    "spending_hh_t": 4500.0,
    "spending_st_t": 350.0,
    "spending_in_t": 1500.0,
    "spending_lo_t": 2500.0,
    "spending_io_t": 150.0,
    "spending_other_t": 1800.0,
    "spending_t_minus_1": 9200.0,
    "spending_t_minus_2": 8800.0,
}


def test_e2e_health_and_system_readiness():
    """Verify system health, API status, and frozen model availability via GET /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dimension"] == "D10"
    assert data["model_artifact_exists"] is True
    assert "models/catboost_model.joblib" in data["model_artifact"]

    # Verify frozen CatBoost model loads cleanly
    model = get_model()
    assert model is not None
    assert hasattr(model, "predict")


def test_e2e_schema_contract_metadata():
    """Verify system metadata and authoritative 14-feature contract via GET /api/schema."""
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()

    assert data["exact_14_feature_order"] == EXACT_14_FEATURE_ORDER
    assert len(data["exact_14_feature_order"]) == 14
    assert sorted(data["allowed_primary_inputs"]) == sorted(list(ALLOWED_PRIMARY_INPUTS))
    assert sorted(data["derived_features"]) == sorted(list(DERIVED_FEATURES))
    assert "ddof=0" in data["derived_feature_rules"]["spending_3m_std"]


def test_e2e_baseline_predict_workflow_parity():
    """Verify complete baseline prediction workflow from HTTP request to model output parity."""
    # 1. HTTP API Call
    response = client.post("/api/predict", json=TEST_BASELINE_STATE)
    assert response.status_code == 200
    api_result = response.json()

    assert "prediction" in api_result
    assert api_result["currency"] == "CZK"
    assert isinstance(api_result["prediction"], float)
    assert api_result["prediction"] > 0.0

    # 2. Direct D7 Inference Gateway Call
    gateway_result = predict_spending(TEST_BASELINE_STATE)
    assert api_result["prediction"] == gateway_result

    # 3. Direct CatBoost Model Parity Check (predict_spending rounds output to 2 decimal places)
    state_obj = FinancialState.from_dict(TEST_BASELINE_STATE)
    feature_df = derive_feature_vector(state_obj)
    model = get_model()
    direct_catboost_pred = float(model.predict(feature_df)[0])
    assert abs(api_result["prediction"] - direct_catboost_pred) < 0.01
    assert api_result["prediction"] == round(direct_catboost_pred, 2)


def test_e2e_counterfactual_simulation_workflow():
    """Verify end-to-end What-If scenario simulation, derived feature recalculation, and deltas."""
    # Baseline Prediction
    res_base = client.post("/api/predict", json=TEST_BASELINE_STATE)
    base_pred = res_base.json()["prediction"]

    # Scenario Simulation (+5,000 CZK Ending Balance, +3,000 CZK Income)
    scenario_payload = {
        "baseline_state": TEST_BASELINE_STATE,
        "scenario_changes": {
            "ending_balance_t": 23500.0,
            "income_credit_t": 35000.0,
            "spending_hh_t": 6500.0,  # +2000 increase in category spending
        }
    }
    res_sim = client.post("/api/simulate", json=scenario_payload)
    assert res_sim.status_code == 200
    sim_data = res_sim.json()

    # Validate predictions & delta calculations
    assert sim_data["baseline_prediction"] == base_pred
    scen_pred = sim_data["scenario_prediction"]
    expected_delta = round(scen_pred - base_pred, 2)
    assert sim_data["absolute_difference"] == expected_delta

    expected_pct = round((expected_delta / base_pred) * 100.0, 2)
    assert sim_data["percentage_difference"] == expected_pct

    # Validate Primary Input Changes
    assert sim_data["scenario_state"]["spending_hh_t"] == 6500.0
    assert sim_data["baseline_state"]["spending_hh_t"] == 4500.0

    # Validate Derived Feature Recalculation (spending_t increased by 2000)
    X_base = derive_feature_vector(FinancialState.from_dict(sim_data["baseline_state"]))
    X_scen = derive_feature_vector(FinancialState.from_dict(sim_data["scenario_state"]))

    expected_base_spending_t = 4500.0 + 350.0 + 1500.0 + 2500.0 + 150.0 + 1800.0  # 10800.0
    expected_scen_spending_t = 6500.0 + 350.0 + 1500.0 + 2500.0 + 150.0 + 1800.0  # 12800.0

    assert X_base["spending_t"].iloc[0] == expected_base_spending_t
    assert X_scen["spending_t"].iloc[0] == expected_scen_spending_t
    assert X_scen["spending_t"].iloc[0] - X_base["spending_t"].iloc[0] == 2000.0


def test_e2e_treeshap_explanation_workflow_additivity():
    """Verify end-to-end TreeSHAP attribution generation and exact additivity property."""
    response = client.post("/api/explain", json=TEST_BASELINE_STATE)
    assert response.status_code == 200
    exp_data = response.json()

    assert "prediction" in exp_data
    assert "base_value" in exp_data
    assert "features" in exp_data
    assert len(exp_data["features"]) == 14

    # Verify exact feature order
    feature_names = [f["feature"] for f in exp_data["features"]]
    assert feature_names == EXACT_14_FEATURE_ORDER

    # Verify exact SHAP efficiency additivity: prediction == base_value + sum(shap_values)
    shap_sum = sum(f["shap_value"] for f in exp_data["features"])
    reconstructed_pred = exp_data["base_value"] + shap_sum
    assert abs(exp_data["prediction"] - reconstructed_pred) < 0.01
    assert exp_data["additivity_delta"] < 1e-4

    # Verify direct D9 explainer parity
    direct_exp = explain_prediction(TEST_BASELINE_STATE)
    assert exp_data["prediction"] == direct_exp["prediction"]
    assert exp_data["base_value"] == direct_exp["base_value"]


def test_e2e_frontend_assets_and_ui_contract():
    """Verify frontend HTML, CSS, JS asset serving and complete UI component contract."""
    # 1. HTML index
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers.get("content-type", "")
    html = res_index.text

    assert "Personal Financial Digital Twin" in html
    assert "Explainable Next-Month Spending Forecasting" in html
    assert "Counterfactual What-If Scenario Visualization" in html
    assert "bar-chart-container" in html
    assert "changed-inputs-list" in html
    assert "recalculated-derived-list" in html

    # 2. CSS stylesheet asset
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "--bg-base" in res_css.text

    # 3. JS application script asset
    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    js = res_js.text
    assert "/api/predict" in js
    assert "/api/simulate" in js
    assert "/api/explain" in js
    assert "/health" in js


def test_e2e_error_handling_and_boundary_guardrails():
    """Verify system-wide error handling, invalid input rejection, and guardrails."""
    # 1. Reject negative debit count
    invalid_state = TEST_BASELINE_STATE.copy()
    invalid_state["debit_count_t"] = -10
    res_inv = client.post("/api/predict", json=invalid_state)
    assert res_inv.status_code == 400
    assert "debit_count_t" in res_inv.json()["detail"]

    # 2. Reject prohibited derived feature override attempt in simulation
    override_payload = {
        "baseline_state": TEST_BASELINE_STATE,
        "scenario_changes": {"spending_3m_mean": 99999.0}
    }
    res_ovr = client.post("/api/simulate", json=override_payload)
    assert res_ovr.status_code == 400
    assert "prohibited" in res_ovr.json()["detail"].lower()

    # 3. Reject unexpected extra fields in prediction
    extra_field_state = TEST_BASELINE_STATE.copy()
    extra_field_state["unrecognized_field"] = 123.45
    res_extra = client.post("/api/predict", json=extra_field_state)
    assert res_extra.status_code == 422

    # 4. Zero baseline division safety check
    zero_baseline = TEST_BASELINE_STATE.copy()
    zero_baseline["spending_hh_t"] = 0.0
    zero_baseline["spending_st_t"] = 0.0
    zero_baseline["spending_in_t"] = 0.0
    zero_baseline["spending_lo_t"] = 0.0
    zero_baseline["spending_io_t"] = 0.0
    zero_baseline["spending_other_t"] = 0.0
    sim_res = simulate_scenario(zero_baseline, {"ending_balance_t": 50000.0})
    assert "absolute_difference" in sim_res
    assert "percentage_difference" in sim_res


def test_e2e_mathematical_fidelity_ddof_zero():
    """Verify mathematical fidelity of derived spending_3m_std feature (ddof=0)."""
    state_obj = FinancialState.from_dict(TEST_BASELINE_STATE)
    feature_df = derive_feature_vector(state_obj)

    spending_t = TEST_BASELINE_STATE["spending_hh_t"] + TEST_BASELINE_STATE["spending_st_t"] + \
                 TEST_BASELINE_STATE["spending_in_t"] + TEST_BASELINE_STATE["spending_lo_t"] + \
                 TEST_BASELINE_STATE["spending_io_t"] + TEST_BASELINE_STATE["spending_other_t"]
    lags = [spending_t, TEST_BASELINE_STATE["spending_t_minus_1"], TEST_BASELINE_STATE["spending_t_minus_2"]]

    expected_mean = sum(lags) / 3.0
    expected_std_ddof0 = float(np.std(lags, ddof=0))

    assert feature_df["spending_t"].iloc[0] == spending_t
    assert abs(feature_df["spending_3m_mean"].iloc[0] - expected_mean) < 1e-5
    assert abs(feature_df["spending_3m_std"].iloc[0] - expected_std_ddof0) < 1e-5
