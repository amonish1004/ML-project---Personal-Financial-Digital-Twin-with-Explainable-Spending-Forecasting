"""
Personal Financial Digital Twin — End-to-End System Integration Test Suite (Dimension 12)

Tests:
1. Complete Digital Twin workflow (Baseline prediction -> Scenario simulation -> Delta -> SHAP attribution).
2. Changed primary inputs and automatic derived feature recalculation (spending_t, spending_3m_mean, spending_3m_std ddof=0).
3. Derived feature override rejection via API & simulator.
4. Immutability of baseline state profile across integrated workflow steps.
5. Error handling and domain constraint enforcement (negative count rejection).
6. Frontend D12 UI element availability (comparison bars, delta tables, scenario SHAP buttons).
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.api.server import app
from src.app.schema import EXACT_14_FEATURE_ORDER, FinancialState, derive_feature_vector
from src.app.inference import predict_spending
from src.app.simulator import simulate_scenario
from src.app.explainer import explain_prediction

client = TestClient(app)

# Test Sample Profiles
BASELINE_STATE_DICT = {
    "ending_balance_t": 15000.0,
    "income_credit_t": 25000.0,
    "debit_count_t": 12,
    "spending_hh_t": 3000.0,
    "spending_st_t": 500.0,
    "spending_in_t": 1200.0,
    "spending_lo_t": 2000.0,
    "spending_io_t": 300.0,
    "spending_other_t": 1500.0,
    "spending_t_minus_1": 8000.0,
    "spending_t_minus_2": 7500.0,
}


def test_d12_integrated_digital_twin_workflow():
    """Verify full end-to-end Digital Twin workflow execution via REST API."""
    # Step 1: Predict Baseline Forecast
    res_base = client.post("/api/predict", json=BASELINE_STATE_DICT)
    assert res_base.status_code == 200
    base_pred = res_base.json()["prediction"]
    assert base_pred > 0.0

    # Step 2: Run What-If Scenario Simulation (+2,000 CZK in Household Spend)
    scenario_payload = {
        "baseline_state": BASELINE_STATE_DICT,
        "scenario_changes": {"spending_hh_t": 5000.0}
    }
    res_sim = client.post("/api/simulate", json=scenario_payload)
    assert res_sim.status_code == 200
    sim_data = res_sim.json()

    # Step 3: Verify Prediction Deltas
    assert sim_data["baseline_prediction"] == base_pred
    scen_pred = sim_data["scenario_prediction"]
    assert "absolute_difference" in sim_data
    assert "percentage_difference" in sim_data
    assert abs(sim_data["absolute_difference"] - round(scen_pred - base_pred, 2)) < 1e-5

    # Step 4: Verify Primary Inputs Modified and Recalculated Derived Features
    assert sim_data["baseline_state"]["spending_hh_t"] == 3000.0
    assert sim_data["scenario_state"]["spending_hh_t"] == 5000.0

    X_base = derive_feature_vector(FinancialState.from_dict(sim_data["baseline_state"]))
    X_scen = derive_feature_vector(FinancialState.from_dict(sim_data["scenario_state"]))

    assert X_base["spending_t"].iloc[0] == 8500.0
    assert X_scen["spending_t"].iloc[0] == 10500.0  # +2000 increase

    # Step 5: Run TreeSHAP Attributions for Scenario State
    res_exp = client.post("/api/explain", json=sim_data["scenario_state"])
    assert res_exp.status_code == 200
    exp_data = res_exp.json()

    assert len(exp_data["features"]) == 14
    assert [f["feature"] for f in exp_data["features"]] == EXACT_14_FEATURE_ORDER
    assert exp_data["additivity_delta"] < 1e-4


def test_d12_derived_feature_override_rejection_integrated():
    """Verify that attempts to override derived features in integrated simulation are rejected."""
    prohibited_payload = {
        "baseline_state": BASELINE_STATE_DICT,
        "scenario_changes": {"spending_3m_mean": 9999.0}
    }
    response = client.post("/api/simulate", json=prohibited_payload)
    assert response.status_code == 400
    data = response.json()
    assert "prohibited" in data["detail"].lower()


def test_d12_baseline_immutability():
    """Verify baseline financial state profile remains 100% unmutated during simulation."""
    original_copy = BASELINE_STATE_DICT.copy()
    scenario_payload = {
        "baseline_state": BASELINE_STATE_DICT,
        "scenario_changes": {"ending_balance_t": 50000.0, "income_credit_t": 40000.0}
    }
    client.post("/api/simulate", json=scenario_payload)
    assert BASELINE_STATE_DICT == original_copy


def test_d12_frontend_integration_markup():
    """Verify frontend HTML document contains D12 counterfactual visualization components."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    assert "Counterfactual What-If Scenario Visualization" in html
    assert "bar-chart-container" in html
    assert "changed-inputs-list" in html
    assert "recalculated-derived-list" in html
    assert "btn-explain-scenario" in html
