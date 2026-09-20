"""
Personal Financial Digital Twin — REST API Integration Test Suite (Dimension 10)

Tests:
1. GET /health status check (HTTP 200 & lightweight artifact check).
2. GET /api/schema metadata contract (HTTP 200 & EXACT_14_FEATURE_ORDER match).
3. POST /api/predict with valid financial state (HTTP 200 & gateway parity).
4. POST /api/predict with invalid financial state (HTTP 400 validation error).
5. POST /api/simulate with valid counterfactual (HTTP 200 & D8 engine parity).
6. POST /api/simulate rejecting direct derived feature override (HTTP 400).
7. POST /api/explain with valid financial state (HTTP 200, 14 attributions, additivity).
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
from src.app.simulator import ALLOWED_PRIMARY_INPUTS, DERIVED_FEATURES
from src.app.explainer import explain_prediction

client = TestClient(app)

# Test Fixture Sample State
SAMPLE_STATE_DICT = {
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


def test_health_endpoint():
    """Verify /health returns HTTP 200 and indicates model artifact readiness safely."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dimension"] == "D10"
    assert data["model_artifact_exists"] is True
    assert data["model_artifact"] == "models/catboost_model.joblib"


def test_schema_endpoint():
    """Verify /api/schema returns HTTP 200 and matches feature contract constants."""
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert data["exact_14_feature_order"] == EXACT_14_FEATURE_ORDER
    assert data["allowed_primary_inputs"] == sorted(list(ALLOWED_PRIMARY_INPUTS))
    assert data["derived_features"] == sorted(list(DERIVED_FEATURES))
    assert "derived_feature_rules" in data


def test_predict_endpoint_valid():
    """Verify POST /api/predict returns HTTP 200 and matches direct predict_spending()."""
    response = client.post("/api/predict", json=SAMPLE_STATE_DICT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["currency"] == "CZK"

    # Verify exact parity with direct D7 prediction gateway call
    expected_pred = predict_spending(SAMPLE_STATE_DICT)
    assert data["prediction"] == expected_pred


def test_predict_endpoint_invalid_input():
    """Verify POST /api/predict rejects invalid domain inputs with HTTP 400."""
    invalid_state = SAMPLE_STATE_DICT.copy()
    invalid_state["debit_count_t"] = -5  # Negative transaction count is invalid

    response = client.post("/api/predict", json=invalid_state)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "debit_count_t" in data["detail"]


def test_predict_endpoint_unexpected_extra_field_rejected():
    """Verify POST /api/predict rejects unexpected extra fields with HTTP 422 validation error."""
    invalid_state = SAMPLE_STATE_DICT.copy()
    invalid_state["unexpected_unknown_field"] = 9999.0

    response = client.post("/api/predict", json=invalid_state)
    assert response.status_code == 422


def test_simulate_endpoint_valid():
    """Verify POST /api/simulate returns HTTP 200 and correct scenario calculations."""
    payload = {
        "baseline_state": SAMPLE_STATE_DICT,
        "scenario_changes": {"spending_hh_t": 5000.0},
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "baseline_prediction" in data
    assert "scenario_prediction" in data
    assert "absolute_difference" in data
    assert "percentage_difference" in data
    assert data["scenario_state"]["spending_hh_t"] == 5000.0
    # spending_t is derived automatically when derive_feature_vector is called
    X_scen = derive_feature_vector(FinancialState.from_dict(data["scenario_state"]))
    expected_new_spending_t = 5000.0 + 500.0 + 1200.0 + 2000.0 + 300.0 + 1500.0
    assert X_scen["spending_t"].iloc[0] == expected_new_spending_t


def test_simulate_endpoint_derived_override_rejected():
    """Verify POST /api/simulate rejects direct derived feature overrides with HTTP 400."""
    payload = {
        "baseline_state": SAMPLE_STATE_DICT,
        "scenario_changes": {"spending_t": 10000.0},  # spending_t is a derived feature!
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "spending_t" in data["detail"]
    assert "prohibited" in data["detail"]


def test_explain_endpoint_valid():
    """Verify POST /api/explain returns HTTP 200 and matches D9 TreeSHAP output."""
    response = client.post("/api/explain", json=SAMPLE_STATE_DICT)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "base_value" in data
    assert "features" in data
    assert len(data["features"]) == 14

    # Verify exact feature order in features list
    feature_names = [f["feature"] for f in data["features"]]
    assert feature_names == EXACT_14_FEATURE_ORDER

    # Verify additivity delta preservation
    assert "additivity_delta" in data
    assert data["additivity_delta"] < 1e-4

    # Verify exact parity with direct explain_prediction() function
    direct_exp = explain_prediction(SAMPLE_STATE_DICT)
    assert data["prediction"] == direct_exp["prediction"]
    assert data["base_value"] == direct_exp["base_value"]


def test_export_pdf_endpoint_valid():
    """Verify POST /api/export/pdf returns HTTP 200 and application/pdf content without CZK wording."""
    payload = {
        "baseline_state": SAMPLE_STATE_DICT,
        "prediction": 8250.50,
    }
    response = client.post("/api/export/pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")
    assert b"CZK" not in response.content


def test_export_excel_endpoint_valid():
    """Verify POST /api/export/excel returns HTTP 200 and spreadsheet content without CZK wording."""
    payload = {
        "baseline_state": SAMPLE_STATE_DICT,
        "prediction": 8250.50,
    }
    response = client.post("/api/export/excel", json=payload)
    assert response.status_code == 200
    assert "spreadsheetml.sheet" in response.headers["content-type"]
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert len(response.content) > 500
    assert b"CZK" not in response.content


