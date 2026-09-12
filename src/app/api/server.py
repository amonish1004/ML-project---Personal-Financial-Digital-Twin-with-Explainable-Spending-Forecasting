"""
Personal Financial Digital Twin — REST API Server (Dimension 10)

Provides a lightweight, non-duplicative FastAPI REST interface over existing
D7 (Inference), D8 (Simulator), and D9 (SHAP Explainer) services.

Architectural Contract:
- Thin HTTP presentation layer.
- Delegates prediction logic to predict_spending() (D7).
- Delegates scenario analysis to simulate_scenario() (D8).
- Delegates attribution logic to explain_prediction() (D9).
- Re-uses exact feature ordering, validation, and derivation from src.app.schema.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

from src.app.schema import (
    EXACT_14_FEATURE_ORDER,
    FinancialState,
)
from src.app.inference import predict_spending
from src.app.simulator import (
    simulate_scenario,
    ALLOWED_PRIMARY_INPUTS,
    DERIVED_FEATURES,
)
from src.app.explainer import explain_prediction

# Instantiate FastAPI Application
app = FastAPI(
    title="Personal Financial Digital Twin API",
    description="Explainable Spending Forecasting REST API Backend (Dimension 10 & 11)",
    version="1.0.0",
)

# Mount Static Files & Serve Frontend Interface
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Personal Financial Digital Twin REST API is running. Visit /docs for Swagger UI."}


# Exception Handlers mapping domain errors to HTTP 400
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


# Request Models
class FinancialStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ending_balance_t: float = Field(..., description="Liquid account balance at month t (CZK)")
    income_credit_t: float = Field(..., description="Cash inflows at month t (CZK)")
    debit_count_t: int = Field(..., description="Count of debit transactions at month t (integer >= 0)")
    spending_hh_t: float = Field(0.0, description="Household debit spending subtotal (CZK)")
    spending_st_t: float = Field(0.0, description="Statement/fee debit spending subtotal (CZK)")
    spending_in_t: float = Field(0.0, description="Insurance debit spending subtotal (CZK)")
    spending_lo_t: float = Field(0.0, description="Loan repayment debit spending subtotal (CZK)")
    spending_io_t: float = Field(0.0, description="Interest outflow debit spending subtotal (CZK)")
    spending_other_t: float = Field(0.0, description="Uncategorized debit spending subtotal (CZK)")
    spending_t_minus_1: float = Field(0.0, description="Total debit spending at month t-1 (CZK)")
    spending_t_minus_2: float = Field(0.0, description="Total debit spending at month t-2 (CZK)")


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_state: FinancialStateRequest = Field(..., description="Baseline financial state profile")
    scenario_changes: Dict[str, Any] = Field(..., description="Dictionary of primary input overrides")


# Endpoints
@app.get("/health", summary="API Health and Model Artifact Check")
def health_check():
    """
    Lightweight health status endpoint.
    Checks API readiness and verifies CatBoost model artifact existence without
    unnecessarily instantiating heavy SHAP TreeExplainer instances.
    """
    model_path = PROJECT_ROOT / "models" / "catboost_model.joblib"
    artifact_exists = model_path.exists()

    return {
        "status": "healthy" if artifact_exists else "degraded",
        "service": "Personal Financial Digital Twin REST API",
        "dimension": "D10",
        "model_artifact_exists": artifact_exists,
        "model_artifact": "models/catboost_model.joblib",
    }


@app.get("/api/schema", summary="Authoritative Feature and Input Schema Metadata")
def get_schema():
    """
    Exposes authoritative feature contract, editable inputs, and derived calculation rules.
    """
    return {
        "exact_14_feature_order": EXACT_14_FEATURE_ORDER,
        "allowed_primary_inputs": sorted(list(ALLOWED_PRIMARY_INPUTS)),
        "derived_features": sorted(list(DERIVED_FEATURES)),
        "derived_feature_rules": {
            "spending_t": "spending_hh_t + spending_st_t + spending_in_t + spending_lo_t + spending_io_t + spending_other_t",
            "spending_3m_mean": "(spending_t + spending_t_minus_1 + spending_t_minus_2) / 3.0",
            "spending_3m_std": "np.std([spending_t, spending_t_minus_1, spending_t_minus_2], ddof=0)",
        },
    }


@app.post("/api/predict", summary="Predict Next-Month Spending")
def predict_endpoint(request: FinancialStateRequest):
    """
    Accepts financial state and returns point prediction via D7 predict_spending gateway.
    """
    state_dict = request.model_dump()
    prediction = predict_spending(state_dict)
    return {
        "prediction": prediction,
        "currency": "CZK",
    }


@app.post("/api/simulate", summary="Counterfactual What-If Scenario Simulation")
def simulate_endpoint(request: SimulateRequest):
    """
    Accepts baseline state and scenario modifications, delegating to D8 simulate_scenario engine.
    """
    baseline_dict = request.baseline_state.model_dump()
    simulation_result = simulate_scenario(
        baseline_state=baseline_dict,
        scenario_changes=request.scenario_changes,
    )
    return simulation_result


@app.post("/api/explain", summary="Real-Time TreeSHAP Explanation")
def explain_endpoint(request: FinancialStateRequest):
    """
    Accepts financial state and returns local feature attributions via D9 explain_prediction service.
    """
    state_dict = request.model_dump()
    explanation_result = explain_prediction(state_dict)
    return explanation_result
