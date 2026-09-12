# Dimension 10 — REST API Backend Service

## 1. Executive Summary & Objective

Dimension 10 establishes a high-performance, non-duplicative **REST API Application Backend Layer** for the **Personal Financial Digital Twin with Explainable Spending Forecasting**.

The primary objective of D10 is to expose the core machine learning inference, what-if counterfactual scenario simulation, and real-time TreeSHAP feature explanation capabilities to web application clients, mobile interfaces, and external consumers via standard HTTP REST endpoints.

> [!IMPORTANT]
> **Architectural Integrity Guarantee**: D10 acts strictly as a **thin HTTP presentation layer**. It delegates all financial validation and feature derivation to Dimension 7 (`src.app.schema`), model inference to Dimension 7 (`src.app.inference`), counterfactual simulation to Dimension 8 (`src.app.simulator`), and TreeSHAP attribution to Dimension 9 (`src.app.explainer`). D10 does **not** duplicate CatBoost model loading, feature derivation mathematics, or SHAP calculations.

---

## 2. Technical Framework Selection

**FastAPI** was selected as the web framework for this student ML project based on the following key evaluation criteria:

1. **Native Pydantic Data Validation**: Automatic type parsing and structured error reporting for incoming HTTP JSON payloads.
2. **Asynchronous & High-Performance Core**: Built on Starlette and ASGI, providing minimal overhead.
3. **Interactive OpenAPI / Swagger Documentation**: Automatically generates interactive UI documentation at `/docs` without extra configuration.
4. **Clean Integration with Existing Codebase**: Seamlessly wraps existing Python dataclasses and functions from `src.app`.

---

## 3. Application Architecture

```
                                 HTTP REST Request
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │        D10 — FastAPI REST API            │
                   │        (src/app/api/server.py)           │
                   └────────────────────┬─────────────────────┘
                                        │ Delegates to
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
    ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
    │   D7 — Inference   │    │   D8 — Simulator   │    │   D9 — Explainer   │
    │ predict_spending() │    │ simulate_scenario()│    │ explain_prediction │
    └──────────┬─────────┘    └─────────┬──────────┘    └─────────┬──────────┘
               │                        │                         │
               └────────────────────────┼─────────────────────────┘
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │    Frozen Model & SHAP Explainer Cache   │
                   │    models/catboost_model.joblib          │
                   └──────────────────────────────────────────┘
```

### Layer Responsibilities:
- **Presentation (D10)**: Receives HTTP JSON, validates schema structure via Pydantic models, maps exceptions to clean HTTP 400 responses, and returns serializable JSON responses.
- **Service Layer (D7/D8/D9)**: Enforces domain rules, checks non-negativity and integer constraints, derives the exact 14-feature vector with `ddof=0`, and executes model predictions/explanations.
- **Model Storage Layer**: Frozen CatBoost model (`models/catboost_model.joblib`) cached via singleton pattern.

---

## 4. REST API Endpoint Specifications

### Endpoint Overview

| Method | Path | Summary | Service Delegated |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | API Health & Model Artifact Check | Disk Check (`catboost_model.joblib`) |
| `GET` | `/api/schema` | Feature Contract & Schema Metadata | `src.app.schema` Constants |
| `POST` | `/api/predict` | Predict Next-Month Spending (CZK) | `predict_spending()` |
| `POST` | `/api/simulate` | Counterfactual What-If Scenario | `simulate_scenario()` |
| `POST` | `/api/explain` | Real-Time TreeSHAP Attribution | `explain_prediction()` |

---

### Endpoint Details & Request/Response Contracts

#### 1. `GET /health`
- **Description**: Lightweight health status endpoint. Verifies API readiness and checks model artifact existence on disk without instantiating heavy SHAP `TreeExplainer` objects.
- **Response Example (Illustrative HTTP 200 Response)**:
```json
{
  "status": "healthy",
  "service": "Personal Financial Digital Twin REST API",
  "dimension": "D10",
  "model_artifact_exists": true,
  "model_artifact": "models/catboost_model.joblib"
}
```

#### 2. `GET /api/schema`
- **Description**: Exposes the authoritative 14-feature order contract, list of user-editable primary inputs, prohibited derived features, and derived feature mathematics.
- **Response Example (Illustrative HTTP 200 Response)**:
```json
{
  "exact_14_feature_order": [
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
    "spending_other_t"
  ],
  "allowed_primary_inputs": [
    "debit_count_t",
    "ending_balance_t",
    "income_credit_t",
    "spending_hh_t",
    "spending_in_t",
    "spending_io_t",
    "spending_lo_t",
    "spending_other_t",
    "spending_st_t",
    "spending_t_minus_1",
    "spending_t_minus_2"
  ],
  "derived_features": [
    "spending_3m_mean",
    "spending_3m_std",
    "spending_t"
  ]
}
```

#### 3. `POST /api/predict`
- **Description**: Predicts next-month total spending in CZK for an account financial state.
- **Request Body (Illustrative Request Example)**:
```json
{
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
  "spending_t_minus_2": 7500.0
}
```
- **Response Example (Illustrative Response Example)**:
```json
{
  "prediction": 6535.61,
  "currency": "CZK"
}
```

#### 4. `POST /api/simulate`
- **Description**: Runs counterfactual scenario analysis comparing baseline financial state against scenario changes.
- **Request Body (Illustrative Request Example)**:
```json
{
  "baseline_state": {
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
    "spending_t_minus_2": 7500.0
  },
  "scenario_changes": {
    "spending_hh_t": 5000.0
  }
}
```
- **Response Example (Illustrative Response Example)**:
```json
{
  "baseline_state": { ... },
  "scenario_state": { ... },
  "baseline_prediction": 6535.61,
  "scenario_prediction": 7120.45,
  "absolute_difference": 584.84,
  "percentage_difference": 8.95
}
```

#### 5. `POST /api/explain`
- **Description**: Computes single-instance local TreeSHAP feature attributions.
- **Response Example (Illustrative Response Example)**:
```json
{
  "prediction": 6535.61,
  "raw_prediction": 6535.6136,
  "base_value": 1612.1063,
  "features": [
    {"feature": "spending_t", "value": 8500.0, "shap_value": 1120.45},
    {"feature": "spending_t_minus_1", "value": 8000.0, "shap_value": 450.21}
  ],
  "reconstructed_prediction": 6535.6136,
  "additivity_delta": 0.000011
}
```

---

## 5. Error Handling & Security Protections

### HTTP Error Mapping
- **Validation Errors (`ValueError`, `TypeError`)**: Intercepted by FastAPI custom exception handlers and mapped to `HTTP 400 Bad Request` with structured JSON error details: `{"detail": "<error_message>"}`.
- **Malformed Request Bodies & Unexpected Extra Fields**: Handled natively by Pydantic with `ConfigDict(extra="forbid")` (`HTTP 422 Unprocessable Entity`).
- **Prohibited Derived Feature Overrides**: Direct attempts to override `spending_t`, `spending_3m_mean`, or `spending_3m_std` in `/api/simulate` return `HTTP 400 Bad Request` with message: `"Direct override of derived feature 'spending_t' is prohibited."`

### Model & Data Protection Guarantees
1. **Model Immutability**: The CatBoost model (`models/catboost_model.joblib`) is accessed in read-only mode and cached in memory.
2. **Dataset Isolation**: No raw or processed datasets are modified or loaded by the API backend.
3. **No Direct Overrides**: Derived features are recalculated strictly using `ddof=0` standard deviation and exact category sums.

---

## 6. Local Server Execution & Interactive Documentation

### Starting the API Server
To start the REST API backend locally using the virtual environment:

```powershell
.\.venv\Scripts\python -m uvicorn src.app.api.server:app --reload
```

Output upon startup:
```text
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Accessing Interactive Swagger UI
Once running, open your web browser to:
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`

---

## 7. Verification & Test Suite Execution

The REST API backend is verified using `fastapi.testclient.TestClient` in `tests/test_api.py`.

### Test Summary

```powershell
.\.venv\Scripts\python -m pytest tests/test_api.py
```

Result: `8 passed, 5 warnings in 2.75s`

### Complete Project Regression Test Suite

```powershell
.\.venv\Scripts\python -m pytest
```

Result:
```text
collected 42 items

tests/test_api.py ........                                               [ 19%]
tests/test_explainer.py ..............                                   [ 52%]
tests/test_inference.py .......                                          [ 69%]
tests/test_simulator.py .............                                    [100%]

======================= 42 passed, 5 warnings in 2.97s =======================
```

| Dimension Test Suite | Tests Passed | Status |
| :--- | :--- | :--- |
| D7 Model Inference Gateway | 7 / 7 | PASSED |
| D8 What-If Scenario Simulator | 13 / 13 | PASSED |
| D9 Individual Real-Time SHAP | 14 / 14 | PASSED |
| D10 REST API Backend Service | 8 / 8 | PASSED |
| **Total Combined Suite** | **42 / 42** | **PASSED** |
