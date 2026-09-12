# Dimension 11 — Interactive Web Frontend Interface

## 1. Executive Summary & Objective

Dimension 11 implements a responsive, modern **Interactive Web Frontend Dashboard** for the **Personal Financial Digital Twin with Explainable Spending Forecasting**.

The web interface enables users to interactively configure account financial profiles, compute next-month spending forecasts in real-time, execute what-if counterfactual scenario simulations, and visualize local TreeSHAP feature attributions directly in the web browser.

> [!IMPORTANT]
> **Strict Presentation Layer Boundary**: The frontend operates strictly as an un-privileged HTTP presentation layer. It communicates **exclusively** with the Dimension 10 REST API backend (`/api/predict`, `/api/simulate`, `/api/explain`, `/health`, `/api/schema`). The browser client does **not** load the CatBoost model, execute SHAP algorithms, or duplicate backend feature derivation logic.

---

## 2. System Architecture & Component Design

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        D11 — Browser Web Client                        │
 │                (HTML5 / Vanilla CSS Glassmorphic Dashboard)            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP REST (JSON)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      D10 — FastAPI REST API Server                     │
 │                        (src/app/api/server.py)                         │
 └───────┬───────────────────────────┬───────────────────────────┬────────┘
         │                           │                           │
         ▼                           ▼                           ▼
 ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
 │ D7 Inference  │           │ D8 Simulator  │           │ D9 Explainer  │
 │  predict()    │           │  simulate()   │           │   explain()   │
 └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     Frozen CatBoost & TreeSHAP Cache                   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Visual Design System & UI Components

### Dark Glassmorphic Aesthetic
The user interface follows modern dark glassmorphism design principles:
- **Color Palette**: Dark slate background (`#0b0f19`), translucent cards (`rgba(26, 34, 52, 0.7)`), subtle borders (`rgba(255, 255, 255, 0.08)`), and vibrant blue (`#3b82f6`) / purple (`#8b5cf6`) accents.
- **Typography**: Clean typography using **Inter** for UI labels and **JetBrains Mono** for numeric financial values.
- **Micro-Interactions**: Smooth hover states, glowing interactive buttons, and toast notifications.

### Dashboard Layout & Controls

1. **Header & Health Status Indicator**:
   - Polling `GET /health` displays a live badge: `API Online • Model Ready`.
2. **Financial Profile Input Card**:
   - **Preset Profile Buttons**: Quick-load buttons ("Default Profile", "High Spender", "Saver Profile") pre-fill standard financial states.
   - **Primary Editable Inputs**: 11 user-editable numeric inputs categorized into:
     - *Primary Account State*: `ending_balance_t`, `income_credit_t`, `debit_count_t`.
     - *Category Outflow Breakdowns*: `spending_hh_t`, `spending_st_t`, `spending_in_t`, `spending_lo_t`, `spending_io_t`, `spending_other_t`.
     - *Historical Spending Lags*: `spending_t_minus_1`, `spending_t_minus_2`.
   - **Auto-Derived Preview Banner**: Displays read-only previews of `spending_t`, `spending_3m_mean`, and `spending_3m_std` (`ddof=0`).
3. **Next-Month Forecast Card (`POST /api/predict`)**:
   - Prominently displays predicted outgoing expenditure in CZK with currency badge and loading indicator.
4. **What-If Scenario Simulator Card (`POST /api/simulate`)**:
   - Allows selecting a primary input field and entering a counterfactual scenario value.
   - Displays baseline forecast, scenario forecast, absolute delta (CZK), and percentage delta (%).
5. **TreeSHAP Feature Attribution Card (`POST /api/explain`)**:
   - Displays base expected value $E[f(X)]$, reconstructed prediction, and additivity delta check.
   - Renders a horizontal waterfall bar list for all 14 features in `EXACT_14_FEATURE_ORDER`.
   - Red/coral bars (`#ef4444`) indicate positive attributions (increases spending forecast); blue/cyan bars (`#3b82f6`) indicate negative attributions (decreases spending forecast).
   - Includes explicit non-causal attribution disclaimer wording.

---

## 4. REST API Endpoint Integration

| Component Action | Method | API Endpoint | Payload Sent | Result Rendered |
| :--- | :--- | :--- | :--- | :--- |
| **Health Check** | `GET` | `/health` | None | Header status badge |
| **Run Forecast** | `POST` | `/api/predict` | `FinancialStateRequest` JSON | Predicted spending value (CZK) |
| **Run Simulation** | `POST` | `/api/simulate` | `SimulateRequest` JSON | Scenario comparison grid & deltas |
| **Explain Forecast**| `POST` | `/api/explain` | `FinancialStateRequest` JSON | TreeSHAP metadata & bar charts |

---

## 5. Local Application Execution

To run the complete integrated web application and API backend locally:

```powershell
.\.venv\Scripts\python -m uvicorn src.app.api.server:app --reload
```

Then open your web browser to:
- **Interactive Web Interface**: `http://127.0.0.1:8000/`
- **Swagger API Documentation**: `http://127.0.0.1:8000/docs`

---

## 6. Verification & Test Suite Execution

Frontend availability and asset delivery are verified in `tests/test_frontend.py`.

### Test Summary

```powershell
.\.venv\Scripts\python -m pytest tests/test_frontend.py
```

Result: `3 passed, 5 warnings in 2.41s`

### Complete Project Regression Test Suite

```powershell
.\.venv\Scripts\python -m pytest
```

Result:
```text
collected 45 items

tests/test_api.py ........                                               [ 17%]
tests/test_explainer.py ..............                                   [ 48%]
tests/test_frontend.py ...                                               [ 55%]
tests/test_inference.py .......                                          [ 71%]
tests/test_simulator.py .............                                    [100%]

======================= 45 passed, 5 warnings in 3.09s =======================
```

| Dimension Test Suite | File | Tests Passed | Status |
| :--- | :--- | :---: | :---: |
| D7 Model Inference Gateway | `tests/test_inference.py` | 7 / 7 | PASSED |
| D8 What-If Scenario Simulator | `tests/test_simulator.py` | 13 / 13 | PASSED |
| D9 Individual Real-Time SHAP | `tests/test_explainer.py` | 14 / 14 | PASSED |
| D10 REST API Backend Service | `tests/test_api.py` | 8 / 8 | PASSED |
| D11 Interactive Web Frontend | `tests/test_frontend.py` | 3 / 3 | PASSED |
| **Total Combined Suite** | **All Modules** | **45 / 45** | **PASSED** |
