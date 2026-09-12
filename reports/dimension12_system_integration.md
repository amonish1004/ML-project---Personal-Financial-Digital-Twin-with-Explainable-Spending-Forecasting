# Dimension 12 — System Integration & Counterfactual Visualization

## 1. Objective

Dimension 12 unifies the individual components built across Dimensions 7 through 11 into a cohesive, interactive **Personal Financial Digital Twin** workflow. 

The primary goals of Dimension 12 are:
1. Provide an integrated browser interface connecting baseline prediction, counterfactual simulation, change attribution, and TreeSHAP explainability.
2. Enable side-by-side comparative visualization of baseline vs counterfactual next-month spending forecasts.
3. Automatically highlight modified primary inputs while recalculating and displaying derived features (`spending_t`, `spending_3m_mean`, `spending_3m_std` with `ddof=0`).
4. Support optional scenario-level TreeSHAP feature contribution explanations without duplicating SHAP logic or introducing client-side calculation.
5. Enforce non-causal language across all UI elements, making it explicit that scenario results are model predictions under past training patterns, not guarantees or causal proofs.

---

## 2. Existing Architecture Reused

Dimension 12 strictly reuses the established 5-layer system architecture without adding new machine learning frameworks, custom prediction engines, or duplicate endpoints.

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        D11 / D12 — Web Frontend                        │
 │            (HTML5 / Vanilla CSS / JavaScript Presentation Layer)       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP REST (JSON)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      D10 — FastAPI REST API Server                     │
 │     Endpoints: GET /health, GET /api/schema, POST /api/predict,        │
 │                POST /api/simulate, POST /api/explain                   │
 └───────┬───────────────────────────┬───────────────────────────┬────────┘
         │                           │                           │
         ▼                           ▼                           ▼
 ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
 │ D7 Inference  │           │ D8 Simulator  │           │ D9 Explainer  │
 │ predict_      │           │ simulate_     │           │ explain_      │
 │ spending()    │           │ scenario()    │           │ prediction()  │
 └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │             Frozen CatBoost Model (models/catboost_model.joblib)       │
 │                     Authoritative 14-Feature Contract                  │
 └────────────────────────────────────────────────────────────────────────┘
```

- **Frontend Client (`src/app/static/app.js`)**: Orchestrates user actions and communicates exclusively via D10 REST API endpoints.
- **REST Gateway (`src/app/api/server.py`)**: Routes JSON payloads to backend services and formats structured HTTP responses.
- **Inference Gateway (`src/app/inference.py`)**: Validates input state, derives features, enforces feature order, and invokes frozen CatBoost.
- **Simulator (`src/app/simulator.py`)**: Computes baseline vs scenario state, recalculates derived values, and measures prediction deltas.
- **Explainer (`src/app/explainer.py`)**: Computes exact TreeSHAP values for baseline or scenario states.

---

## 3. D12 Integrated Digital Twin Workflow

The Digital Twin workflow operates in four sequential steps:

1. **Baseline State Input**: The user loads or inputs an initial 11-primary-feature account state.
2. **Baseline Forecast**: The client calls `POST /api/predict` to obtain the baseline forecast.
3. **What-If Scenario Configuration**: The user modifies one or more primary features in the What-If Scenario panel.
4. **Counterfactual Execution**: Clicking **Run What-If Simulation** calls `POST /api/simulate`, returning:
   - Baseline spending prediction (CZK)
   - Scenario spending prediction (CZK)
   - Absolute delta ($\Delta$) and percentage delta ($\Delta\%$)
   - Recalculated scenario derived features
   - Primary input diff list
5. **Scenario Explanation (Optional)**: Clicking **Explain Scenario Forecast** calls `POST /api/explain` with scenario inputs to inspect TreeSHAP feature attributions.

---

## 4. Baseline Spending Prediction

When the baseline state is submitted:
1. `spending_t` is automatically computed as the sum of category spending (`spending_hh_t` through `spending_other_t`).
2. `spending_3m_mean` and `spending_3m_std` (`ddof=0`) are calculated from `spending_t`, `spending_t_minus_1`, and `spending_t_minus_2`.
3. The 14 features are passed to `CatBoostRegressor.predict()`.
4. The prediction is rendered in CZK in both the Baseline card and the comparison visualizer.

---

## 5. Counterfactual Simulation

Counterfactual simulation allows testing hypothetical financial adjustments (e.g., increasing income, reducing discretionary category spending, or increasing account balance).

- **Primary Input Isolation**: Users may edit only valid primary inputs (`ending_balance_t`, `income_credit_t`, `debit_count_t`, category spendings, and historical spending lags).
- **Derived Feature Re-derivation**: In `simulate_scenario()`, derived features are recalculated from modified primary inputs. Manual overrides of derived features are strictly rejected.
- **Side-by-Side Comparison**: Both predictions are displayed alongside absolute change and percentage change.

---

## 6. Prediction Delta & Visualization

### Metrics Calculation
- **Absolute Delta**:
  $$\Delta = \text{Prediction}_{\text{scenario}} - \text{Prediction}_{\text{baseline}}$$
- **Percentage Delta**:
  $$\Delta\% = \begin{cases} \left( \frac{\Delta}{\text{Prediction}_{\text{baseline}}} \right) \times 100 & \text{if } \text{Prediction}_{\text{baseline}} \neq 0 \\ \text{N/A} & \text{if } \text{Prediction}_{\text{baseline}} = 0 \end{cases}$$

### Visualizer Component
The UI renders a comparative dual-bar visualizer using pure HTML/CSS:
- **Baseline Bar**: Height scaled relative to max prediction, styled in primary blue (`#3b82f6`).
- **Scenario Bar**: Height scaled relative to max prediction, styled in purple (`#8b5cf6`) or emerald green (`#10b981`) depending on delta sign.
- **Delta Indicator Card**: Displays $\pm\text{CZK}$ and $\pm\%$.

---

## 7. Changed-Feature Visualization

The UI provides a dedicated **Changed Inputs Breakdown** panel:
- **Primary Changes Table**: Lists exact original vs modified values and delta for each modified primary input.
- **Empty State Handling**: If no primary inputs are changed, the UI displays *"No primary inputs modified — scenario identical to baseline."*

Example Breakdown:
| Feature | Baseline Value | Scenario Value | Delta |
| :--- | :--- | :--- | :--- |
| `ending_balance_t` | 5000.00 CZK | 15000.00 CZK | +10000.00 CZK |
| `income_credit_t` | 4000.00 CZK | 8000.00 CZK | +4000.00 CZK |

---

## 8. Derived-Feature Handling

Derived features are strictly protected across backend and frontend layers:

1. **Non-Editable**: `spending_t`, `spending_3m_mean`, and `spending_3m_std` are excluded from user input controls.
2. **Backend Validation**: `src/app/schema.py` and `src/app/simulator.py` raise validation errors if derived features are supplied in override dicts.
3. **Recalculated Derived Table**: The UI displays a dedicated read-only table showing updated values post-simulation:
   - `spending_t`: Sum of 6 category spendings.
   - `spending_3m_mean`: 3-month rolling mean.
   - `spending_3m_std`: Population standard deviation (`ddof=0`).

---

## 9. TreeSHAP Explanation Integration

Users can request a TreeSHAP explanation for either the baseline forecast or the counterfactual scenario forecast.

- **Backend Delegation**: Calls `POST /api/explain`.
- **TreeSHAP Attributions**: Returns exact TreeSHAP values relative to expected model base value ($E[f(x)] \approx 2486.29 \text{ CZK}$).
- **Non-Causal Language**: UI headers state: *"These feature attributions explain how the CatBoost model arrived at this prediction based on patterns learned during training."*

---

## 10. Error Handling & Edge Cases

The UI handles all edge cases gracefully without crashing:

| Edge Case | Backend / Frontend Defense | UI Feedback |
| :--- | :--- | :--- |
| **No Changes Made** | Simulator detects 0 diffs; delta is 0.00 CZK. | Displays "No primary inputs modified". |
| **Zero Baseline Prediction** | Division-by-zero check in delta percentage calculation. | Displays `N/A (Baseline is 0)`. |
| **Invalid / Non-numeric Inputs** | Frontend numerical validation + Pydantic schema validation. | Toast error: `"Validation Error: [field] must be a valid number"`. |
| **Derived Feature Override Attempt** | Simulator raises `ValueError`. | Toast error message rejecting derived feature edits. |
| **Server Offline / Network Failure** | Fetch `catch()` block. | Toast error: `"API Connection Error. Ensure server is running."` |
| **Internal Exceptions** | FastAPI exception handlers return HTTP 400/422/500 JSON. | Sanitized message displayed, no stack traces exposed. |

---

## 11. Testing & Verification

Integration behavior was verified with automated tests in `tests/test_integration.py` and the complete test suite:

### Test Suite Summary
- `tests/test_api.py`: 8 passed
- `tests/test_explainer.py`: 14 passed
- `tests/test_frontend.py`: 3 passed
- `tests/test_inference.py`: 7 passed
- `tests/test_integration.py`: 4 passed
- `tests/test_simulator.py`: 13 passed
- **Total**: **49 / 49 tests passed** (100% pass rate).

### Key Integration Tests (`tests/test_integration.py`)
1. `test_d12_workflow_baseline_predict_api`: Verifies baseline prediction via API server matches direct inference.
2. `test_d12_workflow_scenario_simulate_api`: Verifies counterfactual simulation API returns correct predictions, deltas, changed features, and recalculated derived values.
3. `test_d12_workflow_derived_override_rejection`: Verifies API rejects attempts to override derived features in scenario simulation.
4. `test_d12_workflow_scenario_explain_api`: Verifies TreeSHAP explanation endpoint functions for scenario states.

---

## 12. Data & Model Integrity Audit

The system integrity audit confirms:
- **Frozen CatBoost Model**: `models/catboost_model.joblib` was untouched (0 bytes modified, hash unchanged).
- **Raw/Processed Data**: `data/raw/` and `data/processed/` remain untouched.
- **14-Feature Contract**: Immutable feature order (`spending_t`, ..., `spending_other_t`) enforced across D7, D8, D9, D10, D11, and D12.
- **Single Source of Truth**: All predictions and TreeSHAP calculations route exclusively through backend CatBoost engine.

---

## 13. System Limitations

1. **Predictive Model Estimates, Not Financial Guarantees**: Model forecasts represent statistical estimates based on historical patterns in the Czech Bank dataset; they are not guaranteed future spending outcomes.
2. **Associational Scenarios, Not Causal Interventions**: Counterfactual scenario simulations reflect model behavior under shifted input states; they do not prove real-world causal impacts.
3. **Attribution, Not Causation**: TreeSHAP values measure feature importance within the CatBoost decision trees, not physical causal relationships.
4. **Frozen Model Scope**: Predictions are bound by the domain of the training dataset (Czech Bank account histories).
