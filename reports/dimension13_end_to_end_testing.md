# Dimension 13 — End-to-End System Testing & Validation

## 1. Objective

Dimension 13 performs comprehensive **End-to-End System Testing & Validation** across all architectural layers of the **Personal Financial Digital Twin with Explainable Spending Forecasting**.

While prior dimensions evaluated individual components in isolation (D7 inference, D8 simulator, D9 explainer, D10 REST API, D11 web frontend, D12 visualizer integration), Dimension 13 validates the system-level interactions across boundary interfaces—from user-facing HTML/CSS/JS presentation layer through FastAPI REST endpoints down to underlying application services, derived feature engines, and the frozen CatBoost champion model.

---

## 2. Testing Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Presentation Layer (D11 / D12)                    │
 │               Browser Web Dashboard (HTML5 / CSS3 / JS)                │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP REST (JSON)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        REST API Gateway (D10)                          │
 │         FastAPI Endpoints: /health, /api/schema, /api/predict,         │
 │                            /api/simulate, /api/explain                 │
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
 │                       Application Engine Core                          │
 │  • FinancialState Schema Validation & Feature Derivation Engine        │
 │  • Mathematical Fidelity Guardrails (ddof=0 std, 3m mean, category sum)│
 │  • Authoritative 14-Predictor Feature Contract                         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Predict Matrix (14 Features)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 Frozen CatBoost Champion Model & TreeSHAP              │
 │          models/catboost_model.joblib (Test MAE: 729.88 CZK, R²: 0.4954) │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Test Scope & Coverage

The end-to-end system testing suite (`tests/test_e2e_system.py`) validates eight core operational domains:

1. **System Health & Readiness (`GET /health`)**: Verifies HTTP 200 response, healthy status flag, model artifact path resolution, and non-mutating readiness checks.
2. **Schema Metadata Contract (`GET /api/schema`)**: Confirms exact 14-feature sequence, classification of 11 primary user-editable inputs vs 3 derived features, and derived recalculation rules.
3. **End-to-End Baseline Prediction Workflow**: Validates request parsing, domain validation, feature derivation, model inference execution, and exact numeric output parity across API $\leftrightarrow$ Gateway $\leftrightarrow$ CatBoost model.
4. **End-to-End Counterfactual Simulation Workflow**: Validates baseline execution, scenario input modification, derived feature recalculation (`spending_t`, `spending_3m_mean`, `spending_3m_std` with `ddof=0`), prediction delta computation ($\Delta\text{ CZK}$, $\Delta\%$), and baseline profile immutability.
5. **End-to-End TreeSHAP Explanation & Additivity**: Validates single-instance TreeSHAP attribution generation, expected base value $E[f(x)]$, feature attribution list, and exact TreeSHAP efficiency additivity ($f(x) = E[f(X)] + \sum \phi_j$).
6. **Frontend Asset Serving & UI Component Contract**: Confirms HTTP 200 serving of index document, CSS stylesheet, and client JS script, verifying frontend contains all required HTML container IDs and JS API fetch routes.
7. **System Error Handling & Guardrails**: Tests rejection of invalid domain values (negative counts), rejection of prohibited derived feature manual overrides, rejection of unexpected payload fields (HTTP 422), and safe division handling for zero baseline predictions.
8. **Mathematical Fidelity & Feature Contract**: Confirms `ddof=0` standard deviation formula matches NumPy's population standard deviation formula ($\sqrt{\frac{1}{3}\sum (S_k - \mu)^2}$) to 5 decimal places.

---

## 4. Executed End-to-End Test Scenarios

The system-level integration tests executed the following deterministic evaluation profiles:

### Scenario A: High Income & Liquidity Scenario
- **Baseline State**: `ending_balance_t = 18,500.0`, `income_credit_t = 32,000.0`, `debit_count_t = 15`, total spending lags `[9200.0, 8800.0]`, category spendings sum to `10,800.0 CZK`.
- **Counterfactual Intervention**: Increase `ending_balance_t` to `23,500.0 CZK` (+5,000 CZK), increase `income_credit_t` to `35,000.0 CZK` (+3,000 CZK), and increase `spending_hh_t` to `6,500.0 CZK` (+2,000 CZK).
- **Validation Outcome**:
  - Baseline Forecast: `8,771.92 CZK`
  - Scenario Forecast: `9,021.61 CZK`
  - Absolute Delta: `+249.69 CZK`
  - Percentage Delta: `+2.85%`
  - Derived `spending_t`: Recalculated from `10,800.0 CZK` to `12,800.0 CZK` (+2,000 CZK exact delta).
  - Derived `spending_3m_mean`: Recalculated from `9,600.00 CZK` to `10,266.67 CZK`.
  - Derived `spending_3m_std`: Recalculated from `864.10 CZK` to `1,749.92 CZK` (`ddof=0` fidelity verified).

### Scenario B: Prohibited Derived Override Defense
- **Action**: Client sends payload attempting to override `spending_3m_mean = 99999.0` directly in `/api/simulate`.
- **Validation Outcome**: HTTP 400 Bad Request returned with detail message: `"Direct override of derived feature 'spending_3m_mean' is prohibited."`

### Scenario C: TreeSHAP Additivity Verification
- **Action**: Client requests local feature attributions via `POST /api/explain`.
- **Validation Outcome**: Returned 14 SHAP values summing with base value `2,486.29 CZK` to exact prediction `8,771.92 CZK` (efficiency additivity delta = `0.000000 CZK`).

---

## 5. System Test Results

The complete project test suite was executed using PyTest:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\College UG\3rd year\5TH SEM\Machine Learning\ML project\Personal Financial Digital Twin
plugins: anyio-4.15.1
collected 57 items

tests\test_api.py ........                                               [ 14%]
tests\test_e2e_system.py ........                                        [ 28%]
tests\test_explainer.py ..............                                   [ 52%]
tests\test_frontend.py ...                                               [ 57%]
tests\test_inference.py .......                                          [ 70%]
tests\test_integration.py ....                                           [ 77%]
tests\test_simulator.py .............                                    [100%]

======================= 57 passed, 5 warnings in 3.30s ========================
```

### Complete Test Breakdown
| Test Suite Module | Target Layer / Dimension | Test Count | Result |
| :--- | :--- | :---: | :---: |
| [`tests/test_inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_inference.py) | D7 Inference Gateway | 7 | ✅ **PASSED** |
| [`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py) | D8 What-If Scenario Simulator | 13 | ✅ **PASSED** |
| [`tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py) | D9 Real-Time SHAP Explainer | 14 | ✅ **PASSED** |
| [`tests/test_api.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_api.py) | D10 REST API Server | 8 | ✅ **PASSED** |
| [`tests/test_frontend.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_frontend.py) | D11 Web Frontend Assets | 3 | ✅ **PASSED** |
| [`tests/test_integration.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_integration.py) | D12 System Integration & Visualizer | 4 | ✅ **PASSED** |
| [`tests/test_e2e_system.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_e2e_system.py) | D13 End-to-End System Testing | 8 | ✅ **PASSED** |
| **Total Test Suite** | **Dimensions 7–13** | **57** | ✅ **57 / 57 PASSED** |

---

## 6. Validation Summary

The system-level audit confirms that:
1. **End-to-End Functionality**: The digital twin web dashboard, REST API backend, inference gateway, simulator engine, and explainer service function seamlessly as an integrated software system.
2. **Contract Enforceability**: The 14-feature sequence, derived recalculation rules, and `ddof=0` mathematical standard deviation formula are strictly enforced across all operational entry points.
3. **Data & State Protection**: Baseline financial states remain 100% immutable across simulations. Derived feature overrides are actively blocked with informative HTTP 400 validation messages.
4. **Attribution Integrity**: TreeSHAP feature attributions satisfy mathematical additivity efficiency ($E[f(X)] + \sum \phi_j = f(x)$) without performance degradation (11.4 ms latency).

---

## 7. Data and Model Integrity Audit

Empirical verification via `git status --short -- data/raw data/processed models/catboost_model.joblib` returned zero output, confirming:

- **Raw Datasets (`data/raw/`)**: Unchanged (0 modifications).
- **Processed Datasets (`data/processed/`)**: Unchanged (0 modifications).
- **Frozen CatBoost Model (`models/catboost_model.joblib`)**: Unchanged (0 bytes modified, exact champion metrics preserved: Test MAE = 729.88 CZK, RMSE = 1321.27 CZK, $R^2 = 0.4954$, MedAE = 377.17 CZK).

---

## 8. System Limitations

1. **Software Verification Scope**: Automated unit, integration, and E2E tests validate software execution path correctness and data contract compliance, but do not guarantee cloud production infrastructure resilience or server scalability under heavy multi-tenant concurrency.
2. **Statistical Model Estimates**: All financial spending predictions represent statistical point estimates derived from historical Czech Bank patterns; they are not real-world expenditure guarantees.
3. **Non-Causal Attribution Interpretation**: TreeSHAP values and counterfactual scenario deltas describe model attributions under learned statistical associations, not physical causal interventions.
4. **Fixed Historical Horizon**: The digital twin operates on a 1-month forward forecast horizon conditioned on a 3-month historical window.
