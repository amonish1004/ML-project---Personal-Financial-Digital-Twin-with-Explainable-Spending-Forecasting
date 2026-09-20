# Personal Financial Digital Twin with Explainable Spending Forecasting

An academic machine learning and software engineering project developing a **Personal Financial Digital Twin** to model account-level transaction behavior, forecast next-month outgoing debit expenditure with explainable TreeSHAP attributions, simulate counterfactual what-if scenarios, and provide interactive financial goal planning.

---

## Project Status & Dimensions

All 14 execution dimensions (D1–D14) are **100% COMPLETE**, audited, and verified:

| Dimension | Description | Layer | Status |
| :---: | :--- | :--- | :---: |
| **D1** | Problem Formulation & Czech Bank Dataset Validation | Core ML Pipeline | ✅ **COMPLETE** |
| **D2** | Data Preprocessing, Aggregation & EDA | Core ML Pipeline | ✅ **COMPLETE** |
| **D3** | ML Model Implementation & Chronological Partitioning | Core ML Pipeline | ✅ **COMPLETE** |
| **D4** | Model Evaluation Audit & Baseline Diagnostics | Core ML Pipeline | ✅ **COMPLETE** |
| **D5** | Model Explainability & Feature Attribution (TreeSHAP) | Core ML Pipeline | ✅ **COMPLETE** |
| **D6** | Digital Twin Application Architecture & Contract | Application Layer | ✅ **COMPLETE** |
| **D7** | Reusable Model Inference Gateway & Validation | Application Layer | ✅ **COMPLETE** |
| **D8** | What-If Scenario Simulator Engine | Application Layer | ✅ **COMPLETE** |
| **D9** | Individual Real-Time SHAP Explanation Service | Application Layer | ✅ **COMPLETE** |
| **D10** | REST API Backend Application (FastAPI) | Application Layer | ✅ **COMPLETE** |
| **D11** | Interactive Web Dashboard (Dark Glassmorphism UI) | Application Layer | ✅ **COMPLETE** |
| **D12** | System Integration & Counterfactual Visualization | Application Layer | ✅ **COMPLETE** |
| **D13** | End-to-End System Testing & Validation (60/60 Passed) | Application Layer | ✅ **COMPLETE** |
| **D14** | Savings Goal Simulator & Financial Action Planner | Application Layer | ✅ **COMPLETE** |

---

## Overview

The **Personal Financial Digital Twin** combines supervised machine learning, explainable AI (XAI), and interactive scenario simulation into a unified web-based dashboard. Given an account's historical financial transaction behavior through calendar month $t$, the system forecasts total outgoing debit spending in month $t+1$, provides exact feature attributions for the prediction, and allows users to explore hypothetical spending changes.

---

## Key Features

- **Six Interactive Workspace Areas:**
  - **Twin:** Main financial state input workspace and baseline spending forecast.
  - **Explore:** Counterfactual What-If scenario simulator and comparison engine.
  - **Plan:** Interactive Savings Goal Simulator for target allocations.
  - **Insights:** Self-service TreeSHAP feature attributions and waterfall visualizations.
  - **Overview:** Executive financial dashboard and synthesized Financial Action Planner.
  - **About & Methodology:** System documentation, data provenance, and model specifications.
- **Financial Profile Presets:** One-click loading for *Default*, *High Spender*, and *Saver* account profiles.
- **Form Controls & Derived Previews:** 11 editable inputs with real-time read-only derived previews (`spending_t`, `spending_3m_mean`, `spending_3m_std` with `ddof=0`).
- **Next-Month Spending Forecast:** Machine learning forecast of outgoing debit expenditure ($t+1$).
- **What-If Scenario Simulator:** Real-time side-by-side comparison of baseline vs. scenario forecasts, showing absolute ($\Delta\text{ CZK}$) and percentage ($\Delta\%$) changes.
- **Explainable TreeSHAP Attributions:** Local feature attribution breakdown showing positive and negative drivers of individual forecasts.
- **Savings Goal Simulator (Plan):** Target saving allocation calculator (`goal_amount / timeframe_months` with final-month rounding adjustment) to evaluate monthly saving feasibility.
- **Financial Action Planner (Overview):** Synthesizes baseline forecasts, income snapshots, What-If results, and savings goals into clear financial guidance.
- **Dark Glassmorphism Interface:** Modern UI design system featuring live backend API status checks, theme toggles, and cross-tab state preservation.

---

## Machine Learning Problem Formulation

The predictive engine models account-level monthly financial behavior to forecast forward expenditure:

$$\text{Task: } (u, t) \longrightarrow Y_{u, t+1}$$

- **Target Variable ($Y_{u, t+1}$):** `next_month_total_spending` (continuous expenditure in CZK during month $t+1$).
- **Observation Unit:** Account-Month profile $(u, t)$.
- **Spending Definition:** Outgoing expenditure is defined strictly using debit transactions, where `type == 'D'`. Income credits and account balances are tracked separately as predictor inputs and do not count as outgoing expenditure.
- **Chronological Forecasting Boundary:** Features are constructed strictly on or before reference month $t$ to eliminate lookahead bias and temporal target leakage.
- **Independent Balance Input:** `ending_balance_t` is an independent model input representing the account's ending liquid balance for month $t$ and is not mathematically calculated from category spending.

---

## Dataset & Partitioning

The project uses the **PKDD '99 Czech Bank Dataset** (Berka Dataset) stored under [`data/raw/berka/`](data/raw/berka/). The raw files are preserved strictly in their original headerless TSV format.

The supervised monthly panel dataset ([`data/processed/supervised_spending_dataset.csv`](data/processed/supervised_spending_dataset.csv)) contains **171,194 observations** across **4,500 accounts** with zero null values.

Data is partitioned strictly **chronologically (out-of-time)** without random shuffling:

| Partition | Time Range | Account-Month Rows | Percentage | Role in Pipeline |
| :--- | :--- | ---: | ---: | :--- |
| **Train** | 2013-04 to 2016-12 | 71,824 | 41.95% | Initial model training during candidate selection |
| **Validation** | 2017-01 to 2017-12 | 45,980 | 26.86% | Hyperparameter tuning & model selection |
| **Train + Validation** | 2013-04 to 2017-12 | 117,804 | 68.81% | Final model retraining before test evaluation |
| **Test (Held-Out)** | 2018-01 to 2018-12 | 53,390 | 31.19% | Final held-out evaluation benchmark |

For complete feature definitions and data dictionary details, see [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

---

## Model & Feature Contract

The production model uses the **CatBoostRegressor** algorithm serialized at [`models/catboost_model.joblib`](models/catboost_model.joblib).

### Frozen Model Configuration
- **Algorithm:** `CatBoostRegressor`
- **Iterations:** `300`
- **Learning Rate:** `0.05`
- **Tree Depth:** `6`
- **Random Seed:** `42`
- **Loss Function:** `'RMSE'`

### Authoritative 14-Predictor Feature Contract ($X_{u, t}$)
1. `spending_t` — Current month total debit spending (derived)
2. `spending_t_minus_1` — Month $t-1$ debit spending lag
3. `spending_t_minus_2` — Month $t-2$ debit spending lag
4. `spending_3m_mean` — 3-month rolling mean debit spending (derived)
5. `spending_3m_std` — 3-month rolling population std (`ddof=0`, derived)
6. `debit_count_t` — Total debit transaction count
7. `income_credit_t` — Credit income inflow
8. `ending_balance_t` — Ending liquid account balance
9. `spending_hh_t` — Household expense debit subtotal
10. `spending_st_t` — Statement / fee debit subtotal
11. `spending_in_t` — Insurance premium debit subtotal
12. `spending_lo_t` — Loan repayment debit subtotal
13. `spending_io_t` — Interest outflow debit subtotal
14. `spending_other_t` — Uncategorized debit subtotal

---

## Model Performance

Because forecasting monthly spending is a continuous regression problem, model performance is evaluated using standard continuous regression metrics (**MAE**, **RMSE**, **R²**, and **MedAE**). Conventional classification accuracy is methodologically invalid for continuous expenditure targets.

### Metric Definitions
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual spending in CZK.
- **RMSE (Root Mean Squared Error):** Square root of mean squared error in CZK, heavily penalizing larger prediction errors.
- **MedAE (Median Absolute Error):** Median absolute prediction error in CZK, robust to extreme spending spikes.
- **R² (Coefficient of Determination):** Proportion of target variance explained by the model relative to a mean-prediction baseline.

### Model Selection Validation Results (2017 Validation Set)
During model candidate selection, algorithms were trained on **Train** and evaluated on **Validation**:

| Model Candidate | Validation MAE (CZK) | Validation RMSE (CZK) | Validation R² | Validation MedAE (CZK) | Selection Decision |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Naive Persistence Baseline** | 934.27 | 1820.08 | 0.0215 | 396.00 | Baseline Benchmark |
| **Ridge Regression (Scaled)** | 763.99 | 1323.34 | 0.4827 | 444.28 | Linear Candidate |
| **Candidate CatBoost (Train-only)** | **679.11** | **1256.17** | **0.5339** | **348.89** | **SELECTED CHAMPION** |

### Authoritative Final Held-Out Test Evaluation (2018 Test Partition)
After selecting CatBoost, the model was retrained on combined **Train + Validation** data (117,804 rows) and evaluated **once** on the held-out **2018 Test Partition (53,390 rows)**:

| Dataset / Retraining Strategy | Rows | MAE (CZK) | RMSE (CZK) | R² | MedAE (CZK) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Train** | 71,824 | 592.69 | 1151.58 | 0.5857 | 290.83 |
| **Validation** | 45,980 | 654.68 | 1211.18 | 0.5667 | 332.09 |
| **Train + Validation** | 117,804 | 616.88 | 1175.20 | 0.5781 | 306.62 |
| **Final Champion CatBoost (Held-Out Test)** | **53,390** | **729.88** | **1321.27** | **0.4954** | **377.17** |

- **Supplementary Error Tolerances:** **59.55%** of test forecasts fall within $\pm 500\text{ CZK}$, and **79.77%** fall within $\pm 1000\text{ CZK}$ of actual spending.

For detailed evaluation methodology, leakage audit findings, and metric interpretations, see [`reports/model_evaluation.md`](reports/model_evaluation.md).

---

## Application Architecture & Navigation

The application connects the offline machine learning model to a real-time web interface:

```text
Processed Data ➔ 14-Feature Schema ➔ Frozen CatBoost Model ➔ Reusable Gateway ➔ FastAPI Backend ➔ Web Frontend
```

### Application Workspace Navigation
1. **Twin Workspace:** Financial state input form, profile presets (*Default*, *High Spender*, *Saver*), and baseline forecast card.
2. **Explore Workspace:** What-If scenario controls, interactive baseline vs. scenario bar charts, and input delta breakdowns.
3. **Plan Workspace (Savings Goal Simulator):** Target savings goal calculator (`goal_amount / timeframe_months` with final-month adjustment) to compute required monthly saving allocations.
4. **Insights Workspace:** Real-time TreeSHAP feature attributions and horizontal waterfall visualizations.
5. **Overview Workspace:** High-level summary view and the **Financial Action Planner**, which synthesizes forecasts, income, scenario deltas, and savings goals into actionable financial recommendations.
6. **About & Methodology:** Technical documentation, feature descriptions, data provenance, and system architecture.

---

## Model Explainability (TreeSHAP)

The digital twin uses **TreeSHAP** (`TreeExplainer`) to compute local feature attributions for individual account forecasts:
- Calculates the base expected spending $E[f(X)]$ across the training distribution.
- Computes exact feature contribution values $\phi_j$ for each of the 14 predictors.
- Verifies TreeSHAP additivity efficiency ($f(x) = E[f(X)] + \sum \phi_j$) with zero additivity error.
- Highlights primary drivers such as liquid ending balance (`ending_balance_t`), credit income (`income_credit_t`), and historical rolling spending (`spending_3m_mean`).

For technical explainability reports, see [`reports/dimension5_explainability.md`](reports/dimension5_explainability.md) and [`reports/dimension9_individual_shap_explanation.md`](reports/dimension9_individual_shap_explanation.md).

---

## Savings Goal & Financial Planning Layers

The application includes two presentation and planning modules designed to help users translate model forecasts into actionable financial strategies:

- **Savings Goal Simulator (Plan Tab):** Allows users to specify a goal name, target amount (CZK), and timeframe (1–60 months) to compute required monthly savings (`goal_amount / timeframe_months` with final-month rounding adjustment).
- **Financial Action Planner (Overview Tab):** Synthesizes baseline spending forecasts, income snapshots, What-If simulation results, and active savings goals to present a structured financial plan.

*Note: The Savings Goal Simulator and Financial Action Planner are presentation and interpretation layers. They build directly upon existing model forecasts and do not introduce additional machine learning models.*

---

## System Testing & Verification

The project includes an automated test suite verifying all system layers:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

**Test Status:** **`60 passed, 5 warnings in 3.23s`**

| Test Module | Coverage Area | Status |
| :--- | :--- | :---: |
| [`tests/test_inference.py`](tests/test_inference.py) | Model loading, 14-feature contract, `ddof=0` std, determinism, input validation | ✅ **7/7 Passed** |
| [`tests/test_simulator.py`](tests/test_simulator.py) | What-If simulation engine, derived recalculations, zero-baseline safety | ✅ **13/13 Passed** |
| [`tests/test_explainer.py`](tests/test_explainer.py) | TreeSHAP explainer service, additivity verification, latency checks | ✅ **14/14 Passed** |
| [`tests/test_api.py`](tests/test_api.py) | FastAPI REST endpoints (`/health`, `/api/schema`, `/api/predict`, `/api/simulate`, `/api/explain`) | ✅ **8/8 Passed** |
| [`tests/test_frontend.py`](tests/test_frontend.py) | Static asset serving (`index.html`, `style.css`, `app.js`) and UI routing | ✅ **3/3 Passed** |
| [`tests/test_integration.py`](tests/test_integration.py) | Integrated workflow, counterfactual visualizer API parity, derived overrides | ✅ **4/4 Passed** |
| [`tests/test_e2e_system.py`](tests/test_e2e_system.py) | End-to-end system validation, boundary conditions, mathematical fidelity | ✅ **8/8 Passed** |

For the full system testing report, see [`reports/dimension13_end_to_end_testing.md`](reports/dimension13_end_to_end_testing.md).

---

## Running the Project

### Prerequisites
- **Python:** `3.12.x`
- **Virtual Environment:** `.venv`

### Application Startup Command
Launch the FastAPI backend server and static frontend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.app.api.server:app --reload
```

### Server Access URLs
- **Interactive Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **OpenAPI / Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest
```

---

## Project Structure

```text
Personal Financial Digital Twin/
├── README.md                             <-- Project Master Readme
├── requirements.txt                      <-- Python Package Dependencies
├── data/
│   ├── raw/
│   │   └── berka/                        <-- Raw Berka Dataset (Headerless TSVs)
│   └── processed/
│       ├── account_monthly_panel.csv
│       └── supervised_spending_dataset.csv  <-- Supervised Monthly Panel (171,194 rows)
│   └── DATA_DICTIONARY.md               <-- Authoritative Dataset Dictionary
├── docs/
│   └── digital_twin_architecture.md     <-- Application Architecture Contract
├── models/
│   ├── catboost_model.joblib             <-- Frozen Authoritative Champion Model
│   ├── ridge_pipeline.joblib
│   └── experiment_results.json
├── src/
│   ├── app/                              <-- Digital Twin Application Package
│   │   ├── schema.py                     <-- FinancialState & Feature Derivation Engine
│   │   ├── inference.py                  <-- Reusable Prediction Gateway (predict_spending)
│   │   ├── simulator.py                  <-- What-If Scenario Simulator (simulate_scenario)
│   │   ├── explainer.py                  <-- Real-Time TreeSHAP Service (explain_prediction)
│   │   ├── static/                       <-- Web Frontend Assets
│   │   │   ├── index.html                <-- HTML Layout Structure
│   │   │   ├── style.css                 <-- Dark Glassmorphism Design System
│   │   │   └── app.js                    <-- Client Application Script
│   │   └── api/                          <-- REST API Package
│   │       └── server.py                 <-- FastAPI Server Implementation
├── tests/                                <-- Complete System Test Suite (60/60 Passed)
│   ├── test_api.py
│   ├── test_e2e_system.py
│   ├── test_explainer.py
│   ├── test_frontend.py
│   ├── test_inference.py
│   ├── test_integration.py
│   └── test_simulator.py
└── reports/                              <-- Technical & Evaluation Reports
    ├── dimension1_dataset_validation.md
    ├── dimension2_preprocessing_and_eda.md
    ├── dimension3_ml_implementation.md
    ├── dimension4_model_evaluation.md
    ├── dimension5_explainability.md
    ├── dimension8_what_if_simulator.md
    ├── dimension9_individual_shap_explanation.md
    ├── dimension10_rest_api.md
    ├── dimension11_web_frontend.md
    ├── dimension12_system_integration.md
    ├── dimension13_end_to_end_testing.md
    └── model_evaluation.md              <-- Final Evaluation Report
```

---

## Technical Reports & References

- **Dataset Dictionary:** [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md)
- **Model Evaluation Report:** [`reports/model_evaluation.md`](reports/model_evaluation.md)
- **Baseline Model Diagnostics:** [`reports/dimension4_model_evaluation.md`](reports/dimension4_model_evaluation.md)
- **TreeSHAP Explainability Report:** [`reports/dimension5_explainability.md`](reports/dimension5_explainability.md)
- **Digital Twin Architecture Specification:** [`docs/digital_twin_architecture.md`](docs/digital_twin_architecture.md)
- **End-to-End System Testing Report:** [`reports/dimension13_end_to_end_testing.md`](reports/dimension13_end_to_end_testing.md)

---

## System Limitations

- **Model Forecast Estimates:** Forecasts are statistical model estimates based on historical transaction behavior, not deterministic guarantees or financial advice.
- **Historical Data Scope:** The model is trained and evaluated on historical PKDD '99 / Berka Czech banking records.
- **Upper-Tail Spending Volatility:** Large non-recurring expenditure spikes create higher RMSE relative to MedAE ($1321.27\text{ CZK}$ vs. $377.17\text{ CZK}$).
- **Academic Project Boundary:** Designed as an academic machine learning research and application engineering system.

---

## Project Conclusion

All core machine learning pipelines, FastAPI backend services, interactive frontend dashboards, TreeSHAP explainability services, scenario simulators, savings goal tools, automated test suites, and evaluation documentation are **100% complete, fully audited, and empirically validated**.