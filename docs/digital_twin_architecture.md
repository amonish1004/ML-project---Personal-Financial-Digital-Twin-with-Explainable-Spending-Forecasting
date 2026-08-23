# Digital Twin Architecture & Inference Contract Specification

**Project Title:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Document Stage:** STEP 1 — Digital Twin Architecture & Inference Contract  
**Target Model:** Frozen CatBoost Regressor ([`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib))  
**Evaluation Status:** Dimensions 1–5 Complete ($R^2 = 0.4954$, $\text{MAE} = 729.88\text{ CZK}$, $\text{MedAE} = 377.17\text{ CZK}$)

---

## 1. System Overview & Project Status

The Personal Financial Digital Twin project models account-level transaction behavior from the PKDD '99 Czech Bank (Berka) dataset to forecast next-month total outgoing debit expenditure ($Y_{u, t+1}$).

Prior offline engineering phases have established:
- **Dimension 1 (Problem & Dataset)**: Clean definition of account-month observation profiles $(u, t)$ across 4,500 accounts ($171,194$ total rows).
- **Dimension 2 (Data Preprocessing & EDA)**: Leakage-safe monthly aggregation and feature engineering.
- **Dimension 3 (ML Implementation)**: Out-of-time chronological partitioning (Train `2013-04`..`2016-12`, Validation `2017-01`..`2017-12`, Test `2018-01`..`2018-12`). Model selection identified `CatBoostRegressor` ($R^2 = 0.5339$ on Validation) as champion over Persistence and Ridge baselines.
- **Dimension 4 (Model Evaluation Audit)**: Retrained CatBoost on combined Train+Val ($N = 117,804$) and evaluated once on the held-out 2018 Test partition ($N = 53,390$). Final test metrics: $\text{MAE} = 729.88\text{ CZK}$, $\text{RMSE} = 1321.27\text{ CZK}$, $R^2 = 0.4954$, $\text{MedAE} = 377.17\text{ CZK}$.
- **Dimension 5 (Explainability & Feature Attribution)**: Exact TreeSHAP attributions on the 2018 Test partition ($E[f(X)] = 1,612.11\text{ CZK}$, Rank 1: `ending_balance_t`, Rank 2: `income_credit_t`, Rank 3: `spending_3m_mean`).

The project is now transitioning from the completed offline research pipeline to an online, web-based **Personal Financial Digital Twin App**.

---

## 2. Offline Research vs. Online Application Boundary

A strict boundary is maintained between offline model development artifacts and the future online application layer:

```text
========================================================================================
OFFLINE RESEARCH & TRAINING PIPELINE (DIMENSIONS 1–5 — COMPLETE & FROZEN)
========================================================================================
Raw TSVs ➔ Preprocessing ➔ Monthly Panel ➔ Supervised Dataset ➔ Chronological Split 
         ➔ CatBoost Retraining ➔ Model Serialization (models/catboost_model.joblib)
         ➔ Test Audit ➔ TreeSHAP Calculation (models/shap_summary_results.json)
========================================================================================
                                     │
                             READ-ONLY ARTIFACT
                             catboost_model.joblib
                                     │
========================================================================================
ONLINE APPLICATION & DIGITAL TWIN PIPELINE (STEPS 1–8 — IN PROGRESS)
========================================================================================
User UI / Scenario Inputs ➔ Feature Construction & Validation ➔ Reusable Inference Layer
   ➔ Frozen CatBoost Model ➔ Spending Forecast (Y_hat) ➔ TreeSHAP Attributions
   ➔ Baseline vs. Scenario What-If Delta ➔ REST API ➔ Interactive Dashboard
========================================================================================
```

### Boundary Constraints:
- **No Model Retraining**: The online application MUST NOT modify, retrain, or re-fit [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib).
- **Single Inference Gate**: All online predictions (baseline predictions, SHAP attributions, what-if counterfactuals) MUST pass through a single reusable inference function.

---

## 3. Authoritative 14-Feature Inference Contract

The frozen model artifact [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) strictly requires a tabular input matrix $X \in \mathbb{R}^{N \times 14}$ with **exact feature naming and exact column ordering**.

### Feature Schema & Technical Specification:

| Order | Feature Name (`str`) | Type | Unit | Conceptual Meaning | Generation Rule / Source | Editability Classification |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **1** | `spending_t` | `float` | CZK | Total debit spending in reference month $t$ | Sum of debit transactions in month $t$ | **Derived** (Sum of category debits) |
| **2** | `spending_t_minus_1` | `float` | CZK | Total debit spending in month $t-1$ | Historical lag 1 shift of `spending_debit` | **Historical / Contextual** |
| **3** | `spending_t_minus_2` | `float` | CZK | Total debit spending in month $t-2$ | Historical lag 2 shift of `spending_debit` | **Historical / Contextual** |
| **4** | `spending_3m_mean` | `float` | CZK | 3-month rolling mean spending | `np.mean([spending_t, spending_t_minus_1, spending_t_minus_2])` | **Derived** (Auto-computed from lags) |
| **5** | `spending_3m_std` | `float` | CZK | 3-month rolling std of spending | `np.std([spending_t, spending_t_minus_1, spending_t_minus_2], ddof=0)` | **Derived** (Auto-computed from lags) |
| **6** | `debit_count_t` | `int` | Count | Count of debit transactions in month $t$ | Count of debit rows in month $t$ | **Directly User-Editable** (Activity) |
| **7** | `income_credit_t` | `float` | CZK | Total credit inflow in month $t$ | Sum of credit transactions in month $t$ | **Directly User-Editable** (Inflow) |
| **8** | `ending_balance_t` | `float` | CZK | Account ending balance at month $t$ | End-of-month balance snapshot | **Directly User-Editable** (Liquidity) |
| **9** | `spending_hh_t` | `float` | CZK | Household debit spending in month $t$ | Subtotal of debits (`k_symbol == 'HH'`) | **Directly User-Editable** (Category) |
| **10** | `spending_st_t` | `float` | CZK | Bank statement/fee debit in month $t$ | Subtotal of debits (`k_symbol == 'ST'`) | **Directly User-Editable** (Category) |
| **11** | `spending_in_t` | `float` | CZK | Insurance premium debit in month $t$ | Subtotal of debits (`k_symbol == 'IN'`) | **Directly User-Editable** (Category) |
| **12** | `spending_lo_t` | `float` | CZK | Loan repayment debit in month $t$ | Subtotal of debits (`k_symbol == 'LO'`) | **Directly User-Editable** (Category) |
| **13** | `spending_io_t` | `float` | CZK | Interest outflow debit in month $t$ | Subtotal of debits (`k_symbol == 'IO'`) | **Directly User-Editable** (Category) |
| **14** | `spending_other_t` | `float` | CZK | Uncategorized debit in month $t$ | Subtotal of uncategorized debits | **Directly User-Editable** (Category) |

> [!IMPORTANT]
> **Order Enforcement**: Feature columns passed to `model.predict()` or `TreeExplainer(model)` MUST match this exact 14-element sequence:
> `['spending_t', 'spending_t_minus_1', 'spending_t_minus_2', 'spending_3m_mean', 'spending_3m_std', 'debit_count_t', 'income_credit_t', 'ending_balance_t', 'spending_hh_t', 'spending_st_t', 'spending_in_t', 'spending_lo_t', 'spending_io_t', 'spending_other_t']`

---

## 4. Digital Twin Financial-State Definition

### Definition:
In this project, a **Personal Financial Digital Twin** is a computational representation of an individual bank account's historical spending profile $(u, t)$ that maps its 14-feature financial state vector $X_{u, t}$ to:
1. Next-month total spending forecast $\hat{Y}_{u, t+1}$ (via CatBoost Regressor).
2. Local Shapley attribution decomposition $\phi(X_{u, t})$ (via TreeSHAP).
3. Counterfactual what-if scenario comparison $\Delta \hat{Y} = \hat{Y}_{\text{scenario}} - \hat{Y}_{\text{baseline}}$.

### Non-Causal Model Attribution Paradigm:
The digital twin is a **model-based what-if simulator**, NOT a causal economic simulator.
- **Incorrect Causal Interpretation**: *"Increasing your income by 10,000 CZK causes your spending to increase."*
- **Correct Model-Attribution Interpretation**: *"Under the trained CatBoost model, increasing the `income_credit_t` input from 20,000 CZK to 30,000 CZK changes the predicted next-month spending from 2,500 CZK to 2,950 CZK, attributing a +450 CZK model change to income inflow."*

---

## 5. Feature Classification & Mathematical Consistency Rules

To prevent mathematical inconsistencies (such as editing `spending_hh_t` without updating `spending_t`), features are classified into strict input rules:

```text
                                USER UI INPUTS
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
PRIMARY STATE INPUTS         CATEGORY DEBIT INPUTS        HISTORICAL LAG INPUTS
- ending_balance_t           - spending_hh_t              - spending_t_minus_1
- income_credit_t            - spending_st_t              - spending_t_minus_2
- debit_count_t              - spending_in_t
                             - spending_lo_t
                             - spending_io_t
                             - spending_other_t
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                        AUTOMATIC DERIVATION ENGINE
        ┌───────────────────────────────────────────────────────────┐
        │ 1. spending_t = sum(category debits)                      │
        │ 2. spending_3m_mean = mean([spending_t, t-1, t-2])       │
        │ 3. spending_3m_std  = std([spending_t, t-1, t-2], ddof=0) │
        └───────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           14-FEATURE VECTOR (X)
```

### Derivation Formulas:
1. **Current Month Spending (`spending_t`)**:
   $$\text{spending\_t} = \text{spending\_hh\_t} + \text{spending\_st\_t} + \text{spending\_in\_t} + \text{spending\_lo\_t} + \text{spending\_io\_t} + \text{spending\_other\_t}$$
2. **3-Month Rolling Mean (`spending_3m_mean`)**:
   $$\text{spending\_3m\_mean} = \frac{\text{spending\_t} + \text{spending\_t\_minus\_1} + \text{spending\_t\_minus\_2}}{3.0}$$
3. **3-Month Rolling Std (`spending_3m_std`)**:
   $$\text{spending\_3m\_std} = \sqrt{\frac{1}{3} \sum_{k \in \{t, t-1, t-2\}} (\text{spending}_k - \text{spending\_3m\_mean})^2}$$

---

## 6. Proposed Application Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  FRONTEND UI                                      |
|  - Financial State Controls (Balance, Income, Category Debits, Historical Lags)   |
|  - Real-Time Spending Forecast Display (CZK)                                      |
|  - Interactive SHAP Waterfall Decomposition Chart                                 |
|  - What-If Scenario Comparison View (Baseline vs. Counterfactual Delta)           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                    HTTP JSON API
                                          |
+-----------------------------------------v-----------------------------------------+
|                                  BACKEND API                                      |
|  - Request Validation & Schema Enforcer                                           |
|  - Endpoints: /api/predict, /api/explain, /api/simulate, /api/sample-account      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                   Python Invocation
                                          |
+-----------------------------------------v-----------------------------------------+
|                        APPLICATION SERVICES (src/app/)                            |
|                                                                                   |
|  +-----------------------+   +----------------------+   +----------------------+  |
|  | Input State Validator |   | What-If Simulator    |   | SHAP Explainer       |  |
|  | - Schema checking     |   | - Counterfactuals    |   | - TreeSHAP wrapper   |  |
|  | - Auto-derivation     |   | - Delta calculation  |   | - Local attributions |  |
|  +-----------+-----------+   +----------+-----------+   +----------+-----------+  |
|              |                          |                          |              |
|              +--------------------------+--------------------------+              |
|                                         |                                         |
|                                         v                                         |
|                        +----------------------------------+                       |
|                        | Reusable Inference Core          |                       |
|                        | - Single prediction gateway      |                       |
|                        | - Feature order enforcement       |                       |
|                        +----------------+-----------------+                       |
+-----------------------------------------|-----------------------------------------+
                                          |
                                   Model Invocation
                                          |
+-----------------------------------------v-----------------------------------------+
|                       FROZEN MODEL ARTIFACT LAYER                                 |
|  - models/catboost_model.joblib (CatBoostRegressor, 300 trees, 14 features)        |
+-----------------------------------------------------------------------------------+
```

### Proposed Directory Layout for App Code (Step 2+):
```text
Personal Financial Digital Twin/
├── docs/
│   └── digital_twin_architecture.md   <-- Architecture Artifact (Step 1)
├── models/
│   ├── catboost_model.joblib           <-- Frozen Model Weight (Read-Only)
│   ├── experiment_results.json
│   └── shap_summary_results.json
├── src/
│   ├── config.py
│   ├── data/                           <-- Existing ML Pipeline (Read-Only)
│   ├── features/                       <-- Existing ML Pipeline (Read-Only)
│   ├── models/                         <-- Existing ML Pipeline (Read-Only)
│   └── app/                            <-- Application Services Package (Steps 2–5)
│       ├── __init__.py
│       ├── schema.py                   <-- Input Validation & Pydantic Data Models
│       ├── inference.py                <-- Reusable Single Inference Engine (Step 2)
│       ├── simulator.py                <-- What-If Scenario Comparison Engine (Step 3)
│       ├── explainer.py                <-- Real-Time TreeSHAP Service (Step 4)
│       └── api/                        <-- REST API Application (Step 5)
│           ├── __init__.py
│           └── server.py
└── web/                                <-- Frontend Web App (Step 6)
```

---

## 7. Future Data Flows

### A. Baseline Prediction Data Flow
$$\text{User Inputs} \xrightarrow{\text{validate}} \text{FinancialState} \xrightarrow{\text{derive}} X \in \mathbb{R}^{1 \times 14} \xrightarrow{\text{inference.predict()}} \hat{Y}_{\text{baseline}}$$

### B. Individual SHAP Explanation Data Flow
$$X \in \mathbb{R}^{1 \times 14} \xrightarrow{\text{explainer.explain()}} \{\phi_1, \dots, \phi_{14}, E[f(X)]\} \xrightarrow{\text{format}} \text{Waterfall Breakdown JSON}$$

### C. What-If Scenario Data Flow
$$\begin{cases} X_{\text{baseline}} \xrightarrow{\text{inference.predict()}} \hat{Y}_{\text{baseline}} \\ X_{\text{baseline}} \xrightarrow{\text{apply scenario edits}} X_{\text{scenario}} \xrightarrow{\text{derive & validate}} X_{\text{scenario}} \xrightarrow{\text{inference.predict()}} \hat{Y}_{\text{scenario}} \end{cases} \implies \Delta = \hat{Y}_{\text{scenario}} - \hat{Y}_{\text{baseline}}$$

---

## 8. Design Constraints and Technical Risks

1. **Strict Feature Order**: `CatBoostRegressor` does not inspect pandas column names during `predict()` if array inputs are passed; column ordering MUST be strictly validated before passing to model.
2. **Derived Feature Integrity**: Allowing users to edit `spending_t` directly while editing categories creates contradictory feature states. Automatic derivation logic must always re-calculate `spending_t`, `spending_3m_mean`, and `spending_3m_std`.
3. **Model Path Portability**: All path resolutions must use `Path(__file__).resolve()` or `PROJECT_ROOT` from `src.config` to prevent broken loads across working directories.
4. **Range & Type Validation**:
   - `ending_balance_t`: Numeric float ($\ge -100,000\text{ CZK}$).
   - `income_credit_t`: Non-negative float ($\ge 0\text{ CZK}$).
   - All category spendings: Non-negative floats ($\ge 0\text{ CZK}$).
   - `debit_count_t`: Non-negative integer ($\ge 0$).
5. **Thread Safety & Performance**: `TreeExplainer(model)` execution on a single instance takes $< 0.005$ seconds. The explainer instance should be cached in memory upon backend startup rather than re-initialized per request.

---

## 9. Module Responsibility Boundaries

| Module Layer | Primary Responsibility (SHOULD DO) | Strict Prohibition (SHOULD NOT DO) |
| :--- | :--- | :--- |
| **`schema.py`** | Validate raw input types, enforce non-negativity bounds, auto-derive `spending_t` and 3M rolling stats. | Perform model inference or call SHAP. |
| **`inference.py`** | Load frozen model, validate 14-feature column ordering, return point prediction $\hat{Y}$. | Retrain model, modify weights, or format HTTP responses. |
| **`simulator.py`** | Accept baseline state and scenario modifications, compute baseline vs scenario delta $\Delta \hat{Y}$. | Redefining inference logic outside `inference.py`. |
| **`explainer.py`** | Compute real-time TreeSHAP values $\phi_j$ and base value $E[f(X)]$ for a 14-feature vector. | Retrain model or alter feature values. |
| **`api/server.py`** | Handle HTTP routes (`/api/predict`, etc.), parse JSON requests, return JSON responses. | Implement custom feature derivation or model logic directly in route functions. |
| **`frontend/`** | Render interactive controls, draw waterfall charts, display predictions and deltas. | Contain model formulas or calculate SHAP values on client side. |

---

## 10. Planned Implementation Sequence (Steps 2–8)

- **Step 2**: Reusable Prediction / Inference Layer (`src/app/inference.py`).
- **Step 3**: What-If Simulator Engine (`src/app/simulator.py`).
- **Step 4**: SHAP Prediction Explanation Service (`src/app/explainer.py`).
- **Step 5**: Backend API Server (`src/app/api/server.py`).
- **Step 6**: Web Frontend Interface (`web/`).
- **Step 7**: End-to-End Integration & Connection.
- **Step 8**: Application Testing & Verification.

---

## 11. Explicit List of Preserved Files (MUST REMAIN UNCHANGED)

- `data/processed/supervised_spending_dataset.csv`
- `models/catboost_model.joblib`
- `models/experiment_results.json`
- `models/shap_summary_results.json`
- `src/data/*`
- `src/features/*`
- `src/models/evaluator.py`
- `src/models/models.py`
- `src/models/explainability.py`
- `reports/dimension1_dataset_validation.md` through `reports/dimension5_explainability.md`
- `scripts/run_preprocessing.py`, `scripts/build_supervised_dataset.py`, `scripts/run_eda.py`, `scripts/train_evaluate_models.py`, `scripts/generate_shap_explanations.py`
