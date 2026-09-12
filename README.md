# Personal Financial Digital Twin with Explainable Spending Forecasting

An academic machine learning and software engineering project developing a **Personal Financial Digital Twin** to model account-level transaction behavior and forecast next-month outgoing debit expenditure with explainable post-hoc attributions and counterfactual what-if simulation.

---

## Project Dimensions & Development Progress

The project is structured into two distinct execution phases:
- **Dimensions 1–5**: Core Academic ML & Research Pipeline (**COMPLETE**)
- **Dimensions 6–13**: Application Engineering & Digital Twin Deployment Layer (**IN PROGRESS**)

### Overall Dimension Status Summary

| Dimension | Description | Stage / Layer | Status |
| :---: | :--- | :--- | :---: |
| **Dimension 1** | Problem Formulation & Czech Bank Dataset Validation | Core ML Pipeline | ✅ **COMPLETE** |
| **Dimension 2** | Data Preprocessing, Aggregation & EDA | Core ML Pipeline | ✅ **COMPLETE** |
| **Dimension 3** | ML Model Implementation & Chronological Partitioning | Core ML Pipeline | ✅ **COMPLETE** |
| **Dimension 4** | Model Evaluation Audit & Performance Diagnostics | Core ML Pipeline | ✅ **COMPLETE** |
| **Dimension 5** | Model Explainability & Feature Attribution (TreeSHAP) | Core ML Pipeline | ✅ **COMPLETE** |
| **Dimension 6** | Digital Twin Application Architecture & Contract | Application Layer | ✅ **COMPLETE** |
| **Dimension 7** | Reusable Model Inference Gateway & Validation | Application Layer | ✅ **COMPLETE** |
| **Dimension 8** | What-If Scenario Simulator Engine | Application Layer | ✅ **COMPLETE** |
| **Dimension 9** | Individual Real-Time SHAP Explanation Service | Application Layer | ✅ **COMPLETE** |
| **Dimension 10** | REST API Backend Application | Application Layer | ⏳ **NEXT** |
| **Dimension 11** | Interactive Web Frontend Interface | Application Layer | ⏳ **PLANNED** |
| **Dimension 12** | System Integration & Counterfactual Visualization | Application Layer | ⏳ **PLANNED** |
| **Dimension 13** | End-to-End System Testing & Validation | Application Layer | ⏳ **PLANNED** |

---

## Core Problem Statement

Given an account's historical financial transaction behavior through calendar month $t$, predict its total outgoing debit expenditure during calendar month $t+1$:

- **Machine Learning Task:** Supervised Tabular Regression
- **Target Variable ($Y_{u, t+1}$):** `next_month_total_spending` (continuous expenditure in CZK)
- **Observation Unit:** Account-Month profile $(u, t)$
- **Forecast Horizon:** 1 month forward

---

## Dataset & Provenance

The project utilizes the **PKDD '99 Czech Bank Dataset** (Berka Dataset) stored under `data/raw/berka/`. It contains anonymized banking records for **4,500 accounts** spanning 72 calendar months (`1,056,320` raw transaction rows).

- **Date Provenance Note:** The downloaded mirror dataset contains transaction dates linearly shifted by +20 years (`2013-01` through `2018-12`). This shift preserves 100% of relative time deltas, monthly sequences, and lag structures. Detailed documentation is provided in [reports/date_provenance_check.md](reports/date_provenance_check.md).

---

## End-to-End Pipeline & Feature Taxonomy

### Pipeline Flow
```text
Raw Berka TSVs ➔ Preprocessing ➔ Monthly Account Panel ➔ Feature Engineering ➔ Supervised Dataset ➔ Chronological Split ➔ Baseline + Ridge + CatBoost ➔ Evaluation ➔ SHAP Explainability ➔ Reusable Inference Gateway
```

Dimension 2 completed raw data cleaning, monthly aggregation, leakage-safe dataset construction, and EDA. The final supervised dataset ([data/processed/supervised_spending_dataset.csv](data/processed/supervised_spending_dataset.csv)) contains **171,194 samples** across **4,500 accounts** with zero null values.

### Authoritative Predictor Feature Contract ($X_{u, t}$)
The model strictly requires exactly 14 historical predictor features constructed on or before month $t$:

```python
EXACT_14_FEATURE_ORDER = [
    "spending_t",          # Order 1:  Current month total debit spending (derived)
    "spending_t_minus_1",  # Order 2:  Month t-1 debit spending lag
    "spending_t_minus_2",  # Order 3:  Month t-2 debit spending lag
    "spending_3m_mean",    # Order 4:  3-month rolling mean spending (derived)
    "spending_3m_std",     # Order 5:  3-month rolling std spending (derived, ddof=0)
    "debit_count_t",       # Order 6:  Debit transaction count
    "income_credit_t",     # Order 7:  Credit income inflow
    "ending_balance_t",    # Order 8:  Ending liquid balance
    "spending_hh_t",       # Order 9:  Household debit subtotal
    "spending_st_t",       # Order 10: Statement/fee debit subtotal
    "spending_in_t",       # Order 11: Insurance debit subtotal
    "spending_lo_t",       # Order 12: Loan repayment debit subtotal
    "spending_io_t",       # Order 13: Interest outflow debit subtotal
    "spending_other_t"     # Order 14: Uncategorized debit subtotal
]
```

*Note: `account_id`, `reference_month`, `target_month`, and `next_month_total_spending` are metadata/targets and are strictly excluded from predictor matrix $X$.*

---

## ML Implementation & Authoritative Model Results (Dimension 3 & 4)

Dimension 3 & 4 implementation is complete and empirically validated.

### Out-of-Time Dataset Partitioning
Data is partitioned strictly **chronologically (out-of-time)** without random shuffling:
- **TRAIN (2013-04 to 2016-12):** 71,824 samples (41.95%)
- **VALIDATION (2017-01 to 2017-12):** 45,980 samples (26.86%)
- **TEST (2018-01 to 2018-12):** 53,390 samples (31.19%)
- **Total:** 171,194 samples across 4,500 accounts

### Validation Empirical Results & Model Selection
Models were trained on **TRAIN** and evaluated on **VALIDATION** for model selection:

| Model | MAE (CZK) | RMSE (CZK) | R² | MedAE (CZK) | Validation Decision |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Naive Persistence Baseline** | 934.27 | 1820.08 | 0.0215 | 396.00 | Baseline Benchmark |
| **Ridge Regression (Scaled)** | 763.99 | 1323.34 | 0.4827 | 444.28 | Passed Benchmark |
| **CatBoost Regressor** | **679.11** | **1256.17** | **0.5339** | **348.89** | **SELECTED MODEL** |

- **Selection Rationale:** **CatBoost Regressor** was selected based strictly on Validation set performance ($R^2 = 0.5339$).

### Final Test Results (Frozen Champion Model)
The selected CatBoost model was retrained on combined **Train + Validation** data (`2013-04` through `2017-12`) and evaluated once on the held-out **2018 Test Partition**:

| Model / Retraining Strategy | Test MAE (CZK) | Test RMSE (CZK) | Test R² | Test MedAE (CZK) |
| :--- | ---: | ---: | ---: | ---: |
| **Naive Persistence Baseline** | 996.51 | 1918.87 | -0.0643 | 418.54 |
| **Final CatBoost (`models/catboost_model.joblib`)** | **729.88** | **1321.27** | **0.4954** | **377.17** |

- **Model Preservation**: [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) remains the authoritative frozen champion model artifact. No retraining occurred during application engineering.

---

## Dimension 5 — Model Explainability (TreeSHAP)

Dimension 5 evaluated exact TreeSHAP attributions across all 53,390 samples in the 2018 test partition:
- **Base Expected Value $E[f(X)]$**: $1,612.11\text{ CZK}$
- **Top 3 Predictor Drivers**:
  1. `ending_balance_t` (Mean $|SHAP| = 437.69\text{ CZK}$)
  2. `income_credit_t` (Mean $|SHAP| = 386.17\text{ CZK}$)
  3. `spending_3m_mean` (Mean $|SHAP| = 239.47\text{ CZK}$)
- **Local Customer Archetypes**: Evaluated typical spender, high spender, largest under-prediction, and largest over-prediction cases via local waterfall plots.
- **Detailed Academic Report**: See [reports/dimension5_explainability.md](reports/dimension5_explainability.md).

---

## Dimension 6 — Digital Twin Application Architecture

Dimension 6 established the formal transition from offline ML modeling to online application engineering:
- **Architecture Specification Artifact**: See [docs/digital_twin_architecture.md](docs/digital_twin_architecture.md).
- **ML vs. Application Boundary**: Defined strict isolation ensuring the online web application consumes the frozen [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) without retraining or modifying training code.
- **Mathematical Feature Integrity**: Documented that `spending_3m_std` MUST explicitly use `ddof=0` ($\text{std} = \sqrt{\frac{1}{3}\sum (S_k - \mu)^2}$) to match the exact training feature generation.
- **Input Editability & Derivation Rules**: Classified features into directly user-editable state inputs (`ending_balance_t`, `income_credit_t`, `debit_count_t`), category debit inputs (`spending_hh_t`..`spending_other_t`), historical lags (`spending_t_minus_1`, `spending_t_minus_2`), and auto-derived features (`spending_t`, `spending_3m_mean`, `spending_3m_std`).
- **Non-Causal Model Attribution Paradigm**: Framed what-if simulations as model-attribution counterfactuals (*"Under the trained model, changing this input produces a different spending prediction"*), avoiding invalid real-world causal claims.

---

## Dimension 7 — Reusable Model Inference Layer

Dimension 7 implemented and verified the single, reusable prediction gateway for the digital twin application:
- **Application Services Package**: [`src/app/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/)
  - [`src/app/schema.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/schema.py): Input representation (`FinancialState`), validation rules, and feature derivation (`ddof=0` std, 3M mean, category spending summation).
  - [`src/app/inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/inference.py): Primary reusable gateway function `predict_spending()`, singleton model caching, and exact 14-column sequence enforcement.
- **Verified Test Suite**: Executed [`python tests/test_inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_inference.py) (**All 7 Tests Passed**):
  1. Model loading — **PASS**
  2. Feature contract & 14-column sequence — **PASS**
  3. Feature derivation & `ddof=0` mathematical fidelity — **PASS**
  4. Prediction execution & finiteness — **PASS**
  5. Invalid input rejection (NaN, infinity, non-numeric, negative counts) — **PASS**
  6. Determinism & repeatability — **PASS**
  7. Direct CatBoost prediction consistency — **PASS** (Gateway `231.73 CZK` == Direct CatBoost `231.73 CZK`, Difference = `0.000000`).

---

## Dimension 8 — What-If Scenario Simulator Engine

Dimension 8 implemented and verified the counterfactual scenario simulation engine:
- **Simulator Module**: [`src/app/simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/simulator.py)
  - Public function `simulate_scenario(baseline_state, scenario_changes, model_path=None)`
  - Direct re-use of prediction gateway `predict_spending()` from `src.app.inference`
  - Protection of derived features (`spending_t`, `spending_3m_mean`, `spending_3m_std` using `ddof=0`)
  - Baseline state immutability & zero-division safe percentage delta calculation
- **Verified Test Suite**: Executed [`python tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py) (**All 13 Tests Passed**):
  1. Baseline simulation — **PASS**
  2. Scenario prediction — **PASS**
  3. Absolute difference calculation — **PASS**
  4. Percentage difference calculation — **PASS**
  5. Zero baseline safety — **PASS**
  6. Derived `spending_t` recalculation — **PASS**
  7. Derived 3-month mean recalculation — **PASS**
  8. Derived 3-month std (`ddof=0`) — **PASS**
  9. Baseline state immutability — **PASS**
  10. Multiple independent scenarios — **PASS**
  11. Invalid scenario field rejection — **PASS**
  12. Derived feature override prevention — **PASS**
  13. Gateway re-use mock verification — **PASS**
- **Technical Report**: See [reports/dimension8_what_if_simulator.md](reports/dimension8_what_if_simulator.md).

---

## Dimension 9 — Individual Real-Time SHAP Explanation Service

Dimension 9 implemented and verified the online single-instance TreeSHAP attribution service:
- **Explainer Module**: [`src/app/explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/explainer.py)
  - Public function `explain_prediction(state, model_path=None)`
  - Direct re-use of prediction gateway `predict_spending()` from `src.app.inference`
  - Singleton `TreeExplainer` caching (`_EXPLAINER_CACHE`) achieving measured execution latency of **11.42 ms**
  - Verification of exact SHAP efficiency additivity ($f(x) = E[f(X)] + \sum \phi_j$, additivity delta = **0.000000 CZK**)
  - Clean, JSON-serializable dictionary structure for future REST API and frontend waterfall charts
- **Verified Test Suite**: Executed [`python tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py) (**All 14 Tests Passed**):
  1. TreeExplainer initialization — **PASS**
  2. Single-instance explanation — **PASS**
  3. Feature contract & 14-column sequence — **PASS**
  4. Inference gateway consistency — **PASS**
  5. Finiteness & numerical types — **PASS**
  6. Primary feature values fidelity — **PASS**
  7. Derived `spending_t` recalculation — **PASS**
  8. Derived 3-month std (`ddof=0`) — **PASS**
  9. TreeSHAP additivity verification — **PASS**
  10. Invalid input rejection — **PASS**
  11. Singleton explainer caching & latency — **PASS**
  12. Repeatability & determinism — **PASS**
  13. Frozen model preservation — **PASS**
  14. JSON serialization compatibility — **PASS**
- **Technical Report**: See [reports/dimension9_individual_shap_explanation.md](reports/dimension9_individual_shap_explanation.md).

---

## Project Structure

```text
Personal Financial Digital Twin/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── berka/
│   └── processed/
│       ├── account_monthly_panel.csv
│       └── supervised_spending_dataset.csv
├── docs/
│   └── digital_twin_architecture.md     <-- Dimension 6 Architecture Contract
├── models/
│   ├── catboost_model.joblib             <-- Frozen Authoritative Champion Model
│   ├── ridge_pipeline.joblib
│   ├── experiment_results.json
│   └── shap_summary_results.json
├── src/
│   ├── config.py
│   ├── data/                             <-- Core ML Pipeline (Read-Only)
│   ├── features/                         <-- Core ML Pipeline (Read-Only)
│   ├── models/                           <-- Core ML Pipeline (Read-Only)
│   │   └── explainability.py
│   └── app/                              <-- Application Layer Package (Dimension 7+)
│       ├── __init__.py
│       ├── schema.py                     <-- FinancialState & Derivation Engine
│       ├── inference.py                  <-- Reusable Prediction Gateway (predict_spending)
│       ├── simulator.py                  <-- What-If Scenario Simulator Engine (simulate_scenario)
│       └── explainer.py                  <-- Real-Time TreeSHAP Explanation Service (explain_prediction)
├── tests/
│   ├── test_inference.py                 <-- Inference Layer Test Suite (7/7 Passed)
│   ├── test_simulator.py                 <-- Simulator Test Suite (13/13 Passed)
│   └── test_explainer.py                 <-- Real-Time SHAP Explainer Test Suite (14/14 Passed)
├── scripts/
│   ├── run_preprocessing.py
│   ├── build_supervised_dataset.py
│   ├── run_eda.py
│   ├── train_evaluate_models.py
│   └── generate_shap_explanations.py
└── reports/
    ├── figures/
    │   └── shap/
    ├── date_provenance_check.md
    ├── dimension1_dataset_validation.md
    ├── dimension2_preprocessing_and_eda.md
    ├── dimension3_ml_implementation.md
    ├── dimension4_model_evaluation.md
    ├── dimension5_explainability.md
    ├── dimension8_what_if_simulator.md
    └── dimension9_individual_shap_explanation.md
```

---

## Setup & Reproduction

### Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run ML Pipeline & Verification Scripts
```powershell
# 1. Preprocess raw data & build monthly panel
python scripts/run_preprocessing.py

# 2. Construct supervised dataset & run leakage audit
python scripts/build_supervised_dataset.py

# 3. Generate EDA statistics & figures
python scripts/run_eda.py

# 4. Train, evaluate, & serialize ML models (Dimensions 3 & 4)
python scripts/train_evaluate_models.py

# 5. Calculate TreeSHAP attributions & render explainability plots (Dimension 5)
python scripts/generate_shap_explanations.py

# 6. Run Inference Layer Verification Test Suite (Dimension 7)
python tests/test_inference.py

# 7. Run What-If Simulator Test Suite (Dimension 8)
python tests/test_simulator.py

# 8. Run Individual SHAP Explainer Test Suite (Dimension 9)
python tests/test_explainer.py
```

---

## Limitations

- **Upper-Tail Outliers:** Non-recurring spending debits drive higher RMSE metrics relative to MedAE (CatBoost Test MedAE: 377.17 CZK vs MAE: 729.88 CZK, RMSE: 1321.27 CZK).
- **Unexplained Variance:** $R^2 \approx 0.50$ reflects inherent stochastic variance in individual financial spending behavior.
- **Observational Correlation:** SHAP values and digital twin counterfactuals measure model attributions based on observed historical patterns, not causal interventions.
- **Academic Scope:** Designed as an academic financial digital twin simulation; does not provide real-world credit scoring or financial advice.

---

## Future Development Roadmap

The upcoming application engineering dimensions include:
- **Dimension 10 — REST API Backend Application**: Develop lightweight Web API endpoints (`/api/predict`, `/api/explain`, `/api/simulate`).
- **Dimension 11 — Interactive Web Frontend Interface**: Develop browser-based dashboard with financial state sliders and real-time visualization.
- **Dimension 12 — System Integration & Counterfactual Visualization**: Connect frontend user controls to backend inference, simulation, and SHAP services.
- **Dimension 13 — End-to-End System Testing & Validation**: Validate complete web application performance, user interactions, and edge-case behavior.

---

## Technical Documentation & References

For comprehensive technical documentation, refer to:
- [reports/dimension1_dataset_validation.md](reports/dimension1_dataset_validation.md)
- [reports/dimension2_preprocessing_and_eda.md](reports/dimension2_preprocessing_and_eda.md)
- [reports/dimension3_ml_implementation.md](reports/dimension3_ml_implementation.md)
- [reports/dimension4_model_evaluation.md](reports/dimension4_model_evaluation.md)
- [reports/dimension5_explainability.md](reports/dimension5_explainability.md)
- [reports/dimension8_what_if_simulator.md](reports/dimension8_what_if_simulator.md)
- [reports/dimension9_individual_shap_explanation.md](reports/dimension9_individual_shap_explanation.md)
- [docs/digital_twin_architecture.md](docs/digital_twin_architecture.md)