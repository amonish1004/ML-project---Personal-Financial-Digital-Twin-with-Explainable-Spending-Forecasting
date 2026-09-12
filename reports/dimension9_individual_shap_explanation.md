# Dimension 9 — Individual Real-Time SHAP Explanation Service Report

**Project Title:** Personal Financial Digital Twin with Explainable Spending Forecasting
**Stage:** Dimension 9 — Application Layer: Individual Real-Time SHAP Explanation Service
**Module:** [`src/app/explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/explainer.py)
**Test Suite:** [`tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py) (**14/14 Passed**)
**Audit Status:** Verified & Frozen

---

## 1. Executive Summary & Purpose

Dimension 9 implements the **Individual Real-Time SHAP Explanation Service** for the Personal Financial Digital Twin application.

While Dimension 5 provided offline global feature importance rankings and archetype waterfall plots across the 53,390-sample 2018 Test set partition, Dimension 9 establishes an online, single-instance Shapley attribution service capable of generating local TreeSHAP decompositions for any individual account financial state profile ($X_{u, t}$).

### Key Technical Accomplishments:
1. **Single Prediction Gateway Consistency**: `explain_prediction()` re-uses the authoritative prediction gateway `predict_spending()` from [`src/app/inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/inference.py). No competing or duplicate prediction architecture was introduced.
2. **Singleton Explainer Caching**: Implemented `get_explainer()` with a global singleton cache (`_EXPLAINER_CACHE`), avoiding redundant `shap.TreeExplainer` initialization. Empirical benchmarking measured single-instance explanation execution at approximately **11.42 ms**.
3. **Exact Additivity Verification**: Verified analytical TreeSHAP efficiency $f(x) = E[f(X)] + \sum_{j=1}^{14} \phi_j(x)$, achieving an additivity reconstruction delta of **0.000000 CZK**.
4. **Input & Derived Feature Fidelity**: Differentiates between primary user-editable inputs (matching input state) and derived features (`spending_t`, `spending_3m_mean`, `spending_3m_std`), ensuring derived features use exact population standard deviation ($\text{ddof}=0$).
5. **JSON API Compatibility**: Returns a clean, strongly typed, 100% JSON-serializable dictionary structured for future REST API (`src/app/api/server.py`) endpoints and web frontend waterfall charts.
6. **Verified Test Suite**: Created [`tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py) containing 14 automated unit tests, all of which executed and passed cleanly.

---

## 2. Architecture & Service Data Flow

```text
                           USER / API INPUT FINANCIAL STATE
                          (FinancialState or Dict[str, Any])
                                         │
                                         ▼
                             validate_financial_state()
                                         │
                                         ▼
                              derive_feature_vector()
                        (X matrix: 1 row x 14 features)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
           predict_spending()                        get_explainer()
     (Single Prediction Gateway)                   (Singleton TreeExplainer)
                   │                                           │
                   ▼                                           ▼
          point forecast (CZK)                        explainer(X)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                           SINGLE-INSTANCE SHAP ENGINE
                   ┌──────────────────────────────────────────┐
                   │ - Base Value E[f(X)]: 1,612.1063 CZK     │
                   │ - 14 Feature attributions phi_j (CZK)    │
                   │ - Additivity delta: |base + sum(phi) - y| │
                   └──────────────────────────────────────────┘
```

---

## 3. Authoritative Feature Contract & Derivation Rules

The explanation service preserves the exact 14-feature order expected by the frozen CatBoost model:

```python
EXACT_14_FEATURE_ORDER = [
    "spending_t",          # Order 1: Current month spending (Derived: sum of categories)
    "spending_t_minus_1",  # Order 2: Month t-1 spending lag (Primary Input)
    "spending_t_minus_2",  # Order 3: Month t-2 spending lag (Primary Input)
    "spending_3m_mean",    # Order 4: 3-month rolling mean (Derived: (t + t-1 + t-2)/3)
    "spending_3m_std",     # Order 5: 3-month rolling std (Derived: ddof=0 population std)
    "debit_count_t",       # Order 6: Debit count (Primary Input)
    "income_credit_t",     # Order 7: Credit income (Primary Input)
    "ending_balance_t",    # Order 8: Ending balance (Primary Input)
    "spending_hh_t",       # Order 9: Household debit (Primary Input)
    "spending_st_t",       # Order 10: Statement debit (Primary Input)
    "spending_in_t",       # Order 11: Insurance debit (Primary Input)
    "spending_lo_t",       # Order 12: Loan debit (Primary Input)
    "spending_io_t",       # Order 13: Interest debit (Primary Input)
    "spending_other_t"     # Order 14: Other debit (Primary Input)
]
```

---

## 4. Public API Specification

### Signature:
```python
def explain_prediction(
    state: Union[FinancialState, Dict[str, Any], pd.DataFrame],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
```

### Measured Execution Results (Sample Baseline State):
- **Input State**: `ending_balance_t` = 12,000 CZK, `income_credit_t` = 20,000 CZK, `debit_count_t` = 8, category spendings sum = 2,000 CZK.
- **Model Point Forecast (`prediction`)**: `6,535.61 CZK`
- **Raw CatBoost Model Forecast (`raw_prediction`)**: `6,535.6136 CZK`
- **Expected Base Spending (`base_value`)**: `1,612.1063 CZK`
- **Sum of Full-Precision SHAP Attributions**: `4,923.5073 CZK`
- **Reconstructed Prediction**: `6,535.6136 CZK`
- **Additivity Delta**: `0.000000 CZK`
- **Measured Singleton Cached Latency**: `11.42 ms`

#### Exact Instance-Specific SHAP Breakdown for Sample State:
| Feature Name | Feature Value | Instance SHAP Value ($\phi_j$) | Feature Category |
| :--- | :---: | :---: | :---: |
| `ending_balance_t` | 12,000.00 CZK | **+2,643.0817 CZK** | Liquid Balance Snapshot |
| `income_credit_t` | 20,000.00 CZK | **+2,432.4320 CZK** | Monthly Income Inflow |
| `debit_count_t` | 8 count | **-369.0257 CZK** | Debit Transaction Activity |
| `spending_hh_t` | 1,000.00 CZK | **+227.7985 CZK** | Category Debit Outflow |
| `spending_other_t` | 500.00 CZK | **+152.5331 CZK** | Category Debit Outflow |
| `spending_t` | 2,000.00 CZK | **+130.7025 CZK** | Derived Total Debit Spending |
| `spending_3m_std` | 408.25 CZK | **+121.1292 CZK** | Derived 3M Volatility ($\text{ddof}=0$) |
| `spending_st_t` | 200.00 CZK | **+2.8517 CZK** | Category Debit Outflow |
| `spending_io_t` | 0.00 CZK | **-0.2618 CZK** | Category Debit Outflow |
| `spending_t_minus_1` | 1,500.00 CZK | **-18.1623 CZK** | Historical Lag 1 Outflow |
| `spending_lo_t` | 0.00 CZK | **-27.2702 CZK** | Category Debit Outflow |
| `spending_3m_mean` | 1,500.00 CZK | **-94.4714 CZK** | Derived 3M Rolling Mean |
| `spending_t_minus_2` | 1,000.00 CZK | **-82.8358 CZK** | Historical Lag 2 Outflow |
| `spending_in_t` | 300.00 CZK | **-194.9942 CZK** | Category Debit Outflow |

---

## 5. Non-Causal Model Attribution Paradigm & D5 vs D9 Distinction

The individual SHAP explanation service strictly adheres to the non-causal interpretation paradigm:

> [!IMPORTANT]
> **Correct Model-Attribution Interpretation**:
> *"Under the trained frozen CatBoost decision tree ensemble, the account's liquid balance (`ending_balance_t` = 12,000 CZK) contributed **+2,643.0817 CZK** toward this specific month's spending forecast relative to the expected base spending $E[f(X)] = 1,612.1063\text{ CZK}$."*
>
> **Incorrect Causal Interpretation**:
> *"Increasing your ending balance to 12,000 CZK caused your next month's spending to increase by 2,643.08 CZK."*

### Critical Distinction — Global D5 SHAP Statistics vs. Instance-Specific D9 SHAP Values:
- **Dimension 5 Global SHAP Statistic**: Mean absolute SHAP value (e.g. $\text{mean}(|\text{SHAP}|) = 437.69\text{ CZK}$ for `ending_balance_t`) is a **global dataset-level importance metric** averaged across all 53,390 test set observations to rank overall model drivers.
- **Dimension 9 Instance SHAP Value**: Instance-specific TreeSHAP attribution ($\phi_j = +2,643.0817\text{ CZK}$ for `ending_balance_t` on this account state) is an **exact local attribution** calculated for a single individual profile using its unique 14-feature values. Global D5 averages must NEVER be substituted or confused with local D9 attributions.

---

## 6. Verification Test Suite Results

The explainer test suite ([`tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py)) was executed using Python 3.12. All 14 tests passed:

| Test ID | Test Description | Purpose | Result |
| :---: | :--- | :--- | :---: |
| **TEST 1** | Explainer Initialization | Verify `get_explainer()` initializes a valid `shap.TreeExplainer` | ✅ **PASS** |
| **TEST 2** | Single-Instance Execution | Verify `explain_prediction()` runs successfully on `FinancialState` & `dict` | ✅ **PASS** |
| **TEST 3** | Exact 14 Features Sequence | Verify `features` list contains 14 items matching `EXACT_14_FEATURE_ORDER` | ✅ **PASS** |
| **TEST 4** | Inference Gateway Consistency | Verify `explanation["prediction"] == predict_spending(state)` | ✅ **PASS** |
| **TEST 5** | Finiteness & Types | Verify all output numeric fields are finite floats (no `NaN`, no `Inf`) | ✅ **PASS** |
| **TEST 6** | Primary Feature Values | Verify primary inputs in explanation match input state values | ✅ **PASS** |
| **TEST 7** | Derived `spending_t` | Verify category spending sum correctly updates `spending_t` in explanation | ✅ **PASS** |
| **TEST 8** | Derived `3m_std` (`ddof=0`) | Verify `spending_3m_std` feature value uses population std (`ddof=0`) | ✅ **PASS** |
| **TEST 9** | TreeSHAP Additivity | Verify $\left|\text{base} + \sum \phi_j - \text{raw}\right| < 10^{-2}\text{ CZK}$ ($\text{Delta} = 0.000000$) | ✅ **PASS** |
| **TEST 10** | Invalid Input Rejection | Verify malformed/negative inputs raise explicit exceptions | ✅ **PASS** |
| **TEST 11** | Singleton Caching | Verify singleton explainer caching & measure latency ($\sim 11.42\text{ ms}$) | ✅ **PASS** |
| **TEST 12** | Repeatability | Verify 100% deterministic attributions across repeated calls | ✅ **PASS** |
| **TEST 13** | Model Preservation | Verify frozen CatBoost model weights remain unmutated | ✅ **PASS** |
| **TEST 14** | JSON Serialization | Verify explanation dictionary is 100% JSON-serializable | ✅ **PASS** |

---

## 7. System Regression Verification

- **Inference Suite ([`tests/test_inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_inference.py))**: **7/7 Passed**
- **Simulator Suite ([`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py))**: **13/13 Passed**
- **Explainer Suite ([`tests/test_explainer.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_explainer.py))**: **14/14 Passed**
- **Model Weight Preservation**: [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) remained unchanged.
- **Dataset Preservation**: Raw and processed datasets were untouched.
