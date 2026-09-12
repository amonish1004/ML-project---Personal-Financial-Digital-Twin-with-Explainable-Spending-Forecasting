# Dimension 8 — What-If Scenario Simulator Engine Report

**Project Title:** Personal Financial Digital Twin with Explainable Spending Forecasting
**Stage:** Dimension 8 — Application Layer: What-If Scenario Simulator Engine
**Module:** [`src/app/simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/simulator.py)
**Test Suite:** [`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py) (**13/13 Passed**)
**Audit Status:** Verified & Frozen

---

## 1. Executive Summary & Purpose

Dimension 8 implements the **What-If Scenario Simulator Engine** for the Personal Financial Digital Twin.

The simulator allows users to perform counterfactual scenario analysis against any valid baseline financial state profile ($X_{\text{baseline}}$). By modifying one or more user-editable primary financial inputs (e.g. household debits, liquid ending balance, monthly income credit), the engine automatically recalculates all dependent features according to strict mathematical contracts, evaluates both baseline and scenario forecasts via the single authoritative prediction gateway [`predict_spending()`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/inference.py#L65), and computes signed absolute ($\Delta \hat{Y}$) and percentage changes.

### Key Technical Accomplishments:
1. **Single Prediction Gateway Re-Use**: 100% of baseline and scenario predictions route exclusively through `predict_spending()` in [`src/app/inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/app/inference.py). The simulator does NOT load models independently or invoke CatBoost directly.
2. **Derived Feature Integrity**: Derived features (`spending_t`, `spending_3m_mean`, `spending_3m_std`) are protected from manual caller overrides. Any scenario change to category spendings or historical lags automatically triggers re-derivation using population standard deviation ($\text{ddof}=0$).
3. **Baseline Immutability**: All scenario transformations operate on deep copies, leaving the caller's baseline `FinancialState` 100% unmutated and reusable for multiple independent scenarios.
4. **Zero Baseline Division Safety**: Percentage difference calculations return `None` when baseline predictions are zero or near-zero, preventing `NaN` or `Infinity` errors.
5. **Verified Test Suite**: Created [`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py) containing 13 automated unit tests, all of which executed and passed cleanly.

---

## 2. Architecture & Data Flow

```text
                               BASELINE FINANCIAL STATE
                                  (FinancialState)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
          BASELINE PATH                                 SCENARIO PATH
   (validate_financial_state)                    (apply scenario_changes to copy)
                   │                                           │
                   ▼                                           ▼
      derive_feature_vector                       derive_feature_vector
   (spending_t, 3M mean, std ddof=0)           (auto-recalculated spending_t, mean, std)
                   │                                           │
                   ▼                                           ▼
      predict_spending(baseline)                  predict_spending(scenario)
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                             COMPARISON & DELTA ENGINE
                   ┌──────────────────────────────────────────┐
                   │ absolute_difference = scenario - base    │
                   │ percentage_diff = (delta / base) * 100   │
                   └──────────────────────────────────────────┘
```

---

## 3. Input Classification & Derivation Contract

### Allowed Primary User Inputs (11 Fields)
Users and frontend controls may modify any combination of the following 11 primary input fields:
- `ending_balance_t`: Liquid balance at month $t$ (CZK)
- `income_credit_t`: Inflow/credits at month $t$ (CZK)
- `debit_count_t`: Transaction count at month $t$ ($\ge 0$)
- Category Debits: `spending_hh_t`, `spending_st_t`, `spending_in_t`, `spending_lo_t`, `spending_io_t`, `spending_other_t` (CZK $\ge 0$)
- Historical Lags: `spending_t_minus_1`, `spending_t_minus_2` (CZK $\ge 0$)

### Disallowed Derived Features (3 Fields)
Direct manual overrides of the following 3 features are strictly rejected with a `ValueError`:
1. `spending_t`: Sum of 6 category spending fields.
2. `spending_3m_mean`: 3-month rolling mean $\frac{\text{spending\_t} + \text{lag1} + \text{lag2}}{3}$.
3. `spending_3m_std`: 3-month rolling population standard deviation $\text{std}([\text{spending\_t}, \text{lag1}, \text{lag2}], \text{ddof}=0)$.

---

## 4. Public API Specification

### Signature:
```python
def simulate_scenario(
    baseline_state: Union[FinancialState, Dict[str, Any]],
    scenario_changes: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
```

### Return Dictionary Layout:
```python
{
    "baseline_state": Dict[str, Any],
    "scenario_state": Dict[str, Any],
    "baseline_prediction": float,
    "scenario_prediction": float,
    "absolute_difference": float,
    "percentage_difference": Optional[float],
}
```

---

## 5. Non-Causal Model Attribution Paradigm

The what-if simulator is framed strictly as a **model-based counterfactual attribution engine**:

> [!IMPORTANT]
> **Correct Model-Attribution Interpretation**:
> *"Under the trained frozen CatBoost model, increasing household spending (`spending_hh_t`) from 1,000 CZK to 1,500 CZK changes the predicted next-month spending from 6,535.61 CZK to 7,435.06 CZK (an increase of +899.45 CZK or +13.76%)."*
>
> **Incorrect Causal Interpretation**:
> *"Increasing your household spending causes your next month's spending to increase by 899.45 CZK."*

---

## 6. Verification Test Suite Results

The simulator test suite ([`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py)) was executed using Python 3.12. All 13 tests passed:

| Test ID | Test Description | Purpose | Result |
| :---: | :--- | :--- | :---: |
| **TEST 1** | Baseline Simulation | Verify baseline financial state produces a valid float prediction | ✅ **PASS** |
| **TEST 2** | Scenario Prediction | Verify scenario modification produces a valid float prediction | ✅ **PASS** |
| **TEST 3** | Difference Calculation | Verify `absolute_difference == scenario_pred - baseline_pred` | ✅ **PASS** |
| **TEST 4** | Percentage Calculation | Verify `percentage_difference == ((scenario - baseline)/baseline)*100` | ✅ **PASS** |
| **TEST 5** | Zero Baseline Safety | Verify zero baseline returns `percentage_difference = None` without Inf/NaN | ✅ **PASS** |
| **TEST 6** | Derived `spending_t` | Verify category change automatically updates `spending_t` | ✅ **PASS** |
| **TEST 7** | Derived 3-Month Mean | Verify scenario spending updates `spending_3m_mean` correctly | ✅ **PASS** |
| **TEST 8** | Derived 3-Month Std | Verify `spending_3m_std` strictly uses `ddof=0` population formula | ✅ **PASS** |
| **TEST 9** | Baseline Immutability | Verify original `FinancialState` remains 100% unmutated | ✅ **PASS** |
| **TEST 10** | Multiple Scenarios | Verify independent scenarios on same baseline do not contaminate each other | ✅ **PASS** |
| **TEST 11** | Invalid Field Rejection | Verify unknown/unsupported scenario fields raise `ValueError` | ✅ **PASS** |
| **TEST 12** | Derived Feature Protection | Verify direct overrides to `spending_t`, `3m_mean`, `3m_std` raise `ValueError` | ✅ **PASS** |
| **TEST 13** | Gateway Re-Use Check | Verify simulator calls `predict_spending()` via `unittest.mock.patch` | ✅ **PASS** |

---

## 7. System Regression Verification

- **Inference Suite ([`tests/test_inference.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_inference.py))**: **7/7 Passed**
- **Simulator Suite ([`tests/test_simulator.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/tests/test_simulator.py))**: **13/13 Passed**
- **Model Weight Preservation**: [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) remained unchanged.
- **Dataset Preservation**: Raw and processed datasets were untouched.
