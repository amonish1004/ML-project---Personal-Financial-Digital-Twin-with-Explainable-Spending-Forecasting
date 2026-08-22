# Dimension 4: Model Evaluation Audit & Performance Diagnostics Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 22, 2026  
**Status:** Dimension 4 Model Evaluation & Retraining Audit COMPLETE  

---

## 1. Machine Learning Objective & Task Overview

The primary predictive objective of the Personal Financial Digital Twin is formulated as an out-of-time **supervised tabular regression** task:

> **"Given an account's historical financial transaction behavior over 3 consecutive past calendar months ($t-2, t-1, t$), forecast its total outgoing debit expenditure during the following calendar month ($t+1$).**

$$\hat{Y}_{u, t+1} = f(X_{u, t})$$

* **Target Variable ($Y_{u, t+1}$):** `next_month_total_spending` (continuous non-negative expenditure in Czech Koruna - CZK).
* **Observation Unit:** Account-Month profile $(u, t)$.
* **Forecast Horizon:** 1 calendar month forward.
* **Predictor Matrix ($X_{u, t}$):** 14 verified historical financial features constructed strictly on or before month $t$.

---

## 2. Out-of-Time Temporal Partitioning & Leakage Audit

To simulate operational deployment in real-world personal banking, dataset partitioning is enforced strictly **out-of-time (chronologically)** based on `target_month`. Random data shuffling is strictly prohibited.

```text
Full Supervised Panel (171,194 samples, 4,500 accounts)
│
├── TRAIN PARTITION        (2013-04 to 2016-12) : 71,824 samples (41.95%) | 3,252 accounts
├── VALIDATION PARTITION   (2017-01 to 2017-12) : 45,980 samples (26.86%) | 4,289 accounts
└── TEST PARTITION         (2018-01 to 2018-12) : 53,390 samples (31.19%) | 4,487 accounts
```

### Comprehensive Data Leakage Audit Verification
All 10 mandatory leakage safeguards were re-audited and confirmed:

| # | Safety Rule | Enforcement Mechanism | Status |
|:---:|:---|:---|:---:|
| 1 | Target Exclusion | `next_month_total_spending` excluded from feature matrix $X$. | **PASS** |
| 2 | Identifier Exclusion | `account_id` excluded from feature matrix $X$. | **PASS** |
| 3 | Temporal Metadata Exclusion | `reference_month` and `target_month` excluded from feature matrix $X$. | **PASS** |
| 4 | Predictor Windowing | All 14 features use transaction data strictly $\le$ reference month $t$. | **PASS** |
| 5 | Out-of-Time Separation | Partitioning uses strict temporal boundaries ($2013\text{--}2016 \to 2017 \to 2018$). | **PASS** |
| 6 | Scaler Isolation | `StandardScaler` in Ridge is fitted strictly on training partition samples. | **PASS** |
| 7 | Tree Ensemble Scaling | CatBoost gradient decision trees are inherently scale-invariant. | **PASS** |
| 8 | Model Selection Integrity | Model selection was finalized on Validation data (`2017`) prior to final test scoring. | **PASS** |
| 9 | Fixed Hyperparameters | Hyperparameters (`iterations=300`, `lr=0.05`, `depth=6`, `seed=42`) fixed prior to test scoring. | **PASS** |
| 10 | Retraining Protocol | Retraining combined Train + Val (`2013-04` to `2017-12`) without touching Test inputs. | **PASS** |

---

## 3. Stage 1: Candidate Model Validation & Selection (Train $\to$ Val)

Candidate models were trained on **TRAIN** (`2013-04` to `2016-12`, 71,824 samples) and evaluated on **VALIDATION** (`2017-01` to `2017-12`, 45,980 samples) to select the champion architecture:

| Candidate Model | Validation MAE (CZK) | Validation RMSE (CZK) | Validation $R^2$ | Validation MedAE (CZK) | Selection Decision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Persistence Baseline** | 934.27 | 1,820.08 | 0.0215 | 396.00 | Non-Parametric Benchmark |
| **Ridge Regression Pipeline** | 763.99 | 1,323.34 | 0.4827 | 444.28 | Linear Benchmark Passed |
| **CatBoost Regressor** | **679.11** | **1,256.17** | **0.5339** | **348.89** | **CHAMPION SELECTED** |

### Candidate Selection Rationale:
1. **Dominant Predictive Accuracy:** CatBoost outperformed linear and persistence baselines across all continuous metrics ($R^2 = 0.5339$, MAE $= 679.11\text{ CZK}$).
2. **Non-Linear Synergy:** Decision trees naturally capture non-linear relationships between liquid balance buffers, transaction frequency, and category breakdowns without requiring manual interaction terms.

---

## 4. Stage 2: Authoritative Retrained Test Evaluation (Train+Val $\to$ Test)

Following champion selection on Validation, the selected CatBoost model configuration (`iterations=300`, `learning_rate=0.05`, `depth=6`, `random_seed=42`) was retrained on the combined **Train + Validation** partition (`2013-04` to `2017-12`, **117,804 samples**) and evaluated on the held-out **2018 TEST partition** (53,390 samples).

### Authoritative Empirical Metric Table

| Model / Strategy | Partition Evaluated | Test MAE (CZK) | Test RMSE (CZK) | Test $R^2$ | Test MedAE (CZK) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Persistence Baseline** | Test (2018) | 996.51 | 1,918.87 | -0.0643 | 418.54 |
| **Ridge Regression (Scaled)** | Test (2018) | 867.28 | 1,425.37 | 0.4127 | 524.36 |
| **CatBoost Regressor (Train Only)** | Test (2018) | 751.34 | 1,337.41 | 0.4830 | 402.75 |
| **Final CatBoost (Train+Val Retrained)** | **Test (2018)** | **729.88** | **1,321.27** | **0.4954** | **377.17** |

> [!IMPORTANT]
> **Authoritative Final Test Metrics:**  
> Retraining CatBoost on all available historical data (`117,804` samples) reduced Test MAE from **751.34 CZK** to **729.88 CZK**, reduced Test MedAE from **402.75 CZK** to **377.17 CZK**, and increased Test $R^2$ from **0.4830** to **0.4954**.

---

## 5. Residual Diagnostics & Error Distribution Analysis

To move beyond aggregate summary statistics, error distribution diagnostics were calculated on the 53,390 held-out test predictions:

* **Mean Residual (Prediction Bias):** $+60.94\text{ CZK}$  
  *(Slight positive bias indicating the model slightly over-predicts spending by $\approx 60.94\text{ CZK}$ on average across the portfolio).*
* **Residual Standard Deviation:** $1,319.87\text{ CZK}$
* **Median Absolute Error (MedAE):** $377.17\text{ CZK}$  
  *(For over 50% of account-months, predictions are within $\approx 377\text{ CZK}$ of actual expenditure).*

### Cumulative Prediction Error Tolerances

| Error Tolerance Threshold | Count within Bound | Percentage of Test Set |
|:---|:---:|:---:|
| **Within $\pm 100\text{ CZK}$** | 7,939 | **14.87%** |
| **Within $\pm 250\text{ CZK}$** | 19,754 | **37.00%** |
| **Within $\pm 500\text{ CZK}$** | 31,794 | **59.55%** |
| **Within $\pm 1000\text{ CZK}$** | 42,589 | **79.77%** |

> [!NOTE]
> **Operational Finding:** Nearly **60% of all monthly forecasts** fall within a $\pm 500\text{ CZK}$ window, and nearly **80%** fall within $\pm 1,000\text{ CZK}$.

---

## 6. Empirical Quantile Spending Tertile Performance Analysis

To evaluate model consistency across different account expenditure scales, test predictions were evaluated across **spending tertiles**.  
*Leakage-Safe Protocol: Cutoffs ($T_1 = 620.00\text{ CZK}$, $T_2 = 1,753.96\text{ CZK}$) were derived strictly from the Training partition.*

| Spending Tertile | Spending Range (CZK) | Test MAE (CZK) | Test RMSE (CZK) | Test MedAE (CZK) | Subgroup Interpretation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Low Spending ($T_1$)** | $Y \le 620.00$ | **382.16** | 658.06 | **215.05** | High relative precision; median error is only ~215 CZK. |
| **Medium Spending ($T_2$)** | $620.00 < Y \le 1,753.96$ | **586.18** | 892.33 | **395.14** | Stable mid-tier tracking; median error ~395 CZK. |
| **High Spending ($T_3$)** | $Y > 1,753.96$ | **1,244.73** | 2,028.92 | **711.64** | Driven by large non-recurring outliers (e.g. tuition, loans). |

### Key Insight on Subgroup $R^2$:
Within narrow bounded intervals like $T_1$ or $T_2$, total variance ($\sum (y_i - \bar{y})^2$) is artificially restricted, causing sub-segment $R^2$ to be negative while absolute monetary metrics (MAE / MedAE) remain excellent ($215\text{ CZK}$ MedAE for Low Spending).

---

## 7. Model Artifact Integrity & Reproducibility

All experiment metrics and retrained model weights have been serialized to disk:

* **Execution Script:** `python scripts/train_evaluate_models.py`
* **Serialized Model Weights:** [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) (335.2 KB)
* **Linear Baseline Weights:** [`models/ridge_pipeline.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/ridge_pipeline.joblib) (1.9 KB)
* **Structured Experiment Record:** [`models/experiment_results.json`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/experiment_results.json)

---

## 8. Summary Table of Dimension 4 Audit Accomplishments

| Metric / Audit Check | Status / Value |
|:---|:---|
| **Champion Model Architecture** | **CatBoost Regressor (`iterations=300`, `lr=0.05`, `depth=6`)** |
| **Retraining Partition Size** | **117,804 samples (`2013-04` through `2017-12`)** |
| **Test Partition Size** | **53,390 samples (`2018-01` through `2018-12`)** |
| **Authoritative Test MAE** | **$729.88\text{ CZK}$** |
| **Authoritative Test RMSE** | **$1,321.27\text{ CZK}$** |
| **Authoritative Test $R^2$** | **$0.4954$** |
| **Authoritative Test MedAE** | **$377.17\text{ CZK}$** |
| **Predictions within $\pm 500\text{ CZK}$** | **$59.55\%$** |
| **Predictions within $\pm 1,000\text{ CZK}$** | **$79.77\%$** |
| **Code & JSON Artifact Sync** | **100% Synchronized & Reproducible** |
