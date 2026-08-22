# Review 1 — Dimension 3: Supervised Machine Learning Implementation & Evaluation Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 20, 2026  
**Status:** Complete & Empirical Evaluation Validated  

---

## 1. Machine Learning Objective

The primary machine learning objective of the Personal Financial Digital Twin is formulated as a **supervised tabular regression** task:

> **"Given an account's historical financial transaction behavior over past consecutive months, forecast its total outgoing expenditure for the following calendar month."**

The computational twin uses this spending forecast to project account liquidity dynamics, assist in personal budget allocation, and model future monthly financial health.

---

## 2. Target Variable Definition ($Y_{u, t+1}$)

* **Symbolic Notation:** $Y(u, t+1) \in \mathbb{R}_{\ge 0}$
* **Target Column Name:** `next_month_total_spending`
* **Mathematical Definition:** Total aggregated outgoing debit expenditure incurred by account $u$ in calendar month $t+1$:
  $$Y(u, t+1) = \sum_{k \in \text{Debits}_{u, t+1}} \text{Amount}_{u, k, \text{debit}}$$
* **Target Properties:** Non-negative continuous monetary expenditure in Czech Koruna ($\text{CZK}$).
* **Target Summary Statistics (171,194 samples):**
  * Mean: $\$1,614.58\text{ CZK}$
  * Median: $\$1,111.46\text{ CZK}$
  * Standard Deviation: $\$1,825.24\text{ CZK}$
  * Minimum: $\$0.00\text{ CZK}$
  * Maximum: $\$28,308.02\text{ CZK}$

---

## 3. Feature Selection & Taxonomy ($X(u, t)$)

The feature matrix $X(u, t)$ comprises **14 verified numerical predictors** extracted using historical transaction data available strictly on or before reference month $t$:

| # | Feature Name | Description | Temporal Window | Leakage Safe? |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `spending_t` | Total debit spending in reference month $t$ | Month $t$ | YES |
| 2 | `spending_t_minus_1` | Total debit spending in month $t-1$ | Month $t-1$ | YES |
| 3 | `spending_t_minus_2` | Total debit spending in month $t-2$ | Month $t-2$ | YES |
| 4 | `spending_3m_mean` | 3-month moving average spending ($t-2, t-1, t$) | Months $t-2 \dots t$ | YES |
| 5 | `spending_3m_std` | 3-month spending standard deviation | Months $t-2 \dots t$ | YES |
| 6 | `debit_count_t` | Total debit transaction count in month $t$ | Month $t$ | YES |
| 7 | `income_credit_t` | Total credit deposit income in month $t$ | Month $t$ | YES |
| 8 | `ending_balance_t` | Liquid ending balance after last transaction of month $t$ | Month $t$ | YES |
| 9 | `spending_hh_t` | Household category debit spending in month $t$ | Month $t$ | YES |
| 10 | `spending_st_t` | Statement fee category debit spending in month $t$ | Month $t$ | YES |
| 11 | `spending_in_t` | Insurance category debit spending in month $t$ | Month $t$ | YES |
| 12 | `spending_lo_t` | Loan repayment category debit spending in month $t$ | Month $t$ | YES |
| 13 | `spending_io_t` | Interest outward category debit spending in month $t$ | Month $t$ | YES |
| 14 | `spending_other_t` | Uncategorized debit spending in month $t$ | Month $t$ | YES |

---

## 4. Exclusion of Identifiers and Temporal Metadata

The following columns present in [`data/processed/supervised_spending_dataset.csv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/processed/supervised_spending_dataset.csv) were **explicitly excluded** from the input feature matrix $X$:

1. `account_id`: Integer entity identifier. Treating account IDs as continuous numerical features would inject arbitrary ordinal bias, while treating them as raw categorical levels would prevent model generalization to unseen accounts.
2. `reference_month`: Temporal string metadata (`YYYY-MM`). Excluded from raw numerical matrix $X$ to prevent linear models from learning spurious timestamp offsets.
3. `target_month`: Target string metadata (`YYYY-MM`). Used exclusively for chronological partition indexing.
4. `next_month_total_spending`: Target variable $Y(u, t+1)$. Isolated as the ground-truth regression label.

---

## 5. Time-Aware Supervised Panel Formulation

To map panel transaction tables into a tabular supervised regression format without data contamination:
* Each observation represents a unique `(account_id, reference_month)` pair.
* Historical predictors $X(u, t)$ aggregate transaction activity over 3 consecutive historical months ($t-2, t-1, t$).
* Target $Y(u, t+1)$ aggregates outgoing debit expenditures occurring strictly within month $t+1$.
* Only 4-consecutive-calendar-month sequences are retained ($m_{t+1} - m_{t-2} = 3$).

---

## 6. Out-of-Time Temporal Train / Validation / Test Partition

To reflect real-world financial forecasting, data is partitioned strictly **out-of-time (chronologically)** using `target_month`. Random shuffling is strictly prohibited.

```text
Full Supervised Panel (171,194 samples, 4,500 accounts)
│
├── TRAIN PARTITION        (2013-04 to 2016-12) : 71,824 samples (41.95%) | 3,252 accounts
├── VALIDATION PARTITION   (2017-01 to 2017-12) : 45,980 samples (26.86%) | 4,289 accounts
└── TEST PARTITION         (2018-01 to 2018-12) : 53,390 samples (31.19%) | 4,487 accounts
```

* **Partition Alignment Audit:**
  $$\text{Total Samples} = 71,824 + 45,980 + 53,390 = 171,194 \quad (100\% \text{ accounted for})$$
* Zero overlapping target months across partitions.
* The 2018 Test partition remained completely untouched during model development and selection.

---

## 7. Model Strategy & Candidates

Three candidate models were evaluated to establish an academically defensible performance hierarchy:

### 1. Naive Persistence Baseline
* **Deterministic Rule:** $\hat{Y}_{u, t+1} = S_{u, t}$ (`spending_t`)
* **Role:** Establishes the non-parametric domain baseline. Requires 0 learned parameters. Any ML model must outperform persistence to justify operational adoption.

### 2. Ridge Regression Pipeline (Linear Candidate)
* **Architecture:** `StandardScaler` + `Ridge(alpha=1.0, random_state=42)`
* **Role:** Regularized parametric linear baseline. Feature scaling is fitted strictly on the training partition to prevent data leakage. $L_2$ regularization manages multicollinearity between lag spending and moving averages.

### 3. CatBoost Regressor (Nonlinear Ensemble Candidate)
* **Architecture:** `CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42)`
* **Role:** High-capacity gradient-boosted decision tree ensemble. Naturally models non-linear spending interactions, categorical breakdowns, and account volatility.

---

## 8. Evaluation Metrics

Models were evaluated using four standard continuous regression metrics:

1. **Mean Absolute Error (MAE):** Average absolute deviation in CZK currency.
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE):** Quadratic error metric penalizing large forecast misses.
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$):** Proportion of spending variance explained by the model relative to a mean baseline.
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
4. **Median Absolute Error (MedAE):** Central error metric robust against non-recurring spending outliers.
   $$\text{MedAE} = \text{median}(|y_i - \hat{y}_i|)$$

---

## 9. Validation Empirical Results & Model Selection

Candidates were trained on the **TRAIN partition** (`2013-04` through `2016-12`) and evaluated on the **VALIDATION partition** (`2017-01` through `2017-12`):

| Model Name | MAE ($\text{CZK}$) | RMSE ($\text{CZK}$) | $R^2$ | MedAE ($\text{CZK}$) | Validation Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | 934.27 | 1,820.08 | 0.0215 | 396.00 | Rejected (Uninformative) |
| **Ridge Regression (Scaled)** | 763.99 | 1,323.34 | 0.4827 | 444.28 | Benchmark Passed |
| **CatBoost Regressor** | **679.11** | **1,256.17** | **0.5339** | **348.89** | **SELECTED** |

### Model Selection Rationale:
* **Validation Performance:** CatBoost achieved superior performance across all metrics ($R^2 = 0.5339$, MAE = $\$679.11\text{ CZK}$, RMSE = $\$1,256.17\text{ CZK}$, MedAE = $\$348.89\text{ CZK}$).
* **Non-linear Domain Fit:** Decision trees capture non-linear relationships between liquid balances, transaction counts, and category breakdowns without requiring manual feature transformations.
* **Test Isolation:** Model selection was finalized strictly using Validation metrics prior to touching the Test set.

---

## 10. Final Test Evaluation Results

Following model selection, the candidate pipeline was evaluated on the **2018 TEST partition** (`2018-01` through `2018-12`, **53,390 samples**):

1. **Initial Model (Trained on Train 2013–2016)** evaluated once on Test.
2. **Final Selected Candidate (Retrained on Train + Validation 2013–2017)** evaluated once on Test to utilize all permitted historical data.

| Model / Retraining Procedure | Test MAE ($\text{CZK}$) | Test RMSE ($\text{CZK}$) | Test $R^2$ | Test MedAE ($\text{CZK}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | 996.51 | 1,918.87 | -0.0643 | 418.54 |
| **Ridge Regression (Train Only)** | 867.28 | 1,425.37 | 0.4127 | 524.36 |
| **CatBoost Regressor (Train Only)** | 751.34 | 1,337.41 | 0.4830 | 402.75 |
| **Final CatBoost Regressor (Train + Val Retrained)** | **729.88** | **1,321.27** | **0.4954** | **377.17** |

---

## 11. Explicit Data Leakage Audit

> [!IMPORTANT]
> All 10 mandatory leakage checks passed verification:

1. **Target Exclusion:** PASS. `next_month_total_spending` excluded from $X$.
2. **Identifier Exclusion:** PASS. `account_id` excluded from $X$.
3. **Temporal Metadata Exclusion:** PASS. `reference_month` and `target_month` excluded from $X$.
4. **Future Contamination:** PASS. Predictors use data strictly $\le$ month $t$.
5. **No Target-Derived Predictors:** PASS. Features derived purely from debit/credit transaction histories.
6. **No Random Shuffling:** PASS. 100% out-of-time chronological partitioning.
7. **Model Selection Integrity:** PASS. Validation set (`2017`) used strictly for candidate selection.
8. **Test Set Integrity:** PASS. Test set (`2018`) evaluated after model selection completion.
9. **Scaler Fitting Isolation:** PASS. `StandardScaler` fitted strictly on training partition data.
10. **Hyperparameter Integrity:** PASS. Hyperparameters configured prior to test evaluation.

---

## 12. Negative Predictions Investigation

* **Ridge Regression (Unconstrained Linear Model):**
  * Validation: 470 negative predictions ($1.02\%$), minimum prediction $-\$665.52\text{ CZK}$.
  * Test: 527 negative predictions ($0.99\%$), minimum prediction $-\$1,346.48\text{ CZK}$.
  * **Cause:** Linear regression hyperplanes ($y = w^T x + b$) are unconstrained and output negative values for zero-spending/low-balance account edge cases.
* **CatBoost Regressor (Decision Tree Ensemble):**
  * Validation: 2 minor negative predictions (`[-116.50, -39.02]`).
  * Test: 0 negative predictions.
  * **Conclusion:** CatBoost predictions are virtually non-negative across all test samples, aligning with actual debit spending physics ($Y \ge 0$).

---

## 13. Spending Distribution & Outlier Limitations

* **MedAE vs. MAE Gap:**
  * CatBoost Test MedAE is **$\$377.17\text{ CZK}$**, whereas Test MAE is **$\$729.88\text{ CZK}$** and Test RMSE is **$\$1,321.27\text{ CZK}$**.
* **Analytical Interpretation:**
  * For over $50\%$ of account-months, spending forecasts are within $\approx \$377\text{ CZK}$ of actual expenditure.
  * Heavy upper-tail spending outliers (e.g., large single vehicle purchases, annual insurance renewals, or tuition debits) increase squared residual metrics (RMSE).

---

## 14. Reproducibility Assurance

* **Global Random Seed:** `42` enforced across Ridge and CatBoost models.
* **Serialized Artifacts:**
  * [`models/ridge_pipeline.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/ridge_pipeline.joblib)
  * [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) (Final retrained Train+Val model)
  * [`models/experiment_results.json`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/experiment_results.json)
* **Execution Script:** Re-running `python scripts/train_evaluate_models.py` reproduces all reported numbers identically.

---

## 15. Conclusion & Appropriate Scope Claims

> [!NOTE]
> **Correct Metric Interpretation:** An $R^2 \approx 0.50$ means that the final CatBoost model explains approximately half of the variance in next-month expenditure relative to a mean regression benchmark. It does **not** mean 50% prediction accuracy.
> 
> Unpredictable personal expenditure shocks (medical emergencies, unexpected repairs) introduce natural variance in real-world personal banking data. The trained CatBoost model provides a robust, leakage-safe foundation for the Personal Financial Digital Twin pipeline.
