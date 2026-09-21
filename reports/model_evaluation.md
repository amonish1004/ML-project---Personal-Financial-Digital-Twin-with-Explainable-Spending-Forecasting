# Model Evaluation and Performance Testing

## 1. Evaluation Objective

The objective of this report is to document the final empirical evaluation of the frozen **CatBoost** spending forecasting model for the **Personal Financial Digital Twin with Explainable Spending Forecasting** project. 

Model evaluation is conducted across chronological out-of-time partitions: **Train** (2013-04 to 2016-12), **Validation** (2017-01 to 2017-12), combined **Train + Validation** (2013-04 to 2017-12), and the final held-out **Test** partition (2018-01 to 2018-12). All evaluation results reported herein reflect the existing serialized champion model artifact ([`models/catboost_model.joblib`](models/catboost_model.joblib)).

---

## 2. Prediction Task

The predictive core of the Personal Financial Digital Twin models account-level monthly financial behavior to forecast forward expenditure:

$$\text{Task: } (u, t) \longrightarrow Y_{u, t+1}$$

Where:
- **Observation Unit:** An account $u$ at calendar reference month $t$.
- **Target Variable ($Y_{u, t+1}$):** `next_month_total_spending`, representing the total outgoing debit expenditure during calendar month $t+1$.
- **Spending Definition:** Outgoing expenditure is defined strictly using debit transactions, where `type == 'D'`. Inflow transactions (credits) and balance states are tracked separately as predictors and do not count as outgoing expenditure.

---

## 3. Model Configuration

The forecasting model utilizes the **CatBoostRegressor** algorithm with frozen hyperparameters and a fixed 14-feature input schema contract:

### Model Hyperparameters
- **Algorithm:** `CatBoostRegressor`
- **Iterations:** `300`
- **Learning Rate:** `0.05`
- **Tree Depth:** `8`
- **L2 Leaf Regularization:** `5`
- **Random Strength:** `1.0`
- **Random Seed:** `42`
- **Loss Function:** `'RMSE'`
- **Serialized Artifact:** [`models/catboost_model.joblib`](models/catboost_model.joblib)

### Frozen 14-Feature Input Contract ($X_{u, t}$)
1. `spending_t` — Total debit spending in reference month $t$
2. `spending_t_minus_1` — Total debit spending in month $t-1$
3. `spending_t_minus_2` — Total debit spending in month $t-2$
4. `spending_3m_mean` — 3-month rolling mean debit spending
5. `spending_3m_std` — 3-month rolling population standard deviation of debit spending (`ddof=0`)
6. `debit_count_t` — Total number of outgoing debit transactions in month $t$
7. `income_credit_t` — Total credit income inflow in month $t$
8. `ending_balance_t` — Ending liquid account balance at the close of month $t$
9. `spending_hh_t` — Household expense debit subtotal in month $t$
10. `spending_st_t` — Statement / fee debit subtotal in month $t$
11. `spending_in_t` — Insurance premium debit subtotal in month $t$
12. `spending_lo_t` — Loan repayment debit subtotal in month $t$
13. `spending_io_t` — Interest outflow debit subtotal in month $t$
14. `spending_other_t` — Uncategorized debit subtotal in month $t$

*Note: Metadata columns (`account_id`, `reference_month`, `target_month`) and the target variable (`next_month_total_spending`) are strictly excluded from the predictor matrix $X$.*

---

## 4. Dataset and Chronological Splits

The dataset contains **171,194 account-month observations** constructed from historical PKDD '99 / Berka Czech banking records across 4,500 accounts spanning 69 reference months.

To prevent temporal data leakage and mimic prospective real-world deployment, the dataset is partitioned **chronologically (out-of-time)** without random shuffling:

| Partition | Time Range | Account-Month Rows | Percentage | Role in Pipeline |
| :--- | :--- | ---: | ---: | :--- |
| **Train** | 2013-04 to 2016-12 | 71,824 | 41.95% | Initial model training during candidate selection |
| **Validation** | 2017-01 to 2017-12 | 45,980 | 26.86% | Hyperparameter tuning & model candidate selection |
| **Train + Validation** | 2013-04 to 2017-12 | 117,804 | 68.81% | Retraining final model before test evaluation |
| **Test** | 2018-01 to 2018-12 | 53,390 | 31.19% | Final out-of-time held-out performance evaluation |
| **Total Dataset** | 2013-04 to 2018-12 | 171,194 | 100.00% | Full supervised monthly panel |

### Importance of Chronological Splitting
Random $k$-fold cross-validation or random train-test splitting is inappropriate for financial time-series and monthly panel datasets. Random splitting causes future transaction behavior to leak into training folds, artificially inflating metrics. Chronological out-of-time partitioning guarantees that models are evaluated on future unseen calendar periods relative to their training window.

---

## 5. Evaluation Metrics

Because forecasting monthly spending is a **continuous regression task**, conventional classification accuracy (e.g., percentage of correct binary labels) is methodologically invalid. Model performance is evaluated using standard continuous regression metrics:

- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
  Measures the average absolute magnitude of prediction errors in CZK.

- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
  Measures the square root of mean squared errors in CZK. Because errors are squared before averaging, RMSE penalizes larger prediction errors more heavily than MAE.

- **Median Absolute Error (MedAE):**
  $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_N - \hat{y}_N|)$$
  Measures the median absolute prediction error in CZK. MedAE is robust to extreme spending spikes or upper-tail outliers.

- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
  Measures the proportion of variance in the target variable explained by the model relative to a naive mean-prediction baseline.

---

## 6. Final Evaluation Results

The frozen champion CatBoost model ([`models/catboost_model.joblib`](models/catboost_model.joblib)) was evaluated across all data partitions. The results are summarized below:

| Dataset | Rows | MAE (CZK) | RMSE (CZK) | $R^2$ | MedAE (CZK) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Train** | 71,824 | 592.69 | 1151.58 | 0.5857 | 290.83 |
| **Validation** | 45,980 | 654.68 | 1211.18 | 0.5667 | 332.09 |
| **Train + Validation** | 117,804 | 616.88 | 1175.20 | 0.5781 | 306.62 |
| **Test (Held-Out, Previous Production)** | 53,390 | 729.88 | 1321.27 | 0.4954 | 377.17 |
| **Test (Held-Out, Final Champion)** | **53,390** | **725.28** | **1319.25** | **0.4969** | **369.89** |

### Authoritative Final Held-Out Evaluation (2018 Test Set)
The **2018 Test partition (53,390 rows)** serves as the primary held-out benchmark for reporting final model performance. After model optimization experiments on the 2017 Validation set, the selected champion configuration (`depth=8`, `l2_leaf_reg=5`, `random_strength=1.0`) was retrained on Train + Validation and evaluated once on the 2018 Test set:

- **MAE:** $725.28\text{ CZK}$
- **RMSE:** $1319.25\text{ CZK}$
- **$R^2$:** $0.4969$
- **MedAE:** $369.89\text{ CZK}$

---

## 7. Baseline and Candidate Model Comparison

During the model development phase (Dimension 3 & 4), candidate algorithms were trained on the **Train** partition (`2013-04` to `2016-12`) and evaluated on the **Validation** partition (`2017-01` to `2017-12`) to select the champion model architecture.

| Model Candidate | Validation MAE (CZK) | Validation RMSE (CZK) | Validation $R^2$ | Validation MedAE (CZK) | Model Selection Decision |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Naive Persistence Baseline** | 934.27 | 1820.08 | 0.0215 | 396.00 | Naive Benchmark |
| **Ridge Regression (Standardized)** | 763.99 | 1323.34 | 0.4827 | 444.28 | Linear Candidate |
| **Candidate CatBoost (Train-only)** | **679.11** | **1256.17** | **0.5339** | **348.89** | **SELECTED CHAMPION** |

### Retraining & Final Test Evaluation Protocol
1. **Model Selection:** CatBoost was selected as the champion architecture based strictly on Validation $R^2$ ($0.5339$ vs Ridge $0.4827$ vs Persistence $0.0215$).
2. **Model Optimization:** Four controlled experiments were conducted on the 2017 Validation set to optimize hyperparameters. The final champion configuration (`depth=8`, `l2_leaf_reg=5`, `random_strength=1.0`) was selected based on consistent validation improvement (Val $R^2 = 0.5357$, Val MAE $= 673.58$).
3. **Retraining:** The optimized champion was retrained on combined **Train + Validation** data (117,804 rows, `2013-04` to `2017-12`).
4. **Final Held-Out Test Evaluation:** The retrained model ([`models/catboost_model.joblib`](models/catboost_model.joblib)) was evaluated **once** on the 2018 Test partition. The 2018 Test set was strictly held out and never used for hyperparameter tuning or model selection.

---

## 8. Test Performance Interpretation

The final held-out test evaluation metrics provide clear insights into the predictive behavior of the model:

- **Mean Absolute Error ($\text{MAE} = 725.28\text{ CZK}$):** On average, the model's predicted monthly expenditure deviates from actual outgoing spending by approximately 725 CZK.
- **Median Absolute Error ($\text{MedAE} = 369.89\text{ CZK}$):** For 50% of account-months in the held-out test set, the absolute prediction error is $369.89\text{ CZK}$ or less. The fact that MedAE is substantially lower than MAE indicates that typical errors are small, while a minority of accounts with high spending variance pull up the average error.
- **Root Mean Squared Error ($\text{RMSE} = 1319.25\text{ CZK}$):** The gap between RMSE ($1319.25\text{ CZK}$) and MAE ($725.28\text{ CZK}$) reflects the presence of upper-tail spending spikes (such as large one-off loan payments or major purchases), which incur larger squared penalties.
- **Coefficient of Determination ($R^2 = 0.4969$):** The model accounts for approximately **49.69% of the variance** in next-month total spending across the 2018 test set relative to a naive mean forecast.

### Critical Interpretive Guardrails
- **Not "50% Accuracy":** $R^2 = 0.4969$ indicates variance explained relative to a mean baseline, not a classification accuracy percentage.
- **No Absolute Guarantees:** Predictions represent statistical expected values under historical behavioral patterns, not guaranteed future financial outcomes.
- **Regression vs. Classification:** Model evaluation must be interpreted using CZK error magnitudes and variance metrics.

---

## 9. Tolerance-Based Results

To provide operational context for practical financial tracking, error tolerances were evaluated on the 53,390 held-out test predictions:

- **Within $\pm 500\text{ CZK}$ Tolerance:** **60.08%** of test predictions.
- **Within $\pm 1000\text{ CZK}$ Tolerance:** **80.01%** of test predictions.

*Note: These tolerance figures are supplementary diagnostic bounds indicating the proportion of forecasts falling within fixed monetary bands. They are not classification accuracy rates.*

---

## 10. Leakage and Evaluation Integrity

A comprehensive model evaluation audit was conducted to verify the mathematical and temporal integrity of the evaluation pipeline. The audit confirmed:

1. **Target Isolation:** The target variable (`next_month_total_spending`) is strictly excluded from the 14-feature predictor matrix $X$.
2. **Strict Temporal Partitioning:** Data is partitioned chronologically out-of-time without random shuffling across calendar months.
3. **Predictor Window Alignment:** All 14 predictor features are constructed using transaction records on or before reference month $t$.
4. **Held-Out Test Integrity:** The 2018 Test partition was completely isolated during feature engineering, hyperparameter tuning, and candidate model selection.
5. **No Data Contamination:** Model retraining on Train + Validation occurred prior to evaluating on Test, maintaining strict separation between training histories and test evaluation windows.
6. **Zero Target Leakage:** The model audit revealed zero evidence of lookahead bias or target leakage.

---

## 11. Limitations

When reviewing model evaluation results, the following technical and domain limitations should be considered:

- **Spending Volatility and Tail Risk:** Individual consumer spending exhibits natural stochastic variability. Large non-recurring financial events (e.g., annual insurance premiums or vehicle purchases) are difficult to anticipate from 3-month rolling aggregates alone.
- **Unexplained Variance:** An $R^2$ of approximately $0.4969$ reflects inherent unexplained variance in personal financial behavior.
- **Historical Data Context:** The model is trained and evaluated on historical PKDD '99 / Berka Czech banking records. Macroeconomic shifts, inflation, or changing banking regulations could alter baseline spending patterns.
- **Observational Nature:** Predictions reflect statistical correlations observed in historical transaction logs and should not be interpreted as financial advice or guarantees.

---

## 12. Conclusion

The final evaluation audit confirms that the serialized CatBoost champion model ([`models/catboost_model.joblib`](models/catboost_model.joblib)) delivers solid, reproducible performance for next-month spending forecasting. 

Evaluated on the held-out **2018 Test partition (53,390 rows)**, the model achieves:
- **MAE:** $725.28\text{ CZK}$
- **RMSE:** $1319.25\text{ CZK}$
- **$R^2$:** $0.4969$
- **MedAE:** $369.89\text{ CZK}$
- **Tolerance Bounds:** $60.08\%$ within $\pm 500\text{ CZK}$, $80.01\%$ within $\pm 1000\text{ CZK}$

The model outperforms both naive persistence and linear regression baselines across out-of-time evaluation partitions, providing a reliable quantitative backbone for the Personal Financial Digital Twin application layer.
