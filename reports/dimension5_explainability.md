# Dimension 5 — Explainability & Feature Attribution Analysis

**Project Title:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Module:** Post-Hoc Model Explainability (TreeSHAP)  
**Evaluated Partition:** Held-Out 2018 Test Partition ($N = 53,390$ samples across 4,487 accounts)  
**Champion Model:** Retrained CatBoost Regressor ([`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib))  
**Status:** **COMPLETE**

---

## 1. Objective

The objective of **Dimension 5 — Explainability & Feature Attribution Analysis** is to open the "black box" of the final retrained CatBoost Regressor model by providing rigorous, game-theoretic, and non-causal feature attributions. While Dimension 3 and Dimension 4 established the predictive superiority and numerical error diagnostics of CatBoost ($R^2 = 0.4954$, $\text{MAE} = 729.88\text{ CZK}$, $\text{MedAE} = 377.17\text{ CZK}$ on the 2018 test set), academic research and financial digital twin applications mandate interpretable predictions. 

Specifically, Dimension 5 addresses:
1. **Global Feature Attribution**: Identifying which historical financial behaviors (lags, rolling statistics, account balance, category outflows) exert the strongest overall contribution to next-month spending forecasts.
2. **Non-Linear & Interaction Dynamics**: Uncovering how individual predictor values non-linearly modulate predictions and interact with secondary financial indicators.
3. **Local Customer Archetypes**: Explaining individual spending predictions for representative account archetypes (typical spenders, extreme high spenders, and failure-mode residual cases) using additive feature waterfall plots.

---

## 2. Explainability Methodology (TreeSHAP)

To achieve additive, fair, and mathematically consistent feature attributions, this project implements **SHAP (SHapley Additive exPlanations)** based on cooperative game theory. 

For a given feature vector $x \in \mathbb{R}^M$, the prediction $f(x)$ is decomposed into an expected base value $E[f(X)]$ plus the sum of feature attributions $\phi_j(x)$:

$$f(x) = E[f(X)] + \sum_{j=1}^{M} \phi_j(x)$$

where $\phi_j(x)$ represents the Shapley value of feature $j$ for instance $x$, satisfying four fundamental properties:
- **Efficiency**: The sum of feature contributions equals the difference between the instance prediction $f(x)$ and the base expected spending $E[f(X)]$.
- **Symmetry**: Features contributing equally to all possible feature subsets receive identical attributions.
- **Dummy (Null Effect)**: A feature that does not change the model prediction receives a Shapley value of zero ($\phi_j = 0$).
- **Additivity**: Attributions across combined models equal the sum of attributions from individual models.

Because the final selected model is a gradient-boosted decision tree ensemble (`CatBoostRegressor`), we utilize **TreeSHAP** (Lundberg et al., 2020). TreeSHAP computes exact, analytical Shapley values by evaluating conditional expectations directly over tree node paths in polynomial time $O(T L D^2)$, eliminating the sampling variance of permutation-based SHAP methods.

---

## 3. Model and Dataset Used

- **Model Artifact**: Frozen retrained CatBoost Regressor stored at [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib). The model was trained on the combined **TRAIN + VALIDATION** partition (`2013-04` through `2017-12`, $N = 117,804$) using hyperparameters: `iterations=300`, `learning_rate=0.05`, `depth=6`, `random_seed=42`.
- **Supervised Dataset**: [`data/processed/supervised_spending_dataset.csv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/processed/supervised_spending_dataset.csv) ($171,194$ total rows across 4,500 accounts).
- **Predictor Schema**: Exactly 14 verified historical predictor features ($X_{u, t}$):
  - Lags: `spending_t`, `spending_t_minus_1`, `spending_t_minus_2`
  - Rolling Volatility: `spending_3m_mean`, `spending_3m_std`
  - Balance & Inflow Activity: `ending_balance_t`, `income_credit_t`, `debit_count_t`
  - Disaggregated Outflow Categories: `spending_hh_t`, `spending_st_t`, `spending_in_t`, `spending_lo_t`, `spending_io_t`, `spending_other_t`

---

## 4. Test-Set SHAP Evaluation Methodology

SHAP attributions are computed strictly on the held-out **2018 TEST partition** ($N = 53,390$ samples, target months `2018-01` through `2018-12`).

- **Post-Hoc Analysis Only**: The 2018 test partition was **not** used to train, retrain, select hyperparameters, perform feature selection, or fit background reference distributions. The CatBoost model remained completely frozen during SHAP evaluation.
- **Generalization Audit**: Computing SHAP on out-of-time test data evaluates feature importance and attribution patterns under real-world temporal shift, preventing memorization artifacts that occur when SHAP is computed on training partitions.

---

## 5. Leakage and Temporal-Isolation Safeguards

1. **Model Isolation**: Zero training or hyperparameter adjustments were performed during Dimension 5. The model file [`models/catboost_model.joblib`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/catboost_model.joblib) remained read-only.
2. **Exact Analytical TreeSHAP**: `TreeExplainer(model)` computes exact conditional expectations from tree structures without drawing external background samples, completely eliminating reference-sample data leakage.
3. **Metadata & Identifier Exclusion**: Non-predictor columns (`account_id`, `reference_month`, `target_month`, `next_month_total_spending`) were strictly stripped prior to model inference and SHAP matrix construction.

---

## 6. SHAP Computation Details

The SHAP pipeline was executed via [`scripts/generate_shap_explanations.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/scripts/generate_shap_explanations.py).

- **Evaluated Samples**: $N = 53,390$ test observations ($100\%$ of 2018 test set).
- **SHAP Matrix Dimension**: $(53390, 14)$.
- **TreeSHAP Execution Time**: **1.97 seconds** (demonstrating computational efficiency for real-time digital twin inference).
- **Expected / Base Spending Value $E[f(X)]$**: **$1,612.1063\text{ CZK}$**.
  - *Note on Interpretation*: $E[f(X)] = 1,612.11\text{ CZK}$ represents the global mean spending prediction output by the model across the test feature space prior to adding individual feature attributions.

---

## 7. Global Feature Importance

Global feature importance is quantified by the **mean absolute SHAP value** across all 53,390 test observations:

$$\text{Importance}_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_j^{(i)}|$$

The table below summarizes the authoritative empirical ranking across all 14 predictor features:

| Rank | Predictor Feature ($X$) | Category | Mean \|SHAP\| (CZK) | Mean SHAP (CZK) | Std SHAP (CZK) | Attribution Role |
| :---: | :--- | :--- | ---: | ---: | ---: | :--- |
| **1** | `ending_balance_t` | Liquidity Buffer | **437.6909** | +72.7854 | 652.0095 | Dominant Capacity Indicator |
| **2** | `income_credit_t` | Cash Inflow | **386.1704** | -2.6916 | 530.9442 | Inflow Volatility & Scale Anchor |
| **3** | `spending_3m_mean` | Rolling Trend | **239.4737** | +3.2753 | 326.5228 | Multi-Month Baseline Spending |
| **4** | `spending_t` | Immediate Lag | **132.8945** | +3.3345 | 150.9812 | Short-Term Momentum |
| **5** | `spending_t_minus_1` | Month $t-1$ Lag | **109.3653** | +8.0068 | 130.1972 | Secondary Autoregressive Lag |
| **6** | `spending_3m_std` | Outflow Volatility | **61.1843** | -7.3071 | 80.4964 | Spending Instability Adjustment |
| **7** | `spending_hh_t` | Category Debit | **45.7229** | +2.8911 | 69.4333 | Fixed Household Expenditure |
| **8** | `spending_other_t` | Category Debit | **43.6187** | -1.2145 | 52.7479 | Discretionary Outflows |
| **9** | `debit_count_t` | Activity Level | **40.3530** | -6.0828 | 91.8734 | Transaction Frequency |
| **10** | `spending_t_minus_2` | Month $t-2$ Lag | **30.0808** | -0.4031 | 41.7386 | Tertiary Autoregressive Lag |
| **11** | `spending_st_t` | Category Debit | **17.6124** | -5.7540 | 27.8049 | Fee / Statement Outflow |
| **12** | `spending_lo_t` | Category Debit | **12.2719** | +1.8461 | 29.2575 | Fixed Loan Obligations |
| **13** | `spending_in_t` | Category Debit | **8.0946** | -0.2264 | 16.1738 | Insurance Outflow |
| **14** | `spending_io_t` | Category Debit | **0.4313** | +0.1651 | 4.1078 | Interest Outflow |

![Global Mean |SHAP| Feature Importance](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_summary_bar.png)

---

## 8. Interpretation of Top Features

1. **`ending_balance_t` (Rank 1, Mean $|SHAP| = 437.69\text{ CZK}$)**:
   - Account ending balance at month $t$ emerges as the single most influential predictor in the model.
   - *Model Attribution*: Higher liquid balances are associated with higher predicted spending. The model relies heavily on ending balance as an indicator of spending capacity.
2. **`income_credit_t` (Rank 2, Mean $|SHAP| = 386.17\text{ CZK}$)**:
   - Monthly credit inflows represent the second most critical feature.
   - *Model Attribution*: Credit inflows reflect cash flow scale. Higher credit inflows contribute positively to spending forecasts, whereas lower inflows are associated with downward attribution adjustments.
3. **`spending_3m_mean` (Rank 3, Mean $|SHAP| = 239.47\text{ CZK}$)**:
   - 3-month rolling mean expenditure serves as a key autoregressive smoothing feature.
   - *Model Attribution*: While single-month spending (`spending_t`) contains monthly variance, `spending_3m_mean` provides the model with a persistent baseline scale over quarterly horizons.

---

## 9. SHAP Beeswarm Plot Interpretation

The beeswarm plot illustrates the distribution of SHAP attributions for every test observation, combining feature ranking with feature value intensity (color) and directional impact (x-axis position).

![SHAP Beeswarm Distribution](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_summary_beeswarm.png)

### Key Model Attribution Insights from Beeswarm:
- **Asymmetric Upper-Tail Impact**: For `ending_balance_t` and `income_credit_t`, high feature values (bright red points) extend toward positive SHAP values (up to $+3,000\text{ CZK}$ to $+5,000\text{ CZK}$). This indicates that high liquidity and large inflows strongly elevate spending forecasts.
- **Lower Bounded Attribution**: Low feature values (blue points) for balance and income cluster in a bounded negative SHAP range (around $-1,000\text{ CZK}$ to $-1,500\text{ CZK}$), reflecting a lower bound on downward predictions relative to base spending.
- **Autoregressive Alignment**: High `spending_t` and `spending_3m_mean` values consistently map to positive SHAP contributions, showing that prior spending momentum contributes positively to next-month predictions.

---

## 10. Dependence-Plot Analysis

Dependence plots examine how SHAP attributions vary non-linearly as a function of feature values, with automatic color-coded secondary feature interactions.

### Rank 1 Dependence: `ending_balance_t`
![Dependence — ending_balance_t](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_dependence_rank1_ending_balance_t.png)

- **Non-Linear Trend**: For lower balances ($0$ to $20,000\text{ CZK}$), the SHAP attribution increases steeply from negative values ($-1,000\text{ CZK}$) toward positive values. For higher balance ranges, the attribution growth moderates, displaying a non-linear relationship.
- **Interaction Dynamics**: Color interaction indicates that observations with high ending balance *and* high credit inflows receive higher positive attributions.

### Rank 2 Dependence: `income_credit_t`
![Dependence — income_credit_t](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_dependence_rank2_income_credit_t.png)

- **Non-Linear Relationship**: Low credit inflows yield negative or near-zero SHAP attributions. Higher credit inflows ($>10,000\text{ CZK}$) are associated with increasingly positive SHAP contributions to next-month spending forecasts.

### Rank 3 Dependence: `spending_3m_mean`
![Dependence — spending_3m_mean](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_dependence_rank3_spending_3m_mean.png)

- **Monotonic Attribution**: Shows a consistent positive association where higher 3-month average historical spending steadily increases predicted next-month spending.

---

## 11. Local Archetype Analysis

To evaluate local interpretability, four financial account archetypes were deterministically selected from the 2018 test set using exact mathematical rules.

Residual Convention:

$$\text{residual} = \hat{Y} - Y = y_{\text{pred}} - y_{\text{true}}$$

- **Under-prediction**: $y_{\text{pred}} < y_{\text{true}} \implies \text{residual} < 0$ (negative residual).
- **Over-prediction**: $y_{\text{pred}} > y_{\text{true}} \implies \text{residual} > 0$ (positive residual).

| Archetype Name | Selection Rule | Account ID | Month | Actual $Y$ (CZK) | Forecast $\hat{Y}$ (CZK) | Residual ($\hat{Y} - Y$) |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: |
| **Typical / Median Spender** | $Y \approx \text{median}(Y)$ with $|\text{residual}| \le 100\text{ CZK}$ | #487 | 2018-01 | $1,069.86$ | $1,023.70$ | $-46.16\text{ CZK}$ |
| **High Spender** | $\max(Y)$ in Test Partition | #1029 | 2018-05 | $26,459.26$ | $5,353.59$ | $-21,105.67\text{ CZK}$ |
| **Largest Under-Prediction** | $\min(\text{residual})$ in Test Partition | #1029 | 2018-05 | $26,459.26$ | $5,353.59$ | $-21,105.67\text{ CZK}$ |
| **Largest Over-Prediction** | $\max(\text{residual})$ in Test Partition | #3516 | 2018-05 | $3.00$ | $7,090.46$ | $+7,087.46\text{ CZK}$ |

---

## 12. Typical Spender Interpretation

![Waterfall — Typical Median Spender](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_waterfall_typical_median_spender.png)

- **Account Profile**: Account #487 (January 2018). Actual spending = $1,069.86\text{ CZK}$, Predicted = $1,023.70\text{ CZK}$ (Residual = $-46.16\text{ CZK}$).
- **Observed Feature Values**: `ending_balance_t = 371.11 CZK`, `spending_3m_mean = 357.59 CZK`, `income_credit_t = 1,360.97 CZK`, `spending_t = 1,069.86 CZK`.
- **Attribution Breakdown**: Starting from base expected spending $E[f(X)] = 1,612.11\text{ CZK}$, low ending balance ($371.11\text{ CZK}$) and modest 3-month mean spending ($357.59\text{ CZK}$) contribute negative SHAP attributions, bringing the net forecast to $1,023.70\text{ CZK}$ (within $46.16\text{ CZK}$ of actual spending).
- **Digital Twin Utility**: Demonstrates accurate model alignment for routine spending profiles.

---

## 13. High-Spender / Under-Prediction Analysis

![Waterfall — High Spender / Under-Prediction](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_waterfall_high_spender.png)

> [!NOTE]
> **Methodological Coincidence**: Account #1029 (May 2018) is simultaneously the highest spending observation ($Y = 26,459.26\text{ CZK}$) and the largest under-prediction ($\text{residual} = -21,105.67\text{ CZK}$) in the test set.

- **Account Profile**: Account #1029 (May 2018). Actual spending = $26,459.26\text{ CZK}$, Predicted = $5,353.59\text{ CZK}$ (Residual = $-21,105.67\text{ CZK}$).
- **Observed Feature Values**: `ending_balance_t = 10,083.69 CZK`, `income_credit_t = 3,686.50 CZK`, `spending_t = 1,359.26 CZK`, `spending_3m_mean = 3,485.93 CZK`.
- **Attribution Breakdown**: Starting from base spending ($1,612.11\text{ CZK}$), positive attributions from ending balance and 3-month mean spending elevate the prediction to $5,353.59\text{ CZK}$.
- **Diagnostic Explanation**: In May 2018, actual spending reached $26,459.26\text{ CZK}$, whereas historical month $t$ predictors reflected moderate past expenditure (`spending_t = 1,359.26 CZK`, `spending_3m_mean = 3,485.93 CZK`). Because the model relies on historical predictors, it forecast a moderate spending level ($5,353.59\text{ CZK}$), resulting in a large negative residual under-prediction.

---

## 14. Over-Prediction Analysis

![Waterfall — Over-Prediction](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap/shap_waterfall_largest_over_prediction.png)

- **Account Profile**: Account #3516 (May 2018). Actual spending = $3.00\text{ CZK}$, Predicted = $7,090.46\text{ CZK}$ (Residual = $+7,087.46\text{ CZK}$).
- **Observed Feature Values**: `ending_balance_t = 11,338.38 CZK`, `income_credit_t = 16,346.25 CZK`, `spending_t = 20,503.00 CZK`, `spending_3m_mean = 10,222.00 CZK`.
- **Attribution Breakdown**: Base spending ($1,612.11\text{ CZK}$) is increased by positive SHAP attributions from high month $t$ spending (`spending_t = 20,503.00 CZK`), high rolling average (`spending_3m_mean = 10,222.00 CZK`), and income credit (`16,346.25 CZK`), yielding a forecast of $7,090.46\text{ CZK}$.
- **Diagnostic Explanation**: In May 2018, actual spending dropped to $3.00\text{ CZK}$. However, because historical predictors reflected high prior spending momentum, the model attributed high positive contributions to these features, producing an over-prediction.

---

## 15. Model Limitations

1. **Upper-Tail Shrinkage**: Regression loss functions (MSE/MAE) penalize extreme predictions, leading the model to pull forecasts toward conditional expectations during unexpected high-spending months.
2. **Lag Dependency on Trend Shifts**: When an account experiences a sharp drop or spike in transaction activity, autoregressive lag features (`spending_t`, `spending_3m_mean`) adjust with a delay, leading to temporary over- or under-predictions.

---

## 16. Financial Digital Twin Implications

- **Explainable Budgeting**: SHAP waterfall decompositions allow digital twins to present breakdown attributions for forecasted expenditure (e.g., illustrating how balance and historical spending contribute to the predicted spending level).
- **Interactive Feature Auditing**: Digital twin interfaces can simulate counterfactual predictor adjustments (e.g., modifying `ending_balance_t`) and observe resulting shifts in model attribution.

---

## 17. Explainability Limitations

- **Non-Causal Attribution**: SHAP values measure feature attributions within the learned model decision function $f(X)$, **not** causal financial mechanisms. Modifying a feature value in the model alters prediction attribution, but does not represent a causal real-world intervention.
- **Correlated Predictors**: Collinear features (e.g., `spending_t` and `spending_3m_mean`) share attribution credit across tree splits in the ensemble.

---

## 18. Conclusion

Dimension 5 establishes model explainability for the Personal Financial Digital Twin:
- **`ending_balance_t`**, **`income_credit_t`**, and **`spending_3m_mean`** are identified as the top 3 predictor drivers by mean absolute SHAP attribution.
- Analytical TreeSHAP on 53,390 test observations executed in **1.97 seconds**, verifying computational feasibility.
- Local waterfall attributions provide detailed instance-level explanations for both typical spending profiles and large-residual edge cases.

---

## 19. Reproducibility & Generated Artifacts

All SHAP results, plots, and summary statistics are 100% reproducible via:

```powershell
python scripts/generate_shap_explanations.py
```

### Artifact Index:
- **Script**: [`scripts/generate_shap_explanations.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/scripts/generate_shap_explanations.py)
- **Module**: [`src/models/explainability.py`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/src/models/explainability.py)
- **JSON Metrics**: [`models/shap_summary_results.json`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/models/shap_summary_results.json)
- **Global & Local Figures**: [`reports/figures/shap/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/figures/shap)
