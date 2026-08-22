# Personal Financial Digital Twin with Explainable Spending Forecasting

---

## 1. Project Overview

This academic project develops a **Personal Financial Digital Twin**—a computational model designed to represent an individual account's financial transaction dynamics and forecast next-month total expenditure using explainable machine learning.

**Project Status:** Initial Setup & Problem Formulation Phase (Review 1 - Dimension 1).

*Note: Planned features and pipeline components are explicitly documented in their respective sections below and will be populated as each phase is completed and empirically verified.*

---

## 2. Problem Statement

The intended primary machine learning problem is defined as:

> **"Given an account's historical financial behavior over a sequence of months, predict its total expenditure for the following month."**

- **ML Task Type:** Supervised Learning (Tabular Regression)
- **Target Variable:** Numerical total monthly debit expenditure ($Y_{u, t+1} \in \mathbb{R}_{\ge 0}$)
- **Observation Unit:** Account-Month profile ($u, t$)

*Note: The exact target scope, temporal windows, and feature boundary conditions will be finalized following physical data download and schema verification.*

---

## 3. Project Objectives

### Planned Objectives
1. **Financial Transaction Data Acquisition:** Acquire a multi-month panel financial transaction dataset from a reputable public academic source.
2. **Data Preprocessing & Cleaning:** Implement a modular pipeline for date parsing, transaction direction filtering (debits vs. credits), and category normalization.
3. **Monthly Aggregation & Feature Engineering:** Construct account-month aggregated profiles and extract historical lag spending, rolling statistics, and transaction velocity indicators.
4. **Exploratory Data Analysis (EDA):** Conduct empirical statistical checks on spending distributions, temporal stability, and outlier characteristics.
5. **Supervised Regression Modeling:** Implement baseline statistical models and standard open-source ML regression algorithms.
6. **Model Evaluation:** Evaluate model accuracy using standard continuous error metrics (MAE, RMSE, $R^2$).
7. **Explainability:** Apply post-hoc model interpretability techniques (e.g., feature importance, SHAP) to explain spending predictions.
8. **Reproducibility:** Ensure complete script-driven pipeline execution, version-controlled dependency management, and reproducible random seeding.

### Completed Work
- [x] Initial project repository setup and directory structure initialization.
- [x] Candidate dataset research and comparative source evaluation.
- [x] Provisionally selected Candidate 1 (PKDD '99 Czech Bank Dataset / Berka Dataset) pending physical download and verification.

---

## 4. Dataset

*Status: Awaiting physical dataset download and schema verification.*

Detailed empirical dataset statistics (such as total row counts, column data types, missing value ratios, unique account counts, temporal date ranges, and file sizes) will be computed directly from downloaded raw data files and populated in this section.

---

## 5. Dataset Provenance & Reproducibility

- **Original Source:** Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE), Czech Republic (PKDD '99 Discovery Challenge).
- **Download Source:** Pending physical download from official VŠE / public academic mirror.
- **Download Date:** *To be recorded upon download*
- **Dataset Version:** Original 1999 Release
- **Files Included:** `trans.asc`, `account.asc`, `client.asc`, `disp.asc`, `order.asc`, `loan.asc`, `card.asc`, `district.asc`
- **File Sizes & Cryptographic Hashes (SHA-256):** *To be computed upon raw file acquisition*
- **Access / Usage Terms:** Open Academic Research Use (Public Domain / Academic Benchmark)
- **Citation:**
  > Berka, P., & Bruha, E. (1999). *PKDD '99 Discovery Challenge Financial Dataset*. Faculty of Informatics and Statistics, VŠE, Czech Republic.

---

## 6. ML Problem Formulation

### Proposed Formulation (Supervised Tabular Regression)

- **Prediction Unit:** Individual Account ($u$) at calendar month $t$
- **Input Features ($X_{u, t}$):** Engineered historical indicators extracted from months $t, t-1, t-2, \dots$
- **Target Variable ($Y_{u, t+1}$):** Total aggregated debit expenditure incurred by account $u$ in calendar month $t+1$:
  $$Y_{u, t+1} = \sum_{k \in \text{Debits}_{u, t+1}} \text{Amount}_{u, k}$$
- **Forecasting Window:** 1-month forward sliding horizon
- **Regression Justification:** Total expenditure is a continuous, non-negative monetary metric, making supervised regression the mathematically appropriate formulation.

*Note: Final claims regarding sample counts and exact lag depths depend on physical dataset verification.*

---

## 7. Data Preprocessing

*Status: Pipeline design phase. No preprocessing scripts have been executed yet.*

### Planned Preprocessing Steps
1. **Data Loading:** Memory-efficient loading of raw ASCII transaction tables.
2. **Schema & Data Type Validation:** Explicit type casting of identifiers, timestamps, and numerical amounts.
3. **Missing Value & Duplicate Audit:** Empirical detection and systematic handling of null entries and duplicate transactions.
4. **Date Parsing:** Standardizing date fields into UTC timestamp objects.
5. **Transaction Type Mapping:** Explicitly categorizing transactions into debit expenditures vs. credit deposits.
6. **Account Filtering:** Retaining accounts with sufficient consecutive monthly activity to support lag creation.
7. **Monthly Panel Aggregation:** Grouping transaction records by `account_id` and calendar year-month.
8. **Feature & Target Construction:** Shifting temporal windows to create inputs ($X_{u, t}$) and ground-truth targets ($Y_{u, t+1}$).

---

## 8. Exploratory Data Analysis

*Status: Pending dataset acquisition.*

This section will house empirical statistical summary tables and visualization plots generated during EDA:
- Transaction amount distribution plots (skewness, log-transforms)
- Monthly total expenditure distribution across accounts
- Category-wise spending proportions
- Longitudinal transaction volume trends over time
- Missing value heatmaps and correlation matrices

---

## 9. Feature Engineering

*Status: Planned candidates under evaluation.*

### Candidate Feature Taxonomy
- **Lagged Spending:** $S_{u, t}, S_{u, t-1}, S_{u, t-2}$ (total spending in past 1, 2, and 3 months)
- **Rolling Expenditure Statistics:** 3-month moving average $\mu_{u, t}$, 3-month standard deviation $\sigma_{u, t}$ (spending volatility)
- **Transaction Velocity:** Monthly transaction count $N_{u, t}$, average transaction size $\bar{A}_{u, t}$
- **Income & Net Flow:** Total credit deposits $I_{u, t}$, net savings ratio $\frac{I_{u, t} - S_{u, t}}{I_{u, t}}$
- **Category Ratios:** Share of total expenditure allocated to mandatory vs. discretionary categories
- **Calendar Signals:** Month of year, quarter index

---

## 10. ML Methodology

*Status: Planned methodology.*

- **Data Splitting Strategy:** Strict Out-of-Time (Temporal) Train/Validation/Test splitting to prevent data leakage across temporal boundaries.
- **Baseline Models:** Dummy Mean Regressor, Historical Lag-1 Baseline ($Y_{u, t+1} \approx S_{u, t}$).
- **Candidate Supervised Regressors:**
  - Linear Models: Ridge / Lasso Regression
  - Tree-Based Ensembles: Random Forest Regressor, Gradient Boosting / XGBoost
- **Leakage Prevention:** All feature transformations (scaling, imputation, rolling windows) will be fitted strictly on the training partition.
- **Hyperparameter Optimization:** Time-series cross-validation on the training set.

---

## 11. Model Evaluation

*Status: Pending model implementation.*

Models will be evaluated using standard continuous regression metrics calculated on the test dataset:

- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

*Numerical evaluation results will be inserted here following experimental execution.*

---

## 12. Explainability

*Status: Planned implementation.*

To fulfill the "Explainable" requirement of the Financial Digital Twin, post-hoc interpretability methods will be integrated:
- Global feature importance analysis (tree-based impurity / gain scores)
- Permutation feature importance on validation sets
- SHAP (SHapley Additive exPlanations) summary plots and force plots to explain individual account spending predictions during viva presentation.

---

## 13. Results

*Status: No experimental models trained yet.*

Empirical model comparison tables, error distribution plots, and performance summaries will be added here upon completion of experimental runs.

---

## 14. Project Architecture

The current workspace structure containing existing files and directories:

```text
Personal Financial Digital Twin/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── account_monthly_panel.csv
│   │   ├── supervised_spending_dataset.csv
│   │   └── supervised_spending_dataset.parquet
│   └── raw/
│       └── berka/
├── models/
│   ├── catboost_model.joblib
│   ├── experiment_results.json
│   └── ridge_pipeline.joblib
├── notebooks/
├── reports/
│   ├── figures/
│   ├── dimension1_dataset_validation.md
│   ├── dimension2_preprocessing_and_eda.md
│   ├── dimension3_ml_implementation.md
│   └── supervised_dataset_validation.md
├── scripts/
│   ├── build_supervised_dataset.py
│   ├── run_eda.py
│   ├── run_preprocessing.py
│   └── train_evaluate_models.py
└── src/
    ├── config.py
    ├── data/
    │   ├── loader.py
    │   └── preprocessor.py
    ├── features/
    │   └── builder.py
    └── models/
        ├── evaluator.py
        └── models.py
```

*Note: New module files will be added to `src/` and `notebooks/` progressively as code is authored.*

---

## 15. Installation

*Status: Environment configuration phase.*

### Prerequisites
- Python 3.12+
- Virtual environment tool (`venv`)

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/amonish1004/ML-project---Personal-Financial-Digital-Twin-with-Explainable-Spending-Forecasting.git
cd "Personal Financial Digital Twin"

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies (to be updated as requirements.txt is populated)
pip install -r requirements.txt
```

---

## 16. Usage & Reproduction

*Status: Scripts pending implementation.*

Execution instructions for data ingestion, preprocessing, training, and evaluation will be added here as `src/` pipelines are developed.

---

## 17. Review 1 Evidence

### Dimension 1 — Problem & Dataset
- [x] Defined primary ML problem: Next-month total spending regression per account.
- [x] Evaluated 9 candidate datasets across public repositories (Kaggle, UCI, VSE).
- [x] Performed source validation and structural comparison.
- [x] Provisionally selected PKDD '99 Czech Bank Dataset (Berka Dataset) based on real-world banking panel structure and statistical sample volume ($\approx 167,000$ samples across 4,500 accounts).
- [x] Documented detailed findings in [`reports/review1_dimension1_problem_and_dataset.md`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/review1_dimension1_problem_and_dataset.md).

### Dimension 2 — Data Preprocessing & EDA
- [x] Ingested raw headerless PKDD '99 dataset tables into modular Python pipeline (`src/data/loader.py`).
- [x] Normalized transaction direction signs (`-amount` for debits) and extracted chronologically valid month-end liquid balances ($B_{u, t}$).
- [x] Constructed monthly account-level panel (`185,057` account-months) with 100% exact spending and credit income mathematical reconciliation (`data/processed/account_monthly_panel.csv`).
- [x] Built leakage-safe supervised tabular regression dataset (`171,194` samples across 4,500 accounts) using strictly historical 3-month windows ($t-2, t-1, t \to t+1$) with 0 target contamination (`data/processed/supervised_spending_dataset.csv`).
- [x] Performed empirical EDA, category decomposition, statistical outlier auditing, and generated 5 publication-quality figures (`reports/figures/`).
- [x] Documented detailed findings in [`reports/dimension2_preprocessing_and_eda.md`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/dimension2_preprocessing_and_eda.md).

### Dimension 3 — ML Implementation
- [x] Partitioned dataset strictly out-of-time (Train: 2013–2016, Val: 2017, Test: 2018) with zero data leakage (`src/models/evaluator.py`).
- [x] Implemented Naive Persistence Baseline ($R^2 = -0.0643$ Test), Ridge Regression Pipeline ($R^2 = 0.4127$ Test), and CatBoost Regressor ($R^2 = 0.4954$ Test).
- [x] Selected CatBoost Regressor based strictly on Validation set performance ($R^2 = 0.5339$, MAE = $\$679.11\text{ CZK}$).
- [x] Retrained selected CatBoost model on combined Train+Validation data (`2013–2017`) and evaluated once on held-out 2018 Test set ($\text{MAE} = \$729.88\text{ CZK}$, $\text{RMSE} = \$1,321.27\text{ CZK}$, $R^2 = 0.4954$, $\text{MedAE} = \$377.17\text{ CZK}$).
- [x] Serialized model artifacts under `models/` (`ridge_pipeline.joblib`, `catboost_model.joblib`, `experiment_results.json`).
- [x] Documented empirical findings and leakage audit in [`reports/dimension3_ml_implementation.md`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/reports/dimension3_ml_implementation.md).

---

## 18. Limitations

### Current Known & Anticipated Limitations
1. **Historical Dataset Era:** The provisionally selected PKDD '99 dataset covers transaction history from 1993 to 1998; while temporal mathematical patterns remain valid, inflation scales differ from modern currency levels.
2. **Cold Start Accounts:** Accounts with fewer than 3 months of consecutive history cannot generate full 3-month rolling lag features.
3. **Non-Recurring Expense Outliers:** Large unannounced one-off debits (e.g., major emergency expenses) introduce variance in regression targets.

---

## 19. Privacy & Ethical Considerations

- **Dataset Anonymization:** The project relies strictly on anonymized, publicly available open research datasets. No personal identifiable information (PII) or real consumer banking credentials are collected or stored.
- **Responsible Financial Modeling:** Spending predictions are intended solely for personal budgeting assistance and educational digital twin simulation, not for automated credit scoring or loan refusal decisions.

---

## 20. Technologies Used

Currently integrated / verified packages:
- **Language:** Python 3.12.6
- **Environment & Tools:** `venv`, `pip`
- **Data Manipulation & Analysis:** `pandas 3.0.5`, `numpy 2.5.2` (installed for initial inspection)
- **HTTP / Web Communication:** `requests 2.34.2`

*Additional libraries (e.g., `scikit-learn`, `matplotlib`, `seaborn`, `shap`) will be added to this list only after installation and use.*

---

## 21. Reproducibility

- **Python Version:** 3.12.6
- **Operating System:** Windows 11
- **Random Seed:** 42 (Global seed to be enforced across all random operations)
- **Dependency Tracking:** [`requirements.txt`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/requirements.txt)
- **Pipeline Execution:** Pure Python module execution via `src/` to ensure 100% CLI reproducibility.

---

## 22. References

1. Berka, P., & Bruha, E. (1999). *Financial Dataset — PKDD '99 Discovery Challenge*. Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE), Czech Republic.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765-4774.

---

## 23. License

- **Project Source Code License:** MIT License (or specified academic open-source license upon completion).
- **Dataset Usage Terms:** Governed by the original PKDD '99 Open Academic Research License.