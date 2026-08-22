# Personal Financial Digital Twin with Explainable Spending Forecasting

An academic machine learning project developing a **Personal Financial Digital Twin** to model account-level transaction behavior and forecast next-month outgoing debit expenditure.

---

## Project Status

**Review 1 is COMPLETE** across all three core evaluation dimensions:
- **Dimension 1 (Problem & Dataset):** Complete
- **Dimension 2 (Data Preprocessing & EDA):** Complete
- **Dimension 3 (ML Implementation & Evaluation):** Complete

The project is now moving beyond Review 1 toward model explainability and digital twin interface development.

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
Raw Berka TSVs ➔ Preprocessing ➔ Monthly Account Panel ➔ Feature Engineering ➔ Supervised Dataset ➔ Chronological Split ➔ Baseline + Ridge + CatBoost ➔ Evaluation
```

Dimension 2 completed raw data cleaning, monthly aggregation, leakage-safe dataset construction, and EDA. The final supervised dataset ([data/processed/supervised_spending_dataset.csv](data/processed/supervised_spending_dataset.csv)) contains **171,194 samples** across **4,500 accounts** with zero null values.

### Approved Predictor Features ($X_{u, t}$)
14 historical predictor features constructed strictly on or before month $t$:
- **Lag Spending:** `spending_t`, `spending_t_minus_1`, `spending_t_minus_2`
- **Rolling Volatility:** `spending_3m_mean`, `spending_3m_std`
- **Activity & Balance:** `debit_count_t`, `income_credit_t`, `ending_balance_t`
- **Category Breakdown:** `spending_hh_t` (household), `spending_st_t` (fees), `spending_in_t` (insurance), `spending_lo_t` (loans), `spending_io_t` (interest), `spending_other_t` (uncategorized)

*Note: `account_id`, `reference_month`, `target_month`, and `next_month_total_spending` are explicitly excluded from predictor matrix $X$.*

---

## ML Implementation & Empirical Results (Dimension 3)

Dimension 3 implementation is complete and empirically validated.

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

### Final Test Results
The selected CatBoost model was retrained on combined **Train + Validation** data (`2013-04` through `2017-12`) and evaluated once on the held-out **2018 Test Partition**:

| Model / Retraining Strategy | Test MAE (CZK) | Test RMSE (CZK) | Test R² | Test MedAE (CZK) |
| :--- | ---: | ---: | ---: | ---: |
| **Naive Persistence Baseline** | 996.51 | 1918.87 | -0.0643 | 418.54 |
| **Final CatBoost (Train + Val Retrained)** | **729.88** | **1321.27** | **0.4954** | **377.17** |

- **Test Set Isolation:** Test data was **not** used for model selection or hyperparameter tuning.
- **R² Interpretation:** An $R^2 \approx 0.50$ means CatBoost accounts for approximately half of the expenditure variance relative to a mean baseline.

---

## Methodology Status

The project aligns with the recommended **CatBoost + Time-Series Evaluation + SHAP Explainability** framework:

- [x] **CatBoost Regressor:** **IMPLEMENTED** (`models/catboost_model.joblib`)
- [x] **Time-Series / Out-of-Time Evaluation:** **IMPLEMENTED** (Strict chronological partitioning)
- [ ] **SHAP Explainability:** **NOT YET IMPLEMENTED** (Planned next stage)

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
├── src/
│   ├── config.py
│   ├── data/
│   ├── features/
│   └── models/
├── scripts/
│   ├── run_preprocessing.py
│   ├── build_supervised_dataset.py
│   ├── run_eda.py
│   └── train_evaluate_models.py
├── models/
│   ├── catboost_model.joblib
│   ├── ridge_pipeline.joblib
│   └── experiment_results.json
└── reports/
    ├── figures/
    ├── date_provenance_check.md
    ├── dimension1_dataset_validation.md
    ├── dimension2_preprocessing_and_eda.md
    ├── dimension3_ml_implementation.md
    └── supervised_dataset_validation.md
```

---

## Setup & Reproduction

### Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run Pipeline Scripts
```powershell
# 1. Preprocess raw data & build monthly panel
python scripts/run_preprocessing.py

# 2. Construct supervised dataset & run leakage audit
python scripts/build_supervised_dataset.py

# 3. Generate EDA statistics & figures
python scripts/run_eda.py

# 4. Train, evaluate, & serialize ML models
python scripts/train_evaluate_models.py
```

---

## Limitations

- **Upper-Tail Outliers:** Non-recurring spending debits drive higher RMSE metrics relative to MedAE (CatBoost Test MedAE: 377.17 CZK vs MAE: 729.88 CZK, RMSE: 1321.27 CZK).
- **Unexplained Variance:** $R^2 \approx 0.50$ reflects inherent stochastic variance in individual financial spending behavior.
- **Academic Scope:** Designed as an academic financial digital twin simulation; does not provide real-world credit scoring or financial advice.

---

## Future Work

- **SHAP Explainability Integration:** Implement global feature summary plots and local force plots.
- **Digital Twin Interface:** Develop an interactive web dashboard for spending forecasting and scenario simulation.
- **Extended Feature Engineering:** Incorporate seasonal indicators and volatility ratio metrics.

---

## Technical Documentation & References

For comprehensive technical reports, refer to:
- [reports/dimension1_dataset_validation.md](reports/dimension1_dataset_validation.md)
- [reports/dimension2_preprocessing_and_eda.md](reports/dimension2_preprocessing_and_eda.md)
- [reports/dimension3_ml_implementation.md](reports/dimension3_ml_implementation.md)