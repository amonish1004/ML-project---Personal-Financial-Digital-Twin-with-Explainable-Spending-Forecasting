# Review 1 — Dimension 1: Problem & Dataset Assessment

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 18, 2026  
**Status:** Initial Dataset Audit & Problem Verification  

---

## A. Dataset Summary

- **Dataset File Status:** No dataset file currently exists in `data/raw/`, `data/processed/`, or anywhere within the project repository.
- **Physical Inspection Metrics:**
  - **Total Rows:** N/A (0 data files present)
  - **Total Columns:** N/A
  - **Column Names & Data Types:** N/A
  - **Date Coverage:** N/A
  - **Missing Values & Duplicates:** N/A
  - **User & Transaction Identifiers:** N/A
  - **Income & Amount Fields:** N/A
- **Integrity Rule Compliance:** In strict adherence to Project Rule #5 (*"Do not fabricate datasets, observations, metrics, model results, or citations"*), no mock records or synthetic metrics have been fabricated.

---

## B. Dataset Suitability Assessment

Because no dataset file is currently present in [`data/raw/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw), a quantitative suitability audit cannot be executed yet. 

However, to validate any candidate dataset against our primary intended problem (**"Given historical financial behavior, predict next-month total spending per user"**), the dataset **MUST** satisfy the following mandatory structural criteria:

1. **User Identifiers (`user_id` / `account_id`):** Required to aggregate transactions into individual digital twin profiles. Datasets with anonymized or aggregated non-user-linked transactions cannot support user-level forecasting.
2. **Transaction Timestamps (`timestamp` / `date`):** Must cover a continuous timeframe of **at least 4 to 6 consecutive months** per user. This depth is necessary to construct historical lag features ($t-1, t-2, \dots$) and define a future target period ($t+1$).
3. **Transaction Amount & Direction (`amount`, `type` / `debit_credit`):** Must differentiate between outgoing expenditures (debits) and incoming funds (credits/income).
4. **Categorization (Optional but Recommended):** Categories (e.g., Groceries, Utilities, Subscriptions, Entertainment) enhance feature engineering and provide explainability for the digital twin.

---

## C. Recommended ML Problem Formulations

Depending on the dataset provided by the user, we propose three academically defensible supervised machine learning formulations:

### Formulation 1 (Primary Intended Problem): User-Level Next-Month Total Spending Regression
- **Type:** Supervised Tabular Regression
- **Objective:** Predict total debit expenditure $Y_{u, t+1}$ for user $u$ in month $t+1$.
- **Data Requirement:** Panel transaction logs with user IDs spanning $\ge 4-6$ months.
- **Viva Defense Rationale:** Directly aligns with personal financial budgeting and digital twin simulation.

### Formulation 2 (Granular Alternative): Category-Wise Next-Month Spending Forecasting
- **Type:** Multi-Output / Grouped Supervised Regression
- **Objective:** Predict user $u$'s spending across $K$ major expense categories in month $t+1$.
- **Data Requirement:** Categorized transaction logs with user IDs and timestamps.
- **Viva Defense Rationale:** Offers high explainability by identifying *where* budget increases will occur.

### Formulation 3 (Fallback / Single-User or Short History): Sliding-Window Spending Forecast
- **Type:** Time-Series Aggregated Regression
- **Objective:** Predict aggregate 30-day forward spending based on sliding 30-day historical windows.
- **Data Requirement:** Sequential transaction logs (even if user IDs are single or unsegmented).
- **Viva Defense Rationale:** Fallback option if available dataset lacks multi-user panel structure.

---

## D. Target Variable Definition (For Primary Formulation)

For a given user $u$ and target month $t+1$:

$$Y_{u, t+1} = \sum_{k \in \mathcal{T}_{u, t+1}} \text{Amount}_{u, k, \text{debit}}$$

Where $\mathcal{T}_{u, t+1}$ represents the set of all debit/expense transactions performed by user $u$ during calendar month $t+1$.

---

## E. Proposed Input Features

Engineered from historical observation windows (months $t, t-1, t-2$):

1. **Lagged Total Spending:** $S_{u, t}$, $S_{u, t-1}$, $S_{u, t-2}$
2. **Rolling Statistics:** 3-month moving average expenditure $\mu_{u}$, spending volatility (std dev $\sigma_{u}$)
3. **Income & Savings Metrics:** Monthly income $I_{u, t}$, net savings rate $\frac{I_{u, t} - S_{u, t}}{I_{u, t}}$ (if income is present)
4. **Transaction Volume & Velocity:** Transaction count $N_{u, t}$, average transaction size $\bar{A}_{u, t}$
5. **Category Share Features:** Proportion of total spending allocated to essential vs. discretionary categories
6. **Temporal/Calendar Features:** Target month index (1–12), quarter, holiday season indicators

---

## F. Known Limitations & Potential Risks

1. **Missing Data / No File:** Cannot proceed to EDA or feature extraction until a dataset is provided.
2. **Cold Start Problem:** Users with fewer than 3 months of history will have incomplete lag features.
3. **Non-Recurring Outliers:** Sudden large one-off expenses (e.g. medical emergency, major appliance purchase) introduce noise into regression models.
4. **Synthetic Data Risk:** Synthetic fraud datasets (e.g., PaySim) are optimized for transaction fraud classification, not user-level spending forecasting. A personal transaction dataset is required.

---

## G. Recommended Next Step

1. **Obtain / Place Dataset:** Provide or upload a transaction dataset into [`data/raw/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw) (e.g., a CSV file containing transaction records).
2. **Execute Quantitative Inspection:** Upon placement, run an automated Python inspection script to extract exact row counts, column types, missing value percentages, temporal coverage span, and user ID counts.
3. **Finalize Problem Formulation:** Formally confirm whether the dataset supports Formulation 1, 2, or 3 based on empirical evidence.
