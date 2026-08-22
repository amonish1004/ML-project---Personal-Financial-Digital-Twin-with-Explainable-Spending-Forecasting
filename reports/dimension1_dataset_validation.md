# Review 1 — Dimension 1: Final Empirical Dataset Validation Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 18, 2026  
**Dataset Examined:** PKDD '99 Czech Financial Dataset (Berka Dataset) — `data/raw/berka/`  
**Audit Method:** Read-only empirical inspection of raw TSV files  

---

## 1. Dataset Structure & Physical Properties

The read-only audit examined the 8 raw TSV tables located in [`data/raw/berka/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka). The primary files relevant to problem formulation and feature engineering exhibit the following physical properties:

| File Name | File Size | Data Rows | Columns | Primary Key / Join Key | Main Contents |
| :--- | :---: | :---: | :---: | :--- | :--- |
| [`fin_trans.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_trans.tsv) | 52.33 MB | **1,056,319** | 10 | `trans_id`, `account_id` | Individual transaction ledger |
| [`fin_account.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_account.tsv) | 0.09 MB | **4,499** | 4 | `account_id` | Account metadata & creation dates |
| [`fin_disp.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_disp.tsv) | 0.08 MB | **5,368** | 4 | `disp_id`, `account_id` | Client-to-account ownership links |
| [`fin_client.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_client.tsv) | 0.11 MB | **5,368** | 4 | `client_id` | Demographic records |
| [`fin_order.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_order.tsv) | 0.20 MB | **6,470** | 6 | `order_id`, `account_id` | Permanent standing payment orders |
| [`fin_loan.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_loan.tsv) | 0.03 MB | **681** | 7 | `loan_id`, `account_id` | Issued loan records |
| [`fin_card.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_card.tsv) | 0.02 MB | **891** | 4 | `card_id`, `disp_id` | Issued payment cards |
| [`fin_district.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_district.tsv) | 0.01 MB | **76** | 16 | `district_id` | Regional demographic metrics |

---

## 2. Raw Schema & Transaction Semantics

### `fin_trans.tsv` Column Definitions
1. `trans_id` (Integer): Unique transaction record identifier.
2. `account_id` (Integer): Unique account identifier (foreign key to `fin_account.tsv`).
3. `date` (String `YYYY-MM-DD`): Date of transaction execution.
4. `amount` (Float): Transaction monetary value in currency units.
5. `balance` (Float): Running account balance immediately following transaction execution.
6. `type` (String): Direction code of transaction (`D` = Debit/Expense, `C` = Credit/Income, `P` = Interest Payment).
7. `operation` (String): Mode of transaction execution (`WIC`, `ROB`, `CIC`, `COB`, `CCW`).
8. `k_symbol` (String): Characterization code / category symbol (`HH`, `ST`, `PE`, `IN`, `LO`, `IC`, `IO`).
9. `bank` (String): Code of destination/origin bank (optional/sparse).
10. `account` (Float): Destination/origin partner account number (optional/sparse).

### Observed Transaction Type (`type`) Breakdown
- **`D` (Debit / Outgoing Expenditure):** **634,571 transactions (60.07%)** — Represents all outgoing spending, cash withdrawals, and electronic bill payments.
- **`C` (Credit / Incoming Deposits):** **405,083 transactions (38.35%)** — Represents incoming salary, pensions, deposits, and bank collections.
- **`P` (Interest Credit):** **16,666 transactions (1.58%)** — Represents automatic monthly interest credited by the bank.

### Observed Operation (`operation`) Breakdown
- `WIC` (Cash Withdrawal Debit / Interest Credit): 434,918 transactions (418,252 debits, 16,666 interest credits)
- `ROB` (Remittance / Standing Order Outward): 208,283 transactions (100% debits)
- `CIC` (Cash In Credit / Cash Deposit): 156,743 transactions (100% credits)
- `COB` (Collection from Other Bank): 65,226 transactions (100% credits)
- `CCW` (Credit Card Withdrawal): 8,036 transactions (100% debits)
- ` [Blank]` (Direct Bank Transfer): 183,114 transactions (100% credits)

### Observed Category (`k_symbol`) Breakdown
- `IC` (Interest Credited): 183,114
- `ST` (Statement Fee Debit): 155,832
- `HH` (Household / Utility Payment): 118,065
- `PE` (Pension Income Credit): 30,338
- `IN` (Insurance Payment Debit): 18,500
- `LO` (Loan Payment Debit): 13,580
- `IO` (Interest Outward): 1,577
- `[Blank]` (General / Uncategorized Transfer): 535,304

---

## 3. Temporal Coverage & Entity Audits

- **Minimum Date:** `2013-01-01`
- **Maximum Date:** `2018-12-31`
- **Total Temporal Span:** Exactly **72 calendar months** (6 full years: 2013–2018).
- **Unique Accounts (`account_id`):** **4,500** accounts in both `fin_account.tsv` and `fin_trans.tsv`.
- **Unique Clients (`client_id`):** **5,369** clients (4,500 primary account owners + 869 secondary authorized signers).
- **Total Account-Month Observations:** **171,554** aggregated monthly records across all accounts.

---

## 4. Account-Level History & Sample Availability

### Monthly Active Account Coverage
- **Mean Active Months per Account:** 38.12 months (Min: 1, Max: 72).
- **Accounts with $\ge 4$ Active Months:** **4,496 / 4,500 accounts** (99.91%)
- **Accounts with $\ge 6$ Active Months:** **4,489 / 4,500 accounts** (99.76%)
- **Accounts with $\ge 12$ Active Months:** **4,352 / 4,500 accounts** (96.71%)

### Temporal Gap Analysis
- **Total Transitions with Gaps ($>1$ month gap between consecutive transactions):** 924 transitions out of 171,554 account-months ($0.54\%$).
- **Accounts Affected by Temporal Gaps:** 322 accounts (7.15%).
- **Gap Handling Strategy:** Gaps are easily filtered during pipeline windowing by requiring strictly contiguous month indices ($t-3, t-2, t-1, t, t+1$).

### Supervised ML Sample Availability
When requiring a **3-month contiguous historical feature window** ($t-2, t-1, t$) to predict the **1-month forward spending target** ($t+1$):
- **Total Strictly Consecutive Supervised ML Training Samples:** **151,932 samples**
- **Unique Accounts Represented in Final ML Dataset:** **4,435 accounts**
- **Sample Sufficiency:** Provides abundant data for temporal Out-of-Time train/validation/test splitting (e.g., 2013–2016 Train [90k+ samples], 2017 Validation [30k+ samples], 2018 Test [30k+ samples]).

---

## 5. Target Variable & Feature Definitions

### Ground Truth Target Variable ($Y_{u, t+1}$)
The target variable is defined as the total outgoing expenditure incurred by account $u$ in calendar month $t+1$:

$$Y_{u, t+1} = \sum_{k \in \mathcal{D}_{u, t+1}} \text{Amount}_{u, k}$$

Where $\mathcal{D}_{u, t+1}$ is the set of all transactions for account $u$ in month $t+1$ with direction code `type == 'D'` (Debits).

### Candidate Historical Input Features ($X_{u, t}$)
Extracted strictly from historical observation window $t, t-1, t-2$:

1. **Lagged Expenditure:**
   - $S_{u, t}$ (Total spending in month $t$)
   - $S_{u, t-1}$ (Total spending in month $t-1$)
   - $S_{u, t-2}$ (Total spending in month $t-2$)
2. **Rolling Expenditure Statistics:**
   - 3-Month Moving Average: $\mu_{u, t} = \frac{S_{u, t} + S_{u, t-1} + S_{u, t-2}}{3}$
   - 3-Month Spending Volatility (Std Dev): $\sigma_{u, t}$
   - Month-over-Month Spending Growth: $\Delta S_{u, t} = \frac{S_{u, t} - S_{u, t-1}}{S_{u, t-1} + \epsilon}$
3. **Transaction Velocity & Size:**
   - Total Debit Count ($N_{u, t}$)
   - Average Debit Transaction Size ($\bar{A}_{u, t}$)
4. **Income & Liquidity Metrics (Target-Safe):**
   - Total Monthly Credit Deposits ($I_{u, t}$ from `type == 'C'`)
   - Ending Account Balance ($B_{u, t}$ recorded at final transaction of month $t$)
   - Net Monthly Savings Flow ($I_{u, t} - S_{u, t}$)
5. **Category-Level Features (from `k_symbol` & `fin_order.tsv`):**
   - Lagged Household Spending ($S_{u, t, \text{HH}}$)
   - Lagged Loan Payment Spending ($S_{u, t, \text{LO}}$)
   - Discretionary / Uncharacterized Spending Ratio

---

## 6. Data Leakage & Risk Audit

1. **Target Contamination Risk:** Zero risk if temporal boundaries are strictly enforced. All features $X_{u, t}$ must be derived exclusively from transactions with timestamps $\le \text{Last Day of Month } t$.
2. **Income Feature Safety:** Including historical credit deposits ($I_{u, t}$) from month $t$ does **NOT** cause target leakage for month $t+1$ spending, provided no month $t+1$ credit records are included in $X_{u, t}$.
3. **Balance Feature Safety:** `balance` represents the running liquid balance. Taking the last balance entry of month $t$ ($B_{u, t}$) provides a legitimate baseline liquidity signal for month $t+1$ spending.

---

## 7. Known Data Limitations

1. **Uncharacterized Spending Share:** Approximately 50.6% of debit transactions have blank `k_symbol` category labels, representing general debit transfers and point-of-sale cash withdrawals. While overall spending $S_{u, t}$ is exact, fine-grained category decomposition is limited to major fixed bills (household, insurance, loans).
2. **Historical Era / Currency Scale:** The dataset uses normalized CZK currency units spanning 2013–2018; relative spending ratios and lag dynamics remain 100% mathematically valid for machine learning regression.

---

## 8. Final ML Problem Assessment & Recommendation

### Proposed Problem Statement Evaluation
> **"Given an account's historical financial behavior, predict its total spending in the following month."**

- **Empirical Validation Verdict:** **SUPPORTED**
- **Justification:**
  1. The raw dataset contains 1,056,319 transaction records across 4,500 accounts over 72 months.
  2. Outgoing expenditure (`type == 'D'`) is cleanly separable from incoming deposits (`type == 'C'`).
  3. Grouping debits by account and calendar month produces 171,554 monthly observations.
  4. Enforcing a 3-month lag history and a 1-month forward target yields **151,932 strictly consecutive supervised training samples** across 4,435 accounts.
  5. The statistical volume and panel structure fully support training, validating, and explaining supervised ML regression models (Linear Regressors, Random Forests, XGBoost, SHAP).

---

## 9. Final Conclusion

### **CONCLUSION: SUPPORTED**

The PKDD '99 Czech Bank Dataset empirically supports the proposed machine learning problem without requiring any modification to the core research objective.
