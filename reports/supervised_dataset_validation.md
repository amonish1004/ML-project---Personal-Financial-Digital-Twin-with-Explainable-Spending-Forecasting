# Review 1 — Dimension 2: Supervised Dataset Validation & Data Leakage Audit Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 19, 2026  
**Status:** Validated & Leakage-Audited Supervised Regression Dataset  

---

## 1. Executive Summary

This report documents the construction and validation of the supervised regression dataset derived from the monthly account panel ([`data/processed/account_monthly_panel.csv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/processed/account_monthly_panel.csv)).

The dataset is constructed strictly adhering to the mandatory temporal leakage prevention rule:
> **For account u at reference month t, all feature attributes X(u, t) are derived strictly from transactions occurring on or before month t. The target variable Y(u, t+1) is derived exclusively from outgoing debit spending in month t+1.**

---

## 2. Supervised Problem & Target Formulation

- **ML Task:** Supervised Tabular Regression
- **Prediction Unit:** Account-Month Pair (u, t)
- **Target Variable (Y(u, t+1)):** `next_month_total_spending`
  - Defined as total outgoing debit expenditure incurred by account u in month t+1:
    $$Y(u, t+1) = \sum_{k \in \text{Debits}_{u, t+1}} \text{Amount}_{u, k, \text{debit}}$$
  - Stored as non-negative spending magnitude (Y(u, t+1) >= 0.00).

---

## 3. Historical Feature Taxonomy (X(u, t))

Features are extracted using a strictly consecutive 3-month historical window (t-2, t-1, t):

| Feature Name | Description | Window | Leakage Safe? |
| :--- | :--- | :---: | :---: |
| `spending_t` | Total debit spending in reference month t | Month t | YES |
| `spending_t_minus_1` | Total debit spending in month t-1 | Month t-1 | YES |
| `spending_t_minus_2` | Total debit spending in month t-2 | Month t-2 | YES |
| `spending_3m_mean` | 3-Month Moving Average | Months t-2 ... t | YES |
| `spending_3m_std` | 3-Month Spending Standard Deviation | Months t-2 ... t | YES |
| `debit_count_t` | Total debit transaction count in month t | Month t | YES |
| `income_credit_t` | Total credit deposit income in month t | Month t | YES |
| `ending_balance_t` | Ending liquid balance after last transaction of month t | Month t | YES |
| `spending_hh_t` | Household spending in month t | Month t | YES |
| `spending_st_t` | Statement fee spending in month t | Month t | YES |
| `spending_in_t` | Insurance spending in month t | Month t | YES |
| `spending_lo_t` | Loan repayment spending in month t | Month t | YES |
| `spending_io_t` | Interest outward spending in month t | Month t | YES |
| `spending_other_t` | Uncategorized spending in month t | Month t | YES |

---

## 4. Empirical Sample & Coverage Audit Results

- **Total Supervised Regression Samples:** **`171,194`**
- **Unique Accounts Represented:** **`4,500`** (100% of dataset accounts)
- **Reference Month Range (t):** `2013-03 to 2018-11` (69 reference months)
- **Target Month Range (t+1):** `2013-04 to 2018-12` (69 target months)
- **Consecutive Month Rule:** 100% enforced (m_{t+1} - m_{t-2} = 3). Zero samples span non-consecutive calendar months.
- **Null Value Count:** **`0`** across all features and targets.
- **Duplicate Index Check:** **`0`** duplicate (account_id, reference_month) rows.
- **Target Non-Negativity:** **`0`** negative target values (100% >= 0).

---

## 5. Explicit Data Leakage Audit Verification

An empirical 1-to-1 verification was performed against the monthly panel data:

1. **Target Verification:** `next_month_total_spending` matches month t+1 debit spending in panel: **MATCHED (`True`)**
2. **Current Spending Verification:** `spending_t` matches month t debit spending in panel: **MATCHED (`True`)**
3. **Ending Balance Verification:** `ending_balance_t` matches month t ending balance in panel: **MATCHED (`True`)**
4. **Window Boundary Verification:** Zero features consume data beyond month t.

---

## 6. Sample Count Reconciliation & Audit Comparison

- **Current Strict Calendar Windowing Audit:** **`171,194` samples** across **`4,500` accounts**.
- **Earlier Preliminary Audit Mention:** `~151,932` samples across `~4,435` accounts.
- **Reconciliation Explanation:**
  - The preliminary audit script applied a secondary filter requiring active debits (S > 0) in **all 4 consecutive months** (t-2 > 0, t-1 > 0, t > 0, t+1 > 0), which filtered out inactive/zero-spending months (yielding 156,519 samples across 4,454 accounts).
  - The complete calendar panel windowing retains all **171,194 strictly consecutive 4-month calendar sequences** across all 4,500 accounts, providing a complete, unbiased temporal panel.
