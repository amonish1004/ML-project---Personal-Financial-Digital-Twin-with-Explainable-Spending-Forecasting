# Data Dictionary & Raw Schema Documentation

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Dataset:** PKDD '99 Financial Dataset (Czech Bank / Berka Dataset)  
**Data Format:** Raw Headerless Tab-Separated Values (TSV) files in `data/raw/berka/`

> **Note on Raw Data Preservation:**  
> The original raw dataset files in `data/raw/berka/` are headerless TSV files. To preserve exact data integrity, raw files on disk remain strictly unmodified and headerless. Column schemas and semantics are documented separately here and loaded in Python code via `src/config.py` and `src/data/loader.py` using explicit column names (`pandas.read_csv(..., sep='\t', header=None, names=...)`).

---

## 1. Summary of Raw Relational Tables

| File Name | Disk Size | Rows | Columns | Header Status | Primary Key / Joins | Content Description |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `fin_trans.tsv` | 52.33 MB | 1,056,319 | 10 | Headerless TSV | `trans_id`, `account_id` | Individual transaction ledger |
| `fin_account.tsv` | 92.6 KB | 4,500 | 4 | Headerless TSV | `account_id`, `district_id` | Account profiles & branch location |
| `fin_client.tsv` | 110.9 KB | 5,369 | 4 | Headerless TSV | `client_id`, `district_id` | Client demographics & birth dates |
| `fin_disp.tsv` | 88.6 KB | 5,369 | 4 | Headerless TSV | `disp_id`, `client_id`, `account_id` | Client-Account access dispositions |
| `fin_district.tsv` | 6.4 KB | 76 | 16 | Headerless TSV | `district_id` | Demographic & regional statistics |
| `fin_loan.tsv` | 28.3 KB | 681 | 7 | Headerless TSV | `loan_id`, `account_id` | Loan contracts & repayment status |
| `fin_order.tsv` | 210.0 KB | 6,470 | 6 | Headerless TSV | `order_id`, `account_id` | Permanent standing payment orders |
| `fin_card.tsv` | 19.5 KB | 891 | 4 | Headerless TSV | `card_id`, `disp_id` | Issued credit/debit payment cards |

---

## 2. Table-by-Table Data Dictionary

### Table 1: `fin_trans.tsv` (Transaction Ledger — Core Project Source)
*1,056,319 records · 10 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Project Usage / Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `trans_id` | Unique transaction record identifier | Integer | Primary Key | Unique transaction ID |
| 2 | `account_id` | Account identifier associated with transaction | Integer | Foreign Key $\rightarrow$ `fin_account.tsv` | Core entity identifier linking transactions to accounts |
| 3 | `date` | Date of transaction execution (`YYYY-MM-DD`) | String | Temporal index | Aggregated into monthly time periods ($t$) |
| 4 | `amount` | Monetary value of transaction | Float | Numerical amount | Base transaction amount (always positive in raw file) |
| 5 | `balance` | Post-transaction running account balance | Float | Numerical amount | Used for month-end ending balance `ending_balance_t` |
| 6 | `type` | Transaction direction (`D` = Debit, `C` = Credit, `P` = Interest) | String | Categorical code | **Critical filter:** `type == 'D'` defines outgoing expenditure |
| 7 | `operation` | Execution mode (`WIC`, `ROB`, `CIC`, `COB`, `CCW`, blank) | String | Categorical code | Mode of transaction execution (e.g. withdrawal, standing order) |
| 8 | `k_symbol` | Category symbol (`HH`, `ST`, `IN`, `LO`, `IO`, `PE`, `IC`, blank) | String | Categorical code | Category classification used to split debits into 6 spending features |
| 9 | `bank` | Partner bank identifier code | String | Foreign bank ref | Optional/sparse 2-character bank identifier |
| 10 | `account` | Partner account number | Float / Integer | Partner account ref | Optional/sparse partner account reference |

#### Transaction Type (`type`) Categories
- **`D` (Debit / Outgoing Expenditure):** 634,571 transactions (60.07%) — All outgoing spending, bill payments, and cash withdrawals.
- **`C` (Credit / Incoming Deposit):** 405,083 transactions (38.35%) — Salary, pensions, deposits, and bank collections (`income_credit_t`).
- **`P` (Interest Credit):** 16,666 transactions (1.58%) — Monthly interest earned credited by the bank.

#### Category Symbols (`k_symbol`) for Debits (`type == 'D'`)
- `HH` (Household / Utility Payment): Maps to `spending_hh_t`
- `ST` (Statement Fee Debit): Maps to `spending_st_t`
- `IN` (Insurance Payment Debit): Maps to `spending_in_t`
- `LO` (Loan Payment Debit): Maps to `spending_lo_t`
- `IO` (Interest Outward Debit): Maps to `spending_io_t`
- `[Blank]` / Other: Maps to `spending_other_t`

---

### Table 2: `fin_account.tsv` (Account Profiles)
*4,500 records · 4 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `account_id` | Unique account identifier | Integer | Primary Key | Unique account ID |
| 2 | `district_id` | Branch district location identifier | Integer | Foreign Key $\rightarrow$ `fin_district.tsv` | Link to district demographics |
| 3 | `date` | Account creation date (`YYYY-MM-DD`) | String | Temporal | Date account was opened |
| 4 | `frequency` | Statement issuance frequency (`M`, `W`, `I`) | String | Categorical | `M` = Monthly, `W` = Weekly, `I` = Instant |

---

### Table 3: `fin_client.tsv` (Client Demographics)
*5,369 records · 4 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `client_id` | Unique client/customer identifier | Integer | Primary Key | Unique client ID |
| 2 | `birth_date` | Date of birth (`YYYY-MM-DD`) | String | Temporal | Client birth date |
| 3 | `sex` | Gender of client (`F` = Female, `M` = Male) | String | Categorical | Client gender |
| 4 | `district_id` | Client residential district identifier | Integer | Foreign Key $\rightarrow$ `fin_district.tsv` | Link to district demographics |

---

### Table 4: `fin_disp.tsv` (Account Dispositions / Access Rights)
*5,369 records · 4 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `disp_id` | Unique disposition record identifier | Integer | Primary Key | Unique disposition ID |
| 2 | `client_id` | Client identifier | Integer | Foreign Key $\rightarrow$ `fin_client.tsv` | Links client to account |
| 3 | `account_id` | Account identifier | Integer | Foreign Key $\rightarrow$ `fin_account.tsv` | Links account to client |
| 4 | `type` | Access relationship type (`O` = Owner, `U` = User) | String | Categorical | `O` = Owner (primary), `U` = User (authorized signer) |

---

### Table 5: `fin_loan.tsv` (Granted Loans)
*681 records · 7 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `loan_id` | Unique loan identifier | Integer | Primary Key | Unique loan ID |
| 2 | `account_id` | Account identifier | Integer | Foreign Key $\rightarrow$ `fin_account.tsv` | Account granted the loan |
| 3 | `date` | Date loan granted (`YYYY-MM-DD`) | String | Temporal | Loan start date |
| 4 | `amount` | Principal loan amount | Float | Numerical amount | Total loan principal |
| 5 | `duration` | Term length in months | Integer | Numerical (12..60) | Duration of loan term |
| 6 | `payments` | Monthly repayment installment | Float | Numerical amount | Fixed monthly payment |
| 7 | `status` | Loan status (`A`, `B`, `C`, `D`) | String | Categorical flag | `A` = Finished ok, `B` = Finished unpaid, `C` = Running ok, `D` = Running in debt |

---

### Table 6: `fin_order.tsv` (Standing Payment Orders)
*6,470 records · 6 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `order_id` | Unique payment order identifier | Integer | Primary Key | Unique order ID |
| 2 | `account_id` | Source account identifier | Integer | Foreign Key $\rightarrow$ `fin_account.tsv` | Debited account |
| 3 | `bank_to` | Recipient bank code | String | Bank code | 2-character recipient bank identifier |
| 4 | `account_to` | Recipient account number | Integer / String | Account ref | Recipient partner account |
| 5 | `amount` | Monthly payment amount | Float | Numerical amount | Standing debit amount |
| 6 | `k_symbol` | Purpose/category symbol (`HH`, `LO`, `IN`, `ST`, etc.) | String | Categorical code | Standing order category symbol |

---

### Table 7: `fin_card.tsv` (Issued Payment Cards)
*891 records · 4 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `card_id` | Unique card identifier | Integer | Primary Key | Unique card ID |
| 2 | `disp_id` | Disposition identifier | Integer | Foreign Key $\rightarrow$ `fin_disp.tsv` | Links card to disposition/account owner |
| 3 | `type` | Card type (`G` = Gold, `C` = Classic, `J` = Junior) | String | Categorical | Card tier level |
| 4 | `issued` | Date card issued (`YYYY-MM-DD`) | String | Temporal | Card issue date |

---

### Table 8: `fin_district.tsv` (Regional & Demographic Data)
*76 records · 16 columns · Headerless TSV*

| Column # | Original Field | Meaning | Data Type | Key / Relational Status | Notes |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `district_id` / `A1` | Unique district identifier (1..76) | Integer | Primary Key | Unique district ID |
| 2 | `A2` | District name | String | Geographic name | e.g. `Benesov`, `Hl.m. Praha` |
| 3 | `A3` | Region name | String | Geographic name | e.g. `Central Bohemia`, `Prague` |
| 4 | `A4` | Number of inhabitants | Integer | Demographic | Total population |
| 5 | `A5` | Municipalities < 499 inhabitants | Integer | Demographic | Count of small villages |
| 6 | `A6` | Municipalities 500 – 1999 inhabitants | Integer | Demographic | Count of small towns |
| 7 | `A7` | Municipalities 2000 – 9999 inhabitants | Integer | Demographic | Count of medium towns |
| 8 | `A8` | Municipalities > 10000 inhabitants | Integer | Demographic | Count of large towns |
| 9 | `A9` | Number of cities | Integer | Demographic | Count of cities |
| 10 | `A10` | Urban inhabitant ratio (%) | Float / Integer | Demographic ratio | Percentage of urban population |
| 11 | `A11` | Average salary in district | Float / Integer | Economic metric | Regional average income |
| 12 | `A12` | Unemployment rate '95 (%) | Float | Economic metric | 1995 unemployment rate |
| 13 | `A13` | Unemployment rate '96 (%) | Float | Economic metric | 1996 unemployment rate |
| 14 | `A14` | Entrepreneurs per 1000 inhabitants | Integer | Economic metric | Small business density |
| 15 | `A15` | Committed crimes '95 | Integer / Float | Social metric | 1995 crime count |
| 16 | `A16` | Committed crimes '96 | Integer / Float | Social metric | 1996 crime count |

---

## 3. Data Flow & Connection to This Machine Learning Project

```
Headerless Raw TSV Files (data/raw/berka/*.tsv)
                       │
                       ▼
             [ src/data/loader.py ]
   (Applies schemas defined in src/config.py)
                       │
                       ▼
          [ Transaction Type Filtering ]
    (Filters fin_trans.tsv where type == 'D')
                       │
                       ▼
          [ Monthly Outflow Aggregation ]
 (Sums category debits: HH, ST, IN, LO, IO, other)
                       │
                       ▼
         [ Account-Month Panel Dataset ]
  (Computes lag features t-1, t-2 & derived mean/std)
                       │
                       ▼
        [ Supervised ML Training Vector ]
    (14-feature predictor vector -> next_month_total_spending)
```

### Frozen Target & Feature Contract Summary

1. **Prediction Target ($Y_{u, t+1}$):** `next_month_total_spending` — Sum of all outgoing debit transactions (`type == 'D'`) for account $u$ in month $t+1$.
2. **14 Predictor Features:**
   - **Primary User Inputs (11):** `ending_balance_t`, `income_credit_t`, `debit_count_t`, `spending_hh_t`, `spending_st_t`, `spending_in_t`, `spending_lo_t`, `spending_io_t`, `spending_other_t`, `spending_t_minus_1`, `spending_t_minus_2`
   - **Calculated Derived Features (3):** `spending_t` ($\sum$ 6 spending categories), `spending_3m_mean` ($\frac{\text{spending}_t + \text{tm1} + \text{tm2}}{3}$), `spending_3m_std` (population standard deviation, $ddof=0$).
