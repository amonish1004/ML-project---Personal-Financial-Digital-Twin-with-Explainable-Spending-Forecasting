# Dataset Source & Provenance Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 18, 2026  
**Status:** Dataset Acquired & Read-Only Verified  

---

## 1. Dataset Overview

- **Dataset Name:** PKDD '99 Financial Dataset (Czech Bank / Berka Dataset)
- **Original Academic Source:** Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE), Czech Republic  
  *Authors:* Dr. Petr Berka and Eva Bruha (PKDD '99 Discovery Challenge)
- **Primary Source URL:** `http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip`
- **Actual Download Source Used:** Verified public GitHub repository mirror (`https://github.com/dnoeth/1999_Czech_financial_dataset_Teradata`)  
  *Note:* The fallback mirror was utilized due to network host resolution constraints accessing the primary Czech VSE domain from the local execution environment.
- **Download Date:** August 18, 2026
- **Cost:** **FREE** ($0.00 — No payment, subscription, or API key required)

---

## 2. Physical File Statistics (Read-Only Inspection)

- **Total Extracted Disk Space:** 55,432,344 bytes ($\approx 52.86\text{ MB}$)
- **Total Files:** 8 relational TSV data tables
- **Archive SHA-256 Hash:** `N/A` (Individual uncompressed raw TSV files downloaded directly from verified public repository mirror)

### File Level Inspection Table

| File Name | File Size (Bytes) | Size (MB) | Data Rows | Number of Columns | Primary Key / Relational Join Field |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `fin_trans.tsv` | 54,876,077 | 52.33 MB | **1,056,319** | 7 | `trans_id`, `account_id` |
| `fin_order.tsv` | 210,008 | 0.20 MB | **6,470** | 6 | `order_id`, `account_id` |
| `fin_client.tsv` | 110,885 | 0.11 MB | **5,368** | 4 | `client_id` |
| `fin_disp.tsv` | 88,614 | 0.08 MB | **5,368** | 4 | `disp_id`, `client_id`, `account_id` |
| `fin_account.tsv` | 92,603 | 0.09 MB | **4,499** | 4 | `account_id`, `district_id` |
| `fin_loan.tsv` | 28,330 | 0.03 MB | **681** | 7 | `loan_id`, `account_id` |
| `fin_card.tsv` | 19,472 | 0.02 MB | **891** | 4 | `card_id`, `disp_id` |
| `fin_district.tsv` | 6,355 | 0.01 MB | **76** | 16 | `district_id` |

---

## 3. Dataset Description & Structure

The dataset represents anonymized real-world banking operations from a Czech commercial bank over a 6-year period. It is organized into an 8-table relational database schema:

1. **`fin_trans.tsv` (Transactions):** Raw ledger containing 1,056,319 individual transaction entries with transaction ID, account ID, timestamp, debit/credit direction code, amount, post-transaction balance, and category operation symbols.
2. **`fin_account.tsv` (Accounts):** 4,499 unique bank account profiles linked to geographic districts and creation dates.
3. **`fin_client.tsv` (Clients):** 5,368 unique individual client demographic profiles.
4. **`fin_disp.tsv` (Dispositions):** Maps client IDs to account IDs, distinguishing account owners (`OWNER`) from authorized signers (`USER`).
5. **`fin_order.tsv` (Payment Orders):** 6,470 standing permanent payment orders for recurring debits (household, insurance, loans).
6. **`fin_loan.tsv` (Loans):** 681 granted loan records with term length, monthly payment amount, and repayment status flags.
7. **`fin_card.tsv` (Credit Cards):** 891 issued credit cards with card type (Junior, Classic, Gold) and issue date.
8. **`fin_district.tsv` (Districts):** Demographic statistics across 76 Czech administrative regions.

---
## 4. Access & Licensing Information

- **Access:** The dataset was obtained from a publicly accessible repository containing the PKDD '99 Czech financial dataset.
- **Cost:** Free to download.
- **Registration:** No registration or API key was required for the download used in this project.
- **Original licensing status:** The original 1999 dataset distribution does not appear to provide a modern standardized license such as CC BY or MIT in the materials examined.
- **Project usage:** The dataset is being used for an academic machine-learning project.
- **Licensing note:** Because a standardized modern license could not be independently confirmed, the project does not make a broader claim that the dataset is public domain or unrestricted for all forms of redistribution.

---

## 5. Academic Citation

When citing this dataset in project reports or academic publications:

> **Berka, P., & Bruha, E. (1999).** *PKDD '99 Discovery Challenge Financial Dataset*. Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE), Czech Republic.

---

## 6. Raw Data Integrity Confirmation

The raw files in `data/raw/berka/` were preserved without intentional modification after acquisition.

No rows were removed, modified, filtered, renamed, cleaned, or aggregated.

No feature engineering, preprocessing, EDA, or model training has been executed.

File hashes can be generated later if stronger reproducibility verification is required.