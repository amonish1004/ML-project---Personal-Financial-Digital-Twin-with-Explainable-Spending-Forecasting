# Date Provenance & Temporal Discrepancy Investigation Report

**Project:** Personal Financial Digital Twin with Explainable Spending Forecasting  
**Date:** August 18, 2026  
**Subject:** Empirical Investigation of Date Encoding & Range Discrepancy in `data/raw/berka/`  

---

## 1. Executive Summary

During dataset inspection, a discrepancy was noted between original PKDD '99 challenge documentation (which cites dates spanning **1993 to 1998**) and the acquired raw TSV dataset files (which report dates spanning **2013 to 2018**). 

A read-only investigation confirmed that:
1. The raw TSV files in [`data/raw/berka/`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka) contain dates formatted as `YYYY-MM-DD` strings ranging from `2013-01-01` to `2018-12-31`.
2. The download mirror used (`https://github.com/dnoeth/1999_Czech_financial_dataset_Teradata`) intentionally applied a deterministic **+20 year linear temporal shift** to convert historical 1990s dates into modern 2010s dates for Teradata database benchmarking.
3. This linear shift preserves 100% of relative time deltas, monthly sequences, lag intervals, and seasonal patterns.
4. **Academic Reporting Recommendation:** For academic rigor and viva defense, we must report the original historical date range (**1993–1998**) as the ground-truth dataset provenance while documenting the +20 year offset present in the TSV mirror files.

---

## 2. Raw Date Inspection

Read-only inspection of the downloaded raw TSV files revealed the following raw string values:

### Raw Examples from [`fin_trans.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_trans.tsv) (Column 3)
- Row 1: `'2015-03-24'`
- Row 2: `'2015-04-13'`
- Row 3: `'2015-05-13'`
- Row 4: `'2015-06-13'`
- Row 5: `'2015-07-13'`
- **Observed Range in `fin_trans.tsv`:** `2013-01-01` to `2018-12-31` (10-character string format `YYYY-MM-DD`)

### Raw Examples from [`fin_account.tsv`](file:///d:/College%20UG/3rd%20year/5TH%20SEM/Machine%20Learning/ML%20project/Personal%20Financial%20Digital%20Twin/data/raw/berka/fin_account.tsv) (Column 3)
- Account 1 Creation Date: `'2015-03-24'`
- Account 2 Creation Date: `'2013-02-26'`
- Account 3 Creation Date: `'2017-07-07'`
- **Observed Range in `fin_account.tsv`:** `2013-01-01` to `2017-12-29`

---

## 3. Comparison with Original PKDD '99 Documentation

| Attribute | Original PKDD '99 Challenge Specs | Downloaded Mirror (`dnoeth/Teradata`) |
| :--- | :--- | :--- |
| **File Format** | ASCII fixed-width / semicolon `.asc` | Tab-delimited `.tsv` |
| **Date Encoding** | 6-digit integer `YYMMDD` (e.g. `950324`) | 10-character string `YYYY-MM-DD` (e.g. `'2015-03-24'`) |
| **Start Date** | January 1, 1993 (`930101`) | January 1, 2013 (`2013-01-01`) |
| **End Date** | December 31, 1998 (`981231`) | December 31, 2018 (`2018-12-31`) |
| **Temporal Span** | 6 Years (72 Calendar Months) | 6 Years (72 Calendar Months) |

---

## 4. Source Evidence & Cause of Discrepancy

The mirror author (**dnoeth**) created `1999_Czech_financial_dataset_Teradata` to demonstrate database performance on modern SQL platforms (Teradata BTEQ scripts).

As documented in the repository specification:
1. **Date Transformation:** The original 2-digit year format `93` through `98` was converted to `2013` through `2018` by adding exactly **20 years** ($+7,305\text{ days}$).
2. **String Formatting:** Converted numeric `YYMMDD` into standardized ISO `YYYY-MM-DD`.

---

## 5. Mathematical & Machine Learning Impact

Because the transformation is a **strict uniform linear shift**:

1. **Relative Lag Windows:** The duration between consecutive transactions for any account is completely unchanged ($\Delta t_{\text{original}} = \Delta t_{\text{mirror}}$).
2. **Monthly Aggregations:** Grouping by year-month yields the exact same 72-month sequence, account transaction counts, and monthly expenditure totals.
3. **Model Training:** Machine learning features (lagged spending $S_{u, t-1}$, 3-month moving average $\mu_{u, t}$, savings rate) are identical whether computed on 1995 or 2015 labels.

---

## 6. Academic Reporting Recommendation

When presenting the project for academic evaluation, code inspection, and viva defense:

1. **Ground-Truth Citation:** State that the underlying data originates from the **PKDD '99 Discovery Challenge dataset (1993–1998)**.
2. **Provenance Disclosure:** Explicitly disclose that the acquired TSV representation applies a $+20\text{-year}$ calendar shift (`2013–2018`) introduced by the public Teradata benchmark mirror.
3. **Methodological Validity:** Explain that the shift is purely cosmetic/calendar-based and has **zero impact** on time-series continuity, lag feature engineering, or regression model validity.
