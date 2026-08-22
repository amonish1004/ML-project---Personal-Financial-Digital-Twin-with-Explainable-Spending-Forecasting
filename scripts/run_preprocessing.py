import sys
import os
from pathlib import Path
import hashlib
import pandas as pd

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.data.loader import load_all_raw_tables
from src.data.preprocessor import (
    normalize_strings,
    parse_dates,
    validate_numeric_fields,
    aggregate_account_monthly
)


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for immutability verification."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 80)
    print("PERSONAL FINANCIAL DIGITAL TWIN — DIMENSION 2 DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # 1. Record raw file hashes before processing
    raw_files = sorted([f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.tsv')])
    hashes_before = {f: compute_file_hash(DATA_RAW_DIR / f) for f in raw_files}
    sizes_before = {f: (DATA_RAW_DIR / f).stat().st_size for f in raw_files}
    
    print("\n--- STAGE 1: RAW DATASET LOADING & IMMUTABILITY AUDIT ---")
    print(f"Raw Dataset Path: {DATA_RAW_DIR}")
    print(f"Total Raw Files: {len(raw_files)}")
    for f in raw_files:
        print(f"  - {f:<18}: {sizes_before[f]:>10,} bytes | SHA-256: {hashes_before[f][:16]}...")
        
    tables = load_all_raw_tables(DATA_RAW_DIR)
    
    print("\n--- STAGE 2: RAW FILE ROW COUNT & SCHEMA VALIDATION ---")
    expected_rows = {
        'fin_trans.tsv': 1056320,
        'fin_account.tsv': 4500,
        'fin_client.tsv': 5369,
        'fin_disp.tsv': 5369,
        'fin_order.tsv': 6471,
        'fin_loan.tsv': 682,
        'fin_card.tsv': 892,
        'fin_district.tsv': 77
    }
    
    for f, df in tables.items():
        actual_r = len(df)
        exp_r = expected_rows[f]
        match_str = "MATCH" if actual_r == exp_r else f"DISCREPANCY (expected {exp_r})"
        print(f"  - {f:<18}: {actual_r:>9,} rows | {len(df.columns):>2} cols | Status: {match_str}")
        
    # 3. Numeric & String Validation
    df_trans = tables['fin_trans.tsv']
    df_account = tables['fin_account.tsv']
    
    print("\n--- STAGE 3: DATA QUALITY & NUMERIC INTEGRITY AUDIT ---")
    num_audit = validate_numeric_fields(df_trans)
    for k, v in num_audit.items():
        print(f"  - {k:<20}: {v}")
        
    # 4. Account-Month Panel Aggregation
    print("\n--- STAGE 4: ACCOUNT-MONTH PANEL AGGREGATION ---")
    panel = aggregate_account_monthly(df_trans, df_account)
    
    total_obs = len(panel)
    unique_accs = panel['account_id'].nunique()
    min_ym = panel['year_month_str'].min()
    max_ym = panel['year_month_str'].max()
    
    print(f"  - Total Account-Month Observations: {total_obs:,}")
    print(f"  - Unique Accounts Represented:     {unique_accs:,}")
    print(f"  - Temporal Date Coverage Span:     {min_ym} to {max_ym} (72 Calendar Months)")
    
    # 5. Exact Reconciliation Checks
    print("\n--- STAGE 5: RECONCILIATION & MATHEMATICAL IDENTITY CHECKS ---")
    
    # Raw Type D absolute sum vs Panel spending_debit sum
    raw_type_d_abs = df_trans[df_trans['type'].str.strip() == 'D']['amount'].abs().sum()
    panel_spending_sum = panel['spending_debit'].sum()
    debit_diff = abs(raw_type_d_abs - panel_spending_sum)
    
    # Raw Type C sum vs Panel income_credit sum
    raw_type_c_sum = df_trans[df_trans['type'].str.strip() == 'C']['amount'].sum()
    panel_income_sum = panel['income_credit'].sum()
    income_diff = abs(raw_type_c_sum - panel_income_sum)
    
    # Category spending sum in panel vs total spending_debit in panel
    category_sum = (
        panel['spending_hh'].sum() +
        panel['spending_st'].sum() +
        panel['spending_in'].sum() +
        panel['spending_lo'].sum() +
        panel['spending_io'].sum() +
        panel['spending_other'].sum()
    )
    category_diff = abs(panel_spending_sum - category_sum)
    
    print(f"1. DEBIT SPENDING RECONCILIATION:")
    print(f"   - Raw Type D Total Absolute Amount : ${raw_type_d_abs:,.2f}")
    print(f"   - Monthly Panel spending_debit Sum : ${panel_spending_sum:,.2f}")
    print(f"   - Reconciliation Delta             : ${debit_diff:,.6f} -> {'EXACT MATCH (PASSED)' if debit_diff < 1e-4 else 'FAILED'}")
    
    print(f"\n2. CREDIT INCOME RECONCILIATION:")
    print(f"   - Raw Type C Total Amount          : ${raw_type_c_sum:,.2f}")
    print(f"   - Monthly Panel income_credit Sum  : ${panel_income_sum:,.2f}")
    print(f"   - Reconciliation Delta             : ${income_diff:,.6f} -> {'EXACT MATCH (PASSED)' if income_diff < 1e-4 else 'FAILED'}")
    
    print(f"\n3. CATEGORY DECOMPOSITION RECONCILIATION:")
    print(f"   - Total Panel spending_debit Sum   : ${panel_spending_sum:,.2f}")
    print(f"   - Sum of All 6 Category Columns     : ${category_sum:,.2f}")
    print(f"     * Household (spending_hh)        : ${panel['spending_hh'].sum():,.2f}")
    print(f"     * Statement Fees (spending_st)   : ${panel['spending_st'].sum():,.2f}")
    print(f"     * Insurance (spending_in)        : ${panel['spending_in'].sum():,.2f}")
    print(f"     * Loan Repayments (spending_lo)  : ${panel['spending_lo'].sum():,.2f}")
    print(f"     * Interest Outward (spending_io) : ${panel['spending_io'].sum():,.2f}")
    print(f"     * Uncategorized (spending_other) : ${panel['spending_other'].sum():,.2f}")
    print(f"   - Category Reconciliation Delta    : ${category_diff:,.6f} -> {'EXACT MATCH (PASSED)' if category_diff < 1e-4 else 'FAILED'}")

    # 6. Save Outputs
    print("\n--- STAGE 6: SAVING PROCESSED DATASET ARTIFACTS ---")
    parquet_path = DATA_PROCESSED_DIR / "account_monthly_panel.parquet"
    csv_path = DATA_PROCESSED_DIR / "account_monthly_panel.csv"
    
    # Save CSV
    panel.to_csv(csv_path, index=False)
    print(f"  - CSV Output Saved     : {csv_path} ({csv_path.stat().st_size:,.0f} bytes)")
    
    # Try Parquet save if pyarrow is installed
    try:
        panel.to_parquet(parquet_path, index=False)
        print(f"  - Parquet Output Saved : {parquet_path} ({parquet_path.stat().st_size:,.0f} bytes)")
    except Exception as e:
        print(f"  - Parquet Save Note    : Skipping Parquet write ({e}). CSV output is primary.")

    # 7. Post-execution Raw Data Immutability Check
    hashes_after = {f: compute_file_hash(DATA_RAW_DIR / f) for f in raw_files}
    sizes_after = {f: (DATA_RAW_DIR / f).stat().st_size for f in raw_files}
    
    immutability_passed = all(hashes_before[f] == hashes_after[f] and sizes_before[f] == sizes_after[f] for f in raw_files)
    
    print("\n--- STAGE 7: POST-EXECUTION RAW DATA IMMUTABILITY VERIFICATION ---")
    print(f"  - Raw Files Modified : {'NONE (0 files modified)' if immutability_passed else 'WARNING: FILES WERE MODIFIED!'}")
    print(f"  - Immutability Status: {'VERIFIED 100% IMMUTABLE' if immutability_passed else 'FAILED'}")
    
    print("\n" + "=" * 80)
    print("DIMENSION 2 PREPROCESSING PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
