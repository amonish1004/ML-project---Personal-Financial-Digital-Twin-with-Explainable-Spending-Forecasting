import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes string columns by filling NAs with empty string and stripping leading/trailing whitespace.
    Compatible with Pandas 2.x and Pandas 3.x string dtypes.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]) or df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('').astype(str).str.strip()
    return df_clean


def parse_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Parses string date column into datetime64 and creates year_month Period.
    Does NOT alter historical dates (+20 year shift preserved).
    """
    df_parsed = df.copy()
    df_parsed['date_dt'] = pd.to_datetime(df_parsed[date_col], format='%Y-%m-%d', errors='raise')
    df_parsed['year_month'] = df_parsed['date_dt'].dt.to_period('M')
    return df_parsed


def validate_numeric_fields(df_trans: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates numeric integrity of transaction fields.
    """
    results = {
        "null_trans_id": int(df_trans['trans_id'].isnull().sum()),
        "null_account_id": int(df_trans['account_id'].isnull().sum()),
        "null_amount": int(df_trans['amount'].isnull().sum()),
        "null_balance": int(df_trans['balance'].isnull().sum()),
        "min_amount": float(df_trans['amount'].min()),
        "max_amount": float(df_trans['amount'].max()),
        "min_balance": float(df_trans['balance'].min()),
        "max_balance": float(df_trans['balance'].max())
    }
    return results


def aggregate_account_monthly(df_trans: pd.DataFrame, df_account: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data into a monthly account-level panel representation.
    
    Target expenditure definition:
    - Outgoing debits (type == 'D'): Magnitude = -amount (or abs(amount) since all non-zero debits <= 0).
    - Incoming credits (type == 'C'): Magnitude = amount.
    - Interest outward/fee adjustments (type == 'P'): Magnitude = -amount.
    
    Categories for debits (k_symbol):
    - HH: Household / Utility
    - ST: Statement / Bank Service Fee
    - IN: Insurance Payment
    - LO: Loan Repayment
    - IO: Interest Outward
    - OTHER: Uncategorized / General Debit (empty string '')
    
    Month-End Balance (B(u, t)):
    - Balance after chronologically final transaction of each account-month.
    
    Returns
    -------
    pd.DataFrame
        Account-month panel with 185,057 total account-months.
    """
    # 1. Normalize strings and parse dates
    df_clean = normalize_strings(df_trans)
    df_clean = parse_dates(df_clean, date_col='date')
    
    # 2. Sort chronologically for month-end balance extraction
    df_sorted = df_clean.sort_values(by=['account_id', 'date_dt', 'trans_id']).reset_index(drop=True)
    
    # Extract month-end balance (last transaction per account_id and year_month)
    month_end_balances = (
        df_sorted.groupby(['account_id', 'year_month'])
        .last()[['date', 'trans_id', 'balance']]
        .reset_index()
        .rename(columns={
            'balance': 'ending_balance',
            'trans_id': 'last_trans_id',
            'date': 'last_trans_date'
        })
    )
    
    # 3. Compute transaction magnitudes
    # For Type D, use -amount (positive spending magnitude)
    df_sorted['debit_amount'] = np.where(df_sorted['type'] == 'D', -df_sorted['amount'], 0.0)
    df_sorted['credit_amount'] = np.where(df_sorted['type'] == 'C', df_sorted['amount'], 0.0)
    df_sorted['interest_amount'] = np.where(df_sorted['type'] == 'P', -df_sorted['amount'], 0.0)
    
    # Category spending breakdowns for Type D
    df_sorted['spending_hh'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == 'HH'), -df_sorted['amount'], 0.0)
    df_sorted['spending_st'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == 'ST'), -df_sorted['amount'], 0.0)
    df_sorted['spending_in'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == 'IN'), -df_sorted['amount'], 0.0)
    df_sorted['spending_lo'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == 'LO'), -df_sorted['amount'], 0.0)
    df_sorted['spending_io'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == 'IO'), -df_sorted['amount'], 0.0)
    df_sorted['spending_other'] = np.where((df_sorted['type'] == 'D') & (df_sorted['k_symbol'] == ''), -df_sorted['amount'], 0.0)
    
    df_sorted['is_debit'] = np.where(df_sorted['type'] == 'D', 1, 0)
    df_sorted['is_credit'] = np.where(df_sorted['type'] == 'C', 1, 0)
    
    # 4. Group by account_id and year_month
    agg_dict = {
        'debit_amount': 'sum',
        'credit_amount': 'sum',
        'interest_amount': 'sum',
        'is_debit': 'sum',
        'is_credit': 'sum',
        'trans_id': 'count',
        'spending_hh': 'sum',
        'spending_st': 'sum',
        'spending_in': 'sum',
        'spending_lo': 'sum',
        'spending_io': 'sum',
        'spending_other': 'sum'
    }
    
    monthly_panel = df_sorted.groupby(['account_id', 'year_month']).agg(agg_dict).reset_index()
    
    monthly_panel = monthly_panel.rename(columns={
        'debit_amount': 'spending_debit',
        'credit_amount': 'income_credit',
        'interest_amount': 'interest_credit',
        'is_debit': 'debit_count',
        'is_credit': 'credit_count',
        'trans_id': 'total_trans_count'
    })
    
    # Merge month-end balance
    monthly_panel = pd.merge(
        monthly_panel,
        month_end_balances[['account_id', 'year_month', 'ending_balance', 'last_trans_id', 'last_trans_date']],
        on=['account_id', 'year_month'],
        how='left'
    )
    
    # Convert year_month to string format 'YYYY-MM' for clean CSV/Parquet storage
    monthly_panel['year_month_str'] = monthly_panel['year_month'].astype(str)
    
    # Reorder columns logically
    cols_order = [
        'account_id', 'year_month_str', 'spending_debit', 'income_credit', 'interest_credit',
        'debit_count', 'credit_count', 'total_trans_count', 'ending_balance',
        'spending_hh', 'spending_st', 'spending_in', 'spending_lo', 'spending_io', 'spending_other',
        'last_trans_id', 'last_trans_date'
    ]
    
    return monthly_panel[cols_order]
