import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda_analysis():
    print("=" * 80)
    print("PERSONAL FINANCIAL DIGITAL TWIN — DIMENSION 2 EDA & ANALYTICAL PIPELINE")
    print("=" * 80)
    
    panel_csv = DATA_PROCESSED_DIR / "account_monthly_panel.csv"
    sup_csv = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"
    
    if not panel_csv.exists() or not sup_csv.exists():
        raise FileNotFoundError("Processed dataset files missing. Please run preprocessing and feature builder first.")
        
    df_panel = pd.read_csv(panel_csv)
    df_sup = pd.read_csv(sup_csv)
    
    print(f"\n--- 1. PANEL DATASET SUMMARY ---")
    print(f"Panel Rows: {len(df_panel):,} | Columns: {len(df_panel.columns)}")
    print(f"Unique Accounts: {df_panel['account_id'].nunique():,}")
    print(f"Date Coverage: {df_panel['year_month_str'].min()} to {df_panel['year_month_str'].max()} (72 months)")
    print(f"Missing Values in Panel: {df_panel.isnull().sum().sum()}")
    print(f"Duplicate Accounts-Month Rows: {df_panel.duplicated(subset=['account_id', 'year_month_str']).sum()}")
    
    # 2. Transaction & Spending Distribution Statistics
    print(f"\n--- 2. SPENDING & INCOME DISTRIBUTION STATISTICS ---")
    sp = df_panel['spending_debit']
    inc = df_panel['income_credit']
    bal = df_panel['ending_balance']
    
    print(f"Monthly Debit Spending (spending_debit):")
    print(f"  Mean: ${sp.mean():,.2f} | Std: ${sp.std():,.2f} | Median: ${sp.median():,.2f}")
    print(f"  IQR: [25th: ${sp.quantile(0.25):,.2f}, 75th: ${sp.quantile(0.75):,.2f}]")
    print(f"  Min: ${sp.min():,.2f} | Max: ${sp.max():,.2f} | Skewness: {sp.skew():.2f}")
    
    print(f"\nMonthly Income Credit (income_credit):")
    print(f"  Mean: ${inc.mean():,.2f} | Std: ${inc.std():,.2f} | Median: ${inc.median():,.2f}")
    print(f"  IQR: [25th: ${inc.quantile(0.25):,.2f}, 75th: ${inc.quantile(0.75):,.2f}]")
    print(f"  Min: ${inc.min():,.2f} | Max: ${inc.max():,.2f} | Skewness: {inc.skew():.2f}")
    
    print(f"\nEnding Balance (ending_balance):")
    print(f"  Mean: ${bal.mean():,.2f} | Std: ${bal.std():,.2f} | Median: ${bal.median():,.2f}")
    print(f"  Min: ${bal.min():,.2f} | Max: ${bal.max():,.2f}")
    
    # 3. Category Spending Breakdown
    print(f"\n--- 3. CATEGORY SPENDING BREAKDOWN ---")
    cats = ['spending_hh', 'spending_st', 'spending_in', 'spending_lo', 'spending_io', 'spending_other']
    cat_names = ['Household (HH)', 'Statement Fees (ST)', 'Insurance (IN)', 'Loan Payments (LO)', 'Interest Outward (IO)', 'Uncategorized (OTHER)']
    cat_totals = [df_panel[c].sum() for c in cats]
    tot_sp = sp.sum()
    
    for c_name, c_tot in zip(cat_names, cat_totals):
        print(f"  - {c_name:<25}: ${c_tot:>14,.2f} ({c_tot/tot_sp*100:>5.2f}%)")
        
    # 4. Outlier Identification
    print(f"\n--- 4. OUTLIER ANALYSIS (STATISTICAL BOUNDS) ---")
    q1 = sp.quantile(0.25)
    q3 = sp.quantile(0.75)
    iqr = q3 - q1
    upper_iqr_bound = q3 + 1.5 * iqr
    p99 = sp.quantile(0.99)
    
    iqr_outliers = (sp > upper_iqr_bound).sum()
    p99_outliers = (sp > p99).sum()
    
    print(f"  - 1.5 * IQR Upper Bound: ${upper_iqr_bound:,.2f} | Outliers: {iqr_outliers:,} ({iqr_outliers/len(sp)*100:.2f}%)")
    print(f"  - 99th Percentile Bound: ${p99:,.2f} | Outliers: {p99_outliers:,} ({p99_outliers/len(sp)*100:.2f}%)")
    print(f"  - Decision: Outliers represent valid, high-value commercial/personal debits; retained for regression.")
    
    # 5. Supervised Target & Feature Analysis
    print(f"\n--- 5. SUPERVISED DATASET TARGET & FEATURE STATISTICS ---")
    target = df_sup['next_month_total_spending']
    print(f"Target (next_month_total_spending):")
    print(f"  Mean: ${target.mean():,.2f} | Std: ${target.std():,.2f} | Median: ${target.median():,.2f}")
    print(f"  Min: ${target.min():,.2f} | Max: ${target.max():,.2f} | Skewness: {target.skew():.2f}")
    print(f"  Zero Target Count: {(target == 0).sum():,} ({(target == 0).sum()/len(target)*100:.2f}%)")
    
    # Multicollinearity check
    feature_cols = [
        'spending_t', 'spending_t_minus_1', 'spending_t_minus_2',
        'spending_3m_mean', 'spending_3m_std', 'debit_count_t',
        'income_credit_t', 'ending_balance_t', 'spending_hh_t',
        'spending_st_t', 'spending_in_t', 'spending_lo_t', 'spending_other_t',
        'next_month_total_spending'
    ]
    corr = df_sup[feature_cols].corr()
    
    print(f"\nTarget Correlations (Top Predictors):")
    target_corr = corr['next_month_total_spending'].sort_values(ascending=False)
    for feat, r_val in target_corr.items():
        if feat != 'next_month_total_spending':
            print(f"  - {feat:<22}: r = {r_val:+.4f}")
            
    # 6. Generate Figures
    print(f"\n--- 6. GENERATING PUBLICATION-QUALITY FIGURES ---")
    
    # Figure 1: Spending & Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    sns.histplot(sp[sp > 0], bins=50, kde=True, ax=axes[0], color='#1f77b4', log_scale=True)
    axes[0].set_title('A. Monthly Debit Spending Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Monthly Debit Spending ($S_{u, t}$ in CZK)')
    axes[0].set_ylabel('Density / Frequency')
    
    sns.histplot(target[target > 0], bins=50, kde=True, ax=axes[1], color='#2ca02c', log_scale=True)
    axes[1].set_title('B. Target Spending Distribution ($Y_{u, t+1}$ Log Scale)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Next-Month Total Spending ($Y_{u, t+1}$ in CZK)')
    axes[1].set_ylabel('Density / Frequency')
    
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "eda_spending_distribution.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved Figure 1: {fig1_path}")
    
    # Figure 2: Longitudinal Portfolio Trends
    monthly_trend = df_panel.groupby('year_month_str')[['spending_debit', 'income_credit']].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_trend['year_month_str'], monthly_trend['spending_debit'] / 1e6, label='Total Outgoing Debits', color='#d62728', linewidth=2)
    ax.plot(monthly_trend['year_month_str'], monthly_trend['income_credit'] / 1e6, label='Total Incoming Credits', color='#1f77b4', linewidth=2)
    
    ax.set_title('Longitudinal Portfolio Transaction Aggregates (2013–2018)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Calendar Month')
    ax.set_ylabel('Total Monthly Volume (Millions CZK)')
    ax.set_xticks(monthly_trend['year_month_str'][::6])
    ax.set_xticklabels(monthly_trend['year_month_str'][::6], rotation=45)
    ax.legend(frameon=True, facecolor='white', loc='upper left')
    
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "eda_temporal_trends.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved Figure 2: {fig2_path}")
    
    # Figure 3: Category Spending Share
    fig, ax = plt.subplots(figsize=(10, 5))
    short_cat_names = ['Household\n(HH)', 'Statement\n(ST)', 'Insurance\n(IN)', 'Loan\n(LO)', 'Interest Out\n(IO)', 'Uncategorized\n(OTHER)']
    cat_millions = [t / 1e6 for t in cat_totals]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    
    bars = ax.bar(short_cat_names, cat_millions, color=colors, width=0.6)
    ax.set_title('Category Spending Contribution (Total Expenditure Volume)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Volume (Millions CZK)')
    
    for bar, pct in zip(bars, [t/tot_sp*100 for t in cat_totals]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"${yval:,.1f}M\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax.set_ylim(0, max(cat_millions) * 1.25)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "eda_category_breakdown.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved Figure 3: {fig3_path}")
    
    # Figure 4: Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation (r)'}, annot_kws={"size": 8})
    ax.set_title('Supervised Feature Correlation Matrix & Target Alignment', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "eda_feature_correlation.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved Figure 4: {fig4_path}")

    # Figure 5: Account Volatility & Outlier Profile
    acc_stats = df_panel.groupby('account_id')['spending_debit'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(acc_stats['mean'], acc_stats['std'], alpha=0.5, color='#1f77b4', edgecolors='none', s=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Account Spending Volatility: Mean vs. Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Account Mean Monthly Spending (Log Scale CZK)')
    ax.set_ylabel('Account Spending Std Dev (Log Scale CZK)')
    
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "eda_outlier_profile.png"
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved Figure 5: {fig5_path}")
    
    print("\n" + "=" * 80)
    print("EDA PIPELINE EXECUTED & FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_eda_analysis()
