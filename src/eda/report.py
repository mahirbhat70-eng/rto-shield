"""
RTO Shield -- Stage 2: Exploratory Data Analysis

Reads the temporal splits (train/val/test) and produces diagnostic plots
and tables saved to reports/eda/. Reproducible without a notebook runtime.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


SPLIT_DIR = "data/processed"
OUTPUT_DIR = "reports/eda"


def load_splits():
    """Load train/val/test CSVs."""
    dfs = {}
    for name in ['train', 'val', 'test']:
        path = os.path.join(SPLIT_DIR, f"{name}.csv")
        dfs[name] = pd.read_csv(path, dtype={'pincode': str})
    return dfs


def save_fig(fig, name):
    """Save figure to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/{name}")


def eda_class_balance(dfs):
    """Class balance per split."""
    print("\n1. Class Balance (per split)")
    print("-" * 50)
    for name, df in dfs.items():
        rto = df['rto_label'].sum()
        total = len(df)
        rate = rto / total
        print(f"  {name:5s}: {total:>6,} rows | RTO={rto:>5,} ({rate*100:.1f}%) | "
              f"Delivered={total-rto:>5,} ({(1-rate)*100:.1f}%)")


def eda_missing_values(dfs):
    """Missing value check."""
    print("\n2. Missing Value Check")
    print("-" * 50)
    for name, df in dfs.items():
        n_miss = df.isnull().sum().sum()
        print(f"  {name:5s}: {n_miss} missing values {'(OK)' if n_miss == 0 else '(!!)'}")


def eda_distributions(train_df):
    """Histograms for numerical features."""
    print("\n3. Distribution Plots (train set)")
    cols = ['order_value', 'discount_pct', 'account_age_days',
            'device_cluster_size', 'orders_last_24h']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(train_df[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel('')
    axes[5].axis('off')
    fig.suptitle('Feature Distributions (Train Set)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '01_distributions.png')


def eda_rto_by_categorical(train_df):
    """RTO rate by categorical features."""
    print("\n4. RTO Rate by Categorical Features (train set)")
    cats = ['payment_method', 'category', 'pincode_tier', 'courier_id']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(cats):
        grouped = train_df.groupby(col)['rto_label'].agg(['mean', 'count'])
        grouped.columns = ['rto_rate', 'count']
        grouped = grouped.sort_values('rto_rate', ascending=False)
        bars = axes[i].bar(range(len(grouped)), grouped['rto_rate'],
                           color='coral', edgecolor='black', alpha=0.8)
        axes[i].set_xticks(range(len(grouped)))
        axes[i].set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=9)
        axes[i].set_title(f'RTO Rate by {col}', fontsize=11)
        axes[i].set_ylabel('RTO Rate')
        axes[i].axhline(y=train_df['rto_label'].mean(), color='gray',
                        linestyle='--', alpha=0.7, label='Overall')
        axes[i].legend(fontsize=8)
        # Print table
        print(f"\n  {col}:")
        for idx, row in grouped.iterrows():
            print(f"    {str(idx):20s}  RTO={row['rto_rate']:.3f}  n={int(row['count']):,}")
    fig.suptitle('RTO Rate by Categorical Features (Train Set)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '02_rto_by_categorical.png')


def eda_rto_vs_hist_rate(train_df):
    """RTO rate vs historical_pincode_rto_rate (binned line chart)."""
    print("\n5. RTO Rate vs historical_pincode_rto_rate (train set)")
    train_df = train_df.copy()
    train_df['hist_bin'] = pd.cut(train_df['historical_pincode_rto_rate'],
                                   bins=20, include_lowest=True)
    binned = train_df.groupby('hist_bin', observed=True)['rto_label'].agg(['mean', 'count'])
    binned.columns = ['rto_rate', 'count']

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(binned))
    ax1.plot(x, binned['rto_rate'], 'o-', color='crimson', linewidth=2, label='RTO Rate')
    ax1.set_ylabel('RTO Rate', color='crimson')
    ax1.set_xlabel('historical_pincode_rto_rate (binned)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{iv.left:.2f}" for iv in binned.index], rotation=45, fontsize=8)

    ax2 = ax1.twinx()
    ax2.bar(x, binned['count'], alpha=0.3, color='steelblue', label='Count')
    ax2.set_ylabel('Count', color='steelblue')

    fig.suptitle('RTO Rate vs Historical Pincode RTO Rate (Train)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '03_rto_vs_hist_pincode_rate.png')


def eda_correlation_matrix(train_df):
    """Correlation matrix of numerical features."""
    print("\n6. Correlation Matrix (numerical features, train set)")
    num_cols = ['order_value', 'quantity', 'discount_pct', 'cod_charge',
                'account_age_days', 'prior_orders', 'prior_rto_count',
                'historical_pincode_rto_rate', 'orders_last_24h',
                'device_cluster_size', 'rto_label']
    corr = train_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr.values[i, j]) > 0.5 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Correlation Matrix (Train Set)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '04_correlation_matrix.png')

    # Flag near-duplicates
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = abs(corr.values[i, j])
            if r > 0.85:
                print(f"  WARNING: |corr({num_cols[i]}, {num_cols[j]})| = {r:.3f}")


def main():
    print("=" * 60)
    print("RTO Shield -- Stage 2 EDA Report")
    print("=" * 60)

    dfs = load_splits()
    train_df = dfs['train']

    eda_class_balance(dfs)
    eda_missing_values(dfs)
    eda_distributions(train_df)
    eda_rto_by_categorical(train_df)
    eda_rto_vs_hist_rate(train_df)
    eda_correlation_matrix(train_df)

    print("\n" + "=" * 60)
    print("EDA Complete. All figures saved to reports/eda/")
    print("=" * 60)


if __name__ == "__main__":
    main()
