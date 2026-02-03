#!/usr/bin/env python
"""
Generate all figures for the paper from CSV outputs.

Usage:
    python scripts/plot_all_figures.py --convergence outputs/convergence_results.csv
    python scripts/plot_all_figures.py --hedging outputs/hedging_results.csv --out figures/
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def plot_convergence(csv_path: str, out_dir: str) -> None:
    """Generate convergence plot (Figure 1 in paper)."""
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {0.05: 'crimson', 0.1: 'orange', 0.2: 'steelblue'}
    markers = {0.05: 's', 0.1: 'o', 0.2: '^'}
    
    for H in df['H'].unique():
        subset = df[df['H'] == H]
        color = colors.get(H, 'gray')
        marker = markers.get(H, 'o')
        
        slope = subset['slope'].iloc[0]
        r2 = subset['R2'].iloc[0]
        slope_lower = subset['slope_lower'].iloc[0]
        slope_upper = subset['slope_upper'].iloc[0]
        
        label = f'H={H} (slope={slope:.2f} [{slope_lower:.2f}, {slope_upper:.2f}])'
        
        ax.loglog(
            subset['dt'], subset['error_mean'],
            marker=marker, color=color, linewidth=2, markersize=8,
            label=label
        )
        
        # Error bars
        ax.fill_between(
            subset['dt'],
            subset['error_mean'] - subset['error_std'],
            subset['error_mean'] + subset['error_std'],
            color=color, alpha=0.2
        )
    
    ax.set_xlabel('Step Size (Δt)', fontsize=12)
    ax.set_ylabel('Strong Error (RMSE)', fontsize=12)
    ax.set_title('Strong Convergence: Rough vs Standard Volatility', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'convergence_regimes.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_hedging_comparison(csv_path: str, out_dir: str) -> None:
    """Generate hedging comparison plot (Figure 2 in paper)."""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'BlackScholesDelta': 'steelblue',
        'HestonDelta': 'orange',
        'NaiveHedge': 'gray',
        'NeuralHedge': 'crimson',
    }
    
    # Mean PnL bar chart
    ax1 = axes[0]
    strategies = df['Strategy'].tolist()
    mean_pnl = df['MeanPnL'].tolist()
    std_pnl = df['StdPnL'].tolist()
    bar_colors = [colors.get(s, 'gray') for s in strategies]
    
    bars = ax1.bar(strategies, mean_pnl, yerr=std_pnl, color=bar_colors, alpha=0.8,
                   capsize=5, edgecolor='black')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean P&L', fontsize=12)
    ax1.set_title('Hedging Performance Comparison', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # CVaR comparison
    ax2 = axes[1]
    cvar = df['CVaR_0.05'].tolist()
    
    bars = ax2.bar(strategies, cvar, color=bar_colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('5% CVaR (higher is better)', fontsize=12)
    ax2.set_title('Tail Risk Comparison', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, cvar):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'hedging_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_bootstrap_ci(csv_path: str, out_dir: str) -> None:
    """Generate bootstrap CI forest plot."""
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = df['metric'].tolist()
    means = df['mean'].tolist()
    lowers = df['lower'].tolist()
    uppers = df['upper'].tolist()
    
    y_pos = np.arange(len(metrics))
    errors = [[m - l for m, l in zip(means, lowers)],
              [u - m for m, u in zip(means, uppers)]]
    
    ax.errorbar(means, y_pos, xerr=errors, fmt='o', capsize=4,
                color='steelblue', markersize=8)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Bootstrap 95% Confidence Intervals', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'bootstrap_ci.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from CSVs")
    parser.add_argument("--convergence", type=str, help="Convergence results CSV")
    parser.add_argument("--hedging", type=str, help="Hedging results CSV")
    parser.add_argument("--bootstrap", type=str, help="Bootstrap CI CSV")
    parser.add_argument("--out", type=str, default="figures/",
                        help="Output directory")
    args = parser.parse_args()
    
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    if args.convergence and Path(args.convergence).exists():
        plot_convergence(args.convergence, args.out)
    
    if args.hedging and Path(args.hedging).exists():
        plot_hedging_comparison(args.hedging, args.out)
    
    if args.bootstrap and Path(args.bootstrap).exists():
        plot_bootstrap_ci(args.bootstrap, args.out)
    
    print(f"\nAll figures saved to {args.out}")


if __name__ == "__main__":
    main()
