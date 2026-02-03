#!/usr/bin/env python
"""
Run bootstrap confidence interval analysis.

Produces 95% CIs for key metrics from hedging results.

Usage:
    python experiments/run_bootstrap_ci.py --metric-file outputs/hedging_results.csv
    python experiments/run_bootstrap_ci.py --quick --n-bootstrap 100
"""

import sys
import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    SEED,
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_QUICK_SAMPLES,
)
from src.utils.seed import set_all_seeds
from src.metrics.statistics import bootstrap_ci, compute_cvar

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_bootstrap_cis(
    pnl_arrays: dict,
    n_bootstrap: int,
    alpha: float,
    seed: int
) -> list:
    """
    Compute bootstrap CIs for each strategy's metrics.
    
    Parameters
    ----------
    pnl_arrays : dict
        Dictionary mapping strategy -> PnL array.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level.
    seed : int
        Random seed.
        
    Returns
    -------
    list
        Results for each metric.
    """
    results = []
    
    for strategy, pnl in tqdm(pnl_arrays.items(), desc="Bootstrap"):
        # Mean PnL CI
        lower, mean, upper = bootstrap_ci(
            pnl, lambda x: np.mean(x), n_bootstrap, alpha, seed
        )
        results.append({
            "metric": f"{strategy}_MeanPnL",
            "alpha": alpha,
            "lower": lower,
            "mean": mean,
            "upper": upper,
        })
        
        # CVaR CI
        lower, mean, upper = bootstrap_ci(
            pnl, lambda x: compute_cvar(x, 0.05)[0], n_bootstrap, alpha, seed
        )
        results.append({
            "metric": f"{strategy}_CVaR",
            "alpha": alpha,
            "lower": lower,
            "mean": mean,
            "upper": upper,
        })
        
        # Std PnL CI
        lower, mean, upper = bootstrap_ci(
            pnl, lambda x: np.std(x), n_bootstrap, alpha, seed
        )
        results.append({
            "metric": f"{strategy}_StdPnL",
            "alpha": alpha,
            "lower": lower,
            "mean": mean,
            "upper": upper,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run bootstrap CI analysis")
    parser.add_argument("--metric-file", type=str, default="outputs/hedging_results.csv",
                        help="Path to hedging results CSV")
    parser.add_argument("--pnl-dir", type=str, default="outputs/pnls",
                        help="Directory with PnL arrays")
    parser.add_argument("--out", type=str, default="outputs/bootstrap_ci.csv",
                        help="Output CSV path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for CI")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="Number of bootstrap samples")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    # Configure
    if args.quick:
        n_bootstrap = args.n_bootstrap or BOOTSTRAP_QUICK_SAMPLES
        logger.info("Running in QUICK mode for CI")
    else:
        n_bootstrap = args.n_bootstrap or BOOTSTRAP_N_SAMPLES
    
    set_all_seeds(args.seed)
    
    # Load PnL arrays
    pnl_dir = Path(args.pnl_dir)
    pnl_arrays = {}
    
    if pnl_dir.exists():
        for pnl_file in pnl_dir.glob("*_pnls.npy"):
            strategy = pnl_file.stem.replace("_pnls", "")
            pnl_arrays[strategy] = np.load(pnl_file)
            logger.info(f"Loaded {strategy}: {len(pnl_arrays[strategy])} samples")
    else:
        # Generate synthetic data for testing
        logger.warning(f"PnL directory {pnl_dir} not found, using synthetic data")
        np.random.seed(args.seed)
        pnl_arrays = {
            "BlackScholesDelta": np.random.randn(100) * 0.01 + 0.05,
            "HestonDelta": np.random.randn(100) * 0.012 + 0.048,
            "NaiveHedge": np.random.randn(100) * 0.015 + 0.04,
            "NeuralHedge": np.random.randn(100) * 0.01 + 0.06,
        }
    
    # Compute bootstrap CIs
    results = compute_bootstrap_cis(
        pnl_arrays=pnl_arrays,
        n_bootstrap=n_bootstrap,
        alpha=BOOTSTRAP_ALPHA,
        seed=args.seed
    )
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved results to {args.out}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    for r in results:
        print(f"{r['metric']:30s}: [{r['lower']:.4f}, {r['upper']:.4f}] (mean: {r['mean']:.4f})")


if __name__ == "__main__":
    main()
