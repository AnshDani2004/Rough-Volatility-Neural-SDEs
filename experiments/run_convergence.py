#!/usr/bin/env python
"""
Run convergence experiments for rough volatility.

This script computes strong convergence errors for different Hurst parameters
to demonstrate the breakdown of standard Euler-Maruyama convergence rates.

Usage:
    python experiments/run_convergence.py --out outputs/convergence_results.csv
    python experiments/run_convergence.py --quick --samples 5
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
    CONVERGENCE_H_VALUES,
    CONVERGENCE_DT_POWERS,
    CONVERGENCE_N_FINE,
    CONVERGENCE_N_SAMPLES,
    CONVERGENCE_T,
    CONVERGENCE_QUICK_SAMPLES,
    CONVERGENCE_QUICK_DT_POWERS,
)
from src.utils.seed import set_all_seeds
from src.metrics.statistics import estimate_convergence_slope
from src.noise.fbm import DaviesHarte

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_strong_errors(
    H: float,
    n_fine: int,
    dt_powers: list,
    n_samples: int,
    T: float,
    seed: int
) -> tuple:
    """
    Compute strong convergence errors for an SDE driven by fBM.
    
    We simulate dX = σ X dW^H (geometric fBM).
    
    Parameters
    ----------
    H : float
        Hurst parameter.
    n_fine : int
        Fine grid resolution (ground truth).
    dt_powers : list
        Powers of 2 for coarse grids.
    n_samples : int
        Monte Carlo samples.
    T : float
        Time horizon.
    seed : int
        Random seed.
        
    Returns
    -------
    dt_values : np.ndarray
        Step sizes.
    error_means : np.ndarray
        Mean errors.
    error_stds : np.ndarray
        Std of errors.
    """
    set_all_seeds(seed)
    sigma = 0.2
    
    dt_values = []
    error_means = []
    error_stds = []
    
    for power in tqdm(dt_powers, desc=f"H={H}", leave=False):
        n_coarse = 2 ** power
        dt = T / n_coarse
        dt_values.append(dt)
        
        errors = []
        for _ in range(n_samples):
            # Generate fine grid fBM
            fbm = DaviesHarte(n_steps=n_fine, batch_size=1, H=H, T=T)
            fine_increments = fbm.sample()[0]
            
            # Fine grid SDE
            X_fine = np.zeros(n_fine + 1)
            X_fine[0] = 1.0
            for i in range(n_fine):
                X_fine[i + 1] = X_fine[i] + sigma * X_fine[i] * fine_increments[i]
            
            # Aggregate to coarse grid
            step_ratio = n_fine // n_coarse
            coarse_increments = fine_increments.reshape(n_coarse, step_ratio).sum(axis=1)
            
            # Coarse grid SDE
            X_coarse = np.zeros(n_coarse + 1)
            X_coarse[0] = 1.0
            for i in range(n_coarse):
                X_coarse[i + 1] = X_coarse[i] + sigma * X_coarse[i] * coarse_increments[i]
            
            errors.append((X_fine[-1] - X_coarse[-1]) ** 2)
        
        error_means.append(np.sqrt(np.mean(errors)))
        error_stds.append(np.std(np.sqrt(errors)))
    
    return np.array(dt_values), np.array(error_means), np.array(error_stds)


def main():
    parser = argparse.ArgumentParser(description="Run convergence analysis")
    parser.add_argument("--out", type=str, default="outputs/convergence_results.csv",
                        help="Output CSV path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for CI")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of MC samples")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    # Configure
    if args.quick:
        n_samples = args.samples or CONVERGENCE_QUICK_SAMPLES
        dt_powers = CONVERGENCE_QUICK_DT_POWERS
        logger.info("Running in QUICK mode for CI")
    else:
        n_samples = args.samples or CONVERGENCE_N_SAMPLES
        dt_powers = CONVERGENCE_DT_POWERS
    
    H_values = CONVERGENCE_H_VALUES
    n_fine = CONVERGENCE_N_FINE
    T = CONVERGENCE_T
    
    set_all_seeds(args.seed)
    
    # Run experiments
    results = []
    
    for H in tqdm(H_values, desc="Hurst parameters"):
        dt_vals, err_means, err_stds = compute_strong_errors(
            H=H,
            n_fine=n_fine,
            dt_powers=dt_powers,
            n_samples=n_samples,
            T=T,
            seed=args.seed
        )
        
        # Estimate convergence slope
        slope, slope_lower, slope_upper, r2 = estimate_convergence_slope(
            dt_vals, err_means, seed=args.seed
        )
        
        logger.info(f"H={H}: slope={slope:.3f} [{slope_lower:.3f}, {slope_upper:.3f}], R²={r2:.4f}")
        
        for i, dt in enumerate(dt_vals):
            results.append({
                "H": H,
                "dt": dt,
                "error_mean": err_means[i],
                "error_std": err_stds[i],
                "slope": slope,
                "slope_lower": slope_lower,
                "slope_upper": slope_upper,
                "R2": r2,
                "n_samples": n_samples,
            })
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
