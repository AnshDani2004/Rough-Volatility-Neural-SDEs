#!/usr/bin/env python
"""
Calibrate Heston model parameters.

Uses method of moments to fit Heston parameters to simulated vol paths.

Usage:
    python experiments/calibrate_heston.py --vol-paths outputs/vol_paths.npy
    python experiments/calibrate_heston.py --quick
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import SEED, HESTON_INIT_PARAMS
from src.utils.seed import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def heston_moment_loss(
    params: np.ndarray,
    target_mean: float,
    target_var: float,
    target_autocorr: float,
    dt: float
) -> float:
    """
    Loss function for Heston calibration via method of moments.
    
    Compares theoretical Heston moments to empirical moments.
    """
    kappa, theta, sigma = params
    
    # Ensure positive parameters
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    
    # Feller condition check
    if 2 * kappa * theta < sigma**2:
        # Penalty for violating Feller condition
        return 1e10
    
    # Theoretical long-run mean
    theoretical_mean = theta
    
    # Theoretical variance (simplified)
    theoretical_var = (sigma**2 * theta) / (2 * kappa)
    
    # Theoretical autocorrelation at lag 1 (simplified)
    theoretical_autocorr = np.exp(-kappa * dt)
    
    # Compute loss
    loss = (
        (theoretical_mean - target_mean)**2 +
        (theoretical_var - target_var)**2 +
        (theoretical_autocorr - target_autocorr)**2
    )
    
    return loss


def calibrate_heston(
    vol_paths: np.ndarray,
    dt: float = 1/252,
    seed: int = 42
) -> dict:
    """
    Calibrate Heston parameters from variance paths.
    
    Parameters
    ----------
    vol_paths : np.ndarray
        Variance paths of shape (n_paths, n_steps).
    dt : float
        Time step.
    seed : int
        Random seed.
        
    Returns
    -------
    dict
        Calibrated parameters.
    """
    set_all_seeds(seed)
    
    # Compute empirical moments
    flat_vol = vol_paths.flatten()
    target_mean = np.mean(flat_vol)
    target_var = np.var(flat_vol)
    
    # Autocorrelation at lag 1
    vol_centered = vol_paths - np.mean(vol_paths, axis=1, keepdims=True)
    autocorr = np.mean(vol_centered[:, :-1] * vol_centered[:, 1:]) / np.var(flat_vol)
    target_autocorr = max(0.01, min(0.99, autocorr))
    
    logger.info(f"Target moments: mean={target_mean:.4f}, var={target_var:.6f}, acf={target_autocorr:.4f}")
    
    # Initial guess
    x0 = np.array([
        HESTON_INIT_PARAMS["kappa"],
        HESTON_INIT_PARAMS["theta"],
        HESTON_INIT_PARAMS["sigma"]
    ])
    
    # Optimize
    result = minimize(
        heston_moment_loss,
        x0,
        args=(target_mean, target_var, target_autocorr, dt),
        method="Nelder-Mead",
        options={"maxiter": 1000}
    )
    
    kappa, theta, sigma = result.x
    
    # Estimate rho from correlation structure (simplified)
    rho = HESTON_INIT_PARAMS["rho"]  # Use default, proper estimation requires asset paths
    
    params = {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "rho": float(rho),
        "v0": float(target_mean),
        "calibration_loss": float(result.fun),
        "success": bool(result.success),
    }
    
    return params


def main():
    parser = argparse.ArgumentParser(description="Calibrate Heston model")
    parser.add_argument("--vol-paths", type=str, default=None,
                        help="Path to variance paths (.npy)")
    parser.add_argument("--out", type=str, default="outputs/heston_params.json",
                        help="Output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with synthetic data")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    set_all_seeds(args.seed)
    
    if args.vol_paths and Path(args.vol_paths).exists():
        vol_paths = np.load(args.vol_paths)
        logger.info(f"Loaded vol paths: {vol_paths.shape}")
    else:
        # Generate synthetic variance paths for testing
        logger.info("Generating synthetic variance paths...")
        n_paths, n_steps = (100, 252) if not args.quick else (10, 50)
        
        # Simple CIR-like process
        kappa, theta, sigma = 2.0, 0.04, 0.3
        dt = 1/252
        
        vol_paths = np.zeros((n_paths, n_steps))
        vol_paths[:, 0] = theta
        
        for t in range(1, n_steps):
            dW = np.random.randn(n_paths) * np.sqrt(dt)
            vol_paths[:, t] = (
                vol_paths[:, t-1] +
                kappa * (theta - vol_paths[:, t-1]) * dt +
                sigma * np.sqrt(np.maximum(vol_paths[:, t-1], 0)) * dW
            )
            vol_paths[:, t] = np.maximum(vol_paths[:, t], 1e-8)
    
    # Calibrate
    params = calibrate_heston(vol_paths, seed=args.seed)
    
    logger.info(f"Calibrated: κ={params['kappa']:.4f}, θ={params['theta']:.4f}, σ={params['sigma']:.4f}")
    
    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
