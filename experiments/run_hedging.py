#!/usr/bin/env python
"""
Run hedging experiments comparing multiple strategies.

This script compares:
- Neural Hedge (trained RNN)
- Black-Scholes Delta
- Heston Delta  
- Naive Hedge (fixed delta)

Usage:
    python experiments/run_hedging.py --paths 1000 --out outputs/hedging_results.csv
    python experiments/run_hedging.py --quick --paths 100
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    SEED,
    HEDGING_N_PATHS,
    HEDGING_N_STEPS,
    HEDGING_STRIKE_PCT,
    HEDGING_TRANSACTION_COST,
    HEDGING_VOLATILITY,
    HEDGING_T,
    HEDGING_QUICK_PATHS,
    HEDGING_QUICK_STEPS,
    HEDGING_STRATEGIES,
    HESTON_INIT_PARAMS,
)
from src.utils.seed import set_all_seeds
from src.metrics.statistics import compute_cvar, compute_turnover
from src.agents.baselines import BlackScholesDeltaHedge, HestonDeltaHedge, NaiveHedgeAgent
from src.noise.fbm import DaviesHarte
from src.hedging.engine import DeepHedgingAgent, HedgingEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_paths(
    n_paths: int,
    n_steps: int,
    H: float,
    T: float,
    vol: float,
    seed: int
) -> tuple:
    """
    Generate price paths using rough volatility dynamics.
    
    Returns
    -------
    paths : np.ndarray
        Price paths of shape (n_paths, n_steps+1).
    time_grid : np.ndarray
        Time grid of shape (n_steps+1,).
    """
    set_all_seeds(seed)
    
    fbm = DaviesHarte(n_steps=n_steps, batch_size=n_paths, H=H, T=T)
    increments = fbm.sample()
    
    # Geometric-like process
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = 1.0
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(vol * increments[:, t])
    
    time_grid = np.linspace(0, T, n_steps + 1)
    
    return paths, time_grid


def compute_pnl_for_strategy(
    strategy_name: str,
    paths: np.ndarray,
    time_grid: np.ndarray,
    env: HedgingEnvironment,
    neural_agent: DeepHedgingAgent = None,
    heston_params: dict = None
) -> tuple:
    """
    Compute PnL for a given hedging strategy.
    
    Returns
    -------
    pnl : np.ndarray
        PnL values.
    deltas : np.ndarray
        Hedge deltas.
    """
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    
    if strategy_name == "BlackScholesDelta":
        agent = BlackScholesDeltaHedge(
            strike=env.K,
            sigma=env.volatility,
            T=time_grid[-1]
        )
        deltas = agent.compute_deltas(paths, time_grid)
        
    elif strategy_name == "HestonDelta":
        params = heston_params or HESTON_INIT_PARAMS
        agent = HestonDeltaHedge.from_params(params, strike=env.K, T=time_grid[-1])
        deltas = agent.compute_deltas(paths, time_grid)
        
    elif strategy_name == "NaiveHedge":
        agent = NaiveHedgeAgent(strategy="atm", strike=env.K)
        deltas = agent.compute_deltas(paths, time_grid)
        
    elif strategy_name == "NeuralHedge":
        if neural_agent is None:
            raise ValueError("Neural agent required for NeuralHedge strategy")
        
        paths_tensor = torch.tensor(paths, dtype=torch.float32)
        neural_agent.eval()
        with torch.no_grad():
            deltas_tensor = neural_agent(paths_tensor[:, :-1])
        deltas = deltas_tensor.numpy()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Compute PnL using environment
    deltas_tensor = torch.tensor(deltas, dtype=torch.float32)
    paths_tensor = torch.tensor(paths, dtype=torch.float32)
    pnl = env.compute_pnl(paths_tensor, deltas_tensor).numpy()
    
    return pnl, deltas


def main():
    parser = argparse.ArgumentParser(description="Run hedging experiments")
    parser.add_argument("--out", type=str, default="outputs/hedging_results.csv",
                        help="Output CSV path")
    parser.add_argument("--pnl-dir", type=str, default="outputs/pnls",
                        help="Directory to save raw PnL arrays")
    parser.add_argument("--heston-params", type=str, default="outputs/heston_params.json",
                        help="Path to calibrated Heston parameters JSON")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for CI")
    parser.add_argument("--paths", type=int, default=None,
                        help="Number of simulation paths")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap samples for CIs")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    # Import bootstrap_ci here to avoid circular imports
    from src.metrics.statistics import bootstrap_ci
    
    # Configure
    if args.quick:
        n_paths = args.paths or HEDGING_QUICK_PATHS
        n_steps = HEDGING_QUICK_STEPS
        n_bootstrap = 100  # Fewer bootstrap samples in quick mode
        logger.info("Running in QUICK mode for CI")
    else:
        n_paths = args.paths or HEDGING_N_PATHS
        n_steps = HEDGING_N_STEPS
        n_bootstrap = args.n_bootstrap
    
    # Load Heston parameters
    heston_params = HESTON_INIT_PARAMS.copy()
    if Path(args.heston_params).exists():
        with open(args.heston_params, "r") as f:
            heston_params = json.load(f)
        logger.info(f"Loaded Heston params from {args.heston_params}: "
                    f"κ={heston_params.get('kappa', 'N/A'):.2f}, "
                    f"θ={heston_params.get('theta', 'N/A'):.4f}")
    else:
        logger.info("Using default Heston parameters (no calibration file found)")
    
    set_all_seeds(args.seed)
    
    # Generate paths
    logger.info(f"Generating {n_paths} paths with {n_steps} steps...")
    paths, time_grid = generate_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        H=0.1,  # Rough volatility
        T=HEDGING_T,
        vol=HEDGING_VOLATILITY,
        seed=args.seed
    )
    
    # Setup environment
    env = HedgingEnvironment(
        strike_pct=HEDGING_STRIKE_PCT,
        transaction_cost=HEDGING_TRANSACTION_COST,
        initial_price=1.0,
        volatility=HEDGING_VOLATILITY
    )
    
    # Create neural agent (simple untrained for comparison)
    neural_agent = DeepHedgingAgent(input_dim=3, hidden_dim=32, n_layers=2)
    
    # Run all strategies
    results = []
    Path(args.pnl_dir).mkdir(parents=True, exist_ok=True)
    
    strategies = ["BlackScholesDelta", "HestonDelta", "NaiveHedge", "NeuralHedge"]
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("SIGN CONVENTION: Positive PnL = Profit, Negative PnL = Loss")
    logger.info("CVaR (5%): Expected loss in worst 5% of scenarios (more negative = worse)")
    logger.info("=" * 70)
    logger.info("")
    
    for strategy in tqdm(strategies, desc="Strategies"):
        try:
            pnl, deltas = compute_pnl_for_strategy(
                strategy_name=strategy,
                paths=paths,
                time_grid=time_grid,
                env=env,
                neural_agent=neural_agent,
                heston_params=heston_params
            )
            
            # Compute point estimates
            cvar, var = compute_cvar(pnl, alpha=0.05)
            turnover = compute_turnover(deltas)
            mean_pnl = float(np.mean(pnl))
            std_pnl = float(np.std(pnl))
            
            # Bootstrap 95% CIs
            mean_lower, mean_est, mean_upper = bootstrap_ci(
                pnl, lambda x: np.mean(x), n_bootstrap=n_bootstrap, 
                alpha=0.05, seed=args.seed
            )
            
            cvar_lower, cvar_est, cvar_upper = bootstrap_ci(
                pnl, lambda x: compute_cvar(x, 0.05)[0], n_bootstrap=n_bootstrap,
                alpha=0.05, seed=args.seed
            )
            
            results.append({
                "Strategy": strategy,
                "MeanPnL": mean_pnl,
                "MeanPnL_CI_Lower": mean_lower,
                "MeanPnL_CI_Upper": mean_upper,
                "StdPnL": std_pnl,
                "CVaR_5pct": cvar,  # Renamed for clarity
                "CVaR_CI_Lower": cvar_lower,
                "CVaR_CI_Upper": cvar_upper,
                "VaR_5pct": var,
                "Turnover": turnover,
                "Paths": n_paths,
            })
            
            # Save raw PnL
            np.save(Path(args.pnl_dir) / f"{strategy}_pnls.npy", pnl)
            
            # Log with sign interpretation
            sign_mean = "profit" if mean_pnl > 0 else "loss"
            sign_cvar = "profit" if cvar > 0 else "loss"
            logger.info(
                f"{strategy}: Mean={mean_pnl:+.4f} ({sign_mean}) "
                f"[{mean_lower:+.4f}, {mean_upper:+.4f}], "
                f"CVaR={cvar:+.4f} ({sign_cvar}) "
                f"[{cvar_lower:+.4f}, {cvar_upper:+.4f}]"
            )
            
        except Exception as e:
            logger.error(f"Error with {strategy}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logger.info(f"\nSaved results to {args.out}")
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE (for paper)")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<20} {'Mean PnL':>12} {'95% CI':>20} {'CVaR (5%)':>12} {'95% CI':>20}")
    logger.info("-" * 70)
    for r in results:
        mean_ci = f"[{r['MeanPnL_CI_Lower']:+.4f}, {r['MeanPnL_CI_Upper']:+.4f}]"
        cvar_ci = f"[{r['CVaR_CI_Lower']:+.4f}, {r['CVaR_CI_Upper']:+.4f}]"
        logger.info(f"{r['Strategy']:<20} {r['MeanPnL']:>+12.4f} {mean_ci:>20} {r['CVaR_5pct']:>+12.4f} {cvar_ci:>20}")
    
    # Hypothesis tests: compare strategies
    from src.metrics.statistics import bootstrap_paired_test
    
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS TESTS: Neural vs BlackScholes")
    logger.info("H0: No difference | H1: Neural better (one-sided)")
    logger.info("=" * 70)
    
    # Load saved PnLs for comparison
    pnl_files = {
        "NeuralHedge": Path(args.pnl_dir) / "NeuralHedge_pnls.npy",
        "BlackScholesDelta": Path(args.pnl_dir) / "BlackScholesDelta_pnls.npy",
    }
    
    if all(f.exists() for f in pnl_files.values()):
        neural_pnl = np.load(pnl_files["NeuralHedge"])
        bs_pnl = np.load(pnl_files["BlackScholesDelta"])
        
        # Test Mean PnL difference
        mean_diff, mean_p, mean_ci_l, mean_ci_u = bootstrap_paired_test(
            neural_pnl, bs_pnl,
            lambda x: np.mean(x),
            n_bootstrap=n_bootstrap, seed=args.seed,
            alternative="greater"
        )
        
        # Test CVaR difference
        cvar_diff, cvar_p, cvar_ci_l, cvar_ci_u = bootstrap_paired_test(
            neural_pnl, bs_pnl,
            lambda x: compute_cvar(x, 0.05)[0],
            n_bootstrap=n_bootstrap, seed=args.seed,
            alternative="greater"
        )
        
        sig_mean = "***" if mean_p < 0.001 else "**" if mean_p < 0.01 else "*" if mean_p < 0.05 else ""
        sig_cvar = "***" if cvar_p < 0.001 else "**" if cvar_p < 0.01 else "*" if cvar_p < 0.05 else ""
        
        logger.info(f"Mean PnL Difference:  {mean_diff:+.4f}  p = {mean_p:.4f} {sig_mean}")
        logger.info(f"  95% CI for diff:    [{mean_ci_l:+.4f}, {mean_ci_u:+.4f}]")
        logger.info(f"CVaR Difference:      {cvar_diff:+.4f}  p = {cvar_p:.4f} {sig_cvar}")
        logger.info(f"  95% CI for diff:    [{cvar_ci_l:+.4f}, {cvar_ci_u:+.4f}]")
        logger.info("-" * 70)
        logger.info("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        
        # Save hypothesis test results
        hypo_results = {
            "comparison": "NeuralHedge vs BlackScholesDelta",
            "mean_pnl_diff": mean_diff,
            "mean_pnl_pvalue": mean_p,
            "mean_pnl_ci": [mean_ci_l, mean_ci_u],
            "cvar_diff": cvar_diff,
            "cvar_pvalue": cvar_p,
            "cvar_ci": [cvar_ci_l, cvar_ci_u],
        }
        hypo_path = Path(args.out).parent / "hypothesis_tests.json"
        with open(hypo_path, "w") as f:
            json.dump(hypo_results, f, indent=2)
        logger.info(f"\nSaved hypothesis tests to {hypo_path}")


if __name__ == "__main__":
    main()


