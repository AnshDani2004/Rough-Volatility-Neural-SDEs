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
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for CI")
    parser.add_argument("--paths", type=int, default=None,
                        help="Number of simulation paths")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    # Configure
    if args.quick:
        n_paths = args.paths or HEDGING_QUICK_PATHS
        n_steps = HEDGING_QUICK_STEPS
        logger.info("Running in QUICK mode for CI")
    else:
        n_paths = args.paths or HEDGING_N_PATHS
        n_steps = HEDGING_N_STEPS
    
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
    
    for strategy in tqdm(strategies, desc="Strategies"):
        try:
            pnl, deltas = compute_pnl_for_strategy(
                strategy_name=strategy,
                paths=paths,
                time_grid=time_grid,
                env=env,
                neural_agent=neural_agent
            )
            
            cvar, var = compute_cvar(pnl, alpha=0.05)
            turnover = compute_turnover(deltas)
            
            results.append({
                "Strategy": strategy,
                "MeanPnL": float(np.mean(pnl)),
                "StdPnL": float(np.std(pnl)),
                "CVaR_0.05": cvar,
                "VaR_0.05": var,
                "Turnover": turnover,
                "Paths": n_paths,
            })
            
            # Save raw PnL
            np.save(Path(args.pnl_dir) / f"{strategy}_pnls.npy", pnl)
            
            logger.info(f"{strategy}: Mean={np.mean(pnl):.4f}, CVaR={cvar:.4f}")
            
        except Exception as e:
            logger.error(f"Error with {strategy}: {e}")
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
