"""
Hedging Environment for Options.

This module provides a clean interface for hedging experiments,
wrapping the core hedging logic from src.hedging.engine.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HedgingEnv:
    """
    Simplified hedging environment for experiments.
    
    Wraps HedgingEnvironment from src.hedging.engine with a cleaner API.
    
    Parameters
    ----------
    strike : float
        Option strike price.
    transaction_cost : float
        Proportional transaction cost.
    volatility : float
        Implied volatility for premium calculation.
    T : float
        Time to maturity.
    """
    
    def __init__(
        self,
        strike: float = 1.0,
        transaction_cost: float = 0.001,
        volatility: float = 0.2,
        T: float = 1.0
    ):
        from src.hedging.engine import HedgingEnvironment
        
        self.strike = strike
        self.transaction_cost = transaction_cost
        self.volatility = volatility
        self.T = T
        
        self._env = HedgingEnvironment(
            strike_pct=strike,
            transaction_cost=transaction_cost,
            initial_price=1.0,
            volatility=volatility
        )
        
        logger.debug(f"Created HedgingEnv: K={strike}, tc={transaction_cost}")
    
    def compute_pnl(
        self,
        paths: np.ndarray,
        deltas: np.ndarray
    ) -> np.ndarray:
        """
        Compute hedging P&L for given paths and hedge ratios.
        
        Parameters
        ----------
        paths : np.ndarray
            Price paths of shape (n_paths, n_steps+1).
        deltas : np.ndarray
            Hedge ratios of shape (n_paths, n_steps).
            
        Returns
        -------
        np.ndarray
            P&L for each path.
        """
        paths_t = torch.tensor(paths, dtype=torch.float32)
        deltas_t = torch.tensor(deltas, dtype=torch.float32)
        
        pnl = self._env.compute_pnl(paths_t, deltas_t)
        
        return pnl.numpy()
    
    @property
    def premium(self) -> float:
        """Option premium (BS price)."""
        return self._env.premium
