"""
Baseline hedging agents for comparison.

This module provides:
- BlackScholesDeltaHedge: Analytical BS delta
- HestonDeltaHedge: Delta from calibrated Heston model  
- NaiveHedgeAgent: Fixed delta strategies
"""

import logging
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BaseHedgeAgent(ABC):
    """Abstract base class for hedging agents."""
    
    @abstractmethod
    def compute_deltas(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute hedge ratios for given paths.
        
        Parameters
        ----------
        paths : np.ndarray
            Price paths of shape (n_paths, n_steps+1).
        time_grid : np.ndarray
            Time points of shape (n_steps+1,).
            
        Returns
        -------
        np.ndarray
            Hedge ratios of shape (n_paths, n_steps).
        """
        pass


class BlackScholesDeltaHedge(BaseHedgeAgent):
    """
    Black-Scholes delta hedging strategy.
    
    Computes analytical delta for European call option assuming
    constant volatility.
    
    Parameters
    ----------
    strike : float
        Option strike price.
    sigma : float
        Constant volatility.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    """
    
    def __init__(
        self,
        strike: float = 1.0,
        sigma: float = 0.2,
        r: float = 0.0,
        T: float = 1.0
    ):
        self.strike = strike
        self.sigma = sigma
        self.r = r
        self.T = T
        logger.debug(f"Initialized BS delta hedge: K={strike}, σ={sigma}")
    
    def _compute_d1(
        self,
        S: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """Compute d1 in Black-Scholes formula."""
        # Avoid log(0) and division by 0
        S = np.maximum(S, 1e-10)
        tau = np.maximum(tau, 1e-10)
        
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * self.sigma**2) * tau) / \
             (self.sigma * np.sqrt(tau))
        return d1
    
    def compute_deltas(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute BS delta for each timestep.
        
        Parameters
        ----------
        paths : np.ndarray
            Price paths of shape (n_paths, n_steps+1).
        time_grid : np.ndarray
            Time points, should span [0, T].
            
        Returns
        -------
        np.ndarray
            Delta values of shape (n_paths, n_steps).
        """
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        
        # Time to maturity at each point (excluding terminal)
        tau = self.T - time_grid[:-1]  # Shape: (n_steps,)
        
        # Compute delta at each rebalancing point
        deltas = np.zeros((n_paths, n_steps))
        
        for t in range(n_steps):
            S_t = paths[:, t]
            tau_t = tau[t]
            d1 = self._compute_d1(S_t, tau_t)
            deltas[:, t] = norm.cdf(d1)
        
        return deltas


class HestonDeltaHedge(BaseHedgeAgent):
    """
    Heston model delta hedging strategy.
    
    Uses numerical approximation for delta under Heston dynamics.
    
    Parameters
    ----------
    strike : float
        Option strike price.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation between spot and vol.
    v0 : float
        Initial variance.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    """
    
    def __init__(
        self,
        strike: float = 1.0,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
        r: float = 0.0,
        T: float = 1.0
    ):
        self.strike = strike
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.T = T
        logger.debug(f"Initialized Heston delta hedge: κ={kappa}, θ={theta}")
    
    def _heston_delta_approx(
        self,
        S: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Approximate Heston delta using BS delta with instantaneous vol.
        
        For more accurate delta, use characteristic function approach.
        This is a common practitioner approximation.
        """
        # Use instantaneous vol for BS delta
        sigma_inst = np.sqrt(np.maximum(v, 1e-10))
        
        # Avoid numerical issues
        S = np.maximum(S, 1e-10)
        tau = np.maximum(tau, 1e-10)
        
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * sigma_inst**2) * tau) / \
             (sigma_inst * np.sqrt(tau))
        
        return norm.cdf(d1)
    
    def compute_deltas(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        vol_paths: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Heston delta for each timestep.
        
        Parameters
        ----------
        paths : np.ndarray
            Price paths of shape (n_paths, n_steps+1).
        time_grid : np.ndarray
            Time points.
        vol_paths : np.ndarray, optional
            Variance paths. If None, uses v0.
            
        Returns
        -------
        np.ndarray
            Delta values of shape (n_paths, n_steps).
        """
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        tau = self.T - time_grid[:-1]
        
        if vol_paths is None:
            # Use initial variance throughout
            vol_paths = np.full((n_paths, n_points), self.v0)
        
        deltas = np.zeros((n_paths, n_steps))
        
        for t in range(n_steps):
            S_t = paths[:, t]
            v_t = vol_paths[:, t]
            tau_t = tau[t]
            deltas[:, t] = self._heston_delta_approx(S_t, v_t, tau_t)
        
        return deltas
    
    @classmethod
    def from_params(cls, params: dict, strike: float = 1.0, T: float = 1.0):
        """Create instance from parameter dictionary."""
        return cls(
            strike=strike,
            kappa=params.get("kappa", 2.0),
            theta=params.get("theta", 0.04),
            sigma=params.get("sigma", 0.3),
            rho=params.get("rho", -0.7),
            v0=params.get("v0", 0.04),
            T=T
        )


class NaiveHedgeAgent(BaseHedgeAgent):
    """
    Naive hedging strategy with fixed or simple rule-based delta.
    
    Parameters
    ----------
    strategy : str
        One of:
        - 'fixed': Hold constant delta
        - 'atm': Delta = 0.5 always (ATM approximation)
        - 'moneyness': Delta = min(1, max(0, S/K))
    fixed_delta : float
        Delta to use for 'fixed' strategy.
    strike : float
        Strike price (for moneyness strategy).
    """
    
    def __init__(
        self,
        strategy: str = "atm",
        fixed_delta: float = 0.5,
        strike: float = 1.0
    ):
        if strategy not in ["fixed", "atm", "moneyness"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.fixed_delta = fixed_delta
        self.strike = strike
        logger.debug(f"Initialized naive hedge: {strategy}")
    
    def compute_deltas(
        self,
        paths: np.ndarray,
        time_grid: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute naive delta.
        
        Parameters
        ----------
        paths : np.ndarray
            Price paths of shape (n_paths, n_steps+1).
        time_grid : np.ndarray
            Time points (unused for most strategies).
            
        Returns
        -------
        np.ndarray
            Delta values of shape (n_paths, n_steps).
        """
        n_paths, n_points = paths.shape
        n_steps = n_points - 1
        
        if self.strategy == "fixed" or self.strategy == "atm":
            delta = self.fixed_delta if self.strategy == "fixed" else 0.5
            return np.full((n_paths, n_steps), delta)
        
        elif self.strategy == "moneyness":
            # Delta based on moneyness
            deltas = np.zeros((n_paths, n_steps))
            for t in range(n_steps):
                moneyness = paths[:, t] / self.strike
                deltas[:, t] = np.clip(moneyness, 0, 1)
            return deltas
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
