"""
Statistical utilities for risk metrics and bootstrap inference.

This module provides:
- CVaR (Expected Shortfall) and VaR computation
- Bootstrap confidence intervals
- Convergence slope estimation with error bars
"""

import logging
from typing import Callable, Tuple, Optional, List

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_var(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Value at Risk (VaR) at level alpha.
    
    VaR is the alpha-quantile of the loss distribution.
    
    Parameters
    ----------
    pnl : np.ndarray
        Profit and Loss values. Negative = loss.
    alpha : float
        Significance level (default 5%).
        
    Returns
    -------
    float
        VaR value (the alpha-quantile of PnL).
    """
    return float(np.percentile(pnl, alpha * 100))


def compute_cvar(pnl: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the expected value of PnL in the worst alpha-percentile.
    
    Parameters
    ----------
    pnl : np.ndarray
        Profit and Loss values. Negative = loss.
    alpha : float
        Significance level (default 5%).
        
    Returns
    -------
    cvar : float
        CVaR (expected shortfall).
    var : float
        VaR at the same alpha level.
    """
    var = compute_var(pnl, alpha)
    # CVaR = mean of values below VaR
    tail = pnl[pnl <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return cvar, var


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "percentile"
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    values : np.ndarray
        Sample values.
    stat_fn : Callable
        Statistic function to apply to each bootstrap sample.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level (default 5% for 95% CI).
    seed : int
        Random seed for reproducibility.
    method : str
        CI method: 'percentile' or 'bca' (bias-corrected accelerated).
        
    Returns
    -------
    lower : float
        Lower bound of CI.
    mean : float
        Point estimate (mean of bootstrap distribution).
    upper : float
        Upper bound of CI.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    
    # Generate bootstrap samples
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_sample = values[idx]
        boot_stats.append(stat_fn(boot_sample))
    
    boot_stats = np.array(boot_stats)
    
    if method == "percentile":
        lower = np.percentile(boot_stats, alpha / 2 * 100)
        upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)
    elif method == "bca":
        # Bias-corrected and accelerated (simplified version)
        # Full BCA requires jackknife - using simplified percentile with bias correction
        point_est = stat_fn(values)
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < point_est))
        # Acceleration (simplified)
        za = stats.norm.ppf(alpha / 2)
        zb = stats.norm.ppf(1 - alpha / 2)
        
        # Adjusted percentiles
        p_lower = stats.norm.cdf(2 * z0 + za)
        p_upper = stats.norm.cdf(2 * z0 + zb)
        
        lower = np.percentile(boot_stats, p_lower * 100)
        upper = np.percentile(boot_stats, p_upper * 100)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    mean = float(np.mean(boot_stats))
    
    return float(lower), mean, float(upper)


def estimate_convergence_slope(
    dt_values: np.ndarray,
    errors: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Estimate convergence slope with confidence interval via bootstrap.
    
    Fits log(error) = slope * log(dt) + intercept and returns slope with CI.
    
    Parameters
    ----------
    dt_values : np.ndarray
        Step sizes.
    errors : np.ndarray
        Strong errors at each step size.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Significance level.
    seed : int
        Random seed.
        
    Returns
    -------
    slope : float
        Estimated convergence order.
    slope_lower : float
        Lower bound of slope CI.
    slope_upper : float
        Upper bound of slope CI.
    r_squared : float
        R² of the fit.
    """
    log_dt = np.log(dt_values)
    log_err = np.log(errors)
    
    # Point estimate
    slope, intercept, r_value, _, _ = stats.linregress(log_dt, log_err)
    r_squared = r_value ** 2
    
    # Bootstrap CI for slope
    def fit_slope(indices):
        # Check for valid regression (need at least 2 unique x values)
        x_subset = log_dt[indices]
        if len(np.unique(x_subset)) < 2:
            return slope  # Return point estimate if can't fit
        try:
            s, _, _, _, _ = stats.linregress(x_subset, log_err[indices])
            return s
        except ValueError:
            return slope  # Return point estimate on error
    
    rng = np.random.RandomState(seed)
    n = len(dt_values)
    boot_slopes = []
    
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_slopes.append(fit_slope(idx))
    
    boot_slopes = np.array(boot_slopes)
    slope_lower = np.percentile(boot_slopes, alpha / 2 * 100)
    slope_upper = np.percentile(boot_slopes, (1 - alpha / 2) * 100)
    
    return float(slope), float(slope_lower), float(slope_upper), float(r_squared)


def compute_turnover(deltas: np.ndarray) -> float:
    """
    Compute turnover as sum of absolute delta changes.
    
    Parameters
    ----------
    deltas : np.ndarray
        Hedge ratios of shape (n_paths, n_steps).
        
    Returns
    -------
    float
        Mean turnover across paths.
    """
    if deltas.ndim == 1:
        deltas = deltas.reshape(1, -1)
    
    # Add initial delta of 0
    deltas_with_zero = np.concatenate([np.zeros((deltas.shape[0], 1)), deltas], axis=1)
    turnover_per_path = np.abs(np.diff(deltas_with_zero, axis=1)).sum(axis=1)
    
    return float(np.mean(turnover_per_path))


def ks_test_normality(samples: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for normality.
    
    Parameters
    ----------
    samples : np.ndarray
        Sample values to test.
        
    Returns
    -------
    statistic : float
        KS statistic.
    p_value : float
        P-value (reject normality if p < alpha).
    """
    # Standardize samples
    standardized = (samples - np.mean(samples)) / np.std(samples)
    statistic, p_value = stats.kstest(standardized, 'norm')
    return float(statistic), float(p_value)
