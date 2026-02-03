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


def bootstrap_paired_test(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided"
) -> Tuple[float, float, float, float]:
    """
    Bootstrap hypothesis test for paired strategy comparison.
    
    Tests H0: stat(A) = stat(B) vs H1: stat(A) ≠ stat(B) (or one-sided).
    Uses paired bootstrap: resample pairs (a_i, b_i) together to preserve correlation.
    
    Parameters
    ----------
    pnl_a : np.ndarray
        PnL values for strategy A (e.g., NeuralHedge).
    pnl_b : np.ndarray
        PnL values for strategy B (e.g., BlackScholes).
    stat_fn : Callable
        Statistic function (e.g., np.mean or lambda x: compute_cvar(x, 0.05)[0]).
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed.
    alternative : str
        'two-sided', 'greater' (A > B), or 'less' (A < B).
        
    Returns
    -------
    diff : float
        Observed difference stat(A) - stat(B).
    p_value : float
        P-value for the test.
    ci_lower : float
        Lower 95% CI for the difference.
    ci_upper : float
        Upper 95% CI for the difference.
    """
    if len(pnl_a) != len(pnl_b):
        raise ValueError("PnL arrays must have same length for paired test")
    
    n = len(pnl_a)
    rng = np.random.RandomState(seed)
    
    # Observed difference
    obs_diff = stat_fn(pnl_a) - stat_fn(pnl_b)
    
    # Paired bootstrap: resample indices together
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_a = pnl_a[idx]
        boot_b = pnl_b[idx]
        boot_diff = stat_fn(boot_a) - stat_fn(boot_b)
        boot_diffs.append(boot_diff)
    
    boot_diffs = np.array(boot_diffs)
    
    # Center bootstrap distribution at zero for p-value calculation
    # (null hypothesis: no difference)
    centered_diffs = boot_diffs - np.mean(boot_diffs)
    
    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(centered_diffs) >= np.abs(obs_diff))
    elif alternative == "greater":
        # H1: A > B (diff > 0)
        p_value = np.mean(centered_diffs >= obs_diff)
    elif alternative == "less":
        # H1: A < B (diff < 0)
        p_value = np.mean(centered_diffs <= obs_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    # 95% CI for the difference (not centered)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return float(obs_diff), float(p_value), float(ci_lower), float(ci_upper)


def compare_strategies(
    pnl_dict: dict,
    baseline: str = "BlackScholesDelta",
    n_bootstrap: int = 1000,
    seed: int = 42
):
    """
    Compare all strategies against a baseline using bootstrap tests.
    
    Parameters
    ----------
    pnl_dict : dict
        Dictionary mapping strategy names to PnL arrays.
    baseline : str
        Baseline strategy to compare against.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed.
        
    Returns
    -------
    pd.DataFrame
        Comparison results with differences, p-values, and CIs.
    """
    import pandas as pd
    
    if baseline not in pnl_dict:
        raise ValueError(f"Baseline {baseline} not in pnl_dict")
    
    baseline_pnl = pnl_dict[baseline]
    results = []
    
    for strategy, pnl in pnl_dict.items():
        if strategy == baseline:
            continue
        
        # Test for Mean PnL difference
        mean_diff, mean_p, mean_ci_l, mean_ci_u = bootstrap_paired_test(
            pnl, baseline_pnl, 
            lambda x: np.mean(x),
            n_bootstrap=n_bootstrap, seed=seed,
            alternative="greater"  # H1: strategy better than baseline
        )
        
        # Test for CVaR difference (higher CVaR = better, less tail risk)
        cvar_diff, cvar_p, cvar_ci_l, cvar_ci_u = bootstrap_paired_test(
            pnl, baseline_pnl,
            lambda x: compute_cvar(x, 0.05)[0],
            n_bootstrap=n_bootstrap, seed=seed,
            alternative="greater"  # H1: strategy has higher CVaR (better)
        )
        
        results.append({
            "Strategy": strategy,
            "vs_Baseline": baseline,
            "MeanPnL_Diff": mean_diff,
            "MeanPnL_pvalue": mean_p,
            "MeanPnL_CI": f"[{mean_ci_l:+.4f}, {mean_ci_u:+.4f}]",
            "CVaR_Diff": cvar_diff,
            "CVaR_pvalue": cvar_p,
            "CVaR_CI": f"[{cvar_ci_l:+.4f}, {cvar_ci_u:+.4f}]",
        })
    
    return pd.DataFrame(results)

