"""
Numerical Convergence Analysis for Rough Volatility.

This module empirically demonstrates the breakdown of standard Euler-Maruyama
convergence rates when H < 0.5, validating the need for specialized approaches.

See docs/convergence.md for mathematical details.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from scipy import stats
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from noise.fbm import DaviesHarte


def compute_strong_error(
    H: float,
    n_fine: int = 2**14,
    coarse_powers: List[int] = None,
    n_samples: int = 100,
    T: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute strong convergence error for SDE discretization.
    
    We simulate an SDE: dX = X * dW^H (geometric fBM)
    
    Parameters
    ----------
    H : float
        Hurst parameter.
    n_fine : int
        Number of steps for ground truth (very fine grid).
    coarse_powers : List[int]
        Powers of 2 for coarse grids (e.g., [4, 5, 6, ...]).
    n_samples : int
        Number of Monte Carlo samples for error estimation.
    T : float
        Time horizon.
    
    Returns
    -------
    dt_values : np.ndarray
        Step sizes for each coarse grid.
    errors : np.ndarray
        Strong errors (RMSE at terminal time).
    """
    if coarse_powers is None:
        coarse_powers = list(range(4, 11))  # 2^4 to 2^10
    
    dt_values = []
    errors = []
    
    for power in coarse_powers:
        n_coarse = 2 ** power
        dt_fine = T / n_fine
        dt_coarse = T / n_coarse
        dt_values.append(dt_coarse)
        
        # Compute strong error over many samples
        squared_errors = []
        
        for _ in range(n_samples):
            # Generate fBM increments on fine grid
            fbm_fine = DaviesHarte(n_steps=n_fine, batch_size=1, H=H, T=T)
            fine_increments = fbm_fine.sample()[0]
            
            # Simulate SDE on fine grid (ground truth): dX = sigma * X * dW^H
            sigma = 0.2
            X_fine = np.zeros(n_fine + 1)
            X_fine[0] = 1.0
            for i in range(n_fine):
                X_fine[i + 1] = X_fine[i] + sigma * X_fine[i] * fine_increments[i]
            X_true = X_fine[-1]
            
            # Aggregate increments to coarse grid
            step_ratio = n_fine // n_coarse
            coarse_increments = fine_increments.reshape(n_coarse, step_ratio).sum(axis=1)
            
            # Simulate SDE on coarse grid (Euler-Maruyama)
            X_coarse = np.zeros(n_coarse + 1)
            X_coarse[0] = 1.0
            for i in range(n_coarse):
                X_coarse[i + 1] = X_coarse[i] + sigma * X_coarse[i] * coarse_increments[i]
            X_approx = X_coarse[-1]
            
            squared_errors.append((X_true - X_approx) ** 2)
        
        # RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        errors.append(rmse)
    
    return np.array(dt_values), np.array(errors)


def fit_convergence_order(
    dt_values: np.ndarray,
    errors: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit convergence order via linear regression on log-log scale.
    
    Parameters
    ----------
    dt_values : np.ndarray
        Step sizes.
    errors : np.ndarray
        Strong errors.
    
    Returns
    -------
    slope : float
        Convergence order (slope on log-log plot).
    intercept : float
        Y-intercept.
    r_squared : float
        R² of the fit.
    """
    log_dt = np.log(dt_values)
    log_err = np.log(errors)
    
    slope, intercept, r_value, _, _ = stats.linregress(log_dt, log_err)
    
    return slope, intercept, r_value ** 2


def plot_convergence(
    results: Dict[float, Tuple[np.ndarray, np.ndarray]],
    save_path: str = None,
    show: bool = True
) -> None:
    """
    Generate log-log convergence plot.
    
    Parameters
    ----------
    results : Dict[float, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping H -> (dt_values, errors).
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {0.5: 'steelblue', 0.1: 'crimson', 0.3: 'orange'}
    markers = {0.5: 'o', 0.1: 's', 0.3: '^'}
    
    for H, (dt_values, errors) in sorted(results.items(), reverse=True):
        # Fit convergence order
        slope, intercept, r2 = fit_convergence_order(dt_values, errors)
        
        # Plot data points
        color = colors.get(H, 'gray')
        marker = markers.get(H, 'o')
        label = f'H={H} (slope={slope:.2f}, R²={r2:.3f})'
        
        ax.loglog(dt_values, errors, marker=marker, color=color, 
                  linewidth=2, markersize=8, label=label)
        
        # Plot regression line
        dt_fit = np.array([dt_values.min(), dt_values.max()])
        err_fit = np.exp(intercept) * dt_fit ** slope
        ax.loglog(dt_fit, err_fit, '--', color=color, alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Step Size (Δt)', fontsize=12)
    ax.set_ylabel('Strong Error (RMSE)', fontsize=12)
    ax.set_title('Strong Convergence: Standard vs Rough Volatility\n'
                 '(Slope ≈ H confirms theoretical predictions)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()


def run_convergence_analysis(
    H_values: List[float] = None,
    n_samples: int = 100,
    save_path: str = None,
    verbose: bool = True
) -> Dict[float, Dict]:
    """
    Run full convergence analysis for multiple Hurst parameters.
    
    Parameters
    ----------
    H_values : List[float]
        Hurst parameters to test.
    n_samples : int
        Number of Monte Carlo samples.
    save_path : str, optional
        Path to save the figure.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Dict[float, Dict]
        Results for each H value.
    """
    if H_values is None:
        H_values = [0.5, 0.1]  # Standard vs Rough
    
    results = {}
    analysis = {}
    
    for H in H_values:
        if verbose:
            print(f"Computing convergence for H={H}...")
        
        dt_values, errors = compute_strong_error(H=H, n_samples=n_samples)
        slope, intercept, r2 = fit_convergence_order(dt_values, errors)
        
        results[H] = (dt_values, errors)
        analysis[H] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r2,
            'dt_values': dt_values,
            'errors': errors
        }
        
        if verbose:
            print(f"  Convergence order: {slope:.3f}")
            print(f"  R²: {r2:.4f}")
    
    # Plot
    plot_convergence(results, save_path=save_path, show=True)
    
    # Summary
    if verbose:
        print("\n" + "=" * 50)
        print("CONVERGENCE ANALYSIS SUMMARY")
        print("=" * 50)
        for H in H_values:
            print(f"\nH = {H}:")
            print(f"  Convergence order (slope): {analysis[H]['slope']:.3f}")
            print(f"  Expected (theory): ~{H:.1f}")
            match = "✓ Matches" if abs(analysis[H]['slope'] - H) < 0.15 else "✗ Differs"
            print(f"  {match} theoretical prediction")
        
        print("\n" + "=" * 50)
        if 0.1 in H_values and 0.5 in H_values:
            ratio = analysis[0.5]['slope'] / analysis[0.1]['slope']
            print(f"Rough volatility (H=0.1) converges ~{ratio:.1f}x slower!")
            print("This validates the need for specialized numerical methods.")
        print("=" * 50)
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run convergence analysis")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--output", type=str, default="figures/convergence_regimes.png",
                        help="Output path for plot")
    args = parser.parse_args()
    
    run_convergence_analysis(
        H_values=[0.5, 0.1],
        n_samples=args.samples,
        save_path=args.output
    )
