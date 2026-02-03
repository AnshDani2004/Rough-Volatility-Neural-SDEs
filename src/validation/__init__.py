"""Validation utilities package."""

from .convergence_test import (
    run_convergence_analysis,
    compute_strong_error,
    plot_convergence
)

__all__ = [
    "run_convergence_analysis",
    "compute_strong_error",
    "plot_convergence"
]
