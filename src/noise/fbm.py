"""
Fractional Brownian Motion Generator using the Davies-Harte Algorithm.

This module provides an exact method for generating fBM paths with arbitrary
Hurst parameter H ∈ (0, 1). For rough volatility models, we use H ≈ 0.1.

See docs/fbm.md for mathematical details.
"""

import numpy as np
from numpy.fft import fft, ifft


class DaviesHarte:
    """
    Generate fractional Brownian motion increments using the Davies-Harte algorithm.
    
    The algorithm uses circulant embedding and FFT to achieve O(n log n) complexity
    for generating exact fBM paths.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps (increments to generate).
    batch_size : int
        Number of independent fBM paths to generate.
    H : float
        Hurst exponent. Must be in (0, 1). Use H < 0.5 for rough paths.
    T : float
        Time horizon. The paths will span [0, T].
    
    Attributes
    ----------
    dt : float
        Time step size = T / n_steps.
    eigenvalues : np.ndarray
        Precomputed eigenvalues of the circulant embedding matrix.
    
    Example
    -------
    >>> dh = DaviesHarte(n_steps=256, batch_size=100, H=0.1, T=1.0)
    >>> increments = dh.sample()  # Shape: (100, 256)
    >>> paths = np.cumsum(increments, axis=1)  # Cumulative sum gives fBM paths
    """
    
    def __init__(self, n_steps: int, batch_size: int, H: float, T: float):
        if not 0 < H < 1:
            raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if T <= 0:
            raise ValueError(f"Time horizon T must be positive, got {T}")
        
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.H = H
        self.T = T
        self.dt = T / n_steps
        
        # Precompute eigenvalues for efficiency (reused across samples)
        self.eigenvalues = self._build_eigenvalues()
    
    def _autocovariance(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the autocovariance of fBM increments.
        
        For fBM increments ΔB^H_k = B^H_{(k+1)Δt} - B^H_{kΔt}, the autocovariance is:
        
            γ(k) = (Δt^{2H} / 2) * (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
        
        Parameters
        ----------
        k : np.ndarray
            Lag values at which to compute autocovariance.
        
        Returns
        -------
        np.ndarray
            Autocovariance values γ(k).
        """
        H2 = 2 * self.H
        k = np.abs(k)
        return 0.5 * (self.dt ** H2) * (
            np.abs(k + 1) ** H2 - 2 * np.abs(k) ** H2 + np.abs(k - 1) ** H2
        )
    
    def _build_eigenvalues(self) -> np.ndarray:
        """
        Build the eigenvalues of the circulant embedding matrix.
        
        The circulant matrix is constructed by embedding the Toeplitz covariance
        matrix into a circulant matrix of size 2n. The eigenvalues are computed
        via FFT of the first row.
        
        Returns
        -------
        np.ndarray
            Eigenvalues of the circulant matrix (length 2 * n_steps).
        
        Raises
        ------
        ValueError
            If any eigenvalue is negative (Davies-Harte fails for this n/H combo).
        """
        n = self.n_steps
        
        # Build first row of circulant matrix: [γ(0), γ(1), ..., γ(n), γ(n-1), ..., γ(1)]
        # This embeds the n×n Toeplitz matrix into a 2n×2n circulant matrix
        k = np.arange(n + 1)
        gamma = self._autocovariance(k)
        
        # First row: [γ(0), γ(1), ..., γ(n-1), γ(n), γ(n-1), ..., γ(1)]
        first_row = np.concatenate([gamma, gamma[-2:0:-1]])
        
        # Eigenvalues = FFT of first row (for circulant matrices)
        eigenvalues = fft(first_row).real
        
        # Check non-negativity (required for valid sampling)
        if np.any(eigenvalues < -1e-10):
            min_eig = eigenvalues.min()
            raise ValueError(
                f"Davies-Harte failed: negative eigenvalue {min_eig:.6f}. "
                f"Try increasing n_steps or adjusting H."
            )
        
        # Clip tiny negative values from numerical error
        eigenvalues = np.maximum(eigenvalues, 0)
        
        return eigenvalues
    
    def sample(self) -> np.ndarray:
        """
        Generate a batch of fBM increment paths.
        
        Uses the Davies-Harte algorithm:
        1. Generate complex Gaussian noise in Fourier space
        2. Multiply by sqrt(eigenvalues)
        3. Apply inverse FFT
        4. Extract the first n_steps values as fBM increments
        
        Returns
        -------
        np.ndarray
            Array of shape (batch_size, n_steps) containing fBM increments.
            The variance of each increment is Δt^{2H}.
        """
        n = self.n_steps
        m = 2 * n  # Size of circulant embedding
        
        # Square root of eigenvalues for scaling
        sqrt_eig = np.sqrt(self.eigenvalues)
        
        # Generate samples in Fourier space
        # For real-valued output, we need conjugate symmetry
        # Generate random Fourier coefficients with proper structure
        
        all_increments = np.zeros((self.batch_size, n))
        
        for b in range(self.batch_size):
            # Generate complex Gaussian with conjugate symmetry for real output
            # W[0] and W[n] (if m is even) are real, others are complex conjugate pairs
            W = np.zeros(m, dtype=complex)
            
            # W[0] is real
            W[0] = np.random.randn() * sqrt_eig[0]
            
            # W[n] is real (middle element for even m)
            W[n] = np.random.randn() * sqrt_eig[n]
            
            # For k = 1, ..., n-1: W[k] and W[m-k] are conjugate pairs
            for k in range(1, n):
                real_part = np.random.randn()
                imag_part = np.random.randn()
                W[k] = (real_part + 1j * imag_part) * sqrt_eig[k] / np.sqrt(2)
                W[m - k] = np.conj(W[k])
            
            # Inverse FFT 
            samples = ifft(W).real * np.sqrt(m)
            
            # Extract first n components as fBM increments
            all_increments[b, :] = samples[:n]
        
        return all_increments


def generate_fbm_paths(n_steps: int, batch_size: int, H: float, T: float) -> np.ndarray:
    """
    Convenience function to generate fBM paths (not just increments).
    
    Parameters
    ----------
    n_steps : int
        Number of time steps.
    batch_size : int
        Number of paths to generate.
    H : float
        Hurst exponent.
    T : float
        Time horizon.
    
    Returns
    -------
    np.ndarray
        Array of shape (batch_size, n_steps + 1) with fBM paths starting at 0.
    """
    dh = DaviesHarte(n_steps, batch_size, H, T)
    increments = dh.sample()
    
    # Prepend zeros and compute cumulative sum
    paths = np.zeros((batch_size, n_steps + 1))
    paths[:, 1:] = np.cumsum(increments, axis=1)
    
    return paths
