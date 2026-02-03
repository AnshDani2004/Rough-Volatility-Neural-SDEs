"""
Signature-based Loss Functions for Path Distributions.

This module implements the Signature MMD loss, which compares distributions
of paths using their log-signature representations. This is superior to
point-wise losses like MSE for generative models of stochastic processes.

See docs/signature.md for mathematical details.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Try to import iisignature for log-signature computation
try:
    import iisignature
    HAS_IISIGNATURE = True
except ImportError:
    HAS_IISIGNATURE = False
    import warnings
    warnings.warn(
        "iisignature not available. Using fallback signature computation. "
        "Install with: pip install iisignature"
    )


class LogSignature:
    """
    Compute log-signatures of paths.
    
    Uses iisignature if available and working, otherwise falls back to
    a manual implementation using increments and their products.
    
    Parameters
    ----------
    depth : int
        Truncation depth for the log-signature (typically 4-6).
    
    Example
    -------
    >>> logsig = LogSignature(depth=4)
    >>> paths = torch.randn(32, 100, 2)  # (batch, time, channels)
    >>> sigs = logsig(paths)  # (batch, sig_dim)
    """
    
    def __init__(self, depth: int = 4):
        self.depth = depth
        self._prepared = {}  # Cache prepared objects by dimension
        self._use_fallback = {}  # Track if fallback should be used per dimension
    
    def _get_prepared(self, dim: int):
        """Get or create prepared object for given dimension."""
        if dim not in self._prepared:
            if HAS_IISIGNATURE and dim not in self._use_fallback:
                try:
                    self._prepared[dim] = iisignature.prepare(dim, self.depth)
                except RuntimeError:
                    # iisignature failed, use fallback
                    self._use_fallback[dim] = True
                    self._prepared[dim] = None
            else:
                self._prepared[dim] = None
        return self._prepared[dim]
    
    def _should_use_fallback(self, dim: int) -> bool:
        """Check if fallback should be used for this dimension."""
        return not HAS_IISIGNATURE or dim in self._use_fallback
    
    def signature_dim(self, dim: int) -> int:
        """
        Compute the dimension of the log-signature for given input dimension.
        
        Parameters
        ----------
        dim : int
            Input path dimension.
        
        Returns
        -------
        int
            Dimension of the log-signature.
        """
        # Trigger fallback detection by attempting to prepare
        self._get_prepared(dim)
        
        if HAS_IISIGNATURE and not self._should_use_fallback(dim):
            try:
                return iisignature.logsiglength(dim, self.depth)
            except RuntimeError:
                self._use_fallback[dim] = True
        
        # Fallback dimension calculation
        # Level 1: dim
        # Level 2: dim*(dim+1)/2 (upper triangular cross terms)
        # Level 3: dim (squared increments)
        # Level 4: dim (cubed increments)
        return dim + (dim * (dim + 1)) // 2 + dim + dim
    
    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Compute log-signatures for a batch of paths.
        
        Parameters
        ----------
        paths : torch.Tensor
            Paths of shape (batch_size, n_steps, dim).
        
        Returns
        -------
        torch.Tensor
            Log-signatures of shape (batch_size, sig_dim).
        """
        batch_size, n_steps, dim = paths.shape
        
        # Convert to numpy
        paths_np = paths.detach().cpu().numpy()
        
        # Try iisignature first if available
        if HAS_IISIGNATURE and not self._should_use_fallback(dim):
            try:
                s = self._get_prepared(dim)
                if s is not None:
                    logsigs = []
                    for i in range(batch_size):
                        logsig = iisignature.logsig(paths_np[i], s)
                        logsigs.append(logsig)
                    logsigs = np.stack(logsigs, axis=0)
                    return torch.tensor(logsigs, dtype=paths.dtype, device=paths.device)
            except RuntimeError:
                # Mark for fallback
                self._use_fallback[dim] = True
        
        # Use fallback
        logsigs = self._fallback_signature(paths_np)
        return torch.tensor(logsigs, dtype=paths.dtype, device=paths.device)
    
    def _fallback_signature(self, paths: np.ndarray) -> np.ndarray:
        """
        Fallback signature computation when iisignature is unavailable.
        
        Uses increments and their products as a simple approximation.
        
        Parameters
        ----------
        paths : np.ndarray
            Paths of shape (batch_size, n_steps, dim).
        
        Returns
        -------
        np.ndarray
            Approximate signatures of shape (batch_size, sig_dim).
        """
        batch_size, n_steps, dim = paths.shape
        
        # Compute increments
        increments = np.diff(paths, axis=1)  # (batch, n_steps-1, dim)
        
        features = []
        
        # Level 1: Sum of increments (≈ first signature term)
        level1 = increments.sum(axis=1)  # (batch, dim)
        features.append(level1)
        
        # Level 2: Cross terms (≈ second signature term)
        for i in range(dim):
            for j in range(dim):
                if i <= j:
                    # Approximate iterated integral
                    term = np.cumsum(increments[:, :, i], axis=1) * increments[:, :, j]
                    level2 = term.sum(axis=1)  # (batch,)
                    features.append(level2.reshape(-1, 1))
        
        # Level 3 and 4: Higher order terms (simplified)
        # We use powers and products of increments
        inc_squared = increments ** 2
        level3 = inc_squared.sum(axis=1)  # (batch, dim)
        features.append(level3)
        
        inc_cubed = increments ** 3
        level4 = inc_cubed.sum(axis=1)  # (batch, dim)
        features.append(level4)
        
        return np.concatenate(features, axis=1)


def add_time_channel(paths: torch.Tensor) -> torch.Tensor:
    """
    Add a time channel to paths.
    
    Signature computation benefits from having time as an explicit channel,
    making the signature invariant to time reparametrization.
    
    Parameters
    ----------
    paths : torch.Tensor
        Paths of shape (batch_size, n_steps) or (batch_size, n_steps, dim).
    
    Returns
    -------
    torch.Tensor
        Paths with time channel, shape (batch_size, n_steps, dim+1).
    """
    if paths.dim() == 2:
        paths = paths.unsqueeze(-1)  # (batch, n_steps, 1)
    
    batch_size, n_steps, dim = paths.shape
    
    # Create normalized time channel
    time = torch.linspace(0, 1, n_steps, device=paths.device, dtype=paths.dtype)
    time = time.unsqueeze(0).unsqueeze(-1).expand(batch_size, n_steps, 1)
    
    # Concatenate time with paths
    return torch.cat([time, paths], dim=-1)


class SigMMDLoss(nn.Module):
    """
    Signature Maximum Mean Discrepancy Loss.
    
    Computes the MMD between two sets of paths using their log-signatures.
    This loss can be used to train generative models to match path distributions.
    
    Parameters
    ----------
    depth : int
        Log-signature truncation depth.
    add_time : bool
        Whether to add a time channel to paths.
    
    Example
    -------
    >>> loss_fn = SigMMDLoss(depth=4)
    >>> real_paths = torch.randn(32, 100)
    >>> gen_paths = torch.randn(32, 100)
    >>> loss = loss_fn(real_paths, gen_paths)
    """
    
    def __init__(self, depth: int = 4, add_time: bool = True):
        super().__init__()
        self.depth = depth
        self.add_time = add_time
        self.logsig = LogSignature(depth=depth)
    
    def forward(
        self,
        real_paths: torch.Tensor,
        gen_paths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Signature MMD loss.
        
        Parameters
        ----------
        real_paths : torch.Tensor
            Real market paths, shape (batch_size, n_steps) or (batch_size, n_steps, dim).
        gen_paths : torch.Tensor
            Generated paths, same shape as real_paths.
        
        Returns
        -------
        torch.Tensor
            Scalar MMD loss value.
        """
        # Add time channel if requested
        if self.add_time:
            real_paths = add_time_channel(real_paths)
            gen_paths = add_time_channel(gen_paths)
        elif real_paths.dim() == 2:
            real_paths = real_paths.unsqueeze(-1)
            gen_paths = gen_paths.unsqueeze(-1)
        
        # Compute log-signatures
        real_sigs = self.logsig(real_paths)  # (batch, sig_dim)
        gen_sigs = self.logsig(gen_paths)    # (batch, sig_dim)
        
        # Compute MMD
        # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
        # Using L2 kernel: k(x,y) = ||x - y||^2
        
        # Method 1: L2 distance on expected signatures
        real_mean = real_sigs.mean(dim=0)  # (sig_dim,)
        gen_mean = gen_sigs.mean(dim=0)    # (sig_dim,)
        
        mmd_loss = torch.sum((real_mean - gen_mean) ** 2)
        
        return mmd_loss
    
    def compute_full_mmd(
        self,
        real_paths: torch.Tensor,
        gen_paths: torch.Tensor,
        kernel: str = 'rbf',
        bandwidth: float = 1.0
    ) -> torch.Tensor:
        """
        Compute full MMD with explicit kernel computation.
        
        This is more expensive but more accurate than the mean-based approximation.
        
        Parameters
        ----------
        real_paths : torch.Tensor
            Real market paths.
        gen_paths : torch.Tensor
            Generated paths.
        kernel : str
            Kernel type: 'rbf' or 'linear'.
        bandwidth : float
            RBF kernel bandwidth.
        
        Returns
        -------
        torch.Tensor
            Scalar MMD loss value.
        """
        # Add time channel if requested
        if self.add_time:
            real_paths = add_time_channel(real_paths)
            gen_paths = add_time_channel(gen_paths)
        elif real_paths.dim() == 2:
            real_paths = real_paths.unsqueeze(-1)
            gen_paths = gen_paths.unsqueeze(-1)
        
        # Compute log-signatures
        real_sigs = self.logsig(real_paths)  # (n, sig_dim)
        gen_sigs = self.logsig(gen_paths)    # (m, sig_dim)
        
        n = real_sigs.shape[0]
        m = gen_sigs.shape[0]
        
        if kernel == 'linear':
            # Linear kernel: k(x,y) = x^T y
            k_xx = torch.mm(real_sigs, real_sigs.t())
            k_yy = torch.mm(gen_sigs, gen_sigs.t())
            k_xy = torch.mm(real_sigs, gen_sigs.t())
        else:  # RBF kernel
            # k(x,y) = exp(-||x-y||^2 / (2 * bandwidth^2))
            def rbf_kernel(x, y):
                x_sq = (x ** 2).sum(dim=1, keepdim=True)
                y_sq = (y ** 2).sum(dim=1, keepdim=True)
                dists = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())
                return torch.exp(-dists / (2 * bandwidth ** 2))
            
            k_xx = rbf_kernel(real_sigs, real_sigs)
            k_yy = rbf_kernel(gen_sigs, gen_sigs)
            k_xy = rbf_kernel(real_sigs, gen_sigs)
        
        # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
        # Exclude diagonal for unbiased estimate
        mmd = (
            (k_xx.sum() - k_xx.trace()) / (n * (n - 1)) +
            (k_yy.sum() - k_yy.trace()) / (m * (m - 1)) -
            2 * k_xy.mean()
        )
        
        return mmd


class SignatureL2Loss(nn.Module):
    """
    Simple L2 loss on expected log-signatures.
    
    A simplified version of SigMMDLoss that directly compares
    mean log-signatures without explicit kernel computation.
    
    Parameters
    ----------
    depth : int
        Log-signature truncation depth.
    add_time : bool
        Whether to add a time channel.
    """
    
    def __init__(self, depth: int = 4, add_time: bool = True):
        super().__init__()
        self.depth = depth
        self.add_time = add_time
        self.logsig = LogSignature(depth=depth)
    
    def forward(
        self,
        real_paths: torch.Tensor,
        gen_paths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 distance between expected log-signatures.
        
        Parameters
        ----------
        real_paths : torch.Tensor
            Real paths.
        gen_paths : torch.Tensor
            Generated paths.
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Add time channel if requested
        if self.add_time:
            real_paths = add_time_channel(real_paths)
            gen_paths = add_time_channel(gen_paths)
        elif real_paths.dim() == 2:
            real_paths = real_paths.unsqueeze(-1)
            gen_paths = gen_paths.unsqueeze(-1)
        
        # Compute log-signatures
        real_sigs = self.logsig(real_paths)
        gen_sigs = self.logsig(gen_paths)
        
        # L2 distance between means
        loss = torch.sum((real_sigs.mean(dim=0) - gen_sigs.mean(dim=0)) ** 2)
        
        return loss
