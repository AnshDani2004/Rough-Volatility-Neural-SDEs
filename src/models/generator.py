"""
Rough Neural SDE Generator.

This module implements a Neural SDE driven by fractional Brownian motion,
formulated as a Controlled Differential Equation (CDE):

    dZ_t = f(t, Z_t) dt + g(t, Z_t) du(t)

where u(t) is the fBM control path from Phase 1.

See docs/generator.md for mathematical details.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from noise.fbm import DaviesHarte


class MLP(nn.Module):
    """
    Multi-layer perceptron for drift/diffusion networks.
    
    Architecture: Input → Hidden → Hidden → Output
    with Tanh activations between layers.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (typically 2 for (t, X)).
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output dimension (typically 1).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoughNeuralSDE(nn.Module):
    """
    Neural SDE driven by fractional Brownian motion.
    
    The model learns drift f(t, X) and diffusion g(t, X) networks, then
    solves the CDE using a custom Euler-Heun scheme with pre-computed
    fBM increments as the control signal.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for drift and diffusion MLPs.
    state_dim : int
        Dimension of the state space (typically 1 for log-price).
    
    Attributes
    ----------
    drift_net : MLP
        Network computing μ(t, X).
    diffusion_net : MLP
        Network computing σ(t, X) with softplus output for positivity.
    
    Example
    -------
    >>> dh = DaviesHarte(n_steps=100, batch_size=32, H=0.1, T=1.0)
    >>> model = RoughNeuralSDE(hidden_dim=64)
    >>> paths = model(dh)
    >>> print(paths.shape)  # torch.Size([32, 101])
    """
    
    def __init__(self, hidden_dim: int = 64, state_dim: int = 1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Input: (t, X) where t is normalized time in [0, 1] and X is state
        input_dim = 1 + state_dim
        
        # Drift network: (t, X) -> μ
        self.drift_net = MLP(input_dim, hidden_dim, state_dim)
        
        # Diffusion network: (t, X) -> σ (with softplus for positivity)
        self.diffusion_net = MLP(input_dim, hidden_dim, state_dim)
        
        # Softplus for positive diffusion
        self.softplus = nn.Softplus()
        
        # Small constant for numerical stability
        self.eps = 1e-6
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute drift μ(t, x).
        
        Parameters
        ----------
        t : torch.Tensor
            Normalized time in [0, 1], shape (batch_size, 1).
        x : torch.Tensor
            Current state, shape (batch_size, state_dim).
        
        Returns
        -------
        torch.Tensor
            Drift values, shape (batch_size, state_dim).
        """
        # Concatenate time and state
        tx = torch.cat([t, x], dim=-1)
        return self.drift_net(tx)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion σ(t, x) > 0.
        
        Parameters
        ----------
        t : torch.Tensor
            Normalized time in [0, 1], shape (batch_size, 1).
        x : torch.Tensor
            Current state, shape (batch_size, state_dim).
        
        Returns
        -------
        torch.Tensor
            Positive diffusion values, shape (batch_size, state_dim).
        """
        tx = torch.cat([t, x], dim=-1)
        raw = self.diffusion_net(tx)
        # Ensure positivity with softplus
        return self.softplus(raw) + self.eps
    
    def euler_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        du: torch.Tensor
    ) -> torch.Tensor:
        """
        Single Euler step for the CDE.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time, shape (batch_size, 1).
        x : torch.Tensor
            Current state, shape (batch_size, state_dim).
        dt : float
            Time step size.
        du : torch.Tensor
            fBM increment, shape (batch_size, 1).
        
        Returns
        -------
        torch.Tensor
            Next state, shape (batch_size, state_dim).
        """
        mu = self.drift(t, x)
        sigma = self.diffusion(t, x)
        return x + mu * dt + sigma * du
    
    def euler_heun_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        du: torch.Tensor
    ) -> torch.Tensor:
        """
        Single Euler-Heun (predictor-corrector) step for the CDE.
        
        This provides better accuracy than plain Euler for rough paths.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time, shape (batch_size, 1).
        x : torch.Tensor
            Current state, shape (batch_size, state_dim).
        dt : float
            Time step size.
        du : torch.Tensor
            fBM increment, shape (batch_size, 1).
        
        Returns
        -------
        torch.Tensor
            Next state, shape (batch_size, state_dim).
        """
        # Predictor (Euler)
        mu_0 = self.drift(t, x)
        sigma_0 = self.diffusion(t, x)
        x_pred = x + mu_0 * dt + sigma_0 * du
        
        # Corrector (Heun)
        t_next = t + dt
        mu_1 = self.drift(t_next, x_pred)
        sigma_1 = self.diffusion(t_next, x_pred)
        
        # Average of predictor and corrector
        x_next = x + 0.5 * (mu_0 + mu_1) * dt + 0.5 * (sigma_0 + sigma_1) * du
        
        return x_next
    
    def forward(
        self,
        fbm_generator: DaviesHarte,
        x0: Optional[torch.Tensor] = None,
        method: str = 'euler_heun'
    ) -> torch.Tensor:
        """
        Generate price paths driven by fractional Brownian motion.
        
        Parameters
        ----------
        fbm_generator : DaviesHarte
            The fBM generator from Phase 1 (contains n_steps, batch_size, etc.).
        x0 : torch.Tensor, optional
            Initial state. If None, defaults to zeros.
            Shape: (batch_size, state_dim) or scalar.
        method : str
            Integration method: 'euler' or 'euler_heun'.
        
        Returns
        -------
        torch.Tensor
            Price paths of shape (batch_size, n_steps + 1).
        """
        # Extract parameters from generator
        n_steps = fbm_generator.n_steps
        batch_size = fbm_generator.batch_size
        dt = fbm_generator.dt
        T = fbm_generator.T
        
        # Sample fBM increments
        fbm_increments = fbm_generator.sample()  # (batch_size, n_steps)
        fbm_increments = torch.tensor(fbm_increments, dtype=torch.float32)
        
        # Initialize state
        if x0 is None:
            x = torch.zeros(batch_size, self.state_dim)
        elif isinstance(x0, (int, float)):
            x = torch.full((batch_size, self.state_dim), float(x0))
        else:
            x = x0.clone()
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        
        # Store paths
        paths = [x.clone()]
        
        # Choose step function
        step_fn = self.euler_heun_step if method == 'euler_heun' else self.euler_step
        
        # Integrate the CDE
        for i in range(n_steps):
            # Current normalized time
            t = torch.full((batch_size, 1), i * dt / T)
            
            # fBM increment at this step
            du = fbm_increments[:, i:i+1]
            
            # Take a step
            x = step_fn(t, x, dt, du)
            paths.append(x.clone())
        
        # Stack paths: (n_steps + 1, batch_size, state_dim) -> (batch_size, n_steps + 1)
        paths = torch.stack(paths, dim=1)  # (batch_size, n_steps + 1, state_dim)
        
        # Squeeze state dimension if 1D
        if self.state_dim == 1:
            paths = paths.squeeze(-1)  # (batch_size, n_steps + 1)
        
        return paths


def create_generator(
    hidden_dim: int = 64,
    state_dim: int = 1
) -> RoughNeuralSDE:
    """
    Factory function to create a RoughNeuralSDE generator.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for MLPs.
    state_dim : int
        State dimension.
    
    Returns
    -------
    RoughNeuralSDE
        The initialized generator model.
    """
    return RoughNeuralSDE(hidden_dim=hidden_dim, state_dim=state_dim)
