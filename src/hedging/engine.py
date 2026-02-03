"""
Deep Hedging Engine.

This module implements a Deep Hedging agent for options hedging,
using an RNN to learn optimal hedge ratios under CVaR risk measure.

See docs/hedging.md for mathematical details.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.stats import norm


def black_scholes_delta(
    S: np.ndarray,
    K: float,
    T: np.ndarray,
    r: float = 0.0,
    sigma: float = 0.2
) -> np.ndarray:
    """
    Compute Black-Scholes delta for a call option.
    
    Parameters
    ----------
    S : np.ndarray
        Spot price(s).
    K : float
        Strike price.
    T : np.ndarray
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    
    Returns
    -------
    np.ndarray
        Delta values.
    """
    T = np.maximum(T, 1e-8)  # Avoid division by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    sigma: float = 0.2
) -> float:
    """
    Compute Black-Scholes call option price.
    
    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    
    Returns
    -------
    float
        Option price.
    """
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class DeepHedgingAgent(nn.Module):
    """
    RNN-based Deep Hedging Agent.
    
    Observes the path and outputs hedge ratios at each timestep.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (typically 3: S_t, t, delta_{t-1}).
    hidden_dim : int
        RNN hidden state dimension.
    n_layers : int
        Number of RNN layers.
    dropout : float
        Dropout probability.
    
    Example
    -------
    >>> agent = DeepHedgingAgent(input_dim=3, hidden_dim=32)
    >>> paths = torch.randn(32, 50, 1)  # (batch, time, dim)
    >>> deltas = agent(paths)  # (batch, 50)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # RNN for processing sequential data
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Output layer: hedge ratio in [-1, 1]
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bound delta to [-1, 1]
        )
    
    def forward(
        self,
        paths: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hedge ratios for a batch of paths.
        
        Parameters
        ----------
        paths : torch.Tensor
            Price paths of shape (batch_size, n_steps) or (batch_size, n_steps, 1).
        time_grid : torch.Tensor, optional
            Time values of shape (n_steps,). If None, uses uniform grid.
        
        Returns
        -------
        torch.Tensor
            Hedge ratios of shape (batch_size, n_steps).
        """
        if paths.dim() == 2:
            paths = paths.unsqueeze(-1)
        
        batch_size, n_steps, _ = paths.shape
        device = paths.device
        
        # Create time grid if not provided
        if time_grid is None:
            time_grid = torch.linspace(0, 1, n_steps, device=device)
        
        # Build input features: (S_t, t, delta_{t-1})
        deltas = []
        h = None  # RNN hidden state
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for t in range(n_steps):
            # Input at timestep t
            s_t = paths[:, t, :]  # (batch, 1)
            t_val = time_grid[t].expand(batch_size, 1)  # (batch, 1)
            
            features = torch.cat([s_t, t_val, prev_delta], dim=-1)  # (batch, 3)
            features = features.unsqueeze(1)  # (batch, 1, 3)
            
            # RNN step
            out, h = self.rnn(features, h)
            
            # Output delta
            delta_t = self.output_layer(out.squeeze(1))  # (batch, 1)
            deltas.append(delta_t)
            prev_delta = delta_t
        
        return torch.cat(deltas, dim=-1)  # (batch, n_steps)


class CVaRLoss(nn.Module):
    """
    Conditional Value at Risk (CVaR) Loss.
    
    Also known as Expected Shortfall, this measures the expected
    loss in the worst alpha-percentile of scenarios.
    
    Parameters
    ----------
    alpha : float
        Confidence level (e.g., 0.05 for worst 5%).
    
    Example
    -------
    >>> loss_fn = CVaRLoss(alpha=0.05)
    >>> pnl = torch.randn(1000)  # PnL samples
    >>> cvar = loss_fn(pnl)
    """
    
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR of PnL distribution.
        
        Parameters
        ----------
        pnl : torch.Tensor
            PnL values of shape (batch_size,).
            Negative values = losses.
        
        Returns
        -------
        torch.Tensor
            Negative CVaR (for minimization, we want to maximize CVaR).
        """
        # Sort PnL to find worst outcomes
        sorted_pnl, _ = torch.sort(pnl)
        
        # Number of samples in worst alpha%
        n = len(sorted_pnl)
        k = max(1, int(n * self.alpha))
        
        # CVaR = mean of worst k samples
        worst_pnl = sorted_pnl[:k]
        cvar = worst_pnl.mean()
        
        # Return negative because we minimize (and want to maximize PnL)
        return -cvar


class HedgingEnvironment:
    """
    Environment for hedging a call option.
    
    Manages the hedging scenario: selling an ATM call and
    computing the hedging P&L with transaction costs.
    
    Parameters
    ----------
    strike_pct : float
        Strike as percentage of initial price (1.0 = ATM).
    transaction_cost : float
        Transaction cost per unit traded.
    initial_price : float
        Initial underlying price (for premium calculation).
    volatility : float
        Volatility for BS pricing.
    
    Example
    -------
    >>> env = HedgingEnvironment(strike_pct=1.0, transaction_cost=0.001)
    >>> pnl = env.compute_pnl(paths, deltas)
    """
    
    def __init__(
        self,
        strike_pct: float = 1.0,
        transaction_cost: float = 0.001,
        initial_price: float = 1.0,
        volatility: float = 0.2
    ):
        self.strike_pct = strike_pct
        self.transaction_cost = transaction_cost
        self.initial_price = initial_price
        self.volatility = volatility
        
        # Compute strike
        self.K = initial_price * strike_pct
        
        # Option premium (what we receive for selling the call)
        self.premium = black_scholes_price(
            S=initial_price,
            K=self.K,
            T=1.0,
            sigma=volatility
        )
    
    def compute_pnl(
        self,
        paths: torch.Tensor,
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hedging P&L for a batch of paths.
        
        PnL = Premium + Hedge Gains - Option Payoff - Transaction Costs
        
        Parameters
        ----------
        paths : torch.Tensor
            Price paths of shape (batch_size, n_steps+1).
        deltas : torch.Tensor
            Hedge ratios of shape (batch_size, n_steps).
        
        Returns
        -------
        torch.Tensor
            PnL for each path, shape (batch_size,).
        """
        batch_size = paths.shape[0]
        n_steps = deltas.shape[1]
        
        # Starting P&L: option premium received
        pnl = torch.full((batch_size,), self.premium, 
                         device=paths.device, dtype=paths.dtype)
        
        # Hedging gains: sum of delta_t * (S_{t+1} - S_t)
        price_changes = paths[:, 1:n_steps+1] - paths[:, :n_steps]  # (batch, n_steps)
        hedge_gains = (deltas * price_changes).sum(dim=1)
        pnl = pnl + hedge_gains
        
        # Option payoff at maturity (we sold a call, so we pay this)
        S_T = paths[:, -1]
        call_payoff = torch.relu(S_T - self.K)
        pnl = pnl - call_payoff
        
        # Transaction costs: proportional to |delta changes|
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0]  # Initial position
        delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        
        # Cost proportional to |change| * price
        tc = self.transaction_cost * torch.abs(delta_changes) * paths[:, :n_steps]
        pnl = pnl - tc.sum(dim=1)
        
        return pnl


def compute_pnl(
    paths: torch.Tensor,
    deltas: torch.Tensor,
    strike: float = 1.0,
    premium: float = 0.0,
    transaction_cost: float = 0.001
) -> torch.Tensor:
    """
    Simplified P&L computation function.
    
    Parameters
    ----------
    paths : torch.Tensor
        Price paths of shape (batch_size, n_steps+1).
    deltas : torch.Tensor
        Hedge ratios of shape (batch_size, n_steps).
    strike : float
        Option strike price.
    premium : float
        Option premium received.
    transaction_cost : float
        Transaction cost per unit.
    
    Returns
    -------
    torch.Tensor
        PnL for each path.
    """
    batch_size = paths.shape[0]
    n_steps = deltas.shape[1]
    
    # Premium received
    pnl = torch.full((batch_size,), premium, device=paths.device, dtype=paths.dtype)
    
    # Hedging gains
    price_changes = paths[:, 1:n_steps+1] - paths[:, :n_steps]
    hedge_gains = (deltas * price_changes).sum(dim=1)
    pnl = pnl + hedge_gains
    
    # Option payoff (call)
    S_T = paths[:, -1]
    pnl = pnl - torch.relu(S_T - strike)
    
    # Transaction costs
    delta_changes = torch.zeros_like(deltas)
    delta_changes[:, 0] = deltas[:, 0]
    delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    tc = transaction_cost * torch.abs(delta_changes) * paths[:, :n_steps]
    pnl = pnl - tc.sum(dim=1)
    
    return pnl


class DeepHedgingTrainer:
    """
    Trainer for the Deep Hedging agent using TSTR protocol.
    
    Parameters
    ----------
    agent : DeepHedgingAgent
        The hedging agent to train.
    env : HedgingEnvironment
        The hedging environment.
    alpha : float
        CVaR confidence level.
    device : str
        Device for training.
    """
    
    def __init__(
        self,
        agent: DeepHedgingAgent,
        env: HedgingEnvironment,
        alpha: float = 0.05,
        device: str = 'cpu'
    ):
        self.agent = agent.to(device)
        self.env = env
        self.loss_fn = CVaRLoss(alpha=alpha)
        self.device = device
        
        self.history = {'train_loss': [], 'test_loss': [], 'epoch': []}
    
    def train_epoch(
        self,
        train_paths: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch on synthetic paths."""
        self.agent.train()
        
        train_paths = train_paths.to(self.device)
        
        # Get hedge ratios (for all but last timestep)
        n_steps = train_paths.shape[1] - 1
        deltas = self.agent(train_paths[:, :-1])
        
        # Compute PnL
        pnl = self.env.compute_pnl(train_paths, deltas)
        
        # CVaR loss
        loss = self.loss_fn(pnl)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_paths: torch.Tensor) -> Dict[str, float]:
        """Evaluate on test paths (real data in TSTR)."""
        self.agent.eval()
        
        with torch.no_grad():
            test_paths = test_paths.to(self.device)
            deltas = self.agent(test_paths[:, :-1])
            pnl = self.env.compute_pnl(test_paths, deltas)
            
            cvar_loss = self.loss_fn(pnl).item()
            
            return {
                'cvar': -cvar_loss,  # Actual CVaR (not negated)
                'mean_pnl': pnl.mean().item(),
                'std_pnl': pnl.std().item(),
                'min_pnl': pnl.min().item(),
                'max_pnl': pnl.max().item()
            }
    
    def train(
        self,
        train_paths: torch.Tensor,
        test_paths: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3,
        print_every: int = 10
    ):
        """
        Full TSTR training loop.
        
        Parameters
        ----------
        train_paths : torch.Tensor
            Synthetic training paths (from generator).
        test_paths : torch.Tensor
            Real test paths (from market data).
        n_epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        print_every : int
            Print frequency.
        """
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        
        print("Deep Hedging Training (TSTR Protocol)")
        print(f"  - Train paths: {train_paths.shape[0]} (synthetic)")
        print(f"  - Test paths: {test_paths.shape[0]} (real)")
        print(f"  - Epochs: {n_epochs}")
        print("-" * 50)
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_paths, optimizer)
            
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch)
            
            if (epoch + 1) % print_every == 0 or epoch == 0:
                test_metrics = self.evaluate(test_paths)
                self.history['test_loss'].append(-test_metrics['cvar'])
                
                print(f"Epoch {epoch+1:4d} | "
                      f"Train CVaR: {-train_loss:.4f} | "
                      f"Test CVaR: {test_metrics['cvar']:.4f} | "
                      f"Test Mean PnL: {test_metrics['mean_pnl']:.4f}")
        
        print("-" * 50)
        final = self.evaluate(test_paths)
        print(f"Final Test Results:")
        print(f"  CVaR: {final['cvar']:.4f}")
        print(f"  Mean PnL: {final['mean_pnl']:.4f}")
        print(f"  Std PnL: {final['std_pnl']:.4f}")
