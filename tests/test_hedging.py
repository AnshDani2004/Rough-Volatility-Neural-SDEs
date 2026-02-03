"""
Tests for the Deep Hedging module.

See docs/hedging.md for details.
"""

import torch
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hedging.engine import (
    DeepHedgingAgent,
    CVaRLoss,
    HedgingEnvironment,
    compute_pnl,
    black_scholes_delta
)


class TestBlackScholesDelta:
    """Test Black-Scholes delta calculation."""
    
    def test_atm_delta_near_half(self):
        """ATM option with T=1 should have delta ≈ 0.5."""
        delta = black_scholes_delta(
            S=np.array([100.0]),
            K=100.0,
            T=np.array([1.0]),
            sigma=0.2
        )
        assert 0.5 < delta[0] < 0.7, f"ATM delta should be ~0.5-0.6, got {delta[0]}"
    
    def test_deep_itm_delta_near_one(self):
        """Deep ITM option should have delta ≈ 1."""
        delta = black_scholes_delta(
            S=np.array([150.0]),
            K=100.0,
            T=np.array([0.1]),
            sigma=0.2
        )
        assert delta[0] > 0.95, f"Deep ITM delta should be ~1, got {delta[0]}"
    
    def test_deep_otm_delta_near_zero(self):
        """Deep OTM option should have delta ≈ 0."""
        delta = black_scholes_delta(
            S=np.array([50.0]),
            K=100.0,
            T=np.array([0.1]),
            sigma=0.2
        )
        assert delta[0] < 0.05, f"Deep OTM delta should be ~0, got {delta[0]}"


class TestDeepHedgingAgent:
    """Test the RNN-based hedging agent."""
    
    def test_output_shape(self):
        """Verify output shape matches input timesteps."""
        agent = DeepHedgingAgent(input_dim=3, hidden_dim=16)
        
        batch_size = 32
        n_steps = 50
        paths = torch.randn(batch_size, n_steps)
        
        deltas = agent(paths)
        
        assert deltas.shape == (batch_size, n_steps), (
            f"Expected shape ({batch_size}, {n_steps}), got {deltas.shape}"
        )
    
    def test_output_bounded(self):
        """Verify delta output is bounded in [-1, 1]."""
        agent = DeepHedgingAgent(input_dim=3, hidden_dim=16)
        
        paths = torch.randn(100, 50) * 10  # Large values
        deltas = agent(paths)
        
        assert deltas.min() >= -1.0, f"Delta below -1: {deltas.min()}"
        assert deltas.max() <= 1.0, f"Delta above 1: {deltas.max()}"
    
    def test_gradient_flow(self):
        """Verify gradients flow through the agent."""
        agent = DeepHedgingAgent(input_dim=3, hidden_dim=16)
        
        paths = torch.randn(16, 50, requires_grad=True)
        deltas = agent(paths)
        
        loss = deltas.sum()
        loss.backward()
        
        # Check gradients exist
        for param in agent.parameters():
            assert param.grad is not None, "Gradient is None"
            assert not torch.isnan(param.grad).any(), "NaN in gradients"


class TestCVaRLoss:
    """Test CVaR loss computation."""
    
    def test_cvar_lower_than_mean(self):
        """CVaR should focus on tail, giving lower value than mean."""
        loss_fn = CVaRLoss(alpha=0.05)
        
        # PnL with some bad outcomes
        pnl = torch.randn(1000)
        
        cvar = -loss_fn(pnl).item()  # Negate because loss returns negative
        mean = pnl.mean().item()
        
        assert cvar < mean, f"CVaR ({cvar}) should be < mean ({mean})"
    
    def test_cvar_captures_worst_cases(self):
        """CVaR of uniform distribution should be as expected."""
        loss_fn = CVaRLoss(alpha=0.1)
        
        # Uniform distribution from -1 to 1
        pnl = torch.linspace(-1, 1, 1000)
        
        cvar = -loss_fn(pnl).item()
        
        # Worst 10% should be around -0.9 average
        assert cvar < -0.8, f"CVaR should capture worst cases, got {cvar}"


class TestHedgingEnvironment:
    """Test the hedging environment."""
    
    def test_pnl_shape(self):
        """PnL should have correct shape."""
        env = HedgingEnvironment(strike_pct=1.0)
        
        batch_size = 32
        n_steps = 50
        paths = torch.ones(batch_size, n_steps + 1)  # Constant paths
        deltas = torch.zeros(batch_size, n_steps)
        
        pnl = env.compute_pnl(paths, deltas)
        
        assert pnl.shape == (batch_size,), f"Expected ({batch_size},), got {pnl.shape}"
    
    def test_zero_hedge_constant_price(self):
        """Zero hedge on constant price: PnL = premium - payoff."""
        env = HedgingEnvironment(strike_pct=1.0, initial_price=1.0)
        
        paths = torch.ones(10, 51)  # Price stays at 1.0
        deltas = torch.zeros(10, 50)
        
        pnl = env.compute_pnl(paths, deltas)
        
        # ATM call at maturity is worth 0, so PnL = premium
        expected = env.premium
        assert torch.allclose(pnl, torch.full_like(pnl, expected), atol=1e-4)
    
    def test_perfect_hedge(self):
        """Perfect delta hedge should reduce variance."""
        env = HedgingEnvironment(strike_pct=1.0, transaction_cost=0)
        
        # Random paths
        torch.manual_seed(42)
        paths = torch.cumsum(torch.randn(100, 51) * 0.02, dim=1) + 1.0
        
        # Use BS delta as hedge (approximate perfect hedge)
        import numpy as np
        time_remaining = np.linspace(1, 0.02, 50)
        
        deltas_list = []
        for i in range(100):
            path_np = paths[i, :-1].numpy()
            d = black_scholes_delta(path_np, env.K, time_remaining, sigma=0.2)
            deltas_list.append(torch.tensor(d, dtype=torch.float32))
        
        deltas_hedged = torch.stack(deltas_list)
        deltas_unhedged = torch.zeros(100, 50)
        
        pnl_hedged = env.compute_pnl(paths, deltas_hedged)
        pnl_unhedged = env.compute_pnl(paths, deltas_unhedged)
        
        # Hedged should have lower variance
        assert pnl_hedged.std() < pnl_unhedged.std(), (
            f"Hedged std ({pnl_hedged.std():.4f}) should be < "
            f"unhedged std ({pnl_unhedged.std():.4f})"
        )


class TestComputePnL:
    """Test the simplified PnL function."""
    
    def test_basic_computation(self):
        """Test basic PnL calculation."""
        paths = torch.tensor([[1.0, 1.1, 1.2], [1.0, 0.9, 0.8]])
        deltas = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        
        pnl = compute_pnl(paths, deltas, strike=1.0, premium=0.1, transaction_cost=0)
        
        assert pnl.shape == (2,)


class TestCVaRBatchConsistency:
    """Test CVaR computation consistency across batches."""
    
    def test_cvar_batch_consistency(self):
        """CVaR should give consistent results when computed on same data."""
        from src.metrics.statistics import compute_cvar as compute_cvar_np
        
        # Generate fixed PnL data
        np.random.seed(42)
        pnl = np.random.randn(1000) * 0.05 + 0.02
        
        # Compute CVaR multiple times
        cvar1, _ = compute_cvar_np(pnl, alpha=0.05)
        cvar2, _ = compute_cvar_np(pnl, alpha=0.05)
        cvar3, _ = compute_cvar_np(pnl, alpha=0.05)
        
        # Should be exactly the same (deterministic)
        assert cvar1 == cvar2 == cvar3, "CVaR should be deterministic"
    
    def test_cvar_subsample_consistency(self):
        """CVaR on large sample should be close to subsamples."""
        from src.metrics.statistics import compute_cvar as compute_cvar_np
        
        np.random.seed(42)
        pnl_full = np.random.randn(10000) * 0.05 + 0.02
        
        cvar_full, _ = compute_cvar_np(pnl_full, alpha=0.05)
        
        # Subsample CVaRs should be close
        cvars = []
        for i in range(5):
            subset = pnl_full[i*2000:(i+1)*2000]
            cvar, _ = compute_cvar_np(subset, alpha=0.05)
            cvars.append(cvar)
        
        # Subsamples should be within 30% of full
        for cvar in cvars:
            rel_diff = abs(cvar - cvar_full) / abs(cvar_full)
            assert rel_diff < 0.3, f"Subsample CVaR differs too much: {rel_diff:.2%}"


class TestBaselineAgents:
    """Test baseline hedging agents."""
    
    def test_bs_delta_hedge_output_shape(self):
        """BS delta hedge should return correct shape."""
        from src.agents.baselines import BlackScholesDeltaHedge
        
        agent = BlackScholesDeltaHedge(strike=1.0, sigma=0.2, T=1.0)
        
        paths = np.random.randn(50, 101) * 0.02 + 1.0
        time_grid = np.linspace(0, 1, 101)
        
        deltas = agent.compute_deltas(paths, time_grid)
        
        assert deltas.shape == (50, 100), f"Expected (50, 100), got {deltas.shape}"
    
    def test_bs_delta_bounded(self):
        """BS delta should be in [0, 1] for call option."""
        from src.agents.baselines import BlackScholesDeltaHedge
        
        agent = BlackScholesDeltaHedge(strike=1.0, sigma=0.2, T=1.0)
        
        paths = np.abs(np.random.randn(50, 101)) * 0.5 + 0.5  # Positive prices
        time_grid = np.linspace(0, 1, 101)
        
        deltas = agent.compute_deltas(paths, time_grid)
        
        assert deltas.min() >= 0.0, f"Delta below 0: {deltas.min()}"
        assert deltas.max() <= 1.0, f"Delta above 1: {deltas.max()}"
    
    def test_naive_hedge_fixed_delta(self):
        """Naive hedge with fixed delta should return constant."""
        from src.agents.baselines import NaiveHedgeAgent
        
        agent = NaiveHedgeAgent(strategy="fixed", fixed_delta=0.6)
        
        paths = np.random.randn(30, 51) * 0.02 + 1.0
        time_grid = np.linspace(0, 1, 51)
        
        deltas = agent.compute_deltas(paths, time_grid)
        
        assert np.allclose(deltas, 0.6), "Fixed delta should be constant"
    
    def test_heston_delta_output(self):
        """Heston delta should produce valid output."""
        from src.agents.baselines import HestonDeltaHedge
        
        agent = HestonDeltaHedge(strike=1.0, T=1.0)
        
        paths = np.abs(np.random.randn(20, 51)) * 0.3 + 0.8
        time_grid = np.linspace(0, 1, 51)
        
        deltas = agent.compute_deltas(paths, time_grid)
        
        assert deltas.shape == (20, 50), f"Expected (20, 50), got {deltas.shape}"
        assert deltas.min() >= 0.0, f"Heston delta below 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
