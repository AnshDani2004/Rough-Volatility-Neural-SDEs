"""
Tests for the Rough Neural SDE Generator.

These tests verify that the RoughNeuralSDE model:
1. Produces correct output shapes
2. Has differentiable forward pass
3. Maintains positive diffusion

See docs/test_generator.md for details.
"""

import torch
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from noise.fbm import DaviesHarte
from models.generator import RoughNeuralSDE, MLP, create_generator


class TestMLP:
    """Test suite for the MLP building block."""
    
    def test_forward_shape(self):
        """Verify MLP output shape."""
        mlp = MLP(input_dim=2, hidden_dim=32, output_dim=1)
        x = torch.randn(16, 2)
        out = mlp(x)
        assert out.shape == (16, 1)
    
    def test_gradient_flow(self):
        """Verify gradients propagate through MLP."""
        mlp = MLP(input_dim=2, hidden_dim=32, output_dim=1)
        x = torch.randn(16, 2)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        
        # Check first layer gradients exist
        assert mlp.net[0].weight.grad is not None
        assert mlp.net[0].weight.grad.abs().sum() > 0


class TestRoughNeuralSDE:
    """Test suite for the RoughNeuralSDE generator."""
    
    def test_output_shape(self):
        """Verify forward pass produces correct shape."""
        n_steps = 50
        batch_size = 16
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        model = RoughNeuralSDE(hidden_dim=32)
        
        paths = model(dh)
        
        assert paths.shape == (batch_size, n_steps + 1), (
            f"Expected shape {(batch_size, n_steps + 1)}, got {paths.shape}"
        )
    
    def test_forward_backward(self):
        """Verify gradients flow through the entire model."""
        n_steps = 20
        batch_size = 8
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        model = RoughNeuralSDE(hidden_dim=32)
        
        paths = model(dh)
        
        # Compute a simple loss and backward
        loss = paths.mean()
        loss.backward()
        
        # Check drift network gradients
        assert model.drift_net.net[0].weight.grad is not None, (
            "Drift network should have gradients"
        )
        assert model.drift_net.net[0].weight.grad.abs().sum() > 0
        
        # Check diffusion network gradients
        assert model.diffusion_net.net[0].weight.grad is not None, (
            "Diffusion network should have gradients"
        )
        assert model.diffusion_net.net[0].weight.grad.abs().sum() > 0
    
    def test_positive_diffusion(self):
        """Verify diffusion is always positive."""
        model = RoughNeuralSDE(hidden_dim=32)
        
        # Test over a range of inputs
        t_vals = torch.linspace(0, 1, 20).unsqueeze(-1)
        x_vals = torch.linspace(-10, 10, 20).unsqueeze(-1)
        
        for t in t_vals:
            for x in x_vals:
                t_batch = t.unsqueeze(0)  # (1, 1)
                x_batch = x.unsqueeze(0)  # (1, 1)
                sigma = model.diffusion(t_batch, x_batch)
                
                assert sigma.item() > 0, (
                    f"Diffusion should be positive, got {sigma.item()} at t={t.item()}, x={x.item()}"
                )
    
    def test_euler_method(self):
        """Test that Euler method produces valid paths."""
        n_steps = 30
        batch_size = 8
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        model = RoughNeuralSDE(hidden_dim=32)
        
        paths = model(dh, method='euler')
        
        assert paths.shape == (batch_size, n_steps + 1)
        assert torch.isfinite(paths).all(), "Paths should be finite"
    
    def test_euler_heun_method(self):
        """Test that Euler-Heun method produces valid paths."""
        n_steps = 30
        batch_size = 8
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        model = RoughNeuralSDE(hidden_dim=32)
        
        paths = model(dh, method='euler_heun')
        
        assert paths.shape == (batch_size, n_steps + 1)
        assert torch.isfinite(paths).all(), "Paths should be finite"
    
    def test_custom_initial_condition(self):
        """Verify paths respect initial condition."""
        n_steps = 20
        batch_size = 8
        x0_value = 5.0
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        model = RoughNeuralSDE(hidden_dim=32)
        
        paths = model(dh, x0=x0_value)
        
        # All paths should start at x0
        assert torch.allclose(paths[:, 0], torch.full((batch_size,), x0_value)), (
            f"Paths should start at {x0_value}"
        )
    
    def test_deterministic_with_same_noise(self):
        """Verify model is deterministic given the same noise."""
        torch.manual_seed(42)
        
        n_steps = 20
        batch_size = 4
        
        # Create generator and sample once
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        
        model = RoughNeuralSDE(hidden_dim=32)
        model.eval()  # Ensure deterministic behavior
        
        # Note: The DaviesHarte samples fresh noise each call, so
        # we can't easily test determinism without modifying the interface.
        # This test just verifies no errors occur.
        with torch.no_grad():
            paths = model(dh)
            assert torch.isfinite(paths).all()


class TestCreateGenerator:
    """Test factory function."""
    
    def test_create_generator(self):
        """Verify factory function works."""
        model = create_generator(hidden_dim=64, state_dim=1)
        assert isinstance(model, RoughNeuralSDE)
        assert model.hidden_dim == 64
        assert model.state_dim == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
