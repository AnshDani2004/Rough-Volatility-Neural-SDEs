"""
Tests for the Signature Loss functions.

These tests verify that the signature-based loss functions:
1. Correctly compute log-signatures
2. Return low loss for similar distributions
3. Return high loss for different distributions

See docs/test_signature.md for details.
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loss.signature import (
    LogSignature, SigMMDLoss, SignatureL2Loss, add_time_channel
)


class TestLogSignature:
    """Test suite for the LogSignature class."""
    
    def test_output_shape(self):
        """Verify log-signature has expected shape."""
        logsig = LogSignature(depth=4)
        
        batch_size = 16
        n_steps = 50
        dim = 2
        
        paths = torch.randn(batch_size, n_steps, dim)
        sigs = logsig(paths)
        
        assert sigs.shape[0] == batch_size, (
            f"Expected batch size {batch_size}, got {sigs.shape[0]}"
        )
        
        # Signature dimension depends on input dim and depth
        expected_dim = logsig.signature_dim(dim)
        assert sigs.shape[1] == expected_dim, (
            f"Expected signature dim {expected_dim}, got {sigs.shape[1]}"
        )
    
    def test_different_paths_different_sigs(self):
        """Verify different paths produce different signatures."""
        logsig = LogSignature(depth=4)
        
        # Create two clearly different paths
        t = torch.linspace(0, 1, 50)
        path1 = torch.stack([t, torch.sin(2 * np.pi * t)], dim=-1).unsqueeze(0)
        path2 = torch.stack([t, torch.cos(2 * np.pi * t)], dim=-1).unsqueeze(0)
        
        sig1 = logsig(path1)
        sig2 = logsig(path2)
        
        # Signatures should be different
        diff = torch.sum((sig1 - sig2) ** 2).item()
        assert diff > 0.01, "Different paths should have different signatures"
    
    def test_signature_dimension_formula(self):
        """Verify signature dimension computation."""
        logsig = LogSignature(depth=4)
        
        # For dim=2, depth=4, the log-signature dimension should match iisignature
        dim = 2
        sig_dim = logsig.signature_dim(dim)
        
        # Create a dummy path and verify actual output matches
        paths = torch.randn(1, 50, dim)
        sigs = logsig(paths)
        
        assert sigs.shape[1] == sig_dim


class TestAddTimeChannel:
    """Test suite for the add_time_channel function."""
    
    def test_2d_input(self):
        """Test adding time to 2D input."""
        paths = torch.randn(16, 100)  # (batch, time)
        result = add_time_channel(paths)
        
        assert result.shape == (16, 100, 2), (
            f"Expected shape (16, 100, 2), got {result.shape}"
        )
        
        # Time channel should go from 0 to 1
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(result[0, -1, 0], torch.tensor(1.0), atol=1e-5)
    
    def test_3d_input(self):
        """Test adding time to 3D input."""
        paths = torch.randn(16, 100, 3)  # (batch, time, dim)
        result = add_time_channel(paths)
        
        assert result.shape == (16, 100, 4), (
            f"Expected shape (16, 100, 4), got {result.shape}"
        )


class TestSigMMDLoss:
    """Test suite for the SigMMDLoss class."""
    
    def test_zero_for_identical(self):
        """Loss should be approximately zero for identical path sets."""
        loss_fn = SigMMDLoss(depth=4)
        
        paths = torch.randn(32, 50)
        loss = loss_fn(paths, paths)
        
        assert loss.item() < 1e-6, (
            f"Loss for identical paths should be ~0, got {loss.item()}"
        )
    
    def test_positive_for_different(self):
        """Loss should be positive for different distributions."""
        loss_fn = SigMMDLoss(depth=4)
        
        # Two clearly different distributions
        paths1 = torch.randn(32, 50) * 0.1
        paths2 = torch.randn(32, 50) * 2.0 + 5.0
        
        loss = loss_fn(paths1, paths2)
        
        assert loss.item() > 0.1, (
            f"Loss for different paths should be > 0, got {loss.item()}"
        )
    
    def test_discriminative_power(self):
        """Loss should be higher for more different distributions."""
        loss_fn = SigMMDLoss(depth=4)
        
        base = torch.randn(32, 50)
        similar = base + torch.randn(32, 50) * 0.1
        different = torch.randn(32, 50) * 3.0
        
        loss_similar = loss_fn(base, similar)
        loss_different = loss_fn(base, different)
        
        assert loss_different > loss_similar, (
            f"Different distributions should have higher loss: "
            f"similar={loss_similar.item():.4f}, different={loss_different.item():.4f}"
        )
    
    def test_output_is_scalar(self):
        """Verify loss is a scalar tensor."""
        loss_fn = SigMMDLoss(depth=4)
        
        paths1 = torch.randn(16, 50)
        paths2 = torch.randn(16, 50)
        
        loss = loss_fn(paths1, paths2)
        
        assert loss.dim() == 0, "Loss should be a scalar"


class TestSigMMDLossFullMMD:
    """Test the full MMD computation."""
    
    def test_full_mmd_linear_kernel(self):
        """Test full MMD with linear kernel."""
        loss_fn = SigMMDLoss(depth=4)
        
        paths1 = torch.randn(16, 50)
        paths2 = torch.randn(16, 50) * 2.0
        
        loss = loss_fn.compute_full_mmd(paths1, paths2, kernel='linear')
        
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_full_mmd_rbf_kernel(self):
        """Test full MMD with RBF kernel."""
        loss_fn = SigMMDLoss(depth=4)
        
        paths1 = torch.randn(16, 50)
        paths2 = torch.randn(16, 50) * 2.0
        
        loss = loss_fn.compute_full_mmd(paths1, paths2, kernel='rbf', bandwidth=1.0)
        
        assert torch.isfinite(loss), "Loss should be finite"


class TestSignatureL2Loss:
    """Test the simplified L2 signature loss."""
    
    def test_zero_for_identical(self):
        """Loss should be zero for identical paths."""
        loss_fn = SignatureL2Loss(depth=4)
        
        paths = torch.randn(32, 50)
        loss = loss_fn(paths, paths)
        
        assert loss.item() < 1e-6
    
    def test_positive_for_different(self):
        """Loss should be positive for different paths."""
        loss_fn = SignatureL2Loss(depth=4)
        
        paths1 = torch.randn(32, 50)
        paths2 = torch.randn(32, 50) * 2.0
        
        loss = loss_fn(paths1, paths2)
        
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
