"""
Statistical tests for the Davies-Harte fBM generator.

These tests verify that the generated fBM paths have the correct statistical
properties, particularly the variance scaling law: Var(ΔB^H) = Δt^{2H}.

See docs/test_fbm.md for detailed explanations.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from noise.fbm import DaviesHarte, generate_fbm_paths


class TestDaviesHarte:
    """Test suite for the DaviesHarte fBM generator."""
    
    def test_variance_scaling(self):
        """
        Verify that increment variance scales as Δt^{2H}.
        
        This is the key statistical property of fBM. We generate 10,000 paths
        and verify the empirical variance matches the theoretical value within 1%.
        """
        # Parameters
        n_steps = 256
        batch_size = 10_000
        H = 0.1
        T = 1.0
        
        # Generate increments
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=H, T=T)
        increments = dh.sample()
        
        # Theoretical variance: Δt^{2H}
        dt = T / n_steps
        theoretical_variance = dt ** (2 * H)
        
        # Empirical variance (across all paths and time steps)
        empirical_variance = np.var(increments)
        
        # Relative error
        relative_error = abs(empirical_variance - theoretical_variance) / theoretical_variance
        
        # Assert error < 1%
        assert relative_error < 0.01, (
            f"Variance scaling failed: empirical={empirical_variance:.6f}, "
            f"theoretical={theoretical_variance:.6f}, relative_error={relative_error:.4%}"
        )
    
    def test_variance_scaling_multiple_H(self):
        """Test variance scaling for different Hurst parameters."""
        n_steps = 256
        batch_size = 5_000
        T = 1.0
        dt = T / n_steps
        
        for H in [0.1, 0.25, 0.4]:
            dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=H, T=T)
            increments = dh.sample()
            
            theoretical_variance = dt ** (2 * H)
            empirical_variance = np.var(increments)
            relative_error = abs(empirical_variance - theoretical_variance) / theoretical_variance
            
            assert relative_error < 0.02, (
                f"Variance scaling failed for H={H}: "
                f"relative_error={relative_error:.4%}"
            )
    
    def test_increment_shape(self):
        """Verify output has correct shape (batch_size, n_steps)."""
        n_steps = 128
        batch_size = 50
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        increments = dh.sample()
        
        assert increments.shape == (batch_size, n_steps), (
            f"Expected shape {(batch_size, n_steps)}, got {increments.shape}"
        )
    
    def test_eigenvalue_positivity(self):
        """Ensure all eigenvalues are non-negative (Davies-Harte validity)."""
        dh = DaviesHarte(n_steps=256, batch_size=1, H=0.1, T=1.0)
        
        # All eigenvalues should be non-negative
        assert np.all(dh.eigenvalues >= 0), (
            f"Found negative eigenvalues: min={dh.eigenvalues.min()}"
        )
    
    def test_mean_zero(self):
        """Verify that increments have approximately zero mean."""
        n_steps = 256
        batch_size = 10_000
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        increments = dh.sample()
        
        # Sample mean
        sample_mean = np.mean(increments)
        
        # Standard error of the mean
        se = np.std(increments) / np.sqrt(batch_size * n_steps)
        
        # Mean should be within 3 standard errors of zero (99.7% confidence)
        assert abs(sample_mean) < 3 * se, (
            f"Mean {sample_mean:.6f} exceeds 3 standard errors ({3*se:.6f})"
        )
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid Hurst parameter
        with pytest.raises(ValueError, match="Hurst exponent"):
            DaviesHarte(n_steps=100, batch_size=10, H=1.5, T=1.0)
        
        with pytest.raises(ValueError, match="Hurst exponent"):
            DaviesHarte(n_steps=100, batch_size=10, H=0, T=1.0)
        
        # Invalid n_steps
        with pytest.raises(ValueError, match="n_steps"):
            DaviesHarte(n_steps=0, batch_size=10, H=0.1, T=1.0)
        
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size"):
            DaviesHarte(n_steps=100, batch_size=0, H=0.1, T=1.0)
        
        # Invalid time horizon
        with pytest.raises(ValueError, match="Time horizon"):
            DaviesHarte(n_steps=100, batch_size=10, H=0.1, T=-1.0)


class TestGenerateFbmPaths:
    """Test suite for the convenience function generate_fbm_paths."""
    
    def test_path_shape(self):
        """Verify path output has shape (batch_size, n_steps + 1)."""
        n_steps = 128
        batch_size = 20
        
        paths = generate_fbm_paths(n_steps=n_steps, batch_size=batch_size, H=0.1, T=1.0)
        
        assert paths.shape == (batch_size, n_steps + 1), (
            f"Expected shape {(batch_size, n_steps + 1)}, got {paths.shape}"
        )
    
    def test_paths_start_at_zero(self):
        """Verify all paths start at zero."""
        paths = generate_fbm_paths(n_steps=100, batch_size=50, H=0.1, T=1.0)
        
        assert np.allclose(paths[:, 0], 0), "Paths should start at zero"


class TestDistributionalProperties:
    """Statistical tests for fBM distributional properties."""
    
    def test_terminal_variance_scaling(self):
        """
        Verify Var(B^H_T) ≈ T^{2H} at terminal time.
        
        This is a key mathematical property: the variance of fBM at time T
        equals T^{2H}.
        """
        from scipy import stats
        
        n_steps = 256
        batch_size = 5_000
        T = 1.0
        
        for H in [0.1, 0.25, 0.4]:
            paths = generate_fbm_paths(n_steps=n_steps, batch_size=batch_size, H=H, T=T)
            terminal_values = paths[:, -1]
            
            # Theoretical variance: T^{2H}
            theoretical_var = T ** (2 * H)
            empirical_var = np.var(terminal_values)
            
            relative_error = abs(empirical_var - theoretical_var) / theoretical_var
            
            assert relative_error < 0.05, (
                f"Terminal variance for H={H}: empirical={empirical_var:.4f}, "
                f"theoretical={theoretical_var:.4f}, error={relative_error:.2%}"
            )
    
    def test_increment_covariance(self):
        """
        Test that increment covariance matches fBM theory.
        
        For fBM: Cov(B^H_s, B^H_t) = 0.5 * (s^{2H} + t^{2H} - |t-s|^{2H})
        """
        n_steps = 128
        batch_size = 5_000
        H = 0.25
        T = 1.0
        
        paths = generate_fbm_paths(n_steps=n_steps, batch_size=batch_size, H=H, T=T)
        
        # Test covariance at mid and final time
        t_idx = n_steps  # Terminal
        s_idx = n_steps // 2  # Midpoint
        
        t = T
        s = T / 2
        
        # Theoretical covariance
        theoretical_cov = 0.5 * (s**(2*H) + t**(2*H) - abs(t-s)**(2*H))
        
        # Empirical covariance
        empirical_cov = np.cov(paths[:, s_idx], paths[:, t_idx])[0, 1]
        
        relative_error = abs(empirical_cov - theoretical_cov) / abs(theoretical_cov)
        
        assert relative_error < 0.10, (
            f"Covariance test: empirical={empirical_cov:.4f}, "
            f"theoretical={theoretical_cov:.4f}, error={relative_error:.2%}"
        )
    
    def test_increment_normality(self):
        """
        Test that fBM increments are approximately Gaussian.
        
        Uses Kolmogorov-Smirnov test with p > 0.01 threshold.
        """
        from scipy import stats
        
        n_steps = 256
        batch_size = 1_000
        H = 0.1
        T = 1.0
        
        dh = DaviesHarte(n_steps=n_steps, batch_size=batch_size, H=H, T=T)
        increments = dh.sample()
        
        # Flatten and standardize
        flat_incr = increments.flatten()
        standardized = (flat_incr - np.mean(flat_incr)) / np.std(flat_incr)
        
        # KS test against standard normal
        statistic, p_value = stats.kstest(standardized, 'norm')
        
        # p > 0.01 means we cannot reject normality
        assert p_value > 0.01, (
            f"Normality test failed: KS stat={statistic:.4f}, p={p_value:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
