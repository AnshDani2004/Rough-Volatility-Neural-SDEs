"""Simulators package - fBM wrappers."""

# Re-export from noise module for backwards compatibility
from src.noise.fbm import DaviesHarte, generate_fbm_paths

__all__ = ["DaviesHarte", "generate_fbm_paths"]
