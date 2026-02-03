"""
Market Data Loading and Preprocessing.

This module provides utilities for loading financial data from yfinance
and computing realized volatility for training the rough volatility model.

See docs/train.md for details.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, List
from datetime import datetime, timedelta

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def load_spx_data(
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    ticker: str = "^GSPC"
) -> pd.DataFrame:
    """
    Load S&P 500 (SPX) historical data from yfinance.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date. If None, uses today.
    ticker : str
        Yahoo Finance ticker symbol.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data.
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log-returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., adjusted close).
    
    Returns
    -------
    pd.Series
        Log-returns: log(P_t / P_{t-1}).
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_realized_volatility(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute realized volatility as rolling sum of squared returns.
    
    Parameters
    ----------
    returns : pd.Series
        Log-returns series.
    window : int
        Rolling window size.
    
    Returns
    -------
    pd.Series
        Realized volatility series.
    """
    return np.sqrt((returns ** 2).rolling(window=window).sum())


def create_path_windows(
    returns: np.ndarray,
    window_size: int = 50,
    stride: int = 1
) -> np.ndarray:
    """
    Create overlapping windows of return paths for training.
    
    Parameters
    ----------
    returns : np.ndarray
        1D array of log-returns.
    window_size : int
        Length of each path window.
    stride : int
        Step size between consecutive windows.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_windows, window_size) containing path windows.
    """
    n = len(returns)
    n_windows = (n - window_size) // stride + 1
    
    windows = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        start = i * stride
        windows[i] = returns[start:start + window_size]
    
    return windows


def create_cumulative_paths(windows: np.ndarray) -> np.ndarray:
    """
    Convert return windows to cumulative log-price paths.
    
    Parameters
    ----------
    windows : np.ndarray
        Return windows of shape (n_windows, window_size).
    
    Returns
    -------
    np.ndarray
        Cumulative paths of shape (n_windows, window_size + 1).
    """
    n_windows, window_size = windows.shape
    paths = np.zeros((n_windows, window_size + 1))
    paths[:, 1:] = np.cumsum(windows, axis=1)
    return paths


class MarketDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for market return paths.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker.
    start_date : str
        Data start date.
    end_date : str, optional
        Data end date.
    window_size : int
        Path window size.
    stride : int
        Window stride.
    normalize : bool
        Whether to normalize paths by volatility.
    
    Example
    -------
    >>> dataset = MarketDataset("^GSPC", start_date="2020-01-01", window_size=50)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     print(batch.shape)  # (32, 51)
    """
    
    def __init__(
        self,
        ticker: str = "^GSPC",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        window_size: int = 50,
        stride: int = 1,
        normalize: bool = True
    ):
        self.window_size = window_size
        self.normalize = normalize
        
        # Load data
        data = load_spx_data(start_date, end_date, ticker)
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[('Close', ticker)].values
        else:
            prices = data['Close'].values
        
        # Compute returns
        returns = np.diff(np.log(prices))
        
        # Normalize by volatility if requested
        if normalize:
            vol = np.std(returns)
            returns = returns / vol
            self.volatility_scale = vol
        else:
            self.volatility_scale = 1.0
        
        # Create windows
        self.return_windows = create_path_windows(returns, window_size, stride)
        
        # Create cumulative paths (what we train on)
        self.paths = create_cumulative_paths(self.return_windows)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.paths[idx], dtype=torch.float32)


def get_synthetic_data(
    n_samples: int = 1000,
    n_steps: int = 50,
    volatility: float = 0.02
) -> torch.Tensor:
    """
    Generate synthetic data for testing when market data unavailable.
    
    Parameters
    ----------
    n_samples : int
        Number of paths.
    n_steps : int
        Path length.
    volatility : float
        Return volatility.
    
    Returns
    -------
    torch.Tensor
        Synthetic paths of shape (n_samples, n_steps + 1).
    """
    returns = np.random.randn(n_samples, n_steps) * volatility
    paths = np.zeros((n_samples, n_steps + 1))
    paths[:, 1:] = np.cumsum(returns, axis=1)
    return torch.tensor(paths, dtype=torch.float32)
