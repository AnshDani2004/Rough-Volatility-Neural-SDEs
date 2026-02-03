# Market Data Module

This document explains the market data utilities in `market_data.py`.

---

## Functions

| Function | Purpose |
|----------|---------|
| `load_spx_data()` | Fetch SPX data from yfinance |
| `compute_log_returns()` | $r_t = \log(P_t / P_{t-1})$ |
| `compute_realized_volatility()` | Rolling $\sqrt{\sum r^2}$ |
| `create_path_windows()` | Sliding windows for training |

---

## MarketDataset

PyTorch Dataset that:
1. Downloads SPX data from yfinance
2. Computes log-returns
3. Normalizes by volatility
4. Creates overlapping path windows

```python
dataset = MarketDataset("^GSPC", start_date="2020-01-01", window_size=50)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
