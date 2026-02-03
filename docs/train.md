# Training Loop: Data and Optimization

This document explains the training pipeline for the Rough Volatility Neural SDE.

---

## 1. Data Pipeline

### Loading SPX Data

We use `yfinance` to fetch S&P 500 historical data:

```python
import yfinance as yf
spx = yf.download("^GSPC", start="2020-01-01")
```

### Computing Log Returns

For financial modeling, we work with log-returns:

$$r_t = \log(P_t / P_{t-1})$$

### Realized Volatility

Realized volatility over window $w$ is:

$$\text{RV}_t = \sqrt{\sum_{i=0}^{w-1} r_{t-i}^2}$$

---

## 2. Training Objective

Minimize the Signature MMD loss between:
- **Real paths:** Windows of actual market log-returns
- **Generated paths:** Output from `RoughNeuralSDE`

$$\min_{\theta, \phi} \mathcal{L} = \text{SigMMD}(X^{\text{real}}, X^{\text{gen}})$$

---

## 3. Learnable Hurst Parameter

The Hurst exponent can be:

| Mode | Description |
|------|-------------|
| **Fixed** | $H = 0.1$ (forces roughness) |
| **Learnable** | $H = \sigma(h) \cdot 0.5$ where $h$ is learned |

Using sigmoid ensures $H \in (0, 0.5)$ for anti-persistent (rough) behavior.

---

## 4. Training Configuration

```python
config = {
    'n_epochs': 100,
    'batch_size': 64,
    'lr': 1e-3,
    'hidden_dim': 64,
    'H_fixed': 0.1,        # or None for learnable
    'window_size': 50,     # Path length
    'sig_depth': 4,
}
```

---

## 5. Running Training

```bash
python train.py --epochs 100 --lr 0.001 --H 0.1
```
