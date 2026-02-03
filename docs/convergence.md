# Numerical Convergence Analysis

This document explains the convergence analysis in `convergence_test.py`.

---

## The Problem with Rough Volatility

Standard Euler-Maruyama solver has convergence order $O(\Delta t^{0.5})$ for Itô SDEs.

**Key Insight:** This assumes $H = 0.5$ (standard Brownian motion). When $H < 0.5$, this guarantee breaks down.

---

## Strong Convergence

The strong error measures path-wise accuracy:

$$
\epsilon(\Delta t) = \sqrt{\mathbb{E}[(X_T^{\text{true}} - X_T^{\text{approx}})^2]}
$$

### Expected Convergence Orders:

| Regime | Hurst H | Expected Slope |
|--------|---------|----------------|
| Standard BM | 0.5 | ~0.5 |
| Rough Vol | 0.1 | ~0.1 |

The slope on a log-log plot of Error vs $\Delta t$ gives the convergence order.

---

## Method

1. Generate ground truth with fine grid ($N=2^{14}$ steps)
2. Sub-sample to coarser grids ($N=2^4$ to $2^{10}$)
3. Compute RMSE at terminal time
4. Linear regression on log-log scale

---

## Implication

Slow convergence for $H=0.1$ means:
- Need more timesteps for accuracy
- Neural SDE approach becomes valuable (learns dynamics without fine discretization)
