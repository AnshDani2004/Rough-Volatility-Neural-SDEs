# Deep Hedging

This document explains the Deep Hedging module in `engine.py`.

---

## Problem Setup

**Scenario:** Sell an At-The-Money (ATM) Call option on SPX.

**Objective:** Minimize the risk of the hedging P&L:

$$
\text{PnL} = C_0 + \sum_{t=0}^{T-1} \delta_t (S_{t+1} - S_t) - (S_T - K)^+ - \text{TC}
$$

Where:
- $C_0$ = Option premium received
- $\delta_t$ = Hedge ratio at time $t$
- $(S_T - K)^+$ = Call option payoff
- TC = Transaction costs

---

## Risk Measure: CVaR

We minimize **Conditional Value at Risk (CVaR)**, also known as Expected Shortfall:

$$
\text{CVaR}_\alpha(X) = \mathbb{E}[X \mid X \leq \text{VaR}_\alpha(X)]
$$

This focuses on the expected loss in the worst $\alpha$% of scenarios.

---

## Agent Architecture

```
┌─────────────────┐
│   RNN (GRU)     │  ← Observes (S_t, t, delta_{t-1})
├─────────────────┤
│   Dense Layer   │
├─────────────────┤
│     Tanh        │  ← Bounded output [-1, 1]
└─────────────────┘
         ↓
       δ_t (hedge ratio)
```

---

## TSTR Protocol

**Train on Synthetic, Test on Real:**

1. Train the RoughNeuralSDE generator on real SPX data
2. Generate synthetic paths from the trained generator
3. Train the hedging agent on synthetic paths
4. **Test** the hedging agent on held-out real SPX data

This validates whether the generator captures enough structure for practical hedging.
