# Tests for Deep Hedging

This document explains the tests in `test_hedging.py`.

---

## Test Overview

| Test Class | Purpose |
|------------|---------|
| `TestBlackScholesDelta` | Verify BS delta formula |
| `TestDeepHedgingAgent` | Check RNN output shape, bounds, gradients |
| `TestCVaRLoss` | Verify CVaR captures tail risk |
| `TestHedgingEnvironment` | Test PnL computation |

---

## Key Test: Variance Reduction

```python
# Perfect hedge should reduce PnL variance
pnl_hedged = env.compute_pnl(paths, delta_hedged)
pnl_unhedged = env.compute_pnl(paths, delta_zero)

assert pnl_hedged.std() < pnl_unhedged.std()
```
