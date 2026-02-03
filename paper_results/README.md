# Paper Results

This directory maps experiment outputs to figures and tables in the paper.

---

## Sign Convention

**Positive PnL = Profit, Negative PnL = Loss**

CVaR (5%) represents the expected P&L in the worst 5% of scenarios. More negative CVaR indicates higher tail risk.

---

## Table 1: Hedging Strategy Comparison

*3,000 Monte Carlo paths, H = 0.1 (rough volatility), SEED = 42*

| Strategy | Mean PnL | 95% CI | CVaR (5%) | 95% CI |
|----------|----------|--------|-----------|--------|
| BlackScholes Δ | −0.768 | [−0.779, −0.757] | −1.407 | [−1.433, −1.379] |
| Heston Δ | −0.773 | [−0.783, −0.761] | −1.417 | [−1.443, −1.389] |
| Naive (δ=0.5) | −0.005 | [−0.007, −0.002] | −0.190 | [−0.203, −0.177] |
| Neural Hedge | −0.015 | [−0.020, −0.010] | −0.430 | [−0.455, −0.404] |

*CIs computed via 2,000 bootstrap resamples.*

---

## Table 2: Hypothesis Tests (Neural vs Black-Scholes)

*One-sided test: H₁ = Neural outperforms BS*

| Metric | Difference | p-value | 95% CI |
|--------|------------|---------|--------|
| Mean PnL | +0.754 | < 0.001 *** | [+0.741, +0.766] |
| CVaR (5%) | +0.976 | < 0.001 *** | [+0.938, +1.013] |

*Significance: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001*

---

## Figure Mapping

| Figure | Source File | Command |
|--------|-------------|---------|
| Fig. 1: Convergence | `outputs/convergence_results.csv` | `python experiments/run_convergence.py` |
| Fig. 2: Hedging Comparison | `outputs/hedging_results.csv` | `python experiments/run_hedging.py` |

---

## Regenerating Results

```bash
# Full reproduction
python experiments/run_hedging.py \
    --paths 3000 \
    --n-bootstrap 2000 \
    --heston-params outputs/heston_params.json \
    --out outputs/hedging_results.csv

# Generate figures
python scripts/plot_all_figures.py \
    --hedging outputs/hedging_results.csv \
    --out figures/
```

See `REPRODUCE.md` for complete instructions.
