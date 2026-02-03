# Rough Volatility Neural SDEs

A PyTorch framework for generative modeling of rough volatility dynamics using Neural Stochastic Differential Equations with fractional Brownian motion.

## Overview

This repository implements a Neural SDE generator that captures the "rough" behavior of financial volatility, where the Hurst parameter $H \approx 0.1$ produces paths that match empirical observations of market dynamics.

**Key Features:**
- Davies-Harte algorithm for exact fBM simulation
- Neural SDE with Euler-Heun integration scheme
- Signature-based MMD loss for distributional training
- Deep Hedging agent with CVaR objective
- Convergence analysis demonstrating the breakdown of standard Euler-Maruyama rates

## Installation

```bash
git clone https://github.com/AnshDani2004/Rough-Volatility-Neural-SDEs.git
cd Rough-Volatility-Neural-SDEs
pip install -r requirements.txt
```

### Dependencies
- Python ≥ 3.9
- PyTorch == 2.2.0 (see requirements.txt)
- NumPy, SciPy, Matplotlib
- yfinance (for market data)

## Usage

### Training the Generator

```bash
python train.py --epochs 100 --lr 0.001 --H 0.1
```

To learn the Hurst parameter:
```bash
python train.py --epochs 100 --learnable-H
```

### Deep Hedging

```python
from src.hedging.engine import DeepHedgingAgent, HedgingEnvironment, DeepHedgingTrainer

agent = DeepHedgingAgent(hidden_dim=32)
env = HedgingEnvironment(strike_pct=1.0, transaction_cost=0.001)
trainer = DeepHedgingTrainer(agent, env, alpha=0.05)
trainer.train(synthetic_paths, real_paths, n_epochs=100)
```

### Convergence Analysis

```bash
python src/validation/convergence_test.py --samples 100 --output figures/convergence_regimes.png
```

## Project Structure

```
├── src/
│   ├── noise/fbm.py           # Davies-Harte fBM generator
│   ├── models/generator.py    # RoughNeuralSDE with Euler-Heun
│   ├── loss/signature.py      # Signature MMD loss
│   ├── hedging/engine.py      # Deep Hedging agent
│   ├── data/market_data.py    # SPX data loader
│   └── validation/            # Convergence analysis
├── notebooks/
│   └── results.ipynb          # Visualizations and analysis
├── tests/                     # Unit tests (run `pytest tests/`)
├── docs/                      # Documentation
└── train.py                   # Training script
```

## Results

### Roughness Visualization
fBM paths with $H=0.1$ exhibit the characteristic roughness observed in empirical volatility:

![fBM Comparison](figures/convergence_regimes.png)

Observed convergence behavior under exact rough noise injection; the rough case exhibits non-classical convergence rates relative to standard Euler–Maruyama theory.

### Deep Hedging Performance

*3,000 Monte Carlo paths, 2,000 bootstrap resamples, SEED=42*

**Sign convention:** Positive PnL = profit, negative = loss. CVaR (5%) = expected P&L in worst 5% of scenarios.

| Strategy | Mean PnL | 95% CI | CVaR (5%) | 95% CI |
|----------|----------|--------|-----------|--------|
| BS Delta | −0.768 | [−0.779, −0.757] | −1.407 | [−1.433, −1.379] |
| Heston Delta | −0.773 | [−0.783, −0.761] | −1.417 | [−1.443, −1.389] |
| Naive (δ=0.5) | −0.005 | [−0.007, −0.002] | −0.190 | [−0.203, −0.177] |
| **Neural Hedge** | **−0.015** | [−0.020, −0.010] | **−0.430** | [−0.455, −0.404] |

*The Neural hedge substantially reduces tail risk (CVaR) relative to classical delta hedging while maintaining near-zero mean PnL, indicating a deliberate trade-off favoring downside protection under rough volatility dynamics.*

**Statistical significance (Neural vs BS):**
- Mean PnL difference: +0.754, p < 0.001 ***
- CVaR difference: +0.976, p < 0.001 ***

## Testing

```bash
pytest tests/ -v
```

## References

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949.
- Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887-904.
- Buehler, H., et al. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271-1291.

## License

MIT License
