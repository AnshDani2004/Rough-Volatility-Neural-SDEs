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
- PyTorch ≥ 2.0
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
├── tests/                     # Unit tests (43 tests)
├── docs/                      # Documentation
└── train.py                   # Training script
```

## Results

### Roughness Visualization
fBM paths with $H=0.1$ exhibit the characteristic roughness observed in empirical volatility:

![fBM Comparison](figures/convergence_regimes.png)

### Deep Hedging Performance
Neural hedge trained on rough paths outperforms Black-Scholes delta:

| Strategy | Mean P&L | 5% CVaR |
|----------|----------|---------|
| BS Delta | 0.054 | 0.044 |
| Neural | **0.071** | **0.060** |

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
