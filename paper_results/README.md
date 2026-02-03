# Paper Results

This directory maps CSV outputs to figures and tables in the paper.

## Figure Mappings

| Figure | Source File | Script |
|--------|-------------|--------|
| Fig. 1: Convergence | `outputs/convergence_results.csv` | `experiments/run_convergence.py` |
| Fig. 2: Hedging Comparison | `outputs/hedging_results.csv` | `experiments/run_hedging.py` |
| Fig. 3: Bootstrap CIs | `outputs/bootstrap_ci.csv` | `experiments/run_bootstrap_ci.py` |

## Table Mappings

| Table | Source File | Description |
|-------|-------------|-------------|
| Table 1 | `outputs/hedging_results.csv` | Strategy comparison metrics |
| Table 2 | `outputs/bootstrap_ci.csv` | 95% confidence intervals |
| Table 3 | `outputs/convergence_results.csv` | Convergence slopes by H |

## Regenerating Figures

```bash
python scripts/plot_all_figures.py \
    --convergence outputs/convergence_results.csv \
    --hedging outputs/hedging_results.csv \
    --bootstrap outputs/bootstrap_ci.csv \
    --out figures/
```

## Seed Information

All results generated with `SEED=42` for reproducibility.
See `REPRODUCE.md` for complete reproduction instructions.
