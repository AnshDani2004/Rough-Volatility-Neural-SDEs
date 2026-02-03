# Reproducibility Guide

This document provides step-by-step instructions to reproduce all experimental results in the paper.

## Environment Setup

### Option 1: pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Option 2: Docker

```bash
docker build -t rough-vol .
docker run -it rough-vol bash
```

### Verify Installation

```bash
pytest tests/ -v
python scripts/quick_start.sh
```

---

## Seed Policy

All experiments use `SEED=42` by default for reproducibility. Override with `--seed`.

For exact CPU reproducibility:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; torch.use_deterministic_algorithms(True)"
```

> **Note:** GPU results may have minor floating-point variance. Use CPU for exact reproduction.

---

## Experiment 1: Convergence Analysis (Section 4.1)

**Runtime:** ~10 minutes on 8-core CPU

```bash
python experiments/run_convergence.py \
    --samples 100 \
    --out outputs/convergence_results.csv
```

**Quick mode (CI):**
```bash
python experiments/run_convergence.py --quick --samples 5 --out outputs/convergence_quick.csv
```

**Expected output:** `outputs/convergence_results.csv` with columns:
`H,dt,error_mean,error_std,slope,slope_lower,slope_upper,R2,n_samples`

---

## Experiment 2: Hedging Comparison (Section 4.2)

**Runtime:** ~15 minutes on 8-core CPU

```bash
python experiments/run_hedging.py \
    --paths 1000 \
    --out outputs/hedging_results.csv
```

**Quick mode (CI):**
```bash
python experiments/run_hedging.py --quick --paths 100 --out outputs/hedging_quick.csv
```

**Expected output:** `outputs/hedging_results.csv` with columns:
`Strategy,MeanPnL,StdPnL,CVaR_0.05,VaR_0.05,Turnover,Paths`

---

## Experiment 3: Bootstrap Confidence Intervals (Section 4.3)

```bash
python experiments/run_bootstrap_ci.py \
    --metric-file outputs/hedging_results.csv \
    --n-bootstrap 1000 \
    --out outputs/bootstrap_ci.csv
```

**Expected output:** `outputs/bootstrap_ci.csv` with columns:
`metric,alpha,lower,mean,upper`

---

## Calibrating Heston Model

```bash
python experiments/calibrate_heston.py \
    --vol-paths outputs/vol_paths.npy \
    --out outputs/heston_params.json
```

---

## Generating Figures

After running experiments:

```bash
python scripts/plot_all_figures.py \
    --convergence outputs/convergence_results.csv \
    --hedging outputs/hedging_results.csv \
    --out figures/
```

---

## Hardware Requirements

| Experiment | RAM | Time (8-core) |
|------------|-----|---------------|
| Convergence | 4GB | 10 min |
| Hedging | 8GB | 15 min |
| Bootstrap | 2GB | 2 min |

---

## File Mappings

| Paper Figure | Script | Output |
|--------------|--------|--------|
| Fig. 1 | `run_convergence.py` | `convergence_results.csv` |
| Fig. 2 | `run_hedging.py` | `hedging_results.csv` |
| Table 1 | `run_bootstrap_ci.py` | `bootstrap_ci.csv` |

See `paper_results/README.md` for detailed mappings.
