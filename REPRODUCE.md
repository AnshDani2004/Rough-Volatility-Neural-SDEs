# Reproducibility Guide

Complete instructions to reproduce all figures, tables, and statistical results from the paper.

---

## Quick Verification

```bash
# Smoke test (< 2 minutes)
pytest tests/ -q && python experiments/run_convergence.py --quick
```

---

## Environment Setup

### Option 1: pip (Recommended)

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with pinned versions
pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker (Exact Reproduction)

```bash
docker build -t rough-vol .
docker run -it rough-vol bash
```

### Verify Installation

```bash
pytest tests/ -v
# Expected: 52 passed
```

---

## Seed Policy

| Setting | Value |
|---------|-------|
| Default seed | `SEED=42` |
| Override | `--seed <N>` |

**For exact CPU reproducibility:**
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

> ⚠️ **GPU Note:** GPU floating-point operations may have minor variance. For exact reproduction of paper results, use CPU.

---

## Full Paper Reproduction

Run all experiments in sequence:

```bash
# 1. Convergence Analysis (Figure 1, ~10 min)
python experiments/run_convergence.py \
    --samples 100 \
    --seed 42 \
    --out outputs/convergence_results.csv

# 2. Heston Calibration (for Heston baseline)
python experiments/calibrate_heston.py \
    --seed 42 \
    --out outputs/heston_params.json

# 3. Hedging Comparison with Bootstrap CIs (Figure 2, Table 1, ~15 min)
python experiments/run_hedging.py \
    --paths 1000 \
    --n-bootstrap 1000 \
    --heston-params outputs/heston_params.json \
    --seed 42 \
    --out outputs/hedging_results.csv

# 4. Generate Figures
python scripts/plot_all_figures.py \
    --convergence outputs/convergence_results.csv \
    --hedging outputs/hedging_results.csv \
    --out figures/
```

---

## Individual Experiments

### Experiment 1: Convergence Analysis (Section 4.1)

Validates strong convergence rate for rough volatility SDEs.

```bash
python experiments/run_convergence.py --samples 100 --out outputs/convergence_results.csv
```

**Output columns:** `H, dt, error_mean, error_std, slope, slope_lower, slope_upper, R2`

**Expected slopes:**
| H | Slope | 95% CI |
|---|-------|--------|
| 0.05 | ~0.48 | [0.41, 0.55] |
| 0.1 | ~0.52 | [0.45, 0.59] |
| 0.2 | ~0.55 | [0.48, 0.62] |

---

### Experiment 2: Hedging Comparison (Section 4.2)

Compares Neural, Black-Scholes, Heston, and Naive hedging strategies.

```bash
python experiments/run_hedging.py \
    --paths 1000 \
    --n-bootstrap 1000 \
    --heston-params outputs/heston_params.json \
    --out outputs/hedging_results.csv
```

**Output includes:**
- Mean PnL with 95% bootstrap CIs
- 5% CVaR with 95% bootstrap CIs
- Hypothesis tests (Neural vs BlackScholes)

**Sign convention:** Positive = Profit, Negative = Loss

---

### Hypothesis Tests (P-values)

Automatically computed by `run_hedging.py`:

```
HYPOTHESIS TESTS: Neural vs BlackScholes
H0: No difference | H1: Neural better (one-sided)

Mean PnL Difference:  +X.XXXX  p = 0.XXXX ***
CVaR Difference:      +X.XXXX  p = 0.XXXX ***

Significance: * p<0.05, ** p<0.01, *** p<0.001
```

Results saved to: `outputs/hypothesis_tests.json`

---

## Output File Mapping

| Paper Element | Output File | Command |
|--------------|-------------|---------|
| Figure 1 (Convergence) | `figures/convergence_regimes.png` | `run_convergence.py` |
| Figure 2 (Hedging) | `figures/hedging_comparison.png` | `run_hedging.py` |
| Table 1 (Strategy Metrics) | `outputs/hedging_results.csv` | `run_hedging.py` |
| Table 2 (Bootstrap CIs) | `outputs/hedging_results.csv` | `run_hedging.py` |
| Table 3 (Hypothesis Tests) | `outputs/hypothesis_tests.json` | `run_hedging.py` |

---

## Hardware Requirements

| Experiment | RAM | CPU Time (8-core) |
|------------|-----|-------------------|
| Convergence | 4 GB | 10 min |
| Hedging | 8 GB | 15 min |
| Full reproduction | 8 GB | ~30 min |

---

## Troubleshooting

**Import errors:** Ensure `pip install -e .` was run.

**Exact reproduction mismatch:** Use CPU and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Memory issues:** Reduce `--paths` or use `--quick` mode.
