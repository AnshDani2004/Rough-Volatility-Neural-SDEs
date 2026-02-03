#!/bin/bash
# Quick start commands for the Rough Volatility Neural SDE framework
# Run from repository root

set -e

echo "=== Quick Start Script ==="
echo ""

# Install
echo "1. Installing dependencies..."
pip install -e . > /dev/null 2>&1 || pip install -r requirements.txt

# Tests
echo "2. Running tests..."
pytest tests/ -q --tb=no

# Quick experiments
echo "3. Running quick convergence experiment..."
python experiments/run_convergence.py --quick --samples 5 --out outputs/convergence_quick.csv

echo "4. Running quick hedging experiment..."
python experiments/run_hedging.py --quick --paths 100 --out outputs/hedging_quick.csv

echo "5. Running bootstrap CI..."
python experiments/run_bootstrap_ci.py --quick --n-bootstrap 100 --out outputs/bootstrap_quick.csv

# Generate figures
echo "6. Generating figures..."
python scripts/plot_all_figures.py \
    --convergence outputs/convergence_quick.csv \
    --hedging outputs/hedging_quick.csv \
    --out figures/

echo ""
echo "=== All quick tests passed! ==="
echo "Results saved to outputs/"
echo "Figures saved to figures/"
