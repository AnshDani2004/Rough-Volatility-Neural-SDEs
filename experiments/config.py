"""
Experiment configuration.

Central location for all experiment parameters and seeds.
"""

# =============================================================================
# SEED POLICY
# =============================================================================
# Default seed for reproducibility. Override with --seed in CLI.
SEED = 42

# =============================================================================
# CONVERGENCE EXPERIMENT
# =============================================================================
CONVERGENCE_H_VALUES = [0.05, 0.1, 0.2]
CONVERGENCE_DT_POWERS = [4, 5, 6, 7, 8, 9, 10]  # 2^power steps
CONVERGENCE_N_FINE = 2**14
CONVERGENCE_N_SAMPLES = 100
CONVERGENCE_T = 1.0

# Quick mode (for CI)
CONVERGENCE_QUICK_SAMPLES = 5
CONVERGENCE_QUICK_DT_POWERS = [4, 5, 6]

# =============================================================================
# HEDGING EXPERIMENT
# =============================================================================
HEDGING_N_PATHS = 1000
HEDGING_N_STEPS = 50
HEDGING_STRIKE_PCT = 1.0  # ATM
HEDGING_TRANSACTION_COST = 0.001
HEDGING_VOLATILITY = 0.2
HEDGING_T = 1.0

# Quick mode
HEDGING_QUICK_PATHS = 100
HEDGING_QUICK_STEPS = 20

# Strategies to compare
HEDGING_STRATEGIES = [
    "NeuralHedge",
    "BlackScholesDelta",
    "HestonDelta",
    "NaiveHedge",
]

# =============================================================================
# BOOTSTRAP
# =============================================================================
BOOTSTRAP_N_SAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05

# Quick mode
BOOTSTRAP_QUICK_SAMPLES = 100

# =============================================================================
# HESTON CALIBRATION
# =============================================================================
HESTON_INIT_PARAMS = {
    "kappa": 2.0,      # Mean reversion speed
    "theta": 0.04,     # Long-term variance
    "sigma": 0.3,      # Vol of vol
    "rho": -0.7,       # Correlation
    "v0": 0.04,        # Initial variance
}

# =============================================================================
# PATHS
# =============================================================================
OUTPUT_DIR = "outputs"
FIGURES_DIR = "figures"
