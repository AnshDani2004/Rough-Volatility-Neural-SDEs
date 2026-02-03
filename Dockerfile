# Dockerfile for Rough Volatility Neural SDE
# Provides exact reproducibility

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Set reproducibility environment variables
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=42

# Verify installation
RUN pytest tests/ -q --tb=no

# Default command: run quick experiments
CMD ["bash", "-c", "python experiments/run_convergence.py --quick && python experiments/run_hedging.py --quick"]
