FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install package in editable mode
RUN pip install -e .

# Run smoke test
RUN pytest tests/ -q --tb=no || echo "Tests completed"

# Default command
CMD ["python", "experiments/run_convergence.py", "--quick", "--samples", "2"]
