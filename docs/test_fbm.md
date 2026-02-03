# Statistical Tests for fBM Generator

This document explains the tests in `test_fbm.py` that verify the correctness of the Davies-Harte implementation.

---

## Why Statistical Testing?

Unlike deterministic code, stochastic generators cannot be tested with exact equality checks. Instead, we verify **statistical properties** that the generated samples must satisfy.

For fBM with Hurst parameter $H$, the key property is:

> **Variance Scaling Law:** $\text{Var}(\Delta B^H) = \Delta t^{2H}$

---

## Test 1: `test_variance_scaling`

### What It Tests

The variance of fBM increments should scale as $\Delta t^{2H}$.

### Methodology

1. Generate 10,000 paths with $H = 0.1$
2. Compute empirical variance across all paths and time steps
3. Compare to theoretical value $\Delta t^{2H}$
4. Assert relative error < 1%

### Why 10,000 Paths?

By the **Central Limit Theorem**, the sample variance converges to the true variance as $1/\sqrt{N}$:

$$\text{Relative Error} \approx \frac{1}{\sqrt{N \cdot n_{\text{steps}}}}$$

With $N = 10,000$ paths and $n_{\text{steps}} = 256$:

$$\text{Expected Error} \approx \frac{1}{\sqrt{10000 \times 256}} \approx 0.06\%$$

This gives us a comfortable margin below the 1% threshold.

---

## Test 2: `test_increment_shape`

Verifies that the output array has the correct dimensions:
- Shape: `(batch_size, n_steps)`

---

## Test 3: `test_eigenvalue_positivity`

The Davies-Harte algorithm requires all eigenvalues of the circulant embedding to be non-negative. This test ensures the algorithm doesn't silently fail.

---

## Test 4: `test_mean_zero`

fBM increments should have zero mean. We test that the sample mean is within 3 standard errors of zero.

---

## Running the Tests

```bash
cd /Users/ansh/Developer/Rough\ Volatility\ Neural\ SDEs
python -m pytest tests/test_fbm.py -v
```

Expected output:
```
tests/test_fbm.py::test_variance_scaling PASSED
tests/test_fbm.py::test_increment_shape PASSED
tests/test_fbm.py::test_eigenvalue_positivity PASSED
tests/test_fbm.py::test_mean_zero PASSED
```
