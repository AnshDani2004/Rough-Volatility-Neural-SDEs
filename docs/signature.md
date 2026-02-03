# Signature Loss: Theory and Implementation

This document explains the `SigMMDLoss` class in `signature.py` and why signatures are essential for comparing stochastic path distributions.

---

## 1. Why Not Use MSE Loss?

**Problem:** MSE (Mean Squared Error) compares paths point-by-point:
$$L_{\text{MSE}} = \frac{1}{T} \sum_t |X_t^{\text{real}} - X_t^{\text{gen}}|^2$$

| Issue | Explanation |
|-------|-------------|
| **Ignores ordering** | Two paths with same points in different order have same MSE |
| **Sensitive to alignment** | Small time shifts cause large errors |
| **Ignores dynamics** | Doesn't capture how paths *evolve* |

> [!IMPORTANT]
> For generative models, we want to match the **distribution** of paths, not individual paths.

---

## 2. Path Signatures: Capturing Path Geometry

The **signature** of a path $X: [0,T] \to \mathbb{R}^d$ is the sequence of iterated integrals:

$$S(X) = \left(1, \underbrace{\int_0^T dX_t}_{S^1}, \underbrace{\int_0^T \int_0^t dX_s \otimes dX_t}_{S^2}, \ldots\right)$$

### Why Signatures Work

| Property | Meaning |
|----------|---------|
| **Uniqueness** | Different paths (up to reparametrization) have different signatures |
| **Characteristic** | The expected signature $\mathbb{E}[S(X)]$ determines the distribution |
| **Graded structure** | Truncating at depth $k$ captures interactions up to order $k$ |

---

## 3. Log-Signatures: Efficient Representation

The **log-signature** is a more compact encoding via the logarithm in the tensor algebra:

$$\log S(X) = S^1 + \frac{1}{2}[S^1, S^1] + \ldots$$

For depth 4 in dimension $d$, the log-signature has dimension:

$$\dim = d + \frac{d(d-1)}{2} + \frac{d^2(d-1)}{3} + \ldots$$

---

## 4. Maximum Mean Discrepancy (MMD)

**MMD** measures the distance between two distributions using kernel mean embeddings:

$$\text{MMD}^2(P, Q) = \mathbb{E}_{X,X' \sim P}[k(X,X')] + \mathbb{E}_{Y,Y' \sim Q}[k(Y,Y')] - 2\mathbb{E}_{X \sim P, Y \sim Q}[k(X,Y)]$$

### Signature Kernel

For paths, we use a **signature kernel**:
$$k(X, Y) = \langle S(X), S(Y) \rangle_{\text{weighted}}$$

In our implementation, we use the simpler approach of **L2 distance on expected log-signatures**:
$$L = \left\| \mathbb{E}[\text{logsig}(X^{\text{real}})] - \mathbb{E}[\text{logsig}(X^{\text{gen}})] \right\|^2$$

---

## 5. Implementation Details

### Using `iisignature`

```python
import iisignature

# Prepare for log-signature computation
s = iisignature.prepare(dim=2, depth=4)  # dim includes time

# Compute log-signature for each path
logsig = iisignature.logsig(path, s)  # path: (n_steps, dim)
```

### SigMMDLoss Class

```python
class SigMMDLoss(nn.Module):
    def forward(self, real_paths, gen_paths):
        # 1. Add time channel to paths
        # 2. Compute log-signatures
        # 3. Return MMD on signature space
```

---

## 6. Connection to Training

During training:
- Real paths: Actual market data
- Generated paths: Output from `RoughNeuralSDE`
- Loss: `SigMMDLoss(real, generated)`

Minimizing this loss makes the generated path distribution match the real distribution.
