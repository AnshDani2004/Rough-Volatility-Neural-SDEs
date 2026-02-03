# Tests for Rough Neural SDE Generator

This document explains the tests in `test_generator.py`.

---

## Test Overview

| Test | Purpose |
|------|---------|
| `test_output_shape` | Verify paths have shape `(batch, n_steps+1)` |
| `test_forward_backward` | Ensure gradients flow through the model |
| `test_positive_diffusion` | Confirm $\sigma(t, x) > 0$ everywhere |
| `test_euler_vs_euler_heun` | Both methods produce valid paths |
| `test_custom_initial_condition` | Paths respect $X_0$ |

---

## Gradient Flow Test

The model must be trainable, so we verify:

```python
loss = paths.sum()
loss.backward()
assert model.drift_net.net[0].weight.grad is not None
```

---

## Running Tests

```bash
python -m pytest tests/test_generator.py -v
```
