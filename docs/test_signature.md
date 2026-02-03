# Tests for Signature Loss

This document explains the tests in `test_signature.py`.

---

## Test Overview

| Test | Purpose |
|------|---------|
| `test_logsig_output_shape` | Verify log-signature dimension matches expected |
| `test_loss_zero_for_identical` | Loss should be ~0 for identical distributions |
| `test_loss_positive_for_different` | Loss should be > 0 for different distributions |
| `test_add_time_channel` | Verify time channel is correctly added |
| `test_full_mmd` | Test full kernel MMD computation |

---

## Key Test: Discriminative Power

```python
# Same distribution should have low loss
loss_same = loss_fn(paths, paths)  # Should be ~0

# Different distributions should have high loss
loss_diff = loss_fn(paths1, paths2)  # Should be > 0

assert loss_diff > loss_same
```
