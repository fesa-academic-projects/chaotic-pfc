# Performance baselines

Reference hardware for the current baseline:

- **CPU**: Intel Core i7-8550U @ 1.80 GHz (turbo 4.00 GHz, 4C/8T)
- **RAM**: 32 GB
- **GPU**: AMD Radeon RX 550 (2 GB VRAM)
- **Machine**: Lenovo ThinkPad E480
- **OS**: Linux (Void-based, kernel 7.x)
- **Python**: 3.14.5 (CPython)
- **Numba**: 0.61.2

## How to update

After intentional performance changes (new algorithm, SIMD, etc.):

```bash
# Run all benchmarks and save a new baseline
pytest benchmarks/ --benchmark-only --benchmark-save=baseline_vX_Y_Z

# Copy the generated JSON to the baseline directory
mv .benchmarks/*/baseline_vX_Y_Z.json benchmarks/baseline/

# Commit with a note in the changelog
```

## When to update

- After a major release where performance was intentionally improved
- After Numba version bumps that affect JIT compilation
- **Not** after refactors that should not affect performance (if baseline regresses, fix the refactor)
