# Performance baselines

`baseline_v0_7_0.json` is a **historical reference only**. CI no longer
compares against it: the benchmark job compares the pull-request base and head
on the same runner.

Reference hardware for the historical baseline:

- **CPU**: Intel Core i7-8550U @ 1.80 GHz (turbo 4.00 GHz, 4C/8T)
- **RAM**: 32 GB
- **GPU**: AMD Radeon RX 550 (2 GB VRAM)
- **Machine**: Lenovo ThinkPad E480
- **OS**: Linux (Void-based, kernel 7.x)
- **Python**: 3.14.5 (CPython)
- **Numba**: 0.61.2
- **CPU governor**: powersave (the JSON does not record the governor)
- **dirty**: true (`commit_info.dirty` in the JSON)

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

- This file is kept for historical context and is not a CI gate.
- Do not add it back to the benchmark comparison. The CI benchmark job uses
  the pull-request base worktree as its reference.
- If a new historical snapshot is ever wanted, record the machine, governor,
  Python, Numba and the `dirty` flag explicitly.
