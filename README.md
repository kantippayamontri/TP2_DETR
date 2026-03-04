# TP2_DETR

## Training

```bash
bash train_all.sh
```

## Visualization

```bash
python utils/qualitative.py
```

## Reproducibility

This project implements deterministic training to ensure reproducible results across multiple runs. When properly configured, you should get identical training results (same loss values, metrics, and model weights) across different runs.

### Environment Variables

To enable reproducible training, set the following environment variables before running training:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

These variables ensure:

- `PYTHONHASHSEED=0`: Deterministic Python hash functions
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`: Deterministic CUDA operations

### Using Seeds

The training scripts support a `--seed` parameter to control randomness:

```bash
# Train with a specific seed
python ActionFormer/main.py --seed 42

# Or use the training script
bash train.sh --seed 42
```

When you use the same seed with the same environment variables, you will get identical results.

### Expected Behavior

With reproducibility enabled:

- **Identical loss values** across runs with the same seed
- **Identical model weights** after training
- **Identical evaluation metrics** on validation/test sets
- **Deterministic data loading** order and augmentations

### Limitations

Reproducibility is guaranteed only under specific conditions. Understanding these limitations is important for research and production use.

#### Hardware Dependencies

- **Same GPU model required**: Different GPU architectures (e.g., V100 vs A100) may produce different numerical results due to different floating-point implementations and optimized kernels
- **Single GPU only**: Multi-GPU training reproducibility is not currently supported (out of scope)
- **GPU memory layout**: Results may vary if GPU memory allocation patterns differ between runs

#### Software Version Dependencies

- **PyTorch version**: Must use the same PyTorch version (>= 1.8 required for deterministic algorithm support)
- **CUDA version**: Must use the same CUDA version (>= 10.2 required for deterministic CUDA operations)
- **cuDNN version**: Must use the same cuDNN version, as different versions may use different algorithms
- **Python version**: Python >= 3.7 required for guaranteed dictionary ordering

#### Operating System Dependencies

- **Same OS required**: Different operating systems may have different floating-point implementations
- **System libraries**: Differences in BLAS/LAPACK implementations can affect numerical results

#### Partial Determinism with warn_only=True

The implementation uses `torch.use_deterministic_algorithms(True, warn_only=True)`, which means:

- **Some operations may not be fully deterministic**: PyTorch will warn about non-deterministic operations but continue execution
- **Best-effort reproducibility**: Most operations are deterministic, but a few may still have minor numerical variations
- **Check warnings**: Review training logs for any deterministic algorithm warnings

#### Performance Impact

- **Training slowdown**: Deterministic algorithms typically run 2-5% slower than non-deterministic versions
- **cuDNN benchmark disabled**: `torch.backends.cudnn.benchmark = False` prevents automatic algorithm selection, which may reduce performance on some models
- **Memory overhead**: `CUBLAS_WORKSPACE_CONFIG=:4096:8` allocates additional workspace memory for deterministic operations

#### Known Non-Deterministic Operations

Some PyTorch operations may not have deterministic implementations:

- **Certain scatter/gather operations**: May produce warnings with `warn_only=True`
- **Some pooling operations**: Depending on PyTorch version
- **Atomic operations**: Some CUDA atomic operations are inherently non-deterministic

#### Multi-GPU and Distributed Training

- **Not supported**: Reproducibility across multiple GPUs or distributed training is out of scope
- **Different communication patterns**: GPU communication order can introduce non-determinism
- **Gradient synchronization**: AllReduce operations may have non-deterministic ordering

#### Edge Cases

- **Numerical precision**: Floating-point operations are not associative; extremely long training runs may accumulate small differences
- **Dynamic operations**: Operations with dynamic shapes or control flow may be harder to make deterministic
- **Third-party libraries**: External libraries (e.g., data augmentation) may introduce non-determinism if not properly seeded

#### Verification Recommendations

To verify reproducibility in your environment:

1. Run the same training command 2-3 times with identical seeds
2. Compare loss values at each step (should be bit-exact identical)
3. Compare final model weights using checksum or hash
4. Check training logs for any deterministic algorithm warnings

If results differ, check:

- Environment variables are set correctly
- Hardware and software versions match
- No external sources of randomness (e.g., system time, network I/O)

### Example: Running Reproducible Training

```bash
# Set environment variables
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Activate conda environment
conda activate gap

# Run training with seed
bash train_all.sh --seed 42
```

To verify reproducibility, run the same command multiple times and compare the results - they should be identical.
