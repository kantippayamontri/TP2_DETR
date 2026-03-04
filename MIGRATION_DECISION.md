# Migration Guide Decision - Reproducibility Fix

## Decision: No Migration Guide Required

After evaluating the reproducibility fixes implemented in TP2_DETR, we have determined that **a formal migration guide is not necessary**.

## Rationale

### 1. Transparent Changes

All reproducibility fixes are transparent to end users:

- Environment variables (`PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`) are automatically set in shell scripts
- Users who run `train_all.sh` or `train.sh` get reproducibility automatically
- No manual configuration required

### 2. Backward Compatibility

The changes maintain full backward compatibility:

- No API changes or breaking modifications
- Existing code continues to work without modification
- Users can still run training without reproducibility if they bypass the shell scripts
- All changes are additive, not destructive

### 3. Comprehensive Documentation

The README.md already provides complete documentation:

- Clear explanation of reproducibility features
- Environment variable requirements
- Usage examples
- Limitations and expected behavior

### 4. Code-Level Documentation

All modified code sections include detailed comments:

- train.py: Explains sorted dictionary keys for deterministic loss computation
- dataset.py: Documents sorted set operations for deterministic data loading
- main.py: Describes torch.use_deterministic_algorithms() and its requirements
- Shell scripts: Extensive comments on environment variables

## What Changed (Summary)

For reference, here's what was modified:

### Shell Scripts

- `train_all.sh` and `train.sh` now export `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- These are set automatically when using the scripts

### Code Changes

- **train.py**: Dictionary keys sorted before iteration in loss computation
- **dataset.py**: Set operations followed by sorting for deterministic video ordering
- **main.py**: Added `torch.use_deterministic_algorithms(True, warn_only=True)`
- **utils/util.py**: Enhanced seed setup with additional documentation

### Documentation

- **README.md**: New "Reproducibility" section with complete usage guide

## For Users

### If you use the provided shell scripts:

**No action required.** Reproducibility is enabled automatically.

```bash
bash train_all.sh  # Reproducibility enabled by default
```

### If you run Python directly:

Set environment variables before running:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python ActionFormer/main.py --seed 42 [other args]
```

### If you don't need reproducibility:

Continue using your existing workflow. The changes don't break anything.

## Conclusion

The reproducibility fixes are designed to be seamless and non-intrusive. Users benefit from reproducibility automatically when using the standard training scripts, while advanced users who need custom workflows can easily adapt by setting two environment variables. No migration is necessary.

---

**Date**: 2024-03-04  
**Spec**: reproducibility-fix  
**Task**: 10.3 - Create migration guide if needed
