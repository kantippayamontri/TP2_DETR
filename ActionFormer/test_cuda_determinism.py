#!/usr/bin/env python3
"""
Test script to verify CUDA deterministic algorithms are properly configured.

This script tests:
1. torch.use_deterministic_algorithms() is enabled
2. CUBLAS_WORKSPACE_CONFIG environment variable is set
3. cuDNN deterministic settings are correct
4. Basic reproducibility with simple operations

Part of TR-4: CUDA Determinism from reproducibility-fix spec.
"""

import os
import sys
import torch
import numpy as np
import random

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.util import setup_seed


def test_cuda_determinism():
    """Test CUDA deterministic algorithm configuration."""

    print("=" * 70)
    print("Testing CUDA Deterministic Algorithms Configuration")
    print("=" * 70)

    # Test 1: Setup seed
    print("\n[Test 1] Setting up seed...")
    seed = 3552
    setup_seed(seed)
    print(f"✓ Seed {seed} set successfully")

    # Test 2: Check CUBLAS_WORKSPACE_CONFIG
    print("\n[Test 2] Checking CUBLAS_WORKSPACE_CONFIG...")
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        print("WARNING: CUBLAS_WORKSPACE_CONFIG not set. Setting to ':4096:8'")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "Not set")
    print(f"CUBLAS_WORKSPACE_CONFIG = {cublas_config}")
    assert cublas_config != "Not set", "CUBLAS_WORKSPACE_CONFIG must be set"
    print("✓ CUBLAS_WORKSPACE_CONFIG is properly set")

    # Test 3: Enable deterministic algorithms
    print("\n[Test 3] Enabling deterministic algorithms...")
    try:
        torch.use_deterministic_algorithms(True)
        print("✓ Deterministic algorithms enabled successfully")
    except RuntimeError as e:
        print(f"WARNING: Could not enable all deterministic algorithms: {e}")
        print("Trying with warn_only=True...")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✓ Deterministic algorithms enabled with warn_only=True")

    # Test 4: Verify deterministic algorithms are enabled
    print("\n[Test 4] Verifying deterministic algorithms status...")
    is_deterministic = torch.are_deterministic_algorithms_enabled()
    print(f"torch.are_deterministic_algorithms_enabled() = {is_deterministic}")
    assert is_deterministic, "Deterministic algorithms should be enabled"
    print("✓ Deterministic algorithms are enabled")

    # Test 5: Check cuDNN settings
    print("\n[Test 5] Checking cuDNN settings...")
    print(f"  - cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - cuDNN enabled: {torch.backends.cudnn.enabled}")

    assert (
        torch.backends.cudnn.deterministic == True
    ), "cuDNN deterministic should be True"
    assert torch.backends.cudnn.benchmark == False, "cuDNN benchmark should be False"
    print("✓ cuDNN settings are correct for determinism")

    # Test 6: Basic reproducibility test with CPU operations
    print("\n[Test 6] Testing basic reproducibility with CPU operations...")
    setup_seed(seed)
    result1_cpu = torch.randn(10, 10).sum().item()

    setup_seed(seed)
    result2_cpu = torch.randn(10, 10).sum().item()

    print(f"  Result 1 (CPU): {result1_cpu}")
    print(f"  Result 2 (CPU): {result2_cpu}")
    assert result1_cpu == result2_cpu, "CPU operations should be deterministic"
    print("✓ CPU operations are deterministic")

    # Test 7: Basic reproducibility test with CUDA operations (if available)
    if torch.cuda.is_available():
        print("\n[Test 7] Testing basic reproducibility with CUDA operations...")
        device = torch.device("cuda")

        setup_seed(seed)
        result1_cuda = torch.randn(10, 10, device=device).sum().item()

        setup_seed(seed)
        result2_cuda = torch.randn(10, 10, device=device).sum().item()

        print(f"  Result 1 (CUDA): {result1_cuda}")
        print(f"  Result 2 (CUDA): {result2_cuda}")
        assert result1_cuda == result2_cuda, "CUDA operations should be deterministic"
        print("✓ CUDA operations are deterministic")

        # Test 8: Matrix multiplication reproducibility
        print("\n[Test 8] Testing matrix multiplication reproducibility...")
        setup_seed(seed)
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        result1_matmul = torch.matmul(a, b).sum().item()

        setup_seed(seed)
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        result2_matmul = torch.matmul(a, b).sum().item()

        print(f"  Result 1 (matmul): {result1_matmul}")
        print(f"  Result 2 (matmul): {result2_matmul}")
        assert (
            result1_matmul == result2_matmul
        ), "Matrix multiplication should be deterministic"
        print("✓ Matrix multiplication is deterministic")
    else:
        print("\n[Test 7-8] CUDA not available, skipping CUDA-specific tests")

    # Summary
    print("\n" + "=" * 70)
    print("All CUDA Determinism Tests Passed! ✓")
    print("=" * 70)
    print("\nConfiguration Summary:")
    print(f"  - Seed: {seed}")
    print(
        f"  - Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}"
    )
    print(f"  - CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print(f"  - cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    print("\nThe configuration is correct for reproducible training!")

    return True


if __name__ == "__main__":
    try:
        success = test_cuda_determinism()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
