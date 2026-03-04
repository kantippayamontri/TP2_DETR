#!/usr/bin/env python3
"""
Test script for setup_seed function reproducibility.

This script validates that the setup_seed function properly seeds all random
number generators (random, numpy, torch) and ensures reproducible results.
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import random
import numpy as np
import torch
from utils.util import setup_seed


def generate_random_numbers():
    """
    Generate random numbers from different sources.

    Returns:
        dict: Dictionary containing random numbers from each source
    """
    results = {}

    # Python's built-in random module
    results["python_random"] = [random.random() for _ in range(5)]
    results["python_randint"] = [random.randint(0, 100) for _ in range(5)]

    # NumPy random numbers
    results["numpy_random"] = np.random.rand(5).tolist()
    results["numpy_randint"] = np.random.randint(0, 100, size=5).tolist()
    results["numpy_randn"] = np.random.randn(5).tolist()

    # PyTorch random numbers (CPU)
    results["torch_rand"] = torch.rand(5).tolist()
    results["torch_randn"] = torch.randn(5).tolist()
    results["torch_randint"] = torch.randint(0, 100, (5,)).tolist()

    # PyTorch random numbers (CUDA if available)
    if torch.cuda.is_available():
        results["torch_cuda_rand"] = torch.rand(5, device="cuda").cpu().tolist()
        results["torch_cuda_randn"] = torch.randn(5, device="cuda").cpu().tolist()

    return results


def compare_results(results1, results2):
    """
    Compare two sets of random number results.

    Args:
        results1 (dict): First set of results
        results2 (dict): Second set of results

    Returns:
        tuple: (bool, list) - (all_match, list of mismatches)
    """
    mismatches = []

    for key in results1.keys():
        if key not in results2:
            mismatches.append(f"Key '{key}' missing in second results")
            continue

        val1 = results1[key]
        val2 = results2[key]

        # Compare lists element by element
        if val1 != val2:
            mismatches.append(f"Mismatch in '{key}': {val1} != {val2}")

    return len(mismatches) == 0, mismatches


def test_seed_reproducibility(seed=3552):
    """
    Test that setup_seed produces reproducible results.

    Args:
        seed (int): Seed value to test with

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"Testing seed reproducibility with seed={seed}")
    print("=" * 70)

    # First run: setup seed and generate random numbers
    print(f"\n[Run 1] Setting up seed {seed} and generating random numbers...")
    setup_seed(seed)
    results1 = generate_random_numbers()

    print("\nFirst run results:")
    for key, values in results1.items():
        print(f"  {key}: {values[:3]}...")  # Show first 3 values

    # Second run: setup seed again and generate random numbers
    print(f"\n[Run 2] Setting up seed {seed} again and generating random numbers...")
    setup_seed(seed)
    results2 = generate_random_numbers()

    print("\nSecond run results:")
    for key, values in results2.items():
        print(f"  {key}: {values[:3]}...")  # Show first 3 values

    # Compare results
    print("\n" + "=" * 70)
    print("Comparing results...")
    all_match, mismatches = compare_results(results1, results2)

    if all_match:
        print("✓ SUCCESS: All random number generators produce identical results!")
        print("  The setup_seed function correctly seeds all RNG sources.")
        return True
    else:
        print("✗ FAILURE: Random number generators produced different results!")
        print("\nMismatches found:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return False


def test_different_seeds():
    """
    Test that different seeds produce different results.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Testing that different seeds produce different results...")
    print("=" * 70)

    # Generate with seed 1
    setup_seed(1)
    results_seed1 = generate_random_numbers()

    # Generate with seed 2
    setup_seed(2)
    results_seed2 = generate_random_numbers()

    # These should be different
    all_match, _ = compare_results(results_seed1, results_seed2)

    if not all_match:
        print("✓ SUCCESS: Different seeds produce different results!")
        return True
    else:
        print("✗ FAILURE: Different seeds produced identical results!")
        print("  This suggests the seed is not being applied correctly.")
        return False


def test_environment_variables():
    """
    Test that environment variables are set correctly.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Testing environment variable setup...")
    print("=" * 70)

    seed = 12345
    setup_seed(seed)

    # Check PYTHONHASHSEED
    pythonhashseed = os.environ.get("PYTHONHASHSEED")
    if pythonhashseed == str(seed):
        print(f"✓ PYTHONHASHSEED correctly set to '{pythonhashseed}'")
        env_check = True
    else:
        print(f"✗ PYTHONHASHSEED is '{pythonhashseed}', expected '{seed}'")
        env_check = False

    # Check cuDNN settings
    print(f"\ncuDNN settings:")
    print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark} (should be False)")
    print(
        f"  cudnn.deterministic: {torch.backends.cudnn.deterministic} (should be True)"
    )
    print(f"  cudnn.enabled: {torch.backends.cudnn.enabled} (should be True)")

    cudnn_check = (
        not torch.backends.cudnn.benchmark
        and torch.backends.cudnn.deterministic
        and torch.backends.cudnn.enabled
    )

    if cudnn_check:
        print("✓ cuDNN settings are correct for reproducibility")
    else:
        print("✗ cuDNN settings are not configured correctly")

    return env_check and cudnn_check


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("SEED SETUP FUNCTION TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run all tests
    test_results = []

    # Test 1: Reproducibility with seed 3552
    test_results.append(
        ("Reproducibility (seed=3552)", test_seed_reproducibility(3552))
    )

    # Test 2: Reproducibility with different seed
    test_results.append(("Reproducibility (seed=42)", test_seed_reproducibility(42)))

    # Test 3: Different seeds produce different results
    test_results.append(("Different seeds", test_different_seeds()))

    # Test 4: Environment variables
    test_results.append(("Environment variables", test_environment_variables()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe setup_seed function correctly:")
        print("  - Seeds all random number generators (random, numpy, torch)")
        print("  - Produces reproducible results when called with the same seed")
        print("  - Produces different results when called with different seeds")
        print("  - Sets environment variables correctly")
        print("  - Configures cuDNN for deterministic operations")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above and fix the setup_seed function.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
