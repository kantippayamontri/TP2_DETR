#!/usr/bin/env python3
"""
Test script for deterministic model initialization.

This script validates that model parameters are initialized identically across runs
when using the same seed, verifying Property 3 from the design document.

**Validates: Property 3 (Deterministic Model Initialization)**
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import torch
import numpy as np
from utils.util import setup_seed
from models import build_model
import options


def get_model_parameters(model):
    """
    Extract all model parameters as a dictionary.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary mapping parameter names to their values (as numpy arrays)
    """
    params = {}
    for name, param in model.named_parameters():
        # Convert to CPU and numpy for comparison
        params[name] = param.detach().cpu().numpy().copy()
    return params


def compare_parameters(params1, params2, tolerance=0.0):
    """
    Compare two sets of model parameters for equality.

    Args:
        params1 (dict): First set of parameters
        params2 (dict): Second set of parameters
        tolerance (float): Tolerance for floating point comparison (0.0 for exact equality)

    Returns:
        tuple: (bool, list) - (all_match, list of mismatches)
    """
    mismatches = []

    # Check if both have the same parameter names
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            mismatches.append(f"Parameters in run 1 but not in run 2: {missing_in_2}")
        if missing_in_1:
            mismatches.append(f"Parameters in run 2 but not in run 1: {missing_in_1}")
        return False, mismatches

    # Compare each parameter
    for name in params1.keys():
        param1 = params1[name]
        param2 = params2[name]

        # Check shape
        if param1.shape != param2.shape:
            mismatches.append(
                f"Parameter '{name}': Shape mismatch {param1.shape} vs {param2.shape}"
            )
            continue

        # Check values
        if tolerance == 0.0:
            # Exact equality check
            if not np.array_equal(param1, param2):
                max_diff = np.abs(param1 - param2).max()
                mismatches.append(
                    f"Parameter '{name}': Values differ (max diff: {max_diff:.2e})"
                )
        else:
            # Tolerance-based check
            if not np.allclose(param1, param2, atol=tolerance, rtol=0):
                max_diff = np.abs(param1 - param2).max()
                mismatches.append(
                    f"Parameter '{name}': Values differ beyond tolerance "
                    f"(max diff: {max_diff:.2e}, tolerance: {tolerance:.2e})"
                )

    return len(mismatches) == 0, mismatches


def test_deterministic_model_init(seed=3552, device="cpu"):
    """
    Test that model initialization is deterministic across runs with the same seed.

    Args:
        seed (int): Seed value for reproducibility
        device (str): Device to use ('cpu' or 'cuda')

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest: Deterministic Model Initialization")
    print("=" * 70)
    print(f"Seed: {seed}, Device: {device}")

    # Parse args for model building
    sys.argv = [
        "test",
        "--dataset",
        "Thumos14",
        "--feature_type",
        "CLIP",
        "--feature_path",
        "./Thumos14/CLIP_feature_8frame/",
        "--anno_file_path",
        "./Thumos14/annotations/thumos14.json",
        "--feature_info_path",
        "./Thumos14/annotations/feature_info.json",
        "--description_file_path",
        "./Thumos14/annotations/description.json",
        "--device",
        device,
    ]
    args = options.parser.parse_args()

    try:
        # First run: setup seed and initialize model
        print("\n[Run 1] Setting up seed and initializing model...")
        setup_seed(seed)
        torch_device = torch.device(device)

        model1, criterion1, postprocessor1 = build_model(args, torch_device)
        model1.to(torch_device)

        print(
            f"Model initialized with {sum(p.numel() for p in model1.parameters())} parameters"
        )
        print("Extracting parameters from first run...")
        params1 = get_model_parameters(model1)
        print(f"Extracted {len(params1)} parameter tensors")

        # Show sample of first few parameters
        sample_params = list(params1.items())[:3]
        for name, param in sample_params:
            print(
                f"  {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}"
            )

        # Clean up first model
        del model1, criterion1, postprocessor1
        if device == "cuda":
            torch.cuda.empty_cache()

        # Second run: setup seed again and initialize model
        print("\n[Run 2] Setting up seed again and initializing model...")
        setup_seed(seed)

        model2, criterion2, postprocessor2 = build_model(args, torch_device)
        model2.to(torch_device)

        print(
            f"Model initialized with {sum(p.numel() for p in model2.parameters())} parameters"
        )
        print("Extracting parameters from second run...")
        params2 = get_model_parameters(model2)
        print(f"Extracted {len(params2)} parameter tensors")

        # Show sample of first few parameters
        sample_params = list(params2.items())[:3]
        for name, param in sample_params:
            print(
                f"  {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}"
            )

        # Compare parameters
        print("\n" + "=" * 70)
        print("Comparing model parameters...")
        all_match, mismatches = compare_parameters(params1, params2, tolerance=0.0)

        if all_match:
            print("✓ SUCCESS: All model parameters are identical across runs!")
            print("  Model initialization is deterministic with the same seed.")
            print(f"  Verified {len(params1)} parameter tensors for exact equality.")
            return True
        else:
            print("✗ FAILURE: Model parameters differ between runs!")
            print(f"\nFound {len(mismatches)} mismatch(es):")
            for i, mismatch in enumerate(mismatches[:5]):
                print(f"  {i+1}. {mismatch}")
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches) - 5} more")
            return False

    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_seeds_produce_different_params(seed1=42, seed2=123, device="cpu"):
    """
    Test that different seeds produce different model parameters.

    Args:
        seed1 (int): First seed value
        seed2 (int): Second seed value
        device (str): Device to use ('cpu' or 'cuda')

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest: Different Seeds Produce Different Parameters")
    print("=" * 70)
    print(f"Seed 1: {seed1}, Seed 2: {seed2}, Device: {device}")

    # Parse args for model building
    sys.argv = [
        "test",
        "--dataset",
        "Thumos14",
        "--feature_type",
        "CLIP",
        "--feature_path",
        "./Thumos14/CLIP_feature_8frame/",
        "--anno_file_path",
        "./Thumos14/annotations/thumos14.json",
        "--feature_info_path",
        "./Thumos14/annotations/feature_info.json",
        "--description_file_path",
        "./Thumos14/annotations/description.json",
        "--device",
        device,
    ]
    args = options.parser.parse_args()

    try:
        # First seed
        print(f"\n[Seed {seed1}] Initializing model...")
        setup_seed(seed1)
        torch_device = torch.device(device)

        model1, criterion1, postprocessor1 = build_model(args, torch_device)
        model1.to(torch_device)
        params1 = get_model_parameters(model1)

        # Clean up
        del model1, criterion1, postprocessor1
        if device == "cuda":
            torch.cuda.empty_cache()

        # Second seed
        print(f"\n[Seed {seed2}] Initializing model...")
        setup_seed(seed2)

        model2, criterion2, postprocessor2 = build_model(args, torch_device)
        model2.to(torch_device)
        params2 = get_model_parameters(model2)

        # Compare parameters - they should be different
        print("\n" + "=" * 70)
        print("Comparing model parameters...")
        all_match, _ = compare_parameters(params1, params2, tolerance=0.0)

        if not all_match:
            print("✓ SUCCESS: Different seeds produce different model parameters!")
            print("  This confirms that the seed affects model initialization.")
            return True
        else:
            print("✗ FAILURE: Different seeds produced identical parameters!")
            print("  This suggests the seed is not affecting model initialization.")
            return False

    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_model_determinism(seed=3552):
    """
    Test deterministic initialization with a simple PyTorch model.

    This is a simpler test that doesn't require the full model infrastructure.

    Args:
        seed (int): Seed value for reproducibility

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest: Simple Model Deterministic Initialization")
    print("=" * 70)
    print(f"Seed: {seed}")

    try:
        # Define a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 5)
                self.conv = torch.nn.Conv1d(1, 3, kernel_size=3)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        # First run
        print("\n[Run 1] Initializing simple model...")
        setup_seed(seed)
        model1 = SimpleModel()
        params1 = get_model_parameters(model1)

        print(f"Model has {len(params1)} parameter tensors")
        for name, param in params1.items():
            print(f"  {name}: shape={param.shape}, mean={param.mean():.6f}")

        # Second run
        print("\n[Run 2] Initializing simple model again...")
        setup_seed(seed)
        model2 = SimpleModel()
        params2 = get_model_parameters(model2)

        print(f"Model has {len(params2)} parameter tensors")
        for name, param in params2.items():
            print(f"  {name}: shape={param.shape}, mean={param.mean():.6f}")

        # Compare
        print("\n" + "=" * 70)
        print("Comparing parameters...")
        all_match, mismatches = compare_parameters(params1, params2, tolerance=0.0)

        if all_match:
            print("✓ SUCCESS: Simple model initialization is deterministic!")
            print("  setup_seed correctly seeds PyTorch's parameter initialization.")
            return True
        else:
            print("✗ FAILURE: Simple model parameters differ!")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            return False

    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("DETERMINISTIC MODEL INITIALIZATION TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()

    # Run all tests
    test_results = []

    # Test 1: Simple model determinism (quick sanity check)
    test_results.append(
        ("Simple model determinism", test_simple_model_determinism(seed=3552))
    )

    # Test 2: Full model deterministic initialization
    test_results.append(
        (
            "Full model determinism (seed=3552)",
            test_deterministic_model_init(seed=3552, device=device),
        )
    )

    # Test 3: Different seeds produce different parameters
    test_results.append(
        (
            "Different seeds",
            test_different_seeds_produce_different_params(
                seed1=42, seed2=123, device=device
            ),
        )
    )

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
        print("\nThe deterministic model initialization correctly:")
        print("  - Initializes model parameters identically with the same seed")
        print("  - Produces different parameters with different seeds")
        print("  - Works for both simple and complex models")
        print("\n**Validates: Property 3 (Deterministic Model Initialization)**")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
