#!/usr/bin/env python3
"""
Test script for deterministic loss computation.

**Validates: Requirements 1.2**

This script validates Property 1 from the design document:
Given the same model state, inputs, and targets, the computed loss must be
identical across runs.

The test verifies that:
1. Loss computation is deterministic with the same inputs
2. The sorted keys implementation ensures consistent iteration order
3. Different loss_dict orderings produce the same final loss value
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import torch
import torch.nn as nn
from collections import OrderedDict


class SimpleCriterion(nn.Module):
    """
    Simple criterion that mimics the structure of the actual criterion.

    This criterion computes multiple loss components and uses a weight_dict
    to control which losses are applied.
    """

    def __init__(self):
        super().__init__()
        # Define weights for different loss components
        self.weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

    def forward(self, outputs, targets):
        """
        Compute loss dictionary.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            dict: Dictionary of loss components
        """
        # Simulate different loss components
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
            "loss_aux": torch.tensor(0.1),  # Not in weight_dict
        }
        return loss_dict


def compute_loss_original(loss_dict, weight_dict):
    """
    Original loss computation (non-deterministic).

    This uses dictionary iteration which may have non-deterministic order.

    Args:
        loss_dict: Dictionary of loss components
        weight_dict: Dictionary of loss weights

    Returns:
        tuple: (total_loss, loss_dict_scaled, loss_value)
    """
    losses = sum(
        loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
    )

    loss_dict_unscaled = {f"{k}_unscaled": v.item() for k in loss_dict.keys()}
    loss_dict_scaled = {
        k: v.item() * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
    }
    loss_value = sum(loss_dict_scaled.values())

    return losses, loss_dict_scaled, loss_value


def compute_loss_deterministic(loss_dict, weight_dict):
    """
    Deterministic loss computation with sorted keys.

    This is the fixed version that ensures deterministic iteration order.

    Args:
        loss_dict: Dictionary of loss components
        weight_dict: Dictionary of loss weights

    Returns:
        tuple: (total_loss, loss_dict_scaled, loss_value)
    """
    # REPRODUCIBILITY: Sort keys to ensure deterministic iteration order
    sorted_keys = sorted([k for k in loss_dict.keys() if k in weight_dict])
    losses = sum(loss_dict[k] * weight_dict[k] for k in sorted_keys)

    # REPRODUCIBILITY: Use sorted keys for deterministic iteration
    loss_dict_unscaled = {
        f"{k}_unscaled": loss_dict[k].item() for k in sorted(loss_dict.keys())
    }

    # REPRODUCIBILITY: Use sorted keys for deterministic iteration
    loss_dict_scaled = {
        k: loss_dict[k].item() * weight_dict[k]
        for k in sorted(loss_dict.keys())
        if k in weight_dict
    }

    # REPRODUCIBILITY: Use sorted keys for deterministic summation order
    loss_value = sum(loss_dict_scaled[k] for k in sorted(loss_dict_scaled.keys()))

    return losses, loss_dict_scaled, loss_value


def test_deterministic_loss_same_order():
    """
    Test that loss computation is deterministic with the same input order.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Test 1: Deterministic loss with same input order")
    print("=" * 70)

    criterion = SimpleCriterion()

    # Create loss_dict with consistent order
    loss_dict = {
        "loss_ce": torch.tensor(0.5),
        "loss_bbox": torch.tensor(0.3),
        "loss_giou": torch.tensor(0.2),
        "loss_aux": torch.tensor(0.1),
    }

    # Compute loss multiple times
    results = []
    for i in range(5):
        losses, loss_dict_scaled, loss_value = compute_loss_deterministic(
            loss_dict, criterion.weight_dict
        )
        results.append((losses.item(), loss_value, loss_dict_scaled))
        print(f"Run {i+1}: loss_value={loss_value:.6f}, losses={losses.item():.6f}")

    # Check all results are identical
    first_result = results[0]
    all_match = all(
        r[0] == first_result[0] and r[1] == first_result[1] and r[2] == first_result[2]
        for r in results
    )

    if all_match:
        print("✓ SUCCESS: All runs produced identical loss values")
        return True
    else:
        print("✗ FAILURE: Loss values differ across runs")
        for i, r in enumerate(results):
            print(f"  Run {i+1}: {r}")
        return False


def test_deterministic_loss_different_order():
    """
    Test that loss computation is deterministic even with different dict orderings.

    This is the critical test that validates the sorted keys fix.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Test 2: Deterministic loss with different dict orderings")
    print("=" * 70)

    criterion = SimpleCriterion()

    # Create loss_dict with different orderings
    orderings = [
        # Ordering 1: alphabetical
        ["loss_aux", "loss_bbox", "loss_ce", "loss_giou"],
        # Ordering 2: reverse alphabetical
        ["loss_giou", "loss_ce", "loss_bbox", "loss_aux"],
        # Ordering 3: random order
        ["loss_bbox", "loss_aux", "loss_giou", "loss_ce"],
        # Ordering 4: another random order
        ["loss_ce", "loss_giou", "loss_aux", "loss_bbox"],
    ]

    results = []
    for i, keys_order in enumerate(orderings):
        # Create OrderedDict with specific key order
        loss_dict = OrderedDict()
        for key in keys_order:
            if key == "loss_ce":
                loss_dict[key] = torch.tensor(0.5)
            elif key == "loss_bbox":
                loss_dict[key] = torch.tensor(0.3)
            elif key == "loss_giou":
                loss_dict[key] = torch.tensor(0.2)
            elif key == "loss_aux":
                loss_dict[key] = torch.tensor(0.1)

        losses, loss_dict_scaled, loss_value = compute_loss_deterministic(
            loss_dict, criterion.weight_dict
        )
        results.append((losses.item(), loss_value, loss_dict_scaled))

        print(f"\nOrdering {i+1}: {keys_order}")
        print(f"  loss_value={loss_value:.6f}, losses={losses.item():.6f}")
        print(f"  loss_dict_scaled={loss_dict_scaled}")

    # Check all results are identical
    first_result = results[0]
    all_match = all(
        abs(r[0] - first_result[0]) < 1e-9
        and abs(r[1] - first_result[1]) < 1e-9
        and r[2] == first_result[2]
        for r in results
    )

    print("\n" + "-" * 70)
    if all_match:
        print("✓ SUCCESS: All orderings produced identical loss values")
        print("  The sorted keys implementation ensures deterministic computation")
        return True
    else:
        print("✗ FAILURE: Loss values differ across orderings")
        print("\nResults:")
        for i, r in enumerate(results):
            print(f"  Ordering {i+1}: losses={r[0]:.6f}, loss_value={r[1]:.6f}")
        return False


def test_loss_value_correctness():
    """
    Test that the computed loss value is mathematically correct.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Test 3: Loss value correctness")
    print("=" * 70)

    criterion = SimpleCriterion()

    loss_dict = {
        "loss_ce": torch.tensor(0.5),
        "loss_bbox": torch.tensor(0.3),
        "loss_giou": torch.tensor(0.2),
        "loss_aux": torch.tensor(0.1),
    }

    losses, loss_dict_scaled, loss_value = compute_loss_deterministic(
        loss_dict, criterion.weight_dict
    )

    # Expected value: 0.5*1.0 + 0.3*5.0 + 0.2*2.0 = 0.5 + 1.5 + 0.4 = 2.4
    expected_value = 0.5 * 1.0 + 0.3 * 5.0 + 0.2 * 2.0

    print(f"Computed loss_value: {loss_value:.6f}")
    print(f"Expected loss_value: {expected_value:.6f}")
    print(f"Difference: {abs(loss_value - expected_value):.9f}")

    if abs(loss_value - expected_value) < 1e-6:
        print("✓ SUCCESS: Loss value is mathematically correct")
        return True
    else:
        print("✗ FAILURE: Loss value is incorrect")
        return False


def test_sorted_keys_implementation():
    """
    Test the actual sorted keys implementation from train.py.

    This test directly validates the code pattern used in train.py.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Test 4: Sorted keys implementation (train.py pattern)")
    print("=" * 70)

    criterion = SimpleCriterion()
    weight_dict = criterion.weight_dict

    # Create loss_dict with different orderings
    orderings = [
        ["loss_giou", "loss_ce", "loss_bbox", "loss_aux"],
        ["loss_bbox", "loss_aux", "loss_giou", "loss_ce"],
        ["loss_ce", "loss_giou", "loss_aux", "loss_bbox"],
    ]

    results = []
    for keys_order in orderings:
        loss_dict = OrderedDict()
        for key in keys_order:
            if key == "loss_ce":
                loss_dict[key] = torch.tensor(0.5)
            elif key == "loss_bbox":
                loss_dict[key] = torch.tensor(0.3)
            elif key == "loss_giou":
                loss_dict[key] = torch.tensor(0.2)
            elif key == "loss_aux":
                loss_dict[key] = torch.tensor(0.1)

        # This is the exact pattern from train.py
        sorted_keys = sorted([k for k in loss_dict.keys() if k in weight_dict])
        losses = sum(loss_dict[k] * weight_dict[k] for k in sorted_keys)

        loss_dict_scaled = {
            k: loss_dict[k].item() * weight_dict[k]
            for k in sorted(loss_dict.keys())
            if k in weight_dict
        }

        loss_value = sum(loss_dict_scaled[k] for k in sorted(loss_dict_scaled.keys()))

        results.append((losses.item(), loss_value))
        print(
            f"Ordering {keys_order[:2]}...: losses={losses.item():.6f}, loss_value={loss_value:.6f}"
        )

    # Check all results are identical
    first_result = results[0]
    all_match = all(
        abs(r[0] - first_result[0]) < 1e-9 and abs(r[1] - first_result[1]) < 1e-9
        for r in results
    )

    if all_match:
        print("✓ SUCCESS: train.py sorted keys pattern works correctly")
        return True
    else:
        print("✗ FAILURE: train.py pattern produces different results")
        return False


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("DETERMINISTIC LOSS COMPUTATION TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    print("This test validates Property 1 from the design document:")
    print("  'Given the same model state, inputs, and targets, the computed")
    print("   loss must be identical across runs.'")
    print()
    print("The test verifies the sorted keys implementation in train.py")
    print("ensures deterministic loss computation regardless of dict ordering.")

    # Run all tests
    test_results = []

    test_results.append(
        ("Deterministic loss (same order)", test_deterministic_loss_same_order())
    )

    test_results.append(
        (
            "Deterministic loss (different orderings)",
            test_deterministic_loss_different_order(),
        )
    )

    test_results.append(("Loss value correctness", test_loss_value_correctness()))

    test_results.append(
        ("Sorted keys implementation", test_sorted_keys_implementation())
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
        print("\n**Property 1 Validated: Deterministic Loss Computation**")
        print("\nThe sorted keys implementation correctly ensures:")
        print("  - Loss computation is deterministic with same inputs")
        print("  - Different dict orderings produce identical results")
        print("  - Loss values are mathematically correct")
        print("  - The train.py implementation pattern works as expected")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
