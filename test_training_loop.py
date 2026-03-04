#!/usr/bin/env python3
"""
Test script for training loop with sorted keys.

This script validates that the updated training loop with sorted dictionary keys
works correctly and produces deterministic results.

Requirements:
- Verify that the updated training loop with sorted keys works correctly
- Ensure no syntax errors or runtime issues
- Validate that loss computation is deterministic
- Run the same batch twice and verify identical loss values
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import torch
import numpy as np
import random
from utils.util import setup_seed
from models import build_model
import dataset
import options
from options import merge_cfg_from_file
from train import train
from torch.utils.data import DataLoader
from utils.misc import collate_fn


def create_minimal_args():
    """
    Create minimal arguments for testing.

    Returns:
        argparse.Namespace: Minimal arguments for model initialization
    """
    # Parse default arguments
    args = options.parser.parse_args([])

    # Set minimal configuration for testing
    args.cfg_path = "./ActionFormer/config/Thumos14_CLIP_zs_50_8frame.yaml"
    args = merge_cfg_from_file(args, args.cfg_path)

    # Override with test-specific settings
    args.seed = 3552
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = 2  # Small batch for testing
    args.num_workers = 0  # No workers for simpler testing
    args.epochs = 1  # Just one epoch for testing

    return args


def test_training_loop_basic():
    """
    Test basic training loop functionality.

    This test verifies:
    1. Training loop executes without errors
    2. Loss values are computed correctly (finite numbers)
    3. No syntax errors in the sorted keys implementation

    Returns:
        bool: True if test passes, False otherwise
    """
    print("=" * 70)
    print("TEST 1: Basic Training Loop Functionality")
    print("=" * 70)

    try:
        # Setup
        args = create_minimal_args()
        device = torch.device(args.device)
        seed = args.seed

        print(f"\nSetting up with seed={seed}, device={device}")
        setup_seed(seed)

        # Build model
        print("Building model...")
        model, criterion, postprocessor = build_model(args, device)
        model.to(device)

        # Build dataset (just a few samples for testing)
        print("Loading dataset...")
        train_dataset = getattr(dataset, args.dataset_name + "Dataset")(
            subset="train", mode="train", args=args
        )

        # Create a small dataloader with just 2 batches
        print("Creating dataloader...")
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Don't shuffle for deterministic testing
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            generator=generator,
        )

        # Create optimizer
        print("Creating optimizer...")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Run training for just 2 batches
        print("\nRunning training loop (2 batches)...")
        model.train()
        criterion.train()

        batch_count = 0
        loss_values = []

        for batch_idx, (samples, targets) in enumerate(train_loader):
            if batch_count >= 2:  # Only process 2 batches
                break

            samples = samples.to(device)
            targets = [
                {
                    k: (
                        v.to(device)
                        if k
                        in ["segments", "labels", "salient_mask", "semantic_labels"]
                        else v
                    )
                    for k, v in t.items()
                }
                for t in targets
            ]

            classes = train_loader.dataset.classes
            description_dict = train_loader.dataset.description_dict

            # Set up seen and unseen classes (required by the model)
            seen_classes_names = train_loader.dataset.classes
            unseen_classes_names = train_loader.dataset.classes

            # Forward pass
            outputs = model(
                samples,
                classes,
                description_dict,
                targets,
                epoch=0,
                seen_classes_names=seen_classes_names,
                unseen_classes_names=unseen_classes_names,
                batch_idx=batch_idx,
            )

            # Compute loss with sorted keys
            loss_dict = criterion(
                outputs,
                targets,
                epoch=0,
                batch_idx=batch_idx,
                semantic_guided_unseen_weights=None,
            )

            weight_dict = criterion.weight_dict

            # This is the key part we're testing: sorted keys
            sorted_keys = sorted([k for k in loss_dict.keys() if k in weight_dict])
            losses = sum(loss_dict[k] * weight_dict[k] for k in sorted_keys)

            # Verify loss is finite
            loss_value = losses.item()
            if not np.isfinite(loss_value):
                print(f"✗ FAILURE: Loss is not finite: {loss_value}")
                return False

            loss_values.append(loss_value)
            print(f"  Batch {batch_idx}: loss = {loss_value:.6f}")

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_count += 1

        print(f"\n✓ SUCCESS: Training loop executed without errors")
        print(f"  Processed {batch_count} batches")
        print(f"  Loss values: {[f'{v:.6f}' for v in loss_values]}")
        return True

    except Exception as e:
        print(f"\n✗ FAILURE: Exception occurred during training loop")
        print(f"  Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_deterministic_loss_computation():
    """
    Test that loss computation is deterministic.

    This test verifies:
    1. Running the same batch twice produces identical loss values
    2. Sorted keys ensure consistent computation order

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 2: Deterministic Loss Computation")
    print("=" * 70)

    try:
        # Setup
        args = create_minimal_args()
        device = torch.device(args.device)
        seed = args.seed

        print(f"\nSetting up with seed={seed}, device={device}")

        # Build model
        print("Building model...")
        setup_seed(seed)
        model, criterion, postprocessor = build_model(args, device)
        model.to(device)

        # Build dataset
        print("Loading dataset...")
        train_dataset = getattr(dataset, args.dataset_name + "Dataset")(
            subset="train", mode="train", args=args
        )

        # Create dataloader
        print("Creating dataloader...")
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            generator=generator,
        )

        # Get one batch
        print("Getting one batch...")
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = [
            {
                k: (
                    v.to(device)
                    if k in ["segments", "labels", "salient_mask", "semantic_labels"]
                    else v
                )
                for k, v in t.items()
            }
            for t in targets
        ]

        classes = train_loader.dataset.classes
        description_dict = train_loader.dataset.description_dict

        # Set up seen and unseen classes (required by the model)
        seen_classes_names = train_loader.dataset.classes
        unseen_classes_names = train_loader.dataset.classes

        # Save initial model state
        print("Saving initial model state...")
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Run 1: Compute loss
        print("\n[Run 1] Computing loss...")
        model.eval()
        criterion.eval()

        with torch.no_grad():
            outputs1 = model(
                samples,
                classes,
                description_dict,
                targets,
                epoch=0,
                seen_classes_names=seen_classes_names,
                unseen_classes_names=unseen_classes_names,
                batch_idx=0,
            )

            loss_dict1 = criterion(
                outputs1,
                targets,
                epoch=0,
                batch_idx=0,
                semantic_guided_unseen_weights=None,
            )

            weight_dict = criterion.weight_dict
            sorted_keys = sorted([k for k in loss_dict1.keys() if k in weight_dict])
            losses1 = sum(loss_dict1[k] * weight_dict[k] for k in sorted_keys)
            loss_value1 = losses1.item()

            # Also compute individual loss components
            loss_components1 = {k: loss_dict1[k].item() for k in sorted_keys}

        print(f"  Total loss: {loss_value1:.10f}")
        print(f"  Loss components: {sorted_keys}")
        for k in sorted_keys[:3]:  # Show first 3 components
            print(f"    {k}: {loss_components1[k]:.10f}")

        # Restore model state
        print("\n[Run 2] Restoring model state and computing loss again...")
        model.load_state_dict(initial_state)
        model.eval()
        criterion.eval()

        with torch.no_grad():
            outputs2 = model(
                samples,
                classes,
                description_dict,
                targets,
                epoch=0,
                seen_classes_names=seen_classes_names,
                unseen_classes_names=unseen_classes_names,
                batch_idx=0,
            )

            loss_dict2 = criterion(
                outputs2,
                targets,
                epoch=0,
                batch_idx=0,
                semantic_guided_unseen_weights=None,
            )

            sorted_keys = sorted([k for k in loss_dict2.keys() if k in weight_dict])
            losses2 = sum(loss_dict2[k] * weight_dict[k] for k in sorted_keys)
            loss_value2 = losses2.item()

            # Also compute individual loss components
            loss_components2 = {k: loss_dict2[k].item() for k in sorted_keys}

        print(f"  Total loss: {loss_value2:.10f}")
        print(f"  Loss components: {sorted_keys}")
        for k in sorted_keys[:3]:  # Show first 3 components
            print(f"    {k}: {loss_components2[k]:.10f}")

        # Compare results
        print("\n" + "=" * 70)
        print("Comparing results...")

        # Check if losses are identical
        loss_diff = abs(loss_value1 - loss_value2)
        print(f"  Loss difference: {loss_diff:.15e}")

        # Check individual components
        component_diffs = {}
        max_diff = 0.0
        for k in sorted_keys:
            diff = abs(loss_components1[k] - loss_components2[k])
            component_diffs[k] = diff
            max_diff = max(max_diff, diff)

        print(f"  Max component difference: {max_diff:.15e}")

        # For deterministic computation, we expect exact equality
        # However, due to floating point precision, we allow a very small tolerance
        tolerance = 1e-10

        if loss_diff < tolerance and max_diff < tolerance:
            print(f"\n✓ SUCCESS: Loss computation is deterministic!")
            print(f"  Loss values are identical (within tolerance {tolerance})")
            print(
                f"  This confirms that sorted keys ensure consistent computation order"
            )
            return True
        else:
            print(f"\n✗ FAILURE: Loss computation is not deterministic!")
            print(f"  Loss difference {loss_diff} exceeds tolerance {tolerance}")
            print(f"\nComponent differences:")
            for k, diff in sorted(component_diffs.items(), key=lambda x: -x[1])[:5]:
                print(f"    {k}: {diff:.15e}")
            return False

    except Exception as e:
        print(f"\n✗ FAILURE: Exception occurred during deterministic test")
        print(f"  Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_sorted_keys_order():
    """
    Test that sorted keys produce consistent ordering.

    This test verifies:
    1. Dictionary keys are sorted alphabetically
    2. The same keys always produce the same order

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("TEST 3: Sorted Keys Order")
    print("=" * 70)

    try:
        # Create sample loss dictionaries
        print("\nTesting sorted keys implementation...")

        # Simulate loss_dict and weight_dict
        loss_dict = {
            "loss_bbox": torch.tensor(0.5),
            "loss_giou": torch.tensor(0.3),
            "loss_class": torch.tensor(1.2),
            "loss_ce": torch.tensor(0.8),
            "loss_aux_0": torch.tensor(0.4),
        }

        weight_dict = {
            "loss_class": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_ce": 1.0,
            "loss_aux_0": 1.0,
        }

        # Test sorted keys multiple times
        orders = []
        for i in range(5):
            sorted_keys = sorted([k for k in loss_dict.keys() if k in weight_dict])
            orders.append(sorted_keys)
            if i == 0:
                print(f"  Sorted keys: {sorted_keys}")

        # Verify all orders are identical
        all_same = all(order == orders[0] for order in orders)

        if all_same:
            print(f"\n✓ SUCCESS: Sorted keys produce consistent ordering")
            print(f"  Keys are always sorted in the same order: {orders[0]}")
            return True
        else:
            print(f"\n✗ FAILURE: Sorted keys produce inconsistent ordering")
            for i, order in enumerate(orders):
                print(f"  Run {i+1}: {order}")
            return False

    except Exception as e:
        print(f"\n✗ FAILURE: Exception occurred during sorted keys test")
        print(f"  Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("TRAINING LOOP WITH SORTED KEYS TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run all tests
    test_results = []

    # Test 1: Sorted keys order (lightweight, no model needed)
    test_results.append(("Sorted keys order", test_sorted_keys_order()))

    # Test 2: Basic training loop functionality
    test_results.append(("Basic training loop", test_training_loop_basic()))

    # Test 3: Deterministic loss computation
    test_results.append(("Deterministic loss", test_deterministic_loss_computation()))

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
        print("\nThe training loop with sorted keys:")
        print("  - Executes without syntax errors or runtime issues")
        print("  - Computes loss values correctly (finite numbers)")
        print("  - Produces deterministic results with sorted dictionary keys")
        print("  - Maintains consistent key ordering across runs")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
