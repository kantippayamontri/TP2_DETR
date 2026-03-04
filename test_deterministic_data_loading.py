#!/usr/bin/env python3
"""
Test script for deterministic data loading.

This script validates that the DataLoader yields batches in the same order
across runs when given the same seed, verifying the sorted video_list
implementation in dataset.py.

**Validates: Property 2 (Deterministic Data Loading)**
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import torch
import numpy as np
from torch.utils.data import DataLoader
from main import seed_worker, build_dataloader
from dataset import Thumos14Dataset
from options import parser
from utils.misc import collate_fn


def load_batches(dataloader, num_batches=5):
    """
    Load a specified number of batches from the dataloader.

    Args:
        dataloader: DataLoader instance
        num_batches (int): Number of batches to load

    Returns:
        list: List of batch data (video names and features)
    """
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        # batch is a tuple: (features, targets)
        features, targets = batch
        # Store video names and feature shapes for comparison
        batch_info = {
            "video_names": [t["video_name"] for t in targets],
            "feature_shape": (
                features.tensors.shape
                if hasattr(features, "tensors")
                else features.shape
            ),
        }
        batches.append(batch_info)
    return batches


def compare_batch_order(batches1, batches2):
    """
    Compare two sets of batches to verify they have the same order.

    Args:
        batches1 (list): First set of batches
        batches2 (list): Second set of batches

    Returns:
        tuple: (bool, list) - (all_match, list of mismatches)
    """
    if len(batches1) != len(batches2):
        return False, [
            f"Different number of batches: {len(batches1)} vs {len(batches2)}"
        ]

    mismatches = []

    for batch_idx, (batch1, batch2) in enumerate(zip(batches1, batches2)):
        # Compare video names
        names1 = batch1["video_names"]
        names2 = batch2["video_names"]

        if names1 != names2:
            mismatches.append(
                f"Batch {batch_idx}: Video names differ\n"
                f"  Run 1: {names1}\n"
                f"  Run 2: {names2}"
            )

        # Compare feature shapes
        shape1 = batch1["feature_shape"]
        shape2 = batch2["feature_shape"]

        if shape1 != shape2:
            mismatches.append(
                f"Batch {batch_idx}: Feature shapes differ\n"
                f"  Run 1: {shape1}\n"
                f"  Run 2: {shape2}"
            )

    return len(mismatches) == 0, mismatches


def test_deterministic_loading_with_shuffle(seed=3552, num_workers=2):
    """
    Test that DataLoader produces the same batch order across runs with shuffle=True.

    This validates that the generator-based shuffling is deterministic.

    Args:
        seed (int): Seed value for reproducibility
        num_workers (int): Number of workers to use

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest 1: Deterministic loading with shuffle=True")
    print("=" * 70)
    print(f"Seed: {seed}, Workers: {num_workers}")

    # Parse args for dataset
    sys.argv = [
        "test",
        "--dataset_name",
        "Thumos14",
        "--task",
        "close_set",
        "--feature_type",
        "CLIP",
        "--feature_path",
        "../Thumos14/CLIP_feature_8frame/",
        "--anno_file_path",
        "../GAP/data/Thumos14/Thumos14_annotations.json",
        "--feature_info_path",
        "../Thumos14/CLIP_feature_8frame/Thumos14_info.json",
        "--description_file_path",
        "../GAP/data/Thumos14/Thumos14_description_v3.json",
    ]
    args = parser.parse_args()

    try:
        # First run: create dataset and dataloader
        print("\n[Run 1] Creating dataset and DataLoader...")
        dataset1 = Thumos14Dataset(subset="train", mode="train", args=args)
        print(f"Dataset size: {len(dataset1)}")

        dataloader1 = build_dataloader(
            dataset1,
            batch_size=4,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            collate_fn=collate_fn,
        )

        print("Loading batches from first run...")
        batches1 = load_batches(dataloader1, num_batches=5)
        print(f"Loaded {len(batches1)} batches")

        if batches1:
            print(f"First batch video names: {batches1[0]['video_names']}")

        # Second run: create new dataset and dataloader with same seed
        print("\n[Run 2] Creating new dataset and DataLoader with same seed...")
        dataset2 = Thumos14Dataset(subset="train", mode="train", args=args)
        dataloader2 = build_dataloader(
            dataset2,
            batch_size=4,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            collate_fn=collate_fn,
        )

        print("Loading batches from second run...")
        batches2 = load_batches(dataloader2, num_batches=5)
        print(f"Loaded {len(batches2)} batches")

        if batches2:
            print(f"First batch video names: {batches2[0]['video_names']}")

        # Compare results
        print("\n" + "=" * 70)
        print("Comparing batch order...")
        all_match, mismatches = compare_batch_order(batches1, batches2)

        if all_match:
            print("✓ SUCCESS: Batch order is identical across runs!")
            print("  DataLoader with shuffle=True is deterministic.")
            return True
        else:
            print("✗ FAILURE: Batch order differs between runs!")
            print(f"\nFound {len(mismatches)} mismatch(es):")
            for i, mismatch in enumerate(mismatches[:3]):
                print(f"  {i+1}. {mismatch}")
            if len(mismatches) > 3:
                print(f"  ... and {len(mismatches) - 3} more")
            return False

    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_deterministic_loading_without_shuffle(seed=3552, num_workers=2):
    """
    Test that DataLoader produces the same batch order across runs with shuffle=False.

    This validates that the sorted video_list implementation ensures deterministic order.

    Args:
        seed (int): Seed value for reproducibility
        num_workers (int): Number of workers to use

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest 2: Deterministic loading with shuffle=False")
    print("=" * 70)
    print(f"Seed: {seed}, Workers: {num_workers}")

    # Parse args for dataset
    sys.argv = [
        "test",
        "--dataset_name",
        "Thumos14",
        "--task",
        "close_set",
        "--feature_type",
        "CLIP",
        "--feature_path",
        "../Thumos14/CLIP_feature_8frame/",
        "--anno_file_path",
        "../GAP/data/Thumos14/Thumos14_annotations.json",
        "--feature_info_path",
        "../Thumos14/CLIP_feature_8frame/Thumos14_info.json",
        "--description_file_path",
        "../GAP/data/Thumos14/Thumos14_description_v3.json",
    ]
    args = parser.parse_args()

    try:
        # First run: create dataset and dataloader
        print("\n[Run 1] Creating dataset and DataLoader...")
        dataset1 = Thumos14Dataset(subset="train", mode="train", args=args)
        print(f"Dataset size: {len(dataset1)}")

        dataloader1 = build_dataloader(
            dataset1,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            collate_fn=collate_fn,
        )

        print("Loading batches from first run...")
        batches1 = load_batches(dataloader1, num_batches=5)
        print(f"Loaded {len(batches1)} batches")

        if batches1:
            print(f"First batch video names: {batches1[0]['video_names']}")

        # Second run: create new dataset and dataloader with same seed
        print("\n[Run 2] Creating new dataset and DataLoader with same seed...")
        dataset2 = Thumos14Dataset(subset="train", mode="train", args=args)
        dataloader2 = build_dataloader(
            dataset2,
            batch_size=4,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            collate_fn=collate_fn,
        )

        print("Loading batches from second run...")
        batches2 = load_batches(dataloader2, num_batches=5)
        print(f"Loaded {len(batches2)} batches")

        if batches2:
            print(f"First batch video names: {batches2[0]['video_names']}")

        # Compare results
        print("\n" + "=" * 70)
        print("Comparing batch order...")
        all_match, mismatches = compare_batch_order(batches1, batches2)

        if all_match:
            print("✓ SUCCESS: Batch order is identical across runs!")
            print("  Sorted video_list ensures deterministic order.")
            return True
        else:
            print("✗ FAILURE: Batch order differs between runs!")
            print(f"\nFound {len(mismatches)} mismatch(es):")
            for i, mismatch in enumerate(mismatches[:3]):
                print(f"  {i+1}. {mismatch}")
            if len(mismatches) > 3:
                print(f"  ... and {len(mismatches) - 3} more")
            return False

    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sorted_video_list():
    """
    Test that the video_list in the dataset is sorted.

    This directly validates the sorted video_list implementation.

    Returns:
        bool: True if test passes, False otherwise
    """
    print(f"\nTest 3: Verify sorted video_list implementation")
    print("=" * 70)

    # Parse args for dataset
    sys.argv = [
        "test",
        "--dataset_name",
        "Thumos14",
        "--task",
        "close_set",
        "--feature_type",
        "CLIP",
        "--feature_path",
        "../Thumos14/CLIP_feature_8frame/",
        "--anno_file_path",
        "../GAP/data/Thumos14/Thumos14_annotations.json",
        "--feature_info_path",
        "../Thumos14/CLIP_feature_8frame/Thumos14_info.json",
        "--description_file_path",
        "../GAP/data/Thumos14/Thumos14_description_v3.json",
    ]
    args = parser.parse_args()

    try:
        print("\nCreating dataset...")
        dataset = Thumos14Dataset(subset="train", mode="train", args=args)
        video_list = dataset.valid_video_list

        print(f"Dataset size: {len(video_list)}")
        print(f"First 5 videos: {video_list[:5]}")
        print(f"Last 5 videos: {video_list[-5:]}")

        # Check if the list is sorted
        sorted_list = sorted(video_list)
        is_sorted = video_list == sorted_list

        if is_sorted:
            print("\n✓ SUCCESS: video_list is sorted!")
            print("  The sorted() implementation ensures deterministic order.")
            return True
        else:
            print("\n✗ FAILURE: video_list is NOT sorted!")
            # Find first mismatch
            for i, (actual, expected) in enumerate(zip(video_list, sorted_list)):
                if actual != expected:
                    print(f"  First mismatch at index {i}:")
                    print(f"    Actual:   {actual}")
                    print(f"    Expected: {expected}")
                    break
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
    print("DETERMINISTIC DATA LOADING TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Run all tests
    test_results = []

    # Test 1: Sorted video_list
    test_results.append(("Sorted video_list", test_sorted_video_list()))

    # Test 2: Deterministic loading without shuffle
    test_results.append(
        (
            "Deterministic loading (shuffle=False)",
            test_deterministic_loading_without_shuffle(),
        )
    )

    # Test 3: Deterministic loading with shuffle
    test_results.append(
        (
            "Deterministic loading (shuffle=True)",
            test_deterministic_loading_with_shuffle(),
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
        print("\nThe deterministic data loading implementation correctly:")
        print("  - Uses sorted video_list for deterministic order")
        print("  - Produces identical batch order across runs (shuffle=False)")
        print("  - Produces identical batch order with generator (shuffle=True)")
        print("\n**Validates: Property 2 (Deterministic Data Loading)**")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
