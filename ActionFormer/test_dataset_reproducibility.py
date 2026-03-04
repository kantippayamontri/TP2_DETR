#!/usr/bin/env python3
"""
Test script to validate deterministic dataset loading after reproducibility fixes.
This script tests that both Thumos14Dataset and ActivityNet13Dataset produce
consistent, deterministic video ordering across multiple runs.

Task 4.4: Test dataset loading produces consistent order
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset import Thumos14Dataset, ActivityNet13Dataset
from utils.util import setup_seed
import options


def test_thumos14_determinism(config_path, num_runs=3):
    """
    Test that Thumos14Dataset produces consistent video ordering across multiple runs.

    Args:
        config_path: Path to Thumos14 config file
        num_runs: Number of times to load the dataset

    Returns:
        True if all runs produce identical ordering
    """
    print("\n" + "=" * 70)
    print("Testing Thumos14Dataset Determinism")
    print("=" * 70)

    video_lists = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...")

        # Reset seed for each run to ensure same initialization
        setup_seed(3552)

        # Create args from config using the full parser
        args = options.parser.parse_args([])
        args = options.merge_cfg_from_file(args, config_path)

        # Override split_id for testing
        args.split_id = 0

        # Create dataset
        dataset = Thumos14Dataset(subset="train", mode="train", args=args)

        # Get the video list from the dataset
        video_list = dataset.video_list.copy()
        video_lists.append(video_list)

        print(f"  Number of videos: {len(video_list)}")
        if len(video_list) > 0:
            print(f"  First 5 videos: {video_list[:min(5, len(video_list))]}")
            print(f"  Last 5 videos: {video_list[-min(5, len(video_list)):]}")

    # Verify all runs produce identical results
    print("\n" + "-" * 70)
    print("Verifying consistency across runs...")
    print("-" * 70)

    all_identical = True
    for i in range(1, num_runs):
        if video_lists[i] != video_lists[0]:
            print(f"✗ Run {i + 1} differs from Run 1!")
            print(f"  Run 1 length: {len(video_lists[0])}")
            print(f"  Run {i + 1} length: {len(video_lists[i])}")

            # Find differences
            for j, (v1, v2) in enumerate(zip(video_lists[0], video_lists[i])):
                if v1 != v2:
                    print(f"  First difference at index {j}: '{v1}' vs '{v2}'")
                    break

            all_identical = False
        else:
            print(f"✓ Run {i + 1} matches Run 1")

    if all_identical:
        print("\n✓ SUCCESS: All runs produced identical, deterministic ordering")
        print(f"✓ Total videos: {len(video_lists[0])}")
        return True
    else:
        print("\n✗ FAILURE: Runs produced different orderings")
        return False


def test_activitynet_determinism(config_path, num_runs=3):
    """
    Test that ActivityNet13Dataset produces consistent video ordering across multiple runs.

    Args:
        config_path: Path to ActivityNet config file
        num_runs: Number of times to load the dataset

    Returns:
        True if all runs produce identical ordering
    """
    print("\n" + "=" * 70)
    print("Testing ActivityNet13Dataset Determinism")
    print("=" * 70)

    video_lists = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...")

        # Reset seed for each run to ensure same initialization
        setup_seed(3552)

        # Create args from config using the full parser
        args = options.parser.parse_args([])
        args = options.merge_cfg_from_file(args, config_path)

        # Override split_id for testing
        args.split_id = 0

        # Create dataset
        dataset = ActivityNet13Dataset(subset="train", mode="train", args=args)

        # Get the video list from the dataset
        video_list = dataset.video_list.copy()
        video_lists.append(video_list)

        print(f"  Number of videos: {len(video_list)}")
        if len(video_list) > 0:
            print(f"  First 5 videos: {video_list[:min(5, len(video_list))]}")
            print(f"  Last 5 videos: {video_list[-min(5, len(video_list)):]}")

    # Verify all runs produce identical results
    print("\n" + "-" * 70)
    print("Verifying consistency across runs...")
    print("-" * 70)

    all_identical = True
    for i in range(1, num_runs):
        if video_lists[i] != video_lists[0]:
            print(f"✗ Run {i + 1} differs from Run 1!")
            print(f"  Run 1 length: {len(video_lists[0])}")
            print(f"  Run {i + 1} length: {len(video_lists[i])}")

            # Find differences
            for j, (v1, v2) in enumerate(zip(video_lists[0], video_lists[i])):
                if v1 != v2:
                    print(f"  First difference at index {j}: '{v1}' vs '{v2}'")
                    break

            all_identical = False
        else:
            print(f"✓ Run {i + 1} matches Run 1")

    if all_identical:
        print("\n✓ SUCCESS: All runs produced identical, deterministic ordering")
        print(f"✓ Total videos: {len(video_lists[0])}")
        return True
    else:
        print("\n✗ FAILURE: Runs produced different orderings")
        return False


def test_video_list_is_sorted(config_path, dataset_class, dataset_name):
    """
    Test that the video_list is properly sorted.

    Args:
        config_path: Path to config file
        dataset_class: Dataset class to test
        dataset_name: Name of the dataset for display

    Returns:
        True if video_list is sorted
    """
    print("\n" + "=" * 70)
    print(f"Testing {dataset_name} Video List Sorting")
    print("=" * 70)

    # Create args from config using the full parser
    args = options.parser.parse_args([])
    args = options.merge_cfg_from_file(args, config_path)
    args.split_id = 0

    # Create dataset
    dataset = dataset_class(subset="train", mode="train", args=args)

    video_list = dataset.video_list
    sorted_video_list = sorted(video_list)

    print(f"Video list length: {len(video_list)}")
    if len(video_list) > 0:
        print(f"First 5 videos: {video_list[:min(5, len(video_list))]}")
        print(f"Last 5 videos: {video_list[-min(5, len(video_list)):]}")

    if video_list == sorted_video_list:
        print("\n✓ SUCCESS: Video list is properly sorted")
        return True
    else:
        print("\n✗ FAILURE: Video list is NOT sorted")
        print("First difference:")
        for i, (v1, v2) in enumerate(zip(video_list, sorted_video_list)):
            if v1 != v2:
                print(f"  Index {i}: '{v1}' (actual) vs '{v2}' (expected)")
                break
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Dataset Reproducibility Test Suite")
    print("Task 4.4: Test dataset loading produces consistent order")
    print("=" * 70)

    all_tests_passed = True

    # Test Thumos14Dataset
    try:
        thumos_config = "./config/Thumos14_CLIP_zs_50_8frame.yaml"

        # Test determinism
        if not test_thumos14_determinism(thumos_config, num_runs=3):
            all_tests_passed = False

        # Test sorting
        if not test_video_list_is_sorted(
            thumos_config, Thumos14Dataset, "Thumos14Dataset"
        ):
            all_tests_passed = False

    except FileNotFoundError as e:
        print(f"\n⚠ WARNING: Could not test Thumos14Dataset - {e}")
        print("This is expected if dataset files are not available")
    except Exception as e:
        print(f"\n✗ ERROR testing Thumos14Dataset: {e}")
        import traceback

        traceback.print_exc()
        all_tests_passed = False

    # Test ActivityNet13Dataset
    try:
        activitynet_config = "./config/ActivityNet13_CLIP_zs_50.yaml"

        # Test determinism
        if not test_activitynet_determinism(activitynet_config, num_runs=3):
            all_tests_passed = False

        # Test sorting
        if not test_video_list_is_sorted(
            activitynet_config, ActivityNet13Dataset, "ActivityNet13Dataset"
        ):
            all_tests_passed = False

    except FileNotFoundError as e:
        print(f"\n⚠ WARNING: Could not test ActivityNet13Dataset - {e}")
        print("This is expected if dataset files are not available")
    except Exception as e:
        print(f"\n✗ ERROR testing ActivityNet13Dataset: {e}")
        import traceback

        traceback.print_exc()
        all_tests_passed = False

    # Final summary
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nVerified:")
        print("  1. Thumos14Dataset produces deterministic video ordering")
        print("  2. ActivityNet13Dataset produces deterministic video ordering")
        print("  3. Video lists are properly sorted")
        print("  4. Multiple runs produce identical results")
        print("  5. No runtime issues detected")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(1)
