#!/usr/bin/env python3
"""
Test script for DataLoader worker seeding with multiple workers.

This script validates that the seed_worker function and generator properly
ensure reproducible data loading across multiple workers.

**Validates: Requirements TR-5 (Worker Reproducibility)**
"""

import sys
import os

# Add ActionFormer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ActionFormer"))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from main import seed_worker, build_dataloader


class SimpleDataset(Dataset):
    """
    Simple dataset that generates random data for testing.
    This allows us to verify that worker seeding produces reproducible results.
    """

    def __init__(self, size=100, seed=None):
        """
        Args:
            size (int): Number of samples in the dataset
            seed (int): Optional seed for dataset initialization
        """
        self.size = size
        # Store a base array that will be shuffled/transformed by workers
        if seed is not None:
            np.random.seed(seed)
        self.data = np.arange(size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Return data with some random transformation to test worker RNG.
        """
        # Get base value
        value = self.data[idx]

        # Add some random noise (this will be affected by worker seed)
        noise = np.random.randn()
        torch_noise = torch.randn(1).item()

        return {
            "idx": idx,
            "value": value,
            "np_noise": noise,
            "torch_noise": torch_noise,
        }


def load_data_batch(dataloader, num_batches=10):
    """
    Load a specified number of batches from the dataloader.

    Args:
        dataloader: DataLoader instance
        num_batches (int): Number of batches to load

    Returns:
        list: List of batch data
    """
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batches.append(batch)
    return batches


def compare_batches(batches1, batches2):
    """
    Compare two sets of batches for equality.

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
        # Compare each field in the batch
        for key in batch1.keys():
            if key not in batch2:
                mismatches.append(
                    f"Batch {batch_idx}: Key '{key}' missing in second batch"
                )
                continue

            val1 = batch1[key]
            val2 = batch2[key]

            # Convert to numpy for comparison
            if isinstance(val1, torch.Tensor):
                val1 = val1.numpy()
            if isinstance(val2, torch.Tensor):
                val2 = val2.numpy()

            # Compare with tolerance for floating point
            if not np.allclose(val1, val2, rtol=1e-7, atol=1e-7):
                mismatches.append(
                    f"Batch {batch_idx}, key '{key}': Values differ\n"
                    f"  First:  {val1[:3] if len(val1) > 3 else val1}\n"
                    f"  Second: {val2[:3] if len(val2) > 3 else val2}"
                )

    return len(mismatches) == 0, mismatches


def test_worker_seeding_reproducibility(num_workers=4, seed=3552):
    """
    Test that DataLoader with multiple workers produces reproducible results.

    Args:
        num_workers (int): Number of workers to use
        seed (int): Seed value for reproducibility

    Returns:
        bool: True if test passes, False otherwise
    """
    print(
        f"\nTesting worker seeding reproducibility with {num_workers} workers, seed={seed}"
    )
    print("=" * 70)

    # Create dataset
    dataset = SimpleDataset(size=100, seed=seed)

    # First run: create dataloader and load data
    print(f"\n[Run 1] Creating DataLoader with {num_workers} workers...")
    dataloader1 = build_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )

    print("Loading batches from first run...")
    batches1 = load_data_batch(dataloader1, num_batches=10)
    print(f"Loaded {len(batches1)} batches")

    # Show sample from first batch
    if batches1:
        print(f"\nFirst batch sample (indices): {batches1[0]['idx'][:5].tolist()}")
        print(f"First batch sample (np_noise): {batches1[0]['np_noise'][:3].tolist()}")

    # Second run: create new dataloader with same seed and load data
    print(f"\n[Run 2] Creating new DataLoader with same seed...")
    dataset2 = SimpleDataset(size=100, seed=seed)
    dataloader2 = build_dataloader(
        dataset2,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )

    print("Loading batches from second run...")
    batches2 = load_data_batch(dataloader2, num_batches=10)
    print(f"Loaded {len(batches2)} batches")

    # Show sample from first batch
    if batches2:
        print(f"\nFirst batch sample (indices): {batches2[0]['idx'][:5].tolist()}")
        print(f"First batch sample (np_noise): {batches2[0]['np_noise'][:3].tolist()}")

    # Compare results
    print("\n" + "=" * 70)
    print("Comparing batches...")
    all_match, mismatches = compare_batches(batches1, batches2)

    if all_match:
        print("✓ SUCCESS: All batches are identical across runs!")
        print(f"  Worker seeding with {num_workers} workers is reproducible.")
        return True
    else:
        print("✗ FAILURE: Batches differ between runs!")
        print(f"\nFound {len(mismatches)} mismatch(es):")
        for i, mismatch in enumerate(mismatches[:5]):  # Show first 5
            print(f"  {i+1}. {mismatch}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
        return False


def test_different_worker_counts():
    """
    Test reproducibility with different numbers of workers.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "=" * 70)
    print("Testing reproducibility with different worker counts...")
    print("=" * 70)

    worker_counts = [1, 2, 4]
    seed = 3552
    all_passed = True

    for num_workers in worker_counts:
        passed = test_worker_seeding_reproducibility(num_workers=num_workers, seed=seed)
        if not passed:
            all_passed = False

    return all_passed


def test_single_worker_vs_multi_worker():
    """
    Test that single worker and multi-worker produce different but reproducible results.

    Returns:
        bool: True if test passes, False otherwise
    """
    print("\n" + "=" * 70)
    print("Testing single worker vs multi-worker behavior...")
    print("=" * 70)

    seed = 3552
    dataset = SimpleDataset(size=100, seed=seed)

    # Single worker
    print("\nLoading with 1 worker...")
    dataloader_single = build_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        seed=seed,
    )
    batches_single = load_data_batch(dataloader_single, num_batches=5)

    # Multiple workers
    print("Loading with 4 workers...")
    dataset2 = SimpleDataset(size=100, seed=seed)
    dataloader_multi = build_dataloader(
        dataset2,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        seed=seed,
    )
    batches_multi = load_data_batch(dataloader_multi, num_batches=5)

    # Compare - they should have same indices (shuffle order) but may have different noise
    print("\nComparing shuffle order (indices)...")
    indices_match = True
    for i, (b1, b2) in enumerate(zip(batches_single, batches_multi)):
        if not torch.equal(b1["idx"], b2["idx"]):
            indices_match = False
            print(f"  Batch {i}: Indices differ")
            print(f"    Single: {b1['idx'][:5].tolist()}")
            print(f"    Multi:  {b2['idx'][:5].tolist()}")

    if indices_match:
        print("✓ Shuffle order is consistent between single and multi-worker")
    else:
        print("✗ Shuffle order differs between single and multi-worker")

    return indices_match


def main():
    """
    Main test function.
    """
    print("=" * 70)
    print("DATALOADER WORKER SEEDING TEST")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Run all tests
    test_results = []

    # Test 1: Reproducibility with different worker counts
    test_results.append(
        ("Multi-worker reproducibility", test_different_worker_counts())
    )

    # Test 2: Single vs multi-worker consistency
    test_results.append(
        ("Single vs multi-worker", test_single_worker_vs_multi_worker())
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
        print("\nThe worker seeding implementation correctly:")
        print("  - Produces reproducible results with multiple workers")
        print("  - Works consistently across different worker counts")
        print("  - Maintains consistent shuffle order across configurations")
        print("\n**Validates: Requirements TR-5 (Worker Reproducibility)**")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
