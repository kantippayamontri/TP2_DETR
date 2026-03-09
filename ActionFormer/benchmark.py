#!/usr/bin/env python3
"""
Benchmark inference speed of TP2_DETR ActionFormer model.

This script measures:
- Inference time per video
- FPS (videos per second)
- Average latency
- Throughput

Usage:
    python benchmark.py --cfg_path config.yaml --resume checkpoint.pth [options]
    python benchmark.py --cfg_path config.yaml --resume checkpoint.pkl [options]

Checkpoint formats supported:
    - .pth: New format with model_state_dict, optimizer_state_dict, etc.
    - .pkl: Legacy format with direct state dict
"""

import argparse
import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import options
from models import build_model
from options import merge_cfg_from_file
from utils.misc import collate_fn
from utils.util import setup_seed


logger = logging.getLogger(__name__)


def get_benchmark_arg_parser():
    """Parse benchmark-specific arguments."""
    parser = argparse.ArgumentParser(
        "Benchmark inference speed of TP2_DETR ActionFormer.",
        parents=[options.parser],
        add_help=False,
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=100,
        help="Number of videos to benchmark (default: 100)",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=5,
        help="Number of warm-up iterations to ignore (default: 5)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth or .pkl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save benchmark results (optional)",
    )
    return parser


def seed_worker(worker_id):
    """Set different deterministic seed for each worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    collate_fn=None,
    seed=42,
):
    """Build a reproducible DataLoader with deterministic seed handling."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def benchmark_inference(model, data_loader, device, num_videos, warm_up, args):
    """
    Benchmark inference speed of the model.

    Args:
        model: The model to benchmark
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        num_videos: Number of videos to process
        warm_up: Number of warm-up iterations
        args: Additional arguments

    Returns:
        dict: Benchmark statistics
    """
    model.eval()

    # Statistics
    inference_times = []
    video_count = 0
    total_frames = 0

    print("=" * 70)
    print("Starting Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Number of videos: {num_videos}")
    print(f"Warm-up iterations: {warm_up}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # Warm-up phase
    if warm_up > 0:
        print(f"\nWarm-up phase ({warm_up} iterations)...")
        warm_up_count = 0
        for samples, targets in data_loader:
            if warm_up_count >= warm_up:
                break

            samples = samples.to(device)
            classes = data_loader.dataset.classes
            description_dict = data_loader.dataset.description_dict

            # Run inference (warm-up)
            _ = model(samples, classes, description_dict, targets, epoch=0)

            warm_up_count += len(targets)
            print(f"  Warm-up: {warm_up_count}/{warm_up}", end="\r")

        print(f"\n✓ Warm-up complete")

    # Benchmark phase
    print(f"\nBenchmark phase...")
    print("-" * 70)

    for batch_idx, (samples, targets) in enumerate(data_loader):
        if video_count >= num_videos:
            break

        batch_size = len(targets)
        samples = samples.to(device)
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        # Synchronize CUDA before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure inference time
        start_time = time.perf_counter()

        outputs = model(samples, classes, description_dict, targets, epoch=0)

        # Synchronize CUDA after inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Record timing
        batch_time = end_time - start_time
        inference_times.append(batch_time)

        # Update statistics
        video_count += batch_size
        total_frames += samples.shape[0] if len(samples.shape) > 3 else batch_size

        # Print progress
        avg_time = np.mean(inference_times)
        fps = batch_size / batch_time
        print(
            f"Batch {batch_idx + 1}: {batch_time:.4f}s ({fps:.2f} videos/s) | "
            f"Avg: {avg_time:.4f}s | Progress: {video_count}/{num_videos}",
            end="\r",
        )

    print()  # New line after progress

    # Calculate statistics
    inference_times = np.array(inference_times)

    stats = {
        "num_videos": video_count,
        "total_time": np.sum(inference_times),
        "mean_time": np.mean(inference_times),
        "std_time": np.std(inference_times),
        "min_time": np.min(inference_times),
        "max_time": np.max(inference_times),
        "median_time": np.median(inference_times),
        "fps": video_count / np.sum(inference_times),
        "latency_ms": np.mean(inference_times) * 1000,
        "throughput": video_count / np.sum(inference_times),
    }

    return stats


def print_benchmark_results(stats, args):
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Device: {args.device}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Number of videos: {stats['num_videos']}")
    print()
    print(f"Timing Statistics:")
    print(f"  - Total time: {stats['total_time']:.4f}s")
    print(f"  - Mean time per batch: {stats['mean_time']:.4f}s")
    print(f"  - Std time per batch: {stats['std_time']:.4f}s")
    print(f"  - Min time per batch: {stats['min_time']:.4f}s")
    print(f"  - Max time per batch: {stats['max_time']:.4f}s")
    print(f"  - Median time per batch: {stats['median_time']:.4f}s")
    print()
    print(f"Performance Metrics:")
    print(f"  - FPS (videos/second): {stats['fps']:.2f}")
    print(f"  - Average latency: {stats['latency_ms']:.2f}ms")
    print(f"  - Throughput: {stats['throughput']:.2f} videos/s")
    print("=" * 70)


def save_benchmark_results(stats, args, output_file):
    """Save benchmark results to a file."""
    import json

    results = {
        "configuration": {
            "dataset": args.dataset_name,
            "model": args.model_name,
            "device": args.device,
            "batch_size": args.batch_size,
            "checkpoint": args.resume,
        },
        "statistics": stats,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main benchmark function."""
    # Parse arguments
    parser = get_benchmark_arg_parser()
    args = parser.parse_args()

    # Merge config from YAML file
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args, args.cfg_path)

    # Setup
    device = torch.device(args.device)
    seed = args.seed
    setup_seed(seed)

    # Check checkpoint exists
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    print(f"\nLoading dataset: {args.dataset_name}...")
    val_dataset = getattr(dataset, args.dataset_name + "Dataset")(
        subset="inference", mode="inference", args=args
    )

    # Limit dataset size if needed
    if args.num_videos < len(val_dataset):
        print(f"  Using {args.num_videos} out of {len(val_dataset)} videos")

    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        seed=seed,
    )

    print(f"\nBuilding model: {args.model_name}...")
    model, criterion, postprocessor = build_model(args, device)

    print(f"\nLoading checkpoint: {args.resume}...")
    checkpoint = torch.load(args.resume, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New format (.pth): dictionary with model_state_dict, optimizer_state_dict, etc.
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if "best_stats" in checkpoint:
            best_mAP = checkpoint["best_stats"].get("mAP_raw", 0.0) * 100
            print(f"  Best mAP: {best_mAP:.2f}%")
    else:
        # Old format (.pkl): direct state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    # Run benchmark
    stats = benchmark_inference(
        model=model,
        data_loader=val_loader,
        device=device,
        num_videos=args.num_videos,
        warm_up=args.warm_up,
        args=args,
    )

    # Print results
    print_benchmark_results(stats, args)

    # Save results if output file specified
    if args.output_file:
        save_benchmark_results(stats, args, args.output_file)

    return stats


if __name__ == "__main__":
    main()
