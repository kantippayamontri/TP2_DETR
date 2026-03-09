#!/usr/bin/env python3
"""
Count the number of parameters in the TP2_DETR/GAP model.

Usage:
    python count_parameters.py --cfg_path config.yaml [--resume checkpoint.pkl]
"""

import argparse
import torch
import options
from models import build_model
from options import merge_cfg_from_file
from utils.util import setup_seed


def count_parameters(model, verbose=True):
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        verbose: If True, print detailed breakdown

    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    param_details = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params

        if verbose:
            param_details.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "num_params": num_params,
                    "trainable": param.requires_grad,
                }
            )

    results = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "param_details": param_details if verbose else None,
    }

    return results


def print_parameter_summary(results, show_details=False):
    """Print parameter count summary."""
    print("=" * 70)
    print("Model Parameter Count")
    print("=" * 70)
    print(
        f"Total parameters: {results['total_params']:,} ({results['total_params_M']:.2f}M)"
    )
    print(
        f"Trainable parameters: {results['trainable_params']:,} ({results['trainable_params_M']:.2f}M)"
    )
    print(f"Non-trainable parameters: {results['non_trainable_params']:,}")
    print("=" * 70)

    if show_details and results["param_details"]:
        print("\nDetailed Breakdown by Layer:")
        print("-" * 70)
        print(f"{'Layer Name':<50} {'Shape':<20} {'Parameters':>15}")
        print("-" * 70)

        for detail in results["param_details"]:
            shape_str = str(detail["shape"])
            trainable_str = "" if detail["trainable"] else " (frozen)"
            print(
                f"{detail['name']:<50} {shape_str:<20} {detail['num_params']:>15,}{trainable_str}"
            )

        print("-" * 70)


def count_parameters_by_module(model):
    """Count parameters grouped by module."""
    module_params = {}

    for name, param in model.named_parameters():
        # Get the top-level module name
        module_name = name.split(".")[0]

        if module_name not in module_params:
            module_params[module_name] = {
                "total": 0,
                "trainable": 0,
                "non_trainable": 0,
            }

        num_params = param.numel()
        module_params[module_name]["total"] += num_params

        if param.requires_grad:
            module_params[module_name]["trainable"] += num_params
        else:
            module_params[module_name]["non_trainable"] += num_params

    return module_params


def print_module_summary(module_params):
    """Print parameter count by module."""
    print("\nParameter Count by Module:")
    print("-" * 70)
    print(f"{'Module':<30} {'Total':>15} {'Trainable':>15} {'Frozen':>15}")
    print("-" * 70)

    # Sort by total parameters (descending)
    sorted_modules = sorted(
        module_params.items(), key=lambda x: x[1]["total"], reverse=True
    )

    for module_name, counts in sorted_modules:
        print(
            f"{module_name:<30} {counts['total']:>15,} {counts['trainable']:>15,} {counts['non_trainable']:>15,}"
        )

    print("-" * 70)

    # Calculate totals
    total = sum(m["total"] for m in module_params.values())
    trainable = sum(m["trainable"] for m in module_params.values())
    non_trainable = sum(m["non_trainable"] for m in module_params.values())

    print(f"{'TOTAL':<30} {total:>15,} {trainable:>15,} {non_trainable:>15,}")
    print(
        f"{'(in millions)':<30} {total/1e6:>15.2f} {trainable/1e6:>15.2f} {non_trainable/1e6:>15.2f}"
    )
    print("-" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        "Count parameters in TP2_DETR/GAP model",
        parents=[options.parser],
        add_help=False,
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Show detailed parameter breakdown by layer",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load checkpoint to verify parameter count (optional)",
    )

    args = parser.parse_args()

    # Merge config from YAML file
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args, args.cfg_path)

    # Setup
    device = torch.device("cpu")  # Use CPU for parameter counting
    seed = args.seed
    setup_seed(seed)

    print(f"\nBuilding model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Config: {args.cfg_path}")

    # Build model
    model, criterion, postprocessor = build_model(args, device)

    # Load checkpoint if provided
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded successfully")

    print("\nCounting parameters...")

    # Count parameters
    results = count_parameters(model, verbose=args.show_details)

    # Print summary
    print_parameter_summary(results, show_details=args.show_details)

    # Print module breakdown
    module_params = count_parameters_by_module(model)
    print_module_summary(module_params)

    # Additional statistics
    print("\nAdditional Statistics:")
    print("-" * 70)

    # Count parameters in different components
    backbone_params = sum(
        p.numel() for n, p in model.named_parameters() if "backbone" in n
    )
    transformer_params = sum(
        p.numel() for n, p in model.named_parameters() if "transformer" in n
    )
    text_encoder_params = sum(
        p.numel() for n, p in model.named_parameters() if "text_encoder" in n
    )

    print(f"Backbone parameters: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    print(
        f"Transformer parameters: {transformer_params:,} ({transformer_params/1e6:.2f}M)"
    )
    print(
        f"Text encoder parameters: {text_encoder_params:,} ({text_encoder_params/1e6:.2f}M)"
    )
    print("-" * 70)

    return results


if __name__ == "__main__":
    main()
