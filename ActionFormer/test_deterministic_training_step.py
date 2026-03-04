#!/usr/bin/env python3
"""
Test script to verify deterministic training step.

This script tests Property 4 from the reproducibility-fix design document:
"Given the same model state, optimizer state, and batch, a training step
must produce identical results."

**Validates: Requirements 1.2**

The test:
1. Creates a simple model and optimizer
2. Saves initial model and optimizer state
3. Performs a training step (forward, loss, backward, optimizer step)
4. Saves final model state
5. Restores initial state and repeats training step
6. Compares final states for exact equality

Part of Task 7.4 from reproducibility-fix spec.
"""

import os
import sys
import torch
import torch.nn as nn
import copy

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.util import setup_seed


class SimpleModel(nn.Module):
    """Simple neural network for testing training step determinism."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_one_step(model, optimizer, batch_data, batch_target, criterion):
    """
    Perform a single training step.

    Args:
        model: The neural network model
        optimizer: The optimizer
        batch_data: Input batch
        batch_target: Target batch
        criterion: Loss function

    Returns:
        loss: The computed loss value
    """
    optimizer.zero_grad()
    output = model(batch_data)
    loss = criterion(output, batch_target)
    loss.backward()
    optimizer.step()
    return loss


def test_training_step_determinism():
    """Test that training steps are deterministic."""

    print("=" * 70)
    print("Testing Deterministic Training Step (Property 4)")
    print("=" * 70)

    # Setup
    seed = 3552
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Test 1: Setup environment
    print("\n[Test 1] Setting up deterministic environment...")
    setup_seed(seed)

    # Set CUBLAS_WORKSPACE_CONFIG if not already set
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enable deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
        print("✓ Deterministic algorithms enabled")
    except RuntimeError:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✓ Deterministic algorithms enabled (with warn_only=True)")

    # Test 2: Create model and optimizer
    print("\n[Test 2] Creating model and optimizer...")
    setup_seed(seed)
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print(
        f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Test 3: Create a fixed batch
    print("\n[Test 3] Creating fixed batch data...")
    setup_seed(seed)
    batch_size = 4
    batch_data = torch.randn(batch_size, 10, device=device)
    batch_target = torch.randn(batch_size, 5, device=device)
    print(
        f"✓ Batch created: data shape {batch_data.shape}, target shape {batch_target.shape}"
    )

    # Test 4: First training step
    print("\n[Test 4] Performing first training step...")
    # Save initial state
    model_state_initial = copy.deepcopy(model.state_dict())
    optimizer_state_initial = copy.deepcopy(optimizer.state_dict())

    # Run training step
    loss1 = train_one_step(model, optimizer, batch_data, batch_target, criterion)
    final_state1 = copy.deepcopy(model.state_dict())

    print(f"✓ First training step completed, loss: {loss1.item():.8f}")

    # Test 5: Second training step (restore and repeat)
    print("\n[Test 5] Restoring state and performing second training step...")
    # Restore initial state
    model.load_state_dict(model_state_initial)
    optimizer.load_state_dict(optimizer_state_initial)

    # Run training step again
    loss2 = train_one_step(model, optimizer, batch_data, batch_target, criterion)
    final_state2 = copy.deepcopy(model.state_dict())

    print(f"✓ Second training step completed, loss: {loss2.item():.8f}")

    # Test 6: Compare losses
    print("\n[Test 6] Comparing losses...")
    print(f"  Loss 1: {loss1.item():.10f}")
    print(f"  Loss 2: {loss2.item():.10f}")
    print(f"  Difference: {abs(loss1.item() - loss2.item()):.2e}")

    loss_match = torch.equal(loss1, loss2)
    if loss_match:
        print("✓ Losses are identical (exact match)")
    else:
        # Check if they're very close (floating point tolerance)
        loss_close = torch.allclose(loss1, loss2, rtol=0, atol=0)
        if loss_close:
            print("✓ Losses are identical (within tolerance)")
        else:
            print("✗ Losses are different!")
            raise AssertionError(
                f"Training step loss is non-deterministic: {loss1.item()} != {loss2.item()}"
            )

    # Test 7: Compare model parameters
    print("\n[Test 7] Comparing final model parameters...")
    all_params_match = True
    max_diff = 0.0
    mismatched_params = []

    for key in final_state1.keys():
        param1 = final_state1[key]
        param2 = final_state2[key]

        if not torch.equal(param1, param2):
            all_params_match = False
            diff = torch.abs(param1 - param2).max().item()
            max_diff = max(max_diff, diff)
            mismatched_params.append((key, diff))

    if all_params_match:
        print(f"✓ All {len(final_state1)} parameters are identical (exact match)")
    else:
        print(f"✗ Found {len(mismatched_params)} mismatched parameters:")
        for param_name, diff in mismatched_params[:5]:  # Show first 5
            print(f"  - {param_name}: max diff = {diff:.2e}")
        if len(mismatched_params) > 5:
            print(f"  ... and {len(mismatched_params) - 5} more")
        print(f"  Maximum difference: {max_diff:.2e}")
        raise AssertionError(
            f"Model parameters are non-deterministic after training step"
        )

    # Test 8: Third run to triple-check
    print("\n[Test 8] Triple-checking with third training step...")
    model.load_state_dict(model_state_initial)
    optimizer.load_state_dict(optimizer_state_initial)

    loss3 = train_one_step(model, optimizer, batch_data, batch_target, criterion)
    final_state3 = copy.deepcopy(model.state_dict())

    print(f"✓ Third training step completed, loss: {loss3.item():.8f}")

    # Compare with first run
    loss_match_3 = torch.equal(loss1, loss3)
    params_match_3 = all(
        torch.equal(final_state1[k], final_state3[k]) for k in final_state1.keys()
    )

    if loss_match_3 and params_match_3:
        print("✓ Third run matches first run exactly")
    else:
        print("✗ Third run does not match first run")
        raise AssertionError("Training step is not consistently deterministic")

    # Summary
    print("\n" + "=" * 70)
    print("All Deterministic Training Step Tests Passed! ✓")
    print("=" * 70)
    print("\nTest Summary:")
    print(f"  - Device: {device}")
    print(f"  - Seed: {seed}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Loss (all runs): {loss1.item():.10f}")
    print(f"  - All three training steps produced identical results")
    print("\n✓ Property 4 validated: Training steps are deterministic")

    return True


if __name__ == "__main__":
    try:
        success = test_training_step_determinism()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
