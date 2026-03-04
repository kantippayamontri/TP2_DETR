import logging
import math
import sys
from typing import Iterable

import torch
from sklearn.metrics import accuracy_score


logger = logging.getLogger()


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    seen_classes_names: list = None,
    unseen_classes_names: list = None,
    semantic_guided_unseen_weights: torch.Tensor = None,
):
    model.train()
    criterion.train()

    epoch_loss_dict_scaled = {}
    epoch_loss_dict_unscaled = {}
    count = 0

    for batch_idx, (samples, targets) in enumerate(data_loader):
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
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(
            samples,
            classes,
            description_dict,
            targets,
            epoch,
            seen_classes_names=seen_classes_names,
            unseen_classes_names=unseen_classes_names,
            batch_idx=batch_idx,
        )

        loss_dict = criterion(
            outputs,
            targets,
            epoch,
            batch_idx=batch_idx,
            semantic_guided_unseen_weights=semantic_guided_unseen_weights,
        )
        weight_dict = criterion.weight_dict

        # ============================================================================
        # REPRODUCIBILITY: Deterministic loss computation
        # ============================================================================
        # Sort dictionary keys before iteration to ensure deterministic order.
        # Python dictionaries maintain insertion order (Python 3.7+), but when
        # combining losses from multiple sources, the order may vary between runs.
        # Sorting keys ensures consistent computation order, which is critical for
        # bit-exact reproducibility in floating-point operations.
        sorted_keys = sorted([k for k in loss_dict.keys() if k in weight_dict])
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in sorted_keys
        )  # the weight_dict controls which loss is applied

        # Sort keys for deterministic logging and tracking
        loss_dict_unscaled = {
            f"{k}_unscaled": loss_dict[k].item() for k in sorted(loss_dict.keys())
        }  # logging all losses thet are computed (note that some of these are not allpied for backward)

        loss_dict_scaled = {
            k: loss_dict[k].item() * weight_dict[k]
            for k in sorted(loss_dict.keys())
            if k in weight_dict
        }

        # Use sorted keys for deterministic summation order
        loss_value = sum(loss_dict_scaled[k] for k in sorted(loss_dict_scaled.keys()))

        # update the epoch_loss
        epoch_loss_dict_unscaled.update(
            {
                k: epoch_loss_dict_unscaled.get(k, 0.0) + v
                for k, v in loss_dict_unscaled.items()
            }
        )
        epoch_loss_dict_scaled.update(
            {
                k: epoch_loss_dict_scaled.get(k, 0.0) + v
                for k, v in loss_dict_scaled.items()
            }
        )
        count = count + len(targets)
        logger.info(
            f"Train Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_value:{loss_value}, loss_dict_scaled:{loss_dict_scaled}"
        )
        logger.info(
            f"Train Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_dict_unscaled:{loss_dict_unscaled}"
        )

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_scaled)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    epoch_loss_dict_scaled.update(
        {k: v / count for k, v in epoch_loss_dict_scaled.items()}
    )
    epoch_loss_dict_unscaled.update(
        {k: v / count for k, v in epoch_loss_dict_unscaled.items()}
    )
    logger.info(
        f"Train Epoch: {epoch}, epoch_loss_dict_scaled:{epoch_loss_dict_scaled}"
    )
    logger.info(
        f"Train Epoch: {epoch}, epoch_loss_dict_unscaled:{epoch_loss_dict_unscaled}"
    )

    return epoch_loss_dict_scaled
