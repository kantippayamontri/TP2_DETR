#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from utils.misc import collate_fn
import random
import options
from options import merge_cfg_from_file
import numpy as np
from tqdm import tqdm
from utils.util import setup_seed, get_logger, write_to_csv
import dataset
from train import train
from test import test, test_freq
from pretrain import pretrain
from tqdm import tqdm
from datetime import datetime
import math
import json
from pprint import pformat
import shutil

from models import build_model

current_time = datetime.now()
filename = current_time.strftime("%m%d_%H%M")


def update_best_figures(figures_dir="figures", best_dir="best_figures"):
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    for filename in os.listdir(figures_dir):
        src_path = os.path.join(figures_dir, filename)
        dst_path = os.path.join(best_dir, filename)

        # copy2 preserves metadata (timestamps etc.)
        shutil.copy2(src_path, dst_path)


def clear_figure_folder(figures_dir="figures"):
    for filename in os.listdir(figures_dir):
        file_path = os.path.join(figures_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# Computing the parameters of the model
def count_parameters(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        temp = param.numel()
        total_params += temp

        if param.requires_grad:
            trainable_params += temp
    print(
        f"Total parameters: {total_params/1000000} M, Trainable parameters: {trainable_params/1000000} M"
    )
    return total_params, trainable_params


def check_directory(args):

    if not os.path.exists(os.path.join("./results/logs/", args.dataset_name)):
        os.makedirs(os.path.join("./results/logs/", args.dataset_name))
    logger = get_logger(
        os.path.join(
            "./results/logs/",
            args.dataset_name,
            args.model_name + "_" + args.filename + ".log",
        )
    )

    if not os.path.exists(os.path.join("./results/excel", args.dataset_name)):
        os.makedirs(os.path.join("./results/excel", args.dataset_name))
    if not os.path.exists(os.path.join("./results/excel", args.dataset_name, "Train")):
        os.makedirs(os.path.join("./results/excel", args.dataset_name, "Train"))

    if not os.path.exists(os.path.join("./ckpt/", args.dataset_name)):
        os.makedirs(os.path.join("./ckpt/", args.dataset_name))

    return logger


def seed_worker(worker_id):
    """
    Seed worker processes for deterministic DataLoader behavior.

    This function is called by DataLoader for each worker process to ensure
    reproducible data loading when using multiple workers. It seeds all RNG
    sources (NumPy, Python random, PyTorch) with a deterministic seed derived
    from the main process's RNG state.

    Args:
        worker_id: Worker process ID (automatically provided by DataLoader)

    Note:
        - Must be used with a seeded Generator passed to DataLoader
        - Ensures each worker has a different but deterministic seed
        - Critical for reproducible training with num_workers > 0
    """
    # Get deterministic seed from the main process's generator
    # torch.initial_seed() returns the seed set by the DataLoader's generator
    worker_seed = torch.initial_seed() % 2**32

    # Seed all RNG sources for this worker process
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
    """
    Build a deterministic DataLoader with reproducible shuffling and worker seeding.

    This function creates a DataLoader that ensures reproducible behavior across runs:
    - Uses a seeded Generator for deterministic shuffling
    - Applies seed_worker to each worker process for consistent data loading
    - Maintains reproducibility even with shuffle=True and multiple workers

    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data (uses seeded generator for reproducibility)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        collate_fn: Function to collate samples into batches
        seed: Random seed for reproducible shuffling and worker initialization

    Returns:
        DataLoader with deterministic behavior

    Note:
        - The Generator ensures shuffle produces the same order with the same seed
        - worker_init_fn ensures each worker has a deterministic but unique seed
        - Critical for reproducible training across multiple runs
    """
    # Create a seeded generator for deterministic shuffling
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
        worker_init_fn=seed_worker,  # Seed each worker process deterministically
        generator=generator,  # Use seeded generator for reproducible shuffling
    )


if __name__ == "__main__":
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(
            args, args.cfg_path
        )  # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    setup_seed(seed)

    # ============================================================================
    # REPRODUCIBILITY: CUDA deterministic configuration
    # ============================================================================
    # Check and set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations
    # This environment variable must be set for torch.use_deterministic_algorithms()
    # to work properly with CUDA operations. If not set, some operations may fail.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        print(
            "WARNING: CUBLAS_WORKSPACE_CONFIG not set. Setting to ':4096:8' for deterministic CUDA operations"
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args.filename = filename
    logger = check_directory(args)

    # ============================================================================
    # REPRODUCIBILITY: Enable deterministic algorithms
    # ============================================================================
    # torch.use_deterministic_algorithms(True) forces PyTorch to use only
    # deterministic algorithms, ensuring bit-exact reproducibility across runs.
    #
    # warn_only=True allows operations without deterministic implementations
    # (like cumsum_cuda) to run with a warning instead of raising an error.
    # This is necessary because some operations don't have deterministic versions.
    #
    # Requirements for full determinism:
    # 1. CUBLAS_WORKSPACE_CONFIG must be set (checked above)
    # 2. cuDNN deterministic mode enabled (done in setup_seed)
    # 3. All RNG sources seeded (done in setup_seed)
    # 4. DataLoader uses seeded generator and worker_init_fn (done in build_dataloader)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(
        "Deterministic algorithms enabled with warn_only=True (some operations may not be fully deterministic)"
    )

    # ============================================================================
    # REPRODUCIBILITY: Log configuration for verification
    # ============================================================================
    # Log all reproducibility settings to help debug non-deterministic behavior.
    # These logs provide a complete picture of the determinism configuration,
    # making it easier to verify settings and troubleshoot issues.
    logger.info("=============seed: {}, pid: {}=============".format(seed, os.getpid()))
    logger.info("Reproducibility Configuration:")
    logger.info(
        f"  - Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}"
    )
    logger.info(
        f"  - CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}"
    )
    logger.info(f"  - cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"  - cuDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(args)

    # load model
    model, criterion, postprocessor = build_model(args, device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters/1000000} M")

    # load dataset
    train_dataset = getattr(dataset, args.dataset_name + "Dataset")(
        subset="train", mode="train", args=args
    )
    val_dataset = getattr(dataset, args.dataset_name + "Dataset")(
        subset="inference", mode="inference", args=args
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        seed=seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        seed=seed,
    )
    train_val_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        seed=seed,
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            # backbone (ActionFormer)
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            # reference_points, sampling_offsets
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.semantic_guided_loss:
        from models.clip import clip as clip_pkg
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            # text_emb_seen: [S, D]
            seen_classes_names = train_loader.dataset.classes
            # text_emb_unseen: [U, D]
            unseen_classes_names = val_loader.dataset.classes

            act_prompt = []
            for c in seen_classes_names + unseen_classes_names:
                act_prompt.append("a video of a person doing" + " " + c)
            tokens = (
                clip_pkg.tokenize(act_prompt).long().to(device)
            )  # input_ids->input_ids:[150,length]
            text_feats = model.text_encoder(tokens).float()
            text_emb_seen = text_feats[: len(seen_classes_names)]
            text_emb_unseen = text_feats[len(seen_classes_names) :]
            # cosine similarity
            sim_matrix = F.cosine_similarity(
                text_emb_seen.unsqueeze(1),  # [S, 1, D]
                text_emb_unseen.unsqueeze(0),  # [1, U, D]
                dim=-1,  # -> [S, U]
            )
            ## 【Way1】
            # semantic_guided_unseen_weights = torch.exp(-sim_matrix)  # μ(s, u) = exp(-cosine)

            ## 【Way2: seen(BCE), unseen(KL)】
            ## 【Way3: seen(BCE), unseen(BCE + soft target)】
            # semantic_guided_unseen_weights = F.softmax(sim_matrix, dim=1)

            ## 【Way4: seen(BCE), unseen(KL) + loss balance】
            semantic_guided_unseen_weights = sim_matrix

            ## 【Way5: seen(KL), unseen(KL)】
            # sim_matrix = F.cosine_similarity(
            #     text_emb_seen.unsqueeze(1),  # [S, 1, D]
            #     text_feats.unsqueeze(0),  # [1, S+U, D]
            #     dim=-1  # -> [S, S+U]
            # )
            # # normal
            # semantic_guided_unseen_weights = (sim_matrix-sim_matrix.min())/(sim_matrix.max()-sim_matrix.min())
            # # shaper
            # temparature = 0.2
            # shaper_sim_matrix = F.softmax(sim_matrix/temparature, dim=1)
            # semantic_guided_unseen_weights = (shaper_sim_matrix-shaper_sim_matrix.min(axis=1, keepdims=True)[0])/(shaper_sim_matrix.max(axis=1, keepdims=True)[0]-shaper_sim_matrix.min(axis=1, keepdims=True)[0])
    else:
        # text_emb_seen: [S, D]
        seen_classes_names = train_loader.dataset.classes
        # text_emb_unseen: [U, D]
        unseen_classes_names = val_loader.dataset.classes
        semantic_guided_unseen_weights = None

    best_stats = {}
    with torch.autograd.set_detect_anomaly(True):
        clear_figure_folder("best_figures")
        for epoch in tqdm(range(args.epochs)):
            clear_figure_folder()
            epoch_loss_dict_scaled = train(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                seen_classes_names=seen_classes_names,
                unseen_classes_names=unseen_classes_names,
                semantic_guided_unseen_weights=semantic_guided_unseen_weights,
            )
            lr_scheduler.step()

            if epoch % args.train_interval == 0 and args.train_interval != -1:
                train_stats = test(
                    model=model,
                    criterion=criterion,
                    postprocessor=postprocessor,
                    data_loader=train_val_loader,
                    dataset_name=args.dataset_name,
                    epoch=epoch,
                    device=device,
                    args=args,
                )
                logger.info(
                    "||".join(
                        [
                            "Train map @ {} = {:.3f} ".format(
                                train_stats["iou_range"][i],
                                train_stats["per_iou_ap_raw"][i] * 100,
                            )
                            for i in range(len(train_stats["iou_range"]))
                        ]
                    )
                )
                logger.info(
                    "Intermediate Train mAP Avg ALL: {}".format(
                        train_stats["mAP_raw"] * 100
                    )
                )
                logger.info(
                    "Intermediate Train AR@1: {}, AR@5: {}, AR@10: {}, AR@50:{}, AR@100:{}, AUC@100:{}".format(
                        train_stats["AR@1_raw"] * 100,
                        train_stats["AR@5_raw"] * 100,
                        train_stats["AR@10_raw"] * 100,
                        train_stats["AR@50_raw"] * 100,
                        train_stats["AR@100_raw"] * 100,
                        train_stats["AUC_raw"] * 100,
                    )
                )
                write_to_csv(
                    os.path.join(
                        "./results/excel",
                        args.dataset_name,
                        "Train",
                        args.model_name + "_" + args.filename,
                    ),
                    train_stats,
                    epoch,
                )

            if epoch % args.test_interval == 0 and args.test_interval != -1:
                test_stats = test(
                    model=model,
                    criterion=criterion,
                    postprocessor=postprocessor,
                    data_loader=val_loader,
                    dataset_name=args.dataset_name,
                    epoch=epoch,
                    device=device,
                    args=args,
                    seen_classes_names=seen_classes_names,
                    unseen_classes_names=unseen_classes_names,
                )
                logger.info(
                    "||".join(
                        [
                            "Intermediate map @ {} = {:.3f} ".format(
                                test_stats["iou_range"][i],
                                test_stats["per_iou_ap_raw"][i] * 100,
                            )
                            for i in range(len(test_stats["iou_range"]))
                        ]
                    )
                )
                logger.info(
                    "Intermediate mAP Avg ALL: {}".format(test_stats["mAP_raw"] * 100)
                )
                logger.info(
                    "Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}".format(
                        test_stats["AR@1_raw"] * 100,
                        test_stats["AR@5_raw"] * 100,
                        test_stats["AR@10_raw"] * 100,
                        test_stats["AR@50_raw"] * 100,
                        test_stats["AR@100_raw"] * 100,
                        test_stats["AUC_raw"] * 100,
                    )
                )
                write_to_csv(
                    os.path.join(
                        "./results/excel",
                        args.dataset_name,
                        args.model_name + "_" + args.filename,
                    ),
                    test_stats,
                    epoch,
                )

                if args.enable_freqCalibrate:
                    test_freq_stats = test_freq(
                        model=model,
                        criterion=criterion,
                        postprocessor=postprocessor,
                        data_loader=val_loader,
                        dataset_name=args.dataset_name,
                        epoch=epoch,
                        device=device,
                        args=args,
                        type="sigmoid",
                    )
                    logger.info(
                        "||".join(
                            [
                                "Intermediate map @ {} = {:.3f} ".format(
                                    test_freq_stats["iou_range"][i],
                                    test_freq_stats["per_iou_ap_raw"][i] * 100,
                                )
                                for i in range(len(test_freq_stats["iou_range"]))
                            ]
                        )
                    )
                    logger.info(
                        "(FreqCalibrate)Intermediate mAP Avg ALL: {}".format(
                            test_freq_stats["mAP_raw"] * 100
                        )
                    )
                    logger.info(
                        "(FreqCalibrate)Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}".format(
                            test_freq_stats["AR@1_raw"] * 100,
                            test_freq_stats["AR@5_raw"] * 100,
                            test_freq_stats["AR@10_raw"] * 100,
                            test_freq_stats["AR@50_raw"] * 100,
                            test_freq_stats["AR@100_raw"] * 100,
                            test_freq_stats["AUC_raw"] * 100,
                        )
                    )
                    write_to_csv(
                        os.path.join(
                            "./results/excel",
                            args.dataset_name,
                            args.model_name + "_" + args.filename,
                        ),
                        test_freq_stats,
                        epoch,
                    )
                    test_stats = test_freq_stats

                # update best
                if test_stats["mAP_raw"] > best_stats.get("mAP_raw", 0.0):
                    best_stats = test_stats
                    logger.info(
                        "new best metric {:.4f}@epoch{}".format(
                            best_stats["mAP_raw"] * 100, epoch
                        )
                    )
                    update_best_figures()

                logger.info(
                    "Current best metric from {:.4f}@epoch{}".format(
                        best_stats["mAP_raw"] * 100, best_stats["epoch"]
                    )
                )

    iou = best_stats["iou_range"]
    max_map = best_stats["per_iou_ap_raw"]
    max_Avg = best_stats["mAP_raw"]
    best_epoch = best_stats["epoch"]
    logger.info(
        "||".join(
            [
                "MAX map @ {} = {:.3f} ".format(iou[i], max_map[i] * 100)
                for i in range(len(iou))
            ]
        )
    )
    logger.info("MAX mAP Avg ALL: {} in Epoch: {}".format(max_Avg * 100, best_epoch))
    logger.info(
        "MAX AR@10: {}, AR@25: {}, AR@40: {}, AUC: {}".format(
            best_stats["AR@10_raw"] * 100,
            best_stats["AR@25_raw"] * 100,
            best_stats["AR@40_raw"] * 100,
            best_stats["AUC_raw"] * 100,
        )
    )
