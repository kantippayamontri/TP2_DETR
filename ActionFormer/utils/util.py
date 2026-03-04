import csv
import logging
import os
import random

import numpy as np
import torch


#################### fix seed #####################
def setup_seed(seed):
    """
    Setup random seed for reproducibility across all random number generators.

    This function ensures deterministic behavior by seeding all major RNG sources:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - cuDNN backend settings for deterministic operations

    Args:
        seed (int): Random seed value to use across all RNG sources

    Note:
        - PYTHONHASHSEED should also be set in the environment before Python starts
        - torch.use_deterministic_algorithms() should be called in main.py after
          model creation to enable full deterministic behavior
        - Setting cudnn.benchmark=False may reduce performance but ensures reproducibility
    """
    # Seed Python's built-in random module
    random.seed(seed)

    # Set Python hash seed for deterministic dictionary/set ordering
    # Note: This should ideally be set before Python starts via environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed NumPy's random number generator
    np.random.seed(seed)

    # Seed PyTorch's random number generator for CPU
    torch.manual_seed(seed)

    # Seed PyTorch's random number generator for current CUDA device
    torch.cuda.manual_seed(seed)

    # Seed PyTorch's random number generator for all CUDA devices
    torch.cuda.manual_seed_all(seed)

    # Disable cuDNN benchmark mode for deterministic behavior
    # Benchmark mode selects the best algorithm but may vary between runs
    torch.backends.cudnn.benchmark = False

    # Enable cuDNN deterministic mode to ensure reproducible results
    # This forces cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True

    # Enable cuDNN backend (required for deterministic operations)
    # This ensures cuDNN is active and can apply deterministic settings
    torch.backends.cudnn.enabled = True


#################### logger #####################
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def write_to_csv(dname, test_stats, epoch):
    path = dname + "_results.csv"
    data_row = []

    test_stats["per_iou_ap_raw"], test_stats["mAP_raw"],
    data_row.append(epoch)
    for item in test_stats["per_iou_ap_raw"]:
        data_row.append(np.round(item * 100, 6))
    data_row.append(np.round(test_stats["mAP_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@1_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@5_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@10_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@25_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@40_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@50_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AR@100_raw"] * 100, 6))
    data_row.append(np.round(test_stats["AUC_raw"] * 100, 6))

    with open(path, "a+", newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
