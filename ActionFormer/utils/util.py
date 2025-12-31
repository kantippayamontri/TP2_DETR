import torch
import random
import os
import numpy as np
import logging
import csv

#################### fix seed #####################
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

#################### logger #####################
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
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
    data_row=[]

    test_stats['per_iou_ap_raw'], test_stats['mAP_raw'],
    data_row.append(epoch)
    for item in test_stats['per_iou_ap_raw']:
        data_row.append(np.round(item*100,6))
    data_row.append(np.round(test_stats['mAP_raw']*100,6))
    data_row.append(np.round(test_stats['AR@1_raw']*100,6))
    data_row.append(np.round(test_stats['AR@5_raw']*100,6))
    data_row.append(np.round(test_stats['AR@10_raw']*100,6))
    data_row.append(np.round(test_stats['AR@25_raw']*100,6))
    data_row.append(np.round(test_stats['AR@40_raw']*100,6))
    data_row.append(np.round(test_stats['AR@50_raw']*100,6))
    data_row.append(np.round(test_stats['AR@100_raw']*100,6))
    data_row.append(np.round(test_stats['AUC_raw']*100,6))

    with open(path,'a+',newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
