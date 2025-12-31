
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
from test import test
from pretrain import pretrain
from tqdm import tqdm
from datetime import datetime
import math
import json
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,inverse_sigmoid)
import matplotlib.pyplot as plt
import seaborn as sns
from utils.segment_ops import segment_cw_to_t1t2


if __name__ == '__main__':

    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)

    # load dataset
    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    
    sampling_rate = 30  # 30fps
    time_step_aggregation = 8  # aggragate every 8 frame
    aggregated_sampling_rate = sampling_rate / time_step_aggregation  # new sampling rate
    d = 1 / aggregated_sampling_rate

    for samples, targets in train_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['segments', 'labels', 'salient_mask', 'semantic_labels'] else v for k, v in t.items()} for t in targets]
        
        # origin CLIP features
        clip_feat, original_mask = samples.decompose()
        bs, t, c = clip_feat.shape
        

        for batch_idx in range(bs):
            out_bbox = targets[batch_idx]['segments']
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) * t
            out_boxes[:, 0] = torch.ceil(out_boxes[:, 0])
            out_boxes[:, 1] = torch.floor(out_boxes[:, 1])
            out_boxes[:, 0] = torch.min(out_boxes[:, 0], out_boxes[:, 1])
            out_boxes = out_boxes.int()

            for i, (s, e) in enumerate(out_boxes):
                Z = torch.fft.fft(clip_feat[batch_idx, s: e+1, :].cpu(), dim=1)
                frequency_energy_t = torch.sum(torch.abs(Z)**2, dim=1)
                frequency_energy = torch.sum(frequency_energy_t)
                S = frequency_energy/(e-s+1)
'''
# 假設 X 是 (T, D) 的矩陣，包含 T 個時間段的 CLIP 特徵，每個時間段是 8 幀的平均特徵
X = np.random.randn(10, 64)  # 假設有 10 個時間段，每個時間段有 64 維特徵

# 對每個時間段的特徵進行 FFT，假設時間段是行，特徵是列
Z = np.fft.fft(X, axis=0)  # 沿著每個時間段的維度進行 FFT，返回頻域數據

# 計算頻率能量，這是對每個頻率分量的模平方進行求和
frequency_energy = np.sum(np.abs(Z)**2, axis=0)

# 計算動作性分數
segment_length = X.shape[0]  # 這裡是時間段數量 T
s = 1 / segment_length * np.sum(frequency_energy)
actionness_score = 1 / (1 + np.exp(-s))  # 使用 sigmoid 函數
'''