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

def update_best_figures(figures_dir='figures', best_dir='best_figures'):
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    for filename in os.listdir(figures_dir):
        src_path = os.path.join(figures_dir, filename)
        dst_path = os.path.join(best_dir, filename)

        # copy2 preserves metadata (timestamps etc.)
        shutil.copy2(src_path, dst_path)

def clear_figure_folder(figures_dir='figures'):
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
    print(f"Total parameters: {total_params/1000000} M, Trainable parameters: {trainable_params/1000000} M")
    return total_params, trainable_params 


def check_directory(args):

    if not os.path.exists(os.path.join('./results/logs/',args.dataset_name)):
        os.makedirs(os.path.join('./results/logs/',args.dataset_name))
    logger = get_logger(os.path.join('./results/logs/',args.dataset_name,args.model_name+'_'+args.filename +'.log'))

    if not os.path.exists(os.path.join('./results/excel',args.dataset_name)):
        os.makedirs(os.path.join('./results/excel',args.dataset_name))
    if not os.path.exists(os.path.join('./results/excel',args.dataset_name, 'Train')):
        os.makedirs(os.path.join('./results/excel',args.dataset_name, 'Train'))
    
    if not os.path.exists(os.path.join('./ckpt/', args.dataset_name)):
        os.makedirs(os.path.join('./ckpt/', args.dataset_name))

    return logger

def seed_worker(worker_id):
    # 為每個 worker 設定不同的 deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset, 
                     batch_size=1,
                     shuffle=False,
                     num_workers=0,
                     pin_memory=True,
                     drop_last=False,
                     collate_fn=None,
                     seed=42):
    """
    建立一個 deterministic 的 DataLoader，支援 shuffle 和多 worker 的 reproducibility。
    """
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
        generator=generator
    )

if __name__ == '__main__':
    print(f"Run!!!")
    exit()

    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed=args.seed
    setup_seed(seed)

    args.filename = filename
    logger = check_directory(args)
    logger.info('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
    logger.info(args)


    # load model
    model, criterion, postprocessor = build_model(args,device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters/1000000} M')


    # load dataset
    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    train_val_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            # backbone (ActionFormer)
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            # reference_points, sampling_offsets
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
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
            for c in (seen_classes_names+unseen_classes_names):
                act_prompt.append("a video of a person doing"+" "+c)
            tokens = clip_pkg.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
            text_feats = model.text_encoder(tokens).float()
            text_emb_seen = text_feats[:len(seen_classes_names)]
            text_emb_unseen = text_feats[len(seen_classes_names):]
            # cosine similarity
            sim_matrix = F.cosine_similarity(
                text_emb_seen.unsqueeze(1),  # [S, 1, D]
                text_emb_unseen.unsqueeze(0),  # [1, U, D]
                dim=-1  # -> [S, U]
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
        clear_figure_folder('best_figures')
        for epoch in tqdm(range(args.epochs)):
            clear_figure_folder()
            epoch_loss_dict_scaled = train(model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_max_norm, seen_classes_names=seen_classes_names, unseen_classes_names=unseen_classes_names, semantic_guided_unseen_weights=semantic_guided_unseen_weights)
            lr_scheduler.step()
            # torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'last_'+args.model_name+'_'+args.filename+'.pkl'))

            if epoch % args.train_interval == 0 and args.train_interval != -1:
                train_stats = test(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=train_val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args)
                logger.info('||'.join(['Train map @ {} = {:.3f} '.format(train_stats['iou_range'][i],train_stats['per_iou_ap_raw'][i]*100) for i in range(len(train_stats['iou_range']))]))
                logger.info('Intermediate Train mAP Avg ALL: {}'.format(train_stats['mAP_raw']*100))
                logger.info('Intermediate Train AR@1: {}, AR@5: {}, AR@10: {}, AR@50:{}, AR@100:{}, AUC@100:{}'.format(train_stats['AR@1_raw']*100, train_stats['AR@5_raw']*100,train_stats['AR@10_raw']*100,train_stats['AR@50_raw']*100,train_stats['AR@100_raw']*100,train_stats['AUC_raw']*100))
                write_to_csv(os.path.join('./results/excel',args.dataset_name,'Train',args.model_name+'_'+args.filename), train_stats, epoch)

            
            if epoch % args.test_interval == 0 and args.test_interval != -1:
                test_stats = test(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args, seen_classes_names=seen_classes_names, unseen_classes_names=unseen_classes_names)
                logger.info('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_stats['iou_range'][i],test_stats['per_iou_ap_raw'][i]*100) for i in range(len(test_stats['iou_range']))]))
                logger.info('Intermediate mAP Avg ALL: {}'.format(test_stats['mAP_raw']*100))
                logger.info('Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(test_stats['AR@1_raw']*100, test_stats['AR@5_raw']*100, test_stats['AR@10_raw']*100, test_stats['AR@50_raw']*100,test_stats['AR@100_raw']*100,test_stats['AUC_raw']*100))
                write_to_csv(os.path.join('./results/excel',args.dataset_name,args.model_name+'_'+args.filename), test_stats, epoch)

                if args.enable_freqCalibrate:
                    test_freq_stats = test_freq(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args, type='sigmoid')
                    logger.info('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_freq_stats['iou_range'][i],test_freq_stats['per_iou_ap_raw'][i]*100) for i in range(len(test_freq_stats['iou_range']))]))
                    logger.info('(FreqCalibrate)Intermediate mAP Avg ALL: {}'.format(test_freq_stats['mAP_raw']*100))
                    logger.info('(FreqCalibrate)Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(test_freq_stats['AR@1_raw']*100, test_freq_stats['AR@5_raw']*100, test_freq_stats['AR@10_raw']*100, test_freq_stats['AR@50_raw']*100,test_freq_stats['AR@100_raw']*100,test_freq_stats['AUC_raw']*100))
                    write_to_csv(os.path.join('./results/excel',args.dataset_name,args.model_name+'_'+args.filename), test_freq_stats, epoch)
                    test_stats = test_freq_stats

                    # test_freq_stats2 = test_freq(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args, type='sigmoid_enlarge')
                    # logger.info('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_freq_stats2['iou_range'][i],test_freq_stats2['per_iou_ap_raw'][i]*100) for i in range(len(test_freq_stats2['iou_range']))]))
                    # logger.info('(FreqCalibrate)Intermediate mAP Avg ALL: {}'.format(test_freq_stats2['mAP_raw']*100))
                    # logger.info('(FreqCalibrate)Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(test_freq_stats2['AR@1_raw']*100, test_freq_stats2['AR@5_raw']*100, test_freq_stats2['AR@10_raw']*100, test_freq_stats2['AR@50_raw']*100,test_freq_stats2['AR@100_raw']*100,test_freq_stats2['AUC_raw']*100))
                    # write_to_csv(os.path.join('./results/excel',args.dataset_name,args.model_name+'_'+args.filename), test_freq_stats2, epoch)

                # update best
                if test_stats['mAP_raw'] > best_stats.get('mAP_raw',0.0):
                    best_stats = test_stats
                    logger.info('new best metric {:.4f}@epoch{}'.format(best_stats['mAP_raw']*100, epoch))
                    update_best_figures()
                    # torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'best_'+args.model_name+'_'+args.filename+'.pkl'))

                logger.info('Current best metric from {:.4f}@epoch{}'.format(best_stats['mAP_raw']*100, best_stats['epoch']))

    
    iou = best_stats['iou_range']
    max_map = best_stats['per_iou_ap_raw']
    max_Avg = best_stats['mAP_raw']
    best_epoch = best_stats['epoch']
    logger.info('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
    logger.info('MAX mAP Avg ALL: {} in Epoch: {}'.format(max_Avg*100,best_epoch))
    logger.info('MAX AR@10: {}, AR@25: {}, AR@40: {}, AUC: {}'.format(best_stats['AR@10_raw']*100, best_stats['AR@25_raw']*100, best_stats['AR@40_raw']*100, best_stats['AUC_raw']*100))
                


