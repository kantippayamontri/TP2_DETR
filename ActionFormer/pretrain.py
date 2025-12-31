import torch
from typing import Iterable
import logging
import torch.nn.functional as F

logger = logging.getLogger()


def pretrain(model: torch.nn.Module, 
             criterion: torch.nn.Module,
            data_loader: Iterable, 
            optimizer: torch.optim.Optimizer,
            device: torch.device, 
            epoch: int, 
            max_norm: float = 0):
    model.train()
    criterion.train()

    total_salient_loss = 0
    count = 0

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['segments', 'labels', 'salient_mask', 'semantic_labels'] else v for k, v in t.items()} for t in targets]
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(samples, classes, description_dict, targets, epoch, pretrain=True)

        loss_dict = criterion(outputs, targets, epoch, pretrain=True)
        salient_loss = loss_dict['loss_salient']


        optimizer.zero_grad()
        salient_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        count = count + len(targets)
        total_salient_loss += salient_loss.item()
        logger.info(f"Pretrain Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_value:{salient_loss}")
        
    logger.info(f"Pretrain Epoch {epoch}, Salient Loss: {total_salient_loss / count}")



def pretrain_recon(model: torch.nn.Module, 
          data_loader: Iterable, 
          optimizer: torch.optim.Optimizer,
          device: torch.device, 
          epoch: int, 
          max_norm: float = 0):
    model.train()

    epoch_loss = 0
    total_recon_loss = 0
    total_consist_loss = 0
    lambda_recon = 3
    count = 0

    for samples, targets in data_loader:
        samples = samples.to(device)
        clip_feat, _ = samples.decompose() # [b, t, c]
        features , _ = model.backbone(samples) # [[b, c, t], [b, c, t/2], ...]

        pyramid_clip_feats = []
        for l, feat in enumerate(features):
            src, _ = feat.decompose()
            pyramid_clip_feats.append(src)

        recon_loss = compute_reconstruction_loss(pyramid_clip_feats, clip_feat.permute(0,2,1))
        consist_loss = comput_feature_consistency_loss(pyramid_clip_feats)
        losses = recon_loss + lambda_recon*consist_loss


        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        count = count + len(targets)
        epoch_loss += losses.item()
        total_recon_loss += recon_loss.item()
        total_consist_loss += consist_loss.item()
        
    logger.info(f"Backbone Pretrain Epoch {epoch}, Loss: {epoch_loss / count}, Recon Loss: {total_recon_loss/count}, Consist Loss: {total_consist_loss/count}")


def compute_reconstruction_loss(features, clip_feat):
    loss = 0
    for i, feat in enumerate(features):
        if i==0:
            loss += F.mse_loss(feat, clip_feat)
        else:
            scale_factor = clip_feat.shape[-1] // feat.shape[-1]
            upsampled_feat = F.interpolate(feat, scale_factor=scale_factor, mode='nearest')
            # upsampled_feat = F.interpolate(feat, scale_factor=scale_factor, mode='linear', align_corners=False)
            loss += F.mse_loss(upsampled_feat, clip_feat)
    return loss


def comput_feature_consistency_loss(features):
    loss = 0
    level = len(features)
    for lvl in range(level-1):
        scale_factor = features[lvl].shape[-1] // features[lvl+1].shape[-1]
        feat = features[lvl]
        upsampled_feat = F.interpolate(features[lvl+1], scale_factor=scale_factor, mode='nearest')
        # upsampled_feat = F.interpolate(features[lvl+1], scale_factor=scale_factor, mode='linear', align_corners=False)
        loss += F.mse_loss(upsampled_feat, feat)
    return loss