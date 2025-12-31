from utils.misc import (accuracy)
from utils.segment_ops import segment_cw_to_t1t2,segment_iou
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import defaultdict
import itertools
import math
from utils.draw import plot_attention_map, plot_SAattention_map
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger()

def compute_exp_ramp_weight(current_epoch, max_epoch=15, target_weight=2.0, beta=5.0):
    if current_epoch >= max_epoch:
        return target_weight
    scale = (math.exp(beta * current_epoch / max_epoch) - 1) / (math.exp(beta) - 1)
    weight = 1 + (target_weight - 1) * scale
    return weight


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. [bs,num_queries,num_classes]
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs. [bs,num_queries,num_classes]
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # [bs,num_queries,num_classes]
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets) # [bs,num_queries,num_classes]
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes 

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha, split_id, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            base_losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            gamma: gamma in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.split_id = split_id

        self.gamma = args.gamma

        self.actionness_loss = args.actionness_loss 
        self.eval_proposal = args.eval_proposal
        self.enable_classAgnostic = args.enable_classAgnostic

        self.salient_loss = args.salient_loss
        self.salient_loss_impl = args.salient_loss_impl
        self.enable_edgePunish = args.enable_edgePunish
        self.punish = args.punish
        self.enable_softSalient = args.enable_softSalient
        self.as_calibration = args.as_calibration
        self.warmup = True
        self.warmup_epochs = 5

        self.vis_con_loss = args.vis_con_loss
        self.vis_con_loss_temp = args.vis_con_loss_temp

        self.tIoUw_bbox_loss = args.tIoUw_bbox_loss

        self.sparsity_loss = args.sparsity_loss

        self.alignment_loss = args.use_decouple

        self.semantic_guided_loss = args.semantic_guided_loss

        self.SA_LastOnly = args.SA_LastOnly

        self.feedback_loss = args.feedback_loss
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'class_logits' in outputs
        src_logits = outputs['class_logits'] # [bs,num_queries,num_classes]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

    def loss_complete_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Modify the loss_labels, consider the complete of proposal semantic, introding the bg instance in matching stage
        """
        assert 'class_logits' in outputs
        src_logits = outputs['class_logits'] # [bs,num_queries,num_classes+1]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2]-1,
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)


        complete_classes_onehot = target_classes_onehot[idx] # [batch_matched_queries,num_classes+1]
        complete_logits = src_logits[idx] # [batch_matched_queries,num_classes+1]

        loss_ce = sigmoid_focal_loss(complete_logits, complete_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L_reg = L1 + L_tIoU
           Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes # L_1

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_giou'] = loss_giou.sum() / num_boxes # L_tIoU
        return losses
    
    def loss_boxes2(self, outputs, targets, indices, num_boxes):
        """L_reg = tIoU-aware L1 + L_tIoU"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [N, 2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [N, 2]

        # L1 loss
        l1_loss = F.l1_loss(src_boxes, target_boxes, reduction='none').sum(-1)  # [N]

        # Compute tIoU as weight
        with torch.no_grad():
            src_segments = segment_cw_to_t1t2(src_boxes).clamp(min=0, max=1)
            tgt_segments = segment_cw_to_t1t2(target_boxes)
            tiou_scores = segment_iou(src_segments, tgt_segments)  # [N]
            tiou_weights = tiou_scores.pow(2.0)  # gamma 可調，預設為 2.0

        # Apply tIoU-weighted regression loss
        loss_bbox = (l1_loss * tiou_weights).sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(segment_iou(src_segments, tgt_segments))
        loss_giou = loss_giou.sum() / num_boxes

        losses = {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        return losses

    def loss_boxes_refine(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox_refine'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_exclusive(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        matched2all_iou_list = []
        for batch_id, (query_id_list,_) in enumerate(indices):
            batch_pred_boxes = outputs['pred_boxes'][batch_id] # [num_query,2]
            batch_iou = segment_iou(
            segment_cw_to_t1t2(batch_pred_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(batch_pred_boxes).clamp(min=0,max=1)
            ) # [num_query,num_query]
            diag = torch.diag(batch_iou)
            a_diag = torch.diag_embed(diag)
            batch_iou = batch_iou - a_diag

            matched2all_iou = batch_iou[query_id_list] # [matched_query,num_query]
            matched2all_iou_list.append(matched2all_iou)

        batch_matched2all_iou = torch.cat(matched2all_iou_list,dim=0) # [batch_matched2all, num_query]
        loss_exclusive = batch_matched2all_iou.mean()
        losses = {}
        losses['loss_exclusive'] = loss_exclusive
        return losses

    def loss_exclusive_v2(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        all_iou_list = []
        for batch_id, (query_id_list,_) in enumerate(indices):
            batch_pred_boxes = outputs['pred_boxes'][batch_id] # [num_query,2]
            batch_iou = segment_iou(
            segment_cw_to_t1t2(batch_pred_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(batch_pred_boxes).clamp(min=0,max=1)
            ) # [num_query,num_query]
            diag = torch.diag(batch_iou)
            a_diag = torch.diag_embed(diag)
            batch_iou = batch_iou - a_diag

            all_iou_list.append(batch_iou)

        all_iou_list = torch.cat(all_iou_list,dim=0) # [all_iou_list, num_query]
        loss_exclusive = all_iou_list.mean()
        losses = {}
        losses['loss_exclusive'] = 0.5*loss_exclusive
        return losses

    def loss_segmentations(self,outputs, targets, indices, num_boxes):
        '''
        for dense prediction, which is like to maskformer, the loss compute on the memory of the output of encoder
        segmentation_logits: the output of segmentation_logits => [B,T,num_classes]
        feature_list: the feature list after backbone for temporal modeling => list[NestedTensor]
        '''
        # assert 'segmentation_logits' in outputs
        # assert 'segmentation_onehot_labels' in targets[0]

        # # obtain logits
        # segmentation_logits = outputs['segmentation_logits'] # [B,T,num_classes]
        # B,T,C = segmentation_logits.shape

        # # prepare labels
        # segmentation_onehot_labels_pad = torch.zeros_like(segmentation_logits) # [B,T,num_classes], zero in padding region
        # for i, tgt in enumerate(targets):
        #     feat_length = tgt['segmentation_onehot_labels'].shape[0]
        #     segmentation_onehot_labels_pad[i,:feat_length,:] = tgt['segmentation_onehot_labels'] # [feat_length, num_classes]

        # ce_loss = -(segmentation_onehot_labels_pad * F.log_softmax(segmentation_logits, dim=-1)).sum(dim=-1)

        # losses = {}
        # losses['loss_segmentation'] = ce_loss.sum(-1).sum() / B

        assert 'segmentation_logits' in outputs
        assert 'segmentation_labels' in targets[0]
        assert 'logits_mask' in outputs

        # obtain logits
        segmentation_logits = outputs['segmentation_logits'] # [B,T,num_classes+1]
        B,T,C = segmentation_logits.shape

        # prepare labels
        segmentation_labels_pad = torch.full(segmentation_logits.shape[:2], segmentation_logits.shape[2]-1, dtype=torch.int64, device=segmentation_logits.device) # [B,T]
        for i, tgt in enumerate(targets):
            feat_length = tgt['segmentation_labels'].shape[0]
            segmentation_labels_pad[i,:feat_length] = tgt['segmentation_labels'] # [feat_length]

        target_classes_onehot = torch.zeros_like(segmentation_logits) # [bs,T,num_classes+1]
        target_classes_onehot.scatter_(2, segmentation_labels_pad.unsqueeze(-1), 1)
        

        prob = segmentation_logits.sigmoid() # [batch_instance_num,num_classes]
        ce_loss = F.binary_cross_entropy_with_logits(segmentation_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,T,num_classes+1]
        loss = ce_loss * ((1 - p_t) ** 2)
        alpha = 0.25
        if alpha >= 0:
            alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        logits_mask = ~outputs['logits_mask'] # covert False to 1
        loss = torch.einsum("btc,bt->btc",loss,logits_mask) # [b,t,c+1]

        losses = {}
        losses['loss_segmentation'] = loss.sum(-1).mean()
        return losses

    def loss_instances(self,outputs, targets, indices, num_boxes):
        '''
        for instance prediction, modeling the relation of instance region and text 
        instance_logits: the output of instance_logits => [batch_instance_num,num_classes]
        '''
        assert 'instance_logits' in outputs
 
        # obtain logits
        instance_logits = outputs['instance_logits'] #[batch_instance_num,num_classes]
        B,C = instance_logits.shape

        instance_gt = [] 
        for t in targets:
            gt_labels = t['labels'] # [num_instance]
            instance_gt.append(gt_labels)
        instance_gt = torch.cat(instance_gt,dim=0) # [batch_instance_num]->"class id"
        
        # prepare labels
        target_classes_onehot = torch.zeros_like(instance_logits) # [batch_instance_num,num_classes]
        target_classes_onehot.scatter_(1, instance_gt.reshape(-1,1), 1) # [batch_instance_num,num_classes]

        if self.instance_loss_type == "CE":
            loss = -(target_classes_onehot * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)
        elif self.instance_loss_type == "BCE":
            prob = instance_logits.sigmoid() # [batch_instance_num,num_classes]
            ce_loss = F.binary_cross_entropy_with_logits(instance_logits, target_classes_onehot, reduction="none")
            p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,num_queries,num_classes]
            loss = ce_loss * ((1 - p_t) ** 2)
            alpha = 0.25
            if alpha >= 0:
                alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
                loss = alpha_t * loss
        else:
            raise ValueError
        losses = {}
        losses['loss_instance'] = loss.mean()
        return losses

    def loss_matching(self,outputs, targets, indices, num_boxes):
        '''
        for instance prediction, modeling the relation of instance region and text 
        matching_logits: the output of matching_logits => [batch_instance_num,num_classes]
        '''
        assert 'matching_logits' in outputs
 
        # obtain logits
        matching_logits = outputs['matching_logits'] #[batch_instance_num,num_classes]
        B,C = matching_logits.shape

        instance_gt = [] 
        for t in targets:
            gt_labels = t['labels'] # [num_instance]
            instance_gt.append(gt_labels)
        instance_gt = torch.cat(instance_gt,dim=0) # [batch_instance_num]->"class id"
        
        # prepare labels
        target_classes_onehot = torch.zeros_like(matching_logits) # [batch_instance_num,num_classes]
        target_classes_onehot.scatter_(1, instance_gt.reshape(-1,1), 1) # [batch_instance_num,num_classes]

        prob = matching_logits.sigmoid() # [batch_instance_num,num_classes]
        ce_loss = F.binary_cross_entropy_with_logits(matching_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot) # [bs,num_queries,num_classes]
        loss = ce_loss * ((1 - p_t) ** 2)
        alpha = 0.25
        if alpha >= 0:
            alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss

        losses = {}
        losses['loss_matching'] = loss.mean()
        return losses

    def loss_mask(self,outputs, targets, indices, num_boxes):
        '''
        for dense prediction, which is like to maskformer, the loss compute on the memory of the output of encoder
        mask_logits: the output of mask_logits => [B,T,num_classes]
        feature_list: the feature list after backbone for temporal modeling => list[NestedTensor]
        '''
        assert 'mask_logits' in outputs
        assert 'mask_labels' in targets[0]

        # obtain logits
        mask_logits = outputs['mask_logits'].squeeze(2) # [B,T,1]->[B,T]
        B,T= mask_logits.shape

        # prepare labels
        mask_labels_pad = torch.zeros_like(mask_logits) # [B,T], zero in padding region
        for i, tgt in enumerate(targets):
            feat_length = tgt['mask_labels'].shape[0]
            mask_labels_pad[i,:feat_length] = tgt['mask_labels'] # [feat_length]

        ce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_labels_pad, reduction="none").sum(-1) # [B,T]

        losses = {}
        losses['loss_mask'] = ce_loss.sum() / B
        return losses
    
    def loss_actionness(self, outputs, targets, indices, num_boxes, log=False):
        """L_cls
        Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_actionness': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

    def loss_actionness_refine(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_actionness_refine': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses


    def loss_refine_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_refine_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_refine_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_complete_actionness(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,2]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2]-1,
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # complete_classes_onehot = target_classes_onehot[idx] # [batch_matched_queries,num_classes+1]
        # complete_logits = src_logits[idx] # [batch_matched_queries,num_classes+1]

        # loss_ce = sigmoid_focal_loss(complete_logits, complete_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma)
        # losses = {'loss_actionness': loss_ce}
        
        target_classes_onehot = target_classes_onehot[:,:,:] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_actionness': loss_ce}

        return losses

    def loss_queryRelation(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'ROI_relation' in outputs
        ROI_relation = outputs['ROI_relation'] # [bs*q,bs*q]
        assert 'query_relation' in outputs
        query_relation = outputs['query_relation'] # [bs*q,bs*q]

        queryRelation_loss = F.l1_loss(query_relation,ROI_relation)


        losses = {'loss_queryRelation': queryRelation_loss}

        return losses

    def loss_rank(self, outputs, targets, indices, num_boxes, log=True):
        """
            Rank loss, for fine-grained boundary perception
            NOTE: If the num_queries is so small, can not cover all gt, an error will appear here
        """
        def rank_loss(center_logits, inner_logits, outer_logits, margin=0.3):
            # 计算 pairwise ranking loss
            center_logits = center_logits.sigmoid()
            inner_logits = inner_logits.sigmoid()
            outer_logits = outer_logits.sigmoid()

            loss = torch.relu(margin + inner_logits - center_logits) + torch.relu(margin + outer_logits - inner_logits)
            return loss.mean()
        
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        batch_center_logits = []
        batch_inner_logits = []
        batch_outer_logits = []
        for i, (src,tgt) in enumerate(indices):
            batch_idx = torch.full_like(src[0::3],i) # [num_tgt]
            src_idx = src # [num_tgt]
            tgt_idx = tgt # [num_tgt]

            sorted_pairs = sorted(zip(tgt_idx, src_idx))
            sorted_tgt_idx, sorted_src_idx = zip(*sorted_pairs)
            center_tgt, center_src = sorted_tgt_idx[0::3], sorted_src_idx[0::3]
            inner_tgt, inner_src = sorted_tgt_idx[1::3], sorted_src_idx[1::3]
            outer_tgt, outer_src = sorted_tgt_idx[2::3], sorted_src_idx[2::3]

            center_logits = src_logits[(batch_idx,center_src)] # [num_tgt,1]
            batch_center_logits.append(center_logits)
            inner_logits = src_logits[(batch_idx,inner_src)] # [num_tgt,1]
            batch_inner_logits.append(inner_logits)
            outer_logits = src_logits[(batch_idx,outer_src)] # [num_tgt,1]
            batch_outer_logits.append(outer_logits)

        batch_center_logits = torch.cat(batch_center_logits,dim=0)
        batch_inner_logits = torch.cat(batch_inner_logits,dim=0)
        batch_outer_logits = torch.cat(batch_outer_logits,dim=0)
        
        loss_rank = rank_loss(batch_center_logits,batch_inner_logits,batch_outer_logits,margin=0.2)
            
        losses = {'loss_rank': loss_rank}

        return losses

    def loss_salient(self, outputs, targets, indices, num_boxes, epoch, log=True):
        """ L_ad
            Rank loss, for fine-grained boundary perception
            NOTE: If the num_queries is so small, can not cover all gt, an error will appear here
        """

        assert 'salient_logits' in outputs
        assert 'salient_loss_mask' in outputs
        assert 'salient_gt' or 'soft_salient_gt' in outputs
        salient_logits = outputs['salient_logits'] # [bs,t,1]
        salient_logits = salient_logits.squeeze(dim=2) # [bs,t]
        mask = outputs['salient_loss_mask'] # [bs,t]

        fpn_salient_logits = outputs['fpn_salient_logits'] # [bs,t,1]
        fpn_salient_logits = fpn_salient_logits.squeeze(dim=2) # [bs,t]
         

        if self.salient_loss_impl == "BCE":
            prob = salient_logits.sigmoid() # [bs,t]

            salient_gt = outputs['salient_gt'] # [bs,t]  
            ce_loss = F.binary_cross_entropy_with_logits(salient_logits, salient_gt, reduction="none")
            if self.enable_softSalient:
                ce_loss = ce_loss*(outputs['soft_salient_gt']**2)
                # ce_loss = ce_loss*(outputs['soft_salient_gt'])
            p_t = prob * salient_gt + (1 - prob) * (1 - salient_gt) # [bs,t]
            loss = ce_loss * ((1 - p_t) ** self.gamma)

            if self.focal_alpha >= 0:
                alpha_t = self.focal_alpha * salient_gt + (1 - self.focal_alpha) * (1 - salient_gt)
                loss = alpha_t * loss

            # if self.enable_softSalient and not self.as_calibration:
            #     salient_gt = outputs['soft_salient_gt'] # [bs,t]  
            #     # Binary Cross Entropy with logits (soft target)
            #     ce_loss = F.binary_cross_entropy_with_logits(salient_logits, salient_gt, reduction="none")  # [bs, t]

            #     # Optional: focal-style weighting for soft target
            #     if self.warmup:
            #         current_gamma = min(self.gamma, compute_exp_ramp_weight(epoch, max_epoch=10, target_weight=self.gamma))
            #         # current_gamma = min(self.gamma, self.gamma * epoch / self.warmup_epochs) # gamma warmup: 前期學 soft label，後期加強 focus
            #     else:
            #         current_gamma = self.gamma

            #     # Focal weight (recommended for soft labels)
            #     focal_weight = (salient_gt * (1 - prob) + (1 - salient_gt) * prob) ** current_gamma  # [bs, t]
            #     loss = ce_loss * focal_weight  # [bs, t]
            # else:
            #     salient_gt = outputs['salient_gt'] # [bs,t]  
            #     ce_loss = F.binary_cross_entropy_with_logits(salient_logits, salient_gt, reduction="none")
            #     ce_loss = ce_loss*(outputs['soft_salient_gt']**2)
            #     p_t = prob * salient_gt + (1 - prob) * (1 - salient_gt) # [bs,t]
            #     loss = ce_loss * ((1 - p_t) ** self.gamma)

            # if self.focal_alpha >= 0:
            #     salient_gt = outputs['salient_gt']
            #     alpha_t = self.focal_alpha * salient_gt + (1 - self.focal_alpha) * (1 - salient_gt)
            #     loss = alpha_t * loss

            # if self.as_calibration:
            #     w = outputs['soft_salient_gt']
            #     w_mean = w.mean(dim=1, keepdim=True)
            #     loss = (w/w_mean)**2 * loss
            #     # loss = (outputs['soft_salient_gt'])**2 * loss

            un_mask = ~mask
            loss_salient = loss*un_mask
            
            if self.enable_edgePunish and epoch>10:
                center = salient_gt
                left = torch.cat([salient_gt[:, :1], salient_gt[:, :-1]], dim=1)   # left shift, pad with first value
                right = torch.cat([salient_gt[:, 1:], salient_gt[:, -1:]], dim=1)  # right shift, pad with last value
                edge_mask = (center != left) | (center != right)
                punish_weight = compute_exp_ramp_weight(epoch)
                loss_salient[edge_mask] *= punish_weight
                # loss_salient[edge_mask] *= self.punish

            loss_salient = loss_salient.mean(1).sum() / num_boxes 
            
            # # --- Temporal Smoothing Loss (TV Loss) ---
            # # Compute differences between adjacent time steps
            # prob_shifted = prob[:, 1:]  # [bs, t-1]
            # prob_orig = prob[:, :-1]    # [bs, t-1]
            # tv_loss = torch.abs(prob_orig - prob_shifted)  # [bs, t-1]

            # # apply mask to ignore padded regions (optional)
            # if mask is not None:
            #     un_mask_tv = (~mask)[:, 1:] * (~mask)[:, :-1]
            #     tv_loss = tv_loss * un_mask_tv.float()

            # loss_tv = tv_loss.mean(1).sum() / num_boxes

            # # Combine both losses
            # lambda_tv = 0.1  # 可調參數：控制平滑損失的權重
            # total_loss = loss_salient + lambda_tv * loss_tv

            # -----------------------------------------------------------------
            fpn_prob = fpn_salient_logits.sigmoid() # [bs,t]
            fpn_ce_loss = F.binary_cross_entropy_with_logits(fpn_salient_logits, salient_gt, reduction="none")
            fpn_p_t = fpn_prob * salient_gt + (1 - fpn_prob) * (1 - salient_gt) # [bs,t]
            fpn_loss = fpn_ce_loss * ((1 - fpn_p_t) ** self.gamma)

            if self.focal_alpha >= 0:
                alpha_t = self.focal_alpha * salient_gt + (1 - self.focal_alpha) * (1 - salient_gt)
                fpn_loss = alpha_t * fpn_loss
            fpn_loss_salient = fpn_loss*un_mask
            fpn_loss_salient = fpn_loss_salient.mean(1).sum() / num_boxes 

            # fpn_loss_salient = 0

            all_loss_salient = 0.5*fpn_loss_salient + 1.5*loss_salient

        elif self.salient_loss_impl == "CE":

            salient_gt = salient_gt / (torch.sum(salient_gt, dim=1, keepdim=True) + 1e-4) # [b,t]

            loss_salient = -(salient_gt * F.log_softmax(salient_logits, dim=-1)) # [b,t]
            
            un_mask = ~mask
            loss_salient = loss_salient*un_mask
            loss_salient = loss_salient.sum(dim=1).mean()
        else:
            raise ValueError
 
        losses = {'loss_salient': all_loss_salient}

        return losses

    def loss_visual_consistency(self, outputs, targets, indices, num_boxes, log=True, temperature=0.05):
        """L_con
            For each activity, its visual features should remain consistent 
            across different temporal segments of the same video as well as acrosss different videos.
        Parameters:
            outputs (dict): 包含 "actionness_logits", "class_logits", "pred_boxes"
            targets (list): 包含每個目標的標籤和邊界框
            indices (list): 每個批次中的預測框和目標框的索引對
            temperature (float): 溫度參數，控制內積的縮放程度
        """
        zD = outputs['hs'][-1]                          # [b, num_quries, c]
        zD_normalized = F.normalize(zD, p=2, dim=-1)    # [b, num_quries, c]

        idx = self._get_src_permutation_idx(indices) # (batch_idx, query_idx)

        # matched_gt_class = {} 
        # keys = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)])
        # values = zD_normalized[idx]
        # for key, value in zip(keys, values):
        #     if key.item() not in matched_gt_class:
        #         matched_gt_class[key.item()] = []
        #     matched_gt_class[key.item()].append(value)

        matched_gt_class = defaultdict(list) # {key: value = class: matched zDi}
        for t, (_, J) in zip(targets, indices):
            for key, value in zip(t["semantic_labels"][J], zD_normalized[idx]):
                matched_gt_class[key.item()].append(value)

        
        # Example = {c1: [x1, x2, x3], c2:[x4, x5], c3:[x6]}
        loss_visual_consistency = 0
        for cls, embs in matched_gt_class.items():
            if len(embs) < 2:  # no other positive sample
                continue

            # Negative pairs: collect
            negative_pairs = []
            for other_cls, other_embs in matched_gt_class.items():
                if other_cls != cls:
                    negative_pairs.extend(other_embs)
            negative_pairs_tensor = torch.stack(negative_pairs)  # [num_negative, c]

            for i, emb1 in enumerate(embs):

                # Positive pairs: collect
                positive_pairs = [emb2 for j, emb2 in enumerate(embs) if i != j]
                positive_pairs_tensor = torch.stack(positive_pairs)  # [num_positives, c]

                # Positive pairs: compute
                positive_sim = torch.sum(torch.exp(torch.matmul(positive_pairs_tensor, emb1) / temperature))

                # Negative pairs: compute
                negative_sim = torch.sum(torch.exp(torch.matmul(negative_pairs_tensor, emb1) / temperature))

                # Denominator 
                denominator = positive_sim + negative_sim

                for emb2 in positive_pairs:
                    numerator = torch.exp(torch.dot(emb1, emb2) / temperature)
                    loss = -torch.log(numerator / denominator)
                    loss_visual_consistency += loss

        losses = {'loss_visual_consistency': loss_visual_consistency}
        return losses
    
    def loss_sparsity(self, outputs, targets, indices, num_boxes):
        assert 'attn_map' in outputs
        attn_map  = outputs['attn_map']

        attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-6)
        attn_map = attn_map.clamp(min=1e-6)  # 防止 log(0)
        entropy = - (attn_map * attn_map.log()).sum(dim=-1)  # [B, n_layers, n_heads]
        loss_sparsity = entropy.mean()
        losses = {'loss_sparsity': loss_sparsity}
        return losses

    def loss_alignment(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        assert 'actionness_logits' in outputs

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]
    
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        losses = {}
        eps = 1e-6
        # 只有 positive queries（被 assign 到 ground-truth 的）要這樣算 alignment loss
        iou = torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        actionness_score = torch.abs(1-src_logits.sigmoid()[idx]).squeeze(1)
        s = torch.clamp(actionness_score, min=eps, max=1.0)
        u = torch.clamp(iou, min=eps, max=1.0)
        loss_alignment = (s**(0.25) * u**(0.75))
        loss_alignment = loss_alignment.sum() / num_boxes
        losses = {'loss_alignment': loss_alignment}

        return losses

    def loss_semantic_guided(self, outputs, targets, indices, num_boxes, num_seen_classes=10):
        assert 'semantic_guided_logits' in outputs
        # Seen Class Supervision
        src_logits = outputs['semantic_guided_logits'] # [bs,num_queries,num_classes]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]

        target_seen = target_classes_onehot[..., :num_seen_classes][idx] # [batch_matched_queries, num_seen_classes]=[B', S]
        pred_seen = src_logits[..., :num_seen_classes][idx] # [B', S]

        ## 【Way1】
        # loss_seen = F.binary_cross_entropy_with_logits(pred_seen, target_seen, reduction="none").mean(1).sum()
        loss_seen = F.binary_cross_entropy_with_logits(pred_seen, target_seen, reduction="none").sum(1).mean()

        ## 【Way2: Focal Loss scaling】
        # gamma = 2.0
        # prob = torch.sigmoid(pred_seen)
        # pt = target_seen * prob + (1 - target_seen) * (1 - prob)
        # focal_weight = (1 - pt).pow(gamma)
        # loss_seen = F.binary_cross_entropy_with_logits(pred_seen, target_seen, reduction="none")
        # loss_seen = (focal_weight * loss_seen).sum(1).mean()

        ## 【Way3: Class-wise weighting】
        # flat_pred = pred_seen.view(-1, num_seen_classes)
        # flat_target = target_seen.view(-1, num_seen_classes)
        # # 計算每個 class 的正負樣本數量
        # pos_counts = flat_target.sum(dim=0)  # [num_classes]
        # neg_counts = flat_target.shape[0] - pos_counts
        # total_counts = pos_counts + neg_counts + 1e-6
        # # 計算 class-wise weight：越少樣本 → 權重越大
        # pos_weight = (neg_counts / total_counts).clamp(min=0.1, max=10.0)  # [num_classes]
        # loss_seen = F.binary_cross_entropy_with_logits(
        #     flat_pred, flat_target, pos_weight=pos_weight, reduction='mean'
        # )

        #-----------------------------------------
        # Unseen Class Regularization
        ## 【Way1】
        # target_unseen = target_classes_onehot[..., num_seen_classes:][idx]
        # pred_unseen = src_logits[..., num_seen_classes:][idx] # [batch_matched_queries, num_unseen_classes]=[B', U]
        # mu_weights = self.semantic_guided_unseen_weights[target_classes_o] # [B', U]
        # # loss_unseen = (mu_weights*F.binary_cross_entropy_with_logits(pred_unseen, target_unseen, reduction="none")).mean(1).sum()
        # loss_unseen = (mu_weights*F.binary_cross_entropy_with_logits(pred_unseen, target_unseen, reduction="none")).sum(1).mean()
        # lambda_unseen = 2
        ## 【Way2: seen(BCE), unseen(KL)】
        # mu_weights = self.semantic_guided_unseen_weights[target_classes_o] # [B', U]
        # pred_unseen = src_logits[..., num_seen_classes:][idx] # [B', U]
        # log_probs = F.log_softmax(pred_unseen, dim=1)  # log p_i 
        # # 計算 KL(q||p) = Sum(q_i * log(q_i /p_i) = Sum(q_i * (log q_i -log p_i))
        # loss_unseen_kl = F.kl_div(log_probs, mu_weights, reduction='batchmean')
        # loss_unseen = loss_unseen_kl
        # lambda_unseen = 50000
        ## 【Way3: seen(BCE), unseen(BCE + soft target)】
        # mu_weights = self.semantic_guided_unseen_weights[target_classes_o] # [B', U]
        # pred_unseen = src_logits[..., num_seen_classes:][idx] # [B', U]
        # loss_unseen = F.binary_cross_entropy_with_logits(pred_unseen, mu_weights, reduction='none').sum(1).mean()

        ## 【Way4: seen(BCE), unseen(KL) + loss balance】
        mu_weights = self.semantic_guided_unseen_weights[target_classes_o] # [B', U]
        pred_unseen = src_logits[..., num_seen_classes:][idx] # [B', U]
        log_probs = F.log_softmax(pred_unseen, dim=1)
        target_probs = mu_weights / (mu_weights.sum(dim=1, keepdim=True) + 1e-6)
        loss_unseen = F.kl_div(log_probs, target_probs, reduction='batchmean')
        
        attn = torch.sigmoid(src_logits[idx])
        attn_seen = attn[:, :num_seen_classes].sum(1)
        attn_unseen = attn[:, num_seen_classes:].sum(1)
        loss_balance = F.relu(attn_unseen - attn_seen).mean()

        # --- Combine ---
        lambda_unseen = 200.0
        lambda_balance = 2.0
        loss_semantic_guided = (
            loss_seen +
            lambda_unseen * loss_unseen +
            lambda_balance * loss_balance
        ) / num_boxes

        logger.info(f"Train Epoch: {self.epoch}, loss_seen={loss_seen}, loss_unseen={loss_unseen}")

        ## 【Way5: seen(KL), unseen(KL)】
        # mu_weights = self.semantic_guided_unseen_weights[target_classes_o] # [B', S+U]
        # target = mu_weights
        # log_pred = F.log_softmax(src_logits[idx], dim=1) # [B', S+U]
        # loss_kl = F.kl_div(log_pred, target, reduction='batchmean')
        # loss_semantic_guided = (loss_kl) / num_boxes
        

        # attn = torch.sigmoid(src_logits[idx])
        # attn_seen = attn[:, :num_seen_classes].sum(1)
        # attn_unseen = attn[:, num_seen_classes:].sum(1)
        # loss_balance = F.relu(attn_unseen - attn_seen).mean()

        # loss_semantic_guided = (loss_kl+2*loss_balance) / num_boxes
        

        # # Combine
        # lambda_unseen = 0.1
        # loss_semantic_guided = (loss_seen + lambda_unseen * loss_unseen) / num_boxes
        
        if self.batch_idx==0:
            plot_SAattention_map(src_logits[idx], gt_classes_o=target_classes_o, epoch=self.epoch, split_id=self.split_id)

        losses = {'loss_semantic_guided':loss_semantic_guided}
        return losses
        
        # loss_seen = -torch.sum(target_seen * F.log_softmax(pred_seen, dim=-1), dim=-1).mean()
        # loss_unseen = -torch.sum(mu_weights * target_unseen * F.log_softmax(pred_unseen, dim=-1), dim=-1).mean()
        # lambda_unseen = 0.1
        # loss_semantic_guided = (loss_seen + lambda_unseen * loss_unseen) / num_boxes 
    
    def loss_feedback(self, outputs, targets, indices, num_boxes):
        """
        Args:
            A_D: [num_layers, bs, num_queries, num_queries] - self-attn map
            G_D: [num_layers, bs, num_queries, num_queries] - guidance map

        Returns:
            loss: scalar KL divergence summed over decoder layers
        """
        assert 'G_D' in outputs
        assert 'A_D' in outputs

        # log softmax for pred (A_D), softmax for target (G_D)
        A_log = F.log_softmax(outputs['A_D'], dim=-1)      # [L, B, Q, Q]
        G_prob = F.softmax(outputs['G_D'], dim=-1)         # [L, B, Q, Q]
        # print("A_D max:", outputs['A_D'].max().item(), "min:", outputs['A_D'].min().item())
        # print("G_D max:", outputs['G_D'].max().item(), "min:", outputs['G_D'].min().item())

        # compute kl per layer → sum over layers & batch
        loss_feedback = F.kl_div(A_log, G_prob, reduction='batchmean')  # auto flatten [L*B*Q, Q]

        losses = {'loss_feedback':loss_feedback}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def plot_saliency_vs_bbox(self, outputs, targets, indices, epoch):
        

        def center_width_to_span(center, width, total_length):
            c = center * total_length
            w = width * total_length
            return c - w / 2, c + w / 2

        # Prepare salinet_logits
        salient_logits = outputs['salient_logits'] # [bs,t,1]
        salient_logits = salient_logits.squeeze(dim=2) # [bs,t]
        fpn_salient_logits = outputs['fpn_salient_logits'] # [bs,t,1]
        fpn_salient_logits = fpn_salient_logits.squeeze(dim=2) # [bs,t]
         
        logits_batch = salient_logits.sigmoid() # [bs,t]
        fpn_logits_batch = fpn_salient_logits.sigmoid() # [bs,t]

        # Prepare bbox
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        pred_bbox_batch = outputs['pred_boxes'][idx].detach().cpu().numpy() # [batch_matched_queries,2]
        gt_bbox_batch = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy() # [batch_target,2]

        B, T = logits_batch.shape  # batch size, timestamps
        # Group bboxes by batch_idx
        batch_idx, _ = idx
        pred_bbox_dict = defaultdict(list)
        gt_bbox_dict = defaultdict(list)
        for i, bid in enumerate(batch_idx.cpu().numpy()):
            pred_bbox_dict[bid].append(pred_bbox_batch[i])
            gt_bbox_dict[bid].append(gt_bbox_batch[i])

        for bid in range(B):
            logits_concat = logits_batch[bid].detach().cpu().numpy()
            fpn_logits_concat = fpn_logits_batch[bid].detach().cpu().numpy()

            plt.figure(figsize=(14, 3))
            plt.plot(np.arange(T), logits_concat, label="Salient Logits (enc)", color='blue')
            plt.plot(np.arange(T), fpn_logits_concat, label="Salient Logits (fpn)", color='green')
            bbox_label_drawn = False

            for pred_bbox, gt_bbox in zip(pred_bbox_dict[bid], gt_bbox_dict[bid]):
                start, end = center_width_to_span(pred_bbox[0], pred_bbox[1], T)
                if not bbox_label_drawn:
                    plt.axvspan(start, end, color='red', alpha=0.3, label='BBox Prediction')
                    bbox_label_drawn = True
                else:
                    plt.axvspan(start, end, color='red', alpha=0.3)
                # start, end = center_width_to_span(gt_bbox[0], gt_bbox[1], T)
                # plt.axvspan(start, end, color='green', alpha=0.2, label="Salient GT")
            
            # plt.title(f"Video {bid}: Salient Logits vs BBox Prediction")
            plt.xlabel("Timestep")
            plt.ylabel("Saliency Score")
            plt.xlim(0, T-1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'/home/yayun502/GAP/DeformableDETR/figures/Epoch_{epoch}_Video_{bid}.png')
            plt.close()
    
        

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes2 if self.tIoUw_bbox_loss else self.loss_boxes,
            'actionness': self.loss_actionness,
            'actionness_refine': self.loss_actionness_refine,
            'boxes_refine': self.loss_boxes_refine,
            'semantic_guided': self.loss_semantic_guided
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch, batch_idx=-1, semantic_guided_unseen_weights=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
            assert "actionness_logits" in outputs_without_aux
            chosen_logits_type = "actionness_logits"            
        else:
            assert "class_logits" in outputs_without_aux
            chosen_logits_type = "class_logits"
        indices = self.matcher(outputs_without_aux, targets, chosen_logits_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()

        # Compute all the requested losses
        if self.semantic_guided_loss:
            self.epoch = epoch
            self.batch_idx = batch_idx
            self.semantic_guided_unseen_weights = semantic_guided_unseen_weights
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each innermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, chosen_logits_type)
                for loss in self.losses:
                    if loss == 'masks':
                        # masks loss don't have innermediate feature of decoder, ignore it
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if self.salient_loss:
            salient_loss = self.loss_salient(outputs, targets, indices, num_boxes, epoch)
            losses.update(salient_loss)
        
        if self.vis_con_loss:
            consistency_loss = self.loss_visual_consistency(outputs, targets, indices, num_boxes)
            losses.update(consistency_loss)
        
        if self.sparsity_loss:
            sparsity_loss = self.loss_sparsity(outputs, targets, indices, num_boxes)
            losses.update(sparsity_loss)

        if self.alignment_loss:
            alignment_loss = self.loss_alignment(outputs, targets, indices, num_boxes)  
            losses.update(alignment_loss)
        
        if self.semantic_guided_loss and self.SA_LastOnly:
            semantic_guided_loss = self.loss_semantic_guided(outputs, targets, indices, num_boxes)
            losses.update(semantic_guided_loss)

        if self.feedback_loss:
            feedback_loss = self.loss_feedback(outputs, targets, indices, num_boxes)
            losses.update(feedback_loss)

        # draw: salient_logits vs bbox prediction
        if batch_idx==0:
            self.plot_saliency_vs_bbox(outputs, targets, indices, epoch)


        return losses

def build_criterion(args,num_classes,matcher,weight_dict, losses, focal_alpha, split_id):
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             losses=losses,
                             focal_alpha=focal_alpha,
                             split_id=split_id,
                             args=args)
    return criterion