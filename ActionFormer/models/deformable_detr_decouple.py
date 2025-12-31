import torch
import os
import numpy as np
from models.clip import build_text_encoder
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .criterion import build_criterion
from .postprocess import build_postprocess
from .refine_decoder import build_refine_decoder
import copy
import math
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,inverse_sigmoid)
from blocks import downsample
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign

import torch
import torch.nn.functional as F
from torch import nn

from utils.segment_ops import segment_cw_to_t1t2
from utils.dam import attn_map_to_flat_grid
from utils.draw import plot_attention_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    def __init__(self, 
                 backbone, 
                 transformer, 
                 text_encoder, 
                 refine_decoder,
                 logit_scale, 
                 device, 
                 num_classes,
                 args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.device = device
        self.num_classes = num_classes

        self.feature_type = args.feature_type
        self.feature_path = args.feature_path
        self.downsample_rate = args.downsample_rate
        self.separate_predict_head = args.separate_predict_head
        self.num_feature_levels = args.num_feature_levels
        self.enc_stem_layers = args.enc_stem_layers
        self.enc_branch_layers = args.enc_branch_layers
        self.salient_aggregate_type = args.salient_aggregate_type
        self.salient_upsample_type = args.salient_upsample_type
        self.vis_con_loss = args.vis_con_loss
        self.use_SAPM = args.use_SAPM
        self.use_SAattn = args.use_SAattn
        self.enable_softSalient = args.enable_softSalient
        self.as_calibration = args.as_calibration
        self.use_decouple = args.use_decouple
        self.visualize_decouple = args.visualize_decouple

        self.num_queries = args.num_queries
        self.num_feature_levels = args.num_feature_levels
        self.aux_loss = args.aux_loss, 
        self.with_box_refine = args.with_box_refine, 
        self.two_stage = args.two_stage

        self.target_type = args.target_type
        self.aux_loss = args.aux_loss
        self.norm_embed = args.norm_embed
        self.exp_logit_scale = args.exp_logit_scale
        self.ROIalign_strategy = args.ROIalign_strategy
        self.ROIalign_size = args.ROIalign_size
        self.pooling_type = args.pooling_type
        self.eval_proposal = args.eval_proposal 
        self.actionness_loss = args.actionness_loss
        self.enable_classAgnostic = args.enable_classAgnostic
        self.enable_refine = args.enable_refine
        self.enable_posPrior = args.enable_posPrior
        self.enable_freqCalibrate = args.enable_freqCalibrate
        self.enable_refine_freq = args.enable_refine_freq
        self.salient_loss = args.salient_loss
        hidden_dim = transformer.d_model
        
        ## ----------------------------[1] BBOX EMBED-------------------------------------------
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        ## ----------------------------[2] CLASS EMBED-------------------------------------------
        ## Follow Deformable DETR => no consideration about align with text_features
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        ## Follow GAP (Conditional DETR) => consideration about align with text_features
        if self.target_type != "none":
            self.class_embed = nn.Linear(hidden_dim, hidden_dim)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(hidden_dim) * bias_value
        else:
            if not self.enable_classAgnostic or not self.eval_proposal:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
                # init prior_prob setting for focal loss
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        ## --------------------------[3] ACTIONNESS EMBED----------------------------------------
        if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
            self.actionness_embed = nn.Linear(hidden_dim,1)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.actionness_embed.bias.data = torch.ones(1) * bias_value

        ## --------------------------[4] SALIENT EMBED-------------------------------------------
        if self.salient_loss:
            # self.salient_head = nn.Sequential(
            #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #     nn.LeakyReLU(0.2),
            #     nn.Conv1d(hidden_dim, 1, kernel_size=1)
            # )
            # self.salient_heads = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #         nn.LeakyReLU(0.2),
            #         nn.Conv1d(hidden_dim, 1, kernel_size=1)
            #     ) for _ in range(self.num_feature_levels)
            # ])
            self.salient_embed = MLP(hidden_dim*self.num_feature_levels, hidden_dim, 1, 2)
        if self.salient_aggregate_type == 'mix_adaptive':
            self.salient_weights = nn.Parameter(torch.ones(self.num_feature_levels)) 
        else:
            self.salient_weights = None

        ## -----------------------------SEPARATE PREDICT HEAD------------------------------------
        ## Follow Deformable DETR
        if self.separate_predict_head: 
            # if two-stage, the last class_embed and bbox_embed is for region proposal generation
            num_pred = (transformer.decoder.num_layers + 1) if self.two_stage else transformer.decoder.num_layers
            if self.with_box_refine:
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.actionness_embed = _get_clones(self.actionness_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
                nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[1:], -2.0)
                # hack implementation for iterative bounding box refinement
                self.transformer.decoder.bbox_embed = self.bbox_embed
            else:
                nn.init.constant_(self.bbox_embed.layers[-1].bias.data[1:], -2.0)
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.actionness_embed = nn.ModuleList([self.actionness_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
                self.transformer.decoder.bbox_embed = None
            if self.two_stage:
                # hack implementation for two-stage
                self.transformer.decoder.class_embed = self.class_embed
                for box_embed in self.bbox_embed:
                    nn.init.constant_(box_embed.layers[-1].bias.data[1:], 0.0)
        else:
            # align with what are done in the above
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[1:], -2.0)
            self.transformer.decoder.bbox_embed = None

        ## ----------------------------------- OTHERS --------------------------------------------
        ## OTHER: GAP(Conditional DETR)
        if self.enable_refine or self.enable_refine_freq:
            self.refine_decoder = refine_decoder
        # if self.enable_posPrior:
        #     self.query_embed = nn.Embedding(self.num_queries,1)
        #     self.query_embed.weight.data[:, :1].uniform_(0, 1)
        #     self.query_embed.weight.data[:, :1] = inverse_sigmoid(self.query_embed.weight.data[:, :1])
        #     self.query_embed.weight.data[:, :1].requires_grad = False
        # else:
        #     self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        ## OTHER: Deformable DETR
        if not self.two_stage:
            if self.use_SAPM:
                self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
            else:
                if not self.use_decouple:
                    self.query_embed = nn.Embedding(self.num_queries, hidden_dim*2)
                else:
                    self.query_embed = nn.Embedding(self.num_queries*2, hidden_dim*2)
        
    

    def get_text_feats(self, cl_names, description_dict, device, target_type):
        def get_prompt(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append("a video of a person doing"+" "+c)
            return temp_prompt
        
        def get_description(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # NOTE: default the idx of description is 0.
            return temp_prompt
        
        def get_combined_descriptions(cl_names):
            temp_prompt = []
            for c in cl_names:
                sub_actions = []
                for sub_action in description_dict[c]['Elaboration']['Description']:
                    sub_actions.append("a video of a person doing"+" "+sub_action)
                temp_prompt.append(sub_actions) # NOTE: default the idx of description is 0.
            return temp_prompt
        
        if target_type == 'prompt':
            act_prompt = get_prompt(cl_names)
        elif target_type == 'description':
            act_prompt = get_description(cl_names)
        elif target_type == 'combined_description':
            whole_act_prompt = get_prompt(cl_names)
            sub_act_prompt = get_combined_descriptions(cl_names)
        elif target_type == 'name':
            act_prompt = cl_names
        else: 
            raise ValueError("Don't define this text_mode.")
        
        if target_type == 'combined_description':
            whole_text_feats = []
            whole_tokens = clip_pkg.tokenize(whole_act_prompt).long().to(device)
            whole_text_feats = self.text_encoder(whole_tokens).float() 

            sub_text_feats_list = []
            for prompts in sub_act_prompt: 
                tokens = clip_pkg.tokenize(prompts).long().to(device) 
                text_feats = self.text_encoder(tokens).float() 
                sub_text_feats_list.append(text_feats.mean(dim=0))
            sub_text_feats = torch.stack(sub_text_feats_list)

            text_feats = (whole_text_feats + sub_text_feats) / 2  
        else:
            tokens = clip_pkg.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
            text_feats = self.text_encoder(tokens).float()

        return text_feats
    

    def _to_roi_align_format(self, rois, truely_length, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1] # [B,N,1]
        rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * truely_length # [B,N,2]->"start,end"
        # expand the RoIs
        _max = truely_length.repeat(1,N,2)
        _min = torch.zeros_like(_max)
        rois_abs = torch.clamp(rois_abs, min=_min, max=_max)  # (B, N, 2)
        # transfer to 4 dimension coordination
        rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
        rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
        batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
        rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
        # NOTE: stop gradient here to stablize training
        return rois_abs_4d.view((B*N, 5)).detach()


    def _roi_align(self, rois, origin_feat, mask, ROIalign_size, scale_factor=1):
        B,Q,_ = rois.shape
        B,T,C = origin_feat.shape
        truely_length = T-torch.sum(mask,dim=1) # [B]
        rois_abs_4d = self._to_roi_align_format(rois,truely_length,scale_factor)
        feat = origin_feat.permute(0,2,1) # [B,dim,T]
        feat = feat.reshape(B,C,1,T)
        roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,ROIalign_size))
        roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,dim,output_width]
        roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,dim]
        return roi_feat


    # @torch.no_grad()
    def _compute_similarity(self, visual_feats, text_feats):
        '''
        text_feats: [num_classes,dim]
        '''
        if len(visual_feats.shape)==2: # batch_num_instance,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bd,cd->bc",visual_feats,text_feats)*logit_scale
            else:
                logits = torch.einsum("bd,cd->bc",visual_feats,text_feats)
            return logits
        elif len(visual_feats.shape)==3:# batch,num_queries/snippet_length,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)*logit_scale
            else:
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)
            return logits
        elif len(visual_feats.shape)==4:# batch,num_queries,snippet_length,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqld,cd->bqlc",visual_feats,text_feats)*logit_scale
            else:
                logits = torch.einsum("bqld,cd->bqlc",visual_feats,text_feats)
            return logits
        
        else:
            raise NotImplementedError


    def _temporal_pooling(self,pooling_type,coordinate,clip_feat,mask,ROIalign_size,text_feats):
        b,t,_ = coordinate.shape
        if pooling_type == "average":
            roi_feat = self._roi_align(rois=coordinate,origin_feat=clip_feat+1e-4,mask=mask,ROIalign_size=ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            # roi_feat = roi_feat.mean(-2) # [B,Q,dim]
            if self.ROIalign_strategy == "before_pred":
                roi_feat = roi_feat.mean(-2) # [B,Q,dim]
                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,num_classes]
            elif self.ROIalign_strategy == "after_pred":
                roi_feat = roi_feat # [B,Q,L,dim]
                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,L,num_classes]
                ROIalign_logits = ROIalign_logits.mean(-2) # [B,Q,num_classes]
            else:
                raise NotImplementedError
        elif pooling_type == "max":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            roi_feat = roi_feat.max(dim=2)[0] # [bs,num_queries,dim]

            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "center1":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            center_idx = int(roi_feat.shape[2] / 2)
            roi_feat = roi_feat[:,:,center_idx,:] 
            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "center2":
            rois = coordinate # [b,n,2]
            rois_center = rois[:, :, 0:1] # [B,N,1]
            # rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
            truely_length = t-torch.sum(mask,dim=1) # [B]
            truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
            center_idx = (rois_center*truely_length).long() # [b,n,1]
            roi_feat = torch.gather(clip_feat + 1e-4, dim=1, index=center_idx.expand(-1, -1, clip_feat.shape[-1]))
            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "self_attention":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            attention_weights = F.softmax(torch.matmul(roi_feat, roi_feat.transpose(-2, -1)), dim=-1)
            roi_feat_sa = torch.matmul(attention_weights, roi_feat)
            roi_feat_sa = roi_feat_sa.mean(2)
            ROIalign_logits = self._compute_similarity(roi_feat_sa,text_feats)
        elif pooling_type == "slow_fast":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            fast_feat = roi_feat.mean(dim=2) # [b,q,d]
            step = int(self.ROIalign_size // 4)
            slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
            roi_feat_final = (fast_feat + slow_feat)/2
            ROIalign_logits = self._compute_similarity(roi_feat_final,text_feats)
        elif pooling_type == "sparse":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            step = int(self.ROIalign_size // 4)
            slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
            ROIalign_logits = self._compute_similarity(slow_feat,text_feats)
        else:
            raise ValueError

        return ROIalign_logits   


    def batch_find_action_segments(self, salient_gt_batch):
        """
        salient_gt_batch: [bs, t] tensor，binary mask
        return: List[ List[ (start_idx, end_idx) ] ] 長度為 batch size
        """
        bs, t = salient_gt_batch.shape
        padded = F.pad(salient_gt_batch, (1, 1), value=0)  # [bs, t+2]
        left_shifted = padded[:, :-2]  # [bs, t]
        right_shifted = padded[:, 2:]  # [bs, t]

        is_start = (salient_gt_batch == 1) & (left_shifted == 0)  # [bs, t]
        is_end   = (salient_gt_batch == 1) & (right_shifted == 0)  # [bs, t]

        segments_per_batch = []

        for b in range(bs):
            start_idxs = is_start[b].nonzero(as_tuple=False).squeeze(1)
            end_idxs = is_end[b].nonzero(as_tuple=False).squeeze(1)
            segments = list(zip(start_idxs.tolist(), (end_idxs + 1).tolist()))  # [start, end)
            segments_per_batch.append(segments)

        return segments_per_batch  # List of lists of (start, end)


    def compute_soft_salient_gt(self, salient_gt, semantic_labels, video_feats, label_vocab, bg_text_feat=None, as_calibration=False):
        """
        salient_gt: [bs, t] binary mask
        semantic_labels: List[List[str]] 長度為 batch，每個元素是該 batch 的 label list
        video_feats: [bs, t, c] CLIP video features (with c = 512)
        label_vocab: Dict[label_name] -> CLIP encoded tensor [512]
        segments_per_batch: List of (start_idx, end_idx)
        """
        bs, t, c = video_feats.shape
        if as_calibration:
            soft_gt = torch.ones_like(salient_gt, dtype=video_feats.dtype)
        else:
            soft_gt = torch.zeros_like(salient_gt, dtype=video_feats.dtype)
        segments_per_batch = self.batch_find_action_segments(salient_gt)

        for b in range(bs):
            for idx, (start, end) in enumerate(segments_per_batch[b]):
                label = semantic_labels[b][idx]
                text_feat = label_vocab[label] 

                video_seg = video_feats[b, start:end]  # [seg_len, c]

                # cosine similarity
                video_seg = video_seg / video_seg.norm(dim=-1, keepdim=True)     # [T, C]
                text_feat = text_feat / text_feat.norm()                            # [C]
                sim = torch.matmul(video_seg, text_feat)                            # [T]

                if bg_text_feat == None:
                    # normalize similarity 
                    sim_norm = (sim) / (sim.max() + 1e-6)
                else:
                    bg_text_feat = bg_text_feat / bg_text_feat.norm()                   # [C]
                    sim_bg = torch.matmul(video_seg, bg_text_feat)                      # [T]
                    # contrastive max-min normalize similarity
                    sim_norm = (sim-sim_bg.min())/(sim.max()-sim_bg.min()+1e-6)

                soft_gt[b, start:end] = sim_norm

        return soft_gt  # [bs, t]


    def compute_soft_salient_gt_efficient(self, salient_gt, semantic_labels, video_feats, label_vocab, bg_text_feat=None, as_calibration=False):
        bs, t, c = video_feats.shape
        if as_calibration:
            soft_gt = torch.ones_like(salient_gt, dtype=video_feats.dtype)
        else:
            soft_gt = torch.zeros_like(salient_gt, dtype=video_feats.dtype)
        segments_per_batch = self.batch_find_action_segments(salient_gt)

        video_segs = []     # List of [seg_len, c]
        text_feats = []     # List of [c]
        batch_indices = []  # (b, start, end)
        lengths = []        # for padding

        for b in range(bs):
            for idx, (start, end) in enumerate(segments_per_batch[b]):
                label = semantic_labels[b][idx]
                text_feat = label_vocab[label]
                video_seg = video_feats[b, start:end]

                video_segs.append(video_seg)
                text_feats.append(text_feat)
                batch_indices.append((b, start, end))
                lengths.append(end - start)

        # Padding to max length
        max_len = max(lengths)
        padded_segs = torch.zeros((len(video_segs), max_len, c), device=video_feats.device)
        mask = torch.zeros((len(video_segs), max_len), dtype=torch.bool, device=video_feats.device)

        for i, seg in enumerate(video_segs):
            padded_segs[i, :seg.shape[0]] = seg
            mask[i, :seg.shape[0]] = 1

        # Normalize
        padded_segs = F.normalize(padded_segs, dim=-1)             # [N, L, C]
        text_feats = torch.stack(text_feats)                       # [N, C]
        text_feats = F.normalize(text_feats, dim=-1).unsqueeze(1)  # [N, 1, C]

        # Cosine similarity
        sim = F.cosine_similarity(padded_segs, text_feats, dim=-1)  # [N, L]

        if bg_text_feat is not None:
            bg_feat = F.normalize(bg_text_feat, dim=0).unsqueeze(0).unsqueeze(0)  # [1,1,C]
            sim_bg = F.cosine_similarity(padded_segs, bg_feat, dim=-1)            # [N, L]

            # contrastive normalize: (sim - min_bg) / (max_fg - min_bg)
            sim_fg_max = sim.masked_fill(~mask, float('-inf')).max(dim=1, keepdim=True)[0]
            sim_bg_min = sim_bg.masked_fill(~mask, float('inf')).min(dim=1, keepdim=True)[0]
            sim_norm = (sim - sim_bg_min) / (sim_fg_max - sim_bg_min + 1e-6)
        else:
            # simple normalize: sim / max
            sim_max = sim.masked_fill(~mask, float('-inf')).max(dim=1, keepdim=True)[0] + 1e-6
            sim_norm = sim / sim_max

        sim_norm = sim_norm.clamp(min=0.0, max=1.0)

        # Scatter back to soft_gt
        for i, (b, start, end) in enumerate(batch_indices):
            soft_gt[b, start:end] = sim_norm[i, :end - start]

        return soft_gt


    def forward(self, samples: NestedTensor, classes_name, description_dict, targets, epoch, batch_idx=None, pretrain=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # origin CLIP features
        clip_feat, original_mask = samples.decompose()
        bs, t, c = clip_feat.shape

        features, pos = self.backbone(samples) # [[b, c, t], [b, c, t/2], ...], [[b, c, t], [b, c, t/2], ...]

        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                if self.feature_type == "ViFi-CLIP":
                    text_feats = torch.from_numpy(np.load(os.path.join(self.feature_path,'text_features_split75_splitID1.npy'))).float().to(self.device)
                elif self.feature_type == "CLIP":
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]
                else:
                    raise NotImplementedError

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src, mask = downsample(features[-1].tensors, features[-1].mask.unsqueeze(1), self.downsample_rate) # [b, c, t/(2**l)], [b, 1, t/(2**l)]
                    mask = mask.squeeze(1) # [b, t/(2**l)]
                else:
                    src, mask = downsample(srcs[-1].tensors, masks[-1].mask.unsqueeze(1), self.downsample_rate) # [b, c, t/(2**l)], [b, 1, t/(2**l)]
                    mask = mask.squeeze(1) # [b, t/(2**l)]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        '''
        srcs = list:[b, c, t], [b, c, t/2], ...
        masks = list:[b, t], [b, t/2], ...
        pos = list:[b, c, t], [b, c, t/2], ...
        query_embeds = [40, 512(GAP)/1024(Deformable DETR)]
        ---------------------------------------------------
        hs = [dec_layers, b, num_quries, c]
        init_reference = [b, num_quries, 1]
        inter_references = [num_decoder_layers, b, num_quries, 2]
        enc_outputs_class = None
        enc_outputs_coord_unact = None
        '''
        semantic_descriptions = None
        if self.use_SAattn:
            semantic_descriptions = text_feats
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, salient_preparation, sparse_preparation = self.transformer(srcs, masks, pos, query_embeds, weights=self.salient_weights, semantic_descriptions=semantic_descriptions, pretrain=pretrain)

        
        # record result
        out = {}
        memory, level_start_index = salient_preparation
        
        out['memory'] = memory  # [b, sum(t), c]
        out['hs'] = hs

        if self.visualize_decouple and batch_idx==0:
            if not self.use_decouple:
                temporal_shapes, sampling_locations_dec, attn_weights_dec = sparse_preparation
                flat_grid_attn_map_dec = attn_map_to_flat_grid(
                    temporal_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))
                plot_attention_map(flat_grid_attn_map_dec, level_start_index, decouple_type=None, epoch=epoch)
            else:
                temporal_shapes, sampling_locations_dec_cls, sampling_locations_dec_loc, attn_weights_dec_cls, attn_weights_dec_loc = sparse_preparation
                flat_grid_attn_map_dec_cls = attn_map_to_flat_grid(
                    temporal_shapes, level_start_index, sampling_locations_dec_cls, attn_weights_dec_cls).sum(dim=(1,2))
                flat_grid_attn_map_dec_loc = attn_map_to_flat_grid(
                    temporal_shapes, level_start_index, sampling_locations_dec_loc, attn_weights_dec_loc).sum(dim=(1,2))
                plot_attention_map(flat_grid_attn_map_dec_cls, level_start_index, decouple_type='cls', epoch=epoch)
                plot_attention_map(flat_grid_attn_map_dec_loc, level_start_index, decouple_type='loc', epoch=epoch)


        #########################
        tokens = clip_pkg.tokenize("background").long().to(self.device) # input_ids->input_ids:[150,length]
        bg_text_feat = self.text_encoder(tokens).float().squeeze(0)
        #########################

        # generate the salient gt
        if self.salient_loss:
            if self.training: # only generate gt in training phase
                salient_gt = torch.zeros((bs,t),device=self.device) # [bs,t]
                salient_loss_mask = original_mask.clone() # [bs,t]

                for i, tgt in enumerate(targets):
                    salient_mask = tgt['salient_mask'] # [num_tgt,T]
                    # padding the salient mask
                    num_to_pad = t - salient_mask.shape[1]
                    if num_to_pad > 0:
                        padding = torch.ones((salient_mask.shape[0], num_to_pad), dtype=torch.bool, device=salient_mask.device)
                        salient_mask = torch.cat((salient_mask, padding), dim=1)

                    for salient_mask_j in salient_mask:
                        salient_gt[i,:] = (salient_gt[i,:] + (~salient_mask_j).float()).clamp(0,1)
                if self.enable_softSalient:
                    class_text_feat_dict = {name: feat for name, feat in zip(classes_name, text_feats)}
                    semantic_labels = [target['semantic_label_names'] for target in targets]
                    soft_salient_gt = self.compute_soft_salient_gt_efficient(salient_gt, semantic_labels, clip_feat, class_text_feat_dict, as_calibration=self.as_calibration)
                    # soft_salient_gt = self.compute_soft_salient_gt_efficient(salient_gt, semantic_labels, clip_feat, class_text_feat_dict, bg_text_feat=bg_text_feat, as_calibration=self.as_calibration)
                    out['soft_salient_gt'] = soft_salient_gt
                # if not(self.enable_softSalient and not self.as_calibration):
                #     out['salient_gt'] = salient_gt
                out['salient_gt'] = salient_gt
                out['salient_loss_mask'] = salient_loss_mask

            # # compute salient_logits for different scales
            # salient_logits_list = []
            # for lvl in range(self.num_feature_levels):
            #     if lvl == self.num_feature_levels-1:
            #         salient_logits = self.salient_heads[lvl](memory[:, level_start_index[lvl]: , :].permute(0,2,1)).permute(0,2,1)                            # [b, t/(2**lvl), 1]
            #     else:
            #         salient_logits = self.salient_heads[lvl](memory[:, level_start_index[lvl]:level_start_index[lvl+1] , :].permute(0,2,1)).permute(0,2,1)    # [b, t/(2**lvl), 1]
            #     salient_logits_list.append(salient_logits)
            # # upsample for different scales back to [b, t, 1]
            # if self.salient_upsample_type == 'nearest':
            #     salient_logits_upsampled = [
            #         salient if lvl < self.enc_stem_layers else F.interpolate(salient.transpose(1, 2), size=(t,), mode='nearest').transpose(1, 2)
            #         for lvl, salient in enumerate(salient_logits_list)
            #     ]
            # else:
            #     salient_logits_upsampled = [
            #         salient if lvl < self.enc_stem_layers else F.interpolate(salient.transpose(1, 2), size=(t,), mode='linear', align_corners=False).transpose(1, 2)
            #         for lvl, salient in enumerate(salient_logits_list)
            #     ]
            # # aggregate for different scales
            # if self.salient_aggregate_type == 'max':
            #     salient_logits = torch.max(torch.stack(salient_logits_upsampled, dim=0), dim=0)[0]
            # elif self.salient_aggregate_type == 'mean':
            #     salient_logits = torch.mean(torch.stack(salient_logits_upsampled, dim=0), dim=0)
            # elif self.salient_aggregate_type == 'mix':
            #     salient_logits = 0.5 * torch.mean(torch.stack(salient_logits_upsampled, dim=0), dim=0) + \
            #                      0.5 * torch.max(torch.stack(salient_logits_upsampled, dim=0), dim=0)[0]
            # else:
            #     # w_normalized = torch.softmax(self.salient_weights, dim=0)
            #     w_normalized = self.salient_weights / self.salient_weights.sum()
            #     salient_logits = sum(w_normalized[i] * salient_logits_upsampled[i] for i in range(self.num_feature_levels))
            multi_scale_feats = []
            for lvl in range(self.num_feature_levels):
                if lvl == self.num_feature_levels - 1:
                    feat = memory[:, level_start_index[lvl]:, :]  # [b, t_i, c]
                else:
                    feat = memory[:, level_start_index[lvl]:level_start_index[lvl+1], :]  # [b, t_i, c]

                # 2. upsample to [b, t, c] 使得所有尺度都對齊成原始 temporal resolution
                if lvl < self.enc_stem_layers:
                    upsampled_feat = feat  # 保留原始 input 不上采樣
                else:
                    if self.salient_upsample_type == 'nearest':
                        upsampled_feat = F.interpolate(feat.transpose(1, 2), size=(t,), mode='nearest').transpose(1, 2)
                    else:
                        upsampled_feat = F.interpolate(feat.transpose(1, 2), size=(t,), mode='linear', align_corners=False).transpose(1, 2)
                
                multi_scale_feats.append(upsampled_feat)  # [b, t, c]

            # 3. concat along channel dim: [b, t, c * num_feature_levels]
            fused_feat = torch.cat(multi_scale_feats, dim=-1)

            # 4. pass through MLP to get salient_logits: [b, t, 1]
            salient_logits = self.salient_embed(fused_feat)
            out['salient_logits'] = salient_logits

        if pretrain:
            return out

        # # refine encoder
        # if self.enable_refine:
            
        #     # First, predict once for RoI (stop gradient)
        #     with torch.no_grad():
        #         reference_before_sigmoid = inverse_sigmoid(reference)   # [b,num_queries,1], Reference point is the predicted center point.
        #         tmp = self.bbox_embed(hs[-1])                           # [b,num_queries,2], tmp is the predicted offset value.
        #         tmp[..., :1] += reference_before_sigmoid                # [b,num_queries,2], only the center coordination add reference point
        #         outputs_coord = tmp.sigmoid()                           # [b,num_queries,2]
        #         roi_pos = self._roi_align(outputs_coord,pos[-1],original_mask,self.ROIalign_size)    # [bs,num_queries,ROIalign_size,dim]
        #         roi_feat = self._roi_align(outputs_coord,clip_feat,original_mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]

        #     b,q,l,d = roi_feat.shape
        #     refine_hs = self.refine_decoder(hs[-1],clip_feat,roi_feat,
        #                             video_feat_key_padding_mask=original_mask,
        #                             video_pos=pos[-1],
        #                             roi_pos=roi_pos)
            
        #     # Second, predict for output (gradient)
        #     refine_hs = hs[-1] + refine_hs
        #     reference_before_sigmoid = inverse_sigmoid(reference)   # [b,num_queries,1], Reference point is the predicted center point.
        #     tmp = self.bbox_embed(refine_hs)                        # [b,num_queries,2], tmp is the predicted offset value.
        #     tmp[..., :1] += reference_before_sigmoid                # [b,num_queries,2], only the center coordination add reference point
        #     outputs_coord_refined = tmp.sigmoid()                   # [b,num_queries,2]
        #     out['pred_boxes'] = outputs_coord_refined

        outputs_classes = []
        outputs_actionnesses = []
        outputs_coords = []

        if self.enable_refine or self.enable_refine_freq:

            # First, predict once for RoI (stop gradient)
            roi_poss = []
            roi_feats = []
            with torch.no_grad():
                for lvl in range(hs.shape[0]):
                    if lvl == 0:
                        reference = init_reference
                    else:
                        reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    tmp = self.bbox_embed[lvl](hs[lvl])
                    if reference.shape[-1] == 2:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 1
                        tmp[..., :1] += reference
                    outputs_coord = tmp.sigmoid()
                    roi_pos = self._roi_align(outputs_coord,pos[-1],original_mask,self.ROIalign_size)    # [bs,num_queries,ROIalign_size,dim]
                    roi_feat = self._roi_align(outputs_coord,clip_feat,original_mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]

                    # if self.enable_refine_freq:
                    #     Z = torch.fft.fft(roi_feat.cpu(), dim=2).to(clip_feat.device)  # 沿時間維度做 FFT
                    #     roi_feat = torch.abs(Z)**2 # replace original ones from clip features

                    roi_poss.append(roi_pos)
                    roi_feats.append(roi_feat)

            # Second, predict for output (gradient)
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)

                refine_hs = self.refine_decoder(hs[lvl],clip_feat,roi_feats[lvl],
                                        video_feat_key_padding_mask=original_mask,
                                        video_pos=pos[-1],
                                        roi_pos=roi_pos[lvl])
                refine_hs = hs[lvl] + refine_hs
                # -------------------------------------- Prediction Heads ------------------------------------------------
                if self.separate_predict_head:
                    if self.target_type != "none":
                        class_emb = self.class_embed[lvl](refine_hs)                        # [b, num_queries, num_classes]
                        outputs_class = self._compute_similarity(class_emb, text_feats)     # [b, num_queries, num_classes]
                    else:
                        outputs_class = self.class_embed[lvl](refine_hs)                    # [b, num_queries, num_classes]
                    outputs_actionness = self.actionness_embed[lvl](refine_hs)              # [b,num_queries,1]
                    tmp = self.bbox_embed[lvl](refine_hs)                                   # [b,num_queries,2]
                else:
                    if self.target_type != "none":
                        class_emb = self.class_embed(refine_hs)                             # [b, num_queries, num_classes]
                        outputs_class = self._compute_similarity(class_emb, text_feats)     # [b, num_queries, num_classes]
                    else:
                        outputs_class = self.class_embed(refine_hs)                         # [b, num_queries, num_classes]
                    outputs_actionness = self.actionness_embed(refine_hs)                   # [b,num_queries,1]
                    tmp = self.bbox_embed(refine_hs)                                        # [b,num_queries,2]
                # ---------------------------------------------------------------------------------------------------------
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()               
                outputs_classes.append(outputs_class)
                outputs_actionnesses.append(outputs_actionness)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)                # [dec_layers, b, num_queries, num_classes]
            outputs_actionness = torch.stack(outputs_actionnesses)      # [dec_layers, b, num_queries, 1]
            outputs_coord = torch.stack(outputs_coords)                 # [dec_layers, b, num_queries, 2]

        else:
            if not self.use_decouple:
                hs_cls = hs
                hs_loc = hs
                inter_references_loc = inter_references
            else:
                hs_cls = hs[:, :, :self.num_queries, :]
                hs_loc = hs[:, :, self.num_queries:, :]
                inter_references_loc = inter_references[:, :, self.num_queries:, :]
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references_loc[lvl - 1]
                reference = inverse_sigmoid(reference)
                # -------------------------------------- Prediction Heads ------------------------------------------------
                if self.separate_predict_head:
                    if self.target_type != "none":
                        class_emb = self.class_embed[lvl](hs_cls[lvl])                          # [b, num_queries, num_classes]
                        outputs_class = self._compute_similarity(class_emb, text_feats)     # [b, num_queries, num_classes]
                    else:
                        outputs_class = self.class_embed[lvl](hs_cls[lvl])                      # [b, num_queries, num_classes]
                    outputs_actionness = self.actionness_embed[lvl](hs_cls[lvl])                # [b,num_queries,1]
                    tmp = self.bbox_embed[lvl](hs_loc[lvl])                                     # [b,num_queries,2]
                else:
                    if self.target_type != "none":
                        class_emb = self.class_embed(hs_cls[lvl])                               # [b, num_queries, num_classes]
                        outputs_class = self._compute_similarity(class_emb, text_feats)     # [b, num_queries, num_classes]
                    else:
                        outputs_class = self.class_embed(hs_cls[lvl])                           # [b, num_queries, num_classes]
                    outputs_actionness = self.actionness_embed(hs_cls[lvl])                     # [b,num_queries,1]
                    tmp = self.bbox_embed(hs_loc[lvl])                                          # [b,num_queries,2]
                # ---------------------------------------------------------------------------------------------------------
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()               
                outputs_classes.append(outputs_class)
                outputs_actionnesses.append(outputs_actionness)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)                # [dec_layers, b, num_queries, num_classes]
            outputs_actionness = torch.stack(outputs_actionnesses)      # [dec_layers, b, num_queries, 1]
            outputs_coord = torch.stack(outputs_coords)                 # [dec_layers, b, num_queries, 2]

        out['pred_boxes'] = outputs_coord[-1]
        if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
            # compute the class-agnostic foreground score
            out['actionness_logits'] = outputs_actionness[-1]
        if not self.eval_proposal and not self.enable_classAgnostic:
            out['class_logits'] = outputs_class[-1]

        if self.aux_loss:
            if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
                chosen_logits_type = 'actionness_logits'
                out['aux_outputs'] = self._set_aux_loss(outputs_actionness, outputs_coord, chosen_logits_type)
            if not self.eval_proposal and not self.enable_classAgnostic:
                chosen_logits_type = 'class_logits'
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, chosen_logits_type)
        
        if self.enable_freqCalibrate:
            out['fft_magnitudes'] = compute_fft_magnitude_v3(clip_feat, out['pred_boxes'])

        # if self.two_stage:
        #     enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        #     out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # obtain the ROIalign logits
        if not self.training: # only in inference stage
            if self.enable_classAgnostic:
                ROIalign_logits = self._temporal_pooling(self.pooling_type, out['pred_boxes'], clip_feat, original_mask, self.ROIalign_size, text_feats)
                out['class_logits'] = ROIalign_logits 
            elif self.eval_proposal:
                pass
            else:
                assert "class_logits" in out, "please check the code of self.class_embed"

        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_actionness_or_class, outputs_coord, chosen_logits_type):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{chosen_logits_type: a, 'pred_boxes': b} for a, b in zip(outputs_actionness_or_class[:-1], outputs_coord[:-1])]

def compute_fft_magnitude(clip_feats, pred_boxes):
    """
    Args:
        clip_feats: Tensor of shape [bs, t, c]
        pred_boxes: Tensor of shape [bs, num_queries, 2] (start_time, end_time)

    Returns:
        fft_magnitudes: Tensor of shape [bs, num_queries, 1]
    """
    bs, num_queries, _ = pred_boxes.shape
    _, t, c = clip_feats.shape
    
    # 初始化 magnitude tensor
    fft_magnitudes = torch.zeros(bs, num_queries, 1, device=clip_feats.device)

    for b in range(bs):
        for q in range(num_queries):
            start, end = segment_cw_to_t1t2(pred_boxes[b, q]).clamp(min=0,max=1) * t
            
            # 確保索引正確性
            start = torch.ceil(start).long()
            end = torch.floor(end).long()
            # 確保有效區間
            start = torch.min(start, end)
            
            segment = clip_feats[b, start:end+1, :]  # [segment_length, c]
            Z = torch.fft.fft(segment.cpu(), dim=0)  # 沿時間維度做 FFT
            frequency_energy = torch.sum(torch.abs(Z)**2)
            fft_magnitudes[b, q, 0] = frequency_energy/(end-start+1)

    return fft_magnitudes

def compute_fft_magnitude_v3(clip_feats, pred_boxes):
    """
    Args:
        clip_feats: Tensor of shape [bs, t, c]
        pred_boxes: Tensor of shape [bs, num_queries, 2] (start_time, end_time)

    Returns:
        fft_magnitudes: Tensor of shape [bs, num_queries, 1]
    """
    bs, num_queries, _ = pred_boxes.shape
    _, t, c = clip_feats.shape

    # 確保索引正確性
    pred_boxes = segment_cw_to_t1t2(pred_boxes).clamp(0, 1) * t
    starts = torch.ceil(pred_boxes[..., 0]).long()
    ends = torch.floor(pred_boxes[..., 1]).long()
    # 確保有效區間 
    starts = torch.min(starts, ends).clamp(0, t - 1)

    # 初始化 magnitude tensor
    fft_magnitudes = torch.zeros(bs, num_queries, 1, device=clip_feats.device)

    for b in range(bs):
        for q in range(num_queries):
            start, end = starts[b, q], ends[b, q]
            
            segment = clip_feats[b, start:end+1, :]  # [segment_length, c]
            Z = torch.fft.fft(segment.cpu(), dim=0)  # 沿時間維度做 FFT
            frequency_energy = torch.sum(torch.abs(Z)**2)
            fft_magnitudes[b, q, 0] = frequency_energy/(end-start+1)

    return fft_magnitudes

def compute_fft_magnitude_v2(clip_feats, pred_boxes):
    """
    Efficient FFT magnitude computation with vectorization.
    
    Args:
        clip_feats: Tensor of shape [bs, t, c]
        pred_boxes: Tensor of shape [bs, num_queries, 2] (center, width format)

    Returns:
        fft_magnitudes: Tensor of shape [bs, num_queries, 1]
    """
    bs, num_queries, _ = pred_boxes.shape
    _, t, c = clip_feats.shape

    pred_boxes = segment_cw_to_t1t2(pred_boxes).clamp(0, 1) * t
    start = torch.ceil(pred_boxes[..., 0]).long().clamp(0, t - 1)
    end = torch.floor(pred_boxes[..., 1]).long().clamp(0, t - 1)

    # Create index ranges & Generate valid masks
    max_len = (end - start + 1).max().item()  # Find max segment length
    indices = torch.arange(max_len, device=clip_feats.device).unsqueeze(0).unsqueeze(0)  # [1, 1, max_len]
    valid_mask = (indices < (end - start + 1).unsqueeze(-1))        # [bs, num_queries, max_len]

    # Generate time indices for slicing
    time_indices = (start.unsqueeze(-1) + indices).clamp(0, t - 1)  # [bs, num_queries, max_len]
    # Gather clip_feats using batch indices
    batch_indices = torch.arange(bs, device=clip_feats.device).view(-1, 1, 1)
    gathered_feats = clip_feats[batch_indices, time_indices]        # [bs, num_queries, max_len, c]

    # Apply mask (set invalid positions to zero)
    gathered_feats = gathered_feats * valid_mask.unsqueeze(-1)      # [bs, num_queries, max_len, c]

    # Compute FFT in batch mode
    Z = torch.fft.fft(gathered_feats.cpu(), dim=2)                  # [bs, num_queries, max_len, c] (FFT along time axis)
    
    # Z_square = torch.abs(Z.to(clip_feats.device))**2               
    # valid_Z = Z_square * valid_mask.unsqueeze(-1)                   
    # frequency_energy = torch.sum(valid_Z, dim=(2, 3))
    
    # Apply valid mask to the FFT result before summing
    valid_Z = Z.to(clip_feats.device) * valid_mask.unsqueeze(-1)    # [bs, num_queries, max_len, c]
    frequency_energy = torch.sum(torch.abs(valid_Z) ** 2, dim=(2, 3))  # Sum over time and feature dimensions
    fft_magnitudes = (frequency_energy / (end-start+1)).unsqueeze(-1)  # [bs, num_queries, 1]

    return fft_magnitudes

        
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

def build(args, device):
    if args.target_type != "none": # adopt one-hot as target, only used in close_set
        num_classes = int(args.num_classes * args.split / 100)
    else:
        num_classes = args.num_classes

    if args.feature_type == "ViFi-CLIP":
        text_encoder,logit_scale = None, torch.from_numpy(np.load(os.path.join(args.feature_path,'logit_scale.npy'))).float()
    elif args.feature_type == "CLIP":
        text_encoder, logit_scale = build_text_encoder(args,device)
    else:
        raise NotImplementedError
    
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    if args.enable_refine or args.enable_refine_freq:
        refine_decoder = build_refine_decoder(args)
    else:
        refine_decoder = None

    model = DeformableDETR(
        backbone,
        transformer, 
        text_encoder, 
        refine_decoder,
        logit_scale, 
        device, 
        num_classes,
        args=args
    )
    matcher = build_matcher(args)


    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.actionness_loss or args.eval_proposal or args.enable_classAgnostic:
        weight_dict['loss_actionness'] = args.actionness_loss_coef
    if args.salient_loss:
        if args.aux_loss and args.enlarge_aux_loss:
            if args.weightDict != -1:
                weight_dict['loss_salient'] = args.salient_loss_coef*args.dec_layers
            else:
                aux_scale = sum([(args.weightDict)**i for i in range(0, args.dec_layers)])
                weight_dict['loss_salient'] = args.salient_loss_coef*aux_scale
        else:
            weight_dict['loss_salient'] = args.salient_loss_coef
    if args.vis_con_loss:
        weight_dict['loss_visual_consistency'] = args.vis_con_loss_coef
    if args.use_decouple:
        if args.aux_loss:
            if args.weightDict != -1:
                weight_dict['loss_alignment'] = 5.0*args.dec_layers
            else:
                aux_scale = sum([(args.weightDict)**i for i in range(0, args.dec_layers)])
                weight_dict['loss_alignment'] = 5.0*aux_scale
        else:
            weight_dict['loss_alignment'] = 5.0

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        if args.weightDict != -1:
            for i, layer in enumerate(reversed(range(args.dec_layers - 1))):
                aux_weight_dict.update({k + f'_{layer}': v*(args.weightDict**(i+1)) for k, v in weight_dict.items()}) # forming {"loss_bbox_0": v, "loss_giou_0": v, "loss_class_0": v, ..."loss_bbox_L": v, "loss_giou_L": v, "loss_class_L": v}
            # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()}) # Deformable DETR中有使用
            weight_dict.update(aux_weight_dict)
        else:
            for layer in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{layer}': v for k, v in weight_dict.items()}) # forming {"loss_bbox_0": v, "loss_giou_0": v, "loss_class_0": v, ..."loss_bbox_L": v, "loss_giou_L": v, "loss_class_L": v}
            # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()}) # Deformable DETR中有使用
            weight_dict.update(aux_weight_dict)

    if args.eval_proposal or args.enable_classAgnostic:
        losses = ['boxes','actionness'] # default
    elif args.actionness_loss:
        losses = ['labels','actionness','boxes']
    else:
        losses = ['labels', 'boxes']
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor
