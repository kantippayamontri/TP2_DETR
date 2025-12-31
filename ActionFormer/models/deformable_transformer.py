# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from SAPM import MultiScaleTemporalPooling
from VideoSA import VideoGuidedQueryGenerator

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=40, use_SAPM=False, use_SAattn=False, use_VideoSA=False, SA_LastOnly=False, 
                 use_LFT=False, separate_LFOLFT=False, num_queries=40):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_SAPM = use_SAPM
        self.use_SAattn = use_SAattn
        self.use_VideoSA = use_VideoSA
        self.SA_LastOnly = SA_LastOnly

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        usage_SAattn = use_SAattn or use_VideoSA
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, False)
        sa_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, True)
        self.decoder = DeformableTransformerDecoder(decoder_layer, sa_decoder_layer, num_decoder_layers, return_intermediate_dec, usage_SAattn, SA_LastOnly, use_LFT, separate_LFOLFT)
        # enc_n_points_list = [6, 5, 4] 
        # self.encoder = DeformableTransformerEncoder(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points_list)
        # dec_n_points_list = [4, 4, 3, 2, 2]
        # self.decoder = DeformableTransformerDecoder(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, use_SAattn, dec_n_points_list, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 1)

        if use_SAPM:
            self.SAPM = MultiScaleTemporalPooling(in_channels_list=[d_model]*num_feature_levels, out_channels=num_queries, hidden_dim=d_model)
        if use_VideoSA:
            self.VGQG = VideoGuidedQueryGenerator(in_channels_list=[d_model]*num_feature_levels, out_channels=num_queries, hidden_dim=d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, T_ in enumerate(temporal_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + T_)].view(N_, T_, 1)
            valid_T = torch.sum(~mask_flatten_[:, :, 0], 1)

            grid_t = torch.linspace(0, T_ - 1, T_, dtype=torch.float32, device=memory.device).unsqueeze(-1)

            scale = valid_T.unsqueeze(-1).view(N_, 1, 1)
            grid = (grid_t.unsqueeze(0).expand(N_, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 2)
            proposals.append(proposal)
            _cur += T_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio_t = valid_T.float() / T
        valid_ratio = valid_ratio_t.unsqueeze(-1)  # [b, 1]
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, weights=None, semantic_descriptions=None, pretrain=False):
        assert self.two_stage or query_embed is not None
        ###########
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_shape = t
            temporal_shapes.append(temporal_shape)
            src = src.transpose(1, 2)               # [b, t, c]
            mask = mask.flatten(1)                  # [b, t]
            pos_embed = pos_embed.transpose(1, 2)   # [b, t, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # [b, sum(t), c]
        mask_flatten = torch.cat(mask_flatten, 1)   # [b, sum(t)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # [b, sum(t), c]
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_shapes.new_zeros((1, )), temporal_shapes.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # [b, num_levels, 1]

        # encoder
        memory = self.encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, 
                              lvl_pos_embed_flatten, mask_flatten)
        
        if pretrain:
            return None, None, None, None, None, (memory, level_start_index) 

        if self.use_SAPM:
            # 從 memory 拆回 multi-scale 格式：List[B, C, T_i]
            multi_scale_feats = []
            start = 0
            for t in temporal_shapes:
                feat = memory[:, start:start+t, :]  # [B, T_i, C]
                feat = feat.transpose(1, 2).contiguous()  # [B, C, T_i]
                multi_scale_feats.append(feat)
                start += t
            # 使用 SAPM 初始化 tgt（content query）
            tgt = self.SAPM(multi_scale_feats, weights)  # [B, Q, C] (Defualt: FLex ver.)

        if self.use_VideoSA:
             # 從 memory 拆回 multi-scale 格式：List[B, C, T_i]
            multi_scale_feats = []
            start = 0
            for t in temporal_shapes:
                feat = memory[:, start:start+t, :]  # [B, T_i, C]
                feat = feat.transpose(1, 2).contiguous()  # [B, C, T_i]
                multi_scale_feats.append(feat)
                start += t
            # 使用 SAPM-like 初始化 tgt（content query）
            tgt = self.VGQG(multi_scale_feats, weights)  # [B, Q, C]

        ###########
        # prepare input for decoder
        bs, _, c = memory.shape # [b, sum(t), c]
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, temporal_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            if self.use_SAPM:
                query_embed, _ = torch.split(pos_trans_out, c, dim=2) # 只取 positional query
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # 所以 一開始query_embed才設為[num_queries, hidden_dim*2] 
            # query_embed = query_embed[:, 0:512]   ===> 用於positional embedding (locations)
            # tgt = query_embed[:, 512:1024]        ===> 用於initial target features (features)            
            if self.use_SAPM or self.use_VideoSA:
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)   # [b, num_queries, c]
            else:
                query_embed, tgt = torch.split(query_embed, c, dim=1) 
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)   # [b, num_queries, c]
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)                   # [b, num_queries, c]
            reference_points = self.reference_points(query_embed).sigmoid() # [b, num_queries, 1] Initial guess for action instance location/timestamp
            init_reference_out = reference_points
        
        ###########
        # decoder
        hs, inter_references, sampling_locations_dec, attn_weights_dec, sa_attn_weights_dec, self_attn_weights_dec = self.decoder(tgt, reference_points, memory, temporal_shapes, 
                                                level_start_index, valid_ratios, 
                                                query_embed, mask_flatten, 
                                                semantic_descriptions=semantic_descriptions)

        inter_references_out = inter_references

        ###########
        salient_preparation = (memory, level_start_index) # for computing salient_logits
        sparse_preparation = (temporal_shapes, sampling_locations_dec, attn_weights_dec)
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, salient_preparation, sparse_preparation, sa_attn_weights_dec, self_attn_weights_dec
        return hs, init_reference_out, inter_references_out, None, None, salient_preparation, sparse_preparation, sa_attn_weights_dec, self_attn_weights_dec


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=512, d_ffn=2048,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, temporal_shapes, level_start_index, padding_mask=None, tgt=None):
        if tgt is None:
            # self attention
            src2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, temporal_shapes, level_start_index, padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # ffn
            src = self.forward_ffn(src)

            return src, sampling_locations, attn_weights
        else:
            # self attention
            tgt2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(tgt, pos), reference_points, src, temporal_shapes, level_start_index, padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt = self.forward_ffn(tgt)

            return tgt, sampling_locations, attn_weights


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    # def __init__(self, d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points_list):
    #     super().__init__()
    #     self.layers =  nn.ModuleList()
    #     self.num_layers = len(enc_n_points_list)
        
    #     for enc_n_points in enc_n_points_list:
    #         encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
    #                                                         dropout, activation,
    #                                                         num_feature_levels, nhead, enc_n_points)
    #         self.layers.append(encoder_layer)

    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(temporal_shapes):  
            ref_t = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)   # [t,]
            ref_t = ref_t.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * T_)          # [b, t]
            ref = ref_t.unsqueeze(-1)                                                       # [b, t, 1]
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)                              # [b, sum(t), 1]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]             # [b, sum(t), num_levels, 1]
        return reference_points 

    def forward(self, src, temporal_shapes, level_start_index, valid_ratios, 
                pos=None, padding_mask=None):
        output = src # [b, sum(t), c]
        reference_points = self.get_reference_points(temporal_shapes, valid_ratios, device=src.device) # [b, sum(t), num_levels, 1]
        for _, layer in enumerate(self.layers):
            output, sampling_locations, attn_weights = layer(output, pos, reference_points, temporal_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ffn=2048,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, usage_SAattn=False):
        super().__init__()

        self.usage_SAattn = usage_SAattn

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # semantic-aware attention
        if usage_SAattn:
            self.sa_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout_sa = nn.Dropout(dropout)
            self.norm_sa = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_temporal_shapes, level_start_index, src_padding_mask=None, semantic_descriptions=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos) # [b, num_quries, c]
        tgt2, self_attn_weights = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))
        tgt2 = tgt2.transpose(0, 1) 
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2, sampling_locations, attn_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_temporal_shapes, level_start_index, src_padding_mask) # [b, num_quries, c]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # semantic-aware attention
        if self.usage_SAattn and semantic_descriptions is not None:
            semantic_kv = semantic_descriptions.unsqueeze(1).expand(-1, tgt.size(0), -1)  # [num_classes, c]-->[num_classes, b,  c]
            tgt2, sa_attn_weights = self.sa_attn(
                tgt.transpose(0, 1),  # q: [num_queries, b, c]
                semantic_kv,          # k
                semantic_kv           # v
            )
            tgt2 = tgt2.transpose(0, 1)      # [b, num_queries, c]
            tgt = tgt + self.dropout_sa(tgt2)
            tgt = self.norm_sa(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        if self.usage_SAattn:
            return tgt, sampling_locations, attn_weights, sa_attn_weights, self_attn_weights
        else:
            return tgt, sampling_locations, attn_weights, None, self_attn_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, sa_decoder_layer, num_layers, return_intermediate=False, usage_SAattn=False, SA_LastOnly=False, 
                 use_LFT = False, separate_LFOLFT=False):
        
        super().__init__()
        if SA_LastOnly:
            self.layers = _get_clones(decoder_layer, num_layers-1)
            self.layers.append(sa_decoder_layer)
        elif usage_SAattn:
            self.layers = _get_clones(sa_decoder_layer, num_layers)
        else:
            self.layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers
        self.usage_SAattn = usage_SAattn
        self.SA_LastOnly = SA_LastOnly
        self.use_LFT = use_LFT
        self.separate_LFOLFT = separate_LFOLFT
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        # self.ref_point_head = MLP(d_model, d_model, 1, 2)

    # def __init__(self, d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, use_SAattn, dec_n_points_list, return_intermediate=False):
    #     super().__init__()
    #     self.layers = nn.ModuleList()
    #     self.num_layers = len(dec_n_points_list)
    #     self.return_intermediate = return_intermediate
    #     # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
    #     self.bbox_embed = None
    #     self.class_embed = None
    #     # self.ref_point_head = MLP(d_model, d_model, 1, 2)
    #     for dec_n_points in dec_n_points_list:
    #         decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
    #                                                         dropout, activation,
    #                                                         num_feature_levels, nhead, dec_n_points, use_SAattn)
    #         self.layers.append(decoder_layer)

    def forward(self, tgt, reference_points, src, src_temporal_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, semantic_descriptions=None):
        output = tgt # [b, num_quries, c]

        intermediate = [] 
        intermediate_reference_points = []
        sampling_locations_all = []
        attn_weights_all = []
        sa_attn_weights_all = []
        self_attn_weights_all = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]                               # [b, num_quries, num_levels, 1]
            
            output, sampling_locations, attn_weights, sa_attn_weights, self_attn_weights = layer(output, query_pos, reference_points_input, src, src_temporal_shapes, 
                                                             src_level_start_index, src_padding_mask, semantic_descriptions)    # [b, num_quries, c]
            sampling_locations_all.append(sampling_locations)
            attn_weights_all.append(attn_weights)
            self_attn_weights_all.append(self_attn_weights)
            if sa_attn_weights is not None:
                sa_attn_weights_all.append(sa_attn_weights)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)      # [b, num_quries, 2] bbox_embed: predict "Δcenter" & "length"
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + inverse_sigmoid(reference_points) # 更新時間中心 + 長度
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 1
                    new_reference_points = tmp
                    new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points) # 更新時間中心
                    new_reference_points = new_reference_points.sigmoid() 
                if not self.use_LFT:
                    reference_points = new_reference_points.detach()            # LFO: 不讓梯度流過
                else:
                    if self.separate_LFOLFT:
                        if lid < len(self.layers) // 2:                         # 前半層使用 LFO
                            reference_points = new_reference_points.detach()    # LFO: 不讓梯度流過
                        else:                                                   # 後半層使用 LFT
                            reference_points = new_reference_points             # LFT: 讓梯度流過
                    else:
                        reference_points = new_reference_points                 # LFT: 讓梯度流過

            if self.return_intermediate:
                intermediate.append(output) # list: [b, num_quries, c]
                intermediate_reference_points.append(reference_points) # list: [b, num_quries, 2]


        # dimension=[num_layer, b, ...] 
        sampling_locations_all = torch.stack(sampling_locations_all, dim=0)
        attn_weights_all = torch.stack(attn_weights_all, dim=0)
        self_attn_weights_all = torch.stack(self_attn_weights_all, dim=0)

        if sa_attn_weights_all == []:
            sa_attn_weights_all = None
        else:
            sa_attn_weights_all = torch.stack(sa_attn_weights_all, dim=0)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), sampling_locations_all, attn_weights_all, sa_attn_weights_all, self_attn_weights_all

        return output, reference_points, sampling_locations_all, attn_weights_all, sa_attn_weights_all, self_attn_weights_all


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        use_SAPM = args.use_SAPM,
        use_SAattn = args.use_SAattn,
        use_VideoSA = args.use_VideoSA,
        SA_LastOnly = args.SA_LastOnly,
        use_LFT = args.use_LFT,
        separate_LFOLFT = args.separate_LFOLFT,
        num_queries = args.num_queries)

