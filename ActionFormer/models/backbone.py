#!/usr/bin/python3
# -*- encoding: utf-8 -*-


"""
ActionFormer Encoder modified modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from utils.misc import NestedTensor

from .position_encoding import build_position_encoding
from actionformer import ActionFormerFPN, FPNIdentity
from tridet import SGPFPN
from blocks import downsample, LayerNorm, MaskedConv1D, ConvBlock, Sequential, Identity, LayerNorm_TETAD
# from blocks_causal import downsample, LayerNorm, MaskedConv1D, ConvBlock, Sequential, Identity
    
class ActionFormerBackbone(nn.Module):
    '''ActionFormer as the backbone for Deformable DETR.'''
    def __init__(self, args):
        super().__init__()
        mha_win_size_list = [args.win_size]*(args.enc_stem_layers+args.enc_branch_layers)
        self.fpn = ActionFormerFPN(
            d_model=args.hidden_dim, 
            nhead=args.nheads, 
            downsample_rate=args.downsample_rate, 
            num_encoder_layers=(args.enc_stem_layers, args.enc_branch_layers), 
            mha_win_size=mha_win_size_list,
            attn_pdrop=0.0, # Attention 需要完整的信息流，Dropout 可能破壞時序建模能力
            proj_pdrop=0.1, # MLP 負責 feature transformation，加少量 Dropout 可以提高泛化能力
            path_pdrop=0.0, # add
            args=args
        )
        if args.enable_neck:
            self.neck = FPNIdentity(
                in_channels=[args.hidden_dim] * (args.enc_stem_layers+args.enc_branch_layers),
                out_channel=args.hidden_dim,
                scale_factor=args.downsample_rate,
                with_ln=True
            )
        else:
            self.neck = None


    def forward(self, tensor_list: NestedTensor):
        tensors, mask = tensor_list.decompose() # [b,t,c], [b,t]
        src = tensors
        out_feats, out_masks = self.fpn(src, mask, pos_embed=None) #[[b, c, t], [b, c, t/2], [b, c, t/4], [b, c, t/8]], [[b, 1, t], [b, 1, t/2], [b, 1, t/4], [b, 1, t/8]]
        if self.neck != None:
            out_feats, out_masks = self.neck(out_feats, out_masks)

        out = []
        for feat, mask in zip(out_feats, out_masks):
            out.append(NestedTensor(feat, mask))
        
        return out


class SGPBackbone(nn.Module):
    '''TriDet(SGP layers) as the backbone for Deformable DETR.'''
    def __init__(self, args):
        super().__init__()
        # mha_win_size_list = [args.win_size]*(args.enc_stem_layers+args.enc_branch_layers)
        mha_win_size_list = [1]*(args.enc_stem_layers+args.enc_branch_layers)
        self.fpn = SGPFPN(
            d_model=args.hidden_dim,
            num_encoder_layers= (args.enc_stem_layers, args.enc_branch_layers),     
            sgp_mlp_dim=768,               
            downsample_rate=args.downsample_rate,              
            downsample_type='max',          
            sgp_win_size = mha_win_size_list,
            k=1.5,                          
            init_conv_vars=1,               
            path_pdrop=0.0,    
            args=args    
        )

    def forward(self, tensor_list: NestedTensor):
        tensors, mask = tensor_list.decompose() # [b,t,c], [b,t]
        src = tensors
        out_feats, out_masks = self.fpn(src, mask, pos_embed=None) #[[b, c, t], [b, c, t/2], [b, c, t/4], [b, c, t/8]], [[b, 1, t], [b, 1, t/2], [b, 1, t/4], [b, 1, t/8]]

        out = []
        for feat, mask in zip(out_feats, out_masks):
            out.append(NestedTensor(feat, mask))

        return out


class PlainBackbone(nn.Module):
    """Con1D Local Relationship Modeling."""
    def __init__(self, args):
        super().__init__()
        self.fpn = PlainFPN(
            num_encoder_layers= (args.enc_stem_layers, args.enc_branch_layers),     
            downsample_rate=args.downsample_rate,     
            args=args    
        )
    
    def forward(self, tensor_list: NestedTensor):
        tensors, mask = tensor_list.decompose() # [b,t,c], [b,t]
        src = tensors
        out_feats, out_masks = self.fpn(src, mask, pos_embed=None) #[[b, c, t], [b, c, t/2], [b, c, t/4], [b, c, t/8]], [[b, 1, t], [b, 1, t/2], [b, 1, t/4], [b, 1, t/8]]

        out = []
        for feat, mask in zip(out_feats, out_masks):
            out.append(NestedTensor(feat, mask))

        return out


class PlainFPN(nn.Module):

    def __init__(
            self, 
            num_encoder_layers= (1, 3),     # (#stem transformers, #branch transformers)
            downsample_rate=2,              # dowsampling rate for the branch,
            args = None
    ):
        super().__init__()
        assert len(num_encoder_layers) == 2
        self.downsample_rate = downsample_rate
        self.num_encoder_layers = num_encoder_layers
    
    def forward(self, src, mask, pos_embed):
        '''
        input:
            src: [b,t,c]
            mask: [b,t]
            pos_embed: [b,t,c]
        '''
        # permute NxTxC to TxNxC
        bs, t, c = src.shape
        # pos_embed = pos_embed.permute(1, 0, 2) # [t,b,c]

        ## Encoder
        # [b, t, c] --> [b, c, t]
        x = src.permute(0, 2, 1)
        emask = ~(mask.unsqueeze(1))
        out_feats = [] # FPN:([b, c, t], [b, c, t/2], [b, c, t/4], [b, c, t/8]) 
        out_masks = [] # FPN:([b, t], [b, t/2], [b, t/4], [b, t/8]) 
        # (1) stem transformer
        for idx in range(self.num_encoder_layers[0]):
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        # (2) main branch with downsampling
        for idx in range(self.num_encoder_layers[1]):
            x, emask = downsample(x, emask, self.downsample_rate)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        return out_feats, out_masks


class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
        from https://github.com/happyharrycn/actionformer_release/libs/modeling/backbones.py#L168
    """
    def __init__(
        self,
        feature_dim=512,
        hidden_dim=512,
        kernel_size=3,
        arch=(2, 2),
        num_feature_levels=4,
        scale_factor=2,
        with_ln=False,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.arch = arch
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            feature_dim = hidden_dim if idx > 0 else feature_dim
            self.embd.append(
                MaskedConv1D(
                    feature_dim, hidden_dim, kernel_size,
                    stride=1, padding=kernel_size//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm_TETAD(hidden_dim))
                # self.embd_norm.append(BatchNorm1d(hidden_dim))
            else:
                self.embd_norm.append(Identity())

        for layer in self.embd:
            nn.init.xavier_uniform_(layer.conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer.conv.bias)

        # stem network using convs
        self.stem = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size=3, stride=1)
            for _ in range(arch[1])
        ])

        # main branch using convs with pooling
        self.branch = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=self.scale_factor, padding=1),
                LayerNorm_TETAD(hidden_dim),
            )
            # ConvBlock(hidden_dim, kernel_size=3, stride=self.scale_factor)
            for _ in range(num_feature_levels-1)
        ])
        for layer in self.branch:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)
        # init weights
        self.apply(self.__init_weights__)



    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    
    def forward(self, tensor_list: NestedTensor):

        x, mask = tensor_list.decompose() # [b,t,c], [b,t]
        
        x = x.permute(0, 2, 1) # from [b, t, c] -> [b, c, t]
        emask = ~(mask.unsqueeze(1))
        out_feats = [] # FPN:([b, c, t], [b, c, t/2], [b, c, t/4], [b, c, t/8]) 
        out_masks = [] # FPN:([b, t], [b, t/2], [b, t/4], [b, t/8]) 

        # embedding network
        for idx in range(len(self.embd)):
            x, emask = self.embd[idx](x, emask)
            x = self.embd_norm[idx](x, emask)[0]
            # x = self.activation(self.embd_norm[idx](x, mask)[0])

        # stem conv
        for idx in range(len(self.stem)):
            x, emask = self.stem[idx](x, emask)
            if idx==len(self.stem)-1:
                out_feats.append(x)
                out_masks.append(~(emask.squeeze(1)))

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, emask = self.branch[idx](x, emask)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        out = []
        for feat, mask in zip(out_feats, out_masks):
            out.append(NestedTensor(feat, mask))
        
        return out


class Joiner(nn.Sequential):
    '''Return corresponded positional encodings'''
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        out_tensor_list = self[0](tensor_list) # [NestedTensor([b, c, t], [b, t]), NestedTensor([b, c, t/2], [b, t/2]), ...]
        out: List[NestedTensor] = []
        pos = []
        
        for nest in out_tensor_list:
            out.append(nest)
            pos.append(self[1](nest).to(nest.tensors.dtype)) # position encoding

        return out, pos
    

def build_backbone(args):
    if args.plainFPN:
        position_embedding = build_position_encoding(args)
        backbone = PlainBackbone(args)
        model = Joiner(backbone, position_embedding)
    elif args.use_SGP:
        position_embedding = build_position_encoding(args)
        backbone = SGPBackbone(args)
        model = Joiner(backbone, position_embedding)
    elif args.use_TETAD:
        position_embedding = build_position_encoding(args)
        backbone = ConvBackbone()
        model = Joiner(backbone, position_embedding)
    else:
        position_embedding = build_position_encoding(args)
        backbone = ActionFormerBackbone(args)
        model = Joiner(backbone, position_embedding)
    return model
