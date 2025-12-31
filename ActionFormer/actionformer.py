import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional

from blocks import LayerNorm, LocalMaskedMHCA, AffineDropPath
# from blocks_causal import LayerNorm, LocalMaskedMHCA, AffineDropPath

class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False,      # if to add rel position encoding to attention
        normalize_before=True,# Pre-LN or Post-LN
    ):
        super().__init__()
        self.normalize_before = normalize_before
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        self.attn = LocalMaskedMHCA(
            n_embd,
            n_head,
            window_size=mha_win_size,
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            use_rel_pe=use_rel_pe  # only valid for local attention
        )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward_pre(self, x, src_mask=None, pos_embd=None, use_Gating=False, no_normalizedqkv=False):
        mask = src_mask
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        # out, out_mask = self.attn(self.ln1(x), mask, use_Gating=use_Gating, no_normalizedqkv=no_normalizedqkv)
        out_mask_float = out_mask.to(out.dtype)
        
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask
    
    def forward_post(self, x, src_mask=None, pos_embd=None, use_Gating=False, no_normalizedqkv=False):
        mask = src_mask
        # post-LN transformer
        out, out_mask = self.attn(x, mask)
        # out, out_mask = self.attn(x, mask, use_Gating=use_Gating, no_normalizedqkv=no_normalizedqkv)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        out = self.ln1(out)  # LayerNorm applied after residual connection
        # FFN
        out = out + self.drop_path_mlp(self.mlp(out) * out_mask_float) 
        out = self.ln2(out)  # LayerNorm applied after residual connection
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask
    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                use_Gating=False,
                no_normalizedqkv=False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos, use_Gating=use_Gating, no_normalizedqkv=no_normalizedqkv)
        return self.forward_post(src, src_mask, pos, use_Gating=use_Gating, no_normalizedqkv=no_normalizedqkv)


class ActionFormerFPN(nn.Module):

    def __init__(
            self, 
            d_model=512,                    # input feature dimension
            nhead=8,                        # number of head for self-attention in transformers
            downsample_rate = 2,            # downsampling rate for the branch
            mha_win_size = [-1]*4,          # size of local window for mha
            attn_pdrop=0.0,                 # dropout rate for the attention map 
            proj_pdrop=0.0,                 # dropout rate for the projection / MLP
            path_pdrop=0.0,                 # drop path rate
            max_len = 2304,                 # max sequence length
            num_encoder_layers= (1, 3),     # (#stem transformers, #branch transformers)
            use_abs_pe = False,             # use absolute position embedding
            use_rel_pe = False,             # use relative position embedding
            normalize_before=True,         # Pre-LN or Post-LN
            args = None
            ):
        super().__init__()
        assert len(mha_win_size) == (num_encoder_layers[0] + num_encoder_layers[1])
        self.d_model = d_model
        self.nhead = nhead
        self.downsample_rate = downsample_rate
        self.mha_win_size = mha_win_size
        
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe     

        self.normalize_before = normalize_before

        self.args = args

        ## Encoder: actionformer
        # (1) stem network using (vanilla) transformer
        self.encoder_stem = nn.ModuleList()
        for idx in range(num_encoder_layers[0]):
            self.encoder_stem.append(
                TransformerBlock(
                    d_model, 
                    nhead,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
        # (2) main branch using transformer with pooling
        self.encoder_branch = nn.ModuleList()
        for idx in range(num_encoder_layers[1]):
            self.encoder_branch.append(
                TransformerBlock(
                    d_model, 
                    nhead,
                    n_ds_strides=(self.downsample_rate, self.downsample_rate),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[num_encoder_layers[0] + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        
        # init weights
        init_method = getattr(self, f"_reset_parameters_{args.backbone_init_method}", None)
        if init_method is not None:
            init_method()
        else:
            raise ValueError(f"Unknown backbone_init_method: {args.backbone_init_method}")

        
    def _reset_parameters_v0(self):
        # set nn.Linear/nn.Conv1d bias term to 0
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def _reset_parameters_v1(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _reset_parameters_v2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _reset_parameters_v3(self):
        '''v0+v1'''
        # 初始化 weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 初始化 bias
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
    
    def _reset_parameters_v4(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


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
        # print('ActionFormer Stem Network')
        for idx in range(len(self.encoder_stem)):
            x, emask = self.encoder_stem[idx](src=x, src_mask=emask, use_Gating=self.args.use_Gating, no_normalizedqkv=self.args.no_normalizedqkv)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        # (2) main branch with downsampling
        # print('ActionFormer Branch Network')
        for idx in range(len(self.encoder_branch)):
            x, emask = self.encoder_branch[idx](src=x, src_mask=emask, use_Gating=self.args.use_Gating, no_normalizedqkv=self.args.no_normalizedqkv)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        # for lvl, out_feat in enumerate(out_feats):
        #     print(f'Level{lvl}: Size of out_feat = {out_feat.size()}')

        return out_feats, out_masks


class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = #levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks
    

if __name__=='__main__':
    import options
    args = options.parser.parse_args()
    args.enc_stem_layers = 1
    args.enc_branch_layers = 3

    mha_win_size_list = [args.win_size]*(args.enc_stem_layers+args.enc_branch_layers)

    fpn_extractor = ActionFormerFPN(d_model=args.hidden_dim, 
                                    nhead=args.nheads, 
                                    downsample_rate=2, 
                                    mha_win_size=mha_win_size_list,
                                    num_encoder_layers=(args.enc_stem_layers, args.enc_branch_layers),                                    
                                    args=args)
    
    src = torch.randn((16,128,512))
    mask = torch.zeros((16,128),dtype=bool)
    pos = torch.randn((16,128,512))
    fpn, masks = fpn_extractor(src, mask, pos)

    '''
    Deformable DETR
    (1, 256, 88, 133)
    (1, 256, 44, 67)
    (1, 256, 22, 34)
    (1, 256, 11, 17)
    ActionFormer
    (1, 512, 128)
    (1, 512, 64)
    (1, 512, 32)
    (1, 512, 16)
    '''
    

    print(f"fpn.shape:{fpn.shape}")