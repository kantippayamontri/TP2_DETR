import torch
from torch import nn
from torch.nn import functional as F

from blocks import LayerNorm, AffineDropPath, MaskedConv1D, Scale
import math

class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,             # dimension of the input features
            kernel_size=3,      # conv kernel size
            n_ds_stride=1,      # downsampling stride for the current layer
            k=1.5,              # k
            group=1,            # group for cnn
            n_out=None,         # output dimension, if None, set to input dim
            n_hidden=None,      # hidden dim for mlp
            path_pdrop=0.0,     # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1    # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()


class SGPFPN(nn.Module):

    def __init__(
            self, 
            d_model=512,                    # input feature dimension
            num_encoder_layers= (1, 3),     # (#stem transformers, #branch transformers)
            sgp_mlp_dim=768,                # the numnber of dim in SGP
            downsample_rate=2,              # dowsampling rate for the branch,
            downsample_type='max',          # how to downsample feature in FPN
            sgp_win_size = [-1]*4,          # size of local window for mha
            k=1.5,                          # the K in SGP
            init_conv_vars=1,               # initialization of gaussian variance for the weight in SGP
            path_pdrop=0.0,                 # droput rate for drop path
            use_abs_pe=False,               # use absolute position embedding
            args = None
    ):
        super().__init__()
        assert len(num_encoder_layers) == 2
        assert len(sgp_win_size) == (1 + num_encoder_layers[1])
        self.arch = num_encoder_layers
        self.sgp_win_size = sgp_win_size
        self.relu = nn.ReLU(inplace=True)
        self.downsample_rate = downsample_rate
        self.use_abs_pe = use_abs_pe

        # (1) stem network using (vanilla) transformer
        self.encoder_stem = nn.ModuleList()
        for idx in range(num_encoder_layers[0]):
            self.encoder_stem.append(
                SGPBlock(d_model, 1, 1, n_hidden=sgp_mlp_dim, k=k, init_conv_vars=init_conv_vars))

        # (2) main branch using transformer with poolingg
        self.encoder_branch = nn.ModuleList()
        for idx in range(num_encoder_layers[1]):
            self.encoder_branch.append(SGPBlock(d_model, self.sgp_win_size[num_encoder_layers[0] + idx], self.downsample_rate, path_pdrop=path_pdrop,
                                        n_hidden=sgp_mlp_dim, downsample_type=downsample_type, k=k,
                                        init_conv_vars=init_conv_vars))
        
        # init weights
        init_method = getattr(self, f"_reset_parameters_{args.backbone_init_method}", None)
        if init_method is not None:
            init_method()
        else:
            raise ValueError(f"Unknown backbone_init_method: {args.backbone_init_method}")

        
    def _reset_parameters_origin(self):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(self.modules(), (nn.Linear, nn.Conv1d)):
            if self.modules().bias is not None:
                torch.nn.init.constant_(self.modules().bias, 0.)

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
        # print('SGP Stem Network')
        for idx in range(len(self.encoder_stem)):
            x, emask = self.encoder_stem[idx](x=x, mask=emask)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        # (2) main branch with downsampling
        # print('SGP Branch Network')
        for idx in range(len(self.encoder_branch)):
            x, emask = self.encoder_branch[idx](x=x, mask=emask)
            out_feats.append(x)
            out_masks.append(~(emask.squeeze(1)))

        # for lvl, out_feat in enumerate(out_feats):
        #     print(f'Level{lvl}: Size of out_feat = {out_feat.size()}')

        return out_feats, out_masks


if __name__=='__main__':
    import options
    args = options.parser.parse_args()
    args.enc_stem_layers = 1
    args.enc_branch_layers = 3

    mha_win_size_list = [args.win_size]*(args.enc_stem_layers+args.enc_branch_layers)

    # fpn_extractor = SGPFPN(d_model=args.hidden_dim, 
    #                                 nhead=args.nheads, 
    #                                 downsample_rate=2, 
    #                                 mha_win_size=mha_win_size_list,
    #                                 num_encoder_layers=(args.enc_stem_layers, args.enc_branch_layers),                                    
    #                                 args=args)
    fpn_extractor = SGPFPN(d_model=args.hidden_dim,
                            num_encoder_layers= (args.enc_stem_layers, args.enc_branch_layers),     
                            sgp_mlp_dim=768,               
                            downsample_rate=2,              
                            downsample_type='max',          
                            sgp_win_size = mha_win_size_list,
                            k=1.5,                          
                            init_conv_vars=1,               
                            path_pdrop=0.0,                 
                            use_abs_pe=False,  
                            args=args            
                    )
    
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