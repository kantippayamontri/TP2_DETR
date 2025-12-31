import torch
import torch.nn as nn
import torch.nn.functional as F

## ============================ Spatial ===========================
class AttentionMapProjection(nn.Module):
    def __init__(self, in_channels=256, inter_channels=256, out_channels=10, temperature=1.2):
        '''
        out_channels = Q : the number of queries in DETR’s variants
        '''
        super().__init__()
        self.temperature = temperature

        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=inter_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        x = self.conv3(x)

        # reshape to [B, Q, H*W] then apply softmax along spatial dim
        B, Q, H, W = x.shape
        x_flat = x.view(B, Q, -1)                         # [B, Q, H*W]
        A = F.softmax(x_flat * self.temperature, dim=-1)  # normalize over spatial dim
        A = A.view(B, Q, H, W)                            # reshape back
        return A
    
    
def weighted_pooling(F, A):
    """
    Args:
        F: Tensor of shape [B, C, H, W] - input features
        A: Tensor of shape [B, Q, H, W] - attention weights per query

    Returns:
        F_P: Tensor of shape [B, Q, C] - pooled feature per query
    """
    B, C, H, W = F.shape
    _, Q, _, _ = A.shape

    # Flatten spatial dimension
    F_flat = F.view(B, C, H * W)          # [B, C, H*W]
    A_flat = A.view(B, Q, H * W)          # [B, Q, H*W]

    # Normalize A to sum to 1 over spatial dim (in case not softmaxed)
    A_flat = A_flat / (A_flat.sum(dim=-1, keepdim=True) + 1e-6)

    # Weighted pooling: [B, Q, H*W] @ [B, H*W, C] = [B, Q, C]
    F_flat = F_flat.permute(0, 2, 1)      # [B, H*W, C]
    F_P = torch.bmm(A_flat, F_flat)       # [B, Q, C]

    return F_P


class ChannelReweighting(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, F_P):
        """
        Args:
            F_P: Tensor of shape [B, Q, C] - pooled feature per query
        Returns:
            F_O: Tensor of shape [B, Q, C] - refined feature
        """
        weights = self.mlp(F_P)     # [B, Q, C]
        F_O = weights * F_P         # element-wise multiplication
        return F_O


class MultiScaleFeaturePooling(nn.Module):
    def __init__(self, in_channels_list, query_dim=10, hidden_dim=256):
        """
        Args:
            in_channels_list: list of input channel sizes for each scale (e.g., [256, 256, 256])
            query_dim: number of queries (Q)
        """
        super().__init__()
        self.num_scales = len(in_channels_list)

        # Create a separate AMP for each scale
        self.amps = nn.ModuleList([
            AttentionMapProjection(in_channels=c, out_channels=query_dim)
            for c in in_channels_list
        ])

        # Shared channel reweighting module (optional: can be separate too)
        self.cr = ChannelReweighting(dim=hidden_dim)

    def forward(self, multi_scale_feats):
        """
        Args:
            multi_scale_feats: list of tensors, each [B, C, H, W]

        Returns:
            F_P: [B, Q, C] - averaged over scales
        """
        all_F_P = []

        for i, feat in enumerate(multi_scale_feats):
            A = self.amps[i](feat)               # [B, Q, H, W]
            F_P = weighted_pooling(feat, A)      # [B, Q, C]
            F_P = self.cr(F_P)                   # [B, Q, C]
            all_F_P.append(F_P)

        # Average across all scales
        F_P_final = torch.stack(all_F_P, dim=0).mean(dim=0)  # [B, Q, C]
        return F_P_final


## ============================ Temporal ===========================
class TemporalAttentionMapProjection_Flex(nn.Module):
    def __init__(self, in_channels=512, out_channels=40, temperature=1.0):
        """
        out_channels: Q (number of queries)
        Input: [B, C, T]
        Output: [B, Q, T]
        """
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        x = self.proj(x)                        # [B, out_channels, T]
        A = F.softmax(x / self.temperature, dim=-1)
        return A
    

class TemporalAttentionMapProjection(nn.Module):
    def __init__(self, in_channels=256, inter_channels=256, out_channels=10, temperature=1.2):
        '''
        out_channels = Q : the number of queries
        '''
        super().__init__()
        self.temperature = temperature

        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=inter_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv1d(inter_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        x = self.relu1(self.gn1(self.conv1(x)))  # [B, inter_channels, T]
        x = self.relu2(self.gn2(self.conv2(x)))  # [B, inter_channels, T]
        x = self.conv3(x)                        # [B, Q, T]

        A = F.softmax(x * self.temperature, dim=-1)  # softmax over temporal dim
        return A  # [B, Q, T]


def weighted_temporal_pooling(F, A):
    """
    F: [B, C, T]
    A: [B, Q, T]
    return: F_P: [B, Q, C]
    """
    B, C, T = F.shape
    _, Q, _ = A.shape

    # Normalize over temporal dimension
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    F_T = F.permute(0, 2, 1)  # [B, T, C]
    F_P = torch.bmm(A, F_T)   # [B, Q, C]
    return F_P


class MultiScaleTemporalPooling(nn.Module):
    def __init__(self, in_channels_list=256, out_channels=10, hidden_dim=256):
        """
        Args:
            in_channels: number of input channels per scale
            out_channels: number of queries
        """
        super().__init__()
        self.num_scales = len(in_channels_list)
        # Create a separate AMP for each scale
        self.amps = nn.ModuleList([
            TemporalAttentionMapProjection_Flex(in_channels=c, out_channels=out_channels)
            for c in in_channels_list
        ])

        # Shared channel reweighting module (optional: can be separate too)
        self.cr = ChannelReweighting(dim=hidden_dim)

    def forward(self, multi_scale_feats, weights):
        """
        Args:
            multi_scale_feats: list of T temporal features, each [B, C, T_i]

        Returns:
            F_P: [B, Q, C] - averaged over scales
        """
        all_F_P = []

        for i, feat in enumerate(multi_scale_feats):
            A = self.amps[i](feat)                      # [B, Q, T_i]
            F_P = weighted_temporal_pooling(feat, A)    # [B, Q, C]
            F_P = self.cr(F_P)                          # [B, Q, C]
            all_F_P.append(F_P)

        if weights == None:
            # Average across all scales
            F_P_final = torch.stack(all_F_P, dim=0).mean(dim=0)  # [B, Q, C]
        else:
            weights = weights / weights.sum()               # [S]
            weights = weights.view(-1, 1, 1, 1)             # [S, 1, 1, 1]
            all_F_P_tensor = torch.stack(all_F_P, dim=0)    # [S, B, Q, C]
            F_P_final = (all_F_P_tensor * weights).sum(dim=0) # [B, Q, C]
        return F_P_final
    

class IntegrateMultiScaleTemporalPooling(nn.Module):
    def __init__(self, in_channels=256, out_channels=10, hidden_dim=256):
        """
        Args:
            in_channels: number of input channels per scale
            out_channels: number of queries
        """
        super().__init__()

        # Create a separate AMP for each scale
        self.amps = TemporalAttentionMapProjection(in_channels=in_channels, out_channels=out_channels)

        # Shared channel reweighting module (optional: can be separate too)
        self.cr = ChannelReweighting(dim=hidden_dim)

    def forward(self, consecutive_multi_scale_feats):
        """
        Args:
            consecutive_multi_scale_feats: list of T temporal features, each [B, C, sum(T_i)]

        Returns:
            F_P: [B, Q, C] - averaged over scales
        """
        A = self.amps(consecutive_multi_scale_feats)                        # [B, Q, T_i]
        F_P = weighted_temporal_pooling(consecutive_multi_scale_feats, A)   # [B, Q, C]
        F_P = self.cr(F_P)                                                  # [B, Q, C]

        return F_P



if __name__=='__main__':
    ## ============================ Spatial ===========================
    # ## 【整合前】
    # # Features from transformer encoder
    # Feat = torch.randn(2, 256, 64, 64)  # [B, C, H, W] = [2, 256, 64, 64]

    # # Attention Maps
    # AMP = AttentionMapProjection(in_channels=256, out_channels=10)  # 假設想要輸出10個channel(Q=10)
    # A = AMP(Feat)  # [B, Q, H, W] = [2, 10, 64, 64]
    # print(A.shape) 
    
    # # object-specific feature
    # F_P = weighted_pooling(Feat, A)  # [B, Q, C] = [2, 10, 256]
    # print(F_P.shape) 

    # # Pooled Features
    # CR = ChannelReweighting(dim=256)
    # F_O = CR(F_P)   # [B, Q, C]= [2, 10, 256]
    # print(F_O.shape) 

    # ## 【整合後】
    # # 假設三個不同 scale 的 feature maps，分別來自 FPN 的不同層
    # F3 = torch.randn(2, 256, 64, 64)
    # F4 = torch.randn(2, 256, 32, 32)
    # F5 = torch.randn(2, 256, 16, 16)

    # model = MultiScaleFeaturePooling(in_channels_list=[256, 256, 256], query_dim=10)
    # F_P_final = model([F3, F4, F5])   # [B, Q, C] = [2, 10, 256]

    # print(F_P_final.shape)

    ## ============================ Temporal ===========================
    # ## 【整合前】
    # Feat = torch.randn(2, 256, 128)  # [B, C, T]

    # amp = TemporalAttentionMapProjection(in_channels=256, out_channels=10)
    # A = amp(Feat)  # [B, Q, T]
    # print(A.shape)

    # F_P = weighted_temporal_pooling(Feat, A)  # [B, Q, C]
    # print(F_P.shape)

    # cr = ChannelReweighting(dim=256)
    # F_O = cr(F_P)
    # print(F_O.shape)

    ## 【整合後】
    # 假設三個不同 scale 的 feature maps，分別來自 FPN 的不同層
    F3 = torch.randn(2, 256, 64)
    F4 = torch.randn(2, 256, 32)
    F5 = torch.randn(2, 256, 16)

    model = MultiScaleTemporalPooling(in_channels_list=[256, 256, 256], out_channels=10)
    F_P_final = model([F3, F4, F5])   # [B, Q, C] = [2, 10, 256]
    print(F_P_final.shape)

    #----------------------------------------------------------------------------
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters/1000000} M')


