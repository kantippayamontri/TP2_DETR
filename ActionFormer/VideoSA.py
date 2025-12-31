import torch
import torch.nn.functional as F
from torch import nn, Tensor


class VideoGuidedQueryGenerator_ChatGPT(nn.Module):
    def __init__(self, memory_dim, num_queries, hidden_dim):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool temporal dimension
        self.fc = nn.Linear(memory_dim, hidden_dim)
        self.num_queries = num_queries

    def forward(self, memory_feats):
        """
        Args:
            memory_feats: [batch_size, temporal_len, memory_dim]
        Returns:
            queries: [batch_size, num_queries, hidden_dim]
        """
        x = memory_feats.transpose(1, 2)  # [B, C, T]
        pooled = self.pooling(x).squeeze(-1)  # [B, C]
        queries = self.fc(pooled)  # [B, hidden_dim]
        queries = queries.unsqueeze(1).expand(-1, self.num_queries, -1)  # [B, num_queries, hidden_dim]
        return queries

## ===================== SAPM-like =============================
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


class VideoGuidedQueryGenerator(nn.Module):
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
    
