import numpy as np
import torch

def attn_map_to_flat_grid(temporal_shapes, level_start_index, sampling_locations, attention_weights):
    """
    Args:
        temporal_shapes: Tensor of shape [num_levels], 表示每個 temporal level 的長度
        level_start_index: Tensor of shape [num_levels]，表示每個 level 對應到 flat grid 的起始 index
        sampling_locations: Tensor of shape [N, n_layers, Len_q, n_heads, n_levels, n_points, 1]，表示 normalized 的時間位置
        attention_weights:  Tensor of shape [N, n_layers, Len_q, n_heads, n_levels, n_points]

    Returns:
        flat_grid: Tensor of shape [N, n_layers, n_heads, total_len]，flatten 過的注意力圖
    """

    # 基本維度拆解
    N, n_layers, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape

    # 重排並展平成 [N * n_layers * n_heads, Len_q * n_points, n_levels, 1]
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6)  # -> [N, n_layers, n_heads, Len_q, n_points, n_levels, 1]
    sampling_locations = sampling_locations.reshape(N * n_layers * n_heads, Len_q * n_points, n_levels, 1)

    # attention_weights 同樣處理
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4)  # -> [N, n_layers, n_heads, Len_q, n_points, n_levels]
    attention_weights = attention_weights.reshape(N * n_layers * n_heads, Len_q * n_points, n_levels)

    # === Level Normalization ===
    downsample_ratios = temporal_shapes[0] / temporal_shapes.float()  # e.g., [1, 2, 4, 8]
    attention_weights = attention_weights / downsample_ratios.view(1, 1, -1)  # apply normalization

    # 將 sampling_locations 從 normalized [0,1] 映射到真實時間範圍（broadcast temporal_shapes）
    t_float = sampling_locations[..., 0] * temporal_shapes.view(1, 1, -1)  # -> [N', Lq*P, n_levels]

    # 取得左右邊界時間點（整數）
    t_l = t_float.floor().to(torch.int64)   # 下界
    t_h = t_l + 1                            # 上界

    # 線性插值的權重 (距離下界越近，越偏向 margin_l)
    margin_l = 1.0 - (t_float - t_l.float())
    margin_h = 1.0 - margin_l

    # 初始化 flat grid（注意力總和）
    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(temporal_shapes)))  # [N', total_T]
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

    # 處理 lower bound 的貢獻
    valid_mask_l = torch.logical_and(t_l >= 0, t_l < temporal_shapes.view(1, 1, -1))
    idx_l = t_l + level_start_index.view(1, 1, -1)  # 加上 level 起始位置
    idx_l = (idx_l * valid_mask_l).flatten(1, 2)  # -> [N', Lq*P*n_levels]
    weights_l = (attention_weights * margin_l * valid_mask_l).flatten(1)  # -> [N', Lq*P*n_levels]
    flat_grid.scatter_add_(1, idx_l, weights_l)

    # 處理 upper bound 的貢獻
    valid_mask_h = torch.logical_and(t_h >= 0, t_h < temporal_shapes.view(1, 1, -1))
    idx_h = t_h + level_start_index.view(1, 1, -1)
    idx_h = (idx_h * valid_mask_h).flatten(1, 2)
    weights_h = (attention_weights * margin_h * valid_mask_h).flatten(1)
    flat_grid.scatter_add_(1, idx_h, weights_h)

    # 還原維度
    return flat_grid.view(N, n_layers, n_heads, -1)


def attn_map_to_ac_last_layer(temporal_shapes, level_start_index, sampling_locations, attention_weights):
    """
    Args:
        temporal_shapes: [n_levels] - 每個 temporal level 的長度
        level_start_index: [n_levels] - 每個 level 在 flat grid 的起始位置
        sampling_locations: [bs, num_queries, n_heads, n_levels, n_points, 1] - normalized [0,1]
        attention_weights:  [bs, num_queries, n_heads, n_levels, n_points] - 每個 sample 的 weight (已 softmax)

    Returns:
        A_C: [bs, num_queries, total_T] - 每個 query 對 flattened encoder 的 attention map
    """
    bs, num_queries, n_heads, n_levels, n_points, _ = sampling_locations.shape
    total_T = int(torch.sum(temporal_shapes))

    A_C = torch.zeros(bs, num_queries, total_T, device=attention_weights.device)

    # Denormalize: [0,1] → real coordinates
    sampling_t = sampling_locations[..., 0] * temporal_shapes.view(1, 1, 1, -1, 1)  # [bs, nq, h, L, P]
    t_l = sampling_t.floor().clamp(min=0)
    t_h = (t_l + 1).clamp(max=temporal_shapes.view(1, 1, 1, -1, 1) - 1)

    margin_l = 1.0 - (sampling_t - t_l)
    margin_h = 1.0 - margin_l

    # Map to flat index
    # flat index: 加完 level_start_index 後做 clamp
    flat_l = (t_l + level_start_index.view(1, 1, 1, -1, 1)).long().clamp(min=0, max=total_T - 1)
    flat_h = (t_h + level_start_index.view(1, 1, 1, -1, 1)).long().clamp(min=0, max=total_T - 1)


    for margin, flat_idx in [(margin_l, flat_l), (margin_h, flat_h)]:
        weighted_attn = attention_weights * margin  # [bs, nq, h, L, P]
        weighted_attn = weighted_attn.reshape(bs, num_queries, n_heads, -1)   # [bs, nq, h, L*P]
        flat_idx = flat_idx.long().reshape(bs, num_queries, n_heads, -1)      # same

        for b in range(bs):
            for q in range(num_queries):
                indices = flat_idx[b, q]       # [n_heads, L*P]
                weights = weighted_attn[b, q]  # same
                A_C[b, q].scatter_add_(0, indices.view(-1), weights.view(-1))

    return A_C  # [bs, num_queries, total_T]


def attn_map_to_ac_multi_layer(temporal_shapes, level_start_index, sampling_locations_all, attention_weights_all):
    """
    Args:
        temporal_shapes: [n_levels] - 每個 temporal level 的長度
        level_start_index: [n_levels] - 每個 level 在 flat grid 的起始位置
        sampling_locations: [n_layers, bs, num_queries, n_heads, n_levels, n_points, 1] - normalized [0,1]
        attention_weights:  [n_layers, bs, num_queries, n_heads, n_levels, n_points] - 已 softmax 的權重

    Returns:
        A_C: [n_layers, bs, num_queries, total_T] - 每層 decoder 每個 query 對 encoder token 的 attention map
    """
    n_layers, bs, num_queries, n_heads, n_levels, n_points, _ = sampling_locations_all.shape
    total_T = int(torch.sum(temporal_shapes))

    A_C_all = torch.zeros(n_layers, bs, num_queries, total_T, device=attention_weights_all.device)

    # reshape for computation: flatten layer + batch + query
    for l in range(n_layers):
        # Step 1: 取出該層的 sampling 和權重
        sampling_locations = sampling_locations_all[l]  # [bs, nq, h, L, P, 1]
        attention_weights = attention_weights_all[l]   # [bs, nq, h, L, P]

        bs, num_queries, n_heads, n_levels, n_points, _ = sampling_locations.shape
        total_T = int(torch.sum(temporal_shapes))

        A_C = torch.zeros(bs, num_queries, total_T, device=attention_weights.device)

        # Denormalize: [0,1] → real coordinates
        sampling_t = sampling_locations[..., 0] * temporal_shapes.view(1, 1, 1, -1, 1)  # [bs, nq, h, L, P]
        t_l = sampling_t.floor().clamp(min=0)
        t_h = (t_l + 1).clamp(max=temporal_shapes.view(1, 1, 1, -1, 1) - 1)

        margin_l = 1.0 - (sampling_t - t_l)
        margin_h = 1.0 - margin_l

        # Map to flat index
        # flat index: 加完 level_start_index 後做 clamp
        flat_l = (t_l + level_start_index.view(1, 1, 1, -1, 1)).long().clamp(min=0, max=total_T - 1)
        flat_h = (t_h + level_start_index.view(1, 1, 1, -1, 1)).long().clamp(min=0, max=total_T - 1)


        for margin, flat_idx in [(margin_l, flat_l), (margin_h, flat_h)]:
            weighted_attn = attention_weights * margin  # [bs, nq, h, L, P]
            weighted_attn = weighted_attn.reshape(bs, num_queries, n_heads, -1)   # [bs, nq, h, L*P]
            flat_idx = flat_idx.long().reshape(bs, num_queries, n_heads, -1)      # same

            for b in range(bs):
                for q in range(num_queries):
                    indices = flat_idx[b, q]       # [n_heads, L*P]
                    weights = weighted_attn[b, q]  # same
                    A_C[b, q].scatter_add_(0, indices.view(-1), weights.view(-1))

        A_C_all[l] = A_C

    return A_C_all  # [n_layers, bs, num_queries, total_T]


def attn_map_to_ac_multi_layer_vec(temporal_shapes, level_start_index, sampling_locations_all, attention_weights_all):
    """
    Fully vectorized: compute cross-attention maps A_C from Deformable Attention outputs.
    
    Args:
        temporal_shapes: Tensor [n_levels] - each temporal level's length
        level_start_index: Tensor [n_levels] - starting index in the flat encoder
        sampling_locations_all: [n_layers, bs, nq, h, n_levels, n_points, 1] - normalized [0,1]
        attention_weights_all:   [n_layers, bs, nq, h, n_levels, n_points]   - softmax weights

    Returns:
        A_C_all: [n_layers, bs, nq, total_T] - query-to-encoder cross-attention maps
    """
    L, B, Q, H, Lv, P, _ = sampling_locations_all.shape
    device = sampling_locations_all.device
    total_T = int(temporal_shapes.sum())

    # [1,1,1,1,n_levels,1]
    shape_T = temporal_shapes.to(device).view(1, 1, 1, 1, Lv, 1)
    starts = level_start_index.to(device).view(1, 1, 1, 1, Lv, 1)

    # 1. Denormalize & linear interpolation setup
    sampling_t = sampling_locations_all[..., 0] * shape_T
    t_l = sampling_t.floor().clamp(min=0)
    t_h = (t_l + 1).clamp(max=shape_T - 1)

    margin_l = 1.0 - (sampling_t - t_l)
    margin_h = 1.0 - margin_l

    # 2. Flattened encoder indices (clamped)
    flat_l = (t_l + starts).long().clamp(0, total_T - 1)
    flat_h = (t_h + starts).long().clamp(0, total_T - 1)

    # 3. Cat margins and weights along points dimension → [L, B, Q, H, Lv, 2P]
    all_margins = torch.cat([margin_l, margin_h], dim=-1)
    all_weights = torch.cat([attention_weights_all, attention_weights_all], dim=-1)
    all_wts = all_margins * all_weights  # [L, B, Q, H, Lv, 2P]

    all_indices = torch.cat([flat_l, flat_h], dim=-1)  # same shape

    # 4. Reshape to [N, K] for scatter
    N = L * B * Q * H
    K = Lv * 2 * P

    flat_idx = all_indices.permute(0, 1, 2, 3, 5, 4).reshape(N, K)  # [N, K]
    flat_wts = all_wts.permute(0, 1, 2, 3, 5, 4).reshape(N, K)

    # 5. Scatter into [N, total_T]
    A_C_flat = torch.zeros(N, total_T, device=device)
    A_C_flat.scatter_add_(1, flat_idx, flat_wts)

    # 6. Reshape back to [L, B, Q, H, total_T] and sum over heads
    A_C_all = A_C_flat.view(L, B, Q, H, total_T).sum(dim=3)  # [L, B, Q, T]

    return A_C_all.clamp(min=0.0)  # [n_layers, bs, nq, total_T]

def attn_map_to_flat_grid_per_query(temporal_shapes, level_start_index, sampling_locations, attention_weights):
    """
    Args:
        temporal_shapes: Tensor [n_levels] - 每個 temporal level 的長度
        level_start_index: Tensor [n_levels] - 每個 level 在 flat encoder 的起始 index
        sampling_locations: [n_layers, N, Len_q, n_heads, n_levels, n_points, 1] - within [0,1]
        attention_weights:  [n_layers, N, Len_q, n_heads, n_levels, n_points]

    Returns:
        flat_grid: Tensor [n_layers, N, Len_q, total_T] - 每個 query 對 flattened encoder 的 attention map
    """
    sampling_locations = sampling_locations.clamp(0, 1)
    assert not torch.isnan(attention_weights).any(), "attention_weights has NaN"
    assert not torch.isnan(sampling_locations).any(), "sampling_locations has NaN"
    assert (sampling_locations >= 0).all() and (sampling_locations <= 1).all(), "sampling_locations not valid"


    n_layers, N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    device = sampling_locations.device
    total_T = int(temporal_shapes.sum())

    # === Reshape ===
    # [L, N, Q, H, Lv, P] → [L*N*Q*H, Lv, P]
    sampling_locations = sampling_locations[..., 0].reshape(-1, n_levels, n_points)
    attention_weights = attention_weights.reshape(-1, n_levels, n_points)

    # === Interpolation ===
    # Denormalize to real time
    shape_T = temporal_shapes.to(device).view(1, n_levels, 1)  # [1, Lv, 1]
    start_T = level_start_index.to(device).view(1, n_levels, 1)

    t = sampling_locations * shape_T
    t_l = t.floor().clamp(min=0)
    t_h = (t_l + 1).clamp(max=shape_T - 1)

    margin_l = 1.0 - (t - t_l)
    margin_h = 1.0 - margin_l

    # Convert to flat encoder index
    flat_l = (t_l + start_T).long().clamp(0, total_T - 1)
    flat_h = (t_h + start_T).long().clamp(0, total_T - 1)

    # === Flatten to [B', Lv*P]
    B = sampling_locations.shape[0]
    flat_l = flat_l.reshape(B, -1)
    flat_h = flat_h.reshape(B, -1)

    margin_l = margin_l.reshape(B, -1)
    margin_h = margin_h.reshape(B, -1)
    weights = attention_weights.reshape(B, -1)

    # Apply margin
    weight_l = margin_l * weights
    weight_h = margin_h * weights

    # Safe check
    weight_l = torch.nan_to_num(weight_l, nan=0.0, posinf=1.0, neginf=0.0)
    weight_h = torch.nan_to_num(weight_h, nan=0.0, posinf=1.0, neginf=0.0)
    flat_l = flat_l.clamp(0, total_T - 1)
    flat_h = flat_h.clamp(0, total_T - 1)


    # === Scatter-add to output
    grid = torch.zeros(B, total_T, device=device)
    grid.scatter_add_(1, flat_l, weight_l)
    grid.scatter_add_(1, flat_h, weight_h)
    assert not torch.isnan(grid).any(), "NaN in A_C"
    assert (grid >= 0).all(), "Negative attention weights in grid"


    # === Reshape to [n_layers, N, Len_q, total_T]
    grid = grid.view(n_layers, N, Len_q, n_heads, total_T)
    return grid.mean(dim=3)  # [n_layers, N, Len_q, total_T]


def compute_G_D(A_C):
    # A_C: [n_layers, bs, num_queries, total_T]
    G_D = torch.sqrt(torch.matmul(A_C, A_C.transpose(-1, -2)) + 1e-6)  # [n_layers, bs, num_queries, num_queries]
    return G_D
