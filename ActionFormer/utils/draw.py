import torch
import numpy as np
import matplotlib
import os
import math
import seaborn as sns

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_attention_map(attention_weights, level_start_index, decouple_type=None, normalize=False, epoch=-1):
    """
    Plot the attention map with boundaries marked for each temporal level.
    
    Parameters:
        attention_weights (tensor): The attention weights tensor of shape [B, sum(t_i)].
        temporal_shapes (tensor): The tensor containing the length of each temporal level.
        level_start_index (tensor): The tensor containing the start index of each temporal level.
        normalize (bool): Whether to normalize the attention weights.
        decouple_type (str/None): If decoupling, specify which part you focus on ('cls' or 'loc') 
    """
    # Use no_grad to prevent gradient tracking
    with torch.no_grad():
        batch_size = attention_weights.shape[0]
        rows = (batch_size + 1) // 2

        # 圖片更寬、更扁
        fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 2.5))
        axes = axes.flatten()

        for i in range(batch_size):
            attention_weights_flat = attention_weights[i].flatten().detach().cpu().numpy()
            level_boundaries = [level_start_index[0].item()] + level_start_index[1:].cpu().numpy().tolist()

            if normalize:
                attention_weights_flat = (attention_weights_flat - np.min(attention_weights_flat)) / \
                                         (np.max(attention_weights_flat) - np.min(attention_weights_flat) + 1e-6)

            axes[i].imshow(attention_weights_flat[np.newaxis, :], aspect="auto", cmap="viridis")

            for boundary in level_boundaries:
                axes[i].axvline(x=boundary, color='r', linestyle='--')

            axes[i].set_title(f"Sample {i+1} Attention Map", fontsize=10)
            axes[i].set_xlabel("Temporal Index", fontsize=8)

            # 去掉 y 軸刻度與標籤
            axes[i].set_yticks([])
            axes[i].set_ylabel("")

        # 去除多出來的 subplot（如果 batch 是奇數）
        for j in range(batch_size, len(axes)):
            fig.delaxes(axes[j])

        # 在整體圖上加一個 colorbar
        if epoch==-1:
            title = f'Attention({decouple_type}).jpg' if decouple_type is not None else 'Attention.jpg'
        else:
            title = f'Attention_{epoch}({decouple_type}).jpg' if decouple_type is not None else f'Attention_{epoch}.jpg'
        plt.colorbar(axes[0].images[0], ax=axes)
        plt.savefig(title)  # Save the plot as a file
        plt.close(fig)
    
        
def plot_SAattention_map(attention_weights, gt_classes_o=None, normalize=False, epoch=-1, split_id=-1):
    """
    Plot the attention map with boundaries marked for each temporal level.
    
    Parameters:
        attention_weights (tensor): The attention weights tensor of shape [B, sum(t_i)].
        temporal_shapes (tensor): The tensor containing the length of each temporal level.
        level_start_index (tensor): The tensor containing the start index of each temporal level.
        normalize (bool): Whether to normalize the attention weights.
        decouple_type (str/None): If decoupling, specify which part you focus on ('cls' or 'loc') 
    """
    # Use no_grad to prevent gradient tracking
    with torch.no_grad():
        batch_matched_queries_size = attention_weights.shape[0]
        rows = math.ceil(batch_matched_queries_size / 4)

        # 圖片更寬、更扁
        fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 2.5))
        axes = axes.flatten()

        for i in range(batch_matched_queries_size):
            if i >= len(axes):
                break  # Or raise a more informative error
            attention_weights_flat = attention_weights[i].flatten().detach().cpu().numpy()

            if normalize:
                attention_weights_flat = (attention_weights_flat - np.min(attention_weights_flat)) / \
                                         (np.max(attention_weights_flat) - np.min(attention_weights_flat) + 1e-6)

            axes[i].imshow(attention_weights_flat[np.newaxis, :], aspect="auto", cmap="viridis")

            axes[i].set_title(f"Sample {i+1} Attention Map (gt={gt_classes_o[i]})", fontsize=10)
            axes[i].set_xlabel("Temporal Index", fontsize=8)

            # 去掉 y 軸刻度與標籤
            axes[i].set_yticks([])
            axes[i].set_ylabel("")

        # 去除多出來的 subplot（如果 batch 是奇數）
        for j in range(batch_matched_queries_size, len(axes)):
            fig.delaxes(axes[j])

        # 在整體圖上加一個 colorbar
        if epoch==-1:
            title = f'SA_Attention.jpg'
        else:
            title = f'SA_Attention_{epoch}.jpg'
            
        
        if not os.path.exists(f'SAattention/50_test_split{split_id}'):
            os.makedirs(f'SAattention/50_test_split{split_id}')
        save_path = os.path.join(f'SAattention/50_test_split{split_id}', title)

        plt.colorbar(axes[0].images[0], ax=axes)
        plt.savefig(save_path)  # Save the plot as a file
        plt.close(fig)

# # Example usage
# # Simulated data based on your input
# attention_weights = torch.tensor([ 0.0572, 0.9310, 0.4219, 1.8597, 1.3904, 2.0896, 2.4666, 3.9223, 
#                                   3.3681, 5.3814, 3.5669, 3.8703, 3.4443, 3.6525, 5.4680, 4.5333, 
#                                   4.8072, 4.0922, 3.6524, 3.2269, 3.3650, 5.0186, 2.8496, 2.5195,
#                                   2.2389, 2.5760, 2.8606, 2.5310, 3.4708, 4.3237, 4.1471, 4.0568]) # Example tensor
# temporal_shapes = torch.tensor([128, 64, 32, 16])  # Example temporal shapes
# level_start_index = torch.tensor([0, 128, 192, 224])  # Example start indices

# # Call the function with normalization set to True or False
# plot_attention_map(attention_weights, temporal_shapes, level_start_index, normalize=True)


def visualize_gd(A_C, title='G_D: Query-to-Query Attention', cmap='viridis', query_labels=None, epoch=-1, training_mode=False):
    """
    Args:
        A_C: Tensor of shape [bs, num_queries, total_T] - 每個 query 的 cross-attention map
        title: plot 標題
        cmap: heatmap 顏色風格
        query_labels: Optional list of strings，標記 query 序號（如 ['Q0', 'Q1', ...]）
    """
    if training_mode:
        save_dir = 'SelfDETR_train'
    else:
        save_dir = 'SelfDETR'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():
        bs, num_queries, total_T = A_C.shape

        for b in range(bs):
            # Step 1: G_D = sqrt(A_C A_C^T)
            AC = A_C[b]  # [num_queries, total_T]
            GD = torch.sqrt(torch.matmul(AC, AC.T) + 1e-6).detach().cpu().numpy()# [num_queries, num_queries]

            # Step 2: plot
            plt.figure(figsize=(6, 5))
            sns.heatmap(GD, cmap=cmap, 
                        xticklabels=query_labels if query_labels else [f"Q{i}" for i in range(num_queries)],
                        yticklabels=query_labels if query_labels else [f"Q{i}" for i in range(num_queries)])
            plt.title(f"{title} (Batch {b})")
            plt.xlabel("Query")
            plt.ylabel("Query")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"Epoch_{epoch}_Batch_{b}_GD.jpg")
            plt.savefig(save_path)
            plt.close()

def visualize_selfattn(self_attn_map, title='Self-Attn Map: Query-to-Query Attention', cmap='viridis', query_labels=None, epoch=-1, training_mode=False):
    """
    Args:
        A_C: Tensor of shape [bs, num_queries, total_T] - 每個 query 的 cross-attention map
        title: plot 標題
        cmap: heatmap 顏色風格
        query_labels: Optional list of strings，標記 query 序號（如 ['Q0', 'Q1', ...]）
    """
    if training_mode:
        save_dir = 'SelfDETR_train'
    else:
        save_dir = 'SelfDETR'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        bs, num_queries, _ = self_attn_map.shape

        for b in range(bs):
            map = self_attn_map[b].detach().cpu().numpy()
            # Step 2: plot
            plt.figure(figsize=(6, 5))
            sns.heatmap(map, cmap=cmap, 
                        xticklabels=query_labels if query_labels else [f"Q{i}" for i in range(num_queries)],
                        yticklabels=query_labels if query_labels else [f"Q{i}" for i in range(num_queries)])
            plt.title(f"{title} (Batch {b})")
            plt.xlabel("Query")
            plt.ylabel("Query")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"Epoch_{epoch}_Batch_{b}_SelfAttnMap.jpg")
            plt.savefig(save_path)
            plt.close()


