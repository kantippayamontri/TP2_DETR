#!/bin/bash

# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================
# These environment variables must be set before Python starts to ensure
# deterministic behavior across multiple training runs with the same seed.

# Set Python hash seed for reproducibility
# PYTHONHASHSEED=0 ensures deterministic hash-based operations in Python,
# including dictionary/set ordering and hash() function results.
# This must be set before Python interpreter starts.
export PYTHONHASHSEED=0

# Set CUBLAS workspace config for deterministic CUDA operations
# CUBLAS_WORKSPACE_CONFIG=:4096:8 allocates 4096 bytes across 8 workspaces
# for deterministic cuBLAS operations. This is required when using
# torch.use_deterministic_algorithms(True) with CUDA operations.
# Without this, some CUDA operations may fail or produce non-deterministic results.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_LAUNCH_BLOCKING=1 python ActionFormer/main.py \
    --model_name "Thumos14_reweight2" \
    --cfg_path "./ActionFormer/config/Thumos14_CLIP_zs_50_8frame.yaml" \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 35 \
    --postprocess_type "class_agnostic" \
    --postprocess_topk 100 \
    --num_queries 40 \
    --actionness_loss_coef 3 \
    --norm_embed \
    --exp_logit_scale \
    --proposals_weight_type "after_softmax" \
    --enable_classAgnostic \
    --refine_drop_saResidual \
    --salient_loss \
    --salient_loss_coef 5 \
    --split_id 0 \
    --enc_stem_layers 1 \
    --enc_branch_layers 3 \
    --enc_layers 3 \
    --dec_layers 5 \
    --enc_n_points 4 \
    --dec_n_points 4 \
    --win_size 9 \
    --salient_upsample_type "nearest" \
    --salient_aggregate_type "max" \
    --separate_predict_head \
    --eval_proposal \
    --backbone_init_method "v0" \
    --aux_loss \
    --enlarge_aux_loss \
    --enable_softSalient \
    --as_calibration