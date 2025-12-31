# # re-compile CUDA operator
# cd ActionFormer/models/ops/
# rm -rf build dist MultiScaleDeformableAttention.egg-info
# sh ./make.sh
# cd ../../..

# run
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
    --as_calibration \
    # --enable_neck
    # --use_SGP \
    # --train_interval 5 \
    # --enable_refine \
    # --aux_loss \
    # --vis_con_loss \
    # --vis_con_loss_coef 1e-4 \

# CUDA_LAUNCH_BLOCKING=1 python ActionFormer/main.py \
#     --model_name "Train_ActivityNet13" \
#     --cfg_path "./ActionFormer/config/ActivityNet13_CLIP_zs_50.yaml" \
#     --batch_size 16 \
#     --target_type "prompt" \
#     --epochs 100 \
#     --num_queries 30 \
#     --postprocess_type "class_agnostic" \
#     --postprocess_topk 100 \
#     --rescale_length 256 \
#     --norm_embed \
#     --exp_logit_scale \
#     --proposals_weight_type "after_softmax" \
#     --enable_classAgnostic \
#     --actionness_loss_coef 3 \
#     --refine_drop_saResidual \
#     --split_id 0 \
#     --enc_stem_layers 1 \
#     --enc_branch_layers 3 \
#     --enc_layers 2 \
#     --dec_layers 2 \
#     --enc_n_points 4 \
#     --dec_n_points 4 \
#     --win_size 17 \
#     --salient_upsample_type "linear" \
#     --salient_aggregate_type "mix_adaptive" \
#     --separate_predict_head \
#     --salient_loss \
#     --train_interval 1 \
#     --lr 1e-4 \
#     # --lr 5e-5 \
#     # --lr_backbone 2e-4 \
#     # --lr_backbone 1e-2 \
#     # --aux_loss \
#     # --enable_refine \
    