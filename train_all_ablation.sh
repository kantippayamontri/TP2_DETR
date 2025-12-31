for split_id in {0..9}
do
    # Run the command for each split_id
    CUDA_LAUNCH_BLOCKING=1 python ActionFormer/main.py \
        --model_name "Thumos14_SalientType_convPlus" \
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
        --split_id $split_id \
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
        --salient_loss \
        --salient_loss_coef 3 \
        # --plainFPN \
        # --num_feature_levels 1 \
        # --aux_loss \

    echo "Cleaning up any remaining Python processes..."
    # killall python
    sleep 2  # Optional: Add a small delay if necessary for cleanup
done