# #!/bin/bash


# # # Define the list of split_ids you want to run
# # split_ids=(0 3)
# # # Loop through each specified split_id
# # for split_id in "${split_ids[@]}"

# # Loop through split_id from 0 to 9
# for split_id in {7..8}
# do
#     # Run the command for each split_id
#     CUDA_LAUNCH_BLOCKING=1 python ActionFormer/main.py \
#         --model_name "Thumos14_salientEmbed" \
#         --cfg_path "./ActionFormer/config/Thumos14_CLIP_zs_50_8frame.yaml" \
#         --batch_size 16 \
#         --lr 1e-4 \
#         --epochs 35 \
#         --postprocess_type "class_agnostic" \
#         --postprocess_topk 100 \
#         --num_queries 40 \
#         --actionness_loss_coef 3 \
#         --norm_embed \
#         --exp_logit_scale \
#         --proposals_weight_type "after_softmax" \
#         --enable_classAgnostic \
#         --refine_drop_saResidual \
#         --salient_loss \
#         --salient_loss_coef 3 \
#         --split_id $split_id \
#         --enc_stem_layers 1 \
#         --enc_branch_layers 3 \
#         --enc_layers 3 \
#         --dec_layers 5 \
#         --enc_n_points 4 \
#         --dec_n_points 4 \
#         --win_size 9 \
#         --salient_upsample_type "nearest" \
#         --salient_aggregate_type "max" \
#         --separate_predict_head \
#         --eval_proposal \
#         --backbone_init_method "v0" \
#         --aux_loss \
#         # --sparsity_loss \
#         # --sparsity_loss_coef 1
#         # --enlarge_aux_loss \
#         # --enable_softSalient \
#         # --as_calibration \
#         # --weightDict 0.9 \
#         # --as_calibration \
#         # --use_DyT
#         # --enable_softSalient \
#         # --enable_edgePunish 
#         # --plainFPN
#         # --use_SAattn
#         # --use_SGP \
#         # --use_SAPM \
#         # --tIoUw_bbox_loss
#         # --enable_neck
#         # --enable_freqCalibrate \
#         # --use_SGP \
#         # --enable_refine \
#         # --target_type "combined_description"\
#         # --enable_refine \
#         # --enable_refine_freq
        

#     # Optional: Clean up or ensure no processes are lingering
#     echo "Cleaning up any remaining Python processes..."
#     killall python
#     sleep 2  # Optional: Add a small delay if necessary for cleanup
# done



# # for split_id in {0..9}
# split_ids=(9)
# # Loop through each specified split_id
# for split_id in "${split_ids[@]}"
for split_id in {0..9}
do
    # Run the command for each split_id
    CUDA_LAUNCH_BLOCKING=1 python ActionFormer/main.py \
        --model_name "Thumos14_50_testSubmit" \
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
        --aux_loss \
        # --save_result


        # --use_TETAD \
        # --use_Gating \
        # --no_normalizedqkv
        # --with_box_refine \
        # --use_LFT \
        # --separate_LFOLFT \
        # --weightDict 0.9 \

        # --two_stage
        # --dino_two_stage

        # --feedback_loss \
        # --feedback_loss_coef 5 \

        # --use_VideoSA \
        # --semantic_guided_loss \
        # --semantic_guided_loss_coef 2 \
        # --SA_LastOnly \

        # --use_SAPM \

        

        

        

        # --use_decouple \
        # --visualize_decouple \
        
        # --enlarge_aux_loss \
        # --sparsity_loss \
        # --sparsity_loss_coef 1
        # --enlarge_aux_loss \
        # --enable_softSalient \
        # --as_calibration \
        # --weightDict 0.9 \
        # --as_calibration \
        # --use_DyT
        # --enable_softSalient \
        # --enable_edgePunish 
        # --plainFPN
        # --use_SAattn
        # --use_SGP \
        # --use_SAPM \
        # --tIoUw_bbox_loss
        # --enable_neck
        # --enable_freqCalibrate \
        # --use_SGP \
        # --enable_refine \
        # --target_type "combined_description"\
        # --enable_refine \
        # --enable_refine_freq
        
    break

    # Optional: Clean up or ensure no processes are lingering
    echo "Cleaning up any remaining Python processes..."
    # killall python
    sleep 2  # Optional: Add a small delay if necessary for cleanup
done
