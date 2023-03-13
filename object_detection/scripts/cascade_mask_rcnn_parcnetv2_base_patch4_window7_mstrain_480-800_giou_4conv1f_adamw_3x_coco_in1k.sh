MODEL=cascade_mask_rcnn_parcnetv2_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k

bash tools/dist_train.sh \
    configs/cascade_mask_rcnn/${MODEL}.py 8 \
    --work-dir output/${MODEL} --seed 0 \
    > log/${MODEL}.log 2>&1