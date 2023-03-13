MODEL=upernet_parcnetv2_base_512_160k_ms

bash dist_train.sh \
    configs/upernet/${MODEL}.py 8 \
    --work-dir output/${MODEL} --seed 0 \
    > log/${MODEL}.log 2>&1