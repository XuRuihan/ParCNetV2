_base_ = [
    "../_base_/models/upernet_parcnetv2.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]
crop_size = (512, 512)

model = dict(
    pretrained="pretrained/parcnetv2_small.pth.tar",
    backbone=dict(
        in_chans=3,
        depths=[3, 9, 24, 3],
        dims=[64, 128, 320, 512],
        drop_path_rate=0.3,
        init_cfg=dict(
            type="Pretrained", checkpoint="pretrained/parcnetv2_small.pth.tar"
        ),
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=150,),
    auxiliary_head=dict(in_channels=320, num_classes=150),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)

# ConvNeXt Optimizer
optimizer = dict(
    constructor="LRDOptimizerConstructor",
    _delete_=True,
    type="AdamW",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
)

# # AdamW optimizer, no weight decay for position embedding & layer norm in backbone
# optimizer = dict(
#     _delete_=True,
#     type="AdamW",
#     lr=6e-5,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             "absolute_pos_embed": dict(decay_mult=0.0),
#             "relative_position_bias_table": dict(decay_mult=0.0),
#             "norm": dict(decay_mult=0.0),
#         }
#     ),
# )

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# runner = dict(type="IterBasedRunnerAmp")

# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
