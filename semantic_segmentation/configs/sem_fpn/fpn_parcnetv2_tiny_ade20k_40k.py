_base_ = [
    "../_base_/models/fpn_parcnetv2.py",
    "../_base_/datasets/ade20k.py",
    "../_base_/default_runtime.py",
]
# model settings
model = dict(
    pretrained="pretrained/parcnetv2_tiny.pth.tar",
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 12, 3],
        dims=[64, 128, 320, 512],
        drop_path_rate=0.2,
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150),
)

# we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
gpu_multiples = 2
# optimizer
optimizer = dict(type="AdamW", lr=0.0001 * gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=80000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000 // gpu_multiples)
evaluation = dict(interval=8000 // gpu_multiples, metric="mIoU")
data = dict(samples_per_gpu=4)
