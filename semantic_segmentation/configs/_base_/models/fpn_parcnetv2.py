# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="ParCNetV2",
        in_chans=3,
        depths=[3, 3, 12, 3],
        dims=[64, 128, 320, 512],
        drop_path_rate=0.1,
        out_indices=[0, 1, 2, 3],
        norm_cfg=norm_cfg,
    ),
    neck=dict(
        type="FPN", in_channels=[64, 128, 320, 512], out_channels=512, num_outs=4,
    ),
    decode_head=dict(
        type="FPNHead",
        in_channels=[512, 512, 512, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

