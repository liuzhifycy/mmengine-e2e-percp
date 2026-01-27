"""
轻量级模型配置示例

使用自定义的 MobileNetLiteBackbone + LightweightHead
适合移动端部署，参数量和计算量较小

注意: 使用自定义模型前需要先导入 mmlite.models.custom 模块进行注册
"""
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# 自定义插件 - 注册自定义模块
custom_imports = dict(
    imports=['mmlite.models.custom'],
    allow_failed_imports=False,
)

# 模型配置 - 轻量级版本
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    # 使用轻量级 backbone
    backbone=dict(
        type='MobileNetLiteBackbone',
        in_channels=3,
        width_mult=1.0,
        out_indices=(0, 1, 2, 3),  # 输出 4 个阶段的特征图
        init_cfg=dict(type='Kaiming', layer='Conv2d'),
    ),
    # FPN 使用较少通道
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],  # 根据 MobileNetLiteBackbone 的输出
        out_channels=128,  # 较少的通道数
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
    ),
    # 使用轻量级检测头
    bbox_head=dict(
        type='LightweightHead',
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        num_convs=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=0.11,
            loss_weight=1.0,
        ),
    ),
    # 训练配置
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

# 优化器
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
)

# 学习率策略
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

# 工作目录
work_dir = './work_dirs/mobilenet_lite_retinanet'
