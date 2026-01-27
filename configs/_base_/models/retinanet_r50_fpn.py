# RetinaNet with ResNet50-FPN 模型配置
# 参考: https://arxiv.org/abs/1708.02002

# 模型配置
model = dict(
    type='RetinaNet',
    # 数据预处理器
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # ImageNet 均值和标准差
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,  # BGR 转 RGB
        pad_size_divisor=32,  # padding 到 32 的倍数
    ),
    # Backbone: ResNet50
    backbone=dict(
        type='ResNet',
        depth=50,  # ResNet 深度
        num_stages=4,  # 使用的 stage 数量
        out_indices=(0, 1, 2, 3),  # 输出特征层索引 (C2, C3, C4, C5)
        frozen_stages=1,  # 冻结 stem 和 stage1
        norm_cfg=dict(type='BN', requires_grad=True),  # BatchNorm 配置
        norm_eval=True,  # 评估时 BN 使用全局统计量
        style='pytorch',  # pytorch 风格 (stride 在 3x3 卷积)
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),  # 预训练权重
    ),
    # Neck: FPN (Feature Pyramid Network)
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # ResNet 各 stage 输出通道数
        out_channels=256,  # FPN 输出通道数
        start_level=1,  # 从 C3 开始
        add_extra_convs='on_input',  # 在输入特征上添加额外卷积
        num_outs=5,  # 输出 5 个尺度的特征 (P3-P7)
    ),
    # Head: RetinaHead (分类 + 回归)
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,  # COCO 类别数
        in_channels=256,  # 输入通道数 (来自 FPN)
        stacked_convs=4,  # 堆叠卷积层数
        feat_channels=256,  # 特征通道数
        # Anchor 配置
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,  # 基础尺度
            scales_per_octave=3,  # 每个 octave 的尺度数
            ratios=[0.5, 1.0, 2.0],  # 宽高比
            strides=[8, 16, 32, 64, 128],  # 各层 stride (对应 P3-P7)
        ),
        # 边界框编码器
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        # 分类损失: Focal Loss
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,  # 使用 sigmoid 多标签分类
            gamma=2.0,  # focusing parameter
            alpha=0.25,  # 正负样本平衡因子
            loss_weight=1.0,
        ),
        # 回归损失: L1 Loss
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0,
        ),
    ),
    # 训练配置
    train_cfg=dict(
        # Assigner: 分配正负样本
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,  # IoU >= 0.5 为正样本
            neg_iou_thr=0.4,  # IoU < 0.4 为负样本
            min_pos_iou=0,  # 最小正样本 IoU
            ignore_iof_thr=-1,  # 忽略区域 IoF 阈值
        ),
        sampler=dict(
            type='PseudoSampler',  # RetinaNet 使用所有样本
        ),
        allowed_border=-1,  # 允许超出边界的 anchor
        pos_weight=-1,  # 正样本权重 (-1 表示不使用)
        debug=False,
    ),
    # 测试配置
    test_cfg=dict(
        nms_pre=1000,  # NMS 前保留的候选框数
        min_bbox_size=0,  # 最小边界框尺寸
        score_thr=0.05,  # 分数阈值
        nms=dict(type='nms', iou_threshold=0.5),  # NMS 配置
        max_per_img=100,  # 每张图最多检测框数
    ),
)
