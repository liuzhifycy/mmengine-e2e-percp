# PointPillars 配置文件 for KITTI 3D 目标检测
# 基于论文: PointPillars: Fast Encoders for Object Detection from Point Clouds

# ===================== 点云范围和体素设置 =====================
# KITTI 标准设置
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # [x_min, y_min, z_min, x_max, y_max, z_max]
voxel_size = [0.16, 0.16, 4]  # [x, y, z]

# 计算网格大小
# grid_size = (point_cloud_range[3:6] - point_cloud_range[:3]) / voxel_size
# grid_size = [432, 496, 1]  # [x, y, z]

# ===================== 模型配置 =====================
model = dict(
    type='PointPillars',  # 需要在 mmdet3d 中注册或自行实现检测器
    
    # 体素编码器 - 将点云体素化并编码
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,  # x, y, z, intensity
        feat_channels=(64,),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),
    
    # 中间编码器 - 将体素特征散射到伪图像
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[496, 432],  # [H, W] = [ny, nx]
    ),
    
    # 骨干网络 - 提取多尺度特征
    backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    ),
    
    # 特征金字塔网络 - 融合多尺度特征
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    ),
    
    # 检测头 - 输出 3D 边界框
    bbox_head=dict(
        type='Anchor3DHead',  # 需要从 mmdet3d 导入或自行实现
        num_classes=3,  # Car, Pedestrian, Cyclist
        in_channels=384,  # 128 * 3 from FPN
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Car
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Pedestrian
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],  # Cyclist
            ],
            sizes=[
                [3.9, 1.6, 1.56],  # Car
                [0.8, 0.6, 1.73],  # Pedestrian
                [1.76, 0.6, 1.73],  # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2,
        ),
    ),
    
    # 训练配置
    train_cfg=dict(
        assigner=[
            dict(  # Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
            dict(  # Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    
    # 测试配置
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),
)

# ===================== 数据配置 =====================
dataset_type = 'KittiDataset'  # 需要实现或使用 mmdet3d 的
data_root = 'data/kitti/'

# 类别名称
class_names = ['Car', 'Pedestrian', 'Cyclist']

# 数据增强 Pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type='PointShuffle'),
    dict(
        type='VoxelGenerator',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=32,
        max_voxels=(16000, 40000),
    ),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='VoxelGenerator',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=32,
        max_voxels=(16000, 40000),
    ),
]

# DataLoader 配置
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
)
test_evaluator = val_evaluator

# ===================== 训练配置 =====================

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# 学习率调度
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=1e-5,
        begin=0,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# 训练循环
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认 Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 日志配置
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 可视化
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 加载权重
load_from = None
resume = False
