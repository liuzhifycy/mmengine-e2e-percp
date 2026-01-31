_base_ = [
    '../_base_/default_runtime.py'
]

# 导入自定义模型
custom_imports = dict(
    imports=['mmlite.models'],
    allow_failed_imports=False
)

# ======================== 数据集配置 ========================
dataset_root = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd/combined/'

# 类别配置 - 只有一个类别：车牌
class_name = ('plate',)
num_classes = 1
metainfo = dict(classes=class_name)

# ======================== 模型配置 ========================
model = dict(
    type='YOLO11',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(type='YOLO11CSPDarknet'),
    neck=dict(type='YOLO11PAFPN'),
    bbox_head=dict(
        type='YOLO11Head',
        nc=num_classes,  # 修改为1类
    ),
    # 加载预训练权重 (已适配，排除了分类头)
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/ubuntu/e2e-pecp-pdp/mmengine-lite/checkpoints/yolo11m_plate_pretrain.pth'
    )
)

# ======================== 数据增强 Pipeline ========================
# 简化的训练 Pipeline，不使用 Mosaic
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# ======================== DataLoader 配置 ========================
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

# ======================== 评估器配置 ========================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=dataset_root + 'val.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=dataset_root + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

# ======================== 训练配置 ========================
# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=0.0001,
        begin=0,
        end=50,
        by_epoch=True
    )
]

# 训练循环配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=5
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ======================== 其他配置 ========================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

# 工作目录
work_dir = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite/work_dirs/yolo11m_plate_detection'

# 随机种子
randomness = dict(seed=42, deterministic=False)

# 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
