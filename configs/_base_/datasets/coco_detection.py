# COCO 检测数据集配置

# 数据集类型和根目录
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# 后端配置 (本地磁盘)
backend_args = None

# ===================== 数据增强 Pipeline =====================

# 训练 pipeline
train_pipeline = [
    # 加载图像
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # 加载标注
    dict(type='LoadAnnotations', with_bbox=True),
    # 随机缩放
    dict(
        type='RandomResize',
        scale=(1333, 800),  # 目标尺寸
        ratio_range=(0.5, 2.0),  # 缩放比例范围
        keep_ratio=True,  # 保持宽高比
    ),
    # 随机裁剪
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(384, 600),  # 裁剪尺寸范围
        allow_negative_crop=True,  # 允许无目标的裁剪
    ),
    # 随机翻转
    dict(type='RandomFlip', prob=0.5),  # 50% 概率水平翻转
    # 打包数据
    dict(type='PackDetInputs'),
]

# 测试/验证 pipeline
test_pipeline = [
    # 加载图像
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # 缩放到固定尺寸
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # 加载标注 (用于评估)
    dict(type='LoadAnnotations', with_bbox=True),
    # 打包数据
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# ===================== DataLoader 配置 =====================

# 训练 DataLoader
train_dataloader = dict(
    batch_size=2,  # 每个 GPU 的 batch size
    num_workers=2,  # 数据加载进程数
    persistent_workers=True,  # 保持 worker 进程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 随机采样
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 按宽高比分组
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  # 训练集标注
        data_prefix=dict(img='train2017/'),  # 图像目录
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # 过滤配置
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

# 验证 DataLoader
val_dataloader = dict(
    batch_size=1,  # 验证时 batch size 为 1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,  # 不丢弃最后不完整的 batch
    sampler=dict(type='DefaultSampler', shuffle=False),  # 顺序采样
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',  # 验证集标注
        data_prefix=dict(img='val2017/'),
        test_mode=True,  # 测试模式
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# 测试 DataLoader (与验证相同)
test_dataloader = val_dataloader

# ===================== 评估器配置 =====================

# 验证评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',  # 评估边界框
    format_only=False,  # 是否只格式化结果不评估
    backend_args=backend_args,
)

# 测试评估器
test_evaluator = val_evaluator
