_base_ = [
    './yolo11m_coco_reimpl.py'
]

# 改名避免自动注入
dataset_root = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/coco/'

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root=dataset_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
        ),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(type='RandomAffine', scaling_ratio_range=(0.1, 2), border=(-320, -320)),
            dict(type='MixUp', img_scale=(640, 640), ratio_range=(0.8, 1.6), pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=dataset_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)

# 工作目录
work_dir = 'work_dirs/yolo11m_coco_transfer'
