_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    type='YOLO11',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(type='YOLO11CSPDarknet'),
    neck=dict(type='YOLO11PAFPN'),
    bbox_head=dict(type='YOLO11Head', nc=80),
    init_cfg=dict(type='Pretrained', checkpoint='yolo11m_mm.pth')
)

# Optimization
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005),
    clip_grad=dict(max_norm=10.0, norm_type=2)
)

# Augmentation Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-320, -320)), # YOLO style affine
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Dataloader with MultiImageMixDataset
train_dataloader = dict(
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root='data/coco/',
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ]
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader