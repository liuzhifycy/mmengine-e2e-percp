"""
MMEngine-Lite 数据集模块

提供 2D/3D 目标检测的数据集支持:
- CocoDataset: COCO 格式数据集 (2D)
- KittiDataset: KITTI 3D 检测数据集
- NuScenesDataset: nuScenes 3D 检测数据集（占位）

点云数据处理 Pipeline:
- LoadPointsFromFile: 加载点云文件
- LoadAnnotations3D: 加载 3D 标注
- VoxelGenerator: 点云体素化
- 3D 数据增强变换
"""

# 从 mmdet 导入核心数据集
from mmdet.datasets import CocoDataset

# 从 mmengine 导入数据集构建函数
from mmengine.registry import DATASETS, TRANSFORMS

# 从 mmcv 导入基础数据变换
from mmcv.transforms import LoadImageFromFile

# 从 mmdet 导入检测相关数据变换
from mmdet.datasets.transforms import (
    LoadAnnotations,
    PackDetInputs,
    RandomFlip,
    Resize,
)

# 3D 数据集
from .kitti_dataset import Det3DDataSample, KittiDataset, NuScenesDataset

# 3D 点云 Pipeline
from .pipelines import (
    DynamicVoxelization,
    GlobalRotScaleTrans,
    LoadAnnotations3D,
    LoadPointsFromFile,
    ObjectRangeFilter,
    PointsRangeFilter,
    PointShuffle,
    RandomFlip3D,
    VoxelGenerator,
)

# RandomResize 通过 registry 获取
RandomResize = TRANSFORMS.get("RandomResize")

# 数据集构建函数
build_dataset = DATASETS.build

__all__ = [
    # === 2D 数据集 ===
    "CocoDataset",
    # === 3D 数据集 ===
    "KittiDataset",
    "NuScenesDataset",
    "Det3DDataSample",
    # 构建函数
    "build_dataset",
    "DATASETS",
    "TRANSFORMS",
    # === 2D 数据变换 ===
    "LoadImageFromFile",
    "LoadAnnotations",
    "Resize",
    "RandomResize",
    "RandomFlip",
    "PackDetInputs",
    # === 3D 数据变换 ===
    "LoadPointsFromFile",
    "LoadAnnotations3D",
    "VoxelGenerator",
    "DynamicVoxelization",
    "PointsRangeFilter",
    "ObjectRangeFilter",
    "RandomFlip3D",
    "GlobalRotScaleTrans",
    "PointShuffle",
]
