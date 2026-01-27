"""
MMEngine-Lite 数据集模块

直接从 mmdet 和 mmengine 重导出数据集相关组件，
提供统一的数据集构建接口。
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

# RandomResize 通过 registry 获取
RandomResize = TRANSFORMS.get("RandomResize")

# 数据集构建函数
build_dataset = DATASETS.build

__all__ = [
    # 数据集类
    "CocoDataset",
    # 构建函数
    "build_dataset",
    "DATASETS",
    "TRANSFORMS",
    # 数据变换
    "LoadImageFromFile",
    "LoadAnnotations",
    "Resize",
    "RandomResize",
    "RandomFlip",
    "PackDetInputs",
]
