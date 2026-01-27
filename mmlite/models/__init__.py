"""
Models 模块 - 模型组件统一导出

提供 backbones、necks、dense_heads、detectors 等模型组件的统一接口，
直接复用 mmdet 已有实现，并提供自定义模型示例。

新增 3D 点云检测组件（PointPillars）:
- voxel_encoders: PillarFeatureNet 体素特征编码
- middle_encoders: PointPillarsScatter 散射到伪图像
- backbones3d: SECOND 骨干网络
"""

# 2D Detection
from .backbones import ResNet
from .custom import (
    LightweightHead,
    MobileNetLiteBackbone,
    SimpleCNNBackbone,
    SimpleDetectionHead,
)
from .dense_heads import RetinaHead
from .detectors import RetinaNet
from .necks import FPN

# 3D Detection (PointPillars)
from .backbones3d import SECOND, SECONDFPN
from .middle_encoders import PointPillarsScatter
from .voxel_encoders import DynamicPillarFeatureNet, PillarFeatureNet

__all__ = [
    # === 2D Detection ===
    # Backbones
    "ResNet",
    # Custom Backbones
    "SimpleCNNBackbone",
    "MobileNetLiteBackbone",
    # Necks
    "FPN",
    # Dense Heads
    "RetinaHead",
    # Custom Heads
    "SimpleDetectionHead",
    "LightweightHead",
    # Detectors
    "RetinaNet",
    # === 3D Detection (PointPillars) ===
    # Voxel Encoders
    "PillarFeatureNet",
    "DynamicPillarFeatureNet",
    # Middle Encoders
    "PointPillarsScatter",
    # 3D Backbones
    "SECOND",
    "SECONDFPN",
]
