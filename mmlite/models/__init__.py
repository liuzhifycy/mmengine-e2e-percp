"""
Models 模块 - 模型组件统一导出

提供 backbones、necks、dense_heads、detectors 等模型组件的统一接口，
直接复用 mmdet 已有实现，并提供自定义模型示例。
"""

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

__all__ = [
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
]
