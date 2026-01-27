"""
Backbones 模块 - 特征提取骨干网络

直接复用 mmdet 的 ResNet 实现，避免重复造轮子。
"""

from mmdet.models.backbones import ResNet

__all__ = ["ResNet"]
