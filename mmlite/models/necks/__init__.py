"""
Necks 模块 - 特征金字塔网络

直接复用 mmdet 的 FPN 实现，用于多尺度特征融合。
"""

from mmdet.models.necks import FPN

__all__ = ["FPN"]
