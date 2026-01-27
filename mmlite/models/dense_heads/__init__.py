"""
Dense Heads 模块 - 密集预测头

直接复用 mmdet 的 RetinaHead 实现，用于目标检测的分类和回归。
"""

from mmdet.models.dense_heads import RetinaHead

__all__ = ["RetinaHead"]
