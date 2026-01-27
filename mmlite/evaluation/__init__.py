"""
Evaluation 模块 - 评估指标

提供 2D/3D 目标检测的评估指标:
- CocoMetric: COCO 格式数据集评估 (2D)
- KittiMetric: KITTI 3D 检测评估
"""

from mmdet.evaluation import CocoMetric

from .kitti_metric import KittiMetric

__all__ = ["CocoMetric", "KittiMetric"]
