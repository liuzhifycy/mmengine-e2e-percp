"""
Evaluation 模块 - 评估指标

直接复用 mmdet 的 CocoMetric 实现，用于 COCO 格式数据集的评估。
"""

from mmdet.evaluation import CocoMetric

__all__ = ["CocoMetric"]
