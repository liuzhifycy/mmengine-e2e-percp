"""
Detectors 模块 - 完整检测器

直接复用 mmdet 的 RetinaNet 实现，提供端到端的目标检测能力。
"""

from mmdet.models.detectors import RetinaNet

__all__ = ["RetinaNet"]
