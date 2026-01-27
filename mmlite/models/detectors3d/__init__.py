"""3D Detectors - 3D 检测器模块

提供 3D 目标检测器实现，包括:
- PointPillars: 基于柱体的单阶段 3D 检测器
"""

from .pointpillars import PointPillars

__all__ = ["PointPillars"]
