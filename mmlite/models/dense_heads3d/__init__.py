"""3D Dense Heads - 3D 检测头模块

提供 3D 目标检测的检测头实现，包括:
- Anchor3DHead: 基于 anchor 的 3D 检测头（用于 PointPillars, SECOND 等）
- Anchor3DRangeGenerator: 3D anchor 生成器
"""

from .anchor3d_head import Anchor3DHead
from .anchor_generator import Anchor3DRangeGenerator

__all__ = ["Anchor3DHead", "Anchor3DRangeGenerator"]
