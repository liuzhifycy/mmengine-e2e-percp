"""
Middle Encoders - 中间编码器

将体素特征散射到伪图像（BEV）表示，用于后续2D卷积处理。
"""

from .pillar_scatter import PointPillarsScatter

__all__ = [
    "PointPillarsScatter",
]
