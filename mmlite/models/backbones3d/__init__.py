"""
3D Backbones - 用于点云检测的骨干网络

包含 SECOND 等用于处理伪图像（BEV）的骨干网络。
"""

from .second import SECOND, SECONDFPN

__all__ = [
    "SECOND",
    "SECONDFPN",
]
