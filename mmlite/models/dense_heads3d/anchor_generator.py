"""3D Anchor Generator - 3D anchor 生成器

用于 PointPillars、SECOND 等基于 anchor 的 3D 检测器。
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.registry import TASK_UTILS
from torch import Tensor


@TASK_UTILS.register_module()
class Anchor3DRangeGenerator:
    """3D Anchor generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]], optional): 3D sizes of anchors.
            Defaults to [[1.6, 3.9, 1.56]].
        scales (list[int], optional): Scales of anchors in different feature
            levels. Defaults to [1].
        rotations (list[float], optional): Rotations of anchors in a feature
            grid. Defaults to [0, 1.5707963].
        custom_values (tuple[float], optional): Custom values of anchors.
            Defaults to ().
        reshape_out (bool, optional): Whether to reshape the output into
            (N x 4). Defaults to True.
        size_per_range (bool, optional): Whether to use separate sizes for
            different ranges. Defaults to True.
    """

    def __init__(
        self,
        ranges: List[List[float]],
        sizes: List[List[float]] = [[1.6, 3.9, 1.56]],
        scales: List[int] = [1],
        rotations: List[float] = [0, 1.5707963],
        custom_values: Tuple[float, ...] = (),
        reshape_out: bool = True,
        size_per_range: bool = True,
    ) -> None:
        if len(sizes) != len(ranges):
            assert len(sizes) == 1, (
                "Anchor sizes must be the same length as anchor ranges, "
                f"or only have 1 size. Got {len(sizes)} sizes and "
                f"{len(ranges)} ranges."
            )
            sizes = sizes * len(ranges)

        # Validate anchor range format
        if isinstance(ranges[0], (list, tuple)):
            for r in ranges:
                assert len(r) == 6, "Anchor range should be in format [x0, y0, z0, x1, y1, z1]"
        else:
            assert len(ranges) == 6, "Anchor range should be in format [x0, y0, z0, x1, y1, z1]"

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range
        self.cached_anchors = None

    @property
    def num_base_anchors(self) -> int:
        """int: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = np.array(self.sizes).reshape(-1, 3).shape[0]
        return num_rot * num_size

    @property
    def num_levels(self) -> int:
        """int: Number of feature levels."""
        return len(self.scales)

    def grid_anchors(
        self, featmap_sizes: List[Tuple[int, int]], device: str = "cuda"
    ) -> List[Tensor]:
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in multiple
                feature levels.
            device (str, optional): Device where the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            list[Tensor]: Anchors in multiple feature levels. The sizes of
                each tensor should be [N, 4], where N = width * height * num_base_anchors,
                width and height are the sizes of the corresponding feature level,
                4 represent [cx, cy, cz, w, l, h, rot].
        """
        assert len(featmap_sizes) == self.num_levels, (
            f"Number of feature map sizes ({len(featmap_sizes)}) must be "
            f"equal to the number of anchor levels ({self.num_levels})."
        )

        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.scales[i], device=device
            )
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(
        self,
        featmap_size: Tuple[int, int],
        scale: float,
        device: str = "cuda",
    ) -> Tensor:
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device where the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            Tensor: Anchors in the overall feature map.
        """
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    self.rotations,
                    device=device,
                )
            )
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(
        self,
        feature_size: Tuple[int, int],
        anchor_range: List[float],
        scale: float = 1,
        sizes: List[float] = [1.6, 3.9, 1.56],
        rotations: List[float] = [0, 1.5707963],
        device: str = "cuda",
    ) -> Tensor:
        """Generate anchors in a single range.

        Args:
            feature_size (tuple[int, int]): Feature map size. It generates
                anchors based on the feature map size (height, width).
            anchor_range (list[float]): Range of anchors with shape [6].
                The order is [x0, y0, z0, x1, y1, z1].
            scale (float, optional): The scale factor of anchors.
                Defaults to 1.
            sizes (list[float], optional): Anchor size with shape [3].
                The order is [w, l, h]. Defaults to [1.6, 3.9, 1.56].
            rotations (list[float], optional): Rotations of anchors.
                Defaults to [0, 1.5707963].
            device (str, optional): Device where the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            Tensor: Anchors with shape [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = (*feature_size, 1)

        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2],
            anchor_range[5],
            feature_size[2],
            device=device,
        )
        y_centers = torch.linspace(
            anchor_range[1],
            anchor_range[4],
            feature_size[0],
            device=device,
        )
        x_centers = torch.linspace(
            anchor_range[0],
            anchor_range[3],
            feature_size[1],
            device=device,
        )
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # Generate all combinations
        # [x, y, z] x sizes x rotations -> [H, W, D, num_sizes, num_rots, 7]
        # Using meshgrid for efficiency
        ry, rx, rz = torch.meshgrid(y_centers, x_centers, z_centers, indexing="ij")

        # Expand dims for broadcasting
        # [H, W, D] -> [H, W, D, 1, 1]
        rx = rx.unsqueeze(-1).unsqueeze(-1)
        ry = ry.unsqueeze(-1).unsqueeze(-1)
        rz = rz.unsqueeze(-1).unsqueeze(-1)

        # Expand sizes: [num_sizes, 3] -> [1, 1, 1, num_sizes, 1, 3]
        sizes = sizes.view(1, 1, 1, -1, 1, 3)

        # Expand rotations: [num_rots] -> [1, 1, 1, 1, num_rots, 1]
        rotations = rotations.view(1, 1, 1, 1, -1, 1)

        # Broadcast to [H, W, D, num_sizes, num_rots, 7]
        num_sizes = sizes.shape[3]
        num_rots = rotations.shape[4]

        # Expand center coordinates
        rx = rx.expand(-1, -1, -1, num_sizes, num_rots)
        ry = ry.expand(-1, -1, -1, num_sizes, num_rots)
        rz = rz.expand(-1, -1, -1, num_sizes, num_rots)

        # Expand sizes to all positions
        sizes = sizes.expand(
            feature_size[0], feature_size[1], feature_size[2], -1, num_rots, -1
        )
        rotations = rotations.expand(
            feature_size[0], feature_size[1], feature_size[2], num_sizes, -1, -1
        )

        # Combine into anchors [H, W, D, num_sizes, num_rots, 7]
        # Order: [x, y, z, w, l, h, rot]
        anchors = torch.stack(
            [rx, ry, rz], dim=-1
        )  # [H, W, D, num_sizes, num_rots, 3]
        anchors = torch.cat([anchors, sizes, rotations], dim=-1)

        # Add custom values if any
        if self.custom_values:
            custom = torch.tensor(self.custom_values, device=device)
            custom = custom.view(1, 1, 1, 1, 1, -1).expand(
                feature_size[0],
                feature_size[1],
                feature_size[2],
                num_sizes,
                num_rots,
                -1,
            )
            anchors = torch.cat([anchors, custom], dim=-1)

        return anchors

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"anchor_range={self.ranges},\n"
        s += f"scales={self.scales},\n"
        s += f"sizes={self.sizes},\n"
        s += f"rotations={self.rotations},\n"
        s += f"reshape_out={self.reshape_out})"
        return s
