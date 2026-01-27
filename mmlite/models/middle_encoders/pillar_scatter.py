"""
PointPillarsScatter - 将柱体特征散射到伪图像（BEV）

将学习到的稀疏柱体特征散射到密集的伪图像表示，
便于使用标准2D卷积网络进行后续处理。
"""

import torch
import torch.nn as nn
from mmengine.registry import MODELS


@MODELS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image (BEV).

    Args:
        in_channels (int): Number of input feature channels.
        output_shape (list[int]): Required output shape [H, W].
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]  # Height (y direction)
        self.nx = output_shape[1]  # Width (x direction)
        self.in_channels = in_channels

    def forward(self, voxel_features, coors, batch_size=None):
        """Forward function to scatter features.

        Args:
            voxel_features (torch.Tensor): Voxel features (N, C).
            coors (torch.Tensor): Coordinates of each voxel (N, 4).
                Format: (batch_idx, z, y, x).
            batch_size (int, optional): Batch size. If None, infer from coors.

        Returns:
            torch.Tensor: Pseudo image features (B, C, H, W).
        """
        if batch_size is not None:
            return self._forward_batch(voxel_features, coors, batch_size)
        else:
            return self._forward_single(voxel_features, coors)

    def _forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features (N, C).
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Pseudo image (1, C, H, W).
        """
        # Create canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device,
        )

        # Calculate linear indices: y * width + x
        indices = coors[:, 2] * self.nx + coors[:, 3]
        indices = indices.long()
        
        # Transpose features: (N, C) -> (C, N)
        voxels = voxel_features.t()
        
        # Scatter features to canvas
        canvas[:, indices] = voxels
        
        # Reshape to 4D tensor: (1, C, H, W)
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        
        return canvas

    def _forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features for a batch of samples.

        Args:
            voxel_features (torch.Tensor): Voxel features (N, C).
            coors (torch.Tensor): Coordinates with batch index (N, 4).
            batch_size (int): Number of samples in batch.

        Returns:
            torch.Tensor: Batched pseudo images (B, C, H, W).
        """
        batch_canvas = []
        
        for batch_idx in range(batch_size):
            # Create canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Select voxels belonging to this sample
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, :]
            
            # Calculate linear indices
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.long()
            
            # Get features for this sample
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Scatter features to canvas
            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        # Stack all samples: (B, C, H*W)
        batch_canvas = torch.stack(batch_canvas, dim=0)
        
        # Reshape to 4D tensor: (B, C, H, W)
        batch_canvas = batch_canvas.view(
            batch_size, self.in_channels, self.ny, self.nx
        )

        return batch_canvas
