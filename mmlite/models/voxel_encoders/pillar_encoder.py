"""
PillarFeatureNet - PointPillars 核心特征编码器

将体素化后的点云特征编码为高维特征向量。
基于论文: PointPillars: Fast Encoders for Object Detection from Point Clouds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.registry import MODELS


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actual number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in a voxel.
        axis (int): Axis to indicate the padded dimension.

    Returns:
        torch.Tensor: Mask indicating which points are padded.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device)
    max_num = max_num.view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers.
        last_layer (bool): If last_layer, no concatenation of features.
        mode (str): Pooling mode. 'max' or 'avg'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        last_layer=False,
        mode='max',
    ):
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        
        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel features with shape (N, M, C).
            num_voxels (torch.Tensor): Number of points in each pillar.

        Returns:
            torch.Tensor: Features after PFN layer.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        else:
            x_max = torch.mean(x, dim=1, keepdim=True)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@MODELS.register_module()
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int): Number of input features (x, y, z, r, etc.).
        feat_channels (tuple): Number of features in each PFNLayer.
        with_distance (bool): Whether to include Euclidean distance to points.
        with_cluster_center (bool): Whether to include cluster center offset.
        with_voxel_center (bool): Whether to include voxel center offset.
        voxel_size (tuple): Size of voxels (x, y, z).
        point_cloud_range (tuple): Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max).
        norm_cfg (dict): Config dict for normalization layer.
        mode (str): Pooling mode for PFN layers.
        legacy (bool): Whether to use legacy behavior.
    """

    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=(0.16, 0.16, 4),
        point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        legacy=True,
    ):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        
        # Calculate input channels based on features used
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
            
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.in_channels = in_channels
        
        # Create PFN layers
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            last_layer = (i >= len(feat_channels) - 2)
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode,
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Voxel size and point cloud range for offset calculation
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape (N, M, C).
                N is number of voxels, M is max points per voxel, C is feature dim.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of each voxel (batch_idx, z, y, x).

        Returns:
            torch.Tensor: Features of pillars (N, feat_channels[-1]).
        """
        features_ls = [features]
        
        # Add cluster center offset (distance from point to mean of pillar points)
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / \
                         num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Add voxel center offset (distance from point to pillar center)
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (
                    coors[:, 1].to(dtype).unsqueeze(1) * self.vz + self.z_offset)
            else:
                f_center = features[:, :, :3].clone()
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)

        # Add distance feature
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Concatenate all features
        features = torch.cat(features_ls, dim=-1)
        
        # Mask out padded points
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward through PFN layers
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)


@MODELS.register_module()
class DynamicPillarFeatureNet(PillarFeatureNet):
    """Dynamic Pillar Feature Net for dynamic voxelization.
    
    Different from PillarFeatureNet, this module handles variable number
    of points per voxel without a fixed maximum limit.
    """
    
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=(0.16, 0.16, 4),
        point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode='max',
        legacy=True,
    ):
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_distance=with_distance,
            with_cluster_center=with_cluster_center,
            with_voxel_center=with_voxel_center,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            norm_cfg=norm_cfg,
            mode=mode,
            legacy=legacy,
        )
        
        # For dynamic scatter
        try:
            from mmcv.ops import DynamicScatter
            self.pfn_scatter = DynamicScatter(
                voxel_size, point_cloud_range, (mode != 'max')
            )
            self.cluster_scatter = DynamicScatter(
                voxel_size, point_cloud_range, average_points=True
            )
        except ImportError:
            print("Warning: mmcv.ops.DynamicScatter not available. "
                  "Dynamic voxelization will not work.")
            self.pfn_scatter = None
            self.cluster_scatter = None

    def forward(self, features, coors):
        """Forward function for dynamic voxelization.

        Args:
            features (torch.Tensor): Point features (N, C) where N is total points.
            coors (torch.Tensor): Coordinates (N, 4) with (batch_idx, z, y, x).

        Returns:
            tuple: Voxel features and coordinates.
        """
        features_ls = [features]
        
        # Cluster center (need dynamic scatter for mean)
        if self._with_cluster_center and self.cluster_scatter is not None:
            voxel_mean, mean_coors = self.cluster_scatter(features[:, :3].float(), coors)
            # Map back to points
            points_mean = self._map_voxel_center_to_point(coors, voxel_mean, mean_coors)
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Voxel center offset
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        # Distance feature
        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Concatenate
        features = torch.cat(features_ls, dim=-1)

        # Use first PFN layer for point-wise features
        for pfn in self.pfn_layers:
            features = features.unsqueeze(1)  # Add voxel dimension
            features = pfn(features).squeeze(1)

        # Scatter to voxels
        if self.pfn_scatter is not None:
            voxel_feats, voxel_coors = self.pfn_scatter(features.float(), coors)
            return voxel_feats, voxel_coors
        else:
            return features, coors

    def _map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel centers back to points."""
        canvas_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = int(pts_coors[-1, 0] + 1)
        canvas_len = canvas_y * canvas_x * batch_size
        
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        indices = (voxel_coors[:, 0] * canvas_y * canvas_x +
                  voxel_coors[:, 2] * canvas_x +
                  voxel_coors[:, 3])
        canvas[:, indices.long()] = voxel_mean.t()

        voxel_index = (pts_coors[:, 0] * canvas_y * canvas_x +
                      pts_coors[:, 2] * canvas_x +
                      pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point
