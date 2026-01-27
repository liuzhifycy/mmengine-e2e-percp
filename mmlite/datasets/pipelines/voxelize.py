"""
体素化 Pipeline

将点云体素化为柱体（Pillar）或体素（Voxel）表示。
"""

import numpy as np
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class VoxelGenerator:
    """Voxelization for point cloud.

    Converts point cloud to voxel representation for PointPillars.

    Args:
        voxel_size (list[float]): Size of voxels [x, y, z].
        point_cloud_range (list[float]): Range of point cloud.
        max_num_points (int): Maximum number of points per voxel.
        max_voxels (int or tuple): Maximum number of voxels.
            If tuple, (train_max, test_max).
        deterministic (bool): Whether to use deterministic voxelization.
    """

    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points=32,
        max_voxels=16000,
        deterministic=True,
    ):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels)
        
        self.deterministic = deterministic

        # Calculate grid size
        grid_size = (
            (self.point_cloud_range[3:6] - self.point_cloud_range[:3]) /
            self.voxel_size
        )
        self.grid_size = np.round(grid_size).astype(np.int64)

    def _points_to_voxel(self, points, training=True):
        """Convert points to voxels using numpy.

        Args:
            points (np.ndarray): Points with shape (N, C).
            training (bool): Whether in training mode.

        Returns:
            tuple: voxels, coordinates, num_points_per_voxel
        """
        max_voxels = self.max_voxels[0] if training else self.max_voxels[1]

        # Calculate voxel indices
        voxel_indices = (
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

        # Mask out points outside range
        valid_mask = (
            (voxel_indices[:, 0] >= 0) &
            (voxel_indices[:, 0] < self.grid_size[0]) &
            (voxel_indices[:, 1] >= 0) &
            (voxel_indices[:, 1] < self.grid_size[1]) &
            (voxel_indices[:, 2] >= 0) &
            (voxel_indices[:, 2] < self.grid_size[2])
        )

        points = points[valid_mask]
        voxel_indices = voxel_indices[valid_mask]

        if len(points) == 0:
            # Return empty voxels
            return (
                np.zeros((0, self.max_num_points, points.shape[1]), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.int32)
            )

        # Get unique voxels and their indices
        # Use a hash for faster lookup
        voxel_hash = (
            voxel_indices[:, 0] +
            voxel_indices[:, 1] * self.grid_size[0] +
            voxel_indices[:, 2] * self.grid_size[0] * self.grid_size[1]
        )

        # Sort by hash
        sort_idx = np.argsort(voxel_hash)
        voxel_hash_sorted = voxel_hash[sort_idx]
        voxel_indices_sorted = voxel_indices[sort_idx]
        points_sorted = points[sort_idx]

        # Find unique voxels
        unique_hash, inverse_idx, counts = np.unique(
            voxel_hash_sorted, return_inverse=True, return_counts=True
        )

        num_voxels = min(len(unique_hash), max_voxels)

        # Initialize output arrays
        voxels = np.zeros(
            (num_voxels, self.max_num_points, points.shape[1]),
            dtype=np.float32
        )
        coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)

        # Fill voxels
        start_idx = 0
        for i in range(min(len(unique_hash), num_voxels)):
            count = counts[i]
            end_idx = start_idx + count

            # Get points for this voxel
            voxel_points = points_sorted[start_idx:end_idx]
            n_pts = min(count, self.max_num_points)

            if self.deterministic:
                # Use first N points
                voxels[i, :n_pts] = voxel_points[:n_pts]
            else:
                # Random sample
                if count > self.max_num_points:
                    idx = np.random.choice(count, self.max_num_points, replace=False)
                    voxels[i] = voxel_points[idx]
                    n_pts = self.max_num_points
                else:
                    voxels[i, :n_pts] = voxel_points

            coordinates[i] = voxel_indices_sorted[start_idx]
            num_points_per_voxel[i] = n_pts
            start_idx = end_idx

        return voxels, coordinates, num_points_per_voxel

    def __call__(self, results):
        """Voxelize points.

        Args:
            results (dict): Result dict with 'points'.

        Returns:
            dict: Result dict with voxelization results.
        """
        points = results['points']
        training = not results.get('test_mode', False)

        voxels, coordinates, num_points_per_voxel = self._points_to_voxel(
            points, training=training
        )

        results['voxels'] = voxels
        results['voxel_coords'] = coordinates
        results['voxel_num_points'] = num_points_per_voxel

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'voxel_size={self.voxel_size.tolist()}, '
                f'point_cloud_range={self.point_cloud_range.tolist()}, '
                f'max_num_points={self.max_num_points}, '
                f'max_voxels={self.max_voxels})')


@TRANSFORMS.register_module()
class DynamicVoxelization:
    """Dynamic voxelization without max point limit.

    For DynamicPillarFeatureNet which handles variable number of points.

    Args:
        voxel_size (list[float]): Size of voxels.
        point_cloud_range (list[float]): Range of point cloud.
    """

    def __init__(self, voxel_size, point_cloud_range):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

        grid_size = (
            (self.point_cloud_range[3:6] - self.point_cloud_range[:3]) /
            self.voxel_size
        )
        self.grid_size = np.round(grid_size).astype(np.int64)

    def __call__(self, results):
        """Dynamic voxelization.

        Args:
            results (dict): Result dict with 'points'.

        Returns:
            dict: Result dict with 'points' and 'coors' (coordinates).
        """
        points = results['points']

        # Calculate voxel indices
        coors = (
            (points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
        ).astype(np.int32)

        # Filter valid points
        valid_mask = (
            (coors[:, 0] >= 0) &
            (coors[:, 0] < self.grid_size[0]) &
            (coors[:, 1] >= 0) &
            (coors[:, 1] < self.grid_size[1]) &
            (coors[:, 2] >= 0) &
            (coors[:, 2] < self.grid_size[2])
        )

        results['points'] = points[valid_mask]
        results['coors'] = coors[valid_mask]

        return results
