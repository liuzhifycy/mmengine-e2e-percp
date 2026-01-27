"""
点云变换 Pipeline

点云数据增强和过滤操作。
"""

import numpy as np
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointsRangeFilter:
    """Filter points by range.

    Args:
        point_cloud_range (list[float]): Range of point cloud
            [x_min, y_min, z_min, x_max, y_max, z_max].
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, results):
        """Filter points by range.

        Args:
            results (dict): Result dict with 'points'.

        Returns:
            dict: Result dict with filtered points.
        """
        points = results['points']

        # Create range mask
        in_range_mask = (
            (points[:, 0] >= self.pcd_range[0]) &
            (points[:, 0] <= self.pcd_range[3]) &
            (points[:, 1] >= self.pcd_range[1]) &
            (points[:, 1] <= self.pcd_range[4]) &
            (points[:, 2] >= self.pcd_range[2]) &
            (points[:, 2] <= self.pcd_range[5])
        )

        results['points'] = points[in_range_mask]
        return results


@TRANSFORMS.register_module()
class ObjectRangeFilter:
    """Filter objects by range.

    Args:
        point_cloud_range (list[float]): Range of point cloud.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, results):
        """Filter objects by range.

        Args:
            results (dict): Result dict with 'gt_bboxes_3d' and 'gt_labels_3d'.

        Returns:
            dict: Result dict with filtered objects.
        """
        if 'gt_bboxes_3d' not in results:
            return results

        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results.get('gt_labels_3d', None)

        # Get center of bboxes
        if len(gt_bboxes_3d) == 0:
            return results

        # Assume gt_bboxes_3d has shape (N, 7) with (x, y, z, l, w, h, yaw)
        centers = gt_bboxes_3d[:, :3]

        # Create range mask
        in_range_mask = (
            (centers[:, 0] >= self.pcd_range[0]) &
            (centers[:, 0] <= self.pcd_range[3]) &
            (centers[:, 1] >= self.pcd_range[1]) &
            (centers[:, 1] <= self.pcd_range[4]) &
            (centers[:, 2] >= self.pcd_range[2]) &
            (centers[:, 2] <= self.pcd_range[5])
        )

        results['gt_bboxes_3d'] = gt_bboxes_3d[in_range_mask]
        if gt_labels_3d is not None:
            results['gt_labels_3d'] = gt_labels_3d[in_range_mask]

        return results


@TRANSFORMS.register_module()
class PointShuffle:
    """Shuffle point cloud.

    Randomly shuffle the points.
    """

    def __call__(self, results):
        """Shuffle points.

        Args:
            results (dict): Result dict with 'points'.

        Returns:
            dict: Result dict with shuffled points.
        """
        points = results['points']
        indices = np.random.permutation(len(points))
        results['points'] = points[indices]
        return results


@TRANSFORMS.register_module()
class RandomFlip3D:
    """Random flip for 3D point cloud.

    Args:
        flip_ratio_bev_horizontal (float): Flip ratio in horizontal direction.
        flip_ratio_bev_vertical (float): Flip ratio in vertical direction.
    """

    def __init__(
        self,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0,
    ):
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical

    def __call__(self, results):
        """Flip points and bboxes.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with flipped data.
        """
        points = results['points']

        # Horizontal flip (along y-axis)
        if np.random.random() < self.flip_ratio_bev_horizontal:
            points[:, 1] = -points[:, 1]
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
                gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
                # Flip yaw angle
                gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6]
                results['gt_bboxes_3d'] = gt_bboxes_3d

        # Vertical flip (along x-axis)
        if np.random.random() < self.flip_ratio_bev_vertical:
            points[:, 0] = -points[:, 0]
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
                gt_bboxes_3d[:, 0] = -gt_bboxes_3d[:, 0]
                gt_bboxes_3d[:, 6] = np.pi - gt_bboxes_3d[:, 6]
                results['gt_bboxes_3d'] = gt_bboxes_3d

        results['points'] = points
        return results


@TRANSFORMS.register_module()
class GlobalRotScaleTrans:
    """Global rotation, scaling and translation for 3D point cloud.

    Args:
        rot_range (list[float]): Range of rotation angle.
        scale_ratio_range (list[float]): Range of scale ratio.
        translation_std (list[float]): Std of translation noise.
    """

    def __init__(
        self,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        """Apply global transforms.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with transformed data.
        """
        points = results['points']

        # Random rotation
        rot_angle = np.random.uniform(self.rot_range[0], self.rot_range[1])
        rot_mat = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle), 0],
            [np.sin(rot_angle), np.cos(rot_angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points[:, :3] = points[:, :3] @ rot_mat.T

        # Random scale
        scale = np.random.uniform(
            self.scale_ratio_range[0], self.scale_ratio_range[1]
        )
        points[:, :3] *= scale

        # Random translation
        trans = np.random.normal(0, self.translation_std, size=3).astype(np.float32)
        points[:, :3] += trans

        # Apply to bboxes
        if 'gt_bboxes_3d' in results:
            gt_bboxes_3d = results['gt_bboxes_3d']
            # Rotate centers
            gt_bboxes_3d[:, :3] = gt_bboxes_3d[:, :3] @ rot_mat.T
            # Scale
            gt_bboxes_3d[:, :3] *= scale
            gt_bboxes_3d[:, 3:6] *= scale  # Scale dimensions
            # Translate
            gt_bboxes_3d[:, :3] += trans
            # Rotate yaw
            gt_bboxes_3d[:, 6] += rot_angle
            results['gt_bboxes_3d'] = gt_bboxes_3d

        results['points'] = points
        return results
