"""
点云加载 Pipeline

从文件加载点云数据，支持 KITTI、nuScenes 等格式。
"""

import numpy as np
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPointsFromFile:
    """Load points from file.

    Args:
        coord_type (str): Type of point coordinates. 'LIDAR', 'CAMERA', 'DEPTH'.
        load_dim (int): Dimension of points to load.
        use_dim (list[int]): Dimensions to use. Default all.
        shift_height (bool): Whether to shift height.
        use_color (bool): Whether to use color info.
        file_client_args (dict): Arguments for file client.
    """

    def __init__(
        self,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=None,
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend='disk'),
    ):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim if use_dim is not None else list(range(load_dim))
        self.shift_height = shift_height
        self.use_color = use_color
        self.file_client_args = file_client_args

    def _load_points(self, pts_filename):
        """Load points from file.

        Args:
            pts_filename (str): Path to point cloud file.

        Returns:
            np.ndarray: Loaded points (N, load_dim).
        """
        # KITTI format: binary file with float32
        if pts_filename.endswith('.bin'):
            points = np.fromfile(pts_filename, dtype=np.float32)
            points = points.reshape(-1, self.load_dim)
        # nuScenes/Waymo format: numpy file
        elif pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        # PCD format
        elif pts_filename.endswith('.pcd'):
            points = self._load_pcd(pts_filename)
        else:
            raise NotImplementedError(
                f'Unsupported file format: {pts_filename}'
            )
        return points

    def _load_pcd(self, pts_filename):
        """Load PCD format point cloud."""
        # Simple PCD loader - only supports ASCII and binary formats
        with open(pts_filename, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header.append(line)
                if line.startswith('DATA'):
                    break

            # Parse header
            data_format = line.split()[-1]
            points_num = None
            fields = []
            for h in header:
                if h.startswith('POINTS'):
                    points_num = int(h.split()[-1])
                elif h.startswith('FIELDS'):
                    fields = h.split()[1:]

            if points_num is None:
                raise ValueError('Cannot find POINTS in PCD header')

            # Read data
            if data_format == 'ascii':
                points = np.loadtxt(f, dtype=np.float32, max_rows=points_num)
            elif data_format == 'binary':
                points = np.frombuffer(
                    f.read(points_num * self.load_dim * 4),
                    dtype=np.float32
                ).reshape(-1, self.load_dim)
            else:
                raise NotImplementedError(f'PCD format {data_format} not supported')

        return points

    def __call__(self, results):
        """Load points.

        Args:
            results (dict): Result dict containing 'lidar_points' with 'lidar_path'.

        Returns:
            dict: Result dict with 'points' added.
        """
        pts_filename = results['lidar_points']['lidar_path']
        points = self._load_points(pts_filename)

        # Select dimensions to use
        points = points[:, self.use_dim]

        # Shift height if needed (for KITTI)
        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            points[:, 2] = points[:, 2] - floor_height

        # Create points structure
        results['points'] = points
        results['points_shape'] = points.shape

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'coord_type={self.coord_type}, '
                f'load_dim={self.load_dim}, '
                f'use_dim={self.use_dim})')


@TRANSFORMS.register_module()
class LoadAnnotations3D:
    """Load annotations for 3D detection.

    Args:
        with_bbox_3d (bool): Whether to load 3D bounding boxes.
        with_label_3d (bool): Whether to load 3D labels.
        with_velocity (bool): Whether to load velocity.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_velocity=False,
    ):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_velocity = with_velocity

    def _load_bboxes_3d(self, results):
        """Load 3D bounding boxes."""
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_labels_3d(self, results):
        """Load 3D labels."""
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def __call__(self, results):
        """Load annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with annotations loaded.
        """
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)

        return results
