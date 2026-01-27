"""KITTI Dataset - KITTI 3D 目标检测数据集

支持 KITTI 3D 目标检测数据集的加载和处理。
"""

import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.registry import DATASETS
from mmengine.structures import InstanceData


@DATASETS.register_module()
class KittiDataset(BaseDataset):
    """KITTI Dataset for 3D object detection.

    This dataset loads KITTI 3D detection data including point clouds,
    images, calibration, and annotations.

    Args:
        data_root (str): Path to the dataset root directory.
        ann_file (str): Path to the annotation file (pkl or json).
        pipeline (list[dict]): Processing pipeline.
        modality (dict): Modality config. Keys: 'use_lidar', 'use_camera'.
        box_type_3d (str): Type of 3D box. Options: 'LiDAR', 'Camera'.
        filter_empty_gt (bool): Whether to filter samples with no gt boxes.
        test_mode (bool): Whether in test mode.
        pcd_limit_range (list[float]): Point cloud range limit.
        classes (tuple[str]): Class names.
        metainfo (dict, optional): Meta information.
    """

    METAINFO = {
        "classes": ("Car", "Pedestrian", "Cyclist"),
        "palette": [(0, 255, 0), (255, 255, 0), (0, 255, 255)],
    }

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: List[Dict],
        modality: Dict = dict(use_lidar=True, use_camera=False),
        box_type_3d: str = "LiDAR",
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
        classes: Optional[Tuple[str, ...]] = None,
        metainfo: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.modality = modality
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt
        self.pcd_limit_range = pcd_limit_range

        # Update metainfo with classes if provided
        if metainfo is None:
            metainfo = {}
        if classes is not None:
            metainfo["classes"] = classes

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            metainfo=metainfo,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load annotations from ann_file.

        Returns:
            list[dict]: A list of annotation dicts.
        """
        data_list = []

        # Note: BaseDataset already joins data_root + ann_file,
        # so self.ann_file is already the full path
        ann_file_path = self.ann_file
        if osp.exists(ann_file_path):
            annotations = load(ann_file_path)
            if isinstance(annotations, dict):
                # pkl format from mmdet3d data converter
                data_infos = annotations.get("data_list", annotations.get("infos", []))
            else:
                data_infos = annotations
        else:
            # Create data list from directory structure
            data_infos = self._scan_data_dir()

        for info in data_infos:
            data_info = self.parse_data_info(info)
            if data_info is not None:
                data_list.append(data_info)

        return data_list

    def _scan_data_dir(self) -> List[Dict]:
        """Scan data directory to create info list.

        Returns:
            list[dict]: List of data info dicts.
        """
        data_infos = []

        # KITTI directory structure
        lidar_dir = osp.join(self.data_root, "velodyne")
        if not osp.exists(lidar_dir):
            lidar_dir = osp.join(self.data_root, "training", "velodyne")

        if not osp.exists(lidar_dir):
            return data_infos

        # Get all bin files
        bin_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

        for bin_file in bin_files:
            idx = bin_file.replace(".bin", "")
            info = {
                "sample_idx": idx,
                "lidar_points": {"lidar_path": osp.join("velodyne", bin_file)},
            }
            data_infos.append(info)

        return data_infos

    def parse_data_info(self, info: Dict) -> Optional[Dict]:
        """Parse raw data info.

        Args:
            info (dict): Raw data info from annotation file.

        Returns:
            dict or None: Parsed data info for pipeline.
        """
        data_info = {}

        # Sample index
        sample_idx = info.get("sample_idx", info.get("image_idx", ""))
        data_info["sample_idx"] = sample_idx

        # Data root for pipeline to construct full paths
        data_info["data_root"] = self.data_root

        # Point cloud path - keep lidar_points structure for pipeline
        if "lidar_points" in info:
            lidar_info = info["lidar_points"]
            lidar_path = lidar_info.get("lidar_path", lidar_info.get("velodyne_path", ""))
            data_info["lidar_points"] = {
                "lidar_path": lidar_path,
                "num_pts_feats": lidar_info.get("num_pts_feats", 4)
            }
        else:
            lidar_path = info.get("velodyne_path", f"training/velodyne/{sample_idx}.bin")
            data_info["lidar_points"] = {
                "lidar_path": lidar_path,
                "num_pts_feats": 4
            }

        # Also keep lidar_path for compatibility
        data_info["lidar_path"] = osp.join(self.data_root, data_info["lidar_points"]["lidar_path"])

        # Image path (optional)
        if self.modality.get("use_camera", False):
            if "images" in info:
                img_info = info["images"].get("CAM2", info["images"].get("image_2", {}))
                img_path = img_info.get("img_path", "")
            else:
                img_path = info.get("image_path", f"image_2/{sample_idx}.png")
            data_info["img_path"] = osp.join(self.data_root, img_path)

        # Calibration
        if "calib" in info:
            data_info["calib"] = info["calib"]
        else:
            # Handle both int and str sample_idx
            idx_str = f"{sample_idx:06d}" if isinstance(sample_idx, int) else str(sample_idx).zfill(6)
            calib_path = osp.join(self.data_root, "training", "calib", f"{idx_str}.txt")
            if osp.exists(calib_path):
                data_info["calib"] = self._load_calib(calib_path)

        # Annotations (for training)
        if not self.test_mode:
            annos = info.get("annos", info.get("instances", None))
            if annos is not None:
                gt_bboxes_3d, gt_labels_3d = self._parse_annos(annos)

                # Filter empty gt
                if self.filter_empty_gt and len(gt_labels_3d) == 0:
                    return None

                # Store annotations in ann_info for LoadAnnotations3D
                data_info["ann_info"] = {
                    "gt_bboxes_3d": gt_bboxes_3d,
                    "gt_labels_3d": gt_labels_3d,
                }
                # Also keep at top level for compatibility
                data_info["gt_bboxes_3d"] = gt_bboxes_3d
                data_info["gt_labels_3d"] = gt_labels_3d

                # Additional annotations
                if "name" in annos:
                    data_info["gt_names"] = annos["name"]
                if "difficulty" in annos:
                    data_info["difficulty"] = annos["difficulty"]

        return data_info

    def _parse_annos(self, annos: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Parse annotation dict to gt boxes and labels.

        Args:
            annos (dict): Annotation dict from KITTI.

        Returns:
            tuple: (gt_bboxes_3d, gt_labels_3d)
        """
        classes = self.metainfo["classes"]
        class_to_label = {cls: i for i, cls in enumerate(classes)}

        gt_bboxes_3d = []
        gt_labels_3d = []

        # Handle different annotation formats
        if isinstance(annos, list):
            # List of instance dicts (mmdet3d new format)
            for instance in annos:
                name = instance.get("bbox_label_3d", instance.get("name", ""))
                if isinstance(name, int):
                    label = name
                    if label < len(classes):
                        gt_labels_3d.append(label)
                        bbox_3d = instance.get("bbox_3d", [])
                        if len(bbox_3d) >= 7:
                            gt_bboxes_3d.append(bbox_3d[:7])
                elif name in class_to_label:
                    gt_labels_3d.append(class_to_label[name])
                    bbox_3d = instance.get("bbox_3d", [])
                    if len(bbox_3d) >= 7:
                        gt_bboxes_3d.append(bbox_3d[:7])
        else:
            # Dict format (KITTI original format)
            names = annos.get("name", [])
            
            # Get 3D bboxes
            if "gt_bboxes_3d" in annos:
                bboxes_3d = annos["gt_bboxes_3d"]
            elif "location" in annos and "dimensions" in annos and "rotation_y" in annos:
                # Construct from KITTI format
                loc = np.array(annos["location"])  # [N, 3]
                dims = np.array(annos["dimensions"])  # [N, 3] - h, w, l
                rot = np.array(annos["rotation_y"])  # [N]

                if len(loc.shape) == 1:
                    loc = loc.reshape(1, -1)
                    dims = dims.reshape(1, -1)
                    rot = rot.reshape(-1)

                # KITTI format: x, y, z, w, l, h, rot
                # Note: KITTI dims are h, w, l, need to convert to w, l, h
                bboxes_3d = np.concatenate(
                    [loc, dims[:, [1, 2, 0]], rot[:, np.newaxis]], axis=1
                )
            else:
                bboxes_3d = np.zeros((len(names), 7))

            for i, name in enumerate(names):
                if name in class_to_label:
                    gt_labels_3d.append(class_to_label[name])
                    if i < len(bboxes_3d):
                        gt_bboxes_3d.append(bboxes_3d[i])

        gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32).reshape(-1, 7)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        return gt_bboxes_3d, gt_labels_3d

    def _load_calib(self, calib_path: str) -> Dict:
        """Load calibration data from file.

        Args:
            calib_path (str): Path to calibration file.

        Returns:
            dict: Calibration matrices.
        """
        calib = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                calib[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )

        return calib

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the sample.
        """
        data_info = self.get_data_info(idx)
        gt_labels = data_info.get("gt_labels_3d", [])
        return list(set(gt_labels.tolist() if hasattr(gt_labels, "tolist") else gt_labels))

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "(\n"
        s += f"    data_root={self.data_root},\n"
        s += f"    ann_file={self.ann_file},\n"
        s += f"    num_samples={len(self)},\n"
        s += f"    modality={self.modality},\n"
        s += f"    classes={self.metainfo.get('classes')},\n"
        s += f"    test_mode={self.test_mode})"
        return s


@DATASETS.register_module()
class KittiMonoDataset(KittiDataset):
    """KITTI Dataset for monocular 3D detection."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: List[Dict],
        modality: Dict = dict(use_lidar=False, use_camera=True),
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            **kwargs,
        )


class Det3DDataSample:
    """Data sample class for 3D detection.

    A simple container for detection data samples.
    """

    def __init__(self):
        self._gt_instances_3d = None
        self._pred_instances_3d = None
        self._metainfo = {}

    @property
    def gt_bboxes_3d(self):
        if self._gt_instances_3d is not None:
            return self._gt_instances_3d.get("bboxes_3d")
        return None

    @gt_bboxes_3d.setter
    def gt_bboxes_3d(self, value):
        if self._gt_instances_3d is None:
            self._gt_instances_3d = {}
        self._gt_instances_3d["bboxes_3d"] = value

    @property
    def gt_labels_3d(self):
        if self._gt_instances_3d is not None:
            return self._gt_instances_3d.get("labels_3d")
        return None

    @gt_labels_3d.setter
    def gt_labels_3d(self, value):
        if self._gt_instances_3d is None:
            self._gt_instances_3d = {}
        self._gt_instances_3d["labels_3d"] = value

    @property
    def metainfo(self) -> Dict:
        return self._metainfo

    @metainfo.setter
    def metainfo(self, value: Dict):
        self._metainfo = value

    def set_metainfo(self, metainfo: Dict):
        self._metainfo.update(metainfo)


@DATASETS.register_module()
class NuScenesDataset(BaseDataset):
    """NuScenes Dataset for 3D object detection.

    Placeholder for nuScenes dataset support.

    Args:
        data_root (str): Path to the dataset root directory.
        ann_file (str): Path to the annotation file.
        pipeline (list[dict]): Processing pipeline.
        modality (dict): Modality config.
        classes (tuple[str]): Class names.
    """

    METAINFO = {
        "classes": (
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        ),
    }

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: List[Dict],
        modality: Dict = dict(use_lidar=True, use_camera=False),
        classes: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> None:
        self.modality = modality

        metainfo = {}
        if classes is not None:
            metainfo["classes"] = classes

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            metainfo=metainfo,
            **kwargs,
        )

    def load_data_list(self) -> List[Dict]:
        """Load annotations."""
        ann_file_path = osp.join(self.data_root, self.ann_file)
        if osp.exists(ann_file_path):
            annotations = load(ann_file_path)
            if isinstance(annotations, dict):
                return annotations.get("data_list", annotations.get("infos", []))
            return annotations
        return []
