"""PointPillars Detector - PointPillars 3D 检测器

PointPillars: Fast Encoders for Object Detection from Point Clouds
https://arxiv.org/abs/1812.05784

将点云组织成柱体（pillars），通过 PointNet 编码后生成伪图像，
然后使用 2D 卷积进行目标检测。
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class PointPillars(BaseModel):
    """PointPillars 3D object detector.

    PointPillars uses PointNet to encode point cloud into a pseudo-image,
    then applies 2D convolutions for object detection.

    Args:
        voxel_encoder (dict): Config of voxel encoder (PillarFeatureNet).
        middle_encoder (dict): Config of middle encoder (PointPillarsScatter).
        backbone (dict): Config of backbone (SECOND).
        neck (dict, optional): Config of neck (SECONDFPN).
        bbox_head (dict): Config of bbox head (Anchor3DHead).
        train_cfg (dict, optional): Config for training.
        test_cfg (dict, optional): Config for testing.
        init_cfg (dict, optional): Config for initialization.
        data_preprocessor (dict, optional): Config for data preprocessor.
    """

    def __init__(
        self,
        voxel_encoder: Dict,
        middle_encoder: Dict,
        backbone: Dict,
        neck: Optional[Dict] = None,
        bbox_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
        data_preprocessor: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build voxel encoder
        self.voxel_encoder = MODELS.build(voxel_encoder)

        # Build middle encoder (scatter)
        self.middle_encoder = MODELS.build(middle_encoder)

        # Build backbone
        self.backbone = MODELS.build(backbone)

        # Build neck
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        # Build bbox head
        if bbox_head is not None:
            bbox_head["train_cfg"] = train_cfg
            bbox_head["test_cfg"] = test_cfg
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

    def extract_feat(
        self,
        voxels: Tensor,
        num_points: Tensor,
        coors: Tensor,
        batch_size: int,
    ) -> Tuple[Tensor, ...]:
        """Extract features from point cloud.

        Args:
            voxels (Tensor): Voxelized point cloud with shape (N, max_points, C).
            num_points (Tensor): Number of points in each voxel with shape (N,).
            coors (Tensor): Coordinates of voxels with shape (N, 4).
                The format is (batch_idx, z, y, x).
            batch_size (int): Batch size.

        Returns:
            tuple[Tensor]: Multi-scale features from backbone/neck.
        """
        # Voxel encoder: encode point features in each voxel
        voxel_features = self.voxel_encoder(voxels, num_points, coors)

        # Middle encoder: scatter voxel features to pseudo-image
        spatial_features = self.middle_encoder(voxel_features, coors, batch_size)

        # Backbone: extract features from pseudo-image
        x = self.backbone(spatial_features)

        # Neck: multi-scale feature fusion
        if self.neck is not None:
            x = self.neck(x)

        return x

    def _prepare_inputs(self, inputs: Dict) -> Dict:
        """Convert inputs to tensors and move to the correct device.

        Args:
            inputs (dict): Input dict with numpy arrays or lists.

        Returns:
            dict: Input dict with torch tensors.
        """
        import numpy as np

        device = next(self.parameters()).device
        prepared = {}

        for key in ["voxels", "num_points", "coors"]:
            if key not in inputs:
                continue
            value = inputs[key]

            # Handle list from pseudo_collate (batch of samples)
            if isinstance(value, list):
                if len(value) == 1:
                    value = value[0]
                else:
                    # Stack multiple samples - need to add batch index for coors
                    if key == "coors":
                        # Add batch index as first column
                        coors_list = []
                        for i, v in enumerate(value):
                            if isinstance(v, np.ndarray):
                                batch_idx = np.full((len(v), 1), i, dtype=v.dtype)
                                v = np.concatenate([batch_idx, v], axis=1)
                            else:
                                batch_idx = torch.full((len(v), 1), i, dtype=v.dtype, device=v.device)
                                v = torch.cat([batch_idx, v], dim=1)
                            coors_list.append(v)
                        value = np.concatenate(coors_list, axis=0) if isinstance(coors_list[0], np.ndarray) else torch.cat(coors_list, dim=0)
                    else:
                        value = np.concatenate(value, axis=0) if isinstance(value[0], np.ndarray) else torch.cat(value, dim=0)

            # Convert numpy to tensor
            if isinstance(value, np.ndarray):
                if key == "num_points" or key == "coors":
                    value = torch.from_numpy(value).long().to(device)
                else:
                    value = torch.from_numpy(value).float().to(device)
            elif isinstance(value, torch.Tensor):
                value = value.to(device)

            # For coors, add batch index if not present (single sample case)
            if key == "coors" and value.shape[-1] == 3:
                batch_idx = torch.zeros((value.shape[0], 1), dtype=value.dtype, device=device)
                value = torch.cat([batch_idx, value], dim=1)

            prepared[key] = value

        # Handle ground truth data - convert numpy to tensor
        for key in ["gt_bboxes_3d", "gt_labels_3d"]:
            if key not in inputs:
                continue
            value = inputs[key]

            # Handle list from pseudo_collate
            if isinstance(value, list):
                converted = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        if key == "gt_labels_3d":
                            v = torch.from_numpy(v).long().to(device)
                        else:
                            v = torch.from_numpy(v).float().to(device)
                    elif isinstance(v, torch.Tensor):
                        v = v.to(device)
                    converted.append(v)
                prepared[key] = converted
            else:
                if isinstance(value, np.ndarray):
                    if key == "gt_labels_3d":
                        value = torch.from_numpy(value).long().to(device)
                    else:
                        value = torch.from_numpy(value).float().to(device)
                elif isinstance(value, torch.Tensor):
                    value = value.to(device)
                prepared[key] = [value]  # Wrap in list for batch processing

        # Copy other fields
        for key, value in inputs.items():
            if key not in prepared:
                prepared[key] = value

        return prepared

    def forward(
        self,
        inputs: Dict = None,
        data_samples: Optional[List] = None,
        mode: str = "tensor",
        **kwargs,
    ) -> Union[Dict, List, Tuple]:
        """Forward function.

        mmengine's _run_forward passes all data_batch fields as kwargs,
        so we need to extract voxels, num_points, coors from kwargs
        and also extract gt_bboxes_3d, gt_labels_3d for training.

        Args:
            inputs (dict): Input dict (may be None when data comes via kwargs).
            data_samples (list, optional): Data samples (may be None).
            mode (str): Forward mode. Options: 'tensor', 'loss', 'predict'.
            **kwargs: Data from DataLoader including:
                - voxels, num_points, coors: Voxelized point cloud data.
                - gt_bboxes_3d, gt_labels_3d: Ground truth for training.
                - sample_idx, lidar_path, etc.: Metadata (ignored).

        Returns:
            Depending on mode:
                - tensor: Feature tensors.
                - loss: Dict of losses.
                - predict: List of predictions.
        """
        # Handle case where inputs come as kwargs from mmengine's _run_forward
        if inputs is None or len(inputs) == 0:
            inputs = {}
            # Extract voxel-related fields from kwargs
            for key in ["voxels", "num_points", "coors", "batch_size"]:
                if key in kwargs:
                    inputs[key] = kwargs[key]

        # Also extract ground truth from kwargs for training mode
        if mode == "loss" and data_samples is None:
            # Create pseudo data_samples from kwargs
            gt_bboxes_3d = kwargs.get("gt_bboxes_3d", None)
            gt_labels_3d = kwargs.get("gt_labels_3d", None)
            if gt_bboxes_3d is not None and gt_labels_3d is not None:
                # Store in inputs for forward_loss to use
                inputs["gt_bboxes_3d"] = gt_bboxes_3d
                inputs["gt_labels_3d"] = gt_labels_3d
                inputs["metainfo"] = {
                    k: v for k, v in kwargs.items()
                    if k not in ["voxels", "num_points", "coors", "gt_bboxes_3d",
                                 "gt_labels_3d", "batch_size", "mode"]
                }

        # Prepare inputs - convert numpy to tensor and handle pseudo_collate lists
        inputs = self._prepare_inputs(inputs)

        if mode == "tensor":
            return self.forward_tensor(inputs)
        elif mode == "loss":
            return self.forward_loss(inputs, data_samples)
        elif mode == "predict":
            return self.forward_predict(inputs, data_samples)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward_tensor(self, inputs: Dict) -> Tuple[Tensor, ...]:
        """Forward function for tensor mode.

        Args:
            inputs (dict): Input dict.

        Returns:
            tuple[Tensor]: Multi-scale feature tensors.
        """
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]
        batch_size = inputs.get("batch_size", 1)

        return self.extract_feat(voxels, num_points, coors, batch_size)

    def forward_loss(
        self, inputs: Dict, data_samples: Optional[List] = None
    ) -> Dict[str, Tensor]:
        """Forward function for loss mode (training).

        Args:
            inputs (dict): Input dict containing voxels, num_points, coors.
                May also contain gt_bboxes_3d, gt_labels_3d if data_samples is None.
            data_samples (list, optional): Data samples containing ground truth.

        Returns:
            dict[str, Tensor]: Dict of losses.
        """
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]

        # Determine batch size
        if data_samples is not None:
            batch_size = inputs.get("batch_size", len(data_samples))
        else:
            batch_size = inputs.get("batch_size", 1)

        # Extract features
        x = self.extract_feat(voxels, num_points, coors, batch_size)

        # Get predictions from bbox head
        outs = self.bbox_head(x)

        # Parse ground truth - from data_samples or inputs
        if data_samples is not None:
            gt_bboxes_3d = [ds.gt_bboxes_3d for ds in data_samples]
            gt_labels_3d = [ds.gt_labels_3d for ds in data_samples]
            input_metas = [ds.metainfo for ds in data_samples]
        else:
            # Ground truth from inputs (set by forward() from kwargs)
            gt_bboxes_3d = inputs.get("gt_bboxes_3d")
            gt_labels_3d = inputs.get("gt_labels_3d")
            input_metas = inputs.get("metainfo", {})

            # Wrap in lists if not already (for batch_size=1)
            if gt_bboxes_3d is not None and not isinstance(gt_bboxes_3d, list):
                gt_bboxes_3d = [gt_bboxes_3d]
            if gt_labels_3d is not None and not isinstance(gt_labels_3d, list):
                gt_labels_3d = [gt_labels_3d]
            if not isinstance(input_metas, list):
                input_metas = [input_metas]

        # Compute losses
        losses = self.bbox_head.loss(
            *outs, gt_bboxes_3d, gt_labels_3d, input_metas
        )

        return losses

    def forward_predict(
        self, inputs: Dict, data_samples: Optional[List] = None
    ) -> List:
        """Forward function for predict mode (inference).

        Args:
            inputs (dict): Input dict containing voxels, num_points, coors.
            data_samples (list, optional): Data samples.

        Returns:
            list: Detection results for each sample.
        """
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]
        batch_size = inputs.get("batch_size", 1)

        # Extract features
        x = self.extract_feat(voxels, num_points, coors, batch_size)

        # Get predictions from bbox head
        outs = self.bbox_head(x)

        # Parse input metas
        if data_samples is not None:
            input_metas = [ds.metainfo for ds in data_samples]
        else:
            input_metas = [{}] * batch_size

        # Get bboxes
        results = self.bbox_head.get_bboxes(*outs, input_metas)

        # Format results
        predictions = []
        for i, (bboxes, scores, labels) in enumerate(results):
            pred = {
                "bboxes_3d": bboxes,
                "scores_3d": scores,
                "labels_3d": labels,
            }
            predictions.append(pred)

        return predictions


@MODELS.register_module()
class VoxelNet(PointPillars):
    """VoxelNet 3D object detector.

    VoxelNet is similar to PointPillars but uses different voxel encoding.
    This is an alias for backward compatibility.
    """

    pass


@MODELS.register_module()
class SingleStage3DDetector(BaseModel):
    """Base class for single-stage 3D detectors.

    This provides a common interface for single-stage 3D detection models.

    Args:
        voxel_encoder (dict): Config of voxel encoder.
        middle_encoder (dict): Config of middle encoder.
        backbone (dict): Config of backbone.
        neck (dict, optional): Config of neck.
        bbox_head (dict): Config of bbox head.
        train_cfg (dict, optional): Config for training.
        test_cfg (dict, optional): Config for testing.
        init_cfg (dict, optional): Config for initialization.
    """

    def __init__(
        self,
        voxel_encoder: Dict,
        middle_encoder: Dict,
        backbone: Dict,
        neck: Optional[Dict] = None,
        bbox_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
        data_preprocessor: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build components
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        if bbox_head is not None:
            bbox_head["train_cfg"] = train_cfg
            bbox_head["test_cfg"] = test_cfg
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

    def extract_feat(
        self,
        voxels: Tensor,
        num_points: Tensor,
        coors: Tensor,
        batch_size: int,
    ) -> Tuple[Tensor, ...]:
        """Extract features from voxelized point cloud."""
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        spatial_features = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(spatial_features)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(
        self,
        inputs: Dict = None,
        data_samples: Optional[List] = None,
        mode: str = "tensor",
        **kwargs,
    ) -> Union[Dict, List, Tuple]:
        """Forward function with different modes.

        Args:
            inputs (dict): Input dict containing voxels, num_points, coors.
            data_samples (list, optional): Data samples containing annotations.
            mode (str): Forward mode. Options: 'tensor', 'loss', 'predict'.
            **kwargs: Additional arguments from DataLoader (sample_idx, etc.).
                These are ignored but accepted for compatibility with mmengine.
        """
        # Handle case where inputs come as kwargs from mmengine's _run_forward
        if inputs is None:
            inputs = {}
            for key in ["voxels", "num_points", "coors", "batch_size"]:
                if key in kwargs:
                    inputs[key] = kwargs[key]

        if mode == "tensor":
            voxels = inputs["voxels"]
            num_points = inputs["num_points"]
            coors = inputs["coors"]
            batch_size = inputs.get("batch_size", 1)
            return self.extract_feat(voxels, num_points, coors, batch_size)
        elif mode == "loss":
            return self._forward_loss(inputs, data_samples)
        elif mode == "predict":
            return self._forward_predict(inputs, data_samples)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _forward_loss(self, inputs: Dict, data_samples: List) -> Dict[str, Tensor]:
        """Forward for training."""
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]
        batch_size = inputs.get("batch_size", len(data_samples))

        x = self.extract_feat(voxels, num_points, coors, batch_size)
        outs = self.bbox_head(x)

        gt_bboxes_3d = [ds.gt_bboxes_3d for ds in data_samples]
        gt_labels_3d = [ds.gt_labels_3d for ds in data_samples]
        input_metas = [ds.metainfo for ds in data_samples]

        losses = self.bbox_head.loss(*outs, gt_bboxes_3d, gt_labels_3d, input_metas)
        return losses

    def _forward_predict(self, inputs: Dict, data_samples: Optional[List]) -> List:
        """Forward for inference."""
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]
        batch_size = inputs.get("batch_size", 1)

        x = self.extract_feat(voxels, num_points, coors, batch_size)
        outs = self.bbox_head(x)

        if data_samples is not None:
            input_metas = [ds.metainfo for ds in data_samples]
        else:
            input_metas = [{}] * batch_size

        results = self.bbox_head.get_bboxes(*outs, input_metas)

        predictions = []
        for bboxes, scores, labels in results:
            predictions.append({
                "bboxes_3d": bboxes,
                "scores_3d": scores,
                "labels_3d": labels,
            })
        return predictions
