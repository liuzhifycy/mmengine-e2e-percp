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

    def forward(
        self,
        inputs: Dict,
        data_samples: Optional[List] = None,
        mode: str = "tensor",
    ) -> Union[Dict, List, Tuple]:
        """Forward function.

        Args:
            inputs (dict): Input dict containing:
                - voxels (Tensor): Voxelized point cloud.
                - num_points (Tensor): Number of points in each voxel.
                - coors (Tensor): Coordinates of voxels.
                - batch_size (int): Batch size.
            data_samples (list, optional): Data samples containing annotations.
            mode (str): Forward mode. Options: 'tensor', 'loss', 'predict'.

        Returns:
            Depending on mode:
                - tensor: Feature tensors.
                - loss: Dict of losses.
                - predict: List of predictions.
        """
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
        self, inputs: Dict, data_samples: List
    ) -> Dict[str, Tensor]:
        """Forward function for loss mode (training).

        Args:
            inputs (dict): Input dict containing voxels, num_points, coors.
            data_samples (list): Data samples containing:
                - gt_bboxes_3d (Tensor): Ground truth 3D bboxes.
                - gt_labels_3d (Tensor): Ground truth labels.

        Returns:
            dict[str, Tensor]: Dict of losses.
        """
        voxels = inputs["voxels"]
        num_points = inputs["num_points"]
        coors = inputs["coors"]
        batch_size = inputs.get("batch_size", len(data_samples))

        # Extract features
        x = self.extract_feat(voxels, num_points, coors, batch_size)

        # Get predictions from bbox head
        outs = self.bbox_head(x)

        # Parse ground truth
        gt_bboxes_3d = [ds.gt_bboxes_3d for ds in data_samples]
        gt_labels_3d = [ds.gt_labels_3d for ds in data_samples]
        input_metas = [ds.metainfo for ds in data_samples]

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
        inputs: Dict,
        data_samples: Optional[List] = None,
        mode: str = "tensor",
    ) -> Union[Dict, List, Tuple]:
        """Forward function with different modes."""
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
