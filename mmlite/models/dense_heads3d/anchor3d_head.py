"""Anchor3DHead - 3D 检测头

基于 anchor 的 3D 检测头，用于 PointPillars、SECOND 等检测器。
负责预测 3D 边界框的分类分数和回归参数。
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from .anchor_generator import Anchor3DRangeGenerator


def multi_apply(func, *args, **kwargs):
    """Apply function to multiple inputs.

    Args:
        func (callable): A function that will be applied to multiple inputs.
        args: Positional arguments that will be unpacked for each call.
        kwargs: Keyword arguments that will be passed to all calls.

    Returns:
        tuple: Results of multiple calls to func.
    """
    pfunc = func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@MODELS.register_module()
class Anchor3DHead(BaseModule):
    """Anchor-based head for 3D object detection.

    Args:
        num_classes (int): Number of classes for classification.
        in_channels (int): Number of input channels.
        feat_channels (int, optional): Number of feature channels.
            Defaults to 256.
        use_direction_classifier (bool, optional): Whether to use direction
            classifier. Defaults to True.
        anchor_generator (dict, optional): Config dict for anchor generator.
        diff_rad_by_sin (bool, optional): Whether to use sin difference for
            orientation regression. Defaults to True.
        dir_offset (float, optional): The offset of direction angle.
            Defaults to 0.
        dir_limit_offset (float, optional): The offset to limit direction.
            Defaults to 0.
        bbox_coder (dict, optional): Config dict for bbox coder.
        loss_cls (dict, optional): Config dict for classification loss.
        loss_bbox (dict, optional): Config dict for bbox regression loss.
        loss_dir (dict, optional): Config dict for direction loss.
        train_cfg (dict, optional): Config dict for training.
        test_cfg (dict, optional): Config dict for testing.
        init_cfg (dict, optional): Config dict for initialization.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        use_direction_classifier: bool = True,
        anchor_generator: Optional[Dict] = None,
        diff_rad_by_sin: bool = True,
        dir_offset: float = 0,
        dir_limit_offset: float = 0,
        bbox_coder: Optional[Dict] = None,
        loss_cls: Optional[Dict] = None,
        loss_bbox: Optional[Dict] = None,
        loss_dir: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build anchor generator
        if anchor_generator is not None:
            anchor_generator_cfg = anchor_generator.copy()
            anchor_generator_type = anchor_generator_cfg.pop("type", None)
            self.anchor_generator = Anchor3DRangeGenerator(**anchor_generator_cfg)
        else:
            # Default anchor generator for KITTI
            self.anchor_generator = Anchor3DRangeGenerator(
                ranges=[
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78],
                ],
                sizes=[
                    [0.8, 0.6, 1.73],  # Pedestrian
                    [1.76, 0.6, 1.73],  # Cyclist
                    [1.6, 3.9, 1.56],  # Car
                ],
                rotations=[0, 1.5707963],
            )

        self.num_anchors = self.anchor_generator.num_base_anchors
        self.box_code_size = 7  # [x, y, z, w, l, h, rot]

        # Build bbox coder
        if bbox_coder is None:
            self.bbox_coder = BBox3DCoder()
        elif isinstance(bbox_coder, dict):
            # Build from config dict - support DeltaXYZWLHRBBoxCoder type
            bbox_coder_type = bbox_coder.get("type", "BBox3DCoder")
            # For now, we only support BBox3DCoder (same as DeltaXYZWLHRBBoxCoder)
            self.bbox_coder = BBox3DCoder()
        else:
            # Already a coder instance
            self.bbox_coder = bbox_coder

        # Build loss functions
        self.loss_cls = self._build_loss(
            loss_cls, default=dict(type="FocalLoss", gamma=2.0, alpha=0.25, loss_weight=1.0)
        )
        self.loss_bbox = self._build_loss(
            loss_bbox, default=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0)
        )
        if self.use_direction_classifier:
            self.loss_dir = self._build_loss(
                loss_dir, default=dict(type="CrossEntropyLoss", loss_weight=0.2)
            )
        else:
            self.loss_dir = None

        self._init_layers()

    def _build_loss(self, loss_cfg: Optional[Dict], default: Dict) -> nn.Module:
        """Build loss function."""
        if loss_cfg is None:
            loss_cfg = default

        loss_type = loss_cfg.get("type", "SmoothL1Loss")
        loss_weight = loss_cfg.get("loss_weight", 1.0)

        if loss_type == "FocalLoss":
            gamma = loss_cfg.get("gamma", 2.0)
            alpha = loss_cfg.get("alpha", 0.25)
            return FocalLoss(gamma=gamma, alpha=alpha, loss_weight=loss_weight)
        elif loss_type == "SmoothL1Loss":
            beta = loss_cfg.get("beta", 1.0)
            return SmoothL1Loss(beta=beta, loss_weight=loss_weight)
        elif loss_type == "CrossEntropyLoss":
            return CrossEntropyLoss(loss_weight=loss_weight)
        else:
            return SmoothL1Loss(loss_weight=loss_weight)

    def _init_layers(self) -> None:
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.box_code_size, 1
        )
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        """Forward function for Anchor3DHead.

        Args:
            x (tuple[Tensor]): Features from FPN.

        Returns:
            tuple: Contains predictions.
                - cls_scores (list[Tensor]): Classification scores for each
                    feature level with shape (N, num_anchors * num_classes, H, W).
                - bbox_preds (list[Tensor]): Box predictions for each feature
                    level with shape (N, num_anchors * box_code_size, H, W).
                - dir_cls_preds (list[Tensor], optional): Direction predictions
                    for each feature level with shape (N, num_anchors * 2, H, W).
        """
        return multi_apply(self.forward_single, x)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function for single feature level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                - cls_score (Tensor): Cls scores for a single scale level.
                - bbox_pred (Tensor): Bbox predictions for a single scale level.
                - dir_cls_pred (Tensor, optional): Direction class predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
            return cls_score, bbox_pred, dir_cls_pred
        else:
            return cls_score, bbox_pred, None

    def loss(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dir_cls_preds: Optional[List[Tensor]],
        gt_bboxes_3d: List[Tensor],
        gt_labels_3d: List[Tensor],
        input_metas: List[Dict],
    ) -> Dict[str, Tensor]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale
                level with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predictions for each scale level
                with shape (N, num_anchors * box_code_size, H, W).
            dir_cls_preds (list[Tensor], optional): Direction class predictions
                for each scale level with shape (N, num_anchors * 2, H, W).
            gt_bboxes_3d (list[Tensor]): Ground truth 3D bboxes.
            gt_labels_3d (list[Tensor]): Ground truth labels.
            input_metas (list[dict]): Meta info of each sample.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        device = cls_scores[0].device

        # Get anchors for all feature levels
        anchor_list = self.anchor_generator.grid_anchors(
            featmap_sizes, device=str(device)
        )

        # Compute targets
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(
            anchor_list, gt_bboxes_3d, gt_labels_3d, input_metas
        )

        # Concat all levels
        num_images = cls_scores[0].shape[0]
        all_cls_scores = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(num_images, -1, self.num_classes) for s in cls_scores],
            dim=1,
        )
        all_bbox_preds = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(num_images, -1, self.box_code_size) for b in bbox_preds],
            dim=1,
        )
        all_labels = torch.cat(labels_list, dim=1)
        all_label_weights = torch.cat(label_weights_list, dim=1)
        all_bbox_targets = torch.cat(bbox_targets_list, dim=1)
        all_bbox_weights = torch.cat(bbox_weights_list, dim=1)

        # Classification loss
        all_cls_scores = all_cls_scores.reshape(-1, self.num_classes)
        all_labels = all_labels.reshape(-1)
        all_label_weights = all_label_weights.reshape(-1)

        loss_cls = self.loss_cls(
            all_cls_scores, all_labels, all_label_weights, avg_factor=max(num_total_pos, 1)
        )

        # Regression loss
        all_bbox_preds = all_bbox_preds.reshape(-1, self.box_code_size)
        all_bbox_targets = all_bbox_targets.reshape(-1, self.box_code_size)
        all_bbox_weights = all_bbox_weights.reshape(-1, self.box_code_size)

        loss_bbox = self.loss_bbox(
            all_bbox_preds,
            all_bbox_targets,
            all_bbox_weights,
            avg_factor=max(num_total_pos, 1),
        )

        losses = dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

        # Direction classification loss
        if self.use_direction_classifier and dir_cls_preds is not None:
            all_dir_cls_preds = torch.cat(
                [d.permute(0, 2, 3, 1).reshape(num_images, -1, 2) for d in dir_cls_preds if d is not None],
                dim=1,
            )
            all_dir_targets = torch.cat(dir_targets_list, dim=1)
            all_dir_weights = torch.cat(dir_weights_list, dim=1)

            all_dir_cls_preds = all_dir_cls_preds.reshape(-1, 2)
            all_dir_targets = all_dir_targets.reshape(-1)
            all_dir_weights = all_dir_weights.reshape(-1)

            loss_dir = self.loss_dir(
                all_dir_cls_preds,
                all_dir_targets,
                all_dir_weights,
                avg_factor=max(num_total_pos, 1),
            )
            losses["loss_dir"] = loss_dir

        return losses

    def get_targets(
        self,
        anchor_list: List[Tensor],
        gt_bboxes_3d_list: List[Tensor],
        gt_labels_3d_list: List[Tensor],
        input_metas: List[Dict],
    ) -> Tuple:
        """Compute targets for anchors in all images.

        Args:
            anchor_list (list[Tensor]): Anchors of each feature level.
            gt_bboxes_3d_list (list[Tensor]): Ground truth 3D bboxes.
            gt_labels_3d_list (list[Tensor]): Ground truth labels.
            input_metas (list[dict]): Meta info of each image.

        Returns:
            tuple: Targets for training.
        """
        num_imgs = len(input_metas)
        num_levels = len(anchor_list)

        # Initialize outputs
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []
        dir_targets_list = []
        dir_weights_list = []
        num_total_pos = 0
        num_total_neg = 0

        for img_id in range(num_imgs):
            gt_bboxes_3d = gt_bboxes_3d_list[img_id]
            gt_labels_3d = gt_labels_3d_list[img_id]

            # Concat anchors from all levels for this image
            all_anchors = torch.cat(
                [a.reshape(-1, a.shape[-1]) for a in anchor_list], dim=0
            )
            num_anchors = all_anchors.shape[0]

            # Assign targets
            labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds = (
                self._get_targets_single(
                    all_anchors, gt_bboxes_3d, gt_labels_3d
                )
            )

            num_total_pos += len(pos_inds)
            num_total_neg += len(neg_inds)

            # Split by levels
            level_start = 0
            level_labels = []
            level_label_weights = []
            level_bbox_targets = []
            level_bbox_weights = []
            level_dir_targets = []
            level_dir_weights = []

            for level_id, anchors in enumerate(anchor_list):
                level_size = anchors.reshape(-1, anchors.shape[-1]).shape[0]
                level_end = level_start + level_size

                level_labels.append(
                    labels[level_start:level_end].reshape(1, -1)
                )
                level_label_weights.append(
                    label_weights[level_start:level_end].reshape(1, -1)
                )
                level_bbox_targets.append(
                    bbox_targets[level_start:level_end].reshape(1, -1, self.box_code_size)
                )
                level_bbox_weights.append(
                    bbox_weights[level_start:level_end].reshape(1, -1, self.box_code_size)
                )
                level_dir_targets.append(
                    dir_targets[level_start:level_end].reshape(1, -1)
                )
                level_dir_weights.append(
                    dir_weights[level_start:level_end].reshape(1, -1)
                )

                level_start = level_end

            labels_list.append(level_labels)
            label_weights_list.append(level_label_weights)
            bbox_targets_list.append(level_bbox_targets)
            bbox_weights_list.append(level_bbox_weights)
            dir_targets_list.append(level_dir_targets)
            dir_weights_list.append(level_dir_weights)

        # Concat images
        # Shape: [num_levels][num_images, ...]
        labels_list = [
            torch.cat([labels_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]
        label_weights_list = [
            torch.cat([label_weights_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]
        bbox_targets_list = [
            torch.cat([bbox_targets_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]
        bbox_weights_list = [
            torch.cat([bbox_weights_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]
        dir_targets_list = [
            torch.cat([dir_targets_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]
        dir_weights_list = [
            torch.cat([dir_weights_list[img][lvl] for img in range(num_imgs)], dim=0)
            for lvl in range(num_levels)
        ]

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_targets_single(
        self,
        anchors: Tensor,
        gt_bboxes_3d: Tensor,
        gt_labels_3d: Tensor,
    ) -> Tuple:
        """Compute targets for a single image.

        Args:
            anchors (Tensor): All anchors for this image with shape (N, 7+).
            gt_bboxes_3d (Tensor): Ground truth 3D bboxes with shape (M, 7).
            gt_labels_3d (Tensor): Ground truth labels with shape (M,).

        Returns:
            tuple: Targets for this image.
        """
        num_anchors = anchors.shape[0]
        device = anchors.device

        # Initialize
        labels = torch.full((num_anchors,), self.num_classes, dtype=torch.long, device=device)
        label_weights = torch.zeros(num_anchors, dtype=torch.float, device=device)
        bbox_targets = torch.zeros((num_anchors, self.box_code_size), dtype=torch.float, device=device)
        bbox_weights = torch.zeros((num_anchors, self.box_code_size), dtype=torch.float, device=device)
        dir_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
        dir_weights = torch.zeros(num_anchors, dtype=torch.float, device=device)

        if gt_bboxes_3d.numel() == 0:
            # No ground truth
            label_weights.fill_(1.0)
            return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, [], []

        # Compute IoU between anchors and gt
        # Simplified: use center distance for assignment
        anchor_centers = anchors[:, :3]  # [N, 3]
        gt_centers = gt_bboxes_3d[:, :3]  # [M, 3]

        # Distance matrix [N, M]
        dist_matrix = torch.cdist(anchor_centers, gt_centers)

        # Simple assignment: assign to nearest gt if within threshold
        min_dist, assigned_gt = dist_matrix.min(dim=1)

        # Positive threshold (within 2m of gt center as rough approximation)
        pos_thresh = 2.0
        neg_thresh = 4.0

        pos_inds = (min_dist < pos_thresh).nonzero(as_tuple=False).squeeze(-1)
        neg_inds = (min_dist > neg_thresh).nonzero(as_tuple=False).squeeze(-1)

        # Assign labels
        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels_3d[assigned_gt[pos_inds]]
            label_weights[pos_inds] = 1.0
            bbox_weights[pos_inds] = 1.0
            dir_weights[pos_inds] = 1.0

            # Encode bbox targets
            pos_anchors = anchors[pos_inds]
            pos_gt_bboxes = gt_bboxes_3d[assigned_gt[pos_inds]]
            bbox_targets[pos_inds] = self.bbox_coder.encode(pos_anchors, pos_gt_bboxes)

            # Direction targets
            dir_targets[pos_inds] = self._get_direction_target(pos_gt_bboxes)

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds

    def _get_direction_target(self, gt_bboxes_3d: Tensor) -> Tensor:
        """Get direction classification target.

        Args:
            gt_bboxes_3d (Tensor): Ground truth 3D bboxes with shape (N, 7).

        Returns:
            Tensor: Direction targets with shape (N,).
        """
        rot = gt_bboxes_3d[:, 6]
        dir_offset = self.dir_offset
        dir_limit_offset = self.dir_limit_offset

        # Normalize rotation to [0, 2*pi)
        rot = rot - dir_offset
        rot = rot % (2 * np.pi)

        # Binary classification: 0 for [0, pi), 1 for [pi, 2*pi)
        dir_targets = (rot > np.pi).long()
        return dir_targets

    def get_bboxes(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dir_cls_preds: Optional[List[Tensor]],
        input_metas: List[Dict],
        cfg: Optional[Dict] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Transform network outputs to 3D bboxes.

        Args:
            cls_scores (list[Tensor]): Classification scores.
            bbox_preds (list[Tensor]): Box predictions.
            dir_cls_preds (list[Tensor], optional): Direction predictions.
            input_metas (list[dict]): Meta info of each image.
            cfg (dict, optional): Test config.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Detection results.
                Each tuple contains (bboxes, scores, labels).
        """
        cfg = self.test_cfg if cfg is None else cfg
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        device = cls_scores[0].device

        anchor_list = self.anchor_generator.grid_anchors(
            featmap_sizes, device=str(device)
        )

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].permute(1, 2, 0).reshape(-1, self.num_classes)
                for i in range(len(cls_scores))
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, self.box_code_size)
                for i in range(len(bbox_preds))
            ]

            if dir_cls_preds is not None:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].permute(1, 2, 0).reshape(-1, 2)
                    if dir_cls_preds[i] is not None
                    else None
                    for i in range(len(dir_cls_preds))
                ]
            else:
                dir_cls_pred_list = None

            result = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                dir_cls_pred_list,
                anchor_list,
                input_metas[img_id],
                cfg,
            )
            result_list.append(result)

        return result_list

    def _get_bboxes_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        dir_cls_pred_list: Optional[List[Tensor]],
        anchor_list: List[Tensor],
        input_meta: Dict,
        cfg: Dict,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get bboxes for a single image.

        Args:
            cls_score_list (list[Tensor]): Classification scores per level.
            bbox_pred_list (list[Tensor]): Box predictions per level.
            dir_cls_pred_list (list[Tensor], optional): Direction predictions.
            anchor_list (list[Tensor]): Anchors per level.
            input_meta (dict): Meta info of this image.
            cfg (dict): Test config.

        Returns:
            tuple[Tensor, Tensor, Tensor]: (bboxes, scores, labels).
        """
        # Concat all levels
        all_cls_scores = torch.cat(cls_score_list, dim=0)
        all_bbox_preds = torch.cat(bbox_pred_list, dim=0)
        all_anchors = torch.cat([a.reshape(-1, a.shape[-1]) for a in anchor_list], dim=0)

        if dir_cls_pred_list is not None:
            all_dir_cls_preds = torch.cat(
                [d for d in dir_cls_pred_list if d is not None], dim=0
            )
        else:
            all_dir_cls_preds = None

        # Apply sigmoid/softmax to classification scores
        all_cls_scores = all_cls_scores.sigmoid()

        # Get max score and label per anchor
        max_scores, pred_labels = all_cls_scores.max(dim=1)

        # Filter by score threshold
        score_thresh = cfg.get("score_thr", 0.1)
        keep_mask = max_scores > score_thresh
        
        if keep_mask.sum() == 0:
            # No detections
            return (
                torch.zeros((0, 7), device=all_cls_scores.device),
                torch.zeros((0,), device=all_cls_scores.device),
                torch.zeros((0,), dtype=torch.long, device=all_cls_scores.device),
            )

        # Filter predictions
        scores = max_scores[keep_mask]
        labels = pred_labels[keep_mask]
        bbox_preds = all_bbox_preds[keep_mask]
        anchors = all_anchors[keep_mask]

        # Decode bboxes
        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        # Apply direction classification
        if all_dir_cls_preds is not None and self.use_direction_classifier:
            dir_preds = all_dir_cls_preds[keep_mask]
            dir_cls = dir_preds.argmax(dim=1)

            # Adjust rotation based on direction
            rot = bboxes[:, 6]
            dir_offset = self.dir_offset

            # If predicted direction is 1, add pi to rotation
            rot = rot + dir_offset
            rot_adjusted = torch.where(
                dir_cls == 1,
                rot + np.pi,
                rot,
            )
            # Normalize to [-pi, pi]
            rot_adjusted = torch.atan2(torch.sin(rot_adjusted), torch.cos(rot_adjusted))
            bboxes[:, 6] = rot_adjusted

        # NMS
        nms_thresh = cfg.get("nms_thr", 0.01)
        max_num = cfg.get("max_num", 500)
        nms_pre = cfg.get("nms_pre", 1000)  # Limit boxes before NMS

        # Limit number of boxes before NMS to save memory
        if scores.shape[0] > nms_pre:
            _, topk_inds = scores.topk(nms_pre)
            scores = scores[topk_inds]
            labels = labels[topk_inds]
            bboxes = bboxes[topk_inds]

        keep_inds = self._nms_3d(bboxes, scores, nms_thresh)
        keep_inds = keep_inds[:max_num]

        bboxes = bboxes[keep_inds]
        scores = scores[keep_inds]
        labels = labels[keep_inds]

        return bboxes, scores, labels

    def _nms_3d(
        self, bboxes: Tensor, scores: Tensor, nms_thresh: float
    ) -> Tensor:
        """Simple 3D NMS based on BEV IoU.

        Args:
            bboxes (Tensor): 3D bboxes with shape (N, 7).
            scores (Tensor): Scores with shape (N,).
            nms_thresh (float): NMS IoU threshold.

        Returns:
            Tensor: Indices of kept boxes.
        """
        if bboxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=bboxes.device)

        # Sort by score
        order = scores.argsort(descending=True)
        bboxes = bboxes[order]

        keep = []
        while bboxes.shape[0] > 0:
            keep.append(order[0])
            if bboxes.shape[0] == 1:
                break

            # Compute BEV IoU between first box and rest
            ious = self._bev_iou(bboxes[0:1], bboxes[1:])
            
            # Keep boxes with low IoU
            mask = ious < nms_thresh
            bboxes = bboxes[1:][mask]
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=scores.device)

    def _bev_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """Compute BEV IoU (simplified: axis-aligned approximation).

        Args:
            boxes1 (Tensor): First boxes with shape (N, 7).
            boxes2 (Tensor): Second boxes with shape (M, 7).

        Returns:
            Tensor: IoU matrix with shape (M,) for each box2 vs box1.
        """
        # Extract BEV params: x, y, w, l
        x1, y1, w1, l1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 3], boxes1[:, 4]
        x2, y2, w2, l2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 3], boxes2[:, 4]

        # Axis-aligned bounding boxes
        half_w1, half_l1 = w1 / 2, l1 / 2
        half_w2, half_l2 = w2 / 2, l2 / 2

        # Box1 bounds
        x1_min, x1_max = x1 - half_w1, x1 + half_w1
        y1_min, y1_max = y1 - half_l1, y1 + half_l1

        # Box2 bounds
        x2_min, x2_max = x2 - half_w2, x2 + half_w2
        y2_min, y2_max = y2 - half_l2, y2 + half_l2

        # Intersection
        inter_x_min = torch.maximum(x1_min, x2_min)
        inter_x_max = torch.minimum(x1_max, x2_max)
        inter_y_min = torch.maximum(y1_min, y2_min)
        inter_y_max = torch.minimum(y1_max, y2_max)

        inter_w = (inter_x_max - inter_x_min).clamp(min=0)
        inter_l = (inter_y_max - inter_y_min).clamp(min=0)
        inter_area = inter_w * inter_l

        # Union
        area1 = w1 * l1
        area2 = w2 * l2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)
        return iou


class BBox3DCoder:
    """3D Bounding Box Coder.

    Encodes and decodes 3D bounding boxes for anchor-based detection.
    """

    def encode(self, anchors: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Encode ground truth bboxes to target deltas.

        Args:
            anchors (Tensor): Anchors with shape (N, 7) - [x, y, z, w, l, h, rot].
            gt_bboxes (Tensor): Ground truth bboxes with shape (N, 7).

        Returns:
            Tensor: Encoded deltas with shape (N, 7).
        """
        # Anchor params
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        ra = anchors[:, 6]
        diagonal = torch.sqrt(wa ** 2 + la ** 2)

        # GT params
        xg, yg, zg = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2]
        wg, lg, hg = gt_bboxes[:, 3], gt_bboxes[:, 4], gt_bboxes[:, 5]
        rg = gt_bboxes[:, 6]

        # Encode
        dx = (xg - xa) / diagonal
        dy = (yg - ya) / diagonal
        dz = (zg - za) / ha
        dw = torch.log(wg / wa)
        dl = torch.log(lg / la)
        dh = torch.log(hg / ha)
        dr = rg - ra

        deltas = torch.stack([dx, dy, dz, dw, dl, dh, dr], dim=-1)
        return deltas

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        """Decode deltas to 3D bboxes.

        Args:
            anchors (Tensor): Anchors with shape (N, 7).
            deltas (Tensor): Deltas with shape (N, 7).

        Returns:
            Tensor: Decoded bboxes with shape (N, 7).
        """
        # Anchor params
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        ra = anchors[:, 6]
        diagonal = torch.sqrt(wa ** 2 + la ** 2)

        # Delta params
        dx, dy, dz = deltas[:, 0], deltas[:, 1], deltas[:, 2]
        dw, dl, dh = deltas[:, 3], deltas[:, 4], deltas[:, 5]
        dr = deltas[:, 6]

        # Decode
        xg = dx * diagonal + xa
        yg = dy * diagonal + ya
        zg = dz * ha + za
        wg = torch.exp(dw) * wa
        lg = torch.exp(dl) * la
        hg = torch.exp(dh) * ha
        rg = dr + ra

        bboxes = torch.stack([xg, yg, zg, wg, lg, hg, rg], dim=-1)
        return bboxes


class FocalLoss(nn.Module):
    """Focal Loss for classification."""

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, loss_weight: float = 1.0
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor,
        avg_factor: float = 1.0,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predictions with shape (N, C).
            target (Tensor): Labels with shape (N,).
            weight (Tensor): Weights with shape (N,).
            avg_factor (float): Average factor.

        Returns:
            Tensor: Loss value.
        """
        num_classes = pred.shape[1]

        # One-hot encode targets
        target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).float()

        # Sigmoid probabilities
        p = pred.sigmoid()

        # Focal weight
        pt = p * target_one_hot + (1 - p) * (1 - target_one_hot)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weight
        alpha_weight = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction="none")

        # Final loss
        loss = focal_weight * alpha_weight * bce
        loss = loss.sum(dim=1) * weight
        loss = loss.sum() / avg_factor * self.loss_weight

        return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for regression."""

    def __init__(self, beta: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor,
        avg_factor: float = 1.0,
    ) -> Tensor:
        """Forward function."""
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )
        loss = (loss * weight).sum() / avg_factor * self.loss_weight
        return loss


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss for direction classification."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor,
        avg_factor: float = 1.0,
    ) -> Tensor:
        """Forward function."""
        loss = F.cross_entropy(pred, target, reduction="none")
        loss = (loss * weight).sum() / avg_factor * self.loss_weight
        return loss
