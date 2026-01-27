"""KITTI 3D Detection Metric - KITTI 3D 检测评估指标

实现 KITTI 3D 目标检测的评估指标，包括:
- 3D AP (Average Precision)
- BEV AP (Bird's Eye View)
- 2D AP (Image)

支持 Easy, Moderate, Hard 三个难度级别。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.registry import METRICS


@METRICS.register_module()
class KittiMetric(BaseMetric):
    """KITTI evaluation metric for 3D object detection.

    Args:
        ann_file (str, optional): Path to the annotation file.
        metric (str or list[str]): Metrics to be evaluated. Options: 'bbox',
            'bev', '3d'. Defaults to ['bbox', 'bev', '3d'].
        pcd_limit_range (list[float]): Point cloud range limit.
        prefix (str, optional): Prefix for the metric name.
        collect_device (str): Device name used for collecting results.
    """

    default_prefix: Optional[str] = "kitti"

    def __init__(
        self,
        ann_file: Optional[str] = None,
        metric: Union[str, List[str]] = ["bbox", "bev", "3d"],
        pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
        prefix: Optional[str] = None,
        collect_device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        self.ann_file = ann_file
        self.pcd_limit_range = pcd_limit_range

        if isinstance(metric, str):
            metric = [metric]
        for m in metric:
            assert m in ["bbox", "bev", "3d"], f"Invalid metric: {m}"
        self.metrics = metric

    def process(
        self, data_batch: Any, data_samples: Sequence[Dict]
    ) -> None:
        """Process one batch of data samples.

        Args:
            data_batch: Input data batch (not used).
            data_samples: A batch of data samples that contain predictions
                and ground truths.
        """
        for data_sample in data_samples:
            result = {}

            # Get predictions
            pred = data_sample.get("pred_instances_3d", data_sample)
            result["bboxes_3d"] = self._to_numpy(pred.get("bboxes_3d", []))
            result["scores_3d"] = self._to_numpy(pred.get("scores_3d", []))
            result["labels_3d"] = self._to_numpy(pred.get("labels_3d", []))

            # Get ground truths
            gt = data_sample.get("gt_instances_3d", data_sample)
            result["gt_bboxes_3d"] = self._to_numpy(gt.get("gt_bboxes_3d", gt.get("bboxes_3d", [])))
            result["gt_labels_3d"] = self._to_numpy(gt.get("gt_labels_3d", gt.get("labels_3d", [])))

            # Get sample info
            metainfo = data_sample.get("metainfo", {})
            result["sample_idx"] = metainfo.get("sample_idx", "")

            self.results.append(result)

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert data to numpy array."""
        if data is None:
            return np.array([])
        # Handle PyTorch tensors - must move to CPU first if on GPU
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.array(data)

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results: The processed results of each batch.

        Returns:
            dict: The computed metrics.
        """
        # Separate predictions and ground truths
        det_annos = []
        gt_annos = []

        for result in results:
            # Detection results
            det_anno = {
                "name": self._label_to_name(result["labels_3d"]),
                "bbox": self._bbox_3d_to_2d(result["bboxes_3d"]),
                "dimensions": result["bboxes_3d"][:, 3:6] if len(result["bboxes_3d"]) > 0 else np.zeros((0, 3)),
                "location": result["bboxes_3d"][:, :3] if len(result["bboxes_3d"]) > 0 else np.zeros((0, 3)),
                "rotation_y": result["bboxes_3d"][:, 6] if len(result["bboxes_3d"]) > 0 else np.zeros(0),
                "score": result["scores_3d"],
            }
            det_annos.append(det_anno)

            # Ground truth
            gt_anno = {
                "name": self._label_to_name(result["gt_labels_3d"]),
                "bbox": self._bbox_3d_to_2d(result["gt_bboxes_3d"]),
                "dimensions": result["gt_bboxes_3d"][:, 3:6] if len(result["gt_bboxes_3d"]) > 0 else np.zeros((0, 3)),
                "location": result["gt_bboxes_3d"][:, :3] if len(result["gt_bboxes_3d"]) > 0 else np.zeros((0, 3)),
                "rotation_y": result["gt_bboxes_3d"][:, 6] if len(result["gt_bboxes_3d"]) > 0 else np.zeros(0),
                "difficulty": np.zeros(len(result["gt_labels_3d"]), dtype=np.int32),
            }
            gt_annos.append(gt_anno)

        # Compute metrics
        metrics = {}
        class_names = ["Car", "Pedestrian", "Cyclist"]

        for metric_type in self.metrics:
            ap_results = self._eval_ap(det_annos, gt_annos, class_names, metric_type)

            for cls_name, aps in ap_results.items():
                for difficulty, ap in zip(["easy", "moderate", "hard"], aps):
                    key = f"{metric_type}_{cls_name}_{difficulty}"
                    metrics[key] = ap

        # Compute mAP
        for metric_type in self.metrics:
            for difficulty in ["easy", "moderate", "hard"]:
                aps = [
                    metrics.get(f"{metric_type}_{cls}_{difficulty}", 0.0)
                    for cls in class_names
                ]
                metrics[f"{metric_type}_mAP_{difficulty}"] = np.mean(aps)

        # Print results
        self._print_results(metrics, class_names)

        return metrics

    def _label_to_name(self, labels: np.ndarray) -> np.ndarray:
        """Convert label indices to class names."""
        class_names = ["Car", "Pedestrian", "Cyclist"]
        names = []
        for label in labels:
            if 0 <= label < len(class_names):
                names.append(class_names[label])
            else:
                names.append("DontCare")
        return np.array(names)

    def _bbox_3d_to_2d(self, bboxes_3d: np.ndarray) -> np.ndarray:
        """Convert 3D bboxes to 2D bboxes (placeholder)."""
        if len(bboxes_3d) == 0:
            return np.zeros((0, 4))
        # Simple projection (placeholder - actual projection needs calibration)
        # Return dummy 2D boxes based on 3D center
        x = bboxes_3d[:, 0]
        y = bboxes_3d[:, 1]
        w = bboxes_3d[:, 3]
        l = bboxes_3d[:, 4]

        x1 = x - w / 2
        y1 = y - l / 2
        x2 = x + w / 2
        y2 = y + l / 2

        return np.stack([x1, y1, x2, y2], axis=1)

    def _eval_ap(
        self,
        det_annos: List[Dict],
        gt_annos: List[Dict],
        class_names: List[str],
        metric_type: str,
    ) -> Dict[str, List[float]]:
        """Evaluate Average Precision.

        Args:
            det_annos: Detection annotations.
            gt_annos: Ground truth annotations.
            class_names: Class names to evaluate.
            metric_type: Type of metric ('bbox', 'bev', '3d').

        Returns:
            dict: AP results for each class.
        """
        results = {}

        for cls_name in class_names:
            # Filter by class
            dets, gts = self._filter_by_class(det_annos, gt_annos, cls_name)

            # Compute AP for each difficulty
            aps = []
            for difficulty in [0, 1, 2]:  # easy, moderate, hard
                ap = self._compute_ap(dets, gts, difficulty, metric_type)
                aps.append(ap)

            results[cls_name] = aps

        return results

    def _filter_by_class(
        self,
        det_annos: List[Dict],
        gt_annos: List[Dict],
        cls_name: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter annotations by class name."""
        filtered_dets = []
        filtered_gts = []

        for det, gt in zip(det_annos, gt_annos):
            # Filter detections
            det_mask = det["name"] == cls_name
            filtered_det = {
                "bbox": det["bbox"][det_mask],
                "location": det["location"][det_mask],
                "dimensions": det["dimensions"][det_mask],
                "rotation_y": det["rotation_y"][det_mask],
                "score": det["score"][det_mask],
            }
            filtered_dets.append(filtered_det)

            # Filter ground truths
            gt_mask = gt["name"] == cls_name
            filtered_gt = {
                "bbox": gt["bbox"][gt_mask],
                "location": gt["location"][gt_mask],
                "dimensions": gt["dimensions"][gt_mask],
                "rotation_y": gt["rotation_y"][gt_mask],
                "difficulty": gt["difficulty"][gt_mask],
            }
            filtered_gts.append(filtered_gt)

        return filtered_dets, filtered_gts

    def _compute_ap(
        self,
        dets: List[Dict],
        gts: List[Dict],
        difficulty: int,
        metric_type: str,
    ) -> float:
        """Compute Average Precision for one class and difficulty.

        Args:
            dets: Filtered detections.
            gts: Filtered ground truths.
            difficulty: Difficulty level (0=easy, 1=moderate, 2=hard).
            metric_type: Type of metric ('bbox', 'bev', '3d').

        Returns:
            float: Average Precision.
        """
        # Collect all detections and ground truths
        all_scores = []
        all_tp = []
        num_gt = 0

        # IoU threshold based on metric type
        if metric_type == "bbox":
            iou_thresh = 0.7
        elif metric_type == "bev":
            iou_thresh = 0.7
        else:  # 3d
            iou_thresh = 0.7

        for det, gt in zip(dets, gts):
            # Filter by difficulty
            if difficulty < 2:
                gt_mask = gt["difficulty"] <= difficulty
            else:
                gt_mask = np.ones(len(gt["difficulty"]), dtype=bool)

            gt_boxes = gt["location"][gt_mask]
            gt_dims = gt["dimensions"][gt_mask]
            gt_rots = gt["rotation_y"][gt_mask]
            num_gt += len(gt_boxes)

            if len(det["score"]) == 0:
                continue

            det_boxes = det["location"]
            det_dims = det["dimensions"]
            det_rots = det["rotation_y"]
            det_scores = det["score"]

            # Sort by score
            order = np.argsort(-det_scores)
            det_boxes = det_boxes[order]
            det_dims = det_dims[order]
            det_rots = det_rots[order]
            det_scores = det_scores[order]

            # Match detections to ground truths
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for i in range(len(det_boxes)):
                all_scores.append(det_scores[i])

                if len(gt_boxes) == 0:
                    all_tp.append(0)
                    continue

                # Compute IoU
                if metric_type == "bbox":
                    ious = self._compute_iou_2d(det["bbox"][order[i:i+1]], gt["bbox"][gt_mask])
                elif metric_type == "bev":
                    ious = self._compute_iou_bev(
                        det_boxes[i:i+1], det_dims[i:i+1], det_rots[i:i+1],
                        gt_boxes, gt_dims, gt_rots
                    )
                else:  # 3d
                    ious = self._compute_iou_3d(
                        det_boxes[i:i+1], det_dims[i:i+1], det_rots[i:i+1],
                        gt_boxes, gt_dims, gt_rots
                    )

                if len(ious) > 0:
                    max_iou_idx = np.argmax(ious)
                    max_iou = ious[max_iou_idx]

                    if max_iou >= iou_thresh and not gt_matched[max_iou_idx]:
                        all_tp.append(1)
                        gt_matched[max_iou_idx] = True
                    else:
                        all_tp.append(0)
                else:
                    all_tp.append(0)

        if num_gt == 0 or len(all_scores) == 0:
            return 0.0

        # Sort by score
        all_scores = np.array(all_scores)
        all_tp = np.array(all_tp)
        order = np.argsort(-all_scores)
        all_tp = all_tp[order]

        # Compute precision and recall
        tp_cumsum = np.cumsum(all_tp)
        fp_cumsum = np.cumsum(1 - all_tp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return ap * 100  # Convert to percentage

    def _compute_iou_2d(
        self, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute 2D IoU between boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([])

        x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union_area = area1[:, np.newaxis] + area2 - inter_area
        iou = inter_area / np.maximum(union_area, 1e-6)

        return iou.flatten()

    def _compute_iou_bev(
        self,
        boxes1: np.ndarray,
        dims1: np.ndarray,
        rots1: np.ndarray,
        boxes2: np.ndarray,
        dims2: np.ndarray,
        rots2: np.ndarray,
    ) -> np.ndarray:
        """Compute BEV IoU (simplified axis-aligned approximation)."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.array([])

        # Extract x, y, w, l
        x1, y1 = boxes1[:, 0], boxes1[:, 1]
        w1, l1 = dims1[:, 0], dims1[:, 1]

        x2, y2 = boxes2[:, 0], boxes2[:, 1]
        w2, l2 = dims2[:, 0], dims2[:, 1]

        # Axis-aligned bounding boxes
        x1_min = x1 - w1 / 2
        x1_max = x1 + w1 / 2
        y1_min = y1 - l1 / 2
        y1_max = y1 + l1 / 2

        x2_min = x2 - w2 / 2
        x2_max = x2 + w2 / 2
        y2_min = y2 - l2 / 2
        y2_max = y2 + l2 / 2

        # Intersection
        inter_x_min = np.maximum(x1_min[:, np.newaxis], x2_min)
        inter_x_max = np.minimum(x1_max[:, np.newaxis], x2_max)
        inter_y_min = np.maximum(y1_min[:, np.newaxis], y2_min)
        inter_y_max = np.minimum(y1_max[:, np.newaxis], y2_max)

        inter_w = np.maximum(0, inter_x_max - inter_x_min)
        inter_l = np.maximum(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_l

        # Union
        area1 = w1 * l1
        area2 = w2 * l2
        union_area = area1[:, np.newaxis] + area2 - inter_area

        iou = inter_area / np.maximum(union_area, 1e-6)
        return iou.flatten()

    def _compute_iou_3d(
        self,
        boxes1: np.ndarray,
        dims1: np.ndarray,
        rots1: np.ndarray,
        boxes2: np.ndarray,
        dims2: np.ndarray,
        rots2: np.ndarray,
    ) -> np.ndarray:
        """Compute 3D IoU (simplified approximation)."""
        # For simplicity, use BEV IoU * height overlap ratio
        bev_iou = self._compute_iou_bev(boxes1, dims1, rots1, boxes2, dims2, rots2)

        if len(bev_iou) == 0:
            return bev_iou

        # Height overlap
        z1, h1 = boxes1[:, 2], dims1[:, 2]
        z2, h2 = boxes2[:, 2], dims2[:, 2]

        z1_min = z1 - h1 / 2
        z1_max = z1 + h1 / 2
        z2_min = z2 - h2 / 2
        z2_max = z2 + h2 / 2

        inter_z_min = np.maximum(z1_min[:, np.newaxis], z2_min)
        inter_z_max = np.minimum(z1_max[:, np.newaxis], z2_max)
        inter_h = np.maximum(0, inter_z_max - inter_z_min)

        union_h = h1[:, np.newaxis] + h2 - inter_h
        height_ratio = inter_h / np.maximum(union_h, 1e-6)

        # Approximate 3D IoU
        iou_3d = bev_iou.reshape(len(boxes1), -1) * height_ratio
        return iou_3d.flatten()

    def _print_results(
        self, metrics: Dict[str, float], class_names: List[str]
    ) -> None:
        """Print evaluation results."""
        print_log("\n" + "=" * 60, logger="current")
        print_log("KITTI 3D Detection Evaluation Results", logger="current")
        print_log("=" * 60, logger="current")

        for metric_type in self.metrics:
            print_log(f"\n{metric_type.upper()} AP:", logger="current")
            header = f"{'Class':<12} {'Easy':>10} {'Moderate':>10} {'Hard':>10}"
            print_log(header, logger="current")
            print_log("-" * 44, logger="current")

            for cls_name in class_names:
                easy = metrics.get(f"{metric_type}_{cls_name}_easy", 0.0)
                mod = metrics.get(f"{metric_type}_{cls_name}_moderate", 0.0)
                hard = metrics.get(f"{metric_type}_{cls_name}_hard", 0.0)
                print_log(f"{cls_name:<12} {easy:>10.2f} {mod:>10.2f} {hard:>10.2f}", logger="current")

            # mAP
            print_log("-" * 44, logger="current")
            easy = metrics.get(f"{metric_type}_mAP_easy", 0.0)
            mod = metrics.get(f"{metric_type}_mAP_moderate", 0.0)
            hard = metrics.get(f"{metric_type}_mAP_hard", 0.0)
            print_log(f"{'mAP':<12} {easy:>10.2f} {mod:>10.2f} {hard:>10.2f}", logger="current")

        print_log("=" * 60 + "\n", logger="current")
