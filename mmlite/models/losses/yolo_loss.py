import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.yolo_utils import bbox_iou, bbox2dist, dist2bbox, make_anchors

class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection."""

    def __init__(self, topk=13, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.bg_idx), 
                    torch.zeros_like(pd_bboxes), 
                    torch.zeros_like(pd_scores), 
                    torch.zeros_like(pd_scores[..., 0]), 
                    torch.zeros_like(pd_scores[..., 0]))

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.repeat(1, 1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels.clamp_(0)
        
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_bboxes.device)

        return loss_iou, loss_dfl


class YOLO11Loss(nn.Module):
    def __init__(self, nc, strides, reg_max=16, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.reg_max = reg_max
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max)
        self.proj = torch.arange(reg_max, dtype=torch.float)

    def preprocess_mmdet_targets(self, batch_data_samples, device):
        """
        Convert MMDet batch_data_samples to batched targets tensor.
        Output: (bs, max_num_obj, 5) -> [label, x1, y1, x2, y2]
        """
        batch_size = len(batch_data_samples)
        max_num_obj = max([len(sample.gt_instances) for sample in batch_data_samples])
        
        targets = torch.zeros(batch_size, max_num_obj, 5, device=device)
        
        for i, sample in enumerate(batch_data_samples):
            bboxes = sample.gt_instances.bboxes
            labels = sample.gt_instances.labels
            n = len(bboxes)
            if n > 0:
                targets[i, :n, 0] = labels
                targets[i, :n, 1:5] = bboxes
                
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            if self.proj.device != pred_dist.device:
                self.proj = self.proj.to(pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds, batch_data_samples):
        box_preds, cls_preds = preds
        device = box_preds[0].device
        
        # 1. Flatten
        pred_distri = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_preds], 1)
        pred_scores = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_preds], 1)
        
        # 2. Anchors
        anchor_points, stride_tensor = make_anchors(box_preds, self.strides, 0.5)
        
        # 3. Decode
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # 4. Targets
        targets = self.preprocess_mmdet_targets(batch_data_samples, device)
        gt_labels = targets[..., 0:1]
        gt_bboxes = targets[..., 1:5]
        mask_gt = (gt_bboxes.sum(dim=2, keepdim=True) > 0).float()
        
        # 5. Assign
        # Convert to absolute coords for assigner (grid * stride)
        pred_bboxes_abs = pred_bboxes * stride_tensor
        anchor_points_abs = anchor_points * stride_tensor
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_abs.detach(),
            anchor_points_abs,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        # 6. Loss
        target_scores_sum = max(target_scores.sum(), 1)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        if fg_mask.sum():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor, # back to grid scale for loss
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            
        losses = dict(
            loss_box=loss_box * self.box_gain,
            loss_cls=loss_cls * self.cls_gain,
            loss_dfl=loss_dfl * self.dfl_gain
        )
        return losses