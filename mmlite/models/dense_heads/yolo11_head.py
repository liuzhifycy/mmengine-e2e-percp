import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms
from ..layers.yolo11_layers import YOLOConv, DWConv, DFL
from ..losses.yolo_loss import YOLO11Loss
from ..utils.yolo_utils import make_anchors

@MODELS.register_module()
class YOLO11Head(BaseModule):
    """
    YOLO11 Detect Head.
    """
    def __init__(self, 
                 nc=80, 
                 ch=(256, 512, 512), # Input channels from Neck
                 strides=(8, 16, 32),
                 reg_max=16,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4 # outputs per anchor
        self.strides = strides # Store strides
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # Box Branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                YOLOConv(x, c2, 3), 
                YOLOConv(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        # Cls Branch (Non-Legacy)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), YOLOConv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), YOLOConv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # Loss Module
        self.loss_module = YOLO11Loss(nc, strides, reg_max)

    def forward(self, x):
        """
        x: List[Tensor] of features from Neck [P3, P4, P5]
        """
        # Box outputs
        # shape: (bs, 4*reg_max, h, w)
        box_preds = [self.cv2[i](x[i]) for i in range(self.nl)]
        
        # Cls outputs
        # shape: (bs, nc, h, w)
        cls_preds = [self.cv3[i](x[i]) for i in range(self.nl)]
        
        return box_preds, cls_preds

    def loss(self, x, batch_data_samples):
        """
        Calculate loss.
        """
        preds = self(x)
        return self.loss_module(preds, batch_data_samples)

    def predict_by_feat(self, preds, batch_img_metas, cfg=None, rescale=False):
        """
        Transform a batch of output features extracted from the head into bbox results.
        """
        box_preds, cls_preds = preds
        batch_size = box_preds[0].shape[0]
        device = box_preds[0].device
        
        # 1. Flatten & Concat
        pred_distri = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_preds], 1)
        pred_scores = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_preds], 1)
        
        # 2. Anchors
        anchor_points, stride_tensor = make_anchors(box_preds, self.strides, 0.5)
        
        # 3. Decode Box (Use loss_module's decode logic to reuse DFL proj)
        pred_bboxes = self.loss_module.bbox_decode(anchor_points, pred_distri)
        
        # Convert Grid -> Absolute Image Coords
        pred_bboxes = pred_bboxes * stride_tensor # [B, N, 4]
        
        # Sigmoid Scores
        pred_scores = pred_scores.sigmoid() # [B, N, C]
        
        # 4. Post-process (NMS) per image
        results_list = []
        
        # Config defaults
        score_thr = cfg.get('score_thr', 0.001) if cfg else 0.001
        nms_cfg = cfg.get('nms', dict(type='nms', iou_threshold=0.7)) if cfg else dict(type='nms', iou_threshold=0.7)
        max_per_img = cfg.get('max_per_img', 300) if cfg else 300
        
        for i in range(batch_size):
            img_meta = batch_img_metas[i]
            scores = pred_scores[i]
            bboxes = pred_bboxes[i]
            
            # Filter low score (optimization)
            # Find max score per anchor
            max_scores, labels = scores.max(dim=1)
            valid_mask = max_scores > score_thr
            
            scores = max_scores[valid_mask]
            labels = labels[valid_mask]
            bboxes = bboxes[valid_mask]
            
            if len(bboxes) == 0:
                empty_res = InstanceData()
                empty_res.bboxes = torch.zeros((0, 4), device=device)
                empty_res.scores = torch.zeros((0,), device=device)
                empty_res.labels = torch.zeros((0,), dtype=torch.long, device=device)
                results_list.append(empty_res)
                continue

            # Rescale to Original Image size
            if rescale:
                scale_factor = img_meta['scale_factor'] # (w_scale, h_scale)
                # scale_factor is usually (w_scale, h_scale) tuple or tensor
                # bboxes is xyxy.
                # x /= w_scale, y /= h_scale
                bboxes[:, 0::2] /= scale_factor[0]
                bboxes[:, 1::2] /= scale_factor[1]
                
                # Clip to ori_shape
                h, w = img_meta['ori_shape'][:2]
                bboxes[:, 0::2].clamp_(0, w)
                bboxes[:, 1::2].clamp_(0, h)
            else:
                # Clip to img_shape
                h, w = img_meta['img_shape'][:2]
                bboxes[:, 0::2].clamp_(0, w)
                bboxes[:, 1::2].clamp_(0, h)
            
            # NMS
            # Using MMDet batched_nms. 
            # It expects [N, 4], [N], [N]. It performs class-agnostic NMS if idxs are same, 
            # or multi-class if idxs differ.
            dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
            
            if max_per_img > 0 and dets.shape[0] > max_per_img:
                dets = dets[:max_per_img]
                keep = keep[:max_per_img]
                
            labels = labels[keep]
            
            results = InstanceData()
            results.bboxes = dets[:, :4]
            results.scores = dets[:, 4]
            results.labels = labels
            results_list.append(results)
            
        return results_list
