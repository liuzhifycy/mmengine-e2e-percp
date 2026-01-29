import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ..layers.yolo11_layers import YOLOConv, DWConv, DFL
from ..losses.yolo_loss import YOLO11Loss

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