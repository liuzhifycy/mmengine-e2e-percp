import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ..layers.yolo11_layers import YOLOConv, C3k2

@MODELS.register_module()
class YOLO11PAFPN(BaseModule):
    """
    YOLO11 Neck (PAFPN).
    Hardcoded for 'm' scale.
    Inputs: [P3, P4, P5]
    """
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        
        # Top-down pathway
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest') # 11
        self.c3k2_1 = C3k2(512 + 512, 512, n=1, c3k=True)      # 13 (In: P5_up + P4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') # 14
        self.c3k2_2 = C3k2(512 + 512, 256, n=1, c3k=True)      # 16 (In: P4_up + P3) -> Out 1 (Small)

        # Bottom-up pathway
        self.down1 = YOLOConv(256, 256, kernel_size=3, stride=2) # 17
        self.c3k2_3 = C3k2(256 + 512, 512, n=1, c3k=True)        # 19 (In: P3_down + P4_processed) -> Out 2 (Medium)
        
        self.down2 = YOLOConv(512, 512, kernel_size=3, stride=2) # 20
        self.c3k2_4 = C3k2(512 + 512, 512, n=1, c3k=True)        # 22 (In: P4_down + P5) -> Out 3 (Large)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): (P3, P4, P5) from backbone.
        """
        p3, p4, p5 = inputs
        
        # 11-13
        x = self.up1(p5)
        x = torch.cat([x, p4], dim=1) # Concat P5_up with P4
        f_p4 = self.c3k2_1(x)         # Layer 13 Output (Processed P4)
        
        # 14-16
        x = self.up2(f_p4)
        x = torch.cat([x, p3], dim=1) # Concat P4_up with P3
        f_p3 = self.c3k2_2(x)         # Layer 16 Output (Small Object Feature)
        
        # 17-19
        x = self.down1(f_p3)
        x = torch.cat([x, f_p4], dim=1) # Concat P3_down with Processed P4
        f_p4_new = self.c3k2_3(x)       # Layer 19 Output (Medium Object Feature)
        
        # 20-22
        x = self.down2(f_p4_new)
        x = torch.cat([x, p5], dim=1)   # Concat P4_down with Original P5
        f_p5_new = self.c3k2_4(x)       # Layer 22 Output (Large Object Feature)
        
        return (f_p3, f_p4_new, f_p5_new)