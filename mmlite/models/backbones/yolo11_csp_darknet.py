import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ..layers.yolo11_layers import YOLOConv, C3k2, SPPF, C2PSA

@MODELS.register_module()
class YOLO11CSPDarknet(BaseModule):
    """
    YOLO11 Backbone (CSPDarknet based).
    Hardcoded for 'm' scale based on official yolo11m.pt analysis.
    Scale 'm': d=0.5, w=1.0, max_channels=512
    """
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        
        # P1/2
        self.stem = nn.Sequential(
            YOLOConv(3, 64, kernel_size=3, stride=2),   # 0-P1/2
            YOLOConv(64, 128, kernel_size=3, stride=2)  # 1-P2/4
        )
        
        # P3/8
        self.stage1 = nn.Sequential(
            C3k2(128, 256, n=1, c3k=True, e=0.25),      # 2
            YOLOConv(256, 256, kernel_size=3, stride=2) # 3-P3/8
        )
        
        # P4/16
        self.stage2 = nn.Sequential(
            C3k2(256, 512, n=1, c3k=True, e=0.25),      # 4
            YOLOConv(512, 512, kernel_size=3, stride=2) # 5-P4/16
        )
        
        # P5/32 (Part 1)
        self.stage3 = nn.Sequential(
            C3k2(512, 512, n=1, c3k=True),              # 6
            YOLOConv(512, 512, kernel_size=3, stride=2) # 7-P5/32 (Note: 1024 clamped to 512)
        )
        
        # P5/32 (Part 2 - Deep & SPP)
        self.stage4 = nn.Sequential(
            C3k2(512, 512, n=1, c3k=True),              # 8 (Note: 1024 clamped to 512)
            SPPF(512, 512, k=5),                        # 9
            C2PSA(512, 512, n=1)                        # 10
        )

    def forward(self, x):
        """
        Returns: (P3, P4, P5) features.
        """
        x = self.stem(x)         # L0-1
        
        x = self.stage1[0](x)
        x = self.stage1[1](x)
        
        p3 = self.stage2[0](x)   # L4: C3k2 -> Output P3
        x = self.stage2[1](p3)
        
        p4 = self.stage3[0](x)   # L6: C3k2 -> Output P4
        x = self.stage3[1](p4)
        
        x = self.stage4[0](x)
        x = self.stage4[1](x)
        p5 = self.stage4[2](x)   # L10: C2PSA -> Output P5
        
        return (p3, p4, p5)