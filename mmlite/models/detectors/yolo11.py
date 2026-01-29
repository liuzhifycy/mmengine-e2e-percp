from mmdet.registry import MODELS
from mmengine.model import BaseModel
from ..backbones.yolo11_csp_darknet import YOLO11CSPDarknet
from ..necks.yolo11_pafpn import YOLO11PAFPN
from ..dense_heads.yolo11_head import YOLO11Head

@MODELS.register_module()
class YOLO11(BaseModel):
    def __init__(self, 
                 backbone, 
                 neck, 
                 bbox_head, 
                 data_preprocessor=None, 
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            x = self.extract_feat(inputs)
            return self.bbox_head(x)

    def loss(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        return self.bbox_head.loss(x, data_samples)

    def predict(self, inputs, data_samples, rescale=True):
        # TODO: Implement full post-processing (NMS)
        # For now, return raw outputs wrapper
        x = self.extract_feat(inputs)
        preds = self.bbox_head(x)
        return preds # Placeholder
