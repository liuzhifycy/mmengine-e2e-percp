"""
测试模型构建和前向传播
"""
import pytest
import torch


@pytest.fixture
def device():
    """返回可用设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class TestModelBuilding:
    """模型构建测试"""
    
    def test_build_retinanet(self, device):
        """测试 RetinaNet 模型构建"""
        from mmengine.config import Config
        from mmengine.registry import MODELS, DefaultScope
        
        cfg = Config.fromfile('configs/retinanet/retinanet_r50_fpn_1x_coco.py')
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            model = MODELS.build(cfg.model)
        
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'bbox_head')
        
    def test_retinanet_forward(self, device):
        """测试 RetinaNet 前向传播"""
        from mmengine.config import Config
        from mmengine.registry import MODELS, DefaultScope
        
        cfg = Config.fromfile('configs/retinanet/retinanet_r50_fpn_1x_coco.py')
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            model = MODELS.build(cfg.model)
        
        model = model.to(device)
        model.eval()
        
        # 测试前向传播
        x = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            # 测试 backbone
            feats = model.backbone(x)
            assert len(feats) == 4  # ResNet 输出 4 个阶段
            
            # 测试 neck
            neck_feats = model.neck(feats)
            assert len(neck_feats) == 5  # FPN 输出 5 个尺度
            
    def test_retinanet_parameter_count(self, device):
        """测试 RetinaNet 参数量"""
        from mmengine.config import Config
        from mmengine.registry import MODELS, DefaultScope
        
        cfg = Config.fromfile('configs/retinanet/retinanet_r50_fpn_1x_coco.py')
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            model = MODELS.build(cfg.model)
        
        params = sum(p.numel() for p in model.parameters())
        # RetinaNet R50 约 37M 参数
        assert 35_000_000 < params < 40_000_000


class TestCustomModels:
    """自定义模型测试"""
    
    def test_build_simplecnn_backbone(self, device):
        """测试 SimpleCNNBackbone 构建"""
        from mmlite.models.custom import SimpleCNNBackbone
        
        backbone = SimpleCNNBackbone(
            in_channels=3,
            base_channels=64,
            num_stages=4,
        ).to(device)
        
        x = torch.randn(1, 3, 224, 224).to(device)
        outs = backbone(x)
        
        assert len(outs) == 4
        # 检查输出通道
        assert outs[0].shape[1] == 64
        assert outs[1].shape[1] == 128
        assert outs[2].shape[1] == 256
        assert outs[3].shape[1] == 512
        
    def test_build_mobilenet_lite_backbone(self, device):
        """测试 MobileNetLiteBackbone 构建"""
        from mmlite.models.custom import MobileNetLiteBackbone
        
        backbone = MobileNetLiteBackbone(
            in_channels=3,
            width_mult=1.0,
            out_indices=(0, 1, 2, 3),
        ).to(device)
        
        x = torch.randn(1, 3, 640, 640).to(device)
        outs = backbone(x)
        
        assert len(outs) == 4
        # 检查 stride
        assert outs[0].shape[-1] == 160  # stride 4
        assert outs[1].shape[-1] == 80   # stride 8
        assert outs[2].shape[-1] == 40   # stride 16
        assert outs[3].shape[-1] == 20   # stride 32
        
    def test_build_simple_detection_head(self, device):
        """测试 SimpleDetectionHead 构建"""
        from mmengine.registry import MODELS, DefaultScope
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            from mmlite.models.custom import SimpleDetectionHead
            
            head = SimpleDetectionHead(
                num_classes=80,
                in_channels=256,
                feat_channels=256,
                stacked_convs=4,
            ).to(device)
        
        # 测试单尺度前向
        x = torch.randn(1, 256, 80, 80).to(device)
        cls_score, bbox_pred = head.forward_single(x)
        
        # 检查输出形状
        # 9 anchors * 80 classes
        assert cls_score.shape[1] == 9 * 80
        # 9 anchors * 4 coords
        assert bbox_pred.shape[1] == 9 * 4
        
    def test_build_lightweight_head(self, device):
        """测试 LightweightHead 构建"""
        from mmengine.registry import MODELS, DefaultScope
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            from mmlite.models.custom import LightweightHead
            
            head = LightweightHead(
                num_classes=80,
                in_channels=128,
                feat_channels=128,
                num_convs=2,
            ).to(device)
        
        x = torch.randn(1, 128, 80, 80).to(device)
        cls_score, bbox_pred = head.forward_single(x)
        
        assert cls_score.shape[1] == 9 * 80
        assert bbox_pred.shape[1] == 9 * 4


class TestBackboneFreeze:
    """Backbone 冻结测试"""
    
    def test_simplecnn_frozen_stages(self, device):
        """测试 SimpleCNNBackbone 冻结阶段"""
        from mmlite.models.custom import SimpleCNNBackbone
        
        backbone = SimpleCNNBackbone(
            in_channels=3,
            base_channels=64,
            num_stages=4,
            frozen_stages=0,  # 只冻结 stem 和 stage 0
        ).to(device)
        
        backbone.train()
        
        # 检查 stem 是否冻结
        for param in backbone.stem.parameters():
            assert not param.requires_grad
            
        # 检查 stage 0 是否冻结
        for param in backbone.stages[0].parameters():
            assert not param.requires_grad
            
        # 检查 stage 1 是否未冻结
        for param in backbone.stages[1].parameters():
            assert param.requires_grad
