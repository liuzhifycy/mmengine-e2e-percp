"""
测试 ONNX 导出功能
"""
import os
import pytest
import torch


@pytest.fixture
def device():
    """返回可用设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def export_dir(tmp_path):
    """返回临时导出目录"""
    return str(tmp_path)


class TestONNXExport:
    """ONNX 导出测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_export_retinanet_onnx(self, device, export_dir):
        """测试 RetinaNet ONNX 导出"""
        from mmengine.config import Config
        from mmengine.registry import MODELS, DefaultScope
        
        cfg = Config.fromfile('configs/retinanet/retinanet_r50_fpn_1x_coco.py')
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            model = MODELS.build(cfg.model)
        
        model = model.to(device)
        model.eval()
        
        # 创建示例输入
        x = torch.randn(1, 3, 640, 640).to(device)
        
        # 导出 ONNX
        onnx_path = os.path.join(export_dir, 'retinanet_test.onnx')
        
        # 使用 backbone + neck 部分进行简单测试
        class BackboneNeck(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.backbone = model.backbone
                self.neck = model.neck
                
            def forward(self, x):
                feats = self.backbone(x)
                return self.neck(feats)
        
        export_model = BackboneNeck(model)
        
        torch.onnx.export(
            export_model,
            x,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            do_constant_folding=True,
        )
        
        assert os.path.exists(onnx_path)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_onnx_inference(self, device, export_dir):
        """测试 ONNX 推理"""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("需要安装 onnxruntime")
            
        from mmengine.config import Config
        from mmengine.registry import MODELS, DefaultScope
        
        cfg = Config.fromfile('configs/retinanet/retinanet_r50_fpn_1x_coco.py')
        
        with DefaultScope.overwrite_default_scope('mmdet'):
            model = MODELS.build(cfg.model)
        
        model = model.to(device)
        model.eval()
        
        # 简单的 backbone 导出测试
        class SimpleBackbone(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                
            def forward(self, x):
                outs = self.backbone(x)
                return outs[-1]  # 只返回最后一个特征图
        
        export_model = SimpleBackbone(model.backbone)
        x = torch.randn(1, 3, 224, 224).to(device)
        
        onnx_path = os.path.join(export_dir, 'backbone_test.onnx')
        torch.onnx.export(
            export_model,
            x,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
        )
        
        # 使用 ONNXRuntime 推理
        session = ort.InferenceSession(onnx_path)
        x_np = x.cpu().numpy()
        ort_outputs = session.run(None, {'input': x_np})
        
        # 比较输出
        with torch.no_grad():
            torch_output = export_model(x).cpu().numpy()
            
        # 检查输出形状一致
        assert ort_outputs[0].shape == torch_output.shape
