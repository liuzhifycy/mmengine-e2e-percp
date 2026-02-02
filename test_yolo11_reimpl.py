
import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmlite.models import YOLO11
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

def test_inference_and_loss():
    print("Loading Config...")
    cfg = Config.fromfile("configs/yolo11/yolo11m_coco_reimpl.py")
    
    print("Building Model...")
    model = MODELS.build(cfg.model)
    model.init_weights()
    print("Weights loaded successfully.")
    
    # Input
    img = torch.randn(1, 3, 640, 640)
    
    # -------------------------------------------------
    # Test Forward Tensor
    # -------------------------------------------------
    print("Running Forward (Tensor Mode)...")
    preds = model(img, mode='tensor')
    print("Forward Tensor Output Shapes:")
    for i, (box, cls) in enumerate(zip(preds[0], preds[1])):
        print(f"Level {i}: Box {box.shape}, Cls {cls.shape}")

    # -------------------------------------------------
    # Test Loss
    # -------------------------------------------------
    print("\nRunning Forward (Loss Mode)...")
    
    # Create Mock GT
    gt_instances = InstanceData()
    gt_instances.bboxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=img.device) # xyxy
    gt_instances.labels = torch.tensor([0], device=img.device)
    
    data_sample = DetDataSample()
    data_sample.gt_instances = gt_instances
    data_sample.set_metainfo(dict(
        img_shape=(640, 640),
        ori_shape=(640, 640),
        scale_factor=(1.0, 1.0)
    ))
    
    batch_data_samples = [data_sample]
    
    losses = model(img, data_samples=batch_data_samples, mode='loss')
    print("Loss Outputs:")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
        
    print("\nTest Passed!")

if __name__ == "__main__":
    test_inference_and_loss()
