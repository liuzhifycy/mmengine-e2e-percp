"""
测试配置模块
"""
import pytest


@pytest.fixture
def config_path():
    """返回测试配置文件路径"""
    return 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'


@pytest.fixture
def custom_config_path():
    """返回自定义模型配置文件路径"""
    return 'configs/custom/simplecnn_retinanet_1x_coco.py'


class TestConfigLoading:
    """配置文件加载测试"""
    
    def test_load_retinanet_config(self, config_path):
        """测试 RetinaNet 配置文件加载"""
        from mmengine.config import Config
        
        cfg = Config.fromfile(config_path)
        
        # 检查基本配置项
        assert 'model' in cfg
        assert 'train_dataloader' in cfg
        assert 'val_dataloader' in cfg
        assert 'optim_wrapper' in cfg
        
    def test_config_model_structure(self, config_path):
        """测试配置中的模型结构"""
        from mmengine.config import Config
        
        cfg = Config.fromfile(config_path)
        model_cfg = cfg.model
        
        # 检查模型配置
        assert model_cfg.type == 'RetinaNet'
        assert 'backbone' in model_cfg
        assert 'neck' in model_cfg
        assert 'bbox_head' in model_cfg
        
        # 检查 backbone 配置
        assert model_cfg.backbone.type == 'ResNet'
        assert model_cfg.backbone.depth == 50
        
    def test_config_dataset_structure(self, config_path):
        """测试数据集配置结构"""
        from mmengine.config import Config
        
        cfg = Config.fromfile(config_path)
        
        # 检查训练数据加载器
        assert cfg.train_dataloader.batch_size > 0
        assert 'dataset' in cfg.train_dataloader
        assert cfg.train_dataloader.dataset.type == 'CocoDataset'
        
    def test_load_custom_config(self, custom_config_path):
        """测试自定义模型配置文件加载"""
        from mmengine.config import Config
        
        cfg = Config.fromfile(custom_config_path)
        
        # 检查自定义模型配置
        assert cfg.model.backbone.type == 'SimpleCNNBackbone'
        assert cfg.model.bbox_head.type == 'SimpleDetectionHead'
        assert 'custom_imports' in cfg
