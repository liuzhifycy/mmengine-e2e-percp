"""
测试工具脚本
"""
import os
import subprocess
import pytest


class TestTrainScript:
    """训练脚本测试"""
    
    def test_train_script_help(self):
        """测试训练脚本 --help"""
        result = subprocess.run(
            ['python', 'tools/train.py', '--help'],
            capture_output=True,
            text=True,
            cwd='/home/ubuntu/e2e-pecp-pdp/mmengine-lite',
        )
        assert result.returncode == 0
        assert 'config' in result.stdout.lower()
        
    def test_test_script_help(self):
        """测试测试脚本 --help"""
        result = subprocess.run(
            ['python', 'tools/test.py', '--help'],
            capture_output=True,
            text=True,
            cwd='/home/ubuntu/e2e-pecp-pdp/mmengine-lite',
        )
        assert result.returncode == 0
        
    def test_export_script_help(self):
        """测试导出脚本 --help"""
        result = subprocess.run(
            ['python', 'tools/export_onnx.py', '--help'],
            capture_output=True,
            text=True,
            cwd='/home/ubuntu/e2e-pecp-pdp/mmengine-lite',
        )
        assert result.returncode == 0


class TestDistScripts:
    """分布式脚本测试"""
    
    def test_dist_train_script_exists(self):
        """测试分布式训练脚本存在"""
        script_path = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite/scripts/dist_train.sh'
        assert os.path.exists(script_path)
        assert os.access(script_path, os.X_OK)
        
    def test_dist_test_script_exists(self):
        """测试分布式测试脚本存在"""
        script_path = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite/scripts/dist_test.sh'
        assert os.path.exists(script_path)
        assert os.access(script_path, os.X_OK)
