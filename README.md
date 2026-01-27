# MMEngine-Lite

一个轻量级的 MMEngine 训练框架模板，从 e2e-pecp-pdp 项目抽离而来，独立可迁移，支持目标检测模型的完整训练流程。

## 特性

- **轻量独立**: 不依赖特定业务代码，可直接迁移到其他项目
- **功能完整**: 支持训练、推理、评估、可视化、ONNX导出
- **灵活配置**: 基于 mmengine 配置系统，支持配置继承和覆盖
- **分布式支持**: 支持单GPU和多GPU分布式训练
- **断点续训**: 支持从 checkpoint 恢复训练

## 环境要求

- Python >= 3.8
- PyTorch 2.1.0 (推荐，兼容预编译 mmcv)
- CUDA 12.1 (推荐) 或 CUDA 11.8

## 安装

### 1. 创建 Conda 环境

```bash
conda create -n mmlite python=3.10 -y
conda activate mmlite
```

### 2. 安装 PyTorch 2.1.0

```bash
# CUDA 12.1 (推荐)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装预编译的 mmcv

> **重要**: mmcv 需要使用预编译版本，否则 CUDA 扩展不可用。

```bash
# CUDA 12.1 + PyTorch 2.1.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# 或 CUDA 11.8 + PyTorch 2.1.0
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

### 4. 安装其他依赖

```bash
# NumPy (需要 1.x 版本)
pip install "numpy<2"

# mmengine 和 mmdet
pip install mmengine mmdet

# COCO 评估工具
pip install pycocotools

# ONNX 导出（可选）
pip install onnx onnxruntime onnxsim
```

### 5. 安装本项目

```bash
cd mmengine-lite
pip install -e .
```

### 快速安装脚本

```bash
# 完整安装脚本
conda create -n mmlite python=3.10 -y
conda activate mmlite
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "numpy<2" mmengine mmdet pycocotools
pip install onnx onnxruntime onnxsim  # 可选
pip install -e .
```

## 验证安装

```bash
python -c "
import torch
import mmcv
import mmengine
import mmdet
print(f'PyTorch: {torch.__version__}')
print(f'mmcv: {mmcv.__version__}')
print(f'mmengine: {mmengine.__version__}')
print(f'mmdet: {mmdet.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
# 测试 mmcv CUDA ops
from mmcv.ops import nms
print('mmcv CUDA ops: OK')
"
```

## 项目结构

```
mmengine-lite/
├── configs/                    # 配置文件
│   ├── _base_/                # 基础配置
│   │   ├── models/            # 模型配置
│   │   ├── datasets/          # 数据集配置
│   │   ├── schedules/         # 训练策略配置
│   │   └── default_runtime.py # 默认运行时配置
│   ├── retinanet/             # RetinaNet 完整配置
│   └── custom/                # 自定义模型配置示例
├── mmlite/                    # 核心包
│   ├── datasets/              # 数据集模块
│   ├── models/                # 模型模块
│   │   └── custom/            # 自定义 Backbone/Head 示例
│   ├── engine/                # 训练引擎
│   └── evaluation/            # 评估指标
├── tools/                     # 工具脚本
│   ├── train.py               # 训练入口
│   ├── test.py                # 测试入口
│   ├── visualize.py           # 可视化工具
│   └── export_onnx.py         # ONNX 导出
├── deploy/                    # 部署工具
│   ├── onnx2trt.py            # ONNX 转 TensorRT
│   └── trt_infer.py           # TensorRT 推理
├── tests/                     # 单元测试
├── scripts/                   # Shell 脚本
│   ├── dist_train.sh          # 分布式训练
│   └── dist_test.sh           # 分布式测试
├── requirements.txt
├── setup.py
└── README.md
```

## 数据准备

### COCO 数据集

下载 COCO 2017 数据集并按如下结构组织：

```
data/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── instances_test2017.json
    ├── train2017/
    ├── val2017/
    └── test2017/
```

也可以使用软链接：

```bash
mkdir -p data
ln -s /path/to/coco data/coco
```

### 下载预训练权重

```bash
# 使用 mim 下载 (推荐)
pip install openmim
mim download mmdet --config retinanet_r50_fpn_1x_coco --dest ./checkpoints

# 或手动下载
# https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
```

## 快速开始

### 训练

```bash
# 单GPU训练
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py

# 指定工作目录
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --work-dir work_dirs/my_exp

# 混合精度训练
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --amp

# 断点续训
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --resume
# 或指定 checkpoint
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --resume work_dirs/latest.pth
```

### 分布式训练

```bash
# 8卡训练
bash scripts/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8

# 带额外参数
bash scripts/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8 --amp

# 自定义端口 (避免冲突)
PORT=29501 bash scripts/dist_train.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py 8
```

### 测试/评估

```bash
# 单GPU测试
python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco.py work_dirs/epoch_12.pth

# 保存可视化结果
python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco.py work_dirs/epoch_12.pth --show-dir vis_results/

# 分布式测试
bash scripts/dist_test.sh configs/retinanet/retinanet_r50_fpn_1x_coco.py work_dirs/epoch_12.pth 8
```

### 可视化

```bash
# 单张图片推理
python tools/visualize.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint work_dirs/epoch_12.pth \
    --input demo.jpg \
    --output vis_output/

# 批量处理目录
python tools/visualize.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint work_dirs/epoch_12.pth \
    --input images_dir/ \
    --output vis_output/ \
    --score-thr 0.5
```

### ONNX 导出

```bash
# 基本导出
python tools/export_onnx.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint work_dirs/epoch_12.pth \
    --output-file exports/model.onnx

# 简化模型
python tools/export_onnx.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint work_dirs/epoch_12.pth \
    --output-file exports/model.onnx \
    --simplify

# 指定输入形状
python tools/export_onnx.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint work_dirs/epoch_12.pth \
    --output-file exports/model.onnx \
    --input-shape 1,3,640,640
```

## 配置说明

### 基础配置继承

```python
# configs/retinanet/retinanet_r50_fpn_1x_coco.py
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',      # 模型配置
    '../_base_/datasets/coco_detection.py',       # 数据集配置
    '../_base_/schedules/schedule_1x.py',         # 训练策略
    '../_base_/default_runtime.py',               # 运行时配置
]
```

### 配置覆盖

```python
# 覆盖学习率
optim_wrapper = dict(optimizer=dict(lr=0.02))

# 覆盖训练轮数
train_cfg = dict(max_epochs=24)

# 覆盖数据路径
train_dataloader = dict(dataset=dict(data_root='/new/data/path/'))
```

### 命令行配置覆盖

```bash
python tools/train.py configs/xxx.py \
    --cfg-options train_dataloader.batch_size=4 \
    --cfg-options optim_wrapper.optimizer.lr=0.01
```

## 扩展指南

### 添加新模型

1. 在 `mmlite/models/` 中添加模型文件
2. 在对应的 `__init__.py` 中注册导出
3. 创建配置文件

### 自定义模型示例

项目包含自定义 Backbone 和 Head 的完整示例：

```python
# mmlite/models/custom/ 包含:
# - SimpleCNNBackbone: 简单 CNN backbone (12.6M 参数)
# - MobileNetLiteBackbone: 轻量级 backbone (2.2M 参数)
# - SimpleDetectionHead: 简单检测头
# - LightweightHead: 轻量级检测头 (深度可分离卷积)
```

使用自定义模型配置：

```bash
# SimpleCNN + SimpleDetectionHead
python tools/train.py configs/custom/simplecnn_retinanet_1x_coco.py

# MobileNetLite + LightweightHead (轻量级)
python tools/train.py configs/custom/mobilenet_lite_1x_coco.py
```

### 添加新数据集

1. 在 `mmlite/datasets/` 中添加数据集类
2. 使用 `@DATASETS.register_module()` 装饰器注册
3. 创建数据集配置文件

### 添加自定义 Hook

1. 在 `mmlite/engine/hooks/` 中添加 Hook 类
2. 使用 `@HOOKS.register_module()` 装饰器注册
3. 在配置文件的 `custom_hooks` 中添加

## TensorRT 部署

### ONNX 转 TensorRT

```bash
# FP32 精度
python deploy/onnx2trt.py exports/model.onnx -o exports/model_fp32.engine

# FP16 精度 (推荐)
python deploy/onnx2trt.py exports/model.onnx -o exports/model_fp16.engine --fp16

# INT8 量化 (需要校准数据)
python deploy/onnx2trt.py exports/model.onnx -o exports/model_int8.engine --int8 \
    --calib-data data/coco/val2017 --calib-cache exports/calib.cache
```

### TensorRT 推理

```bash
# 性能测试
python deploy/trt_infer.py exports/model_fp16.engine --benchmark

# 图片推理
python deploy/trt_infer.py exports/model_fp16.engine -i image.jpg -o output.jpg
```

## 单元测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_models.py -v
```

## 常见问题

### Q: 安装时报错 "No module named 'mmcv._ext'"

A: mmcv 没有正确安装 CUDA 扩展。请使用预编译版本：
```bash
pip uninstall mmcv -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

### Q: NumPy 版本冲突

A: PyTorch 2.1 需要 NumPy 1.x：
```bash
pip install "numpy<2"
```

### Q: 如何使用预训练权重？

A: 在模型配置中指定 `init_cfg`:

```python
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    )
)
```

### Q: 如何调整 batch size 和学习率？

A: 学习率通常与 batch size 成线性关系。使用 `--auto-scale-lr` 可自动调整:

```bash
python tools/train.py configs/xxx.py --auto-scale-lr
```

### Q: 如何查看训练日志？

A: 日志保存在 `work_dirs/` 目录下，也可使用 TensorBoard:

```bash
tensorboard --logdir work_dirs/
```

## 版本兼容性

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| Python | 3.10 | 3.8+ 均可 |
| PyTorch | 2.1.0 | 需要与预编译 mmcv 匹配 |
| CUDA | 12.1 | 或 11.8 |
| mmcv | 2.1.0 | 使用预编译版本 |
| mmengine | 0.10.7 | 最新稳定版 |
| mmdet | 3.3.0 | 3.0-3.3 均可 |
| NumPy | 1.26.4 | 必须 < 2.0 |

## 许可证

Apache License 2.0
