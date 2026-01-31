# License Plate Recognition System

基于 YOLO11 + HyperLPR3 的中国车牌识别系统。

## 系统架构

```
输入图像 → YOLO11 (车牌检测) → 裁剪车牌区域 → HyperLPR3 (字符识别) → 车牌号
```

## 性能指标

### YOLO11 检测模型 (我们训练的)

| 指标 | 值 |
|------|-----|
| mAP@0.5 | **99.0%** |
| mAP@0.5:0.95 | **86.0%** |
| mAP@0.75 | **98.9%** |
| 检测率 | **100%** |

### 端到端识别性能

| 方案 | 检测率 | 精确识别率 | 部分匹配率 (>=5字符) |
|------|--------|------------|---------------------|
| HyperLPR3 (内置检测+识别) | 100% | 97.80% | 99.80% |
| **YOLO11 (我们训练) + HyperLPR3** | **100%** | **93.00%** | **95.80%** |

> **精确识别率**: 预测车牌号与真实车牌号完全一致的比例

## 目录结构

```
mmengine-lite/
├── configs/plate_recognition/
│   └── yolo11m_plate_detection.py          # YOLO11 训练配置
├── tools/plate_recognition/
│   ├── README.md                           # 本文档
│   ├── prepare_combined_data.py            # 数据预处理脚本
│   ├── adapt_yolo11_weights.py             # 权重适配脚本
│   └── inference_pipeline.py               # 端到端推理脚本
├── checkpoints/
│   └── yolo11m_plate_pretrain.pth          # YOLO11 预训练权重 (适配后)
├── work_dirs/yolo11m_plate_detection/
│   ├── best_coco_plate_precision_epoch_15.pth  # 最佳检测模型 (推荐使用)
│   ├── epoch_20.pth                        # Epoch 20 检测模型
│   └── ...
└── data/ccpd/
    ├── CCPD2019/                           # 原始数据集
    ├── CCPD2020/                           # 原始数据集
    └── combined/                           # 合并后的数据集 (COCO格式)
        ├── train.json                      # 训练集标注 (18000张)
        ├── val.json                        # 验证集标注 (1001张)
        ├── test.json                       # 测试集标注 (2000张)
        └── train/val/test/images/          # 图片目录
```

## 预训练权重位置

| 权重文件 | 路径 | 说明 |
|---------|------|------|
| YOLO11 预训练 (适配后) | `checkpoints/yolo11m_plate_pretrain.pth` | 从 COCO 预训练权重适配，排除了分类头 |
| **最佳检测模型** | `work_dirs/yolo11m_plate_detection/best_coco_plate_precision_epoch_15.pth` | **推荐使用**，mAP@0.5=99.0% |
| Epoch 20 模型 | `work_dirs/yolo11m_plate_detection/epoch_20.pth` | 最后保存的检查点 |

## 环境配置

```bash
# 激活 conda 环境
source /home/ubuntu/mambaforge/etc/profile.d/conda.sh
conda activate mmlite

# 依赖
# - PyTorch 2.1.0 + CUDA 12.1
# - mmengine 0.10.7, mmdet 3.3.0, mmcv 2.1.0
# - hyperlpr3
```

## 使用方法

### 1. 单张图片推理

```bash
cd /home/ubuntu/e2e-pecp-pdp/mmengine-lite

# 使用 HyperLPR3 内置检测 + 识别 (最高精度)
python tools/plate_recognition/inference_pipeline.py \
    --use_hyperlpr_detection \
    --image path/to/image.jpg

# 使用我们训练的 YOLO11 检测 + HyperLPR3 识别
python tools/plate_recognition/inference_pipeline.py \
    --image path/to/image.jpg
```

### 2. 批量处理

```bash
python tools/plate_recognition/inference_pipeline.py \
    --image_dir path/to/images/ \
    --output_dir path/to/output/
```

### 3. 评估模式

```bash
# 评估 YOLO11 + HyperLPR3 组合
python tools/plate_recognition/inference_pipeline.py \
    --evaluate \
    --test_dir data/ccpd/CCPD2019/ccpd_base/ \
    --max_samples 500
```

### 4. 训练 YOLO11 检测模型

```bash
# 数据预处理
python tools/plate_recognition/prepare_combined_data.py

# 训练
python tools/train.py configs/plate_recognition/yolo11m_plate_detection.py

# 测试
python tools/test.py configs/plate_recognition/yolo11m_plate_detection.py \
    work_dirs/yolo11m_plate_detection/best_coco_plate_precision_epoch_15.pth
```

## 输出格式

```json
{
  "image_path": "path/to/image.jpg",
  "image_size": [720, 1160],
  "plates": [
    {
      "bbox": [254, 462, 499, 543],
      "detection_confidence": 0.995,
      "plate_number": "皖AD18887",
      "recognition_confidence": 0.995
    }
  ]
}
```

## 数据集

使用 CCPD (Chinese City Parking Dataset):
- **CCPD2019**: 基础数据集
- **CCPD2020**: 绿牌数据集

合并后数据分布:
- 训练集: 18,000 张 (CCPD2019: 10,000 + CCPD2020: 8,000)
- 验证集: 1,001 张
- 测试集: 2,000 张

## 错误分析

错误样本保存在 `work_dirs/plate_errors/`:
- `recognition_wrong_*.jpg`: 识别结果与真实值不一致
- `ocr_failed_*.jpg`: OCR 未能识别出结果

每张图片标注:
- 绿色框: YOLO11 检测到的车牌位置
- GT (绿色): 真实车牌号
- Pred (红色): 预测车牌号
