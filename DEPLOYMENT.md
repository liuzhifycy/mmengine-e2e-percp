# 部署指南 (Deployment Guide)

本文档详细介绍了车牌识别系统 (E2E_HZTK) 在不同平台上的部署流程。

## 1. 概览 (Overview)

| 平台 | 脚本路径 | 核心功能 | 适用场景 |
|----------|--------|--------------|-------------|
| **NVIDIA GPU** | `tools/plate_recognition/deploy_e2e_hztk_trt.py` | TensorRT FP16 加速, 异步推理 | 高性能服务器/工控机 |
| **Rockchip NPU** | `e2e_hztk_deploy_package/deploy_rknn_board.py` | RKNN NPU 加速, RTSP 视频流支持, 3核调度优化 | 边缘计算/RK3588 开发板 |
| **开发/评估** | `tools/plate_recognition/inference_pipeline.py` | YOLO11 + HyperLPR3 混合流水线, CCPD 数据集评估 | 算法验证/基线对比 |

---

## 2. NVIDIA TensorRT 部署

适用于 NVIDIA GPU (如 RTX 4090, Jetson Orin) 的高性能推理方案。

### 2.1 环境要求
*   **系统**: Linux (Ubuntu 20.04+)
*   **硬件**: NVIDIA GPU (Compute Capability 6.0+)
*   **软件**: CUDA 11.8/12.x, TensorRT 8.x/10.x
*   **Python 依赖**:
    ```bash
    pip install tensorrt pycuda opencv-python numpy
    ```

### 2.2 模型准备
使用导出工具生成 ONNX 模型并构建 TensorRT 引擎：
```bash
python tools/plate_recognition/export_e2e_hztk.py --output-dir exports/e2e_hztk --trt --fp16
```
*   **生成文件**: `y5fu_640x_fp16.engine` (检测), `rpv3_mdict_160_fp16.engine` (识别), `litemodel_cls_96x_fp16.engine` (分类)

### 2.3 使用方法
**单图推理:**
```bash
python tools/plate_recognition/deploy_e2e_hztk_trt.py \
    --engine-dir exports/e2e_hztk \
    --image test_pic/car.jpg \
    --output-dir vis_output/
```

**性能基准测试:**
```bash
python tools/plate_recognition/deploy_e2e_hztk_trt.py --engine-dir exports/e2e_hztk --benchmark
```

---

## 3. Rockchip RK3588 部署 (RKNN)

适用于瑞芯微 RK3588 平台的边缘侧 NPU 部署方案。

### 3.1 环境要求
*   **硬件**: RK3588 / RK3588S 开发板
*   **系统**: Linux (Debian/Ubuntu/Buildroot)
*   **Python 依赖** (板端):
    ```bash
    # 安装 rknn-toolkit-lite2 (通常由板卡厂商 SDK 提供)
    pip install rknn_toolkit_lite2_*.whl
    pip install opencv-python numpy
    ```

### 3.2 模型准备 (PC 端)
1.  **导出 ONNX**:
    ```bash
    python tools/plate_recognition/export_e2e_hztk.py --output-dir exports/e2e_hztk
    ```
2.  **转换为 RKNN**:
    ```bash
    cd e2e_hztk_deploy_package
    # 确保已有导出的 ONNX 文件，或将其复制到当前目录
    python rknn_convert.py
    ```
    *   **生成文件**: `hztk_det.rknn`, `hztk_rec.rknn`, `hztk_cls.rknn`

### 3.3 使用方法 (板端)
将 `e2e_hztk_deploy_package/` 文件夹完整传输到开发板。

**图片推理:**
```bash
python deploy_rknn_board.py --image test.jpg
```

**RTSP 视频流:**
```bash
python deploy_rknn_board.py --rtsp "rtsp://user:pass@ip:554/stream" --no-display
```
*注: 脚本针对 RK3588 的 3核心 NPU 进行了专门的调度优化，以实现最佳性能。*

---

## 4. 开发与评估流水线 (PyTorch/ONNX)

用于算法验证、数据集评估及基线对比。采用训练好的 YOLO11 检测器配合 HyperLPR3 识别器。

### 4.1 环境要求
*   **Python 依赖**:
    ```bash
    pip install "mmdet>=3.0.0" "mmengine>=0.8.0" hyperlpr3
    ```

### 4.2 使用方法
**单图测试 (混合模式):**
```bash
# 使用训练好的 YOLO11 检测器 + HyperLPR3 识别器
python tools/plate_recognition/inference_pipeline.py \
    --image test_pic/car.jpg \
    --yolo_config configs/plate_recognition/yolo11m_plate_detection.py \
    --yolo_checkpoint work_dirs/yolo11m_plate_detection/best.pth
```

**数据集评估 (CCPD):**
```bash
python tools/plate_recognition/inference_pipeline.py \
    --evaluate \
    --test_dir data/ccpd/test \
    --save_vis \
    --output_dir work_dirs/eval_results
```

---

## 5. 部署工件目录结构

```
e2e-pecp-pdp/
├── deploy/                 # 通用 ONNX/TRT 工具
├── exports/                # 生成的模型文件 (ONNX/Engine)
├── tools/
│   └── plate_recognition/
│       ├── deploy_e2e_hztk_trt.py  # TensorRT 推理脚本
│       ├── export_e2e_hztk.py      # 模型导出工具
│       └── inference_pipeline.py   # PyTorch/评估脚本
└── e2e_hztk_deploy_package/        # RKNN 部署包
    ├── deploy_rknn_board.py        # RKNN 推理脚本
    └── rknn_convert.py             # RKNN 转换脚本
```