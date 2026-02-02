# RK3588 车牌识别部署指南 (E2E_HZTK)

本部署包专为 Rockchip RK3588/RK3588S 平台设计，提供高性能的端到端车牌识别方案。

## 📦 目录结构

```
e2e_hztk_deploy_package/
├── hztk_det.rknn         # 检测模型 (需转换)
├── hztk_rec.rknn         # 识别模型 (需转换)
├── hztk_cls.rknn         # 分类模型 (需转换)
├── rknn_convert.py       # 模型转换脚本 (PC端运行)
├── deploy_rknn_board.py  # 推理部署脚本 (板端运行)
└── install_on_board.sh   # 自动安装脚本 (可选)
```

---

## 🛠️ 第一步：环境准备

### 1. PC 端 (用于模型转换)
需要一台 x86 Linux 主机来将 ONNX 模型转换为 RKNN 格式。
*   安装 `rknn-toolkit2`:
    ```bash
    pip install rknn-toolkit2
    ```

### 2. 板端 (RK3588)
需要在开发板上安装推理引擎和依赖库。
*   **Python 版本**: 3.8 / 3.9 (推荐使用 Miniconda)
*   **安装依赖**:
    ```bash
    # 安装 RKNN Lite2 (请根据 python 版本选择对应的 whl 包)
    # 通常位于开发板 SDK 的 external/rknn-toolkit2/rknn_toolkit_lite2/packages/ 目录
    pip install rknn_toolkit_lite2_cp39_cp39_linux_aarch64.whl
    
    # 安装其他依赖
    pip install opencv-python numpy pillow
    ```
*   **字体支持 (可选)**:
    为了在结果图片上正确绘制中文车牌，建议安装中文字体：
    ```bash
    sudo apt-get install fonts-noto-cjk
    ```

---

## 🔄 第二步：模型转换 (在 PC 上执行)

1.  将导出的 ONNX 模型 (如 `y5fu_640x.onnx`, `rpv3_mdict_160.onnx`) 放入本目录。
2.  运行转换脚本：
    ```bash
    # 默认会自动寻找并转换相关 ONNX 模型
    python rknn_convert_v2.py
    ```
    *成功后会生成 `hztk_det.rknn`, `hztk_rec.rknn`, `hztk_cls.rknn`。*

---

## 📡 第三步：传输到开发板

使用 SCP 将整个部署包发送到开发板。

**示例信息:**
*   **IP:** 60.174.225.149
*   **Port:** 22165
*   **User:** root
*   **Password:** detection*1234

```bash
# 在 PC 上执行
scp -P 22165 -r e2e_hztk_deploy_package/ root@60.174.225.149:/root/
```

---

## 🚀 第四步：运行推理 (在板端执行)

SSH 登录到开发板并进入目录：
```bash
ssh -p 22165 root@60.174.225.149
cd /root/e2e_hztk_deploy_package
```

### 1. 单张图片推理
```bash
python deploy_rknn_board.py --image test_car.jpg --output-dir result/
```

### 2. 批量图片推理
```bash
python deploy_rknn_board.py --image-dir ./images --output-dir ./results
```

### 3. RTSP 视频流 (核心功能)
直接连接网络摄像头进行实时识别。

**基本用法 (显示窗口):**
```bash
python deploy_rknn_board.py --rtsp "rtsp://admin:password@192.168.1.64:554/stream1"
```

**后台运行 (无显示器模式):**
适合在无桌面环境的服务器上运行，只打印日志或保存结果。
```bash
python deploy_rknn_board.py \
    --rtsp "rtsp://..." \
    --no-display \
    --output-dir ./stream_results \
    --save-interval 5    # 每5帧保存一次截图
```

**保存策略:**
*   默认：不保存图片。
*   `--output-dir`: 开启保存功能。
*   默认仅保存**检测到车牌**的帧。
*   `--save-all`: 强制保存所有帧 (慎用，会占用大量空间)。

### 4. 性能基准测试
测试 NPU 推理耗时和 FPS。
```bash
python deploy_rknn_board.py --benchmark
```
*预期性能 (RK3588): 单帧全流程耗时约 10-15ms (60+ FPS)*

---

## ⚠️ 常见问题

1.  **ImportError: No module named 'rknnlite'**
    *   请确保已安装 `rknn_toolkit_lite2` 的 whl 包。
    
2.  **中文乱码 / 方框**
    *   脚本依赖 PIL 绘制中文。如果系统缺少中文字体，会回退到 OpenCV 绘制 (不支持中文)。
    *   解决方法：`sudo apt install fonts-noto-cjk` 或将字体文件放入 `/usr/share/fonts/`。

3.  **RTSP 连接失败**
    *   请先用 VLC 或 FFmpeg 确认 RTSP 地址是否可用。
    *   尝试添加 `--no-display` 排除显示驱动问题。