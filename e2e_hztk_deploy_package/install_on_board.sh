#!/bin/bash

# RK3588 板端 cp39 离线安装脚本
# 用法: 在 Python 3.9 环境下运行 bash install_on_board.sh

echo ">>> 1. 检查 Python 版本..."
PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "当前环境 Python 版本: $PY_VERSION"

if [ "$PY_VERSION" != "3.9" ]; then
    echo "❌ 错误: 当前 Python 版本是 $PY_VERSION，而您的 whl 包是 cp39 (适配 3.9)。"
    echo "请先通过 'conda activate' 切换到一个 Python 3.9 的环境！"
    exit 1
fi

echo ">>> 2. 安装离线依赖 (如果当前环境已安装则跳过)..."
# 注意：如果环境里没 numpy，且没网，这里会报错。
# 如果报错，请在 PC 端下载 numpy 的 aarch64 cp39 whl 包传上来安装。
pip install numpy==1.26.4 opencv-python --no-index --find-links . 2>/dev/null || echo "警告: 无法从本地获取 numpy，尝试普通安装..."
pip install numpy==1.26.4 opencv-python

echo ">>> 3. 安装 RKNN Toolkit Lite2 (cp39)..."
WHL_FILE=$(ls rknn_toolkit_lite2*cp39*.whl 2>/dev/null | head -n 1)

if [ -z "$WHL_FILE" ]; then
    echo "❌ 错误: 未找到标记为 cp39 的 .whl 安装包!"
    ls -l
    exit 1
fi

echo "找到包: $WHL_FILE"
pip install "$WHL_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 安装成功!"
else
    echo "❌ 安装失败! 请检查报错信息。"
    exit 1
fi

echo ">>> 4. 验证环境..."
python -c "from rknnlite.api import RKNNLite; print('RKNNLite imported successfully!')"

echo "========================================"
echo "使用以下命令运行推理:"
echo "python deploy_rknn_board.py --image test.jpg"
echo "========================================"
