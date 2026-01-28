# =============================================================================
# MMEngine-Lite Docker Image
# =============================================================================
# 基于 NVIDIA CUDA 12.1 + PyTorch 2.1.0 的深度学习训练环境
# 支持 2D 目标检测 (RetinaNet) 和 3D 点云检测 (PointPillars)
#
# 构建命令:
#   docker build -t mmlite:latest -f docker/Dockerfile .
#   或者: podman build -t mmlite:latest -f docker/Dockerfile .
#
# 运行命令:
#   docker run --gpus all -it --rm -v $(pwd):/workspace mmlite:latest
# =============================================================================

# 使用 NVIDIA 官方 PyTorch 镜像作为基础
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# 固定 numpy 版本 (必须在安装其他包之前)
# =============================================================================
# PyTorch 2.1.0 和 OpenMMLab 需要 numpy < 2.0
RUN pip install --no-cache-dir "numpy>=1.20.0,<2.0.0"

# =============================================================================
# 安装 OpenMMLab 生态系统
# =============================================================================

# 安装预编译的 mmcv (使用官方预编译版本，避免编译)
# 使用 --no-deps 避免 mmcv 升级 numpy
RUN pip install --no-cache-dir --no-deps \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 安装 mmcv 的其他依赖 (排除 numpy)
RUN pip install --no-cache-dir addict yapf

# 安装 mmengine
RUN pip install --no-cache-dir "mmengine>=0.10.0,<0.11.0"

# 安装 mmdet
RUN pip install --no-cache-dir "mmdet>=3.0.0,<3.4.0"

# 安装 mmdet3d
RUN pip install --no-cache-dir "mmdet3d>=1.2.0,<1.5.0"

# =============================================================================
# 安装其他依赖
# =============================================================================

RUN pip install --no-cache-dir \
    "pycocotools>=2.0.0" \
    "matplotlib>=3.5.0,<4.0.0" \
    "opencv-python>=4.5.0" \
    "scipy>=1.7.0" \
    "tqdm>=4.60.0" \
    "pyyaml>=5.4.0" \
    "tensorboard>=2.10.0"

# ONNX 导出支持 (可选)
RUN pip install --no-cache-dir \
    "onnx>=1.12.0" \
    "onnxruntime>=1.12.0" \
    "onnxsim>=0.4.0"

# =============================================================================
# 确保 numpy 版本正确 (最后再次固定)
# =============================================================================
RUN pip install --no-cache-dir "numpy>=1.20.0,<2.0.0" && \
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# =============================================================================
# 复制项目代码
# =============================================================================

# 复制项目文件
COPY . /workspace/mmlite

# 设置 PYTHONPATH
ENV PYTHONPATH="/workspace/mmlite:${PYTHONPATH}"

# 设置默认工作目录
WORKDIR /workspace/mmlite

# =============================================================================
# 健康检查和入口点
# =============================================================================

# 验证安装
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')" && \
    python -c "import mmdet; print(f'MMDet: {mmdet.__version__}')" && \
    python -c "import mmdet3d; print(f'MMDet3D: {mmdet3d.__version__}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 默认命令
CMD ["/bin/bash"]
