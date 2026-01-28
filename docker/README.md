# MMEngine-Lite Docker 部署指南

本文档介绍如何使用 Docker 部署和运行 mmengine-lite 项目。

## 系统要求

- Docker 20.10+
- NVIDIA Docker Runtime (nvidia-docker2) - 用于 GPU 支持
- NVIDIA 驱动 >= 525.60.13 (支持 CUDA 12.1)
- 至少 16GB 内存
- 至少 50GB 磁盘空间

### 检查 NVIDIA Docker 支持

```bash
# 检查 nvidia-docker 是否安装
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

如果未安装，请参考 [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

## 快速开始

### 1. 构建镜像

```bash
cd /path/to/mmengine-lite
./docker/build.sh
```

构建选项：
```bash
# 不使用缓存重新构建
./docker/build.sh --no-cache

# 指定标签
./docker/build.sh --tag v1.0

# 构建并推送到仓库
./docker/build.sh --push --tag v1.0
```

### 2. 运行容器

```bash
# 交互式 shell（默认）
./docker/run.sh

# 运行训练
./docker/run.sh train configs/retinanet/retinanet_r50_fpn_1x_coco.py

# 后台运行 Jupyter
./docker/run.sh --detach jupyter

# 不使用 GPU
./docker/run.sh --no-gpu bash
```

## 目录结构

```
docker/
├── Dockerfile          # Docker 镜像定义
├── docker-compose.yml  # Docker Compose 配置
├── build.sh           # 构建脚本
├── run.sh             # 运行脚本
└── README.md          # 本文档
```

## 使用 Docker Compose

Docker Compose 提供了更方便的多容器管理：

```bash
cd /path/to/mmengine-lite

# 启动开发环境
docker-compose -f docker/docker-compose.yml up -d dev

# 查看日志
docker-compose -f docker/docker-compose.yml logs -f dev

# 进入容器
docker-compose -f docker/docker-compose.yml exec dev bash

# 启动 TensorBoard
docker-compose -f docker/docker-compose.yml up -d tensorboard

# 停止所有服务
docker-compose -f docker/docker-compose.yml down
```

## 数据卷挂载

默认挂载配置：

| 主机路径 | 容器路径 | 说明 |
|---------|---------|------|
| `./` | `/workspace` | 项目代码 |
| `./data` | `/workspace/data` | 数据集 |
| `./work_dirs` | `/workspace/work_dirs` | 训练输出 |
| `./checkpoints` | `/workspace/checkpoints` | 模型权重 |

自定义数据目录：
```bash
./docker/run.sh --data /path/to/your/data --work /path/to/outputs
```

## 端口映射

| 端口 | 服务 |
|------|------|
| 8888 | Jupyter Notebook |
| 6006 | TensorBoard |

## 常用命令

### 训练

```bash
# 2D 检测 - RetinaNet
./docker/run.sh train configs/retinanet/retinanet_r50_fpn_1x_coco.py

# 3D 检测 - PointPillars (纯 mmlite)
./docker/run.sh train configs/pointpillars/pointpillars_hv_secfpn_8xb6_kitti-3d-3class.py

# 3D 检测 - PointPillars (混合模式)
./docker/run.sh train configs/pointpillars_mmdet3d/pointpillars_hybrid_v2.py
```

### 测试

```bash
./docker/run.sh test configs/retinanet/retinanet_r50_fpn_1x_coco.py checkpoints/retinanet.pth
```

### 可视化

```bash
# 进入容器后运行
python tools/visualize.py \
    --config configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoint checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \
    --input data/coco/train2017/000000000009.jpg \
    --output vis_output/
```

## 迁移到其他机器

### 方法一：推送到 Docker Registry

```bash
# 在源机器上
./docker/build.sh --push --tag v1.0

# 在目标机器上
docker pull your-registry/mmengine-lite:v1.0
```

### 方法二：导出镜像文件

```bash
# 在源机器上 - 导出镜像
docker save mmengine-lite:latest | gzip > mmengine-lite.tar.gz

# 传输到目标机器
scp mmengine-lite.tar.gz user@target:/path/to/

# 在目标机器上 - 导入镜像
gunzip -c mmengine-lite.tar.gz | docker load
```

### 方法三：复制项目并重新构建

```bash
# 复制项目目录（不包含大文件）
rsync -avz --exclude='data' --exclude='work_dirs' --exclude='checkpoints' \
    mmengine-lite/ user@target:/path/to/mmengine-lite/

# 在目标机器上重新构建
cd /path/to/mmengine-lite
./docker/build.sh
```

## 故障排除

### GPU 不可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 如果失败，重新安装 nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 内存不足

增加共享内存大小（在 run.sh 或 docker-compose.yml 中）：
```bash
--shm-size=16g  # 或更大
```

### CUDA 内存碎片

设置 PyTorch 内存配置（已在容器中默认设置）：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 构建失败

```bash
# 清除 Docker 缓存后重试
docker builder prune
./docker/build.sh --no-cache
```

## 镜像内容

基础镜像：`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`

预装软件：
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1
- OpenMMLab 全家桶:
  - mmcv 2.1.0
  - mmengine 0.10.7
  - mmdet 3.3.0
  - mmdet3d 1.4.0
- 常用工具: git, vim, tmux, htop

## 自定义镜像

如需修改镜像，编辑 `docker/Dockerfile`：

```dockerfile
# 添加额外的 Python 包
RUN pip install your-package

# 添加系统依赖
RUN apt-get update && apt-get install -y your-package
```

然后重新构建：
```bash
./docker/build.sh --no-cache
```

## 参考链接

- [MMEngine 文档](https://mmengine.readthedocs.io/)
- [MMDetection 文档](https://mmdetection.readthedocs.io/)
- [MMDetection3D 文档](https://mmdetection3d.readthedocs.io/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
