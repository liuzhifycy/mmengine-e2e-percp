#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# MMEngine-Lite 分布式训练启动脚本
#
# 功能说明:
#     使用 torchrun (PyTorch 2.0+) 启动分布式训练
#
# 使用方法:
#     # 基本用法: 指定配置文件和 GPU 数量
#     bash scripts/dist_train.sh configs/xxx.py 8
#
#     # 指定工作目录
#     bash scripts/dist_train.sh configs/xxx.py 8 --work-dir ./work_dirs/exp1
#
#     # 启用混合精度训练
#     bash scripts/dist_train.sh configs/xxx.py 8 --amp
#
#     # 断点续训（自动从最新检查点恢复）
#     bash scripts/dist_train.sh configs/xxx.py 8 --resume
#
#     # 断点续训（指定检查点路径）
#     bash scripts/dist_train.sh configs/xxx.py 8 --resume /path/to/checkpoint.pth
#
#     # 自定义端口（避免端口冲突）
#     PORT=29501 bash scripts/dist_train.sh configs/xxx.py 8
#
# 参数说明:
#     $1: CONFIG  - 配置文件路径（必需）
#     $2: GPUS    - GPU 数量（必需）
#     ${@:3}      - 其他参数，传递给 train.py
#
# 环境变量:
#     PORT        - 主节点端口，默认 29500
#     NNODES      - 节点数量，默认 1
#     NODE_RANK   - 当前节点编号，默认 0
#     MASTER_ADDR - 主节点地址，默认 127.0.0.1
# ============================================================================

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 解析参数
CONFIG=$1
GPUS=$2

# 参数检查
if [ -z "$CONFIG" ]; then
    echo "错误: 请指定配置文件路径"
    echo "用法: bash scripts/dist_train.sh CONFIG GPUS [其他参数...]"
    exit 1
fi

if [ -z "$GPUS" ]; then
    echo "错误: 请指定 GPU 数量"
    echo "用法: bash scripts/dist_train.sh CONFIG GPUS [其他参数...]"
    exit 1
fi

# 设置默认环境变量
PORT=${PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 设置 PYTHONPATH
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# 设置 PyTorch 内存配置（推荐）
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# 打印训练信息
echo "============================================"
echo "MMEngine-Lite 分布式训练"
echo "============================================"
echo "配置文件: $CONFIG"
echo "GPU 数量: $GPUS"
echo "节点数量: $NNODES"
echo "当前节点: $NODE_RANK"
echo "主节点地址: $MASTER_ADDR:$PORT"
echo "其他参数: ${@:3}"
echo "============================================"

# 使用 torchrun 启动分布式训练
python3 -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    "${ROOT_DIR}/tools/train.py" \
    "$CONFIG" \
    --launcher pytorch \
    ${@:3}
