#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# MMEngine-Lite 分布式测试启动脚本
#
# 功能说明:
#     使用 torchrun (PyTorch 2.0+) 启动分布式测试
#
# 使用方法:
#     # 基本用法: 指定配置文件、权重文件和 GPU 数量
#     bash scripts/dist_test.sh configs/xxx.py checkpoint.pth 8
#
#     # 指定工作目录保存结果
#     bash scripts/dist_test.sh configs/xxx.py checkpoint.pth 8 --work-dir ./results
#
#     # 保存可视化结果
#     bash scripts/dist_test.sh configs/xxx.py checkpoint.pth 8 --show-dir ./vis
#
#     # 自定义端口（避免端口冲突）
#     PORT=29501 bash scripts/dist_test.sh configs/xxx.py checkpoint.pth 8
#
# 参数说明:
#     $1: CONFIG     - 配置文件路径（必需）
#     $2: CHECKPOINT - 模型权重文件路径（必需）
#     $3: GPUS       - GPU 数量（必需）
#     ${@:4}         - 其他参数，传递给 test.py
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
CHECKPOINT=$2
GPUS=$3

# 参数检查
if [ -z "$CONFIG" ]; then
    echo "错误: 请指定配置文件路径"
    echo "用法: bash scripts/dist_test.sh CONFIG CHECKPOINT GPUS [其他参数...]"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "错误: 请指定模型权重文件路径"
    echo "用法: bash scripts/dist_test.sh CONFIG CHECKPOINT GPUS [其他参数...]"
    exit 1
fi

if [ -z "$GPUS" ]; then
    echo "错误: 请指定 GPU 数量"
    echo "用法: bash scripts/dist_test.sh CONFIG CHECKPOINT GPUS [其他参数...]"
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

# 打印测试信息
echo "============================================"
echo "MMEngine-Lite 分布式测试"
echo "============================================"
echo "配置文件: $CONFIG"
echo "权重文件: $CHECKPOINT"
echo "GPU 数量: $GPUS"
echo "节点数量: $NNODES"
echo "当前节点: $NODE_RANK"
echo "主节点地址: $MASTER_ADDR:$PORT"
echo "其他参数: ${@:4}"
echo "============================================"

# 使用 torchrun 启动分布式测试
python3 -m torch.distributed.run \
    --nproc_per_node=$GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    "${ROOT_DIR}/tools/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    ${@:4}
