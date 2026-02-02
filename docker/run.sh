#!/bin/bash
# run.sh - Run mmengine-lite container (supports Docker and Podman)
# Usage: ./run.sh [options] [command]

set -e

# 自动检测容器引擎 (优先使用 podman)
if command -v podman &> /dev/null; then
    CONTAINER_ENGINE="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_ENGINE="docker"
else
    echo "错误: 未找到 docker 或 podman"
    exit 1
fi

# Configuration
IMAGE_NAME="${IMAGE_NAME:-mmlite}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-mmlite-dev}"

# Directories (can be overridden)
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
WORK_DIR="${WORK_DIR:-$(pwd)/work_dirs}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$(pwd)/checkpoints}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Container engine: $CONTAINER_ENGINE"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --detach        Run container in background"
    echo "  -n, --name NAME     Container name (default: mmlite-dev)"
    echo "  -t, --tag TAG       Image tag (default: latest)"
    echo "  --no-gpu            Run without GPU support"
    echo "  --data DIR          Mount data directory (default: ./data)"
    echo "  --work DIR          Mount work_dirs directory (default: ./work_dirs)"
    echo "  --checkpoints DIR   Mount checkpoints directory (default: ./checkpoints)"
    echo ""
    echo "Commands:"
    echo "  train CONFIG        Run training with specified config"
    echo "  test CONFIG CKPT    Run testing with config and checkpoint"
    echo "  bash                Start interactive bash shell (default)"
    echo "  jupyter             Start Jupyter notebook server"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Interactive shell"
    echo "  $0 train configs/retinanet/retinanet_r50_fpn_1x_coco.py"
    echo "  $0 --detach jupyter                   # Jupyter in background"
    echo "  $0 --no-gpu bash                      # Without GPU"
}

# Parse arguments
DETACH=""
USE_GPU=true
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--detach)
            DETACH="-d"
            shift
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --work)
            WORK_DIR="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        train)
            shift
            COMMAND="python tools/train.py $*"
            break
            ;;
        test)
            shift
            COMMAND="python tools/test.py $*"
            break
            ;;
        bash)
            COMMAND="/bin/bash"
            shift
            ;;
        jupyter)
            COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
            shift
            ;;
        *)
            # Pass remaining args as command
            COMMAND="$*"
            break
            ;;
    esac
done

# Default command
if [ -z "$COMMAND" ]; then
    COMMAND="/bin/bash"
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

# Check if image exists
if ! $CONTAINER_ENGINE image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo -e "${RED}Error: Image ${IMAGE_NAME}:${IMAGE_TAG} not found.${NC}"
    echo -e "${YELLOW}Please build the image first: ./docker/build.sh${NC}"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$WORK_DIR" "$CHECKPOINTS_DIR"

# Check if container with same name is running
if $CONTAINER_ENGINE ps -q -f name="^${CONTAINER_NAME}$" 2>/dev/null | grep -q .; then
    echo -e "${YELLOW}Container ${CONTAINER_NAME} is already running.${NC}"
    echo "Executing command in existing container..."
    $CONTAINER_ENGINE exec -it "$CONTAINER_NAME" $COMMAND
    exit 0
fi

# Remove stopped container with same name
if $CONTAINER_ENGINE ps -aq -f name="^${CONTAINER_NAME}$" 2>/dev/null | grep -q .; then
    echo -e "${YELLOW}Removing stopped container ${CONTAINER_NAME}...${NC}"
    $CONTAINER_ENGINE rm "$CONTAINER_NAME"
fi

# 构建 GPU 参数
GPU_ARGS=""
NVIDIA_VOLUME_ARGS=""
if [ "$USE_GPU" = true ]; then
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        # Podman GPU 支持
        if [ -c "/dev/nvidia0" ]; then
            # 检查 Podman 版本是否支持 CDI (4.1.0+)
            PODMAN_VERSION=$(podman --version | grep -oP '\d+\.\d+' | head -1)
            PODMAN_MAJOR=$(echo "$PODMAN_VERSION" | cut -d. -f1)
            
            if [ "$PODMAN_MAJOR" -ge 4 ] && command -v nvidia-ctk &> /dev/null && [ -f /etc/cdi/nvidia.yaml ]; then
                # Podman 4.1+ 支持 CDI
                GPU_ARGS="--device nvidia.com/gpu=all"
            else
                # 旧版本 Podman (3.x) - GPU 支持有限
                echo -e "${YELLOW}警告: Podman ${PODMAN_VERSION} 不完全支持 GPU${NC}"
                echo -e "${YELLOW}建议升级到 Podman 4.1+ 以获得完整的 GPU 支持${NC}"
                echo -e "${YELLOW}或使用本地 conda 环境运行训练${NC}"
                echo ""
                
                GPU_ARGS=""
                # 挂载所有 NVIDIA 设备
                for dev in /dev/nvidia*; do
                    [ -c "$dev" ] && GPU_ARGS="$GPU_ARGS --device $dev"
                done
                # 挂载 DRI 设备 (仅字符设备，排除目录)
                for dev in /dev/dri/card* /dev/dri/renderD*; do
                    [ -c "$dev" ] && GPU_ARGS="$GPU_ARGS --device $dev"
                done
            fi
            GPU_ARGS="$GPU_ARGS --security-opt=label=disable"
        fi
    else
        # Docker 使用 --gpus
        GPU_ARGS="--gpus all"
    fi
fi

echo -e "${GREEN}Starting container ${CONTAINER_NAME}...${NC}"
echo "  Engine: ${CONTAINER_ENGINE}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  GPU: ${GPU_ARGS:-disabled}"
echo "  Data: ${DATA_DIR}"
echo "  Work: ${WORK_DIR}"
echo "  Checkpoints: ${CHECKPOINTS_DIR}"
echo ""

# 构建 ulimit 参数 (Podman rootless 不支持 memlock)
ULIMIT_ARGS="--ulimit stack=67108864"
if [ "$CONTAINER_ENGINE" = "docker" ]; then
    # Docker 支持 memlock
    ULIMIT_ARGS="$ULIMIT_ARGS --ulimit memlock=-1"
fi

# Run container
$CONTAINER_ENGINE run \
    $DETACH \
    -it \
    --rm \
    --name "$CONTAINER_NAME" \
    $GPU_ARGS \
    --shm-size=8g \
    $ULIMIT_ARGS \
    -v "$(pwd):/workspace:Z" \
    -v "${DATA_DIR}:/workspace/data:Z" \
    -v "${WORK_DIR}:/workspace/work_dirs:Z" \
    -v "${CHECKPOINTS_DIR}:/workspace/checkpoints:Z" \
    -w /workspace \
    -p 8888:8888 \
    -p 6006:6006 \
    -e DISPLAY=$DISPLAY \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    $COMMAND
