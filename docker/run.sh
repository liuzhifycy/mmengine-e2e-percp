#!/bin/bash
# run.sh - Run mmengine-lite Docker container
# Usage: ./run.sh [options] [command]

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-mmengine-lite}"
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
GPU_ARGS="--gpus all"
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
            GPU_ARGS=""
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

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo -e "${RED}Error: Image ${IMAGE_NAME}:${IMAGE_TAG} not found.${NC}"
    echo -e "${YELLOW}Please build the image first: ./build.sh${NC}"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$WORK_DIR" "$CHECKPOINTS_DIR"

# Check if container with same name is running
if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo -e "${YELLOW}Container ${CONTAINER_NAME} is already running.${NC}"
    echo "Executing command in existing container..."
    docker exec -it "$CONTAINER_NAME" $COMMAND
    exit 0
fi

# Remove stopped container with same name
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo -e "${YELLOW}Removing stopped container ${CONTAINER_NAME}...${NC}"
    docker rm "$CONTAINER_NAME"
fi

echo -e "${GREEN}Starting container ${CONTAINER_NAME}...${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  GPU: ${GPU_ARGS:-disabled}"
echo "  Data: ${DATA_DIR}"
echo "  Work: ${WORK_DIR}"
echo "  Checkpoints: ${CHECKPOINTS_DIR}"
echo ""

# Run container
docker run \
    $DETACH \
    -it \
    --rm \
    --name "$CONTAINER_NAME" \
    $GPU_ARGS \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -v "${DATA_DIR}:/workspace/data" \
    -v "${WORK_DIR}:/workspace/work_dirs" \
    -v "${CHECKPOINTS_DIR}:/workspace/checkpoints" \
    -p 8888:8888 \
    -p 6006:6006 \
    -e DISPLAY=$DISPLAY \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    $COMMAND
