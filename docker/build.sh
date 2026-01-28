#!/bin/bash
# =============================================================================
# MMEngine-Lite Docker 构建脚本
# =============================================================================
# 使用方法:
#   ./docker/build.sh              # 构建镜像
#   ./docker/build.sh --no-cache   # 不使用缓存构建
#   ./docker/build.sh --push       # 构建并推送到 Docker Hub
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
IMAGE_NAME="mmlite"
IMAGE_TAG="latest"
DOCKERFILE="docker/Dockerfile"
BUILD_CONTEXT="."

# 解析参数
NO_CACHE=""
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}构建 MMEngine-Lite Docker 镜像${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "镜像名称: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "Dockerfile: ${YELLOW}${DOCKERFILE}${NC}"
echo ""

# 构建镜像
echo -e "${GREEN}开始构建...${NC}"
docker build \
    ${NO_CACHE} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f ${DOCKERFILE} \
    ${BUILD_CONTEXT}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}构建完成!${NC}"
echo -e "${GREEN}========================================${NC}"

# 显示镜像信息
docker images ${IMAGE_NAME}:${IMAGE_TAG}

# 推送镜像 (可选)
if [ "$PUSH" = true ]; then
    echo ""
    echo -e "${GREEN}推送镜像到 Docker Hub...${NC}"
    docker push ${IMAGE_NAME}:${IMAGE_TAG}
fi

echo ""
echo -e "${GREEN}使用方法:${NC}"
echo -e "  运行容器:    ${YELLOW}./docker/run.sh${NC}"
echo -e "  进入容器:    ${YELLOW}docker exec -it mmlite-dev bash${NC}"
echo -e "  使用compose: ${YELLOW}cd docker && docker-compose up -d${NC}"
