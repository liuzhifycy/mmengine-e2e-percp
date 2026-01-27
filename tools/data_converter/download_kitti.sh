#!/bin/bash
# KITTI 3D Detection 数据下载脚本
# 
# KITTI 数据集官网: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
# 需要注册账号后下载
#
# 数据集大小约 12GB，包含:
# - data_object_velodyne.zip (29GB 解压后约 12GB)
# - data_object_calib.zip (1.6MB)
# - data_object_label_2.zip (5MB)
# - data_object_image_2.zip (12GB, 可选，仅用于可视化)

set -e

DATA_ROOT="${1:-data/kitti}"

echo "============================================"
echo "KITTI 3D Detection Dataset Preparation"
echo "============================================"
echo ""
echo "Target directory: $DATA_ROOT"
echo ""

# 创建目录结构
mkdir -p $DATA_ROOT/training/velodyne
mkdir -p $DATA_ROOT/training/calib
mkdir -p $DATA_ROOT/training/label_2
mkdir -p $DATA_ROOT/training/image_2
mkdir -p $DATA_ROOT/testing/velodyne
mkdir -p $DATA_ROOT/testing/calib
mkdir -p $DATA_ROOT/testing/image_2
mkdir -p $DATA_ROOT/ImageSets

echo "Directory structure created."
echo ""
echo "Please download the following files from KITTI website:"
echo "http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"
echo ""
echo "Required files:"
echo "  1. data_object_velodyne.zip (Velodyne point clouds)"
echo "  2. data_object_calib.zip (Camera calibration)"
echo "  3. data_object_label_2.zip (Training labels)"
echo ""
echo "Optional files (for visualization):"
echo "  4. data_object_image_2.zip (Left color images)"
echo ""
echo "After downloading, extract them to $DATA_ROOT:"
echo "  unzip data_object_velodyne.zip -d $DATA_ROOT"
echo "  unzip data_object_calib.zip -d $DATA_ROOT"
echo "  unzip data_object_label_2.zip -d $DATA_ROOT"
echo "  unzip data_object_image_2.zip -d $DATA_ROOT  # optional"
echo ""

# 创建 ImageSets 分割文件
echo "Creating train/val/test splits..."

# KITTI 官方只提供 training (7481 samples) 和 testing (7518 samples)
# 通常将 training 按 3712:3769 分成 train:val
# 这里使用 mmdet3d 的标准分割

# 生成 train.txt (前 3712 个样本)
for i in $(seq -f "%06g" 0 3711); do
    echo $i
done > $DATA_ROOT/ImageSets/train.txt

# 生成 val.txt (后 3769 个样本)  
for i in $(seq -f "%06g" 3712 7480); do
    echo $i
done > $DATA_ROOT/ImageSets/val.txt

# 生成 trainval.txt (所有 training 样本)
for i in $(seq -f "%06g" 0 7480); do
    echo $i
done > $DATA_ROOT/ImageSets/trainval.txt

# 生成 test.txt (testing 样本)
for i in $(seq -f "%06g" 0 7517); do
    echo $i
done > $DATA_ROOT/ImageSets/test.txt

echo "ImageSets created:"
echo "  - train.txt: 3712 samples"
echo "  - val.txt: 3769 samples"
echo "  - trainval.txt: 7481 samples"
echo "  - test.txt: 7518 samples"
echo ""
echo "After extracting data, run the data converter:"
echo "  python tools/data_converter/kitti_converter.py --data-root $DATA_ROOT"
echo ""
echo "Done!"
