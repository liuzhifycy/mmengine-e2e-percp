#!/usr/bin/env python3
"""
准备车牌识别微调数据集

功能:
1. 从 PIC annotations.json 裁剪检测到的车牌区域
2. 从 CCPD2020 抽取并裁剪绿牌
3. 生成训练/验证集 label 文件

用法:
    python prepare_finetune_data.py \
        --pic-annotations ../../../PIC_ccpd_format_full/annotations.json \
        --ccpd2020-dir ../../../data/ccpd/CCPD2020/ccpd_green \
        --output-dir ../../../finetune_data \
        --ccpd-samples 2000 \
        --val-ratio 0.1
"""

import os
import sys
import cv2
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


# 省份简称映射 (CCPD 使用索引)
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# 字母数字字符 (CCPD 使用索引)
ALPHANUMS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]


def parse_ccpd_filename(filename: str) -> dict:
    """解析 CCPD 文件名获取车牌信息
    
    CCPD 文件名格式:
    时间戳-区域-检测框坐标-四点坐标-车牌字符索引-亮度-模糊度.jpg
    
    例如: 00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg
    车牌字符索引: 0_0_3_24_28_24_31_33
    - 第1位: 省份索引 (0=皖)
    - 第2位: 城市字母索引 (0=A)
    - 第3-7位: 后5位字符索引
    - 对于新能源绿牌，有8位字符索引
    """
    try:
        parts = filename.replace('.jpg', '').split('-')
        if len(parts) < 5:
            return None
        
        # 车牌字符索引部分
        char_indices = parts[4].split('_')
        if len(char_indices) < 7:
            return None
        
        # 解析车牌
        province_idx = int(char_indices[0])
        if province_idx >= len(PROVINCES):
            return None
        
        plate_chars = [PROVINCES[province_idx]]
        
        for i, idx_str in enumerate(char_indices[1:]):
            idx = int(idx_str)
            if idx >= len(ALPHANUMS):
                return None
            plate_chars.append(ALPHANUMS[idx])
        
        plate_number = ''.join(plate_chars)
        
        # 解析检测框坐标 (格式: x1&y1_x2&y2)
        bbox_part = parts[2]
        bbox_coords = bbox_part.split('_')
        if len(bbox_coords) >= 2:
            x1y1 = bbox_coords[0].split('&')
            x2y2 = bbox_coords[1].split('&')
            if len(x1y1) >= 2 and len(x2y2) >= 2:
                x1, y1 = int(x1y1[0]), int(x1y1[1])
                x2, y2 = int(x2y2[0]), int(x2y2[1])
                bbox = [x1, y1, x2, y2]
            else:
                bbox = None
        else:
            bbox = None
        
        return {
            'plate_number': plate_number,
            'bbox': bbox,
            'is_green': len(char_indices) == 8  # 新能源绿牌有8位字符
        }
    except Exception as e:
        return None


def crop_plate_from_ccpd(image_path: str, bbox: list, expand_ratio: float = 0.1) -> tuple:
    """从 CCPD 图片裁剪车牌区域
    
    Args:
        image_path: 图片路径
        bbox: [x1, y1, x2, y2] 检测框
        expand_ratio: 扩展比例
    
    Returns:
        (plate_image, success)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, False
        
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 扩展边界
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * expand_ratio))
        y1 = max(0, int(y1 - bh * expand_ratio))
        x2 = min(w, int(x2 + bw * expand_ratio))
        y2 = min(h, int(y2 + bh * expand_ratio))
        
        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            return None, False
        
        return plate_img, True
    except Exception as e:
        return None, False


def process_pic_annotations(annotations_path: str, output_dir: str, base_dir: str) -> list:
    """处理 PIC annotations.json，裁剪车牌区域
    
    Args:
        annotations_path: annotations.json 路径
        output_dir: 输出目录
        base_dir: PIC 数据集基础目录 (用于解析相对路径)
    
    Returns:
        list of (image_path, plate_number) tuples
    """
    print(f"Processing PIC annotations from {annotations_path}")
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    plates_dir = os.path.join(output_dir, 'plates')
    os.makedirs(plates_dir, exist_ok=True)
    
    results = []
    skipped = 0
    
    for ann in tqdm(annotations, desc="Processing PIC"):
        # 只处理检测到车牌的图片
        if ann.get('bbox') is None:
            skipped += 1
            continue
        
        # 获取原图路径
        image_path = ann['image_path']
        if image_path.startswith('./'):
            image_path = os.path.join(base_dir, image_path[2:])
        elif not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path)
        
        if not os.path.exists(image_path):
            skipped += 1
            continue
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue
        
        # 裁剪车牌区域
        bbox = ann['bbox']
        x1, y1, x2, y2 = bbox
        
        # 稍微扩展边界
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * 0.05))
        y1 = max(0, int(y1 - bh * 0.05))
        x2 = min(w, int(x2 + bw * 0.05))
        y2 = min(h, int(y2 + bh * 0.05))
        
        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            skipped += 1
            continue
        
        # 生成输出文件名
        gt_plate = ann['gt_plate']
        original_name = os.path.basename(ann['image_path']).replace('.jpg', '').replace('.png', '')
        output_name = f"pic_{gt_plate}_{original_name}.jpg"
        output_path = os.path.join(plates_dir, output_name)
        
        # 保存裁剪的车牌图片
        cv2.imwrite(output_path, plate_img)
        
        results.append((output_path, gt_plate))
    
    print(f"  Processed: {len(results)}, Skipped: {skipped}")
    return results


def process_ccpd2020(ccpd_dir: str, output_dir: str, num_samples: int = 2000) -> list:
    """处理 CCPD2020 绿牌数据
    
    Args:
        ccpd_dir: CCPD2020/ccpd_green 目录
        output_dir: 输出目录
        num_samples: 抽取数量
    
    Returns:
        list of (image_path, plate_number) tuples
    """
    print(f"Processing CCPD2020 green plates from {ccpd_dir}")
    
    plates_dir = os.path.join(output_dir, 'plates')
    os.makedirs(plates_dir, exist_ok=True)
    
    # 收集所有图片
    all_images = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(ccpd_dir, split)
        if os.path.exists(split_dir):
            for filename in os.listdir(split_dir):
                if filename.endswith('.jpg'):
                    all_images.append(os.path.join(split_dir, filename))
    
    print(f"  Found {len(all_images)} images in CCPD2020 green")
    
    # 随机抽样
    if len(all_images) > num_samples:
        random.seed(42)
        all_images = random.sample(all_images, num_samples)
    
    results = []
    skipped = 0
    
    for image_path in tqdm(all_images, desc="Processing CCPD2020"):
        filename = os.path.basename(image_path)
        parsed = parse_ccpd_filename(filename)
        
        if parsed is None or parsed['bbox'] is None:
            skipped += 1
            continue
        
        plate_img, success = crop_plate_from_ccpd(image_path, parsed['bbox'])
        if not success:
            skipped += 1
            continue
        
        # 生成输出文件名
        plate_number = parsed['plate_number']
        output_name = f"ccpd_{plate_number}_{os.path.splitext(filename)[0][-8:]}.jpg"
        output_path = os.path.join(plates_dir, output_name)
        
        # 保存裁剪的车牌图片
        cv2.imwrite(output_path, plate_img)
        
        results.append((output_path, plate_number))
    
    print(f"  Processed: {len(results)}, Skipped: {skipped}")
    return results


def create_label_files(data: list, output_dir: str, val_ratio: float = 0.1):
    """创建训练/验证标签文件
    
    Args:
        data: list of (image_path, plate_number)
        output_dir: 输出目录
        val_ratio: 验证集比例
    """
    print(f"Creating label files with {len(data)} samples, val_ratio={val_ratio}")
    
    # 打乱数据
    random.seed(42)
    random.shuffle(data)
    
    # 划分训练/验证集
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    # 写入标签文件
    train_label_path = os.path.join(output_dir, 'train_label.txt')
    val_label_path = os.path.join(output_dir, 'val_label.txt')
    
    with open(train_label_path, 'w', encoding='utf-8') as f:
        for image_path, plate_number in train_data:
            # 使用相对路径
            rel_path = os.path.relpath(image_path, output_dir)
            f.write(f"{rel_path}\t{plate_number}\n")
    
    with open(val_label_path, 'w', encoding='utf-8') as f:
        for image_path, plate_number in val_data:
            rel_path = os.path.relpath(image_path, output_dir)
            f.write(f"{rel_path}\t{plate_number}\n")
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Labels saved to: {train_label_path}, {val_label_path}")
    
    # 统计车牌类型分布
    blue_count = sum(1 for _, p in data if len(p) == 7)
    green_count = sum(1 for _, p in data if len(p) == 8)
    print(f"  Blue plates (7 chars): {blue_count}")
    print(f"  Green plates (8 chars): {green_count}")


def main():
    parser = argparse.ArgumentParser(description='Prepare plate recognition finetune data')
    parser.add_argument('--pic-annotations', type=str, 
                        default='../../../PIC_ccpd_format_full/annotations.json',
                        help='Path to PIC annotations.json')
    parser.add_argument('--ccpd2020-dir', type=str,
                        default='../../../data/ccpd/CCPD2020/ccpd_green',
                        help='Path to CCPD2020/ccpd_green directory')
    parser.add_argument('--output-dir', type=str,
                        default='../../../finetune_data',
                        help='Output directory')
    parser.add_argument('--ccpd-samples', type=int, default=2000,
                        help='Number of CCPD samples to use')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    pic_annotations = args.pic_annotations
    if not os.path.isabs(pic_annotations):
        pic_annotations = os.path.normpath(os.path.join(script_dir, pic_annotations))
    
    ccpd2020_dir = args.ccpd2020_dir
    if not os.path.isabs(ccpd2020_dir):
        ccpd2020_dir = os.path.normpath(os.path.join(script_dir, ccpd2020_dir))
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Plate Recognition Finetune Data Preparation")
    print("=" * 60)
    print(f"PIC annotations: {pic_annotations}")
    print(f"CCPD2020 dir: {ccpd2020_dir}")
    print(f"Output dir: {output_dir}")
    print(f"CCPD samples: {args.ccpd_samples}")
    print(f"Val ratio: {args.val_ratio}")
    print("=" * 60)
    
    all_data = []
    
    # 处理 PIC 数据
    if os.path.exists(pic_annotations):
        # 获取 PIC 数据集基础目录
        base_dir = os.path.dirname(os.path.dirname(pic_annotations))
        pic_data = process_pic_annotations(pic_annotations, output_dir, base_dir)
        all_data.extend(pic_data)
    else:
        print(f"Warning: PIC annotations not found: {pic_annotations}")
    
    # 处理 CCPD2020 数据
    if os.path.exists(ccpd2020_dir):
        ccpd_data = process_ccpd2020(ccpd2020_dir, output_dir, args.ccpd_samples)
        all_data.extend(ccpd_data)
    else:
        print(f"Warning: CCPD2020 dir not found: {ccpd2020_dir}")
    
    # 创建标签文件
    if all_data:
        create_label_files(all_data, output_dir, args.val_ratio)
        
        # 保存数据统计信息
        stats = {
            'total_samples': len(all_data),
            'pic_samples': len([d for d in all_data if 'pic_' in d[0]]),
            'ccpd_samples': len([d for d in all_data if 'ccpd_' in d[0]]),
            'blue_plates': sum(1 for _, p in all_data if len(p) == 7),
            'green_plates': sum(1 for _, p in all_data if len(p) == 8),
        }
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStats saved to: {stats_path}")
    else:
        print("Error: No data processed!")
        sys.exit(1)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
