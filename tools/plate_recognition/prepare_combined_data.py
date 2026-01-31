#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 CCPD2019 + CCPD2020 数据集，生成 COCO 格式标注
训练集目标：~18000张 (CCPD2019: 10000张 + CCPD2020: 8000张)
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random

random.seed(42)

# 省份简称
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# 字母和数字
CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名 (2019/2020 通用)
    """
    try:
        parts = filename.rsplit('.', 1)[0].split('-')
        if len(parts) < 5:
            return None
        
        # 解析边界框
        bbox_str = parts[2]
        bbox_parts = bbox_str.split('_')
        x1, y1 = map(int, bbox_parts[0].split('&'))
        x2, y2 = map(int, bbox_parts[1].split('&'))
        
        # 解析四个顶点坐标
        vertices_str = parts[3]
        vertices_parts = vertices_str.split('_')
        vertices = []
        for vp in vertices_parts:
            vx, vy = map(int, vp.split('&'))
            vertices.append([vx, vy])
        
        # 解析车牌号
        plate_indices_str = parts[4]
        plate_indices = list(map(int, plate_indices_str.split('_')))
        
        plate_chars = []
        if plate_indices[0] < len(PROVINCES):
            plate_chars.append(PROVINCES[plate_indices[0]])
        for idx in plate_indices[1:]:
            if idx < len(CHARS):
                plate_chars.append(CHARS[idx])
        plate_number = ''.join(plate_chars)
        
        return {
            'bbox': [x1, y1, x2, y2],
            'vertices': vertices,
            'plate_number': plate_number
        }
    except Exception as e:
        return None


def collect_ccpd2019_images(base_dir, num_samples=10000):
    """从 CCPD2019 各子集中抽取图片"""
    base_dir = Path(base_dir)
    all_images = []
    
    # 优先从 ccpd_base 抽取
    base_path = base_dir / 'ccpd_base'
    if base_path.exists():
        images = list(base_path.glob('*.jpg'))
        random.shuffle(images)
        all_images.extend(images[:min(len(images), num_samples)])
    
    # 如果不够，从其他子集补充
    if len(all_images) < num_samples:
        other_dirs = ['ccpd_blur', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather', 'ccpd_db']
        for subdir in other_dirs:
            subdir_path = base_dir / subdir
            if subdir_path.exists():
                images = list(subdir_path.glob('*.jpg'))
                random.shuffle(images)
                need = num_samples - len(all_images)
                all_images.extend(images[:min(len(images), need)])
                if len(all_images) >= num_samples:
                    break
    
    return all_images[:num_samples]


def collect_ccpd2020_images(base_dir, train_count=8000, val_count=1001, test_count=2000):
    """从 CCPD2020 收集图片"""
    base_dir = Path(base_dir) / 'ccpd_green'
    
    train_images = list((base_dir / 'train').glob('*.jpg'))
    val_images = list((base_dir / 'val').glob('*.jpg'))
    test_images = list((base_dir / 'test').glob('*.jpg'))
    
    # 合并所有图片重新分配
    all_images = train_images + val_images + test_images
    random.shuffle(all_images)
    
    return {
        'train': all_images[:train_count],
        'val': all_images[train_count:train_count + val_count],
        'test': all_images[train_count + val_count:train_count + val_count + test_count]
    }


def process_split(image_list, output_dir, split_name):
    """处理一个数据分割"""
    output_dir = Path(output_dir)
    images_dir = output_dir / split_name / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'plate'}]
    }
    
    image_id = 0
    annotation_id = 0
    
    for img_path in tqdm(image_list, desc=f'Processing {split_name}'):
        info = parse_ccpd_filename(img_path.name)
        if info is None:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        x1, y1, x2, y2 = info['bbox']
        
        # 边界检查
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 复制图片
        new_filename = f'{image_id:06d}.jpg'
        dst_path = images_dir / new_filename
        shutil.copy(img_path, dst_path)
        
        # COCO 标注
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        coco_data['images'].append({
            'id': image_id,
            'file_name': new_filename,
            'width': w,
            'height': h
        })
        
        coco_data['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 0,
            'bbox': [x1, y1, bbox_w, bbox_h],
            'area': bbox_w * bbox_h,
            'iscrowd': 0
        })
        
        image_id += 1
        annotation_id += 1
    
    # 保存 COCO JSON
    json_path = output_dir / f'{split_name}.json'
    with open(json_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f'{split_name}: {image_id} images')
    return image_id


def main():
    base_dir = Path('/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd')
    ccpd2019_dir = base_dir / 'CCPD2019'
    ccpd2020_dir = base_dir / 'CCPD2020'
    output_dir = base_dir / 'combined'
    
    # 清理旧数据
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print("=" * 50)
    print("收集 CCPD2019 数据 (10000张)...")
    ccpd2019_train = collect_ccpd2019_images(ccpd2019_dir, num_samples=10000)
    print(f"CCPD2019: {len(ccpd2019_train)} 张")
    
    print("=" * 50)
    print("收集 CCPD2020 数据...")
    ccpd2020_data = collect_ccpd2020_images(ccpd2020_dir, train_count=8000, val_count=1001, test_count=2000)
    print(f"CCPD2020 Train: {len(ccpd2020_data['train'])} 张")
    print(f"CCPD2020 Val: {len(ccpd2020_data['val'])} 张")
    print(f"CCPD2020 Test: {len(ccpd2020_data['test'])} 张")
    
    print("=" * 50)
    print("处理训练集 (CCPD2019 + CCPD2020)...")
    train_images = ccpd2019_train + ccpd2020_data['train']
    random.shuffle(train_images)
    process_split(train_images, output_dir, 'train')
    
    print("=" * 50)
    print("处理验证集...")
    process_split(ccpd2020_data['val'], output_dir, 'val')
    
    print("=" * 50)
    print("处理测试集...")
    process_split(ccpd2020_data['test'], output_dir, 'test')
    
    print("=" * 50)
    print("数据处理完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
