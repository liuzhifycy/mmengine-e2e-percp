#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCPD 数据集预处理脚本
将 CCPD2020 (新能源车牌) 转换为:
1. COCO 格式 - 用于 YOLO11 车牌检测训练
2. 裁剪+矫正的车牌图片 - 用于 LPRNet 字符识别训练
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


# 省份简称
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# 字母和数字 (车牌第二位及之后)
CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，提取车牌信息
    
    CCPD2020 文件名格式:
    时间戳-角度信息-边界框坐标-四个顶点坐标-车牌号索引-亮度-模糊度.jpg
    
    例如: 00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg
    
    Returns:
        dict: 包含 bbox, vertices, plate_number 的字典，解析失败返回 None
    """
    try:
        parts = filename.rsplit('.', 1)[0].split('-')
        if len(parts) < 5:
            return None
        
        # 解析边界框 (格式: x1&y1_x2&y2)
        bbox_str = parts[2]
        bbox_parts = bbox_str.split('_')
        x1, y1 = map(int, bbox_parts[0].split('&'))
        x2, y2 = map(int, bbox_parts[1].split('&'))
        
        # 解析四个顶点坐标 (格式: x1&y1_x2&y2_x3&y3_x4&y4)
        # 顺序: 右下, 左下, 左上, 右上
        vertices_str = parts[3]
        vertices_parts = vertices_str.split('_')
        vertices = []
        for vp in vertices_parts:
            vx, vy = map(int, vp.split('&'))
            vertices.append([vx, vy])
        
        # 解析车牌号 (格式: 省份索引_字符1索引_字符2索引_...)
        plate_indices_str = parts[4]
        plate_indices = list(map(int, plate_indices_str.split('_')))
        
        # 转换为车牌号字符串
        # CCPD2020 是新能源车牌，8位字符
        plate_chars = []
        plate_chars.append(PROVINCES[plate_indices[0]])  # 省份
        for idx in plate_indices[1:]:
            plate_chars.append(CHARS[idx])
        plate_number = ''.join(plate_chars)
        
        return {
            'bbox': [x1, y1, x2, y2],
            'vertices': vertices,  # [右下, 左下, 左上, 右上]
            'plate_number': plate_number
        }
    except Exception as e:
        return None


def order_points(pts):
    """
    将四个顶点按照 [左上, 右上, 右下, 左下] 顺序排列
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 左上角点的和最小，右下角点的和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 右上角点的差值最小，左下角点的差值最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect


def four_point_transform(image, pts):
    """
    透视变换矫正车牌
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算新图像宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算新图像高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 目标坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    # 透视变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def process_ccpd_split(src_dir, dst_detection_dir, dst_recognition_dir, split_name):
    """
    处理 CCPD 数据集的一个分割 (train/val/test)
    
    Args:
        src_dir: 源图片目录
        dst_detection_dir: 检测数据输出目录
        dst_recognition_dir: 识别数据输出目录
        split_name: 分割名称
    """
    src_path = Path(src_dir)
    
    # 输出目录
    det_images_dir = Path(dst_detection_dir) / split_name / 'images'
    det_labels_dir = Path(dst_detection_dir) / split_name / 'labels'
    rec_dir = Path(dst_recognition_dir) / split_name
    
    det_images_dir.mkdir(parents=True, exist_ok=True)
    det_labels_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO 格式数据
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'plate'}]
    }
    
    image_id = 0
    annotation_id = 0
    
    # 遍历所有图片
    image_files = list(src_path.glob('*.jpg'))
    
    for img_file in tqdm(image_files, desc=f'Processing {split_name}'):
        # 解析文件名
        info = parse_ccpd_filename(img_file.name)
        if info is None:
            continue
        
        # 读取图片
        img = cv2.imread(str(img_file))
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
        
        # === 1. 检测数据 (COCO 格式) ===
        # 复制图片
        dst_img_path = det_images_dir / img_file.name
        shutil.copy(img_file, dst_img_path)
        
        # 添加到 COCO 数据
        coco_data['images'].append({
            'id': image_id,
            'file_name': img_file.name,
            'width': w,
            'height': h
        })
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        coco_data['annotations'].append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 0,
            'bbox': [x1, y1, bbox_w, bbox_h],  # COCO 格式: [x, y, width, height]
            'area': bbox_w * bbox_h,
            'iscrowd': 0
        })
        
        # === 2. YOLO 格式标签 ===
        # 归一化坐标
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = bbox_w / w
        bh = bbox_h / h
        
        label_file = det_labels_dir / (img_file.stem + '.txt')
        with open(label_file, 'w') as f:
            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        
        # === 3. 识别数据 (裁剪+矫正) ===
        try:
            vertices = np.array(info['vertices'], dtype=np.float32)
            # 透视变换矫正
            plate_img = four_point_transform(img, vertices)
            # 统一尺寸为 94x24 (LPRNet 输入尺寸)
            plate_img = cv2.resize(plate_img, (94, 24), interpolation=cv2.INTER_CUBIC)
            
            # 保存，文件名包含车牌号
            plate_number = info['plate_number']
            rec_img_name = f'{plate_number}_{image_id}.jpg'
            rec_img_path = rec_dir / rec_img_name
            cv2.imwrite(str(rec_img_path), plate_img)
        except Exception as e:
            pass
        
        image_id += 1
        annotation_id += 1
    
    # 保存 COCO JSON
    coco_json_path = Path(dst_detection_dir) / f'{split_name}.json'
    with open(coco_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f'{split_name}: {image_id} images processed')
    return image_id


def main():
    # 路径配置
    base_dir = Path('/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd')
    ccpd2020_dir = base_dir / 'CCPD2020' / 'ccpd_green'
    
    output_detection_dir = base_dir / 'processed' / 'detection'
    output_recognition_dir = base_dir / 'processed' / 'recognition'
    
    # 处理各个分割
    total = 0
    for split in ['train', 'val', 'test']:
        src_dir = ccpd2020_dir / split
        if src_dir.exists():
            count = process_ccpd_split(
                src_dir, 
                output_detection_dir, 
                output_recognition_dir,
                split
            )
            total += count
    
    print(f'\nTotal: {total} images processed')
    print(f'Detection data saved to: {output_detection_dir}')
    print(f'Recognition data saved to: {output_recognition_dir}')


if __name__ == '__main__':
    main()
