#!/usr/bin/env python3
"""
评估绿牌（新能源车牌）识别率

比较原始 ONNX 模型和微调后 ONNX 模型在绿牌上的表现
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict

# 车牌字符集
PLATE_CHARS = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民", 
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", 
    "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", 
    "黑", "领", "使", "澳",
]

BLANK_IDX = 0


def decode_prediction(output: np.ndarray) -> str:
    """解码模型输出"""
    indices = np.argmax(output, axis=-1)
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != BLANK_IDX and idx != prev_idx:
            if idx < len(PLATE_CHARS):
                chars.append(PLATE_CHARS[idx])
        prev_idx = idx
    return ''.join(chars)


def preprocess_image(image_path: str, img_h: int = 48, img_w: int = 160) -> np.ndarray:
    """预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def crop_plate(image_path: str, bbox: list) -> np.ndarray:
    """从原图裁剪车牌区域"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    x1, y1, x2, y2 = bbox
    # 确保坐标有效
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    plate_img = img[y1:y2, x1:x2]
    
    # 预处理
    plate_img = cv2.resize(plate_img, (160, 48))
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    plate_img = plate_img.astype(np.float32) / 255.0
    plate_img = np.transpose(plate_img, (2, 0, 1))
    plate_img = np.expand_dims(plate_img, axis=0)
    
    return plate_img


def load_labels(label_file: str, base_dir: str):
    """加载标签文件"""
    samples = []
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 6:
                # 完整格式: 图片路径\t标签\t预测\t是否正确\t边界框\t颜色
                img_path = parts[0]
                label = parts[1]
                bbox_str = parts[4]
                color = parts[5]
                
                # 解析边界框
                try:
                    bbox = eval(bbox_str)
                except:
                    bbox = None
                
                # 构建完整路径
                if img_path.startswith('./'):
                    img_path = img_path[2:]
                full_path = os.path.join(base_dir, img_path)
                
                samples.append({
                    'path': full_path,
                    'label': label,
                    'bbox': bbox,
                    'color': color
                })
    
    return samples


def evaluate_models(original_onnx: str, finetuned_onnx: str, samples: list):
    """评估两个模型"""
    print(f"Loading original model: {original_onnx}")
    original_session = ort.InferenceSession(original_onnx, providers=['CPUExecutionProvider'])
    
    print(f"Loading finetuned model: {finetuned_onnx}")
    finetuned_session = ort.InferenceSession(finetuned_onnx, providers=['CPUExecutionProvider'])
    
    # 按颜色分类统计
    stats = {
        'green': {'original_correct': 0, 'finetuned_correct': 0, 'total': 0},
        'blue': {'original_correct': 0, 'finetuned_correct': 0, 'total': 0},
        'all': {'original_correct': 0, 'finetuned_correct': 0, 'total': 0}
    }
    
    # 错误样本
    errors = {
        'green': {'improved': [], 'regressed': [], 'both_wrong': []},
        'blue': {'improved': [], 'regressed': [], 'both_wrong': []}
    }
    
    print("\n" + "=" * 80)
    print("评估车牌识别率")
    print("=" * 80)
    
    for sample in samples:
        img_path = sample['path']
        label = sample['label']
        bbox = sample['bbox']
        color = sample['color']
        
        # 尝试裁剪车牌
        if bbox:
            input_data = crop_plate(img_path, bbox)
        else:
            input_data = preprocess_image(img_path)
        
        if input_data is None:
            continue
        
        # 原始模型推理
        orig_output = original_session.run(None, {original_session.get_inputs()[0].name: input_data})[0]
        orig_decoded = decode_prediction(orig_output[0])
        
        # 微调模型推理
        fine_output = finetuned_session.run(None, {finetuned_session.get_inputs()[0].name: input_data})[0]
        fine_decoded = decode_prediction(fine_output[0])
        
        orig_correct = (orig_decoded == label)
        fine_correct = (fine_decoded == label)
        
        # 更新统计
        color_key = color if color in stats else 'blue'
        stats[color_key]['total'] += 1
        stats['all']['total'] += 1
        
        if orig_correct:
            stats[color_key]['original_correct'] += 1
            stats['all']['original_correct'] += 1
        if fine_correct:
            stats[color_key]['finetuned_correct'] += 1
            stats['all']['finetuned_correct'] += 1
        
        # 记录错误样本
        if orig_decoded != fine_decoded:
            if fine_correct and not orig_correct:
                errors[color_key]['improved'].append({
                    'label': label, 'original': orig_decoded, 'finetuned': fine_decoded
                })
            elif orig_correct and not fine_correct:
                errors[color_key]['regressed'].append({
                    'label': label, 'original': orig_decoded, 'finetuned': fine_decoded
                })
            elif not orig_correct and not fine_correct:
                errors[color_key]['both_wrong'].append({
                    'label': label, 'original': orig_decoded, 'finetuned': fine_decoded
                })
    
    # 打印结果
    print("\n" + "-" * 80)
    print(f"{'类别':<10} {'样本数':<10} {'原始准确率':<15} {'微调准确率':<15} {'提升':<10}")
    print("-" * 80)
    
    for color_name, color_stats in [('绿牌', stats['green']), ('蓝牌', stats['blue']), ('总计', stats['all'])]:
        total = color_stats['total']
        if total == 0:
            continue
        
        orig_acc = 100 * color_stats['original_correct'] / total
        fine_acc = 100 * color_stats['finetuned_correct'] / total
        improvement = fine_acc - orig_acc
        
        print(f"{color_name:<10} {total:<10} {orig_acc:>10.2f}%     {fine_acc:>10.2f}%     {improvement:>+8.2f}%")
    
    print("-" * 80)
    
    # 打印改善的样本
    for color_name, color_key in [('绿牌', 'green'), ('蓝牌', 'blue')]:
        if errors[color_key]['improved']:
            print(f"\n🟢 {color_name}改善样本 ({len(errors[color_key]['improved'])} 个):")
            for e in errors[color_key]['improved'][:10]:
                print(f"   {e['label']}: {e['original']} -> {e['finetuned']} ✓")
        
        if errors[color_key]['regressed']:
            print(f"\n🔴 {color_name}退化样本 ({len(errors[color_key]['regressed'])} 个):")
            for e in errors[color_key]['regressed'][:10]:
                print(f"   {e['label']}: {e['original']} -> {e['finetuned']} ✗")
    
    return stats


def main():
    # 路径配置
    base_dir = '/home/ubuntu/e2e-pecp-pdp/mmengine-lite'
    label_file = os.path.join(base_dir, 'PIC_ccpd_format_full/labels.txt')
    original_onnx = os.path.join(base_dir, 'e2e_hztk_deploy_package/hztk_rec_original.onnx')
    finetuned_onnx = os.path.join(base_dir, 'e2e_hztk_deploy_package/hztk_rec_finetuned_v2.onnx')
    
    # 检查文件
    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return 1
    
    if not os.path.exists(original_onnx):
        print(f"Error: Original ONNX not found: {original_onnx}")
        return 1
    
    if not os.path.exists(finetuned_onnx):
        print(f"Error: Finetuned ONNX not found: {finetuned_onnx}")
        return 1
    
    # 加载样本
    print(f"Loading labels from: {label_file}")
    samples = load_labels(label_file, base_dir)
    print(f"Loaded {len(samples)} samples")
    
    # 统计颜色分布
    color_counts = defaultdict(int)
    for s in samples:
        color_counts[s['color']] += 1
    print(f"Color distribution: {dict(color_counts)}")
    
    # 评估
    stats = evaluate_models(original_onnx, finetuned_onnx, samples)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
