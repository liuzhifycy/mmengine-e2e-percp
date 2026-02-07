#!/usr/bin/env python3
"""
CCPD 数据集评测脚本
从 CCPD2019 和 CCPD2020 各抽取 500 张图片评测三个 OCR 模型:
1. Original (hztk_rec) - 原始模型，使用通用字典
2. PP-OCRv3 - PP-OCRv3 预训练模型，使用通用字典
3. Finetuned - 车牌微调模型，使用车牌专用字典
"""

import os
import random
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
import json
from collections import defaultdict

# CCPD 字符映射
# 省份
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "警", "学", "O"]  # O 是特殊字符

# 字母（无 I 和 O）
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
             'W', 'X', 'Y', 'Z', 'O']  # 最后的O是特殊字符

# 字母数字（无 I 和 O）
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
       'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', 
       '6', '7', '8', '9', 'O']  # 最后的O是特殊字符


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名获取车牌信息
    格式: area-tilt-bbox-vertices-plate_idx-brightness-blur.jpg
    例如: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    
    plate_idx 格式: 省份_字母_字母数字_字母数字_字母数字_字母数字_字母数字
    对于绿牌(8位): 省份_字母_字母数字_字母数字_字母数字_字母数字_字母数字_字母数字
    """
    basename = os.path.basename(filename)
    name = basename.rsplit('.', 1)[0]
    parts = name.split('-')
    
    if len(parts) < 5:
        return None, None, None
    
    # 获取边界框坐标
    try:
        bbox_str = parts[2]  # 例如 "154&383_386&473"
        coords = bbox_str.split('_')
        left_up = [int(x) for x in coords[0].split('&')]
        right_down = [int(x) for x in coords[1].split('&')]
        bbox = [left_up[0], left_up[1], right_down[0], right_down[1]]
    except:
        bbox = None
    
    # 获取四个顶点坐标（用于透视变换）
    try:
        vertices_str = parts[3]  # 例如 "386&473_177&454_154&383_363&402"
        vertices = []
        for v in vertices_str.split('_'):
            x, y = v.split('&')
            vertices.append([int(x), int(y)])
        # 顶点顺序：右下、左下、左上、右上
    except:
        vertices = None
    
    # 获取车牌号
    try:
        plate_idx_str = parts[4]  # 例如 "0_0_22_27_27_33_16"
        indices = [int(x) for x in plate_idx_str.split('_')]
        
        if len(indices) == 7:  # 蓝牌 7 位
            plate = PROVINCES[indices[0]] + ALPHABETS[indices[1]]
            for i in range(2, 7):
                plate += ADS[indices[i]]
        elif len(indices) == 8:  # 绿牌 8 位
            plate = PROVINCES[indices[0]] + ALPHABETS[indices[1]]
            for i in range(2, 8):
                plate += ADS[indices[i]]
        else:
            plate = None
    except Exception as e:
        plate = None
    
    return plate, bbox, vertices


def crop_plate_from_image(img, bbox=None, vertices=None, use_perspective=True):
    """
    从图片中裁剪车牌区域
    """
    if vertices and use_perspective and len(vertices) == 4:
        # 使用透视变换
        # 顶点顺序：右下、左下、左上、右上 -> 需要转换为：左上、右上、右下、左下
        src_pts = np.float32([vertices[2], vertices[3], vertices[0], vertices[1]])
        
        # 计算目标尺寸
        width = max(
            np.linalg.norm(np.array(vertices[3]) - np.array(vertices[2])),
            np.linalg.norm(np.array(vertices[0]) - np.array(vertices[1]))
        )
        height = max(
            np.linalg.norm(np.array(vertices[2]) - np.array(vertices[1])),
            np.linalg.norm(np.array(vertices[3]) - np.array(vertices[0]))
        )
        
        width = int(width)
        height = int(height)
        
        if width < 10 or height < 10:
            return None
        
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        plate_img = cv2.warpPerspective(img, M, (width, height))
        return plate_img
    
    elif bbox:
        # 使用边界框裁剪
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return img[y1:y2, x1:x2]
    
    return None


def preprocess_image(img, target_size=(160, 48)):
    """
    预处理图像用于 OCR
    """
    h, w = img.shape[:2]
    ratio = w / float(h)
    target_w, target_h = target_size
    
    # 保持宽高比缩放
    if ratio > target_w / target_h:
        new_w = target_w
        new_h = int(new_w / ratio)
    else:
        new_h = target_h
        new_w = int(new_h * ratio)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # 创建目标大小的画布（灰色填充）
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 127
    
    # 将缩放后的图像放到画布左上角
    canvas[:new_h, :new_w] = resized
    
    # 归一化
    canvas = canvas.astype(np.float32)
    canvas = (canvas - 127.5) / 127.5
    
    # 转换为 NCHW 格式
    canvas = np.transpose(canvas, (2, 0, 1))
    canvas = np.expand_dims(canvas, axis=0)
    
    return canvas


class OCRModel:
    """OCR 模型封装"""
    
    def __init__(self, model_path, dict_path, input_name='x'):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = input_name
        
        # 加载字典
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.chars = ['blank'] + [line.strip() for line in f if line.strip()]
        
        print(f"Loaded model: {model_path}")
        print(f"  Input name: {input_name}")
        print(f"  Dict size: {len(self.chars)} chars")
    
    def predict(self, img):
        """
        预测车牌号
        """
        input_data = preprocess_image(img)
        
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_data})
        elapsed = (time.time() - start_time) * 1000
        
        # CTC 解码
        output = outputs[0]  # [1, T, C]
        pred_indices = np.argmax(output[0], axis=1)
        
        # 去除重复和空白
        result = []
        prev_idx = -1
        for idx in pred_indices:
            if idx != 0 and idx != prev_idx:  # 0 是 blank
                if idx < len(self.chars):
                    result.append(self.chars[idx])
            prev_idx = idx
        
        return ''.join(result), elapsed


def sample_images(directory, n=500, extensions=['.jpg', '.jpeg', '.png']):
    """
    从目录中随机抽取 n 张图片
    """
    all_files = []
    for ext in extensions:
        all_files.extend(Path(directory).glob(f'*{ext}'))
        all_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    all_files = list(set(all_files))
    
    if len(all_files) <= n:
        return [str(f) for f in all_files]
    
    return [str(f) for f in random.sample(all_files, n)]


def main():
    random.seed(42)  # 固定随机种子以便复现
    
    # 数据集路径
    ccpd2019_dir = "/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd/CCPD2019/ccpd_base"
    ccpd2020_dir = "/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd/CCPD2020/ccpd_green/test"
    
    # 模型路径
    original_model = "/home/ubuntu/e2e-pecp-pdp/mmengine-lite/e2e_hztk_deploy_package/hztk_rec.onnx"
    ppocrv3_model = "/home/ubuntu/e2e-pecp-pdp/PaddleOCR/inference/ch_PP-OCRv3_rec_infer/ch_ppocr_v3_rec.onnx"
    finetuned_model = "/home/ubuntu/e2e-pecp-pdp/mmengine-lite/e2e_hztk_deploy_package/plate_rec_finetuned.onnx"
    
    # 字典路径
    original_dict = "/home/ubuntu/e2e-pecp-pdp/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt"
    ppocrv3_dict = "/home/ubuntu/e2e-pecp-pdp/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt"
    finetuned_dict = "/home/ubuntu/e2e-pecp-pdp/PaddleOCR/ppocr/utils/dict/plate_dict.txt"
    
    # 检查文件是否存在
    for f in [original_model, ppocrv3_model, finetuned_model, original_dict, ppocrv3_dict, finetuned_dict]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return
    
    # 加载模型
    print("Loading models...")
    model_original = OCRModel(original_model, original_dict, input_name='data')
    model_ppocrv3 = OCRModel(ppocrv3_model, ppocrv3_dict, input_name='x')
    model_finetuned = OCRModel(finetuned_model, finetuned_dict, input_name='x')
    
    # 抽取样本
    print(f"\nSampling images...")
    ccpd2019_samples = sample_images(ccpd2019_dir, n=500)
    ccpd2020_samples = sample_images(ccpd2020_dir, n=500)
    
    print(f"  CCPD2019 (blue plates): {len(ccpd2019_samples)} samples")
    print(f"  CCPD2020 (green plates): {len(ccpd2020_samples)} samples")
    
    # 评测结果
    results = {
        'original': {'ccpd2019': [], 'ccpd2020': [], 'times': []},
        'ppocrv3': {'ccpd2019': [], 'ccpd2020': [], 'times': []},
        'finetuned': {'ccpd2019': [], 'ccpd2020': [], 'times': []},
    }
    
    errors = {
        'original': {'ccpd2019': [], 'ccpd2020': []},
        'ppocrv3': {'ccpd2019': [], 'ccpd2020': []},
        'finetuned': {'ccpd2019': [], 'ccpd2020': []},
    }
    
    # 处理 CCPD2019 (蓝牌)
    print("\n" + "="*60)
    print("Evaluating CCPD2019 (Blue Plates)...")
    print("="*60)
    
    valid_count = 0
    for i, img_path in enumerate(ccpd2019_samples):
        plate_gt, bbox, vertices = parse_ccpd_filename(img_path)
        
        if not plate_gt:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        plate_img = crop_plate_from_image(img, bbox, vertices)
        if plate_img is None or plate_img.size == 0:
            continue
        
        valid_count += 1
        
        # Original model
        pred_orig, time_orig = model_original.predict(plate_img)
        correct_orig = (pred_orig == plate_gt)
        results['original']['ccpd2019'].append(correct_orig)
        results['original']['times'].append(time_orig)
        if not correct_orig:
            errors['original']['ccpd2019'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_orig
            })
        
        # PP-OCRv3 model
        pred_v3, time_v3 = model_ppocrv3.predict(plate_img)
        correct_v3 = (pred_v3 == plate_gt)
        results['ppocrv3']['ccpd2019'].append(correct_v3)
        results['ppocrv3']['times'].append(time_v3)
        if not correct_v3:
            errors['ppocrv3']['ccpd2019'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_v3
            })
        
        # Finetuned model
        pred_ft, time_ft = model_finetuned.predict(plate_img)
        correct_ft = (pred_ft == plate_gt)
        results['finetuned']['ccpd2019'].append(correct_ft)
        results['finetuned']['times'].append(time_ft)
        if not correct_ft:
            errors['finetuned']['ccpd2019'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_ft
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(ccpd2019_samples)} images...")
    
    print(f"  Valid samples: {valid_count}")
    
    # 处理 CCPD2020 (绿牌)
    print("\n" + "="*60)
    print("Evaluating CCPD2020 (Green Plates)...")
    print("="*60)
    
    valid_count = 0
    for i, img_path in enumerate(ccpd2020_samples):
        plate_gt, bbox, vertices = parse_ccpd_filename(img_path)
        
        if not plate_gt:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        plate_img = crop_plate_from_image(img, bbox, vertices)
        if plate_img is None or plate_img.size == 0:
            continue
        
        valid_count += 1
        
        # Original model
        pred_orig, time_orig = model_original.predict(plate_img)
        correct_orig = (pred_orig == plate_gt)
        results['original']['ccpd2020'].append(correct_orig)
        results['original']['times'].append(time_orig)
        if not correct_orig:
            errors['original']['ccpd2020'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_orig
            })
        
        # PP-OCRv3 model
        pred_v3, time_v3 = model_ppocrv3.predict(plate_img)
        correct_v3 = (pred_v3 == plate_gt)
        results['ppocrv3']['ccpd2020'].append(correct_v3)
        results['ppocrv3']['times'].append(time_v3)
        if not correct_v3:
            errors['ppocrv3']['ccpd2020'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_v3
            })
        
        # Finetuned model
        pred_ft, time_ft = model_finetuned.predict(plate_img)
        correct_ft = (pred_ft == plate_gt)
        results['finetuned']['ccpd2020'].append(correct_ft)
        results['finetuned']['times'].append(time_ft)
        if not correct_ft:
            errors['finetuned']['ccpd2020'].append({
                'file': os.path.basename(img_path),
                'gt': plate_gt,
                'pred': pred_ft
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(ccpd2020_samples)} images...")
    
    print(f"  Valid samples: {valid_count}")
    
    # 打印结果
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for model_name in ['original', 'ppocrv3', 'finetuned']:
        print(f"\n{model_name.upper()} Model:")
        
        ccpd2019_acc = sum(results[model_name]['ccpd2019']) / len(results[model_name]['ccpd2019']) * 100 if results[model_name]['ccpd2019'] else 0
        ccpd2020_acc = sum(results[model_name]['ccpd2020']) / len(results[model_name]['ccpd2020']) * 100 if results[model_name]['ccpd2020'] else 0
        
        total_correct = sum(results[model_name]['ccpd2019']) + sum(results[model_name]['ccpd2020'])
        total_samples = len(results[model_name]['ccpd2019']) + len(results[model_name]['ccpd2020'])
        total_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        avg_time = np.mean(results[model_name]['times']) if results[model_name]['times'] else 0
        
        print(f"  CCPD2019 (Blue):  {ccpd2019_acc:.2f}% ({sum(results[model_name]['ccpd2019'])}/{len(results[model_name]['ccpd2019'])})")
        print(f"  CCPD2020 (Green): {ccpd2020_acc:.2f}% ({sum(results[model_name]['ccpd2020'])}/{len(results[model_name]['ccpd2020'])})")
        print(f"  Total:            {total_acc:.2f}% ({total_correct}/{total_samples})")
        print(f"  Avg Time:         {avg_time:.2f}ms")
    
    # 对比表格
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<15} {'CCPD2019(Blue)':<18} {'CCPD2020(Green)':<18} {'Total':<12} {'Avg Time':<10}")
    print("-"*80)
    
    for model_name in ['original', 'ppocrv3', 'finetuned']:
        ccpd2019_acc = sum(results[model_name]['ccpd2019']) / len(results[model_name]['ccpd2019']) * 100 if results[model_name]['ccpd2019'] else 0
        ccpd2020_acc = sum(results[model_name]['ccpd2020']) / len(results[model_name]['ccpd2020']) * 100 if results[model_name]['ccpd2020'] else 0
        total_correct = sum(results[model_name]['ccpd2019']) + sum(results[model_name]['ccpd2020'])
        total_samples = len(results[model_name]['ccpd2019']) + len(results[model_name]['ccpd2020'])
        total_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
        avg_time = np.mean(results[model_name]['times']) if results[model_name]['times'] else 0
        
        print(f"{model_name:<15} {ccpd2019_acc:>6.2f}%          {ccpd2020_acc:>6.2f}%           {total_acc:>6.2f}%     {avg_time:>6.2f}ms")
    
    # 保存错误案例
    output_dir = "/home/ubuntu/e2e-pecp-pdp/mmengine-lite/work_dirs/ccpd_eval_new"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'errors.json'), 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    # 保存结果摘要
    summary = {
        'original': {
            'ccpd2019_acc': sum(results['original']['ccpd2019']) / len(results['original']['ccpd2019']) * 100 if results['original']['ccpd2019'] else 0,
            'ccpd2020_acc': sum(results['original']['ccpd2020']) / len(results['original']['ccpd2020']) * 100 if results['original']['ccpd2020'] else 0,
            'ccpd2019_samples': len(results['original']['ccpd2019']),
            'ccpd2020_samples': len(results['original']['ccpd2020']),
            'avg_time_ms': float(np.mean(results['original']['times'])) if results['original']['times'] else 0,
        },
        'ppocrv3': {
            'ccpd2019_acc': sum(results['ppocrv3']['ccpd2019']) / len(results['ppocrv3']['ccpd2019']) * 100 if results['ppocrv3']['ccpd2019'] else 0,
            'ccpd2020_acc': sum(results['ppocrv3']['ccpd2020']) / len(results['ppocrv3']['ccpd2020']) * 100 if results['ppocrv3']['ccpd2020'] else 0,
            'ccpd2019_samples': len(results['ppocrv3']['ccpd2019']),
            'ccpd2020_samples': len(results['ppocrv3']['ccpd2020']),
            'avg_time_ms': float(np.mean(results['ppocrv3']['times'])) if results['ppocrv3']['times'] else 0,
        },
        'finetuned': {
            'ccpd2019_acc': sum(results['finetuned']['ccpd2019']) / len(results['finetuned']['ccpd2019']) * 100 if results['finetuned']['ccpd2019'] else 0,
            'ccpd2020_acc': sum(results['finetuned']['ccpd2020']) / len(results['finetuned']['ccpd2020']) * 100 if results['finetuned']['ccpd2020'] else 0,
            'ccpd2019_samples': len(results['finetuned']['ccpd2019']),
            'ccpd2020_samples': len(results['finetuned']['ccpd2020']),
            'avg_time_ms': float(np.mean(results['finetuned']['times'])) if results['finetuned']['times'] else 0,
        }
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("  - errors.json: Error cases for analysis")
    print("  - summary.json: Accuracy summary")


if __name__ == "__main__":
    main()
