#!/usr/bin/env python3
"""
使用 ONNX 模型评估 PIC 数据集并生成 CCPD 格式数据集

用法:
    python evaluate_onnx_dataset.py --input-dir ./PIC_dataset --output-dir ./PIC_ccpd_format
"""

import os
import sys
import cv2
import numpy as np
import math
import time
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort

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

# 分类标签
PLATE_TYPES = {0: "blue", 1: "green", 2: "yellow"}  # 蓝牌、绿牌、黄牌


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox 预处理"""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def nms(boxes, scores, iou_thresh=0.5):
    """NMS"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


class PlateRecognizerONNX:
    """基于 ONNX 的车牌识别器"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.det_size = 640
        
        det_path = os.path.join(model_dir, 'hztk_det.onnx')
        rec_path = os.path.join(model_dir, 'hztk_rec.onnx')
        cls_path = os.path.join(model_dir, 'hztk_cls.onnx')
        
        print("Loading ONNX models...")
        
        # 使用 CPU 或 CUDA provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.det_session = ort.InferenceSession(det_path, providers=providers)
        print(f"  Det model loaded: {det_path}")
        
        self.rec_session = ort.InferenceSession(rec_path, providers=providers)
        print(f"  Rec model loaded: {rec_path}")
        
        self.cls_session = ort.InferenceSession(cls_path, providers=providers)
        print(f"  Cls model loaded: {cls_path}")
        
        # 获取输入输出信息
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.cls_input_name = self.cls_session.get_inputs()[0].name
        
        # 检查输入格式 (NCHW or NHWC)
        det_input_shape = self.det_session.get_inputs()[0].shape
        print(f"  Det input shape: {det_input_shape}")
        
        # 判断是 NCHW 还是 NHWC
        if len(det_input_shape) == 4:
            if det_input_shape[1] == 3:  # NCHW
                self.input_format = 'NCHW'
            else:  # NHWC
                self.input_format = 'NHWC'
        else:
            self.input_format = 'NCHW'  # 默认
        
        print(f"  Input format: {self.input_format}")
        print("PlateRecognizerONNX initialized")
    
    def preprocess_det(self, image: np.ndarray) -> tuple:
        """检测模型预处理"""
        padded, ratio, (dw, dh) = letterbox(image, (self.det_size, self.det_size))
        
        # BGR -> RGB, 归一化
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        if self.input_format == 'NCHW':
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, 0)  # [1, C, H, W]
        else:
            img = np.expand_dims(img, 0)  # [1, H, W, C]
        
        return img, ratio, (dw, dh)
    
    def postprocess_det(self, output: np.ndarray, ratio: tuple, 
                        pad_size: tuple, orig_shape: tuple, 
                        conf_thresh: float = 0.5) -> list:
        """检测后处理"""
        output = output.squeeze()
        
        # 处理不同的输出格式
        if output.ndim == 2:
            if output.shape[0] == 5:  # [5, num_boxes]
                output = output.T  # 转为 [num_boxes, 5]
        
        dw, dh = pad_size
        orig_h, orig_w = orig_shape
        
        if output.shape[1] >= 5:
            mask = output[:, 4] > conf_thresh
        else:
            return []
        
        filtered = output[mask]
        
        if len(filtered) == 0:
            return []
        
        cx, cy, bw, bh = filtered[:,0], filtered[:,1], filtered[:,2], filtered[:,3]
        confs = filtered[:, 4]
        
        x1 = (cx - bw/2 - dw) / ratio[0]
        y1 = (cy - bh/2 - dh) / ratio[1]
        x2 = (cx + bw/2 - dw) / ratio[0]
        y2 = (cy + bh/2 - dh) / ratio[1]
        
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        valid = (x2 - x1) > 10
        x1, y1, x2, y2, confs = x1[valid], y1[valid], x2[valid], y2[valid], confs[valid]
        
        if len(x1) == 0:
            return []
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, confs, 0.5)
        
        detections = []
        for i in keep:
            detections.append((
                int(boxes[i, 0]), int(boxes[i, 1]),
                int(boxes[i, 2]), int(boxes[i, 3]),
                float(confs[i])
            ))
        
        return detections
    
    def preprocess_rec(self, plate_img: np.ndarray) -> np.ndarray:
        """识别模型预处理"""
        imgH, imgW = 48, 160
        h, w = plate_img.shape[:2]
        wh_ratio = w / float(h)
        
        max_wh_ratio = max(wh_ratio, imgW / imgH)
        target_w = int(imgH * max_wh_ratio)
        target_w = max(min(target_w, 160), 48)
        
        ratio_imgH = math.ceil(imgH * wh_ratio)
        ratio_imgH = max(ratio_imgH, 48)
        resized_w = target_w if ratio_imgH > target_w else int(ratio_imgH)
        
        resized = cv2.resize(plate_img, (resized_w, imgH))
        resized = resized.astype(np.float32)
        resized = (resized - 127.5) / 127.5
        
        # Padding
        padded = np.zeros((imgH, imgW, 3), dtype=np.float32)
        padded[:, 0:resized_w, :] = resized
        
        if self.input_format == 'NCHW':
            padded = np.transpose(padded, (2, 0, 1))  # HWC -> CHW
        
        return np.expand_dims(padded, 0)
    
    def decode_plate(self, output: np.ndarray) -> tuple:
        """解码车牌
        
        输出格式: [1, seq_len, num_classes] 或 [seq_len, num_classes]
        hztk_rec.onnx 输出: [1, 20, 78] - 20 个时间步，78 个类别
        """
        prod = output.squeeze()  # [20, 78]
        
        # hztk_rec 输出已经是 [seq_len, num_classes] 格式，不需要转置
        # 注意: seq_len=20, num_classes=78, 所以 shape[0] < shape[1]
        # 但这是正确的格式，不应该转置！
        
        indices = np.argmax(prod, axis=-1)
        max_probs = np.max(prod, axis=-1)
        
        chars, confs = [], []
        prev_idx = -1
        for i, idx in enumerate(indices):
            if idx == 0 or idx == prev_idx:
                prev_idx = idx
                continue
            if idx < len(PLATE_CHARS):
                chars.append(PLATE_CHARS[int(idx)])
                confs.append(float(max_probs[i]))
            prev_idx = idx
        
        return "".join(chars), float(np.mean(confs)) if confs else 0.0
    
    def preprocess_cls(self, plate_img: np.ndarray) -> np.ndarray:
        """分类模型预处理"""
        img = cv2.resize(plate_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        if self.input_format == 'NCHW':
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        return np.expand_dims(img, 0)
    
    def classify_plate(self, output: np.ndarray) -> tuple:
        """分类解码"""
        output = output.squeeze()
        idx = int(np.argmax(output))
        conf = float(output[idx])
        return idx, PLATE_TYPES.get(idx, "unknown"), conf
    
    def recognize(self, image: np.ndarray, conf_thresh: float = 0.5) -> list:
        """完整识别流程"""
        results = []
        
        # 检测
        det_input, ratio, pad_size = self.preprocess_det(image)
        det_output = self.det_session.run(None, {self.det_input_name: det_input})[0]
        detections = self.postprocess_det(det_output, ratio, pad_size, 
                                          image.shape[:2], conf_thresh)
        
        # 识别每个检测框
        for x1, y1, x2, y2, det_conf in detections:
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # OCR
            rec_input = self.preprocess_rec(plate_img)
            rec_output = self.rec_session.run(None, {self.rec_input_name: rec_input})[0]
            plate_number, rec_conf = self.decode_plate(rec_output)
            
            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_output = self.cls_session.run(None, {self.cls_input_name: cls_input})[0]
            cls_idx, plate_type, cls_conf = self.classify_plate(cls_output)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'det_conf': det_conf,
                'plate_number': plate_number,
                'rec_conf': rec_conf,
                'plate_type_idx': cls_idx,
                'plate_type': plate_type,
                'cls_conf': cls_conf,
            })
        
        return results


def extract_plate_from_filename(filename: str) -> str:
    """从文件名提取车牌号真值
    格式: 时间戳_车牌号.jpg
    例如: 1209095112874_皖AD55283.jpg -> 皖AD55283
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])  # 处理车牌号中可能有下划线的情况
    return None


def is_green_plate_by_gt(plate_number: str) -> bool:
    """根据车牌号判断是否为新能源车牌（绿牌）
    新能源车牌规则：
    - 小型新能源车牌：省份简称+字母+D/F+5位字符 或 省份简称+字母+5位字符+D/F
    - 大型新能源车牌：省份简称+字母+5位字符+D/F
    - 通常第3位或最后一位是D或F
    - 车牌长度为8位（普通车牌7位）
    """
    if not plate_number:
        return False
    
    # 新能源车牌通常是8位
    if len(plate_number) == 8:
        return True
    
    # 也有一些7位的新能源临时方案，第3位是D/F
    if len(plate_number) >= 3:
        if plate_number[2] in ['D', 'F']:
            return True
    
    return False


def generate_ccpd_annotation(image_path: str, gt_plate: str, detection: dict, 
                            image_shape: tuple) -> dict:
    """生成类CCPD格式的标注
    
    返回格式：
    {
        'image_path': 原始图片路径,
        'gt_plate': 真值车牌号,
        'pred_plate': 预测车牌号,
        'is_correct': 预测是否正确,
        'bbox': [x1, y1, x2, y2],
        'bbox_normalized': [x1/w, y1/h, x2/w, y2/h],
        'det_conf': 检测置信度,
        'rec_conf': 识别置信度,
        'plate_type_pred': 预测的车牌类型 (blue/green/yellow),
        'plate_type_gt': 真值车牌类型,
        'cls_conf': 分类置信度,
        'image_size': [w, h],
    }
    """
    h, w = image_shape[:2]
    bbox = detection['bbox']
    
    # 判断真值的车牌类型
    gt_is_green = is_green_plate_by_gt(gt_plate)
    gt_type = 'green' if gt_is_green else 'blue'  # 简化处理，默认非绿牌为蓝牌
    
    return {
        'image_path': image_path,
        'gt_plate': gt_plate,
        'pred_plate': detection['plate_number'],
        'is_correct': detection['plate_number'] == gt_plate,
        'bbox': bbox,
        'bbox_normalized': [
            bbox[0] / w, bbox[1] / h, 
            bbox[2] / w, bbox[3] / h
        ],
        'det_conf': detection['det_conf'],
        'rec_conf': detection['rec_conf'],
        'plate_type_pred': detection['plate_type'],
        'plate_type_pred_idx': detection['plate_type_idx'],
        'plate_type_gt': gt_type,
        'cls_conf': detection['cls_conf'],
        'image_size': [w, h],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX models on PIC dataset')
    parser.add_argument('--input-dir', type=str, required=True, 
                        help='Input directory containing images (PIC_dataset)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for CCPD format dataset')
    parser.add_argument('--model-dir', type=str, 
                        default='/home/ubuntu/e2e-pecp-pdp/mmengine-lite/e2e_hztk_deploy_package',
                        help='ONNX models directory')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process')
    
    args = parser.parse_args()
    
    # 初始化识别器
    recognizer = PlateRecognizerONNX(args.model_dir)
    
    # 收集所有图片
    image_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))
    
    print(f"Found {len(image_files)} images")
    
    if args.max_samples:
        image_files = image_files[:args.max_samples]
        print(f"Processing first {len(image_files)} images")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'visualizations', 'correct'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'visualizations', 'wrong'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'visualizations', 'not_detected'), exist_ok=True)
    
    # 统计
    stats = {
        'total': 0,
        'detected': 0,
        'correct': 0,
        'partial_correct': 0,  # 至少5个字符正确
        'not_detected': 0,
        'plate_type_correct': 0,
        'by_type': {
            'blue': {'total': 0, 'correct': 0},
            'green': {'total': 0, 'correct': 0},
        }
    }
    
    annotations = []
    
    # 处理每张图片
    for img_path in tqdm(image_files, desc="Processing"):
        filename = os.path.basename(img_path)
        gt_plate = extract_plate_from_filename(filename)
        
        if not gt_plate:
            continue
        
        stats['total'] += 1
        
        # 统计车牌类型
        gt_is_green = is_green_plate_by_gt(gt_plate)
        gt_type = 'green' if gt_is_green else 'blue'
        stats['by_type'][gt_type]['total'] += 1
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            stats['not_detected'] += 1
            continue
        
        # 推理
        results = recognizer.recognize(image, args.conf_thresh)
        
        if not results:
            stats['not_detected'] += 1
            
            # 保存未检测到的图片
            if args.save_vis:
                vis_path = os.path.join(args.output_dir, 'visualizations', 'not_detected', filename)
                shutil.copy(img_path, vis_path)
            
            continue
        
        stats['detected'] += 1
        
        # 取第一个检测结果（假设每张图只有一个车牌）
        detection = results[0]
        pred_plate = detection['plate_number']
        
        # 判断识别是否正确
        is_correct = (pred_plate == gt_plate)
        
        if is_correct:
            stats['correct'] += 1
            stats['by_type'][gt_type]['correct'] += 1
        else:
            # 部分正确检查
            if pred_plate and len(pred_plate) >= 5 and len(gt_plate) >= 5:
                matches = sum(1 for p, g in zip(pred_plate, gt_plate) if p == g)
                if matches >= 5:
                    stats['partial_correct'] += 1
        
        # 检查车牌类型分类是否正确
        pred_is_green = detection['plate_type'] == 'green'
        if gt_is_green == pred_is_green:
            stats['plate_type_correct'] += 1
        
        # 生成标注
        annotation = generate_ccpd_annotation(
            img_path, gt_plate, detection, image.shape
        )
        annotations.append(annotation)
        
        # 复制图片到输出目录
        new_filename = f"{gt_plate}_{os.path.basename(img_path)}"
        new_path = os.path.join(args.output_dir, 'images', new_filename)
        shutil.copy(img_path, new_path)
        
        # 保存可视化
        if args.save_vis:
            # 绘制检测框和文本
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加文本
            label = f"Pred: {pred_plate}"
            cv2.putText(image, label, (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            gt_label = f"GT: {gt_plate}"
            cv2.putText(image, gt_label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            
            status = "correct" if is_correct else "wrong"
            vis_path = os.path.join(args.output_dir, 'visualizations', status, filename)
            cv2.imwrite(vis_path, image)
    
    # 计算指标
    total = stats['total']
    metrics = {
        'total_images': total,
        'detected': stats['detected'],
        'detection_rate': stats['detected'] / total if total > 0 else 0,
        'correct': stats['correct'],
        'recognition_accuracy': stats['correct'] / total if total > 0 else 0,
        'partial_correct': stats['partial_correct'],
        'partial_accuracy': (stats['correct'] + stats['partial_correct']) / total if total > 0 else 0,
        'not_detected': stats['not_detected'],
        'plate_type_accuracy': stats['plate_type_correct'] / stats['detected'] if stats['detected'] > 0 else 0,
        'by_type': {}
    }
    
    for ptype, pstats in stats['by_type'].items():
        if pstats['total'] > 0:
            metrics['by_type'][ptype] = {
                'total': pstats['total'],
                'correct': pstats['correct'],
                'accuracy': pstats['correct'] / pstats['total']
            }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total images:        {metrics['total_images']}")
    print(f"Detected:            {metrics['detected']} ({metrics['detection_rate']*100:.2f}%)")
    print(f"Not detected:        {metrics['not_detected']}")
    print(f"Correct:             {metrics['correct']} ({metrics['recognition_accuracy']*100:.2f}%)")
    print(f"Partial correct:     {metrics['partial_correct']}")
    print(f"Combined accuracy:   {metrics['partial_accuracy']*100:.2f}%")
    print(f"Plate type accuracy: {metrics['plate_type_accuracy']*100:.2f}%")
    print()
    print("By plate type:")
    for ptype, pmetrics in metrics['by_type'].items():
        print(f"  {ptype}: {pmetrics['correct']}/{pmetrics['total']} ({pmetrics['accuracy']*100:.2f}%)")
    print("=" * 60)
    
    # 保存标注和指标
    annotation_file = os.path.join(args.output_dir, 'annotations.json')
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"\nAnnotations saved to: {annotation_file}")
    
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")
    
    # 生成 CCPD 格式的标签文件
    ccpd_labels_file = os.path.join(args.output_dir, 'labels.txt')
    with open(ccpd_labels_file, 'w', encoding='utf-8') as f:
        for ann in annotations:
            # 格式: image_path gt_plate pred_plate is_correct bbox plate_type
            line = (f"{ann['image_path']}\t{ann['gt_plate']}\t{ann['pred_plate']}\t"
                   f"{int(ann['is_correct'])}\t{ann['bbox']}\t{ann['plate_type_gt']}\t{ann['plate_type_pred']}\n")
            f.write(line)
    print(f"Labels saved to: {ccpd_labels_file}")


if __name__ == '__main__':
    main()
