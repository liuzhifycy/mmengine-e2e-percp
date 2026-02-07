#!/usr/bin/env python3
"""
轻量级车牌识别器 — 面向低成本ARM部署

从 HyperLPR3 提取独立的 ONNX 识别模型, 不依赖检测阶段.
配合 roi_plate_preprocessor.py 的 ROI 裁剪, 构成完整的:
  固定ROI裁剪 → 预处理 → ONNX识别 → 车牌号输出

特点:
  - 仅依赖 onnxruntime + opencv + numpy (ARM可用)
  - 模型大小 10MB (rpv3_mdict_160), 精度高
  - 单帧识别 ~5-15ms (x86 CPU), ARM估计 20-50ms
  - 支持全部中国车牌类型 (蓝/绿/黄/白/黑)

用法:
  # 识别单张车牌图片 (已裁剪好的车牌区域)
  python lightweight_plate_recognizer.py --image plate_crop.jpg

  # 在 CCPD 预裁剪测试集上跑精度评估
  python lightweight_plate_recognizer.py --eval-ccpd data/ccpd/processed/recognition/test

  # 端到端: 从原图 + ROI 直接出识别结果
  python lightweight_plate_recognizer.py --e2e --image full_image.jpg --roi 520,384,1069,497

  # 在 test_pic 上完整演示 (HyperLPR3定位 + 独立识别)
  python lightweight_plate_recognizer.py --demo

  # 在 CCPD 全图上端到端评估 (ROI裁剪 + 识别)
  python lightweight_plate_recognizer.py --eval-e2e data/ccpd/combined --num-samples 500
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ============================================================================
# 字符表 (与 HyperLPR3 完全一致)
# ============================================================================
PLATE_CHARS = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民",
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏",
    "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁",
    "黑", "领", "使", "澳",
]

# 默认 ONNX 模型路径
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.hyperlpr3/20230229/onnx/rpv3_mdict_160_r3.onnx"
)


# ============================================================================
# 图像预处理 (与 HyperLPR3 的 encode_images 一致)
# ============================================================================
def encode_image_for_rec(image: np.ndarray, target_h: int = 48,
                         target_w: int = 160) -> np.ndarray:
    """
    将车牌图片编码为模型输入张量

    步骤:
      1. 按宽高比等比缩放到 target_h 高度
      2. 右侧零填充到 target_w 宽度
      3. 归一化到 [-1, 1]
      4. HWC -> CHW, 增加 batch 维度

    Args:
        image: BGR 车牌图像 (任意尺寸, 3通道)
        target_h: 模型输入高度 (48)
        target_w: 模型输入宽度 (160)

    Returns:
        np.ndarray: shape (1, 3, target_h, target_w), float32
    """
    imgC = 3
    h, w = image.shape[:2]
    ratio = w / float(h)
    resized_w = max(int(math.ceil(target_h * ratio)), 48)
    resized_w = min(resized_w, target_w)

    resized = cv2.resize(image, (resized_w, target_h))
    resized = resized.astype(np.float32)

    # 归一化: [0, 255] -> [-1, 1]
    resized = (resized.transpose((2, 0, 1)) - 127.5) / 127.5

    # 右侧零填充
    padded = np.zeros((imgC, target_h, target_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized

    return np.expand_dims(padded, 0)


# ============================================================================
# CTC 解码
# ============================================================================
def ctc_decode(logits: np.ndarray, chars: List[str] = PLATE_CHARS,
               blank_idx: int = 0) -> Tuple[str, float]:
    """
    CTC greedy 解码

    Args:
        logits: shape (T, C) 或 (1, T, C), 模型输出 logits
        chars: 字符表
        blank_idx: CTC blank 的索引

    Returns:
        (plate_text, confidence)
    """
    if logits.ndim == 3:
        logits = logits[0]  # (T, C)

    # Greedy: 取每个时间步的 argmax
    indices = np.argmax(logits, axis=1)  # (T,)
    probs = np.max(logits, axis=1)       # (T,)

    # 去重 + 去 blank
    char_list = []
    conf_list = []
    prev_idx = -1
    for t in range(len(indices)):
        idx = int(indices[t])
        if idx == blank_idx:
            prev_idx = idx
            continue
        if idx == prev_idx:
            continue
        if idx < len(chars):
            char_list.append(chars[idx])
        conf_list.append(float(probs[t]))
        prev_idx = idx

    text = ''.join(char_list)
    confidence = float(np.mean(conf_list)) if conf_list else 0.0
    return text, confidence


# ============================================================================
# 轻量识别器
# ============================================================================
class LightweightPlateRecognizer:
    """
    独立的轻量级车牌字符识别器

    仅加载识别模型 (~10MB ONNX), 不需要检测/分类模型.
    输入: 已裁剪的车牌图片 (任意尺寸, BGR)
    输出: (车牌号文本, 置信度)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 use_gpu: bool = False):
        """
        Args:
            model_path: ONNX 模型路径
            use_gpu: 是否使用 GPU (ARM 上一般 False)
        """
        import onnxruntime as ort
        ort.set_default_logger_severity(3)  # 抑制警告

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu \
            else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 48, 160]
        self.target_h = self.input_shape[2]  # 48
        self.target_w = self.input_shape[3]  # 160

        # 预热: 跑一次空推理
        dummy = np.zeros((1, 3, self.target_h, self.target_w), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        识别单张车牌图片

        Args:
            plate_image: BGR 车牌图像 (已裁剪, 任意尺寸)

        Returns:
            (plate_text, confidence)
        """
        if plate_image is None or plate_image.size == 0:
            return '', 0.0

        # 预处理
        blob = encode_image_for_rec(plate_image, self.target_h, self.target_w)

        # 推理
        output = self.session.run([self.output_name], {self.input_name: blob})
        logits = output[0]  # (1, T, C)

        # 解码
        text, conf = ctc_decode(logits)
        return text, conf

    def recognize_batch(self, images: List[np.ndarray],
                        batch_size: int = 1) -> List[Tuple[str, float]]:
        """
        批量识别 (当前逐张, 未来可做 batch 推理)
        """
        results = []
        for img in images:
            results.append(self.recognize(img))
        return results

    def __call__(self, plate_image: np.ndarray) -> Tuple[str, float]:
        return self.recognize(plate_image)


# ============================================================================
# 端到端管线: ROI裁剪 + 预处理 + 识别
# ============================================================================
class E2EPlateRecognitionPipeline:
    """
    端到端车牌识别管线 (无检测, 适用于固定摄像头)

    流程: 原图 → ROI裁剪 → 预处理 → ONNX识别 → 车牌号

    依赖 roi_plate_preprocessor.py 中的预处理器
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 preprocess_mode: str = 'simple'):
        """
        Args:
            model_path: ONNX 模型路径
            preprocess_mode: 'simple' 或 'precise'
        """
        self.recognizer = LightweightPlateRecognizer(model_path)

        # 导入预处理器
        from roi_plate_preprocessor import SimpleROIPreprocessor, PreciseROIPreprocessor
        if preprocess_mode == 'precise':
            self.preprocessor = PreciseROIPreprocessor(target_size=(160, 48))
        else:
            self.preprocessor = SimpleROIPreprocessor(target_size=(160, 48))
        self.mode = preprocess_mode

    def run(self, image: np.ndarray,
            roi: Tuple[int, int, int, int]) -> dict:
        """
        端到端识别

        Args:
            image: 原始完整图像 (BGR)
            roi: (x1, y1, x2, y2) 车牌所在区域

        Returns:
            dict: {
                'plate_text': 识别的车牌号,
                'confidence': 置信度,
                'preprocess_ms': 预处理耗时,
                'recognize_ms': 识别耗时,
                'total_ms': 总耗时,
            }
        """
        t_total = time.perf_counter()

        # Step 1: ROI裁剪+预处理
        t0 = time.perf_counter()
        result = self.preprocessor.process(image, roi)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        if 'error' in result:
            return {
                'plate_text': '',
                'confidence': 0.0,
                'preprocess_ms': preprocess_ms,
                'recognize_ms': 0,
                'total_ms': (time.perf_counter() - t_total) * 1000,
                'error': result['error'],
            }

        # 使用彩色裁剪图 (识别模型接收彩色)
        if self.mode == 'precise' and 'plate_corrected' in result:
            plate_img = result['plate_corrected']
        else:
            plate_img = result['plate_color']

        # Step 2: 识别
        t1 = time.perf_counter()
        plate_text, confidence = self.recognizer.recognize(plate_img)
        recognize_ms = (time.perf_counter() - t1) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return {
            'plate_text': plate_text,
            'confidence': confidence,
            'preprocess_ms': round(preprocess_ms, 2),
            'recognize_ms': round(recognize_ms, 2),
            'total_ms': round(total_ms, 2),
        }

    def __call__(self, image, roi):
        return self.run(image, roi)


# ============================================================================
# 评估: CCPD 预裁剪测试集
# ============================================================================
def eval_recognition_accuracy(test_dir: str, model_path: str = DEFAULT_MODEL_PATH,
                              num_samples: int = 0,
                              output_file: str = None):
    """
    在 CCPD 预裁剪测试集上评估纯识别精度

    测试集目录结构: test_dir/ 下的图片, 文件名格式 {车牌号}_{id}.jpg
    例如: 京ADG4380_2728.jpg

    评估指标:
      - 完全匹配准确率 (exact match)
      - 字符级准确率
      - 平均推理耗时
    """
    recognizer = LightweightPlateRecognizer(model_path)

    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if num_samples > 0:
        image_files = image_files[:num_samples]

    total = len(image_files)
    exact_match = 0
    char_correct = 0
    char_total = 0
    times = []
    errors = []

    print(f'\n{"="*60}')
    print(f'车牌识别精度评估')
    print(f'测试集: {test_dir} ({total} 张)')
    print(f'模型: {model_path}')
    print(f'{"="*60}\n')

    for idx, fname in enumerate(image_files):
        # 从文件名提取 GT
        stem = Path(fname).stem
        parts = stem.rsplit('_', 1)
        gt_text = parts[0] if len(parts) >= 2 else stem

        # 读取并识别
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        t0 = time.perf_counter()
        pred_text, conf = recognizer.recognize(image)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        # 统计
        if pred_text == gt_text:
            exact_match += 1
        else:
            errors.append((fname, gt_text, pred_text, conf))

        # 字符级准确率
        for i in range(min(len(gt_text), len(pred_text))):
            if gt_text[i] == pred_text[i]:
                char_correct += 1
        char_total += len(gt_text)

        if (idx + 1) % 500 == 0:
            print(f'  进度: {idx+1}/{total}, '
                  f'当前准确率: {exact_match/(idx+1)*100:.1f}%')

    # 统计结果
    times = np.array(times)
    exact_acc = exact_match / total * 100 if total > 0 else 0

    print(f'\n{"="*60}')
    print(f'评估结果')
    print(f'{"="*60}')
    print(f'  总样本数: {total}')
    print(f'  完全匹配: {exact_match}/{total} ({exact_acc:.1f}%)')
    if char_total > 0:
        print(f'  字符准确率: {char_correct}/{char_total} '
              f'({char_correct/char_total*100:.1f}%)')
    print(f'\n  推理耗时:')
    print(f'    平均: {times.mean():.2f} ms')
    print(f'    中位: {np.median(times):.2f} ms')
    print(f'    P95:  {np.percentile(times, 95):.2f} ms')
    print(f'    P99:  {np.percentile(times, 99):.2f} ms')

    # 显示部分错误
    if errors:
        print(f'\n  部分错误 (前20个):')
        for fname, gt, pred, conf in errors[:20]:
            mark = ''
            # 标记是哪些位错了
            diff = []
            for i in range(max(len(gt), len(pred))):
                g = gt[i] if i < len(gt) else '_'
                p = pred[i] if i < len(pred) else '_'
                if g != p:
                    diff.append(f'{i}:{g}→{p}')
            mark = ', '.join(diff)
            print(f'    {fname}: GT={gt} PRED={pred} '
                  f'(conf={conf:.3f}) [{mark}]')

    # 保存详细结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results = {
            'total': total,
            'exact_match': exact_match,
            'exact_accuracy': exact_acc,
            'char_accuracy': char_correct / char_total * 100 if char_total else 0,
            'avg_time_ms': float(times.mean()),
            'median_time_ms': float(np.median(times)),
            'errors': [
                {'file': f, 'gt': g, 'pred': p, 'conf': float(c)}
                for f, g, p, c in errors
            ],
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n  详细结果保存到: {output_file}')

    return {
        'exact_accuracy': exact_acc,
        'char_accuracy': char_correct / char_total * 100 if char_total else 0,
        'avg_time_ms': float(times.mean()),
        'total': total,
        'errors': errors,
    }


# ============================================================================
# 评估: 端到端 (CCPD 全图, ROI裁剪+识别)
# ============================================================================
def eval_e2e_ccpd(data_dir: str, model_path: str = DEFAULT_MODEL_PATH,
                  preprocess_mode: str = 'simple',
                  num_samples: int = 500,
                  expand_ratio: float = 0.3,
                  output_dir: str = None):
    """
    在 CCPD 全图上端到端评估:
      标注bbox扩展 → ROI裁剪+预处理 → 识别 → 与文件名GT比较

    这模拟了实际固定摄像头场景: ROI不完全精确, 需要一定容错
    """
    # 导入预处理器
    sys.path.insert(0, os.path.dirname(__file__))
    from roi_plate_preprocessor import (SimpleROIPreprocessor, PreciseROIPreprocessor,
                                        expand_roi)

    ann_path = os.path.join(data_dir, 'test.json')
    img_dir = os.path.join(data_dir, 'test', 'images')

    with open(ann_path) as f:
        ann = json.load(f)

    # image_id -> annotation
    id2ann = {}
    for a in ann['annotations']:
        id2ann[a['image_id']] = a

    # image_id -> plate text (从文件名提取, CCPD格式)
    id2gt = {}
    for img_info in ann['images']:
        # CCPD 文件名编码了车牌信息, 但这里用 category_id 等无法直接得到
        # 需要用另一种方式: annotations 中有 plate_number 字段? 检查一下
        id2gt[img_info['id']] = img_info.get('file_name', '')

    recognizer = LightweightPlateRecognizer(model_path)
    if preprocess_mode == 'precise':
        preprocessor = PreciseROIPreprocessor(target_size=(160, 48))
    else:
        preprocessor = SimpleROIPreprocessor(target_size=(160, 48))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = min(num_samples, len(ann['images']))
    preprocess_times = []
    recognize_times = []
    total_times = []
    recognized_plates = []

    print(f'\n{"="*60}')
    print(f'端到端评估 (CCPD)')
    print(f'数据: {data_dir} ({total} 张)')
    print(f'预处理模式: {preprocess_mode}')
    print(f'ROI扩展比例: {expand_ratio}')
    print(f'{"="*60}\n')

    for idx, img_info in enumerate(ann['images'][:total]):
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        a = id2ann.get(img_info['id'])
        if a is None:
            continue

        bx, by, bw, bh = a['bbox']
        gt_bbox = (int(bx), int(by), int(bx + bw), int(by + bh))

        # 模拟粗ROI: 扩展
        coarse_roi = expand_roi(gt_bbox, ratio=expand_ratio,
                                img_shape=image.shape)

        # Step 1: 预处理
        t0 = time.perf_counter()
        prep_result = preprocessor.process(image, coarse_roi)
        t_prep = (time.perf_counter() - t0) * 1000

        if 'error' in prep_result:
            continue

        # 使用彩色图给识别模型
        if preprocess_mode == 'precise' and 'plate_corrected' in prep_result:
            plate_img = prep_result['plate_corrected']
        else:
            plate_img = prep_result['plate_color']

        # Step 2: 识别
        t1 = time.perf_counter()
        plate_text, conf = recognizer.recognize(plate_img)
        t_rec = (time.perf_counter() - t1) * 1000

        t_total = t_prep + t_rec
        preprocess_times.append(t_prep)
        recognize_times.append(t_rec)
        total_times.append(t_total)
        recognized_plates.append({
            'file': img_info['file_name'],
            'text': plate_text,
            'conf': conf,
        })

        # 保存部分可视化
        if output_dir and idx < 30:
            vis = image.copy()
            x1, y1, x2, y2 = coarse_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            gx1, gy1, gx2, gy2 = gt_bbox
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

            # 写识别结果
            label = f'{plate_text} ({conf:.2f})'
            cv2.putText(vis, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f'e2e_{idx:04d}.jpg'), vis)

            # 保存裁剪的车牌
            cv2.imwrite(
                os.path.join(output_dir, f'e2e_{idx:04d}_plate.jpg'),
                plate_img
            )

        if (idx + 1) % 100 == 0:
            print(f'  进度: {idx+1}/{total}')

    # 统计
    prep_t = np.array(preprocess_times)
    rec_t = np.array(recognize_times)
    tot_t = np.array(total_times)

    print(f'\n{"="*60}')
    print(f'端到端评估结果')
    print(f'{"="*60}')
    print(f'  有效样本: {len(total_times)}')
    print(f'\n  耗时统计:')
    print(f'    预处理: {prep_t.mean():.2f} ms (中位: {np.median(prep_t):.2f})')
    print(f'    识别:   {rec_t.mean():.2f} ms (中位: {np.median(rec_t):.2f})')
    print(f'    总计:   {tot_t.mean():.2f} ms (中位: {np.median(tot_t):.2f})')
    print(f'    FPS:    {1000/tot_t.mean():.1f}')

    print(f'\n  识别结果样例 (前20个):')
    for r in recognized_plates[:20]:
        print(f'    {r["file"]}: {r["text"]} (conf={r["conf"]:.3f})')

    if output_dir:
        # 保存所有结果
        summary = {
            'total_samples': len(total_times),
            'preprocess_mode': preprocess_mode,
            'expand_ratio': expand_ratio,
            'timing': {
                'preprocess_avg_ms': float(prep_t.mean()),
                'recognize_avg_ms': float(rec_t.mean()),
                'total_avg_ms': float(tot_t.mean()),
                'fps': float(1000 / tot_t.mean()),
            },
            'plates': recognized_plates,
        }
        summary_path = os.path.join(output_dir, 'e2e_results.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'\n  结果保存到: {output_dir}')

    return {
        'preprocess_avg_ms': float(prep_t.mean()),
        'recognize_avg_ms': float(rec_t.mean()),
        'total_avg_ms': float(tot_t.mean()),
        'fps': float(1000 / tot_t.mean()),
    }


# ============================================================================
# Demo: 在 test_pic 上完整演示
# ============================================================================
def run_demo(test_pic_dir: str, output_dir: str,
             model_path: str = DEFAULT_MODEL_PATH):
    """
    在 test_pic 图片上完整演示:
      1. 用 HyperLPR3 获取车牌位置 (仅作为GT参考)
      2. 用独立识别器直接识别裁剪结果
      3. 对比 HyperLPR3 全流程 vs 独立识别器的结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # 初始化
    recognizer = LightweightPlateRecognizer(model_path)

    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        has_lpr = True
    except ImportError:
        has_lpr = False
        print('Warning: HyperLPR3 not available, using manual ROI')

    sys.path.insert(0, os.path.dirname(__file__))
    from roi_plate_preprocessor import SimpleROIPreprocessor, PreciseROIPreprocessor, expand_roi

    simple_proc = SimpleROIPreprocessor(target_size=(160, 48))
    precise_proc = PreciseROIPreprocessor(target_size=(160, 48))

    image_files = sorted([
        f for f in os.listdir(test_pic_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print(f'No images found in {test_pic_dir}')
        return

    print(f'\n{"="*60}')
    print(f'端到端车牌识别 Demo')
    print(f'{"="*60}\n')

    for fname in image_files:
        img_path = os.path.join(test_pic_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        stem = Path(fname).stem
        print(f'\n--- {fname} ({image.shape[1]}x{image.shape[0]}) ---')

        # HyperLPR3 全流程 (参考)
        if has_lpr:
            t0 = time.perf_counter()
            lpr_results = catcher(image)
            lpr_time = (time.perf_counter() - t0) * 1000

            if lpr_results:
                lpr_text, lpr_conf, lpr_type, lpr_bbox = lpr_results[0]
                bbox = tuple(int(v) for v in lpr_bbox)
                print(f'  [HyperLPR3 全流程] {lpr_text} '
                      f'(conf={lpr_conf:.3f}, {lpr_time:.1f}ms)')
            else:
                print(f'  [HyperLPR3] 未检测到车牌')
                continue
        else:
            h, w = image.shape[:2]
            bbox = (w // 4, h // 2, w * 3 // 4, h * 3 // 4)

        # ===== 方案A: 精确ROI + 直接识别 =====
        t0 = time.perf_counter()
        crop_a = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        text_a, conf_a = recognizer.recognize(crop_a)
        time_a = (time.perf_counter() - t0) * 1000
        print(f'  [方案A] 精确ROI裁剪 + 识别: {text_a} '
              f'(conf={conf_a:.3f}, {time_a:.1f}ms)')

        # ===== 方案B: SimpleROI预处理 + 识别 =====
        coarse_roi = expand_roi(bbox, ratio=0.3, img_shape=image.shape)
        t0 = time.perf_counter()
        r_simple = simple_proc.process(image, coarse_roi)
        plate_b = r_simple['plate_color']
        text_b, conf_b = recognizer.recognize(plate_b)
        time_b = (time.perf_counter() - t0) * 1000
        print(f'  [方案B] SimpleROI + 识别: {text_b} '
              f'(conf={conf_b:.3f}, {time_b:.1f}ms)')

        # ===== 方案C: PreciseROI预处理 + 识别 =====
        t0 = time.perf_counter()
        r_precise = precise_proc.process(image, coarse_roi)
        if 'plate_corrected' in r_precise:
            plate_c = r_precise['plate_corrected']
        else:
            plate_c = r_precise['plate_color']
        text_c, conf_c = recognizer.recognize(plate_c)
        time_c = (time.perf_counter() - t0) * 1000
        print(f'  [方案C] PreciseROI + 识别: {text_c} '
              f'(conf={conf_c:.3f}, {time_c:.1f}ms, '
              f'method={r_precise.get("method", "?")})')

        # 保存可视化
        vis = image.copy()
        # 精确bbox
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (255, 0, 0), 2)
        # 粗ROI
        cx1, cy1, cx2, cy2 = coarse_roi
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        # 识别结果
        cv2.putText(vis, f'A:{text_a}', (bbox[0], bbox[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis, f'B:{text_b}', (bbox[0], bbox[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        cv2.imwrite(os.path.join(output_dir, f'{stem}_e2e.jpg'), vis)

        # 保存各方案的裁剪
        cv2.imwrite(os.path.join(output_dir, f'{stem}_cropA.jpg'), crop_a)
        cv2.imwrite(os.path.join(output_dir, f'{stem}_cropB.jpg'), plate_b)
        cv2.imwrite(os.path.join(output_dir, f'{stem}_cropC.jpg'), plate_c)

    print(f'\n所有结果保存到: {output_dir}')


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='轻量级车牌识别器 (独立ONNX, 面向ARM部署)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='识别单张车牌图片')
    group.add_argument('--eval-ccpd', type=str, metavar='DIR',
                       help='在CCPD预裁剪测试集上评估精度')
    group.add_argument('--eval-e2e', type=str, metavar='DIR',
                       help='在CCPD全图上端到端评估')
    group.add_argument('--demo', action='store_true',
                       help='在 test_pic 上完整演示')

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='ONNX模型路径')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI坐标: x1,y1,x2,y2 (配合 --image 使用)')
    parser.add_argument('--e2e', action='store_true',
                        help='端到端模式: 从原图+ROI识别')
    parser.add_argument('--preprocess-mode', type=str, default='simple',
                        choices=['simple', 'precise'],
                        help='预处理模式 (default: simple)')
    parser.add_argument('--num-samples', type=int, default=0,
                        help='评估样本数 (0=全部)')
    parser.add_argument('--expand-ratio', type=float, default=0.3,
                        help='端到端评估ROI扩展比例 (default: 0.3)')
    parser.add_argument('--output-dir', type=str, default='recognition_output',
                        help='输出目录')
    parser.add_argument('--test-pic-dir', type=str, default=None,
                        help='test_pic 目录')

    args = parser.parse_args()

    if args.image and not args.e2e:
        # 直接识别裁剪好的车牌图片
        image = cv2.imread(args.image)
        if image is None:
            print(f'Error: 无法读取 {args.image}')
            sys.exit(1)

        recognizer = LightweightPlateRecognizer(args.model)

        # 多次推理取平均
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            text, conf = recognizer.recognize(image)
            times.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(times[1:])  # 排除首次 warmup
        print(f'识别结果: {text}')
        print(f'置信度:   {conf:.4f}')
        print(f'耗时:     {avg:.2f} ms (平均, 排除首次)')
        print(f'图像尺寸: {image.shape}')

    elif args.image and args.e2e:
        # 端到端: 原图 + ROI
        if not args.roi:
            print('Error: 端到端模式需要 --roi')
            sys.exit(1)

        image = cv2.imread(args.image)
        if image is None:
            print(f'Error: 无法读取 {args.image}')
            sys.exit(1)

        roi = tuple(int(x) for x in args.roi.split(','))
        sys.path.insert(0, os.path.dirname(__file__))
        pipeline = E2EPlateRecognitionPipeline(
            args.model, args.preprocess_mode
        )
        result = pipeline.run(image, roi)
        print(f'识别结果: {result["plate_text"]}')
        print(f'置信度:   {result["confidence"]:.4f}')
        print(f'预处理:   {result["preprocess_ms"]:.2f} ms')
        print(f'识别:     {result["recognize_ms"]:.2f} ms')
        print(f'总计:     {result["total_ms"]:.2f} ms')

    elif args.eval_ccpd:
        eval_recognition_accuracy(
            args.eval_ccpd, args.model,
            num_samples=args.num_samples,
            output_file=os.path.join(args.output_dir, 'eval_results.json')
                        if args.output_dir else None
        )

    elif args.eval_e2e:
        eval_e2e_ccpd(
            args.eval_e2e, args.model,
            preprocess_mode=args.preprocess_mode,
            num_samples=args.num_samples or 500,
            expand_ratio=args.expand_ratio,
            output_dir=args.output_dir
        )

    elif args.demo:
        test_pic = args.test_pic_dir
        if test_pic is None:
            candidates = [
                'test_pic',
                '../test_pic',
                os.path.join(os.path.dirname(__file__), '..', '..', 'test_pic'),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    test_pic = c
                    break
        if test_pic is None:
            print('Error: 找不到 test_pic 目录')
            sys.exit(1)

        run_demo(test_pic, args.output_dir, args.model)


if __name__ == '__main__':
    main()
