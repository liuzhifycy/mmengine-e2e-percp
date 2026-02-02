#!/usr/bin/env python
"""
E2E_HZTK TensorRT 部署推理脚本

使用 TensorRT 引擎进行车牌识别推理，支持：
- 车牌检测 (y5fu_640x / y5fu_320x)
- 车牌识别 OCR (rpv3_mdict_160)
- 车牌分类 (litemodel_cls_96x)

使用方法:
    # 单张图片推理
    python tools/plate_recognition/deploy_e2e_hztk_trt.py \
        --image path/to/image.jpg \
        --engine-dir exports/e2e_hztk
    
    # 批量推理
    python tools/plate_recognition/deploy_e2e_hztk_trt.py \
        --image-dir path/to/images \
        --engine-dir exports/e2e_hztk \
        --output-dir output/
    
    # 性能测试
    python tools/plate_recognition/deploy_e2e_hztk_trt.py \
        --benchmark \
        --engine-dir exports/e2e_hztk

依赖:
    pip install tensorrt pycuda opencv-python numpy
"""

import argparse
import math
import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available, falling back to ONNX Runtime")


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

# 车牌类型
PLATE_TYPES = {
    0: "蓝牌",
    1: "绿牌", 
    2: "黄牌",
}


class TRTEngine:
    """TensorRT 引擎封装类"""
    
    def __init__(self, engine_path: str):
        """
        加载 TensorRT 引擎
        
        Args:
            engine_path: TensorRT 引擎文件路径
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # 分配内存
        self.input_size = int(np.prod(self.input_shape) * np.float32().itemsize)
        self.output_size = int(np.prod(self.output_shape) * np.float32().itemsize)
        
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        self.stream = cuda.Stream()
        
        print(f"  Input: {self.input_name} {list(self.input_shape)}")
        print(f"  Output: {self.output_name} {list(self.output_shape)}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_data: 输入数据 (NCHW 格式)
            
        Returns:
            输出数据
        """
        # 确保数据类型和形状正确
        input_data = np.ascontiguousarray(input_data.astype(np.float32))
        
        # 复制输入到 GPU
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # 设置输入输出地址
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 复制输出到 CPU
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        
        self.stream.synchronize()
        
        return output_data
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


class PlateRecognizerTRT:
    """基于 TensorRT 的车牌识别器"""
    
    def __init__(self, engine_dir: str, use_640x: bool = True):
        """
        初始化识别器
        
        Args:
            engine_dir: TensorRT 引擎目录
            use_640x: 是否使用 640x640 检测模型（False 使用 320x320）
        """
        self.engine_dir = engine_dir
        
        # 加载检测引擎
        det_engine = "y5fu_640x_fp16.engine" if use_640x else "y5fu_320x_fp16.engine"
        det_path = os.path.join(engine_dir, det_engine)
        if not os.path.exists(det_path):
            det_path = os.path.join(engine_dir, det_engine.replace("_fp16", "_sim_fp16"))
        self.det_engine = TRTEngine(det_path)
        self.det_size = 640 if use_640x else 320
        
        # 加载识别引擎
        rec_path = os.path.join(engine_dir, "rpv3_mdict_160_fp16.engine")
        if not os.path.exists(rec_path):
            rec_path = os.path.join(engine_dir, "rpv3_mdict_160_r3_fp16.engine")
        self.rec_engine = TRTEngine(rec_path)
        
        # 加载分类引擎
        cls_path = os.path.join(engine_dir, "litemodel_cls_96x_fp16.engine")
        self.cls_engine = TRTEngine(cls_path)
        
        print(f"\nPlateRecognizerTRT initialized with engines from: {engine_dir}")
    
    def preprocess_det(self, image: np.ndarray) -> tuple:
        """
        检测模型预处理
        
        Args:
            image: BGR 图像
            
        Returns:
            (预处理后的图像, 缩放比例, padding)
        """
        h, w = image.shape[:2]
        
        # 等比例缩放
        scale = min(self.det_size / h, self.det_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Padding 到目标尺寸
        pad_h = self.det_size - new_h
        pad_w = self.det_size - new_w
        
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # BGR to RGB, HWC to CHW, normalize
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        
        return img, scale, (pad_w, pad_h)
    
    def postprocess_det(self, output: np.ndarray, scale: float, 
                        orig_shape: tuple, conf_thresh: float = 0.5) -> list:
        """
        检测模型后处理
        
        Args:
            output: 模型输出
            scale: 缩放比例
            orig_shape: 原始图像尺寸 (H, W)
            conf_thresh: 置信度阈值
            
        Returns:
            检测框列表 [(x1, y1, x2, y2, conf), ...]
        """
        # YOLOv5 输出格式: [1, num_anchors, 6] (x, y, w, h, conf, cls)
        output = output.squeeze()
        
        detections = []
        for det in output:
            conf = det[4]
            if conf < conf_thresh:
                continue
            
            cx, cy, w, h = det[:4]
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale
            
            # 裁剪到图像边界
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        
        # NMS
        if len(detections) > 1:
            detections = self.nms(detections, 0.5)
        
        return detections
    
    def nms(self, detections: list, iou_thresh: float) -> list:
        """简单的 NMS 实现"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self.compute_iou(best[:4], d[:4]) < iou_thresh
            ]
        
        return keep
    
    def compute_iou(self, box1: tuple, box2: tuple) -> float:
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def preprocess_rec(self, plate_img: np.ndarray) -> np.ndarray:
        """
        识别模型预处理
        
        Args:
            plate_img: 裁剪的车牌图像 (BGR)
            
        Returns:
            预处理后的图像 [1, 3, 48, 160]
        """
        # Target dimensions
        imgH, imgW = 48, 160
        limited_max_width = 160
        limited_min_width = 48
        
        h, w = plate_img.shape[:2]
        wh_ratio = w / float(h)
        
        # Calculate dynamic width while maintaining aspect ratio
        max_wh_ratio = max(wh_ratio, imgW / imgH)
        target_w = int(imgH * max_wh_ratio)
        target_w = max(min(target_w, limited_max_width), limited_min_width)
        
        # Calculate resized width
        ratio_imgH = math.ceil(imgH * wh_ratio)
        ratio_imgH = max(ratio_imgH, limited_min_width)
        if ratio_imgH > target_w:
            resized_w = target_w
        else:
            resized_w = int(ratio_imgH)
        
        # Resize image (BGR input, no color conversion)
        resized_image = cv2.resize(plate_img, (resized_w, imgH))
        
        # Normalize: (img - 127.5) / 127.5, then transpose to CHW
        resized_image = resized_image.astype(np.float32)
        resized_image = (resized_image.transpose((2, 0, 1)) - 127.5) / 127.5
        
        # Zero-padding to target width
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return np.expand_dims(padding_im, 0)
    
    def decode_plate(self, output: np.ndarray) -> tuple:
        """
        解码车牌识别结果
        
        Args:
            output: 模型输出 [1, seq_len, num_classes]
            
        Returns:
            (车牌号, 置信度)
        """
        # output shape: [1, seq_len, num_classes] -> [seq_len, num_classes]
        prod = output.squeeze()
        
        # CTC 解码
        indices = np.argmax(prod, axis=-1)  # [seq_len]
        max_probs = np.max(prod, axis=-1)   # [seq_len]
        
        # CTC 解码（去重和去 blank）
        # ignored_tokens = [0] (blank)
        plate_chars = []
        confidences = []
        prev_idx = -1
        
        for i, idx in enumerate(indices):
            # Skip blank (idx == 0) and duplicates
            if idx == 0:
                prev_idx = idx
                continue
            if idx == prev_idx:
                continue
            
            if idx < len(PLATE_CHARS):
                plate_chars.append(PLATE_CHARS[int(idx)])
                confidences.append(float(max_probs[i]))
            prev_idx = idx
        
        plate_number = "".join(plate_chars)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return plate_number, avg_conf
    
    def preprocess_cls(self, plate_img: np.ndarray) -> np.ndarray:
        """分类模型预处理"""
        img = cv2.resize(plate_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img
    
    def classify_plate(self, output: np.ndarray) -> tuple:
        """解码分类结果"""
        output = output.squeeze()
        idx = np.argmax(output)
        conf = float(output[idx])
        plate_type = PLATE_TYPES.get(idx, "未知")
        return plate_type, conf
    
    def recognize(self, image: np.ndarray, conf_thresh: float = 0.5) -> list:
        """
        完整的车牌识别流程
        
        Args:
            image: BGR 图像
            conf_thresh: 检测置信度阈值
            
        Returns:
            识别结果列表
        """
        results = []
        
        # 1. 检测
        det_input, scale, padding = self.preprocess_det(image)
        det_output = self.det_engine.infer(det_input)
        detections = self.postprocess_det(det_output, scale, image.shape[:2], conf_thresh)
        
        # 2. 对每个检测框进行识别
        for x1, y1, x2, y2, det_conf in detections:
            # 裁剪车牌区域
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # 识别
            rec_input = self.preprocess_rec(plate_img)
            rec_output = self.rec_engine.infer(rec_input)
            plate_number, rec_conf = self.decode_plate(rec_output)
            
            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_output = self.cls_engine.infer(cls_input)
            plate_type, cls_conf = self.classify_plate(cls_output)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'det_conf': det_conf,
                'plate_number': plate_number,
                'rec_conf': rec_conf,
                'plate_type': plate_type,
                'cls_conf': cls_conf,
            })
        
        return results
    
    def benchmark(self, num_iters: int = 100, warmup: int = 10) -> dict:
        """
        性能测试
        
        Args:
            num_iters: 测试迭代次数
            warmup: 预热次数
            
        Returns:
            性能指标
        """
        # 生成随机输入
        det_input = np.random.randn(1, 3, self.det_size, self.det_size).astype(np.float32)
        rec_input = np.random.randn(1, 3, 48, 160).astype(np.float32)
        cls_input = np.random.randn(1, 3, 96, 96).astype(np.float32)
        
        # 预热
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            self.det_engine.infer(det_input)
            self.rec_engine.infer(rec_input)
            self.cls_engine.infer(cls_input)
        
        # 测试检测
        print(f"Benchmarking detection ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.det_engine.infer(det_input)
        det_time = (time.time() - start) / num_iters * 1000
        
        # 测试识别
        print(f"Benchmarking recognition ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.rec_engine.infer(rec_input)
        rec_time = (time.time() - start) / num_iters * 1000
        
        # 测试分类
        print(f"Benchmarking classification ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.cls_engine.infer(cls_input)
        cls_time = (time.time() - start) / num_iters * 1000
        
        total_time = det_time + rec_time + cls_time
        
        return {
            'detection_ms': det_time,
            'recognition_ms': rec_time,
            'classification_ms': cls_time,
            'total_ms': total_time,
            'fps': 1000 / total_time,
        }


def visualize_result(image: np.ndarray, results: list, output_path: str = None):
    """可视化识别结果"""
    from PIL import Image, ImageDraw, ImageFont
    
    # 获取中文字体
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 24)
                break
            except:
                continue
    
    # 转换为 PIL 图像
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    for res in results:
        x1, y1, x2, y2 = res['bbox']
        
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # 标签
        label = f"{res['plate_number']} ({res['plate_type']})"
        
        # 背景
        if font:
            bbox = draw.textbbox((x1, y1 - 30), label, font=font)
            draw.rectangle(bbox, fill=(255, 255, 255))
            draw.text((x1, y1 - 30), label, font=font, fill=(0, 128, 0))
    
    # 转回 OpenCV 格式
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved: {output_path}")
    
    return result_img


def main():
    parser = argparse.ArgumentParser(description='E2E_HZTK TensorRT Deployment')
    parser.add_argument('--engine-dir', type=str, default='exports/e2e_hztk',
                        help='TensorRT engine directory')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Image directory for batch inference')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--use-320x', action='store_true', help='Use 320x320 detection model (faster)')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    if not TRT_AVAILABLE:
        print("Error: TensorRT is not available")
        sys.exit(1)
    
    # 初始化识别器
    recognizer = PlateRecognizerTRT(args.engine_dir, use_640x=not args.use_320x)
    
    if args.benchmark:
        # 性能测试
        print("\n" + "=" * 50)
        print("Performance Benchmark")
        print("=" * 50)
        
        metrics = recognizer.benchmark()
        
        print(f"\nResults:")
        print(f"  Detection:      {metrics['detection_ms']:.2f} ms")
        print(f"  Recognition:    {metrics['recognition_ms']:.2f} ms")
        print(f"  Classification: {metrics['classification_ms']:.2f} ms")
        print(f"  Total:          {metrics['total_ms']:.2f} ms")
        print(f"  FPS:            {metrics['fps']:.1f}")
    
    elif args.image:
        # 单张图片推理
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Cannot read image: {args.image}")
            sys.exit(1)
        
        results = recognizer.recognize(image, args.conf_thresh)
        
        print(f"\nResults for {args.image}:")
        for i, res in enumerate(results):
            print(f"  [{i+1}] {res['plate_number']} ({res['plate_type']})")
            print(f"      bbox: {res['bbox']}, det_conf: {res['det_conf']:.3f}, rec_conf: {res['rec_conf']:.3f}")
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, os.path.basename(args.image))
            visualize_result(image, results, output_path)
    
    elif args.image_dir:
        # 批量推理
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        total_time = 0
        total_images = 0
        
        for img_file in image_files[:100]:  # 限制最多 100 张
            img_path = os.path.join(args.image_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            start = time.time()
            results = recognizer.recognize(image, args.conf_thresh)
            elapsed = time.time() - start
            
            total_time += elapsed
            total_images += 1
            
            print(f"{img_file}: ", end="")
            if results:
                print(f"{results[0]['plate_number']} ({elapsed*1000:.1f}ms)")
            else:
                print(f"No plate detected ({elapsed*1000:.1f}ms)")
            
            if args.output_dir and results:
                output_path = os.path.join(args.output_dir, img_file)
                visualize_result(image, results, output_path)
        
        if total_images > 0:
            avg_time = total_time / total_images * 1000
            print(f"\nProcessed {total_images} images")
            print(f"Average time: {avg_time:.1f} ms/image ({1000/avg_time:.1f} FPS)")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()