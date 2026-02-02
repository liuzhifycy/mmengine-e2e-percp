#!/usr/bin/env python3
"""
E2E_HZTK RK3588 NPU 部署推理脚本

使用 RKNN 模型在 RK3588 板子上进行车牌识别推理，支持：
- 车牌检测 (hztk_det.rknn)
- 车牌识别 OCR (hztk_rec.rknn - CRNN)
- 车牌分类 (hztk_cls.rknn)

使用方法:
    # 单张图片推理
    python deploy_rknn_board.py --image ./test_pic/20260201-145636.jpg
    
    # 批量推理
    python deploy_rknn_board.py --image-dir ./test_pic --output-dir ./results
    
    # 性能测试
    python deploy_rknn_board.py --benchmark

依赖:
    pip install rknn-toolkit-lite2 opencv-python numpy pillow
"""

import cv2
import numpy as np
import math
import time
import os
import argparse
import sys

# 尝试导入 rknnlite
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("Warning: rknnlite not found. Please run on RK3588 board with rknn-toolkit-lite2 installed.")

# 车牌字符集 (与 HyperLPR3 tokenize.py 一致)
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


class RKNNModel:
    """RKNN 模型封装类"""
    
    def __init__(self, model_path: str, core_mask=None):
        """
        加载 RKNN 模型
        
        Args:
            model_path: RKNN 模型文件路径
            core_mask: NPU 核心掩码 (RKNNLite.NPU_CORE_0/1/2 或组合)
        """
        if not RKNN_AVAILABLE:
            raise RuntimeError("rknnlite not available")
        
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
            
        self.rknn = RKNNLite()
        
        print(f"Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Load RKNN model failed: {model_path}, ret={ret}")
        
        # 初始化运行时
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Init runtime failed, ret={ret}")
        
        print(f"  Model loaded successfully")
    
    def infer(self, input_data: np.ndarray) -> list:
        """
        执行推理
        
        Args:
            input_data: 输入数据 (NCHW 格式)
            
        Returns:
            输出列表
        """
        # RKNN 输入格式：list of numpy arrays
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs
    
    def release(self):
        """释放资源"""
        self.rknn.release()


class PlateRecognizerRKNN:
    """基于 RKNN 的车牌识别器"""
    
    def __init__(self, model_dir: str = "./"):
        """
        初始化识别器
        
        Args:
            model_dir: RKNN 模型目录
        """
        self.model_dir = model_dir
        self.det_size = 640  # 检测模型输入尺寸
        
        # 加载模型，分配到不同 NPU 核心以提高并行度
        # RK3588 有 3 个 NPU 核心
        det_path = os.path.join(model_dir, 'hztk_det.rknn')
        rec_path = os.path.join(model_dir, 'hztk_rec.rknn')
        cls_path = os.path.join(model_dir, 'hztk_cls.rknn')
        
        # 检测模型使用单核或自动
        self.det_model = RKNNModel(det_path, RKNNLite.NPU_CORE_0)
        # 识别和分类模型可以共享另一个核心
        self.rec_model = RKNNModel(rec_path, RKNNLite.NPU_CORE_1)
        self.cls_model = RKNNModel(cls_path, RKNNLite.NPU_CORE_2)
        
        print(f"\nPlateRecognizerRKNN initialized with models from: {model_dir}")
    
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
        
        # BGR to RGB, HWC to CHW, normalize to [0, 1]
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)   # Add batch dim
        
        return img, scale, (pad_w, pad_h)
    
    def postprocess_det(self, outputs: list, scale: float, 
                        orig_shape: tuple, conf_thresh: float = 0.5) -> list:
        """
        检测模型后处理
        
        Args:
            outputs: 模型输出列表
            scale: 缩放比例
            orig_shape: 原始图像尺寸 (H, W)
            conf_thresh: 置信度阈值
            
        Returns:
            检测框列表 [(x1, y1, x2, y2, conf), ...]
        """
        # RKNN 输出可能是列表，取第一个
        output = outputs[0] if isinstance(outputs, list) else outputs
         
        # 对于车牌检测通常是 [1, 25200, 15] 或类似
        # 格式: [x, y, w, h, obj_conf, cls1, cls2, ...]
        output = output.squeeze()  # Remove batch dim
        
        detections = []
        for det in output:
            obj_conf = det[4]
            if obj_conf < conf_thresh:
                continue
            
            # 获取类别置信度
            cls_scores = det[5:]
            cls_idx = np.argmax(cls_scores)
            cls_conf = cls_scores[cls_idx]
            
            # 综合置信度
            conf = obj_conf * cls_conf
            if conf < conf_thresh:
                continue
            
            # 解码边界框 (center_x, center_y, w, h)
            cx, cy, bw, bh = det[:4]
            x1 = (cx - bw / 2) / scale
            y1 = (cy - bh / 2) / scale
            x2 = (cx + bw / 2) / scale
            y2 = (cy + bh / 2) / scale
            
            # 裁剪到图像边界
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            if x2 > x1 and y2 > y1:
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
        识别模型预处理 (与 HyperLPR3 encode_images 一致)
        
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
        
        # Resize image (BGR input, no color conversion needed for this model)
        resized_image = cv2.resize(plate_img, (resized_w, imgH))
        
        # Normalize: (img - 127.5) / 127.5, then transpose to CHW
        resized_image = resized_image.astype(np.float32)
        resized_image = (resized_image.transpose((2, 0, 1)) - 127.5) / 127.5
        
        # Zero-padding to target width
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        
        return np.expand_dims(padding_im, 0)
    
    def decode_plate(self, outputs: list) -> tuple:
        """
        解码车牌识别结果 (CTC 解码)
        
        Args:
            outputs: 模型输出列表
            
        Returns:
            (车牌号, 置信度)
        """
        # RKNN 输出是列表
        output = outputs[0] if isinstance(outputs, list) else outputs
        
        # output shape: [1, seq_len, num_classes] -> [seq_len, num_classes]
        prod = output.squeeze()
        
        # CTC 解码
        indices = np.argmax(prod, axis=-1)  # [seq_len]
        max_probs = np.max(prod, axis=-1)   # [seq_len]
        
        # CTC 解码（去重和去 blank）
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
    
    def classify_plate(self, outputs: list) -> tuple:
        """解码分类结果"""
        output = outputs[0] if isinstance(outputs, list) else outputs
        output = output.squeeze()
        idx = int(np.argmax(output))
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
        det_outputs = self.det_model.infer(det_input)
        detections = self.postprocess_det(det_outputs, scale, image.shape[:2], conf_thresh)
        
        # 2. 对每个检测框进行识别
        for x1, y1, x2, y2, det_conf in detections:
            # 裁剪车牌区域
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # 识别
            rec_input = self.preprocess_rec(plate_img)
            rec_outputs = self.rec_model.infer(rec_input)
            plate_number, rec_conf = self.decode_plate(rec_outputs)
            
            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_outputs = self.cls_model.infer(cls_input)
            plate_type, cls_conf = self.classify_plate(cls_outputs)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'det_conf': det_conf,
                'plate_number': plate_number,
                'rec_conf': rec_conf,
                'plate_type': plate_type,
                'cls_conf': cls_conf,
            })
        
        return results
    
    def benchmark(self, num_iters: int = 50, warmup: int = 5) -> dict:
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
            self.det_model.infer(det_input)
            self.rec_model.infer(rec_input)
            self.cls_model.infer(cls_input)
        
        # 测试检测
        print(f"Benchmarking detection ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.det_model.infer(det_input)
        det_time = (time.time() - start) / num_iters * 1000
        
        # 测试识别
        print(f"Benchmarking recognition ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.rec_model.infer(rec_input)
        rec_time = (time.time() - start) / num_iters * 1000
        
        # 测试分类
        print(f"Benchmarking classification ({num_iters} iterations)...")
        start = time.time()
        for _ in range(num_iters):
            self.cls_model.infer(cls_input)
        cls_time = (time.time() - start) / num_iters * 1000
        
        total_time = det_time + rec_time + cls_time
        
        return {
            'detection_ms': det_time,
            'recognition_ms': rec_time,
            'classification_ms': cls_time,
            'total_ms': total_time,
            'fps': 1000 / total_time,
        }
    
    def release(self):
        """释放资源"""
        self.det_model.release()
        self.rec_model.release()
        self.cls_model.release()


def visualize_result(image: np.ndarray, results: list, output_path: str = None):
    """可视化识别结果"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        use_pil = True
    except ImportError:
        use_pil = False
    
    if use_pil:
        # 尝试加载中文字体 - 按优先级排列
        font_paths = [
            # RK3588/OrangePi 板子上确认存在的中文字体
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 思源黑体
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",  # 思源宋体
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid 回退
            # Ubuntu/Debian 常见中文字体
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 24)
                    print(f"[INFO] 使用中文字体: {fp}")
                    break
                except Exception as e:
                    print(f"[WARN] 加载字体失败 {fp}: {e}")
                    continue
        
        if font is None:
            print("[WARN] 未找到中文字体，中文可能显示异常")
        
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
            
            # 背景和文字
            if font:
                try:
                    bbox = draw.textbbox((x1, y1 - 30), label, font=font)
                    draw.rectangle(bbox, fill=(255, 255, 255))
                    draw.text((x1, y1 - 30), label, font=font, fill=(0, 128, 0))
                except:
                    draw.text((x1, y1 - 20), label, fill=(0, 255, 0))
            else:
                draw.text((x1, y1 - 20), label, fill=(0, 255, 0))
        
        # 转回 OpenCV 格式
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        # 使用 OpenCV 绘制 (不支持中文)
        result_img = image.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{res['plate_number']} ({res['plate_type']})"
            cv2.putText(result_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved: {output_path}")
    
    return result_img


def main():
    parser = argparse.ArgumentParser(description='E2E_HZTK RK3588 NPU Deployment')
    parser.add_argument('--model-dir', type=str, default='./',
                        help='RKNN models directory')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Image directory for batch inference')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    if not RKNN_AVAILABLE:
        print("Error: rknnlite is not available")
        print("Please run this script on RK3588 board with rknn-toolkit-lite2 installed")
        sys.exit(1)
    
    # 初始化识别器
    recognizer = PlateRecognizerRKNN(args.model_dir)
    
    try:
        if args.benchmark:
            # 性能测试
            print("\n" + "=" * 50)
            print("Performance Benchmark (RK3588 NPU)")
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
            
            start = time.time()
            results = recognizer.recognize(image, args.conf_thresh)
            elapsed = time.time() - start
            
            print(f"\nResults for {args.image} ({elapsed*1000:.1f} ms):")
            if results:
                for i, res in enumerate(results):
                    print(f"  [{i+1}] {res['plate_number']} ({res['plate_type']})")
                    print(f"      bbox: {res['bbox']}, det_conf: {res['det_conf']:.3f}, rec_conf: {res['rec_conf']:.3f}")
            else:
                print("  No plate detected")
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, os.path.basename(args.image))
                visualize_result(image, results, output_path)
        
        elif args.image_dir:
            # 批量推理
            image_files = [f for f in os.listdir(args.image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"No images found in {args.image_dir}")
                sys.exit(1)
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
            
            total_time = 0
            total_images = 0
            
            for img_file in image_files:
                img_path = os.path.join(args.image_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Cannot read {img_path}")
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
                
                if args.output_dir:
                    output_path = os.path.join(args.output_dir, img_file)
                    visualize_result(image, results, output_path)
            
            if total_images > 0:
                avg_time = total_time / total_images * 1000
                print(f"\nProcessed {total_images} images")
                print(f"Average time: {avg_time:.1f} ms/image ({1000/avg_time:.1f} FPS)")
        
        else:
            parser.print_help()
    
    finally:
        recognizer.release()


if __name__ == '__main__':
    main()
