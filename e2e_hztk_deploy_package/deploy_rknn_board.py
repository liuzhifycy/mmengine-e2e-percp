#!/usr/bin/env python3
"""
E2E_HZTK RK3588 NPU 部署推理脚本

支持：单张图片、图片目录、RTSP视频流

使用方法:
    python deploy_rknn_board.py --image ./test.jpg
    python deploy_rknn_board.py --image-dir ./test_pic --output-dir ./results
    python deploy_rknn_board.py --rtsp "rtsp://admin:password@ip:554/stream1"
    python deploy_rknn_board.py --benchmark
"""

import cv2
import numpy as np
import math
import time
import os
import argparse
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("Warning: rknnlite not available, running in debug mode")

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

PLATE_TYPES = {0: "蓝牌", 1: "绿牌", 2: "黄牌"}


def get_chinese_font(size=20):
    """获取中文字体"""
    if not PIL_AVAILABLE:
        return None
    
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    
    try:
        return ImageFont.load_default()
    except Exception:
        return None


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


class PlateRecognizerRKNN:
    """基于 RKNN 的车牌识别器 (NHWC 输入格式)"""
    
    def __init__(self, model_dir: str = "./"):
        self.model_dir = model_dir
        self.det_size = 640
        
        if not RKNN_AVAILABLE:
            print("Warning: RKNN not available, methods will return empty results")
            self.det_rknn = None
            self.rec_rknn = None
            self.cls_rknn = None
            return
        
        det_path = os.path.join(model_dir, 'hztk_det.rknn')
        rec_path = os.path.join(model_dir, 'hztk_rec.rknn')
        cls_path = os.path.join(model_dir, 'hztk_cls.rknn')
        
        print("Loading RKNN models...")
        
        self.det_rknn = RKNNLite()
        ret = self.det_rknn.load_rknn(det_path)
        if ret != 0:
            raise RuntimeError(f"Load det model failed: {det_path}")
        self.det_rknn.init_runtime()
        print(f"  Det model loaded: {det_path}")
        
        self.rec_rknn = RKNNLite()
        ret = self.rec_rknn.load_rknn(rec_path)
        if ret != 0:
            raise RuntimeError(f"Load rec model failed: {rec_path}")
        self.rec_rknn.init_runtime()
        print(f"  Rec model loaded: {rec_path}")
        
        self.cls_rknn = RKNNLite()
        ret = self.cls_rknn.load_rknn(cls_path)
        if ret != 0:
            raise RuntimeError(f"Load cls model failed: {cls_path}")
        self.cls_rknn.init_runtime()
        print(f"  Cls model loaded: {cls_path}")
        
        print("PlateRecognizerRKNN initialized (NHWC format)")
    
    def preprocess_det(self, image: np.ndarray) -> tuple:
        """检测模型预处理 - NHWC 格式"""
        padded, ratio, (dw, dh) = letterbox(image, (self.det_size, self.det_size))
        
        # BGR -> RGB, 归一化, 保持 NHWC
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # [1, H, W, C] - NHWC
        
        return img, ratio, (dw, dh)
    
    def postprocess_det(self, output: np.ndarray, ratio: tuple, 
                        pad_size: tuple, orig_shape: tuple, 
                        conf_thresh: float = 0.5) -> list:
        """检测后处理"""
        output = output.squeeze()
        dw, dh = pad_size
        orig_h, orig_w = orig_shape
        
        mask = output[:, 4] > conf_thresh
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
        """识别模型预处理 - NHWC 格式"""
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
        
        # Padding, 保持 NHWC
        padded = np.zeros((imgH, imgW, 3), dtype=np.float32)
        padded[:, 0:resized_w, :] = resized
        
        return np.expand_dims(padded, 0)  # [1, H, W, C] - NHWC
    
    def decode_plate(self, output: np.ndarray) -> tuple:
        """解码车牌"""
        prod = output.squeeze()
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
        """分类模型预处理 - NHWC 格式"""
        img = cv2.resize(plate_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)  # [1, H, W, C] - NHWC
    
    def classify_plate(self, output: np.ndarray) -> tuple:
        """分类解码"""
        output = output.squeeze()
        idx = int(np.argmax(output))
        conf = float(output[idx])
        #wang
        #-
        #return PLATE_TYPES.get(idx, "未知"), conf
        #+
        plate_type=""
        return plate_type, conf

    
    def recognize(self, image: np.ndarray, conf_thresh: float = 0.5) -> list:
        """完整识别流程"""
        if self.det_rknn is None:
            return []
        
        results = []
        
        # 检测
        det_input, ratio, pad_size = self.preprocess_det(image)
        det_output = self.det_rknn.inference(inputs=[det_input])[0]
        detections = self.postprocess_det(det_output, ratio, pad_size, 
                                          image.shape[:2], conf_thresh)
        
        # 识别每个检测框
        for x1, y1, x2, y2, det_conf in detections:
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # OCR
            rec_input = self.preprocess_rec(plate_img)
            rec_output = self.rec_rknn.inference(inputs=[rec_input])[0]
            plate_number, rec_conf = self.decode_plate(rec_output)
            
            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_output = self.cls_rknn.inference(inputs=[cls_input])[0]
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
    
    def benchmark(self, num_iters: int = 50, warmup: int = 5) -> dict:
        """性能测试 - NHWC 格式"""
        if self.det_rknn is None:
            return {'error': 'RKNN not available'}
        
        det_input = np.random.randn(1, self.det_size, self.det_size, 3).astype(np.float32)
        rec_input = np.random.randn(1, 48, 160, 3).astype(np.float32)
        cls_input = np.random.randn(1, 96, 96, 3).astype(np.float32)
        
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            self.det_rknn.inference(inputs=[det_input])
            self.rec_rknn.inference(inputs=[rec_input])
            self.cls_rknn.inference(inputs=[cls_input])
        
        print(f"Benchmarking ({num_iters} iterations)...")
        
        start = time.time()
        for _ in range(num_iters):
            self.det_rknn.inference(inputs=[det_input])
        det_time = (time.time() - start) / num_iters * 1000
        
        start = time.time()
        for _ in range(num_iters):
            self.rec_rknn.inference(inputs=[rec_input])
        rec_time = (time.time() - start) / num_iters * 1000
        
        start = time.time()
        for _ in range(num_iters):
            self.cls_rknn.inference(inputs=[cls_input])
        cls_time = (time.time() - start) / num_iters * 1000
        
        total = det_time + rec_time + cls_time
        
        return {
            'detection_ms': det_time,
            'recognition_ms': rec_time,
            'classification_ms': cls_time,
            'total_ms': total,
            'fps': 1000 / total,
        }
    
    def release(self):
        """释放资源"""
        if self.det_rknn:
            self.det_rknn.release()
        if self.rec_rknn:
            self.rec_rknn.release()
        if self.cls_rknn:
            self.cls_rknn.release()


def visualize_result(image: np.ndarray, results: list, font=None) -> np.ndarray:
    """可视化结果"""
    result_img = image.copy()
    
    for res in results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #wang
        #-
        #label = f"{res['plate_number']} ({res['plate_type']})"
        #+
        label = f"{res['plate_number']}"
        
        if PIL_AVAILABLE and font is not None:
            pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            text_y = max(y1 - 30, 5)
            try:
                bbox = draw.textbbox((x1, text_y), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 255, 0))
                draw.text((x1, text_y), label, font=font, fill=(0, 0, 0))
            except Exception:
                draw.text((x1, text_y), label, font=font, fill=(0, 255, 0))
            result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(result_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result_img


def run_rtsp_stream(recognizer, rtsp_url: str, conf_thresh: float = 0.5, 
                    display: bool = True, save_dir: str = None,
                    save_all: bool = False, save_interval: int = 1):
    """处理 RTSP 视频流
    
    Args:
        recognizer: 车牌识别器
        rtsp_url: RTSP 流地址
        conf_thresh: 置信度阈值
        display: 是否显示窗口
        save_dir: 保存目录（设置后会保存可视化图片）
        save_all: True=保存所有帧，False=仅保存检测到车牌的帧
        save_interval: 保存间隔（每N帧保存一次）
    """
    # 检查是否有显示器，没有则自动禁用显示
    if display:
        display_available = os.environ.get('DISPLAY') is not None
        if not display_available:
            print("Warning: No display available, disabling GUI")
            display = False
    
    print(f"Connecting to RTSP: {rtsp_url}")
    
    # 使用 FFMPEG 后端，设置 TCP 传输
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
    
    if not cap.isOpened():
        print(f"无法打开RTSP流: {rtsp_url}")
        print("请检查：")
        print("1. 网络是否可达")
        print("2. 用户名和密码是否正确")
        print("3. 摄像头是否支持RTSP且该流存在")
        return
    
    print("=" * 60)
    print("成功连接RTSP流，开始处理...")
    print(f"  显示模式: {'开启' if display else '关闭'}")
    print(f"  置信度阈值: {conf_thresh}")
    if save_dir:
        print(f"  保存目录: {save_dir}")
        print(f"  保存模式: {'所有帧' if save_all else '仅检测到车牌的帧'}")
        print(f"  保存间隔: 每 {save_interval} 帧")
    else:
        print(f"  保存目录: 未设置 (使用 -o 或 --output-dir 指定)")
    print("  按 Ctrl+C 退出")
    print("=" * 60)
    
    font = get_chinese_font(size=24) if PIL_AVAILABLE else None
    frame_count = 0
    detect_count = 0
    fps_list = []
    last_print_time = time.time()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 读取帧失败，尝试重连...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            frame_count += 1
            
            # 每帧都处理
            start = time.time()
            results = recognizer.recognize(frame, conf_thresh)
            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(fps)
            
            # 打印识别结果
            if results:
                detect_count += len(results)
                for res in results:
                    print(f"[Frame {frame_count:05d}] 检测到车牌: {res['plate_number']} | "
                          f"类型: {res['plate_type']} | "
                          f"检测置信度: {res['det_conf']:.3f} | "
                          f"识别置信度: {res['rec_conf']:.3f} | "
                          f"耗时: {elapsed*1000:.1f}ms")
            
            # 每5秒打印一次统计信息
            now = time.time()
            if now - last_print_time >= 5.0:
                avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
                print(f"[STAT] 已处理: {frame_count} 帧 | "
                      f"检测到: {detect_count} 个车牌 | "
                      f"平均FPS: {avg_fps:.1f}")
                last_print_time = now
            
            # 判断是否需要保存
            should_save = False
            if save_dir and frame_count % save_interval == 0:
                if save_all:
                    should_save = True
                elif results:  # 仅保存检测到车牌的帧
                    should_save = True
            
            # 可视化和保存
            if display or should_save:
                vis_frame = visualize_result(frame, results, font)
                avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
                cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if should_save:
                    save_path = os.path.join(save_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(save_path, vis_frame)
                    print(f"[SAVE] {save_path}")
                
                if display:
                    cv2.imshow("Plate Recognition", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] 用户退出")
                        break
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断 (Ctrl+C)")
    
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("运行统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  检测到车牌数: {detect_count}")
        if fps_list:
            print(f"  平均 FPS: {np.mean(fps_list):.1f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='E2E_HZTK RK3588 NPU Deployment')
    parser.add_argument('--model-dir', type=str, default='./', help='RKNN models directory')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Image directory')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for saving visualized images')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--no-display', action='store_true', help='Disable display for RTSP')
    parser.add_argument('--save-all', action='store_true', help='Save all frames (not just detected ones)')
    parser.add_argument('--save-interval', type=int, default=1, help='Save every N frames (default: 1)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    recognizer = PlateRecognizerRKNN(args.model_dir)
    
    try:
        if args.benchmark:
            print("\n" + "=" * 50)
            print("Performance Benchmark (RK3588 NPU)")
            print("=" * 50)
            
            metrics = recognizer.benchmark()
            
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
            else:
                print(f"\nResults:")
                print(f"  Detection:      {metrics['detection_ms']:.2f} ms")
                print(f"  Recognition:    {metrics['recognition_ms']:.2f} ms")
                print(f"  Classification: {metrics['classification_ms']:.2f} ms")
                print(f"  Total:          {metrics['total_ms']:.2f} ms")
                print(f"  FPS:            {metrics['fps']:.1f}")
        
        elif args.rtsp:
            run_rtsp_stream(
                recognizer, 
                args.rtsp, 
                args.conf_thresh,
                display=not args.no_display,
                save_dir=args.output_dir,
                save_all=args.save_all,
                save_interval=args.save_interval
            )
        
        elif args.image:
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Cannot read {args.image}")
                sys.exit(1)
            
            start = time.time()
            results = recognizer.recognize(image, args.conf_thresh)
            elapsed = time.time() - start
            
            print(f"\nResults for {args.image} ({elapsed*1000:.1f} ms):")
            if results:
                for i, res in enumerate(results):
                    print(f"  [{i+1}] {res['plate_number']} ({res['plate_type']})")
                    print(f"      bbox: {res['bbox']}, det: {res['det_conf']:.3f}, rec: {res['rec_conf']:.3f}")
            else:
                print("  No plate detected")
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                font = get_chinese_font(size=24)
                vis_img = visualize_result(image, results, font)
                output_path = os.path.join(args.output_dir, os.path.basename(args.image))
                cv2.imwrite(output_path, vis_img)
                print(f"Saved: {output_path}")
        
        elif args.image_dir:
            image_files = [f for f in os.listdir(args.image_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"No images found in {args.image_dir}")
                sys.exit(1)
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
            
            font = get_chinese_font(size=24)
            total_time = 0
            total_images = 0
            
            for img_file in image_files:
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
                    print(f"No plate ({elapsed*1000:.1f}ms)")
                
                if args.output_dir:
                    vis_img = visualize_result(image, results, font)
                    output_path = os.path.join(args.output_dir, img_file)
                    cv2.imwrite(output_path, vis_img)
            
            if total_images > 0:
                avg = total_time / total_images * 1000
                print(f"\nProcessed {total_images} images, avg: {avg:.1f} ms ({1000/avg:.1f} FPS)")
        
        else:
            parser.print_help()
    
    finally:
        recognizer.release()


if __name__ == '__main__':
    main()
