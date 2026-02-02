import cv2
import numpy as np
import time
import os
import argparse
import sys

# 尝试导入 rknnlite
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("Error: rknnlite not found. Please run this script on the RK3588 board with rknn-toolkit-lite2 installed.")
    print("Install via: pip install rknn_toolkit_lite2_cp38_cp38_linux_aarch64.whl (check python version)")
    # 为了防止在PC上编辑时报错，这里仅做提示，不退出，方便代码查看
    pass

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

PLATE_TYPES = {0: "Blue", 1: "Green", 2: "Yellow"}

class RKNNModel:
    def __init__(self, model_path, core_mask=RKNNLite.NPU_CORE_0):
        self.rknn = RKNNLite()
        
        print(f"Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print(f"Load RKNN model failed: {model_path}")
            sys.exit(ret)
            
        print("Init runtime environment")
        # 异步模式通常性能更好，但在Python API中直接调用inference通常是同步的
        # target=rk3588
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            print("Init runtime environment failed")
            sys.exit(ret)
            
    def infer(self, input_data):
        # input_data should be NCHW or NHWC depending on model config.
        # RKNN inputs are usually list of numpy arrays
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs
    
    def release(self):
        self.rknn.release()

class PlateRecognizerRKNN:
    def __init__(self, model_dir):
        # 加载模型
        # 可以分配不同的 NPU 核心以提高并行度
        # RK3588 有 3 个核心: NPU_CORE_0, NPU_CORE_1, NPU_CORE_2
        self.det_model = RKNNModel(os.path.join(model_dir, 'hztk_det.rknn'), RKNNLite.NPU_CORE_0)
        self.rec_model = RKNNModel(os.path.join(model_dir, 'hztk_rec.rknn'), RKNNLite.NPU_CORE_1) # 放在不同核心
        self.cls_model = RKNNModel(os.path.join(model_dir, 'hztk_cls.rknn'), RKNNLite.NPU_CORE_1) 
        
        self.det_size = 640 # 假设使用640模型

    def preprocess_det(self, image):
        h, w = image.shape[:2]
        scale = min(self.det_size / h, self.det_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        pad_h = self.det_size - new_h
        pad_w = self.det_size - new_w
        
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # RKNN model usually expects RGB [0, 1] or [0, 255] depending on conversion config.
        # 假设我们在转换时保留了与 ONNX 一致的预处理（归一化）
        # ONNX: BGR -> RGB -> /255.0 -> NCHW
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # NCHW
        img = np.expand_dims(img, 0)
        
        return img, scale, (pad_w, pad_h)

    def preprocess_rec(self, plate_img):
        imgH, imgW = 48, 160
        h, w = plate_img.shape[:2]
        ratio = w / float(h)
        if ratio > imgW / imgH:
            ratio = imgW / imgH
        
        new_w = int(imgH * ratio)
        resized = cv2.resize(plate_img, (new_w, imgH))
        
        # Padding
        padded = np.zeros((imgH, imgW, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        
        # Preprocess
        img = padded.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img

    def preprocess_cls(self, plate_img):
        img = cv2.resize(plate_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img

    def decode_plate(self, output):
        # output shape: [1, seq_len, num_classes]
        prod = output[0].squeeze() # Remove batch dim if needed, RKNN outputs list
        indices = np.argmax(prod, axis=-1)
        
        plate_chars = []
        prev_idx = -1
        for idx in indices:
            if idx == 0: # blank
                prev_idx = idx
                continue
            if idx == prev_idx:
                continue
            if idx < len(PLATE_CHARS):
                plate_chars.append(PLATE_CHARS[idx])
            prev_idx = idx
            
        return "".join(plate_chars)

    def recognize(self, image):
        results = []
        
        # 1. Detection
        det_input, scale, _ = self.preprocess_det(image)
        det_outputs = self.det_model.infer(det_input)
        
        # YOLOv5 postprocess (Simplified)
        # 假设输出已经是 [1, 25200, 85] 或类似格式
        # 这里需要根据具体 YOLO 模型的输出节点进行解析
        # 为了简化，我们假设模型输出已经经过了解码或者我们可以直接处理
        # 注意：RKNN 的 YOLO 输出通常需要特定的后处理代码，因为 NPU 不支持部分 Grid Sample 操作
        # 这里仅为框架示例，实际部署可能需要移植 yolov5_postprocess
        
        # ... (Detection Post-processing code would go here) ...
        # 模拟一个检测结果用于测试流程
        # x1, y1, x2, y2, conf
        detections = [] # 需要实现 YOLO 后处理
        
        # 2. Loop detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0: continue
            
            # Rec
            rec_in = self.preprocess_rec(plate_img)
            rec_out = self.rec_model.infer(rec_in)
            plate_str = self.decode_plate(rec_out)
            
            # Cls
            cls_in = self.preprocess_cls(plate_img)
            cls_out = self.cls_model.infer(cls_in)
            cls_idx = np.argmax(cls_out[0])
            plate_type = PLATE_TYPES.get(cls_idx, "Unknown")
            
            results.append({
                'plate': plate_str,
                'type': plate_type,
                'bbox': [x1, y1, x2, y2]
            })
            
        return results

    def release(self):
        self.det_model.release()
        self.rec_model.release()
        self.cls_model.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./', help='Path to RKNN models')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    args = parser.parse_args()
    
    recognizer = PlateRecognizerRKNN(args.model_dir)
    
    img = cv2.imread(args.image)
    if img is None:
        print("Image not found")
        return
        
    start = time.time()
    results = recognizer.recognize(img)
    dt = time.time() - start
    print(f"Inference time: {dt*1000:.2f} ms")
    print("Results:", results)
    
    recognizer.release()

if __name__ == '__main__':
    main()
