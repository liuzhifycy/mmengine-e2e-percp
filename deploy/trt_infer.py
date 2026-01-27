#!/usr/bin/env python
"""
TensorRT 推理工具

使用 TensorRT 引擎进行目标检测推理。

使用方法:
    # 单张图片推理
    python deploy/trt_infer.py exports/retinanet_fp16.engine -i image.jpg -o output.jpg
    
    # 批量推理
    python deploy/trt_infer.py exports/retinanet_fp16.engine -i images/ -o outputs/
    
    # 性能测试
    python deploy/trt_infer.py exports/retinanet_fp16.engine --benchmark

依赖:
    pip install tensorrt pycuda opencv-python
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np


def check_dependencies():
    """检查依赖"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install required packages:")
        print("  pip install tensorrt pycuda")
        return False


class TRTInference:
    """TensorRT 推理类"""
    
    def __init__(self, engine_path: str):
        """
        初始化 TensorRT 推理引擎
        
        Args:
            engine_path: TensorRT 引擎文件路径
        """
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.cuda = cuda
        self.trt = trt
        
        # 加载引擎
        print(f"Loading TensorRT engine: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)
            
            # 分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem,
                })
                print(f"  Input: {name}, shape={shape}, dtype={dtype}")
            else:
                self.outputs.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem,
                })
                print(f"  Output: {name}, shape={shape}, dtype={dtype}")
        
        # 获取输入形状
        self.input_shape = tuple(self.inputs[0]['shape'])
        print(f"Engine loaded. Input shape: {self.input_shape}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: BGR 格式的图像 (H, W, C)
            
        Returns:
            预处理后的图像 (1, C, H, W)
        """
        _, c, h, w = self.input_shape
        
        # Resize
        img = cv2.resize(image, (w, h))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        
        # HWC to NCHW
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return np.ascontiguousarray(img)
    
    def infer(self, image: np.ndarray) -> list:
        """
        执行推理
        
        Args:
            image: 预处理后的图像 (N, C, H, W)
            
        Returns:
            输出列表
        """
        # 复制输入到 GPU
        np.copyto(self.inputs[0]['host'], image.ravel())
        self.cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream,
        )
        
        # 执行推理
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 复制输出到 CPU
        outputs = []
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        for out in self.outputs:
            outputs.append(out['host'].reshape(out['shape']).copy())
        
        return outputs
    
    def benchmark(self, num_warmup: int = 10, num_iterations: int = 100) -> dict:
        """
        性能测试
        
        Args:
            num_warmup: 预热次数
            num_iterations: 测试次数
            
        Returns:
            性能指标字典
        """
        # 创建随机输入
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # 预热
        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            self.infer(dummy_input)
        
        # 测试
        print(f"Benchmarking ({num_iterations} iterations)...")
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            self.infer(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'fps': 1000 / np.mean(times),
        }


def detect_image(
    engine: TRTInference,
    image_path: str,
    output_path: str = None,
    score_threshold: float = 0.3,
):
    """
    检测单张图片
    
    Args:
        engine: TensorRT 推理引擎
        image_path: 输入图片路径
        output_path: 输出图片路径 (可选)
        score_threshold: 置信度阈值
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    orig_h, orig_w = image.shape[:2]
    
    # 预处理
    input_tensor = engine.preprocess(image)
    
    # 推理
    start = time.perf_counter()
    outputs = engine.infer(input_tensor)
    end = time.perf_counter()
    
    print(f"Inference time: {(end - start) * 1000:.2f} ms")
    print(f"Outputs: {[o.shape for o in outputs]}")
    
    # TODO: 后处理 - 解析检测结果
    # 这里需要根据具体模型输出格式进行解析
    # RetinaNet 输出格式: cls_scores, bbox_preds (多尺度)
    
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TensorRT Inference')
    parser.add_argument('engine', type=str, help='TensorRT engine path')
    parser.add_argument('-i', '--input', type=str, default=None, help='Input image or directory')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output image or directory')
    parser.add_argument('--score-threshold', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 加载引擎
    engine = TRTInference(args.engine)
    
    # 性能测试
    if args.benchmark:
        results = engine.benchmark(args.warmup, args.iterations)
        print("\n=== Benchmark Results ===")
        print(f"Mean:   {results['mean']:.2f} ms")
        print(f"Std:    {results['std']:.2f} ms")
        print(f"Min:    {results['min']:.2f} ms")
        print(f"Max:    {results['max']:.2f} ms")
        print(f"Median: {results['median']:.2f} ms")
        print(f"FPS:    {results['fps']:.1f}")
        return
    
    # 推理
    if args.input:
        if os.path.isfile(args.input):
            detect_image(engine, args.input, args.output, args.score_threshold)
        elif os.path.isdir(args.input):
            # 批量处理
            output_dir = args.output or 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            
            for f in os.listdir(args.input):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(args.input, f)
                    output_path = os.path.join(output_dir, f)
                    detect_image(engine, input_path, output_path, args.score_threshold)
        else:
            print(f"Error: Input not found: {args.input}")
            sys.exit(1)
    else:
        print("No input specified. Use -i to specify input or --benchmark for performance test.")


if __name__ == '__main__':
    main()
