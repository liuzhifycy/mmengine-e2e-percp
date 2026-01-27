#!/usr/bin/env python
"""
ONNX 转 TensorRT 引擎

将 ONNX 模型转换为 TensorRT 引擎，支持 FP32、FP16 和 INT8 量化。

使用方法:
    # FP32 精度
    python deploy/onnx2trt.py exports/retinanet.onnx -o exports/retinanet_fp32.engine
    
    # FP16 精度 (推荐)
    python deploy/onnx2trt.py exports/retinanet.onnx -o exports/retinanet_fp16.engine --fp16
    
    # INT8 量化 (需要校准数据)
    python deploy/onnx2trt.py exports/retinanet.onnx -o exports/retinanet_int8.engine --int8 \
        --calib-data data/coco/val2017 --calib-cache exports/calib.cache

依赖:
    pip install tensorrt pycuda
"""
import argparse
import os
import sys

import numpy as np


def check_tensorrt():
    """检查 TensorRT 是否可用"""
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        print("Error: TensorRT is not installed.")
        print("Please install TensorRT:")
        print("  pip install tensorrt")
        print("Or use NVIDIA NGC containers with TensorRT pre-installed.")
        return False


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size: int = 4,
    min_shape: tuple = None,
    opt_shape: tuple = None,
    max_shape: tuple = None,
    calib_data_path: str = None,
    calib_cache_path: str = None,
    calib_batch_size: int = 8,
    calib_num_batches: int = 50,
):
    """
    将 ONNX 模型转换为 TensorRT 引擎
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 TensorRT 引擎路径
        fp16: 是否使用 FP16 精度
        int8: 是否使用 INT8 量化
        max_batch_size: 最大批次大小
        workspace_size: 工作空间大小 (GB)
        min_shape: 最小输入形状 (N, C, H, W)
        opt_shape: 最优输入形状
        max_shape: 最大输入形状
        calib_data_path: INT8 校准数据路径
        calib_cache_path: INT8 校准缓存路径
        calib_batch_size: 校准批次大小
        calib_num_batches: 校准批次数量
    
    Returns:
        bool: 是否成功
    """
    import tensorrt as trt
    
    # 创建 logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 创建 builder
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建 network
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    
    # 创建 ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX 模型
    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(error)}")
            return False
    
    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")
    
    # 创建 builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    
    # 设置精度
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        else:
            print("Warning: FP16 not supported on this platform, using FP32")
    
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 mode enabled")
            
            # 创建 INT8 校准器
            if calib_data_path:
                calibrator = ImageCalibrator(
                    data_path=calib_data_path,
                    cache_file=calib_cache_path,
                    batch_size=calib_batch_size,
                    num_batches=calib_num_batches,
                    input_shape=opt_shape or (1, 3, 640, 640),
                )
                config.int8_calibrator = calibrator
        else:
            print("Warning: INT8 not supported on this platform")
    
    # 设置动态 shape (如果提供)
    if min_shape and opt_shape and max_shape:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"Dynamic shape: min={min_shape}, opt={opt_shape}, max={max_shape}")
    
    # 构建引擎
    print("Building TensorRT engine... (this may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine")
        return False
    
    # 保存引擎
    print(f"Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved successfully! Size: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")
    return True


class ImageCalibrator:
    """INT8 量化校准器
    
    使用校准数据集生成量化参数
    """
    
    def __init__(
        self,
        data_path: str,
        cache_file: str = None,
        batch_size: int = 8,
        num_batches: int = 50,
        input_shape: tuple = (1, 3, 640, 640),
    ):
        import tensorrt as trt
        
        self.data_path = data_path
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_shape = input_shape
        self.current_batch = 0
        
        # 获取图像列表
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} images for calibration")
        
        # 分配 GPU 内存
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.device_input = cuda.mem_alloc(
            batch_size * np.prod(input_shape[1:]) * np.float32().itemsize
        )
        
    def _get_image_files(self):
        """获取图像文件列表"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = []
        
        for root, _, files in os.walk(self.data_path):
            for f in files:
                if f.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, f))
        
        return image_files[:self.batch_size * self.num_batches]
    
    def _preprocess(self, image_path: str) -> np.ndarray:
        """预处理图像"""
        import cv2
        
        _, _, h, w = self.input_shape
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize
        img = cv2.resize(img, (w, h))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        return img
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """获取一个批次的校准数据"""
        import pycuda.driver as cuda
        
        if self.current_batch >= self.num_batches:
            return None
        
        # 准备批次数据
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_files))
        
        batch_data = []
        for i in range(start_idx, end_idx):
            img = self._preprocess(self.image_files[i])
            if img is not None:
                batch_data.append(img)
        
        if len(batch_data) == 0:
            return None
        
        # 填充不足的部分
        while len(batch_data) < self.batch_size:
            batch_data.append(batch_data[-1])
        
        batch_array = np.array(batch_data, dtype=np.float32)
        
        # 复制到 GPU
        cuda.memcpy_htod(self.device_input, batch_array)
        
        self.current_batch += 1
        print(f"Calibration batch {self.current_batch}/{self.num_batches}")
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """读取校准缓存"""
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """写入校准缓存"""
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('onnx', type=str, help='Input ONNX model path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output engine path')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--batch-size', type=int, default=1, help='Max batch size')
    
    # 动态 shape
    parser.add_argument('--min-shape', type=str, default=None, help='Min shape: N,C,H,W')
    parser.add_argument('--opt-shape', type=str, default=None, help='Optimal shape: N,C,H,W')
    parser.add_argument('--max-shape', type=str, default=None, help='Max shape: N,C,H,W')
    
    # INT8 校准
    parser.add_argument('--calib-data', type=str, default=None, help='Calibration data path')
    parser.add_argument('--calib-cache', type=str, default=None, help='Calibration cache path')
    parser.add_argument('--calib-batch-size', type=int, default=8, help='Calibration batch size')
    parser.add_argument('--calib-num-batches', type=int, default=50, help='Number of calibration batches')
    
    args = parser.parse_args()
    
    # 检查 TensorRT
    if not check_tensorrt():
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        suffix = '_int8' if args.int8 else ('_fp16' if args.fp16 else '_fp32')
        args.output = args.onnx.replace('.onnx', f'{suffix}.engine')
    
    # 解析 shape
    def parse_shape(s):
        if s is None:
            return None
        return tuple(map(int, s.split(',')))
    
    min_shape = parse_shape(args.min_shape)
    opt_shape = parse_shape(args.opt_shape)
    max_shape = parse_shape(args.max_shape)
    
    # INT8 需要校准数据
    if args.int8 and args.calib_data is None:
        print("Warning: INT8 mode requires calibration data (--calib-data)")
        print("Using default calibration (may not be optimal)")
    
    # 构建引擎
    success = build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        max_batch_size=args.batch_size,
        workspace_size=args.workspace,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        calib_data_path=args.calib_data,
        calib_cache_path=args.calib_cache,
        calib_batch_size=args.calib_batch_size,
        calib_num_batches=args.calib_num_batches,
    )
    
    if success:
        print("\nConversion successful!")
        print(f"Engine saved to: {args.output}")
    else:
        print("\nConversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
