#!/usr/bin/env python
"""
E2E_HZTK 部署导出工具

将 E2E_HZTK 的 ONNX 模型复制到项目目录，并可选择转换为 TensorRT 引擎。

E2E_HZTK 包含以下模型:
- y5fu_640x_sim.onnx: 车牌检测模型 (YOLOv5-based, 640x640)
- y5fu_320x_sim.onnx: 车牌检测模型 (YOLOv5-based, 320x320, 更快)
- rpv3_mdict_160_r3.onnx: 车牌识别模型 (OCR)
- litemodel_cls_96x_r1.onnx: 车牌分类模型 (颜色/类型)

使用方法:
    # 复制 ONNX 模型到项目目录
    python tools/plate_recognition/export_e2e_hztk.py --output-dir exports/e2e_hztk
    
    # 同时转换为 TensorRT FP16 引擎
    python tools/plate_recognition/export_e2e_hztk.py --output-dir exports/e2e_hztk --trt --fp16
    
    # 仅复制指定模型
    python tools/plate_recognition/export_e2e_hztk.py --output-dir exports/e2e_hztk --models det_640x rec

依赖:
    pip install hyperlpr3 onnx
    pip install tensorrt pycuda  # 如需 TensorRT 转换
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# E2E_HZTK 模型配置
E2E_HZTK_MODELS = {
    'det_640x': {
        'filename': 'y5fu_640x_sim.onnx',
        'description': '车牌检测模型 (640x640, 高精度)',
        'input_shape': (1, 3, 640, 640),
    },
    'det_320x': {
        'filename': 'y5fu_320x_sim.onnx', 
        'description': '车牌检测模型 (320x320, 快速)',
        'input_shape': (1, 3, 320, 320),
    },
    'rec': {
        'filename': 'rpv3_mdict_160_r3.onnx',
        'description': '车牌识别模型 (OCR)',
        'input_shape': (1, 3, 48, 160),
    },
    'cls': {
        'filename': 'litemodel_cls_96x_r1.onnx',
        'description': '车牌分类模型 (颜色/类型)',
        'input_shape': (1, 3, 96, 96),
    },
}


def get_e2e_hztk_model_dir():
    """获取 E2E_HZTK 模型目录"""
    home = os.environ.get('HOME', os.path.expanduser('~'))
    model_dir = os.path.join(home, '.hyperlpr3', '20230229', 'onnx')
    
    if not os.path.exists(model_dir):
        print(f"Error: E2E_HZTK model directory not found: {model_dir}")
        print("Please run E2E_HZTK at least once to download models:")
        print("  python -c \"import hyperlpr3; lpr = hyperlpr3.LicensePlateCatcher(); print('Models downloaded')\"")
        return None
    
    return model_dir


def copy_onnx_models(output_dir: str, models: list = None):
    """
    复制 E2E_HZTK ONNX 模型到指定目录
    
    Args:
        output_dir: 输出目录
        models: 要复制的模型列表，None 表示全部
        
    Returns:
        复制的模型路径列表
    """
    model_dir = get_e2e_hztk_model_dir()
    if model_dir is None:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    if models is None:
        models = list(E2E_HZTK_MODELS.keys())
    
    copied_models = []
    
    print(f"\nCopying ONNX models to: {output_dir}")
    print("-" * 50)
    
    for model_key in models:
        if model_key not in E2E_HZTK_MODELS:
            print(f"Warning: Unknown model '{model_key}', skipping")
            continue
        
        model_info = E2E_HZTK_MODELS[model_key]
        src_path = os.path.join(model_dir, model_info['filename'])
        dst_path = os.path.join(output_dir, model_info['filename'])
        
        if not os.path.exists(src_path):
            print(f"Warning: Model not found: {src_path}")
            continue
        
        shutil.copy2(src_path, dst_path)
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        
        print(f"  {model_key}: {model_info['filename']}")
        print(f"    - {model_info['description']}")
        print(f"    - Input shape: {model_info['input_shape']}")
        print(f"    - Size: {size_mb:.2f} MB")
        
        copied_models.append({
            'key': model_key,
            'onnx_path': dst_path,
            'info': model_info,
        })
    
    return copied_models


def convert_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True, 
                        input_shape: tuple = None, workspace: int = 4):
    """
    将 ONNX 模型转换为 TensorRT 引擎
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出引擎路径
        fp16: 是否使用 FP16 精度
        input_shape: 输入形状
        workspace: 工作空间大小 (GB)
        
    Returns:
        bool: 是否成功
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT not installed. Please install:")
        print("  pip install tensorrt pycuda")
        return False
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 创建 builder
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    
    # 解析 ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"  Loading ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"    ONNX Parse Error: {parser.get_error(error)}")
            return False
    
    # 配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    FP16 mode enabled")
    
    # 构建引擎
    print("    Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("    Failed to build engine")
        return False
    
    # 保存
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"    Saved: {engine_path} ({size_mb:.2f} MB)")
    
    return True


def verify_onnx_model(onnx_path: str):
    """验证 ONNX 模型"""
    try:
        import onnx
        import onnxruntime as ort
        
        # 加载并检查模型
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # 获取输入输出信息
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"    Inputs:")
        for inp in inputs:
            print(f"      - {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"    Outputs:")
        for out in outputs:
            print(f"      - {out.name}: {out.shape} ({out.type})")
        
        return True
        
    except Exception as e:
        print(f"    Verification failed: {e}")
        return False


def create_deployment_config(output_dir: str, models: list):
    """创建部署配置文件"""
    import json
    
    config = {
        'version': '1.0',
        'framework': 'E2E_HZTK',
        'models': {}
    }
    
    for model in models:
        config['models'][model['key']] = {
            'onnx': os.path.basename(model['onnx_path']),
            'description': model['info']['description'],
            'input_shape': model['info']['input_shape'],
        }
        
        # 检查是否有 TensorRT 引擎
        engine_path = model['onnx_path'].replace('.onnx', '_fp16.engine')
        if os.path.exists(engine_path):
            config['models'][model['key']]['trt_fp16'] = os.path.basename(engine_path)
    
    config_path = os.path.join(output_dir, 'deploy_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\nDeployment config saved: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Export E2E_HZTK models for deployment')
    parser.add_argument('--output-dir', type=str, default='exports/e2e_hztk',
                        help='Output directory for exported models')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        choices=['det_640x', 'det_320x', 'rec', 'cls'],
                        help='Models to export (default: all)')
    parser.add_argument('--trt', action='store_true',
                        help='Convert to TensorRT engines')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision for TensorRT')
    parser.add_argument('--workspace', type=int, default=4,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX models')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list:
        print("\nAvailable E2E_HZTK models:")
        print("-" * 50)
        for key, info in E2E_HZTK_MODELS.items():
            print(f"  {key}:")
            print(f"    - File: {info['filename']}")
            print(f"    - Description: {info['description']}")
            print(f"    - Input shape: {info['input_shape']}")
        return
    
    # 复制 ONNX 模型
    copied_models = copy_onnx_models(args.output_dir, args.models)
    
    if not copied_models:
        print("\nNo models were copied.")
        sys.exit(1)
    
    # 验证模型
    if args.verify:
        print("\nVerifying ONNX models:")
        print("-" * 50)
        for model in copied_models:
            print(f"  {model['key']}:")
            verify_onnx_model(model['onnx_path'])
    
    # 转换为 TensorRT
    if args.trt:
        print("\nConverting to TensorRT:")
        print("-" * 50)
        
        for model in copied_models:
            suffix = '_fp16' if args.fp16 else '_fp32'
            engine_path = model['onnx_path'].replace('.onnx', f'{suffix}.engine')
            
            print(f"  {model['key']}:")
            success = convert_to_tensorrt(
                model['onnx_path'],
                engine_path,
                fp16=args.fp16,
                input_shape=model['info']['input_shape'],
                workspace=args.workspace,
            )
            
            if success:
                model['engine_path'] = engine_path
    
    # 创建部署配置
    create_deployment_config(args.output_dir, copied_models)
    
    print("\n" + "=" * 50)
    print("Export completed!")
    print(f"Output directory: {args.output_dir}")
    print("\nExported files:")
    for f in os.listdir(args.output_dir):
        size_mb = os.path.getsize(os.path.join(args.output_dir, f)) / 1024 / 1024
        print(f"  - {f} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()