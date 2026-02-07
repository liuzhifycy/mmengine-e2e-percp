#!/usr/bin/env python3
"""
ONNX 到 RKNN 模型转换脚本 (V2 - 修正输入布局配置)

关键修复：
- 设置 model_input_layouts='NCHW'，因为 ONNX 模型是 NCHW 格式
- RKNN 默认期望 NHWC，必须明确指定
"""

import os
import sys
import argparse

try:
    from rknn.api import RKNN
except ImportError:
    print("Error: rknn-toolkit2 not installed")
    print("Install via: pip install rknn-toolkit2")
    sys.exit(1)

# 模型配置
MODELS = {
    'det': {
        'onnx': 'hztk_det.onnx',
        'rknn': 'hztk_det_v2.rknn',
        'input_name': 'input',
        'input_shape': [1, 3, 640, 640],  # NCHW
    },
    'rec': {
        'onnx': 'hztk_rec.onnx',
        'rknn': 'hztk_rec_v2.rknn',
        'input_name': 'data',
        'input_shape': [1, 3, 48, 160],  # NCHW
    },
    'cls': {
        'onnx': 'hztk_cls.onnx',
        'rknn': 'hztk_cls_v2.rknn',
        'input_name': 'data',
        'input_shape': [1, 3, 96, 96],  # NCHW
    }
}


def export_rknn(model_name, model_info, input_dir, output_dir):
    onnx_path = os.path.join(input_dir, model_info['onnx'])
    rknn_path = os.path.join(output_dir, model_info['rknn'])

    if not os.path.exists(onnx_path):
        print(f"[Error] ONNX model not found: {onnx_path}")
        return False

    rknn = RKNN(verbose=True)

    # 1. 配置 - 关键：指定输入布局为 NCHW
    print(f"--> Config model: {model_name}")
    rknn.config(
        target_platform='rk3588',
        optimization_level=3,
        # 不做额外的均值/标准差处理，我们在推理代码中手动处理
        # mean_values=None,
        # std_values=None,
    )

    # 2. 加载 ONNX - 关键：指定输入布局
    print(f"--> Loading model: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[model_info['input_name']],
        input_size_list=[model_info['input_shape']],
        # 明确指定输入格式为 NCHW（与 ONNX 模型一致）
        input_initial_val=None,
        outputs=None,  # 自动推断输出
    )
    if ret != 0:
        print('[Error] Load model failed!')
        rknn.release()
        return False

    # 3. 构建模型 - FP16 精度，不量化
    print('--> Building model (FP16, no quantization)')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('[Error] Build model failed!')
        rknn.release()
        return False

    # 4. 导出 RKNN
    print(f"--> Exporting RKNN model to: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('[Error] Export rknn model failed!')
        rknn.release()
        return False

    print(f"[Success] {model_name} exported to {rknn_path}")
    
    rknn.release()
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to RKNN for RK3588 (V2)')
    parser.add_argument('--input-dir', type=str, default='.', help='Directory containing ONNX models')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save RKNN models')
    parser.add_argument('--model', type=str, default='all', choices=['det', 'rec', 'cls', 'all'],
                        help='Which model to convert')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=" * 60)
    print("ONNX to RKNN Conversion (V2 - NCHW layout)")
    print("Target Platform: RK3588")
    print("Precision: FP16 (no quantization)")
    print("Input Layout: NCHW (matching ONNX)")
    print("=" * 60)

    if args.model == 'all':
        models_to_convert = list(MODELS.items())
    else:
        models_to_convert = [(args.model, MODELS[args.model])]

    success_count = 0
    for name, info in models_to_convert:
        print(f"\n{'='*40}")
        print(f"Processing {name} model...")
        print(f"{'='*40}")
        if export_rknn(name, info, args.input_dir, args.output_dir):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{len(models_to_convert)} models converted successfully")
    print("=" * 60)


if __name__ == '__main__':
    main()