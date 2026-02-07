#!/usr/bin/env python3
"""
诊断脚本: 对比 ONNX Runtime 和 onnx2torch 的输出

目的:
1. 验证两者使用相同的预处理和输入
2. 对比最终输出的差异
3. 尝试定位差异出现在模型的哪个层

用法:
    source /home/ubuntu/mambaforge/etc/profile.d/conda.sh && conda activate mmlite
    python diagnose_onnx2torch.py --image <plate_image>
"""

import os
import sys
import cv2
import math
import numpy as np
import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from onnx2torch import convert

# 字符集
PLATE_CHARS = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民", 
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", 
    "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", 
    "黑", "领", "使", "澳",
]


def preprocess_image(image: np.ndarray, target_h: int = 48, target_w: int = 160) -> np.ndarray:
    """
    预处理图像 (与参考实现一致)
    
    来自 lightweight_plate_recognizer.py 的 encode_image_for_rec
    """
    h, w = image.shape[:2]
    ratio = w / float(h)
    resized_w = max(int(math.ceil(target_h * ratio)), 48)
    resized_w = min(resized_w, target_w)
    
    resized = cv2.resize(image, (resized_w, target_h))
    resized = resized.astype(np.float32)
    
    # 关键: 先 transpose 再归一化
    resized = (resized.transpose((2, 0, 1)) - 127.5) / 127.5
    
    # 右侧零填充
    padded = np.zeros((3, target_h, target_w), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    
    return np.expand_dims(padded, 0)  # (1, 3, 48, 160)


def ctc_decode(logits: np.ndarray, chars=PLATE_CHARS, blank_idx=0):
    """CTC greedy 解码"""
    if logits.ndim == 3:
        logits = logits[0]  # (T, C)
    
    indices = np.argmax(logits, axis=1)
    probs = np.max(logits, axis=1)
    
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


def analyze_model_structure(onnx_model):
    """分析 ONNX 模型结构"""
    print("\n" + "=" * 60)
    print("ONNX 模型结构分析")
    print("=" * 60)
    
    # 输入
    print("\n输入:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    # 输出
    print("\n输出:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    # Initializers (权重)
    print(f"\nInitializers (权重): {len(onnx_model.graph.initializer)} 个")
    
    # 统计算子类型
    op_types = {}
    for node in onnx_model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    
    print(f"\n算子统计 (共 {len(onnx_model.graph.node)} 个):")
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    
    # 检查最后几个节点
    print("\n最后 5 个节点:")
    for node in onnx_model.graph.node[-5:]:
        print(f"  {node.op_type}: inputs={list(node.input)[:3]}... -> outputs={list(node.output)}")
    
    return op_types


def compare_outputs(onnx_path: str, image_path: str, device='cuda'):
    """对比 ONNX Runtime 和 onnx2torch 的输出"""
    
    print("\n" + "=" * 60)
    print("ONNX Runtime vs onnx2torch 输出对比")
    print("=" * 60)
    
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 无法读取图像 {image_path}")
        return
    print(f"\n图像: {image_path}")
    print(f"  尺寸: {image.shape}")
    
    # 2. 预处理
    input_tensor = preprocess_image(image)
    print(f"\n预处理后:")
    print(f"  形状: {input_tensor.shape}")
    print(f"  范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"  均值: {input_tensor.mean():.6f}")
    print(f"  标准差: {input_tensor.std():.6f}")
    
    # 3. ONNX Runtime 推理
    print("\n" + "-" * 40)
    print("ONNX Runtime 推理")
    print("-" * 40)
    
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    ort_output = session.run([output_name], {input_name: input_tensor})[0]
    print(f"  输出形状: {ort_output.shape}")  # 应该是 (1, 20, 78)
    print(f"  输出范围: [{ort_output.min():.6f}, {ort_output.max():.6f}]")
    print(f"  输出均值: {ort_output.mean():.6f}")
    print(f"  输出和: {ort_output.sum(axis=-1)[0, :5]}...")  # 每个位置的和应该接近1 (softmax)
    
    # 检查是否已经过 softmax
    row_sums = ort_output.sum(axis=-1)
    is_softmax = np.allclose(row_sums, 1.0, atol=0.01)
    print(f"  已过 Softmax: {is_softmax} (行和: {row_sums[0, 0]:.4f})")
    
    # 解码
    ort_text, ort_conf = ctc_decode(ort_output)
    print(f"\n  解码结果: {ort_text}")
    print(f"  置信度: {ort_conf:.4f}")
    
    # 显示前几个位置的 top-3 预测
    print("\n  前 5 个位置的 Top-3 预测:")
    for t in range(min(5, ort_output.shape[1])):
        probs = ort_output[0, t]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(PLATE_CHARS[i], probs[i]) for i in top3_idx]
        print(f"    位置 {t}: {top3}")
    
    # 4. onnx2torch 推理
    print("\n" + "-" * 40)
    print("onnx2torch 推理")
    print("-" * 40)
    
    onnx_model = onnx.load(onnx_path)
    torch_model = convert(onnx_model)
    torch_model = torch_model.to(device)
    torch_model.eval()
    
    # 打印模型结构摘要
    total_params = sum(p.numel() for p in torch_model.parameters())
    total_buffers = sum(b.numel() for b in torch_model.buffers())
    print(f"  参数数量: {total_params:,}")
    print(f"  Buffer 数量: {total_buffers:,}")
    
    # 转换输入
    torch_input = torch.from_numpy(input_tensor).to(device)
    
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    
    torch_output_np = torch_output.cpu().numpy()
    print(f"\n  输出形状: {torch_output_np.shape}")
    print(f"  输出范围: [{torch_output_np.min():.6f}, {torch_output_np.max():.6f}]")
    print(f"  输出均值: {torch_output_np.mean():.6f}")
    print(f"  输出和: {torch_output_np.sum(axis=-1)[0, :5]}...")
    
    # 检查是否已经过 softmax
    row_sums_torch = torch_output_np.sum(axis=-1)
    is_softmax_torch = np.allclose(row_sums_torch, 1.0, atol=0.01)
    print(f"  已过 Softmax: {is_softmax_torch} (行和: {row_sums_torch[0, 0]:.4f})")
    
    # 解码
    torch_text, torch_conf = ctc_decode(torch_output_np)
    print(f"\n  解码结果: {torch_text}")
    print(f"  置信度: {torch_conf:.4f}")
    
    # 显示前几个位置的 top-3 预测
    print("\n  前 5 个位置的 Top-3 预测:")
    for t in range(min(5, torch_output_np.shape[1])):
        probs = torch_output_np[0, t]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(PLATE_CHARS[i], probs[i]) for i in top3_idx]
        print(f"    位置 {t}: {top3}")
    
    # 5. 对比差异
    print("\n" + "-" * 40)
    print("输出差异分析")
    print("-" * 40)
    
    diff = np.abs(ort_output - torch_output_np)
    print(f"  绝对差异 - 最大: {diff.max():.6f}")
    print(f"  绝对差异 - 平均: {diff.mean():.6f}")
    print(f"  绝对差异 - 标准差: {diff.std():.6f}")
    
    # 相对差异
    rel_diff = diff / (np.abs(ort_output) + 1e-10)
    print(f"  相对差异 - 最大: {rel_diff.max():.6f}")
    print(f"  相对差异 - 平均: {rel_diff.mean():.6f}")
    
    # 比较 argmax 是否一致
    ort_argmax = np.argmax(ort_output, axis=-1)
    torch_argmax = np.argmax(torch_output_np, axis=-1)
    argmax_match = (ort_argmax == torch_argmax).mean()
    print(f"\n  Argmax 一致率: {argmax_match * 100:.1f}%")
    
    if argmax_match < 1.0:
        mismatch_pos = np.where(ort_argmax[0] != torch_argmax[0])[0]
        print(f"  不一致的位置: {mismatch_pos[:10]}...")
        for pos in mismatch_pos[:5]:
            ort_idx = ort_argmax[0, pos]
            torch_idx = torch_argmax[0, pos]
            print(f"    位置 {pos}: ORT={PLATE_CHARS[ort_idx]} ({ort_idx}), "
                  f"Torch={PLATE_CHARS[torch_idx]} ({torch_idx})")
    
    # 6. 检查模型内部结构
    print("\n" + "-" * 40)
    print("onnx2torch 模型内部结构")
    print("-" * 40)
    
    # 打印模型的子模块
    print("\n  顶层模块:")
    for name, module in torch_model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        buffer_count = sum(b.numel() for b in module.buffers())
        print(f"    {name}: params={param_count:,}, buffers={buffer_count:,}")
    
    # 检查 initializers 模块
    if hasattr(torch_model, 'initializers'):
        init_module = torch_model.initializers
        print(f"\n  Initializers 模块:")
        
        # 统计 parameters 和 buffers
        init_params = list(init_module.named_parameters())
        init_buffers = list(init_module.named_buffers())
        print(f"    Parameters: {len(init_params)} 个")
        print(f"    Buffers: {len(init_buffers)} 个")
        
        # 打印一些 buffer 信息
        if init_buffers:
            print(f"\n    部分 Buffers (前5个):")
            for name, buf in init_buffers[:5]:
                print(f"      {name}: shape={list(buf.shape)}, dtype={buf.dtype}")
    
    return {
        'ort_text': ort_text,
        'ort_conf': ort_conf,
        'torch_text': torch_text,
        'torch_conf': torch_conf,
        'diff_max': diff.max(),
        'diff_mean': diff.mean(),
        'argmax_match': argmax_match,
    }


def trace_intermediate_outputs(onnx_path: str, image_path: str, device='cuda'):
    """
    追踪中间层输出，找出 ONNX Runtime 和 onnx2torch 的差异点
    """
    print("\n" + "=" * 60)
    print("中间层输出追踪")
    print("=" * 60)
    
    # 读取图像和预处理
    image = cv2.imread(image_path)
    input_tensor = preprocess_image(image)
    
    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    
    # 获取所有中间节点的名称
    intermediate_names = []
    for node in onnx_model.graph.node:
        for output in node.output:
            intermediate_names.append(output)
    
    print(f"\n共 {len(intermediate_names)} 个中间节点")
    
    # 选择一些关键节点来追踪
    # 通常关注: 第一个 Conv 后, 最后一个 Conv 前, Softmax 前后
    key_nodes = []
    
    # 找第一个和最后一个 Conv/MatMul
    for node in onnx_model.graph.node:
        if node.op_type in ['Conv', 'MatMul', 'Softmax', 'Reshape']:
            key_nodes.append((node.op_type, node.output[0]))
    
    print(f"\n关键节点 (前10个):")
    for op_type, name in key_nodes[:10]:
        print(f"  {op_type}: {name}")
    
    # 使用 ONNX Runtime 获取中间层输出
    print("\n追踪关键中间层...")
    
    # 添加中间输出
    model_with_outputs = onnx_model
    for op_type, name in key_nodes[:5]:  # 只追踪前 5 个
        try:
            # 创建输出
            from onnx import helper, TensorProto
            intermediate_output = helper.make_tensor_value_info(
                name, TensorProto.FLOAT, None
            )
            model_with_outputs.graph.output.append(intermediate_output)
        except:
            pass
    
    # 保存临时模型
    temp_model_path = '/tmp/temp_model_with_outputs.onnx'
    onnx.save(model_with_outputs, temp_model_path)
    
    # 用 ONNX Runtime 推理
    session = ort.InferenceSession(temp_model_path, providers=['CPUExecutionProvider'])
    output_names = [o.name for o in session.get_outputs()]
    
    print(f"\n输出节点: {output_names}")
    
    results = session.run(output_names, {session.get_inputs()[0].name: input_tensor})
    
    print("\n中间层输出:")
    for name, result in zip(output_names, results):
        print(f"  {name}: shape={result.shape}, range=[{result.min():.4f}, {result.max():.4f}]")
    
    # 清理
    os.remove(temp_model_path)


def check_model_weights(onnx_path: str, device='cuda'):
    """检查模型权重是否正确加载"""
    print("\n" + "=" * 60)
    print("模型权重检查")
    print("=" * 60)
    
    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    
    # 获取所有 initializers
    onnx_initializers = {init.name: onnx.numpy_helper.to_array(init) 
                         for init in onnx_model.graph.initializer}
    
    print(f"\nONNX Initializers: {len(onnx_initializers)} 个")
    
    # 打印前几个的统计信息
    print("\n部分 Initializers 统计:")
    for i, (name, weights) in enumerate(onnx_initializers.items()):
        if i >= 10:
            break
        print(f"  {name}: shape={weights.shape}, "
              f"range=[{weights.min():.4f}, {weights.max():.4f}], "
              f"mean={weights.mean():.4f}")
    
    # 转换模型
    torch_model = convert(onnx_model)
    torch_model = torch_model.to(device)
    
    # 检查 torch 模型中的权重
    print("\n\nonnx2torch 模型权重对比:")
    
    # 获取所有 parameters 和 buffers
    torch_params = dict(torch_model.named_parameters())
    torch_buffers = dict(torch_model.named_buffers())
    
    print(f"  Parameters: {len(torch_params)} 个")
    print(f"  Buffers: {len(torch_buffers)} 个")
    
    # 检查 initializers 模块
    if hasattr(torch_model, 'initializers'):
        init_module = torch_model.initializers
        
        # 对比一些权重
        matched = 0
        mismatched = 0
        
        for attr_name in dir(init_module):
            if attr_name.startswith('onnx_initializer_'):
                torch_weight = getattr(init_module, attr_name)
                if isinstance(torch_weight, torch.Tensor):
                    # 找到对应的 ONNX initializer
                    # onnx2torch 使用 onnx_initializer_N 命名，需要找到原始名称
                    idx = int(attr_name.split('_')[-1])
                    
                    # 获取原始权重名称
                    if idx < len(onnx_model.graph.initializer):
                        orig_init = onnx_model.graph.initializer[idx]
                        orig_name = orig_init.name
                        orig_weights = onnx.numpy_helper.to_array(orig_init)
                        
                        torch_np = torch_weight.cpu().numpy()
                        
                        if orig_weights.shape == torch_np.shape:
                            diff = np.abs(orig_weights - torch_np).max()
                            if diff < 1e-6:
                                matched += 1
                            else:
                                mismatched += 1
                                if mismatched <= 5:
                                    print(f"\n  权重不匹配: {orig_name}")
                                    print(f"    ONNX: shape={orig_weights.shape}, mean={orig_weights.mean():.6f}")
                                    print(f"    Torch: shape={torch_np.shape}, mean={torch_np.mean():.6f}")
                                    print(f"    最大差异: {diff:.6f}")
                        else:
                            mismatched += 1
                            if mismatched <= 5:
                                print(f"\n  形状不匹配: {orig_name}")
                                print(f"    ONNX: {orig_weights.shape}")
                                print(f"    Torch: {torch_np.shape}")
        
        print(f"\n权重对比结果:")
        print(f"  匹配: {matched}")
        print(f"  不匹配: {mismatched}")


def test_simple_forward(onnx_path: str, device='cuda'):
    """测试简单的前向传播，使用固定输入"""
    print("\n" + "=" * 60)
    print("固定输入测试")
    print("=" * 60)
    
    # 创建固定输入 (全 0.5)
    fixed_input = np.full((1, 3, 48, 160), 0.5, dtype=np.float32)
    print(f"\n固定输入: shape={fixed_input.shape}, value=0.5")
    
    # ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    ort_output = session.run([output_name], {input_name: fixed_input})[0]
    print(f"\nONNX Runtime 输出:")
    print(f"  shape: {ort_output.shape}")
    print(f"  前3个位置的 argmax: {np.argmax(ort_output[0, :3], axis=-1)}")
    print(f"  前3个位置的 max prob: {np.max(ort_output[0, :3], axis=-1)}")
    
    # onnx2torch
    onnx_model = onnx.load(onnx_path)
    torch_model = convert(onnx_model)
    torch_model = torch_model.to(device)
    torch_model.eval()
    
    torch_input = torch.from_numpy(fixed_input).to(device)
    
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    
    torch_output_np = torch_output.cpu().numpy()
    print(f"\nonnx2torch 输出:")
    print(f"  shape: {torch_output_np.shape}")
    print(f"  前3个位置的 argmax: {np.argmax(torch_output_np[0, :3], axis=-1)}")
    print(f"  前3个位置的 max prob: {np.max(torch_output_np[0, :3], axis=-1)}")
    
    # 差异
    diff = np.abs(ort_output - torch_output_np)
    print(f"\n差异:")
    print(f"  最大: {diff.max():.6f}")
    print(f"  平均: {diff.mean():.6f}")


def main():
    parser = argparse.ArgumentParser(description='诊断 onnx2torch 转换问题')
    parser.add_argument('--onnx', type=str, 
                        default='../../../e2e_hztk_deploy_package/hztk_rec.onnx',
                        help='ONNX 模型路径')
    parser.add_argument('--image', type=str, default=None,
                        help='测试图像路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.onnx):
        args.onnx = os.path.normpath(os.path.join(script_dir, args.onnx))
    
    print(f"ONNX 模型: {args.onnx}")
    print(f"设备: {args.device}")
    
    # 1. 分析模型结构
    onnx_model = onnx.load(args.onnx)
    analyze_model_structure(onnx_model)
    
    # 2. 检查权重
    check_model_weights(args.onnx, args.device)
    
    # 3. 固定输入测试
    test_simple_forward(args.onnx, args.device)
    
    # 4. 如果提供了图像，进行完整对比
    if args.image:
        if not os.path.isabs(args.image):
            args.image = os.path.normpath(os.path.join(script_dir, args.image))
        
        compare_outputs(args.onnx, args.image, args.device)
        
    else:
        # 尝试找一个测试图像
        finetune_dir = os.path.join(script_dir, '../../../finetune_data/train')
        if os.path.exists(finetune_dir):
            images = [f for f in os.listdir(finetune_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                test_image = os.path.join(finetune_dir, images[0])
                print(f"\n使用测试图像: {test_image}")
                compare_outputs(args.onnx, test_image, args.device)


if __name__ == '__main__':
    main()
