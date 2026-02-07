#!/usr/bin/env python3
"""
将微调后的 PyTorch 权重注入原始 ONNX 模型

这种方法比重新导出更可靠，因为它保持了原始 ONNX 模型的结构，
只是替换了训练后更新的权重。

用法:
    python export_onnx.py --checkpoint checkpoints/stage1_best.pth --output finetuned_hztk_rec.onnx

验证:
    脚本会自动验证导出的 ONNX 模型
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnx2torch import convert


# 车牌字符集 (与 hztk_rec.onnx 一致)
PLATE_CHARS = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", 
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民", 
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", 
    "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", 
    "黑", "领", "使", "澳",
]

BLANK_IDX = 0


def decode_prediction(output: np.ndarray) -> str:
    """解码模型输出"""
    indices = np.argmax(output, axis=-1)
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx != BLANK_IDX and idx != prev_idx:
            if idx < len(PLATE_CHARS):
                chars.append(PLATE_CHARS[idx])
        prev_idx = idx
    return ''.join(chars)


def preprocess_image(image_path: str, img_h: int = 48, img_w: int = 160) -> np.ndarray:
    """预处理图片"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """加载检查点"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best accuracy: {checkpoint.get('best_acc', 'unknown')}")
    else:
        state_dict = checkpoint
    
    return state_dict


def build_pytorch_to_onnx_mapping(onnx_model, pytorch_state_dict):
    """
    建立 PyTorch 参数名到 ONNX initializer 名的映射
    
    onnx2torch 命名规则:
    - backbone 层: backbone.Conv_N.weight -> onnx_initializer_N (大致)
    - initializers 模块: initializers.onnx_initializer_N -> 对应 ONNX 中的某个 initializer
    """
    # 获取 ONNX initializers
    onnx_initializers = {init.name: init for init in onnx_model.graph.initializer}
    
    print(f"\nONNX model has {len(onnx_initializers)} initializers")
    print(f"PyTorch state_dict has {len(pytorch_state_dict)} parameters")
    
    # 按形状和数值匹配
    mapping = {}
    
    # 首先，创建一个基于形状的索引
    shape_to_onnx = {}
    for name, init in onnx_initializers.items():
        arr = numpy_helper.to_array(init)
        shape_key = tuple(arr.shape)
        if shape_key not in shape_to_onnx:
            shape_to_onnx[shape_key] = []
        shape_to_onnx[shape_key].append((name, arr))
    
    # 对每个 PyTorch 参数，找到形状匹配的 ONNX initializer
    matched_onnx = set()
    
    for pt_name, pt_tensor in pytorch_state_dict.items():
        pt_arr = pt_tensor.cpu().numpy()
        shape_key = tuple(pt_arr.shape)
        
        if shape_key in shape_to_onnx:
            candidates = shape_to_onnx[shape_key]
            
            # 找到数值最接近的
            best_match = None
            best_diff = float('inf')
            
            for onnx_name, onnx_arr in candidates:
                if onnx_name in matched_onnx:
                    continue
                diff = np.mean(np.abs(pt_arr - onnx_arr))
                if diff < best_diff:
                    best_diff = diff
                    best_match = onnx_name
            
            if best_match is not None:
                mapping[pt_name] = best_match
                matched_onnx.add(best_match)
    
    print(f"Matched {len(mapping)} parameters")
    
    return mapping


def inject_weights_to_onnx(onnx_model_path: str, pytorch_state_dict: dict, output_path: str):
    """将 PyTorch 权重注入 ONNX 模型"""
    print(f"\nLoading original ONNX model: {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    
    # 建立映射
    mapping = build_pytorch_to_onnx_mapping(onnx_model, pytorch_state_dict)
    
    # 创建 ONNX initializer 名称到索引的映射
    init_name_to_idx = {init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)}
    
    # 更新权重
    updated_count = 0
    for pt_name, onnx_name in mapping.items():
        if onnx_name in init_name_to_idx:
            idx = init_name_to_idx[onnx_name]
            pt_tensor = pytorch_state_dict[pt_name].cpu().numpy()
            
            # 创建新的 initializer
            new_init = numpy_helper.from_array(pt_tensor, name=onnx_name)
            onnx_model.graph.initializer[idx].CopyFrom(new_init)
            updated_count += 1
    
    print(f"Updated {updated_count} weights in ONNX model")
    
    # 保存
    print(f"Saving to: {output_path}")
    onnx.save(onnx_model, output_path)
    
    # 验证
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    return output_path


def verify_finetuned_model(original_onnx: str, finetuned_onnx: str, checkpoint_path: str, 
                           onnx_model_for_pytorch: str, test_images: list):
    """验证微调后的模型"""
    print("\n" + "=" * 60)
    print("Verifying finetuned model...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 PyTorch 模型
    print("\nLoading PyTorch model...")
    onnx_model = onnx.load(onnx_model_for_pytorch)
    pytorch_model = convert(onnx_model)
    
    # 转换 buffers 为 parameters
    if hasattr(pytorch_model, 'initializers'):
        initializers_module = pytorch_model.initializers
        buffers_to_convert = []
        for attr_name in dir(initializers_module):
            if attr_name.startswith('onnx_initializer_'):
                buf = getattr(initializers_module, attr_name)
                if isinstance(buf, torch.Tensor) and buf.numel() > 10:
                    buffers_to_convert.append((attr_name, buf.clone().detach()))
        
        for attr_name, buf_data in buffers_to_convert:
            if attr_name in dict(initializers_module.named_buffers()):
                delattr(initializers_module, attr_name)
            param = nn.Parameter(buf_data, requires_grad=True)
            initializers_module.register_parameter(attr_name, param)
    
    pytorch_model = pytorch_model.to(device)
    
    # 加载微调后的权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    
    # 加载 ONNX 模型
    original_session = ort.InferenceSession(original_onnx, providers=['CPUExecutionProvider'])
    finetuned_session = ort.InferenceSession(finetuned_onnx, providers=['CPUExecutionProvider'])
    
    # 对比结果
    original_correct = 0
    finetuned_correct = 0
    pytorch_correct = 0
    pytorch_onnx_match = 0
    total = 0
    
    print("\n" + "-" * 80)
    print(f"{'Label':<15} {'Original':<15} {'Finetuned':<15} {'PyTorch':<15} {'Match'}")
    print("-" * 80)
    
    for img_path, label in test_images[:20]:  # 只显示前20个
        if not os.path.exists(img_path):
            continue
        
        input_data = preprocess_image(img_path)
        
        # 原始 ONNX
        orig_output = original_session.run(None, {original_session.get_inputs()[0].name: input_data})[0]
        orig_decoded = decode_prediction(orig_output[0])
        
        # 微调 ONNX
        fine_output = finetuned_session.run(None, {finetuned_session.get_inputs()[0].name: input_data})[0]
        fine_decoded = decode_prediction(fine_output[0])
        
        # PyTorch
        with torch.no_grad():
            torch_input = torch.from_numpy(input_data).to(device)
            torch_output = pytorch_model(torch_input).cpu().numpy()
        torch_decoded = decode_prediction(torch_output[0])
        
        orig_match = (orig_decoded == label)
        fine_match = (fine_decoded == label)
        pt_match = (torch_decoded == label)
        pt_onnx_match = (fine_decoded == torch_decoded)
        
        if orig_match:
            original_correct += 1
        if fine_match:
            finetuned_correct += 1
        if pt_match:
            pytorch_correct += 1
        if pt_onnx_match:
            pytorch_onnx_match += 1
        
        total += 1
        
        # 显示结果
        status = "✓" if fine_match else "✗"
        match_status = "=" if pt_onnx_match else "≠"
        print(f"{label:<15} {orig_decoded:<15} {fine_decoded:<15} {torch_decoded:<15} {match_status} {status}")
    
    print("-" * 80)
    
    # 统计剩余样本
    for img_path, label in test_images[20:]:
        if not os.path.exists(img_path):
            continue
        
        input_data = preprocess_image(img_path)
        
        orig_output = original_session.run(None, {original_session.get_inputs()[0].name: input_data})[0]
        orig_decoded = decode_prediction(orig_output[0])
        
        fine_output = finetuned_session.run(None, {finetuned_session.get_inputs()[0].name: input_data})[0]
        fine_decoded = decode_prediction(fine_output[0])
        
        with torch.no_grad():
            torch_input = torch.from_numpy(input_data).to(device)
            torch_output = pytorch_model(torch_input).cpu().numpy()
        torch_decoded = decode_prediction(torch_output[0])
        
        if orig_decoded == label:
            original_correct += 1
        if fine_decoded == label:
            finetuned_correct += 1
        if torch_decoded == label:
            pytorch_correct += 1
        if fine_decoded == torch_decoded:
            pytorch_onnx_match += 1
        
        total += 1
    
    print(f"\nTotal samples: {total}")
    print(f"Original ONNX accuracy:  {original_correct}/{total} = {100*original_correct/total:.2f}%")
    print(f"Finetuned ONNX accuracy: {finetuned_correct}/{total} = {100*finetuned_correct/total:.2f}%")
    print(f"PyTorch accuracy:        {pytorch_correct}/{total} = {100*pytorch_correct/total:.2f}%")
    print(f"PyTorch-ONNX match rate: {pytorch_onnx_match}/{total} = {100*pytorch_onnx_match/total:.2f}%")
    print(f"\nImprovement: {100*(finetuned_correct-original_correct)/total:+.2f}%")
    
    return finetuned_correct / total if total > 0 else 0


def load_test_samples(data_dir: str, max_samples: int = 625):
    """加载测试样本"""
    samples = []
    # 尝试多个可能的文件名
    for filename in ['val_label.txt', 'val_labels.txt']:
        val_label_file = os.path.join(data_dir, filename)
        if os.path.exists(val_label_file):
            break
    else:
        val_label_file = os.path.join(data_dir, 'val_label.txt')
    if os.path.exists(val_label_file):
        with open(val_label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path, label = parts
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(data_dir, img_path)
                    samples.append((img_path, label))
                    if len(samples) >= max_samples:
                        break
    return samples


def main():
    parser = argparse.ArgumentParser(description='Export finetuned model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage1_best.pth',
                        help='Path to finetuned checkpoint')
    parser.add_argument('--onnx-model', type=str, 
                        default='../../../e2e_hztk_deploy_package/hztk_rec.onnx',
                        help='Path to original ONNX model')
    parser.add_argument('--output', type=str, default='finetuned_hztk_rec.onnx',
                        help='Output ONNX path')
    parser.add_argument('--data-dir', type=str, default='../../../finetune_data',
                        help='Data directory for comparison')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with original model')
    parser.add_argument('--max-samples', type=int, default=625,
                        help='Max samples for comparison')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(script_dir, args.checkpoint)
    if not os.path.isabs(args.onnx_model):
        args.onnx_model = os.path.join(script_dir, args.onnx_model)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(script_dir, args.data_dir)
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model not found: {args.onnx_model}")
        return 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载检查点
    state_dict = load_checkpoint(args.checkpoint, device)
    
    # 注入权重
    inject_weights_to_onnx(args.onnx_model, state_dict, args.output)
    
    # 验证
    if args.compare:
        test_samples = load_test_samples(args.data_dir, max_samples=args.max_samples)
        if test_samples:
            verify_finetuned_model(
                args.onnx_model, args.output, args.checkpoint,
                args.onnx_model, test_samples
            )
        else:
            print("No test samples found for comparison")
    
    print("\n" + "=" * 60)
    print(f"Export complete: {args.output}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
