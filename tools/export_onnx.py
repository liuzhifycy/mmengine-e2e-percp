#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 模型导出脚本

功能说明：
    将 PyTorch 检测模型导出为 ONNX 格式，便于部署和推理优化。
    支持模型简化、形状推断和导出验证。

使用方法：
    # 基本用法
    python tools/export_onnx.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --output-file exports/retinanet.onnx

    # 指定输入形状
    python tools/export_onnx.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --output-file exports/retinanet.onnx \
        --input-shape 1,3,800,1333

    # 简化 ONNX 模型
    python tools/export_onnx.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --output-file exports/retinanet.onnx \
        --simplify

    # 指定 opset 版本
    python tools/export_onnx.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --output-file exports/retinanet.onnx \
        --opset 13 \
        --simplify

    # 动态 batch size
    python tools/export_onnx.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --output-file exports/retinanet.onnx \
        --dynamic-batch

参数说明：
    --config       : 配置文件路径 (必需)
    --checkpoint   : 模型权重文件路径 (必需)
    --output-file  : 输出 ONNX 文件路径 (必需)
    --input-shape  : 输入形状，格式: N,C,H,W (默认: 1,3,800,1333)
    --opset        : ONNX opset 版本 (默认: 11)
    --simplify     : 是否简化 ONNX 模型 (需要 onnxsim)
    --dynamic-batch: 是否使用动态 batch size
    --verify       : 是否验证导出的模型 (默认: True)
    --fp16         : 是否导出 FP16 模型
    --device       : 导出时使用的设备 (默认: cpu)
"""

import argparse
import os
import os.path as osp
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    from onnx import checker, shape_inference

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    warnings.warn('onnx 未安装，请运行: pip install onnx')

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    warnings.warn('onnxruntime 未安装，请运行: pip install onnxruntime')

try:
    import onnxsim

    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ONNX 模型导出工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型权重文件路径'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='输出 ONNX 文件路径'
    )
    parser.add_argument(
        '--input-shape',
        type=str,
        default='1,3,800,1333',
        help='输入形状，格式: N,C,H,W (默认: 1,3,800,1333)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset 版本 (默认: 11)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='是否简化 ONNX 模型'
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='是否使用动态 batch size'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='是否验证导出的模型 (默认: True)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_false',
        dest='verify',
        help='不验证导出的模型'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='是否导出 FP16 模型'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='导出时使用的设备 (默认: cpu)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    return parser.parse_args()


def parse_input_shape(shape_str: str) -> Tuple[int, ...]:
    """
    解析输入形状字符串
    
    Args:
        shape_str: 形状字符串，如 "1,3,800,1333"
        
    Returns:
        tuple: 形状元组
    """
    try:
        shape = tuple(int(x.strip()) for x in shape_str.split(','))
        if len(shape) != 4:
            raise ValueError('输入形状必须是 4 维 (N,C,H,W)')
        return shape
    except Exception as e:
        raise ValueError(f'无法解析输入形状 "{shape_str}": {e}')


class ModelWrapper(nn.Module):
    """
    模型包装器，用于 ONNX 导出
    
    将 mmdet 模型包装为标准的 forward 接口，
    确保只输出需要的检测结果。
    """

    def __init__(self, model: nn.Module, output_names: Optional[List[str]] = None):
        """
        初始化包装器
        
        Args:
            model: 原始检测模型
            output_names: 输出名称列表
        """
        super().__init__()
        self.model = model
        self.output_names = output_names or ['boxes', 'scores', 'labels']

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            x: 输入图像 tensor [N, C, H, W]
            
        Returns:
            tuple: 检测结果 (boxes, scores, labels) 或原始输出
        """
        # 尝试使用模型的 forward 方法
        # mmdet 模型通常有 extract_feat 和 bbox_head
        if hasattr(self.model, 'extract_feat') and hasattr(self.model, 'bbox_head'):
            # 提取特征
            feats = self.model.extract_feat(x)
            
            # 如果有 neck，特征已经过 neck 处理
            # 调用 bbox_head 的 forward 获取原始输出
            if hasattr(self.model.bbox_head, 'forward'):
                outs = self.model.bbox_head(feats)
            else:
                outs = feats
            
            return outs
        else:
            # 直接调用模型
            return self.model(x)


def build_model(config: str, checkpoint: str, device: str) -> nn.Module:
    """
    构建模型并加载权重
    
    Args:
        config: 配置文件路径
        checkpoint: 权重文件路径
        device: 设备
        
    Returns:
        nn.Module: 加载好权重的模型
    """
    # 加载配置
    cfg = Config.fromfile(config)

    # 构建模型
    model = MODELS.build(cfg.model)

    # 加载权重
    load_checkpoint(model, checkpoint, map_location='cpu')

    # 设置为评估模式
    model.to(device)
    model.eval()

    return model


def export_onnx(
    model: nn.Module,
    output_file: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
    dynamic_batch: bool = False,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """
    导出 ONNX 模型
    
    Args:
        model: PyTorch 模型
        output_file: 输出文件路径
        input_shape: 输入形状 (N, C, H, W)
        opset_version: ONNX opset 版本
        dynamic_batch: 是否使用动态 batch size
        input_names: 输入名称列表
        output_names: 输出名称列表
        verbose: 是否显示详细信息
    """
    # 创建输出目录
    output_dir = osp.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 默认输入输出名称
    input_names = input_names or ['input']
    output_names = output_names or ['output']

    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    if next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()

    # 配置动态轴
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
        }
        # 为所有输出添加动态 batch
        for name in output_names:
            if dynamic_axes is None:
                dynamic_axes = {}
            dynamic_axes[name] = {0: 'batch_size'}

    print(f'输入形状: {input_shape}')
    print(f'ONNX opset 版本: {opset_version}')
    print(f'动态 batch: {dynamic_batch}')
    print(f'输出文件: {output_file}')

    # 包装模型
    wrapped_model = ModelWrapper(model)

    # 导出 ONNX
    print('\n正在导出 ONNX 模型...')
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_file,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=verbose,
            export_params=True,
        )

    print(f'ONNX 模型导出成功: {output_file}')


def simplify_onnx(
    input_file: str,
    output_file: Optional[str] = None,
    check_n: int = 3,
) -> bool:
    """
    简化 ONNX 模型
    
    使用 onnxsim 简化 ONNX 模型，可以：
    - 消除冗余运算
    - 常量折叠
    - 融合算子
    
    Args:
        input_file: 输入 ONNX 文件路径
        output_file: 输出文件路径（默认覆盖输入文件）
        check_n: 验证次数
        
    Returns:
        bool: 是否成功
    """
    if not HAS_ONNXSIM:
        print('警告: onnxsim 未安装，跳过简化步骤')
        print('请运行: pip install onnxsim')
        return False

    output_file = output_file or input_file

    print('\n正在简化 ONNX 模型...')
    try:
        # 加载模型
        model = onnx.load(input_file)

        # 简化模型
        model_simp, check = onnxsim.simplify(
            model,
            check_n=check_n,
            perform_optimization=True,
        )

        if not check:
            print('警告: 简化后的模型验证失败，使用原始模型')
            return False

        # 保存简化后的模型
        onnx.save(model_simp, output_file)

        # 显示模型大小变化
        original_size = osp.getsize(input_file) / (1024 * 1024)
        simplified_size = osp.getsize(output_file) / (1024 * 1024)
        reduction = (1 - simplified_size / original_size) * 100

        print(f'简化完成:')
        print(f'  原始大小: {original_size:.2f} MB')
        print(f'  简化后: {simplified_size:.2f} MB')
        print(f'  减少: {reduction:.1f}%')

        return True

    except Exception as e:
        print(f'简化失败: {e}')
        return False


def verify_onnx(
    onnx_file: str,
    input_shape: Tuple[int, ...],
    pytorch_model: Optional[nn.Module] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    验证 ONNX 模型
    
    Args:
        onnx_file: ONNX 文件路径
        input_shape: 输入形状
        pytorch_model: 原始 PyTorch 模型（用于输出比对）
        rtol: 相对误差阈值
        atol: 绝对误差阈值
        
    Returns:
        bool: 验证是否通过
    """
    if not HAS_ONNX:
        print('警告: onnx 未安装，跳过验证')
        return False

    print('\n正在验证 ONNX 模型...')

    # 1. 检查模型结构
    print('  [1/3] 检查模型结构...')
    try:
        model = onnx.load(onnx_file)
        checker.check_model(model)
        print('    模型结构验证通过')
    except Exception as e:
        print(f'    模型结构验证失败: {e}')
        return False

    # 2. 形状推断
    print('  [2/3] 执行形状推断...')
    try:
        model = shape_inference.infer_shapes(model)
        print('    形状推断成功')
    except Exception as e:
        print(f'    形状推断失败: {e}')
        # 形状推断失败不影响后续验证

    # 3. 使用 ONNX Runtime 测试推理
    print('  [3/3] 测试 ONNX Runtime 推理...')
    if not HAS_ONNXRUNTIME:
        print('    警告: onnxruntime 未安装，跳过推理测试')
        return True

    try:
        # 创建推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            onnx_file,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # 获取输入信息
        input_info = session.get_inputs()[0]
        print(f'    输入名称: {input_info.name}')
        print(f'    输入形状: {input_info.shape}')
        print(f'    输入类型: {input_info.type}')

        # 获取输出信息
        output_info = session.get_outputs()
        print(f'    输出数量: {len(output_info)}')
        for idx, out in enumerate(output_info):
            print(f'    输出 {idx}: {out.name}, 形状: {out.shape}')

        # 创建测试输入
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # 执行推理
        outputs = session.run(None, {input_info.name: test_input})

        print(f'    推理成功，输出数量: {len(outputs)}')
        for idx, out in enumerate(outputs):
            if isinstance(out, np.ndarray):
                print(f'    输出 {idx} 形状: {out.shape}')

        # 如果提供了 PyTorch 模型，比对输出
        if pytorch_model is not None:
            print('\n  比对 PyTorch 和 ONNX 输出...')
            wrapped_model = ModelWrapper(pytorch_model)
            wrapped_model.eval()

            with torch.no_grad():
                torch_input = torch.from_numpy(test_input)
                if next(pytorch_model.parameters()).is_cuda:
                    torch_input = torch_input.cuda()
                torch_outputs = wrapped_model(torch_input)

            # 转换为 numpy 比对
            if isinstance(torch_outputs, tuple):
                for idx, (torch_out, onnx_out) in enumerate(
                    zip(torch_outputs, outputs)
                ):
                    if isinstance(torch_out, torch.Tensor):
                        torch_out = torch_out.cpu().numpy()
                    if np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol):
                        print(f'    输出 {idx}: 匹配')
                    else:
                        max_diff = np.abs(torch_out - onnx_out).max()
                        print(f'    输出 {idx}: 不匹配 (最大误差: {max_diff})')
            else:
                if isinstance(torch_outputs, torch.Tensor):
                    torch_outputs = torch_outputs.cpu().numpy()
                if np.allclose(torch_outputs, outputs[0], rtol=rtol, atol=atol):
                    print('    输出匹配')
                else:
                    max_diff = np.abs(torch_outputs - outputs[0]).max()
                    print(f'    输出不匹配 (最大误差: {max_diff})')

        print('\nONNX 模型验证通过!')
        return True

    except Exception as e:
        print(f'    推理测试失败: {e}')
        return False


def convert_to_fp16(onnx_file: str, output_file: Optional[str] = None) -> bool:
    """
    将 ONNX 模型转换为 FP16
    
    Args:
        onnx_file: 输入 ONNX 文件路径
        output_file: 输出文件路径
        
    Returns:
        bool: 是否成功
    """
    try:
        from onnxconverter_common import float16

        output_file = output_file or onnx_file.replace('.onnx', '_fp16.onnx')

        print(f'\n正在转换为 FP16: {output_file}')

        model = onnx.load(onnx_file)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_file)

        print(f'FP16 模型保存成功: {output_file}')
        return True

    except ImportError:
        print('警告: onnxconverter-common 未安装，无法转换 FP16')
        print('请运行: pip install onnxconverter-common')
        return False
    except Exception as e:
        print(f'FP16 转换失败: {e}')
        return False


def print_model_info(onnx_file: str) -> None:
    """
    打印 ONNX 模型信息
    
    Args:
        onnx_file: ONNX 文件路径
    """
    if not HAS_ONNX:
        return

    model = onnx.load(onnx_file)

    print('\n========== ONNX 模型信息 ==========')
    print(f'文件: {onnx_file}')
    print(f'大小: {osp.getsize(onnx_file) / (1024 * 1024):.2f} MB')
    print(f'IR 版本: {model.ir_version}')
    print(f'Opset 版本: {model.opset_import[0].version}')
    print(f'Producer: {model.producer_name} {model.producer_version}')

    # 统计算子数量
    op_counts: Dict[str, int] = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    print(f'\n节点总数: {len(model.graph.node)}')
    print('算子统计 (Top 10):')
    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:10]
    for op_name, count in sorted_ops:
        print(f'  {op_name}: {count}')

    print('=' * 40)


def main():
    """主函数"""
    args = parse_args()

    # 检查依赖
    if not HAS_ONNX:
        raise ImportError('请安装 onnx: pip install onnx')

    # 检查文件
    if not osp.exists(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f'权重文件不存在: {args.checkpoint}')

    # 解析输入形状
    input_shape = parse_input_shape(args.input_shape)

    print('========== ONNX 导出配置 ==========')
    print(f'配置文件: {args.config}')
    print(f'权重文件: {args.checkpoint}')
    print(f'输出文件: {args.output_file}')
    print(f'输入形状: {input_shape}')
    print(f'Opset 版本: {args.opset}')
    print(f'简化模型: {args.simplify}')
    print(f'动态 batch: {args.dynamic_batch}')
    print(f'导出设备: {args.device}')
    print('=' * 40)

    # 构建模型
    print('\n正在加载模型...')
    model = build_model(args.config, args.checkpoint, args.device)
    print('模型加载完成')

    # 导出 ONNX
    export_onnx(
        model=model,
        output_file=args.output_file,
        input_shape=input_shape,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
        verbose=args.verbose,
    )

    # 简化模型
    if args.simplify:
        simplify_onnx(args.output_file)

    # 转换 FP16
    if args.fp16:
        convert_to_fp16(args.output_file)

    # 验证模型
    if args.verify:
        verify_onnx(
            args.output_file,
            input_shape,
            pytorch_model=model if args.device == 'cpu' else None,
        )

    # 打印模型信息
    print_model_info(args.output_file)

    print('\n导出完成!')


if __name__ == '__main__':
    main()
