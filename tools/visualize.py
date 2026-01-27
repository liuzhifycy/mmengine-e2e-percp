#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测结果可视化脚本

功能说明：
    使用训练好的检测模型对输入图片进行推理，并将检测结果可视化保存。
    支持单张图片或目录批量处理。

使用方法：
    # 基本用法 - 单张图片推理
    python tools/visualize.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --input demo/test.jpg \
        --output results/

    # 批量处理目录中的图片
    python tools/visualize.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --input demo/images/ \
        --output results/ \
        --score-thr 0.5

    # 指定设备
    python tools/visualize.py \
        --config configs/retinanet/retinanet_r50_fpn.py \
        --checkpoint work_dirs/retinanet/epoch_12.pth \
        --input demo/test.jpg \
        --output results/ \
        --device cuda:0

参数说明：
    --config      : 配置文件路径 (必需)
    --checkpoint  : 模型权重文件路径 (必需)
    --input       : 输入图片路径或图片目录 (必需)
    --output      : 输出保存目录 (必需)
    --score-thr   : 置信度阈值，低于此阈值的检测框不显示 (默认: 0.3)
    --device      : 推理设备，如 'cuda:0' 或 'cpu' (默认: cuda:0)
    --show        : 是否显示可视化窗口 (默认: False)
    --wait-time   : 显示窗口等待时间，0 表示等待按键 (默认: 0)
"""

import argparse
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS, VISUALIZERS
from mmengine.runner import load_checkpoint
from mmengine.visualization import Visualizer

# 支持的图片格式
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='检测结果可视化工具',
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
        '--input',
        type=str,
        required=True,
        help='输入图片路径或图片目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出保存目录'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='置信度阈值 (默认: 0.3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推理设备 (默认: cuda:0)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='是否显示可视化窗口'
    )
    parser.add_argument(
        '--wait-time',
        type=int,
        default=0,
        help='显示窗口等待时间，0 表示等待按键 (默认: 0)'
    )
    return parser.parse_args()


def get_image_list(input_path):
    """
    获取输入图片列表
    
    Args:
        input_path: 图片路径或目录路径
        
    Returns:
        list: 图片路径列表
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() in IMG_EXTENSIONS:
            return [str(input_path)]
        else:
            raise ValueError(f'不支持的图片格式: {input_path.suffix}')
    elif input_path.is_dir():
        # 目录，获取所有图片
        image_list = []
        for ext in IMG_EXTENSIONS:
            image_list.extend(input_path.glob(f'*{ext}'))
            image_list.extend(input_path.glob(f'*{ext.upper()}'))
        image_list = sorted([str(p) for p in image_list])
        if not image_list:
            raise ValueError(f'目录中没有找到图片: {input_path}')
        return image_list
    else:
        raise FileNotFoundError(f'输入路径不存在: {input_path}')


def build_model(config, checkpoint, device):
    """
    构建检测模型
    
    Args:
        config: 配置文件路径或 Config 对象
        checkpoint: 模型权重文件路径
        device: 推理设备
        
    Returns:
        model: 加载好权重的模型
    """
    # 加载配置
    if isinstance(config, str):
        cfg = Config.fromfile(config)
    else:
        cfg = config
    
    # 构建模型
    model = MODELS.build(cfg.model)
    
    # 加载权重
    checkpoint_data = load_checkpoint(model, checkpoint, map_location='cpu')
    
    # 获取类别信息 (如果 checkpoint 中包含)
    if 'meta' in checkpoint_data and 'CLASSES' in checkpoint_data['meta']:
        model.CLASSES = checkpoint_data['meta']['CLASSES']
    elif hasattr(cfg, 'class_names'):
        model.CLASSES = cfg.class_names
    else:
        # 默认使用 COCO 类别
        model.CLASSES = get_coco_classes()
    
    # 设置为评估模式
    model.to(device)
    model.eval()
    
    return model, cfg


def get_coco_classes():
    """获取 COCO 数据集类别名称"""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


def preprocess_image(image_path, device):
    """
    图片预处理
    
    Args:
        image_path: 图片路径
        device: 推理设备
        
    Returns:
        tuple: (原始图片, 预处理后的数据)
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'无法读取图片: {image_path}')
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为 tensor 并归一化
    # 标准的 ImageNet 归一化参数
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    img_normalized = (img_rgb.astype(np.float32) - mean) / std
    
    # HWC -> CHW -> NCHW
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1))
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img, img_tensor


def inference(model, img_tensor, img_shape):
    """
    执行推理
    
    Args:
        model: 检测模型
        img_tensor: 预处理后的图片 tensor
        img_shape: 原始图片尺寸 (H, W)
        
    Returns:
        tuple: (boxes, scores, labels)
    """
    with torch.no_grad():
        # 构造输入数据
        # mmdet 模型通常需要 data_samples 或 batch_inputs
        batch_inputs = img_tensor
        
        # 构造 meta 信息
        batch_data_samples = None
        
        try:
            # 尝试使用 mmdet 的推理接口
            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': img_shape,
                'ori_shape': img_shape,
                'scale_factor': (1.0, 1.0),
                'pad_shape': img_shape,
            })
            batch_data_samples = [data_sample]
            
            # 使用模型的 predict 方法
            results = model.predict(batch_inputs, batch_data_samples)
            
            if results and hasattr(results[0], 'pred_instances'):
                pred = results[0].pred_instances
                boxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                labels = pred.labels.cpu().numpy()
                return boxes, scores, labels
                
        except Exception as e:
            print(f'使用标准推理接口失败: {e}')
            print('尝试使用 forward 方法...')
        
        # 备用方案：直接调用 forward
        try:
            outputs = model(batch_inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # 根据输出格式解析结果
            if isinstance(outputs, dict):
                boxes = outputs.get('boxes', outputs.get('bboxes', []))
                scores = outputs.get('scores', [])
                labels = outputs.get('labels', [])
            else:
                # 假设输出格式为 [N, 5] 或 [N, 6]
                outputs = outputs.cpu().numpy()
                if outputs.shape[-1] == 5:
                    boxes = outputs[:, :4]
                    scores = outputs[:, 4]
                    labels = np.zeros(len(scores), dtype=np.int64)
                elif outputs.shape[-1] == 6:
                    boxes = outputs[:, :4]
                    scores = outputs[:, 4]
                    labels = outputs[:, 5].astype(np.int64)
                else:
                    boxes = outputs[:, :4]
                    scores = np.ones(len(outputs))
                    labels = np.zeros(len(outputs), dtype=np.int64)
            
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
                
            return boxes, scores, labels
            
        except Exception as e:
            print(f'推理失败: {e}')
            return np.array([]), np.array([]), np.array([])


def visualize_results(
    image,
    boxes,
    scores,
    labels,
    class_names,
    score_thr=0.3,
    output_path=None,
    show=False,
    wait_time=0
):
    """
    可视化检测结果
    
    Args:
        image: BGR 格式的原始图片
        boxes: 检测框 [N, 4] (x1, y1, x2, y2)
        scores: 置信度 [N]
        labels: 类别标签 [N]
        class_names: 类别名称列表
        score_thr: 置信度阈值
        output_path: 输出保存路径
        show: 是否显示窗口
        wait_time: 窗口等待时间
    """
    # 过滤低置信度结果
    valid_mask = scores >= score_thr
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    # 复制图片用于绘制
    vis_image = image.copy()
    
    # 颜色调色板
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(len(class_names) + 1, 3), dtype=np.uint8)
    
    # 绘制检测框
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        color = tuple(int(c) for c in palette[label])
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label_idx = int(label)
        if label_idx < len(class_names):
            class_name = class_names[label_idx]
        else:
            class_name = f'class_{label_idx}'
        text = f'{class_name}: {score:.2f}'
        
        # 计算文本尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            vis_image,
            text,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
    
    # 添加统计信息
    info_text = f'Detections: {len(boxes)} (thr={score_thr})'
    cv2.putText(
        vis_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    # 保存结果
    if output_path:
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f'结果已保存: {output_path}')
    
    # 显示窗口
    if show:
        window_name = 'Detection Results'
        cv2.imshow(window_name, vis_image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    
    return vis_image


def main():
    """主函数"""
    args = parse_args()
    
    # 检查设备
    if 'cuda' in args.device and not torch.cuda.is_available():
        print('警告: CUDA 不可用，使用 CPU 进行推理')
        args.device = 'cpu'
    
    print(f'配置文件: {args.config}')
    print(f'权重文件: {args.checkpoint}')
    print(f'推理设备: {args.device}')
    print(f'置信度阈值: {args.score_thr}')
    
    # 检查文件是否存在
    if not osp.exists(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(f'权重文件不存在: {args.checkpoint}')
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 构建模型
    print('正在加载模型...')
    model, cfg = build_model(args.config, args.checkpoint, args.device)
    class_names = getattr(model, 'CLASSES', get_coco_classes())
    print(f'模型加载完成，类别数: {len(class_names)}')
    
    # 获取图片列表
    image_list = get_image_list(args.input)
    print(f'共找到 {len(image_list)} 张图片')
    
    # 遍历处理每张图片
    for idx, image_path in enumerate(image_list):
        print(f'\n[{idx + 1}/{len(image_list)}] 处理: {image_path}')
        
        # 预处理
        image, img_tensor = preprocess_image(image_path, args.device)
        img_shape = image.shape[:2]  # (H, W)
        
        # 推理
        boxes, scores, labels = inference(model, img_tensor, img_shape)
        print(f'  检测到 {len(boxes)} 个目标')
        
        # 可视化
        image_name = osp.basename(image_path)
        output_path = osp.join(args.output, f'vis_{image_name}')
        
        visualize_results(
            image=image,
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=class_names,
            score_thr=args.score_thr,
            output_path=output_path,
            show=args.show,
            wait_time=args.wait_time
        )
    
    print(f'\n可视化完成，结果保存在: {args.output}')


if __name__ == '__main__':
    main()
