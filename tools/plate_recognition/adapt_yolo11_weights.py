#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YOLO11m 预训练权重适配为车牌检测模型
只保留 backbone 和 neck 的权重，head 权重重新初始化
"""

import torch
from pathlib import Path


def adapt_weights(src_path, dst_path, num_classes=1):
    """
    适配预训练权重到单类检测模型
    
    Args:
        src_path: 原始权重路径 (yolo11m_mm.pth)
        dst_path: 输出权重路径
        num_classes: 目标类别数
    """
    print(f'Loading weights from: {src_path}')
    checkpoint = torch.load(src_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 过滤掉 head 中与类别数相关的权重
    adapted_state_dict = {}
    excluded_keys = []
    
    for key, value in state_dict.items():
        # 排除检测头中的分类分支权重 (cv3)
        # 这些权重的输出维度与类别数相关
        if 'bbox_head.cv3' in key:
            excluded_keys.append(key)
            continue
        
        # 保留 backbone、neck 和 box 分支 (cv2) 的权重
        adapted_state_dict[key] = value
    
    print(f'Total keys: {len(state_dict)}')
    print(f'Adapted keys: {len(adapted_state_dict)}')
    print(f'Excluded keys ({len(excluded_keys)}):')
    for k in excluded_keys:
        print(f'  - {k}')
    
    # 保存适配后的权重
    torch.save({'state_dict': adapted_state_dict}, dst_path)
    print(f'Saved adapted weights to: {dst_path}')


def main():
    base_dir = Path('/home/ubuntu/e2e-pecp-pdp/mmengine-lite')
    src_path = base_dir / 'yolo11m_mm.pth'
    dst_path = base_dir / 'checkpoints' / 'yolo11m_plate_pretrain.pth'
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    adapt_weights(src_path, dst_path, num_classes=1)


if __name__ == '__main__':
    main()
