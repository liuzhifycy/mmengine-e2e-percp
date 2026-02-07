#!/usr/bin/env python3
"""
PyTorch 车牌识别模型微调训练脚本

特性:
1. 从 ONNX 模型转换为 PyTorch
2. 支持分阶段微调 (冻结backbone -> 全参数)
3. CTC Loss 训练
4. 完整的 TensorBoard 日志记录
5. 支持断点续训

用法:
    # 阶段一: 冻结 backbone，只训练 head
    python finetune_plate_rec.py --stage 1 --epochs 10
    
    # 阶段二: 全参数微调
    python finetune_plate_rec.py --stage 2 --epochs 20 --resume checkpoints/stage1_best.pth

TensorBoard:
    tensorboard --logdir runs/
"""

import os
import sys
import cv2
import json
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import onnx
from onnx2torch import convert
from tqdm import tqdm


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

# 字符到索引的映射
CHAR2IDX = {char: idx for idx, char in enumerate(PLATE_CHARS)}
BLANK_IDX = 0


class PlateDataset(Dataset):
    """车牌数据集"""
    
    def __init__(self, label_file: str, data_dir: str, img_h: int = 48, img_w: int = 160,
                 augment: bool = False):
        """
        Args:
            label_file: 标签文件路径 (格式: image_path\tplate_number)
            data_dir: 数据目录
            img_h: 图片高度
            img_w: 图片宽度
            augment: 是否数据增强
        """
        self.data_dir = data_dir
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        
        # 加载标签
        self.samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                image_path, plate_number = parts
                # 转换为绝对路径
                if not os.path.isabs(image_path):
                    image_path = os.path.join(data_dir, image_path)
                self.samples.append((image_path, plate_number))
        
        print(f"Loaded {len(self.samples)} samples from {label_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def encode_label(self, plate_number: str) -> list:
        """将车牌号编码为索引列表"""
        indices = []
        for char in plate_number:
            if char in CHAR2IDX:
                indices.append(CHAR2IDX[char])
            else:
                # 未知字符跳过
                print(f"Warning: Unknown char '{char}' in plate '{plate_number}'")
        return indices
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """预处理图片 (与 ONNX 推理时一致)
        
        重要: 必须与 lightweight_plate_recognizer.py 中的 encode_image_for_rec 完全一致!
        步骤:
          1. 按宽高比等比缩放到 target_h 高度
          2. 先 transpose (HWC -> CHW) 再归一化到 [-1, 1]
          3. 右侧零填充到 target_w 宽度
        """
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        # 使用 ceil 计算目标宽度 (与参考实现一致)
        resized_w = max(int(math.ceil(self.img_h * ratio)), 48)
        resized_w = min(resized_w, self.img_w)
        
        # Resize
        resized = cv2.resize(img, (resized_w, self.img_h))
        resized = resized.astype(np.float32)
        
        # 关键: 先 transpose 再归一化 (与参考实现一致)
        resized = (resized.transpose((2, 0, 1)) - 127.5) / 127.5
        
        # 右侧零填充
        padded = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        
        return padded
    
    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """数据增强"""
        # 随机亮度调整
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # 随机对比度调整
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # 随机高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 10, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # 随机模糊
        if random.random() > 0.8:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        return img
    
    def __getitem__(self, idx):
        image_path, plate_number = self.samples[idx]
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            # 返回空白图片
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        
        # 数据增强
        if self.augment:
            img = self.augment_image(img)
        
        # 预处理
        img = self.preprocess(img)
        
        # 编码标签
        label = self.encode_label(plate_number)
        
        return torch.FloatTensor(img), label, plate_number


def collate_fn(batch):
    """自定义 collate 函数，处理变长标签"""
    images = []
    labels = []
    label_lengths = []
    plate_numbers = []
    
    for img, label, plate_number in batch:
        images.append(img)
        labels.extend(label)
        label_lengths.append(len(label))
        plate_numbers.append(plate_number)
    
    images = torch.stack(images, dim=0)
    labels = torch.IntTensor(labels)
    label_lengths = torch.IntTensor(label_lengths)
    
    return images, labels, label_lengths, plate_numbers


def decode_prediction(output: torch.Tensor) -> list:
    """解码模型输出
    
    Args:
        output: [batch, seq_len, num_classes] 模型输出
    
    Returns:
        list of decoded plate numbers
    """
    # output: [B, T, C]
    batch_size = output.size(0)
    output = output.detach().cpu().numpy()
    
    results = []
    for b in range(batch_size):
        pred = output[b]  # [T, C]
        indices = np.argmax(pred, axis=-1)
        
        # CTC 解码: 去重 + 去blank
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx != BLANK_IDX and idx != prev_idx:
                if idx < len(PLATE_CHARS):
                    chars.append(PLATE_CHARS[idx])
            prev_idx = idx
        
        results.append(''.join(chars))
    
    return results


def compute_accuracy(predictions: list, targets: list) -> tuple:
    """计算准确率
    
    Returns:
        (exact_match_accuracy, char_accuracy)
    """
    correct = 0
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
        
        # 字符级准确率
        for i, c in enumerate(target):
            total_chars += 1
            if i < len(pred) and pred[i] == c:
                correct_chars += 1
    
    exact_acc = correct / len(predictions) if predictions else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    
    return exact_acc, char_acc


class PlateRecTrainer:
    """车牌识别训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建输出目录
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"stage{args.stage}_{timestamp}"
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, self.run_name))
        print(f"TensorBoard logs: {os.path.join(args.log_dir, self.run_name)}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 数据集
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 损失函数
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        
        # 优化器和调度器
        self.optimizer, self.scheduler = self._create_optimizer()
        
        # 训练状态
        self.start_epoch = 0
        self.best_acc = 0.0
        self.global_step = 0
        
        # 恢复训练
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _load_model(self):
        """加载并转换 ONNX 模型"""
        print(f"Loading ONNX model from {self.args.onnx_model}")
        
        onnx_model = onnx.load(self.args.onnx_model)
        model = convert(onnx_model)
        
        # 关键: 将 ONNX initializers (buffers) 转换为可训练参数
        # onnx2torch 将 SVTR transformer 和 FC head 的权重存储为 buffers，需要转换
        model = self._convert_buffers_to_params(model)
        
        model = model.to(self.device)
        
        # 打印模型结构摘要
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 阶段一: 冻结 backbone
        if self.args.stage == 1:
            self._freeze_backbone(model)
        
        return model
    
    def _convert_buffers_to_params(self, model):
        """将 ONNX initializers (buffers) 转换为可训练参数
        
        onnx2torch 将 SVTR transformer 和 FC head 的权重存储为 buffers (不可训练)，
        我们需要将它们转换为 nn.Parameter 以便进行微调。
        """
        print("Converting ONNX initializer buffers to trainable parameters...")
        
        # 找到 initializers 模块
        if hasattr(model, 'initializers'):
            initializers_module = model.initializers
            
            # 收集需要转换的 buffer 名称和数据
            buffers_to_convert = []
            for attr_name in dir(initializers_module):
                if attr_name.startswith('onnx_initializer_'):
                    buf = getattr(initializers_module, attr_name)
                    if isinstance(buf, torch.Tensor) and buf.numel() > 10:
                        buffers_to_convert.append((attr_name, buf.clone().detach()))
            
            # 删除旧的 buffers 并注册为 parameters
            converted_count = 0
            for attr_name, buf_data in buffers_to_convert:
                # 删除 buffer
                if attr_name in dict(initializers_module.named_buffers()):
                    delattr(initializers_module, attr_name)
                
                # 创建可训练参数并注册
                param = nn.Parameter(buf_data, requires_grad=True)
                initializers_module.register_parameter(attr_name, param)
                converted_count += 1
            
            print(f"Converted {converted_count} buffers to trainable parameters")
            
            # 验证转换
            new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after conversion: {new_trainable:,}")
        else:
            print("Warning: No 'initializers' module found in model")
        
        return model
    
    def _freeze_backbone(self, model):
        """冻结 backbone 层，只训练 neck 层
        
        注意: onnx2torch 转换后，SVTR transformer 和 head 的权重被存储为常量，
        只有卷积层（backbone 和 neck 中的 Conv）是可训练参数。
        
        策略: 冻结 backbone 的卷积层，保留 neck 的卷积层可训练。
        """
        print("Freezing backbone layers (keeping neck trainable)...")
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            # 冻结 backbone 层，保留 neck 层
            if name.startswith('backbone'):
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen layers: {frozen_count}, Trainable layers: {trainable_count}")
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        # 根据参数决定是否使用数据增强
        use_augment = not getattr(self.args, 'no_augmentation', False)
        if not use_augment:
            print("Data augmentation DISABLED")
        
        train_dataset = PlateDataset(
            label_file=os.path.join(self.args.data_dir, 'train_label.txt'),
            data_dir=self.args.data_dir,
            augment=use_augment
        )
        
        val_dataset = PlateDataset(
            label_file=os.path.join(self.args.data_dir, 'val_label.txt'),
            data_dir=self.args.data_dir,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """创建优化器和学习率调度器"""
        # 只优化可训练参数
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.args.stage == 1:
            # 阶段一: 使用指定学习率
            lr = self.args.lr
        else:
            # 阶段二: 较小学习率
            lr = self.args.lr * 0.1
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=self.args.weight_decay)
        
        # 创建带 warmup 的余弦退火调度器
        total_steps = self.args.epochs * len(self.train_loader)
        warmup_steps = getattr(self.args, 'warmup_epochs', 0) * len(self.train_loader)
        min_lr = getattr(self.args, 'min_lr', lr * 0.01)
        
        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                # Warmup 阶段：线性增加
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 余弦退火阶段
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # 确保不低于 min_lr
                return max(min_lr / lr, cosine_decay)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return optimizer, scheduler
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Resumed from epoch {self.start_epoch}, best_acc: {self.best_acc:.4f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'global_step': self.global_step,
            'args': vars(self.args),
        }
        
        # 保存最新
        latest_path = self.checkpoint_dir / f'stage{self.args.stage}_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / f'stage{self.args.stage}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # 每隔一定 epoch 保存
        if (epoch + 1) % self.args.save_interval == 0:
            epoch_path = self.checkpoint_dir / f'stage{self.args.stage}_epoch{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (images, labels, label_lengths, plate_numbers) in enumerate(pbar):
            images = images.to(self.device)
            # CTC loss 要求 labels 和 lengths 在 CPU 上
            # labels = labels.to(self.device)
            # label_lengths = label_lengths.to(self.device)
            
            # 前向传播
            outputs = self.model(images)  # [B, T, C]
            
            # CTC Loss 需要 log_probs
            # 注意: ONNX 模型输出已经过 Softmax (是概率分布)，直接取 log 即可
            # 不要再做 log_softmax，否则会造成"双重 Softmax"，导致梯度被压得极小
            log_probs = torch.log(outputs.clamp(min=1e-10))
            log_probs = log_probs.permute(1, 0, 2)  # [T, B, C]
            
            # 输入长度 (所有样本相同) - 也需要在 CPU 上
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.int32)
            
            # 计算损失 - log_probs 保持在 GPU 上以保持梯度传播
            # PyTorch CTCLoss 支持 log_probs 在 GPU，targets/lengths 在 CPU
            # 确保 labels 和 lengths 在 CPU 上且为正确类型
            labels_cpu = labels.cpu().to(torch.int32) if labels.device.type != 'cpu' else labels.to(torch.int32)
            label_lengths_cpu = label_lengths.cpu().to(torch.int32) if label_lengths.device.type != 'cpu' else label_lengths.to(torch.int32)
            input_lengths_cpu = input_lengths.cpu().to(torch.int32) if input_lengths.device.type != 'cpu' else input_lengths.to(torch.int32)
            
            loss = self.criterion(log_probs, labels_cpu, input_lengths_cpu, label_lengths_cpu)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            
            # 解码预测
            predictions = decode_prediction(outputs)
            all_predictions.extend(predictions)
            all_targets.extend(plate_numbers)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # TensorBoard 记录 (每 N 步)
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.global_step)
            
            self.global_step += 1
        
        # 计算 epoch 统计
        avg_loss = total_loss / len(self.train_loader)
        exact_acc, char_acc = compute_accuracy(all_predictions, all_targets)
        
        # TensorBoard epoch 统计
        self.writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Exact_Accuracy', exact_acc, epoch)
        self.writer.add_scalar('Train/Char_Accuracy', char_acc, epoch)
        
        print(f"Train - Loss: {avg_loss:.4f}, Exact Acc: {exact_acc:.4f}, Char Acc: {char_acc:.4f}")
        
        return avg_loss, exact_acc
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 用于 TensorBoard 显示样例
        sample_images = []
        sample_preds = []
        sample_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")
        
        for batch_idx, (images, labels, label_lengths, plate_numbers) in enumerate(pbar):
            images = images.to(self.device)
            # CTC loss 要求 labels 和 lengths 在 CPU 上
            # labels = labels.to(self.device)
            # label_lengths = label_lengths.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # CTC Loss
            # 注意: ONNX 模型输出已经过 Softmax，直接取 log
            log_probs = torch.log(outputs.clamp(min=1e-10))
            log_probs = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.int32)
            
            loss = self.criterion(log_probs.cpu(), labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # 解码预测
            predictions = decode_prediction(outputs)
            all_predictions.extend(predictions)
            all_targets.extend(plate_numbers)
            
            # 收集样例 (前几个 batch)
            if batch_idx < 2:
                sample_images.extend(images[:4].cpu())
                sample_preds.extend(predictions[:4])
                sample_targets.extend(plate_numbers[:4])
        
        # 计算统计
        avg_loss = total_loss / len(self.val_loader)
        exact_acc, char_acc = compute_accuracy(all_predictions, all_targets)
        
        # TensorBoard 记录
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Exact_Accuracy', exact_acc, epoch)
        self.writer.add_scalar('Val/Char_Accuracy', char_acc, epoch)
        
        # 记录预测样例
        for i, (img, pred, target) in enumerate(zip(sample_images[:8], sample_preds[:8], sample_targets[:8])):
            # 反归一化图片
            img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
            img_np = ((img_np * 127.5) + 127.5).clip(0, 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # 添加到 TensorBoard
            self.writer.add_image(
                f'Val/Sample_{i}_GT_{target}_Pred_{pred}',
                img_np,
                epoch,
                dataformats='HWC'
            )
        
        # 记录混淆样例 (错误预测)
        wrong_samples = [(p, t) for p, t in zip(all_predictions, all_targets) if p != t]
        if wrong_samples:
            wrong_text = "\n".join([f"GT: {t} -> Pred: {p}" for p, t in wrong_samples[:20]])
            self.writer.add_text('Val/Wrong_Predictions', wrong_text, epoch)
        
        print(f"Val - Loss: {avg_loss:.4f}, Exact Acc: {exact_acc:.4f}, Char Acc: {char_acc:.4f}")
        
        return avg_loss, exact_acc
    
    def train(self):
        """主训练循环"""
        # Eval-only 模式
        if getattr(self.args, 'eval_only', False):
            print("\n=== Evaluation Only Mode ===")
            val_loss, val_acc = self.validate(0)
            print(f"Validation - Loss: {val_loss:.4f}, Exact Acc: {val_acc:.4f}")
            self.writer.close()
            return
        
        print(f"\nStarting training stage {self.args.stage}")
        print(f"Epochs: {self.args.epochs}, Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
        if warmup_epochs > 0:
            print(f"Warmup epochs: {warmup_epochs}")
        print("=" * 60)
        
        # Early stopping 参数
        early_stopping_patience = getattr(self.args, 'early_stopping', 0)
        no_improvement_count = 0
        
        # 记录超参数
        self.writer.add_hparams(
            {
                'stage': self.args.stage,
                'epochs': self.args.epochs,
                'batch_size': self.args.batch_size,
                'lr': self.args.lr,
                'weight_decay': self.args.weight_decay,
            },
            {}
        )
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 保存检查点
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            self._save_checkpoint(epoch, is_best)
            
            print(f"Epoch {epoch+1} - Best Acc: {self.best_acc:.4f}")
            
            # Early stopping
            if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                print(f"Early stopping triggered after {no_improvement_count} epochs without improvement")
                break
            
            print("-" * 60)
        
        self.writer.close()
        print(f"\nTraining completed! Best accuracy: {self.best_acc:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {os.path.join(self.args.log_dir, self.run_name)}")


def main():
    parser = argparse.ArgumentParser(description='Finetune plate recognition model')
    
    # 数据相关
    parser.add_argument('--data-dir', type=str, default='../../../finetune_data',
                        help='Data directory containing train_label.txt and val_label.txt')
    parser.add_argument('--onnx-model', type=str, 
                        default='../../../e2e_hztk_deploy_package/hztk_rec.onnx',
                        help='Path to ONNX model')
    
    # 训练相关
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage: 1=freeze backbone, 2=full finetune')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (use small value like 1e-5 for finetuning pretrained models)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 输出相关
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval (steps)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save interval (epochs)')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # 新增参数
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation (recommended for finetuning)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on the validation set')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.normpath(os.path.join(script_dir, args.data_dir))
    
    if not os.path.isabs(args.onnx_model):
        args.onnx_model = os.path.normpath(os.path.join(script_dir, args.onnx_model))
    
    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.normpath(os.path.join(script_dir, args.checkpoint_dir))
    
    if not os.path.isabs(args.log_dir):
        args.log_dir = os.path.normpath(os.path.join(script_dir, args.log_dir))
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 开始训练
    trainer = PlateRecTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
