#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMEngine-Lite 训练入口脚本

功能说明:
    - 支持单GPU和分布式训练
    - 支持断点续训 (--resume)
    - 支持混合精度训练 (--amp)
    - 使用 mmengine.runner.Runner 管理训练流程

使用方法:
    # 单GPU训练
    python tools/train.py configs/xxx.py

    # 单GPU训练 + 指定工作目录
    python tools/train.py configs/xxx.py --work-dir ./work_dirs/exp1

    # 分布式训练 (使用 torchrun)
    torchrun --nproc_per_node=8 tools/train.py configs/xxx.py --launcher pytorch

    # 断点续训 (自动从最新检查点恢复)
    python tools/train.py configs/xxx.py --resume

    # 断点续训 (指定检查点路径)
    python tools/train.py configs/xxx.py --resume /path/to/checkpoint.pth

    # 启用混合精度训练
    python tools/train.py configs/xxx.py --amp

    # 覆盖配置文件中的参数
    python tools/train.py configs/xxx.py --cfg-options model.backbone.depth=50
"""

import argparse
import logging
import os
import os.path as osp
import sys

# 确保 mmlite 包可以被导入
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# 导入 mmlite 模块以触发注册
import mmlite.datasets  # noqa: F401
import mmlite.evaluation  # noqa: F401
import mmlite.models  # noqa: F401


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MMEngine-Lite 训练脚本")

    # 必需参数
    parser.add_argument("config", help="配置文件路径")

    # 可选参数
    parser.add_argument("--work-dir", help="保存日志和模型的工作目录")

    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="启用自动混合精度 (AMP) 训练",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="断点续训。不指定路径时自动从 work_dir 中的最新检查点恢复；"
        "也可以指定具体的检查点路径",
    )

    parser.add_argument(
        "--auto-scale-lr",
        action="store_true",
        help="根据 GPU 数量自动缩放学习率",
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="覆盖配置文件中的设置，格式为 key=value。"
        '支持嵌套格式，如 key="[a,b]" 或 key=a,b',
    )

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="分布式训练启动器。none 表示非分布式训练",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于结果可复现",
    )

    # PyTorch >= 2.0 使用 --local-rank，旧版本使用 --local_rank
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    args = parser.parse_args()

    # 设置 LOCAL_RANK 环境变量（兼容不同 PyTorch 版本）
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    """主函数"""
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    # 设置分布式启动器
    cfg.launcher = args.launcher

    # 合并命令行参数到配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 确定工作目录优先级: 命令行参数 > 配置文件 > 默认（基于配置文件名）
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # 使用配置文件名作为默认工作目录
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # 处理混合精度训练
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.get("type", "OptimWrapper")
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "配置文件中已启用 AMP 训练。",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                f"--amp 仅支持 OptimWrapper 类型，但配置中是 {optim_wrapper}"
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # 处理学习率自动缩放
    if args.auto_scale_lr:
        if (
            "auto_scale_lr" in cfg
            and "enable" in cfg.auto_scale_lr
            and "base_batch_size" in cfg.auto_scale_lr
        ):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                "配置文件中未找到 auto_scale_lr 相关配置。"
                "请确保配置文件包含 auto_scale_lr.enable 和 auto_scale_lr.base_batch_size"
            )

    # 处理断点续训
    # 优先级: --resume 参数 > 配置文件中的 resume 设置
    if args.resume == "auto":
        # 自动从 work_dir 中的最新检查点恢复
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        # 从指定的检查点恢复
        cfg.resume = True
        cfg.load_from = args.resume

    # 设置随机种子
    if args.seed is not None:
        cfg.seed = args.seed

    # 构建 Runner
    # 如果配置文件中指定了 runner_type，则从 registry 构建自定义 Runner
    # 否则使用默认的 Runner
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # 开始训练
    runner.train()


if __name__ == "__main__":
    main()
