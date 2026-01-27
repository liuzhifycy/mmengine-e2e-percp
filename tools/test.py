#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMEngine-Lite 测试/推理入口脚本

功能说明:
    - 支持单GPU和分布式测试
    - 支持可视化结果 (--show)
    - 支持保存可视化结果 (--show-dir)
    - 使用 mmengine.runner.Runner 管理测试流程

使用方法:
    # 单GPU测试
    python tools/test.py configs/xxx.py /path/to/checkpoint.pth

    # 指定工作目录保存评估结果
    python tools/test.py configs/xxx.py checkpoint.pth --work-dir ./results

    # 分布式测试 (使用 torchrun)
    torchrun --nproc_per_node=8 tools/test.py configs/xxx.py checkpoint.pth --launcher pytorch

    # 显示可视化结果（每张图像显示）
    python tools/test.py configs/xxx.py checkpoint.pth --show

    # 保存可视化结果到指定目录
    python tools/test.py configs/xxx.py checkpoint.pth --show-dir ./vis_results

    # 覆盖配置文件中的参数
    python tools/test.py configs/xxx.py checkpoint.pth --cfg-options model.backbone.depth=50
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MMEngine-Lite 测试脚本")

    # 必需参数
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("checkpoint", help="模型权重文件路径")

    # 可选参数
    parser.add_argument(
        "--work-dir",
        help="保存评估指标和日志的目录",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="显示可视化结果",
    )

    parser.add_argument(
        "--show-dir",
        help="保存可视化结果的目录",
    )

    parser.add_argument(
        "--wait-time",
        type=float,
        default=2,
        help="显示每张图像的等待时间（秒），仅在 --show 时有效",
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
        help="分布式测试启动器。none 表示非分布式测试",
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

    # 设置模型权重路径
    cfg.load_from = args.checkpoint

    # 处理可视化设置
    if args.show or args.show_dir:
        # 确保 default_hooks 中有 visualization hook
        cfg = _setup_visualization(cfg, args)

    # 构建 Runner
    # 如果配置文件中指定了 runner_type，则从 registry 构建自定义 Runner
    # 否则使用默认的 Runner
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # 开始测试
    runner.test()


def _setup_visualization(cfg, args):
    """
    设置可视化相关配置

    Args:
        cfg: 配置对象
        args: 命令行参数

    Returns:
        更新后的配置对象
    """
    # 确保 default_hooks 存在
    if cfg.get("default_hooks", None) is None:
        cfg.default_hooks = dict()

    # 设置 visualization hook
    cfg.default_hooks.visualization = dict(
        type="VisualizationHook",
        draw=True,
    )

    # 设置 visualizer
    if cfg.get("visualizer", None) is None:
        cfg.visualizer = dict(
            type="Visualizer",
            vis_backends=[dict(type="LocalVisBackend")],
        )

    # 设置 show 和 show_dir
    if args.show:
        cfg.visualizer.show = True
        cfg.visualizer.wait_time = args.wait_time

    if args.show_dir:
        # 设置保存路径
        cfg.visualizer.save_dir = args.show_dir
        # 确保目录存在
        os.makedirs(args.show_dir, exist_ok=True)

    return cfg


if __name__ == "__main__":
    main()
