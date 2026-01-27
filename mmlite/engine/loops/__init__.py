"""
Loops 模块 - 重导出 mmengine 的训练循环

训练循环定义了模型训练/验证/测试的迭代逻辑：
- EpochBasedTrainLoop: 基于 epoch 的训练循环
- IterBasedTrainLoop: 基于迭代次数的训练循环
- ValLoop: 验证循环
- TestLoop: 测试循环
"""

from mmengine.runner import (
    BaseLoop,
    EpochBasedTrainLoop,
    IterBasedTrainLoop,
    TestLoop,
    ValLoop,
)

__all__ = [
    "BaseLoop",
    "EpochBasedTrainLoop",
    "IterBasedTrainLoop",
    "ValLoop",
    "TestLoop",
]
