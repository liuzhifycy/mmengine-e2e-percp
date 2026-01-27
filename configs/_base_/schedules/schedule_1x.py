# 1x 训练策略配置
# 1x = 12 epochs, 在 8 和 11 epoch 时学习率衰减

# ===================== 训练/验证/测试循环配置 =====================

# 训练循环配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,  # 总训练轮数 (1x schedule)
    val_interval=1,  # 每 1 个 epoch 验证一次
)

# 验证循环配置
val_cfg = dict(type='ValLoop')

# 测试循环配置
test_cfg = dict(type='TestLoop')

# ===================== 优化器配置 =====================

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # 基础学习率 (对应 8 GPUs, batch_size=2/gpu)
        momentum=0.9,  # 动量
        weight_decay=0.0001,  # 权重衰减
    ),
)

# ===================== 学习率调度器配置 =====================

# 学习率调度器
param_scheduler = [
    # 线性预热
    dict(
        type='LinearLR',
        start_factor=0.001,  # 起始因子 (lr = lr * start_factor)
        by_epoch=False,  # 按迭代计算
        begin=0,
        end=500,  # 预热 500 次迭代
    ),
    # 多步衰减
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,  # 12 个 epoch
        by_epoch=True,  # 按 epoch 计算
        milestones=[8, 11],  # 在第 8 和 11 个 epoch 衰减
        gamma=0.1,  # 衰减因子
    ),
]

# ===================== 自动缩放学习率配置 =====================

# 自动缩放学习率
# 基于: base_batch_size = 8 GPUs * 2 samples/gpu = 16
# 实际学习率 = lr * (actual_batch_size / base_batch_size)
auto_scale_lr = dict(enable=False, base_batch_size=16)
