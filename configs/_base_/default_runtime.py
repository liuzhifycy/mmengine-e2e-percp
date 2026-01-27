# 默认运行时配置

# ===================== 默认钩子配置 =====================

default_hooks = dict(
    # 计时器钩子 - 记录迭代耗时
    timer=dict(type='IterTimerHook'),

    # 日志记录钩子
    logger=dict(
        type='LoggerHook',
        interval=50,  # 每 50 次迭代打印一次日志
    ),

    # 参数调度器钩子 - 更新学习率等
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 检查点保存钩子
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每 1 个 epoch 保存一次
        max_keep_ckpts=3,  # 最多保留 3 个检查点
        save_best='auto',  # 自动保存最佳模型
    ),

    # Sampler 种子钩子 - 确保分布式训练可复现
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 可视化钩子
    visualization=dict(type='DetVisualizationHook'),
)

# ===================== 环境配置 =====================

env_cfg = dict(
    # 是否使用 cudnn benchmark
    cudnn_benchmark=False,

    # 多进程配置
    mp_cfg=dict(
        mp_start_method='fork',  # multiprocessing 启动方式
        opencv_num_threads=0,  # OpenCV 线程数 (0=自动)
    ),

    # 分布式配置
    dist_cfg=dict(backend='nccl'),  # 分布式后端
)

# ===================== 可视化配置 =====================

# 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),  # 本地可视化
    # 可选: TensorBoard
    # dict(type='TensorboardVisBackend'),
    # 可选: WandB
    # dict(type='WandbVisBackend'),
]

# 可视化器
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

# ===================== 日志配置 =====================

# 日志处理器
log_processor = dict(
    type='LogProcessor',
    window_size=50,  # 滑动窗口大小 (用于平滑损失)
    by_epoch=True,  # 按 epoch 记录
)

# 日志级别
log_level = 'INFO'

# ===================== 加载与恢复配置 =====================

# 从检查点加载模型 (仅加载权重)
load_from = None

# 从检查点恢复训练 (加载权重、优化器状态、epoch 等)
resume = False

# ===================== 默认作用域 =====================

# 默认作用域 (用于注册表查找)
default_scope = 'mmdet'

# ===================== 随机性配置 =====================

# 随机性配置 (用于可复现性)
randomness = dict(
    seed=None,  # 随机种子 (None=随机)
    deterministic=False,  # 是否使用确定性算法
)
