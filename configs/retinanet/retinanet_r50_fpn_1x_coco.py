# RetinaNet R50 FPN 1x COCO 配置
# 完整配置文件，继承自基础配置

# ===================== 基础配置继承 =====================

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',  # 模型配置
    '../_base_/datasets/coco_detection.py',   # 数据集配置
    '../_base_/schedules/schedule_1x.py',     # 训练策略配置
    '../_base_/default_runtime.py',           # 运行时配置
]

# ===================== 模型配置覆盖 (可选) =====================

# 如需修改模型配置，可在此覆盖
# model = dict(
#     bbox_head=dict(
#         num_classes=80,  # 修改类别数
#     ),
# )

# ===================== 数据配置覆盖 (可选) =====================

# 如需修改数据配置，可在此覆盖
# train_dataloader = dict(
#     batch_size=4,  # 修改 batch size
# )

# ===================== 训练策略覆盖 (可选) =====================

# 如需修改学习率，可在此覆盖
# optim_wrapper = dict(
#     optimizer=dict(lr=0.02),  # 修改学习率
# )

# ===================== 实验配置 =====================

# 工作目录 (保存日志和检查点)
work_dir = './work_dirs/retinanet_r50_fpn_1x_coco'
