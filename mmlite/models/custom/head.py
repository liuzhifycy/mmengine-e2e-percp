"""
自定义 Detection Head 示例

演示如何创建自定义的检测头并注册到 mmdet 的 MODELS registry。
"""
import torch
import torch.nn as nn
from mmdet.models.dense_heads import AnchorHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.model import bias_init_with_prob, normal_init


@MODELS.register_module()
class SimpleDetectionHead(AnchorHead):
    """简单的检测头示例

    基于 AnchorHead，实现一个简化的检测头。
    使用共享卷积层进行分类和回归。

    Args:
        num_classes (int): 类别数 (不包括背景)
        in_channels (int): 输入特征通道数
        feat_channels (int): 中间特征通道数，默认 256
        stacked_convs (int): 堆叠的卷积层数，默认 4
        anchor_generator (dict): anchor 生成器配置
        bbox_coder (dict): bbox 编码器配置
        loss_cls (dict): 分类损失配置
        loss_bbox (dict): 回归损失配置

    Example:
        >>> head = SimpleDetectionHead(
        ...     num_classes=80,
        ...     in_channels=256,
        ...     feat_channels=256,
        ... )
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        anchor_generator: dict = dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder: dict = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls: dict = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox: dict = dict(
            type='L1Loss',
            loss_weight=1.0,
        ),
        train_cfg: dict = None,
        test_cfg: dict = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self):
        """初始化网络层"""
        # 共享卷积层
        self.shared_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            self.shared_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.feat_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 分类分支
        self.cls_out = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1,
        )

        # 回归分支
        self.reg_out = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            3,
            padding=1,
        )

    def init_weights(self):
        """初始化权重"""
        super().init_weights()
        for m in self.shared_convs:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    normal_init(layer, std=0.01)

        # 分类层使用特殊的偏置初始化 (focal loss)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reg_out, std=0.01)

    def forward_single(self, x: torch.Tensor) -> tuple:
        """单尺度前向传播

        Args:
            x: 单尺度特征图

        Returns:
            tuple: (cls_score, bbox_pred)
        """
        feat = x
        for conv in self.shared_convs:
            feat = conv(feat)

        cls_score = self.cls_out(feat)
        bbox_pred = self.reg_out(feat)

        return cls_score, bbox_pred


@MODELS.register_module()
class LightweightHead(AnchorHead):
    """轻量级检测头
    
    使用深度可分离卷积减少计算量，适合移动端部署。

    Args:
        num_classes (int): 类别数
        in_channels (int): 输入通道数
        feat_channels (int): 特征通道数
        num_convs (int): 卷积层数量
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 128,
        num_convs: int = 2,
        anchor_generator: dict = dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder: dict = dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls: dict = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox: dict = dict(
            type='SmoothL1Loss',
            beta=0.11,
            loss_weight=1.0,
        ),
        train_cfg: dict = None,
        test_cfg: dict = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self):
        """初始化网络层 - 使用深度可分离卷积"""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.num_convs):
            in_ch = self.in_channels if i == 0 else self.feat_channels
            # 分类分支
            self.cls_convs.append(self._depthwise_separable_conv(in_ch, self.feat_channels))
            # 回归分支
            self.reg_convs.append(self._depthwise_separable_conv(in_ch, self.feat_channels))

        # 输出层
        self.cls_out = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            1,
        )
        self.reg_out = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            1,
        )

    def _depthwise_separable_conv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """深度可分离卷积块"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def init_weights(self):
        """初始化权重"""
        super().init_weights()
        for convs in [self.cls_convs, self.reg_convs]:
            for conv in convs:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reg_out, std=0.01)

    def forward_single(self, x: torch.Tensor) -> tuple:
        """单尺度前向传播"""
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        cls_score = self.cls_out(cls_feat)
        bbox_pred = self.reg_out(reg_feat)

        return cls_score, bbox_pred
