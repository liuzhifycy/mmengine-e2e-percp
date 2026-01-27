"""
自定义 Backbone 示例

演示如何创建自定义 backbone 并注册到 mmdet 的 MODELS registry。
这个示例创建了一个简单的 CNN backbone，适合学习和教学目的。

使用方法:
    1. 在配置文件中使用 type='SimpleCNNBackbone'
    2. 或者在代码中直接导入使用
"""
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class SimpleCNNBackbone(BaseModule):
    """简单的 CNN Backbone 示例

    一个轻量级的 CNN backbone，输出多尺度特征图用于 FPN。
    
    架构:
        - 4个卷积阶段 (stages)
        - 每个阶段包含: Conv -> BN -> ReLU -> MaxPool
        - 输出 4 个尺度的特征图 (C2, C3, C4, C5)

    Args:
        in_channels (int): 输入通道数，默认 3 (RGB图像)
        base_channels (int): 基础通道数，默认 64
        num_stages (int): 卷积阶段数，默认 4
        out_indices (tuple): 输出的阶段索引，默认 (0, 1, 2, 3)
        frozen_stages (int): 冻结的阶段数，默认 -1 (不冻结)
        init_cfg (dict): 初始化配置
    
    Example:
        >>> backbone = SimpleCNNBackbone(base_channels=64)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> outs = backbone(x)
        >>> for out in outs:
        ...     print(out.shape)
        torch.Size([1, 64, 56, 56])   # C2: stride 4
        torch.Size([1, 128, 28, 28])  # C3: stride 8
        torch.Size([1, 256, 14, 14])  # C4: stride 16
        torch.Size([1, 512, 7, 7])    # C5: stride 32
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        out_indices: tuple = (0, 1, 2, 3),
        frozen_stages: int = -1,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # 输入卷积 (stem)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 构建各个阶段
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            stage = self._make_stage(in_ch, out_ch, stride=stride)
            self.stages.append(stage)
            in_ch = out_ch

        # 记录每个阶段的输出通道数 (供 FPN 使用)
        self.out_channels = tuple(base_channels * (2 ** i) for i in range(num_stages))

        # 冻结指定阶段
        self._freeze_stages()

    def _make_stage(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """创建一个卷积阶段
        
        包含两个卷积块:
            - Conv 3x3 -> BN -> ReLU
            - Conv 3x3 -> BN -> ReLU
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """冻结指定数量的阶段"""
        if self.frozen_stages >= 0:
            # 冻结 stem
            for param in self.stem.parameters():
                param.requires_grad = False
            
            # 冻结 stages
            for i in range(self.frozen_stages + 1):
                if i < len(self.stages):
                    for param in self.stages[i].parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        """前向传播

        Args:
            x: 输入图像 tensor, shape (N, C, H, W)

        Returns:
            tuple: 多尺度特征图列表
        """
        x = self.stem(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode: bool = True):
        """设置训练模式，同时保持冻结层的 eval 状态"""
        super().train(mode)
        self._freeze_stages()
        # 冻结的 BN 层保持 eval 模式
        if mode and self.frozen_stages >= 0:
            for m in self.stem.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for i in range(self.frozen_stages + 1):
                if i < len(self.stages):
                    for m in self.stages[i].modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()


@MODELS.register_module()
class MobileNetLiteBackbone(BaseModule):
    """轻量级 MobileNet 风格 Backbone
    
    使用深度可分离卷积 (Depthwise Separable Convolution) 减少参数量。
    输出 4 个阶段的特征图，stride 分别为 4, 8, 16, 32。
    
    Args:
        in_channels (int): 输入通道数
        width_mult (float): 宽度乘数，用于调整通道数
        out_indices (tuple): 输出的阶段索引 (0, 1, 2, 3)
        init_cfg (dict): 初始化配置
    
    Example:
        >>> backbone = MobileNetLiteBackbone(width_mult=1.0)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> outs = backbone(x)
        >>> for out in outs:
        ...     print(out.shape)
        torch.Size([1, 64, 160, 160])   # stride 4
        torch.Size([1, 128, 80, 80])    # stride 8
        torch.Size([1, 256, 40, 40])    # stride 16
        torch.Size([1, 512, 20, 20])    # stride 32
    """

    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        out_indices: tuple = (0, 1, 2, 3),
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.width_mult = width_mult
        self.out_indices = out_indices

        def _make_divisible(v, divisor=8):
            """确保通道数是 divisor 的倍数"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 定义各阶段的配置: (输出通道, stride, 重复次数)
        # 总共 5 个阶段，输出 4 个特征图
        stage_configs = [
            (32, 2, 1),   # stem: stride 2
            (64, 2, 2),   # stage 0: stride 4, 输出 C2
            (128, 2, 2),  # stage 1: stride 8, 输出 C3
            (256, 2, 2),  # stage 2: stride 16, 输出 C4
            (512, 2, 2),  # stage 3: stride 32, 输出 C5
        ]
        
        self._stage_configs = stage_configs

        # 构建各阶段
        self.stages = nn.ModuleList()
        in_ch = in_channels
        
        for i, (out_ch, stride, repeats) in enumerate(stage_configs):
            out_ch = _make_divisible(out_ch * width_mult)
            stage_layers = []
            for j in range(repeats):
                s = stride if j == 0 else 1
                if i == 0 and j == 0:
                    # stem: 标准卷积
                    stage_layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, s, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU6(inplace=True),
                    ))
                else:
                    # 其他层: 深度可分离卷积
                    stage_layers.append(self._depthwise_separable_conv(in_ch, out_ch, s))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*stage_layers))
        
        # 记录各阶段的输出通道 (stage 1-4 对应 out_indices 0-3)
        self.out_channels = tuple(
            _make_divisible(stage_configs[i + 1][0] * width_mult) 
            for i in out_indices
        )

    def _depthwise_separable_conv(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        """深度可分离卷积块"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """前向传播"""
        # stem
        x = self.stages[0](x)
        
        # stages 1-4
        outs = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            if (i - 1) in self.out_indices:  # stage 1 对应 out_indices 0
                outs.append(x)
        
        return tuple(outs)
