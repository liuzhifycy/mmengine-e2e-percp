"""
SECOND Backbone - 用于 PointPillars 的骨干网络

基于论文: SECOND: Sparsely Embedded Convolutional Detection
用于处理散射后的伪图像（BEV表示），提取多尺度特征。
"""

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class SECOND(BaseModule):
    """SECOND backbone for point cloud detection.

    This backbone processes the pseudo-image (BEV) from PointPillarsScatter
    and extracts multi-scale features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (list[int]): Number of output channels for each block.
        layer_nums (list[int]): Number of layers in each block.
        layer_strides (list[int]): Strides of the first layer in each block.
        norm_cfg (dict): Config dict for normalization layer.
        conv_cfg (dict): Config dict for convolution layer.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        assert len(out_channels) == len(layer_nums) == len(layer_strides)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_nums = layer_nums
        
        # Build blocks
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        
        for i, (out_ch, layer_num, stride) in enumerate(
            zip(out_channels, layer_nums, layer_strides)
        ):
            block = self._make_layer(
                in_filters[i],
                out_ch,
                layer_num,
                stride=stride,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
            )
            blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)

    def _make_layer(
        self,
        in_channels,
        out_channels,
        num_blocks,
        stride=1,
        norm_cfg=None,
        conv_cfg=None,
    ):
        """Build a layer with multiple convolution blocks.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            num_blocks (int): Number of conv blocks.
            stride (int): Stride of first block.
            norm_cfg (dict): Normalization config.
            conv_cfg (dict): Convolution config.

        Returns:
            nn.Sequential: Layer module.
        """
        layers = []
        
        # First block with stride
        layers.append(
            build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        )
        layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(
                build_conv_layer(
                    conv_cfg,
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input pseudo-image (B, C, H, W).

        Returns:
            list[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class SECONDFPN(BaseModule):
    """FPN for SECOND backbone.

    Upsamples and fuses multi-scale features from SECOND backbone.

    Args:
        in_channels (list[int]): Number of input channels for each level.
        out_channels (list[int]): Number of output channels for each level.
        upsample_strides (list[int]): Strides for upsampling each level.
        norm_cfg (dict): Config dict for normalization layer.
        upsample_cfg (dict): Config dict for upsampling layer.
        conv_cfg (dict): Config dict for convolution layer.
        use_conv_for_no_stride (bool): Use conv for no stride (stride=1) case.
    """

    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        conv_cfg=dict(type='Conv2d', bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        
        assert len(in_channels) == len(out_channels) == len(upsample_strides)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        deblocks = []
        for i, (in_ch, out_ch, stride) in enumerate(
            zip(in_channels, out_channels, upsample_strides)
        ):
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=stride,
                    stride=stride,
                )
            else:
                # Use conv for stride=1 case
                stride = round(1 / stride)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_ch,
                    out_ch,
                    kernel_size=stride,
                    stride=stride,
                )
            
            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_ch)[1],
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (list[torch.Tensor]): Multi-scale features from backbone.

        Returns:
            torch.Tensor: Fused feature map.
        """
        assert len(x) == len(self.deblocks)
        
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        
        # Return as tuple for consistency with FPN interface
        return (out,)


def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): Config dict for upsample layer.

    Returns:
        nn.Module: Upsample layer.
    """
    if cfg is None:
        cfg_ = dict(type='deconv')
    else:
        cfg_ = cfg.copy()
    
    layer_type = cfg_.pop('type')
    
    if layer_type == 'deconv':
        layer = nn.ConvTranspose2d(*args, **kwargs)
    elif layer_type == 'nearest':
        # Nearest neighbor upsample + conv
        scale_factor = kwargs.get('stride', 2)
        in_channels = kwargs.get('in_channels')
        out_channels = kwargs.get('out_channels')
        
        layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, **cfg_),
        )
    elif layer_type == 'bilinear':
        scale_factor = kwargs.get('stride', 2)
        in_channels = kwargs.get('in_channels')
        out_channels = kwargs.get('out_channels')
        
        layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, **cfg_),
        )
    else:
        raise NotImplementedError(f'Upsample type {layer_type} is not supported.')
    
    return layer
