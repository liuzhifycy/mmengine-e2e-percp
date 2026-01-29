import torch
import torch.nn as nn
import math
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version
import mmcv

# 检查 mmcv 版本，确保兼容性
MMCV_VERSION = digit_version(mmcv.__version__)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class YOLOConv(BaseModule):
    """
    YOLO 系列标准的 Conv block: Conv2d + BatchNorm + SiLU
    对应 ultralytics 中的 Conv
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 dilation=1,
                 act_cfg=dict(type='SiLU', inplace=True),
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg=None):
        super().__init__(init_cfg)
        
        # Auto-pad logic
        padding = autopad(kernel_size, padding, dilation)
        
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,  # BN 后面不需要 bias
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        return self.conv(x)


class Bottleneck(BaseModule):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, init_cfg=None):
        """Initializes module."""
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Standard bottleneck forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(BaseModule):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, init_cfg=None):
        """Initialize CSP Bottleneck module."""
        super().__init__(init_cfg)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c1, c_, 1, 1)
        self.cv3 = YOLOConv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, init_cfg=None):
        """Initialize C3k module."""
        super().__init__(c1, c2, n, shortcut, g, e, init_cfg)
        c_ = int(c2 * e)
        # 覆盖 C3 的 self.m
        # 注意: 这里 k=(k, k) 对应 ultralytics 的 Bottleneck 参数
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(BaseModule):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, init_cfg=None):
        """Initialize C2f module."""
        super().__init__(init_cfg)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(BaseModule):
    """Attention module that performs self-attention on the input tensor."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5, init_cfg=None):
        super().__init__(init_cfg)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        
        # Modified: act=False, but uses BN (default YOLOConv behavior)
        self.qkv = YOLOConv(dim, h, 1, act_cfg=None) 
        self.proj = YOLOConv(dim, dim, 1, act_cfg=None)
        self.pe = YOLOConv(dim, dim, 3, 1, groups=dim, act_cfg=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(BaseModule):
    """PSABlock class implementing a Position-Sensitive Attention block."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, init_cfg=None):
        super().__init__(init_cfg)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        
        self.ffn = nn.Sequential(
            YOLOConv(c, c * 2, 1), 
            # Modified: act=False, but uses BN
            YOLOConv(c * 2, c, 1, act_cfg=None) 
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, init_cfg=None):
        """Initialize C3k2 module."""
        super().__init__(c1, c2, n, shortcut, g, e, init_cfg)
        self.c = int(c2 * e)  # hidden channels
        modules = []
        for _ in range(n):
            if attn:
                m = nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
            elif c3k:
                m = C3k(self.c, self.c, 2, shortcut, g)
            else:
                m = Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            modules.append(m)
            
        self.m = nn.ModuleList(modules)


class C2PSA(BaseModule):
    """C2PSA module with attention mechanism."""

    def __init__(self, c1, c2, n=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class SPPF(BaseModule):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""

    def __init__(self, c1, c2, k=5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = c1 // 2  # hidden channels
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class DWConv(YOLOConv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, dilation=d, groups=math.gcd(c1, c2), act_cfg=dict(type='SiLU', inplace=True) if act else None)


class DFL(BaseModule):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
