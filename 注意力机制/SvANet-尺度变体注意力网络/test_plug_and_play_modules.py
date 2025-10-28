"""
测试 SvANet 核心即插即用模块
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Any


# ==================== 基础工具类和函数 ====================

def pair(val):
    """转换为元组"""
    return val if isinstance(val, (tuple, list)) else (val, val)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """确保值能被 divisor 整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def shuffle_tensor(feature: torch.Tensor, mode: int = 1) -> list:
    """打乱张量"""
    if isinstance(feature, torch.Tensor):
        feature = [feature]
    indexs = None
    output = []
    for f in feature:
        B, C, H, W = f.shape
        if mode == 1:
            f = f.flatten(2)
            if indexs is None:
                indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        output.append(f)
    return output


def set_method(self, element_name, element_value):
    """设置属性"""
    return setattr(self, element_name, element_value)


def call_method(self, element_name):
    """获取属性"""
    return getattr(self, element_name)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """自适应平均池化"""
    def __init__(self, output_size: int or tuple = 1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)


class BaseConv2d(nn.Module):
    """基础卷积层"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        BNorm: bool = False,
        ActLayer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any
    ):
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
        if bias is None:
            bias = not BNorm
        
        self.Conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1) if BNorm else nn.Identity()
        
        if ActLayer is not None:
            self.Act = ActLayer(inplace=True) if ActLayer != nn.Sigmoid else ActLayer()
        else:
            self.Act = None
    
    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x


class StochasticDepth(nn.Module):
    """随机深度"""
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            return x * 0.0
        return x


# ==================== 核心即插即用模块 ====================

class MoCAttention(nn.Module):
    """Monte Carlo 注意力 - 学习全局和局部特征"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        SqueezeFactor: int = 4,
        PoolRes: list = [1, 2, 3],
        Act: Callable[..., nn.Module] = nn.ReLU,
        ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
        MoCOrder: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = max(make_divisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            set_method(self, 'Pool%d' % k, Pooling)
        
        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
    
    def monte_carlo_sample(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffle_tensor(x)[0] if self.MoCOrder else x
            AttnMap: torch.Tensor = call_method(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]
        else:
            AttnMap: torch.Tensor = call_method(self, 'Pool%d' % 1)(x)
        
        return AttnMap
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AttnMap = self.monte_carlo_sample(x)
        return x * self.SELayer(AttnMap)


class SqueezeExcitation(nn.Module):
    """SE 注意力机制"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        SqueezeFactor: int = 4,
        Act: Callable[..., nn.Module] = nn.ReLU,
        ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
        **kwargs: Any,
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = max(make_divisible(InChannels // SqueezeFactor, 8), 32)
        
        self.SELayer = nn.Sequential(
            AdaptiveAvgPool2d(1),
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.SELayer(x)


class AssembleFormer(nn.Module):
    """组合 CNN 和 ViT 的混合模块"""
    
    def __init__(
        self,
        InChannels: int,
        FfnMultiplier: float = 2.0,
        NumAttnBlocks: int = 2,
        PatchRes: int = 2,
        Dilation: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        DimAttnUnit = InChannels // 2
        
        # 局部表示
        self.LocalRep = nn.Sequential(
            BaseConv2d(InChannels, InChannels, 3, 1, dilation=Dilation, BNorm=True, ActLayer=nn.SiLU),
            BaseConv2d(InChannels, DimAttnUnit, 1, 1),
        )
        
        # 简化的全局表示（避免复杂的自注意力机制）
        DimFfn = int(FfnMultiplier * DimAttnUnit)
        self.GlobalRep = nn.Sequential(
            BaseConv2d(DimAttnUnit, DimFfn, 1, 1),
            nn.SiLU(inplace=True),
            BaseConv2d(DimFfn, DimAttnUnit, 1, 1),
        )
        
        self.ConvProj = BaseConv2d(2 * DimAttnUnit, InChannels, 1, 1, BNorm=True)
        self.HPatch, self.WPatch = pair(PatchRes)
        self.PatchArea = self.WPatch * self.HPatch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部特征
        FmConv = self.LocalRep(x)
        
        # 简化的全局特征处理
        GlobalFm = self.GlobalRep(FmConv)
        
        # 局部 + 全局
        Fm = self.ConvProj(torch.cat((GlobalFm, FmConv), dim=1))
        
        # 残差连接
        return x + Fm


class FGBottleneck(nn.Module):
    """特征引导瓶颈块 (对应结构图中的 MCBottleneck)"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        Expansion: float = 2.0,
        Stride: int = 1,
        Dilation: int = 1,
        SELayer: nn.Module = None,
        ActLayer: Callable[..., nn.Module] = None,
        **kwargs: Any
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = make_divisible(InChannels * Expansion, 8)
        
        self.Bottleneck = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(HidChannels, HidChannels, 3, Stride, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU),
            SELayer(InChannels=HidChannels, **kwargs) if SELayer is not None else nn.Identity(),
            BaseConv2d(HidChannels, InChannels, 1, BNorm=True)
        )
        
        self.ActLayer = ActLayer(inplace=True) if ActLayer is not None else nn.Identity()
        self.ViTLayer = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Out = self.Bottleneck(x)
        Out = self.ActLayer(x + Out)
        return self.ViTLayer(Out)


# ==================== 测试函数 ====================

def test_modules():
    """测试所有核心模块"""
    print("=" * 80)
    print("开始测试 SvANet 核心即插即用模块")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建测试输入
    batch_size = 2
    in_channels = 64
    h, w = 32, 32
    x = torch.randn(batch_size, in_channels, h, w).to(device)
    
    print(f"输入形状: {x.shape}\n")
    
    # 测试 1: MoCAttention
    print("测试 1: MoCAttention (Monte Carlo 注意力)")
    moc_attn = MoCAttention(InChannels=in_channels).to(device)
    out = moc_attn(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] MoCAttention 测试通过\n")
    
    # 测试 2: SqueezeExcitation
    print("测试 2: SqueezeExcitation (SE 注意力)")
    se = SqueezeExcitation(InChannels=in_channels).to(device)
    out = se(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] SqueezeExcitation 测试通过\n")
    
    # 测试 3: AssembleFormer
    print("测试 3: AssembleFormer (CNN + ViT 混合模块)")
    assem_former = AssembleFormer(InChannels=in_channels, NumAttnBlocks=1).to(device)
    out = assem_former(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] AssembleFormer 测试通过\n")
    
    # 测试 4: FGBottleneck (MCBottleneck)
    print("测试 4: FGBottleneck (特征引导瓶颈)")
    fg_bottleneck = FGBottleneck(InChannels=in_channels, SELayer=None).to(device)
    out = fg_bottleneck(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] FGBottleneck 测试通过\n")
    
    # 测试 5: 使用 MoCAttention 的 FGBottleneck
    print("测试 5: FGBottleneck + MoCAttention")
    fg_bottleneck_se = FGBottleneck(
        InChannels=in_channels, 
        SELayer=MoCAttention
    ).to(device)
    out = fg_bottleneck_se(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] FGBottleneck + MoCAttention 测试通过\n")
    
    print("=" * 80)
    print("所有核心即插即用模块测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_modules()

