import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class LinearProjection(nn.Module):
    """线性投影层，用于生成Q、K、V"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class ASSA(nn.Module):
    """
    自适应稀疏自注意力模块 (Adaptive Sparse Self-Attention)
    
    该模块实现了两个分支：
    1. 动态稀疏注意力 (DSA): 使用Softmax归一化
    2. 静态稀疏注意力 (SSA): 使用ReLU + Square操作
    
    两个分支通过可学习权重进行自适应融合
    """
    def __init__(self, dim, win_size, num_heads, token_projection='linear', 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))

        # 计算相对位置索引
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # 线性投影层
        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ASSA特有的组件
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        # 可学习的融合权重
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [B_, N, C]
            attn_kv: 可选的键值对输入
            mask: 可选的注意力掩码
            
        Returns:
            输出特征 [B_, N, C]
        """
        B_, N, C = x.shape
        
        # 生成Q、K、V
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], 
            self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        # 处理掩码
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)  # DSA分支
            attn1 = self.relu(attn) ** 2  # SSA分支
        else:
            attn0 = self.softmax(attn)  # DSA分支
            attn1 = self.relu(attn) ** 2  # SSA分支

        # 自适应融合两个分支
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2

        attn = self.attn_drop(attn)

        # 应用注意力到值向量
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


def window_partition(x, win_size, dilation_rate=1):
    """将输入分割成窗口"""
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, 
                    padding=4*(dilation_rate-1), stride=win_size)
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)
        windows = windows.permute(0, 2, 3, 1).contiguous()
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    """将窗口重新组合成完整特征图"""
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, 
                  padding=4*(dilation_rate-1), stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def test_assa_module():
    """测试ASSA模块的功能"""
    print("开始测试ASSA模块...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试参数
    batch_size = 2
    height, width = 32, 32
    channels = 64
    win_size = (8, 8)
    num_heads = 8
    
    # 创建ASSA模块
    assa = ASSA(
        dim=channels,
        win_size=win_size,
        num_heads=num_heads,
        attn_drop=0.1,
        proj_drop=0.1
    ).to(device)
    
    print(f"ASSA模块参数:")
    print(f"  - 输入维度: {channels}")
    print(f"  - 窗口大小: {win_size}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 总参数量: {sum(p.numel() for p in assa.parameters()):,}")
    
    # 创建测试输入
    x = torch.randn(batch_size, height * width, channels).to(device)
    print(f"输入张量形状: {x.shape}")
    
    # 测试窗口分割
    x_windows = x.view(batch_size, height, width, channels)
    x_windows = window_partition(x_windows, win_size[0])
    x_windows = x_windows.view(-1, win_size[0] * win_size[1], channels)
    print(f"窗口分割后形状: {x_windows.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = assa(x_windows)
        print(f"输出张量形状: {output.shape}")
        
        # 检查输出是否合理
        assert output.shape == x_windows.shape, f"输出形状不匹配: {output.shape} vs {x_windows.shape}"
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        print("[OK] 前向传播测试通过")
    
    # 测试梯度
    x_windows.requires_grad_(True)
    output = assa(x_windows)
    loss = output.sum()
    loss.backward()
    
    assert x_windows.grad is not None, "梯度计算失败"
    print("[OK] 反向传播测试通过")
    
    # 测试融合权重
    print(f"融合权重: w1={torch.exp(assa.w[0]) / torch.sum(torch.exp(assa.w)):.4f}, "
          f"w2={torch.exp(assa.w[1]) / torch.sum(torch.exp(assa.w)):.4f}")
    
    # 测试不同输入尺寸
    print("\n测试不同输入尺寸...")
    test_sizes = [(16, 16), (24, 24), (32, 32)]
    for h, w in test_sizes:
        x_test = torch.randn(1, h * w, channels).to(device)
        x_test_windows = x_test.view(1, h, w, channels)
        x_test_windows = window_partition(x_test_windows, win_size[0])
        x_test_windows = x_test_windows.view(-1, win_size[0] * win_size[1], channels)
        
        with torch.no_grad():
            output_test = assa(x_test_windows)
            print(f"  输入尺寸 {h}x{w}: 输出形状 {output_test.shape}")
    
    print("\n[SUCCESS] ASSA模块测试全部通过!")


if __name__ == "__main__":
    test_assa_module()
