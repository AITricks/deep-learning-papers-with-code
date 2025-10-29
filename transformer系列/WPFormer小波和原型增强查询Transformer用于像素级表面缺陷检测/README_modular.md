# WPFormer 模块化版本

这是从原始 `WPFormer.py` 提取的即插即用模块化组件，支持独立测试和使用。

## 文件说明

- `modular_wpformer.py` - 核心模块化组件
- `setup_torchv5.py` - 环境设置脚本
- `run_test.py` - 测试运行脚本
- `README_modular.md` - 说明文档

## 核心模块

### 1. MSCW (多尺度上下文权重模块)
- 对应结构图中的 MSCM
- 实现局部和全局注意力机制

### 2. WCA (小波增强交叉注意力)
- 对应结构图中的 WCA 模块
- 集成小波变换和交叉注意力

### 3. PCA (原型引导交叉注意力)
- 对应结构图中的 PCA 模块
- 基于原型的注意力机制

### 4. D2TDecoder (D2T解码器)
- 结合 WCA 和 PCA 的完整解码器
- 对应结构图中的 D2T Decoder

### 5. SegHead (分割头)
- 对应结构图中的 SegHead
- 生成分类和掩码预测

### 6. DDFusion (双域融合)
- 特征融合模块
- 支持不同尺度的特征融合

## 快速开始

### 方法1: 自动环境设置

```bash
# 1. 运行环境设置脚本
python setup_torchv5.py

# 2. 激活环境
conda activate torchv5

# 3. 运行测试
python run_test.py
```

### 方法2: 手动环境设置

```bash
# 1. 创建conda环境
conda create -n torchv5 python=3.8 -y

# 2. 激活环境
conda activate torchv5

# 3. 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 安装其他依赖
pip install numpy matplotlib pillow opencv-python

# 5. 运行测试
python modular_wpformer.py
```

## 使用示例

### 独立使用单个模块

```python
import torch
from modular_wpformer import WCA, PCA, D2TDecoder, SegHead

# 创建测试数据
batch_size = 2
num_queries = 16
d_model = 64
seq_len = 64

query = torch.randn(batch_size, num_queries, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

# 使用WCA模块
wca = WCA(d_model)
wca_output = wca(query, key, value)

# 使用PCA模块
pca = PCA(d_model)
pca_output = pca(query, key, value)

# 使用D2TDecoder
d2t = D2TDecoder(d_model)
d2t_output = d2t(query, key, value)
```

### 组合使用多个模块

```python
from modular_wpformer import D2TDecoder, SegHead

# 创建完整的处理管道
decoder = D2TDecoder(d_model=64)
seg_head = SegHead(channel=64)

# 处理流程
decoded_features = decoder(query, key, value)
class_pred, mask_pred, _ = seg_head(decoded_features.transpose(0, 1), mask_features)
```

## 模块特性

### 即插即用设计
- 每个模块都是独立的 `nn.Module`
- 清晰的输入输出接口
- 支持任意组合使用

### 灵活配置
- 可调节的模型维度
- 可配置的注意力头数
- 支持不同的dropout率

### 易于测试
- 内置测试函数
- 详细的形状检查
- 完整的错误处理

## 注意事项

1. **小波变换**: 当前使用简化的小波池化实现，实际使用时建议导入完整的小波模块
2. **依赖关系**: 确保安装了正确版本的PyTorch
3. **内存使用**: 大模型可能需要更多GPU内存
4. **数值稳定性**: 建议在训练时使用梯度裁剪

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用CPU模式: `torch.device('cpu')`

2. **导入错误**
   - 检查Python路径
   - 确保所有文件在同一目录

3. **版本兼容性**
   - 确保PyTorch版本 >= 1.8.0
   - 检查CUDA版本兼容性

### 获取帮助

如果遇到问题，请检查：
1. PyTorch版本是否正确
2. 所有依赖是否已安装
3. 输入数据形状是否正确
4. 是否有足够的内存

## 扩展使用

这些模块可以轻松集成到其他项目中：

```python
# 在你的项目中使用
from modular_wpformer import D2TDecoder

class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d2t_decoder = D2TDecoder(d_model=256)
        # 添加其他组件...
    
    def forward(self, x):
        # 使用D2T解码器
        output = self.d2t_decoder(query, key, value)
        return output
```
