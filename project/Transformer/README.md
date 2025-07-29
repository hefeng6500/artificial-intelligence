# Transformer 架构演示项目

这个项目提供了一个完整的 Transformer 模型实现，包含详细的中文注释和可视化工具，帮助理解 Transformer 架构的核心概念和工作原理。

## 📁 项目结构

```
Transformer/
├── transformer_model.py         # 核心 Transformer 模型实现
├── transformer_demo.ipynb       # 交互式 Jupyter Notebook 演示
├── train_demo.py                # 训练和推理演示
├── attention_visualization.py    # 注意力权重可视化工具
├── test_transformer.py          # 单元测试
├── font_setup_guide.md          # 中文字体设置指南
├── requirements.txt             # 项目依赖
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 交互式学习（推荐）

启动 Jupyter Notebook 进行交互式学习：

```bash
jupyter notebook transformer_demo.ipynb
```

这个 notebook 包含：
- 详细的架构解释和可视化
- 逐步的代码实现
- 交互式的注意力可视化
- 完整的训练和推理演示

### 3. 运行基本演示

```bash
# 运行完整的训练和推理演示
python train_demo.py

# 运行注意力可视化演示
python attention_visualization.py
```

### 4. 运行测试

```bash
python test_transformer.py
```

### 5. 解决中文字体显示问题

如果在可视化中看到中文显示为方框，请参考：

```bash
# 查看字体设置指南
cat font_setup_guide.md
```

或者在代码中添加字体设置：

```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
```

### 3. 使用模型

```python
from transformer_model import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# 前向传播
output, enc_attn, dec_attn = model(src, tgt)
```

## 🏗️ Transformer 架构详解

### 核心组件

#### 1. 多头注意力机制 (Multi-Head Attention)

多头注意力是 Transformer 的核心创新，它允许模型同时关注序列中不同位置的信息：

- **Query (Q)**: 查询向量，表示当前位置想要查找的信息
- **Key (K)**: 键向量，表示每个位置可以提供的信息
- **Value (V)**: 值向量，表示每个位置的实际内容

**计算公式**:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**多头机制的优势**:
- 每个头可以学习不同类型的依赖关系
- 增强模型的表达能力
- 提供更丰富的特征表示

#### 2. 位置编码 (Positional Encoding)

由于 Transformer 没有循环结构，需要显式地添加位置信息：

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

**特点**:
- 使用正弦和余弦函数编码位置
- 允许模型处理任意长度的序列
- 相对位置关系可以通过三角函数性质学习

#### 3. 前馈网络 (Feed-Forward Network)

每个 Transformer 块包含一个位置前馈网络：

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

**作用**:
- 对每个位置独立进行非线性变换
- 增加模型的非线性表达能力
- 通常将维度先扩展再压缩

#### 4. 残差连接和层归一化

每个子层都使用残差连接和层归一化：

```
LayerNorm(x + Sublayer(x))
```

**优势**:
- 缓解梯度消失问题
- 加速训练收敛
- 提高模型稳定性

### 编码器-解码器架构

#### 编码器 (Encoder)
- 由 N 个相同的层堆叠而成
- 每层包含多头自注意力和前馈网络
- 处理输入序列，生成上下文表示

#### 解码器 (Decoder)
- 同样由 N 个相同的层堆叠而成
- 每层包含三个子层：
  1. 掩码多头自注意力
  2. 编码器-解码器注意力
  3. 前馈网络
- 生成输出序列

## 🎯 演示任务说明

### 数字序列变换任务

为了便于理解，我们设计了一个简单的任务：

**输入**: `[1, 2, 3, 4, 5]`
**输出**: `[2, 3, 4, 5, 6]` (每个数字加1)

这个任务虽然简单，但能够很好地展示 Transformer 的学习能力。

### 训练过程

1. **数据生成**: 自动生成训练数据
2. **模型训练**: 使用 Adam 优化器训练
3. **损失监控**: 实时显示训练损失
4. **模型保存**: 保存训练好的模型权重

### 推理演示

1. **序列生成**: 逐步生成输出序列
2. **准确率计算**: 评估模型性能
3. **注意力可视化**: 展示模型的注意力模式

## 📊 注意力可视化功能

### 可视化类型

1. **注意力热力图**: 显示单个注意力头的权重分布
2. **多头注意力**: 同时显示所有注意力头
3. **跨层分析**: 观察不同层的注意力变化
4. **统计分析**: 注意力权重的统计特性
5. **流向图**: 直观显示注意力流向

### 使用示例

```python
from attention_visualization import AttentionVisualizer

# 创建可视化工具
visualizer = AttentionVisualizer(model)

# 提取注意力权重
enc_attn, dec_attn = visualizer.extract_attention_weights(src, tgt)

# 绘制热力图
visualizer.plot_attention_heatmap(enc_attn, tokens, layer=0, head=0)

# 分析注意力模式
analysis = visualizer.analyze_attention_patterns(enc_attn, tokens)
```

## 🔧 模型参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `d_model` | 模型维度 | 512 |
| `num_heads` | 注意力头数 | 8 |
| `num_layers` | 层数 | 6 |
| `d_ff` | 前馈网络维度 | 2048 |
| `vocab_size` | 词汇表大小 | 取决于任务 |
| `max_len` | 最大序列长度 | 5000 |
| `dropout` | Dropout 比率 | 0.1 |

## 📈 性能优化建议

### 训练优化

1. **学习率调度**: 使用 warmup + 衰减策略
2. **梯度裁剪**: 防止梯度爆炸
3. **批量大小**: 根据显存调整
4. **混合精度**: 使用 FP16 加速训练

### 推理优化

1. **模型量化**: 减少模型大小
2. **序列并行**: 并行处理多个序列
3. **缓存机制**: 缓存中间结果
4. **早停策略**: 提前结束生成

## 🎓 学习建议

### 理解顺序

1. **基础概念**: 先理解注意力机制的基本原理
2. **组件分析**: 逐个学习各个组件的作用
3. **整体架构**: 理解编码器-解码器的协作
4. **训练过程**: 观察模型的学习过程
5. **可视化分析**: 通过可视化加深理解

### 实践建议

1. **修改参数**: 尝试不同的模型参数组合
2. **改变任务**: 设计其他简单的序列任务
3. **分析注意力**: 观察不同层和头的注意力模式
4. **对比实验**: 比较有无某些组件的效果

## 🔍 常见问题

### Q: 为什么需要位置编码？
A: Transformer 没有循环或卷积结构，无法感知序列中的位置信息，因此需要显式添加位置编码。

### Q: 多头注意力的优势是什么？
A: 不同的头可以关注不同类型的依赖关系，如语法关系、语义关系等，增强模型的表达能力。

### Q: 为什么要使用残差连接？
A: 残差连接可以缓解深层网络的梯度消失问题，使得训练更加稳定。

### Q: 如何选择模型参数？
A: 参数选择需要根据具体任务和计算资源来决定。一般来说，更大的模型有更强的表达能力，但也需要更多的计算资源。

## 📚 参考资料

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
3. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 带注释的 Transformer 实现

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。