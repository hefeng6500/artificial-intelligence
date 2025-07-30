# Transformer 中英文翻译模型

一个基于 Transformer 架构的中英文翻译模型项目，支持从原始CSV数据集到完整训练和推理的端到端流程。

## 项目简介

本项目实现了一个完整的神经机器翻译系统，使用 Transformer 架构进行中文到英文的翻译。项目包含数据预处理、模型训练、推理测试和演示功能，特别适用于学习和研究神经机器翻译技术。

## 主要功能

- 🚀 **完整的 Transformer 实现**: 包含编码器、解码器、多头注意力机制
- 📊 **端到端数据处理**: 从原始CSV到训练就绪的数据集
- 🔤 **双语分词器**: 支持中英文文本的智能分词和编码
- 🎯 **多种解码策略**: 贪心搜索、束搜索、采样生成
- 📈 **训练监控**: 实时损失跟踪和验证指标
- 💾 **模型管理**: 完整的模型保存和加载机制
- 🧪 **推理测试**: 独立的推理测试脚本
- 🎨 **演示功能**: 交互式翻译演示

## 项目结构

```
transformer-input-method/
├── config/                    # 配置文件
│   ├── model_config.py        # 模型架构配置
│   └── training_config.py     # 训练参数配置
├── data/                      # 数据目录
│   ├── demo/                  # 演示数据
│   ├── models/                # 训练好的模型
│   │   └── tokenizer/         # 分词器文件
│   ├── processed/             # 处理后的训练数据
│   │   ├── train.json         # 训练集
│   │   ├── val.json           # 验证集
│   │   └── test.json          # 测试集
│   ├── process_data/          # 数据处理脚本
│   │   └── prepare_data.py    # 数据预处理脚本
│   └── raw/                   # 原始数据集
│       └── damo_mt_testsets_zh2en_spoken_iwslt1617.csv
├── models/                    # 模型定义
│   ├── attention.py           # 多头注意力机制
│   ├── decoder.py             # Transformer解码器
│   ├── encoder.py             # Transformer编码器
│   ├── embedding.py           # 词嵌入层
│   ├── positional.py          # 位置编码
│   └── transformer.py         # 完整Transformer模型
├── training/                  # 训练模块
│   ├── loss.py                # 损失函数
│   ├── optimizer.py           # 优化器配置
│   └── trainer.py             # 训练器实现
├── utils/                     # 工具模块
│   ├── data_loader.py         # 数据加载器
│   ├── metrics.py             # 评估指标
│   ├── tokenizer.py           # 双语分词器
│   └── visualization.py       # 可视化工具
├── demo.py                    # 交互式演示脚本
├── train_with_csv.py          # CSV数据训练脚本
├── inference_test.py          # 推理测试脚本
└── README.md                  # 项目文档
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于 GPU 加速)
- Windows/Linux/macOS

### 依赖安装

#### 方法一：使用 pip 安装

```bash
# 核心依赖
pip install torch torchvision torchaudio
pip install transformers tokenizers
pip install numpy pandas matplotlib seaborn
pip install scikit-learn nltk rouge-score
pip install tqdm tensorboard

# 可选依赖（用于可视化）
pip install plotly kaleido
```

#### 方法二：使用 requirements.txt（如果存在）

```bash
pip install -r requirements.txt
```

### 快速演示

运行演示脚本来体验完整功能：

```bash
python demo.py
```

演示包括：
- 分词器功能展示
- 模型架构介绍
- 训练过程演示
- 翻译功能测试
- 注意力可视化
- 评估指标计算

## 快速开始

### 一键运行流程

```bash
# 1. 数据预处理
cd data/process_data
python prepare_data.py

# 2. 模型训练
cd ../..
python train_with_csv.py

# 3. 推理测试
python inference_test.py

# 4. 交互式演示
python demo.py
```

### 预期输出

- **数据预处理**: 生成训练、验证、测试集JSON文件
- **模型训练**: 输出训练进度，保存模型和分词器
- **推理测试**: 显示示例翻译结果
- **演示**: 提供交互式翻译界面

## 完整使用指南

### 1. 数据准备

#### 步骤1：准备原始数据

确保原始CSV数据集位于正确位置：
```
data/raw/damo_mt_testsets_zh2en_spoken_iwslt1617.csv
```

#### 步骤2：数据预处理

运行数据预处理脚本，将CSV数据转换为训练格式：

```bash
cd data/process_data
python prepare_data.py
```

这将生成以下文件：
- `data/processed/train.json` - 训练集（80%）
- `data/processed/val.json` - 验证集（10%）
- `data/processed/test.json` - 测试集（10%）

### 2. 模型训练

#### 使用CSV数据训练（推荐）

```bash
python train_with_csv.py
```

训练过程将：
- 自动创建分词器
- 训练Transformer模型
- 保存模型到 `data/models/`
- 保存分词器到 `data/models/tokenizer/`
- 生成训练日志

#### 训练输出

训练完成后，将在 `data/models/` 目录下生成：
- `final_model.pt` - 训练好的模型
- `test_model.pth` - 测试模型检查点
- `tokenizer/` - 分词器文件
- `config.json` - 模型配置

### 3. 模型推理测试

训练完成后，运行推理测试：

```bash
python inference_test.py
```

这将测试模型的翻译能力，输出示例翻译结果。

### 4. 交互式演示

运行演示脚本体验翻译功能：

```bash
python demo.py
```

演示功能包括：
- 交互式中译英翻译
- 模型性能测试
- 注意力可视化（如果可用）

### 5. 配置说明

#### 训练配置 (`config/training_config.py`)

```python
class TrainingConfig:
    # 模型架构参数
    d_model = 512          # 模型维度
    n_heads = 8            # 注意力头数
    n_layers = 6           # 层数
    d_ff = 2048           # 前馈网络维度
    
    # 训练参数
    batch_size = 32        # 批次大小
    learning_rate = 1e-4   # 学习率
    num_epochs = 50        # 训练轮数
    
    # 数据参数
    max_seq_length = 128   # 最大序列长度
    vocab_size = 10000     # 词汇表大小
    
    # 路径配置
    data_dir = "data/processed"
    model_save_dir = "data/models"
```

## 详细使用

### 数据格式支持

除了CSV数据外，还支持 JSON 和 TXT 格式：

**JSON 格式**：
```json
[
  {"source": "你好，世界！", "target": "Hello, world!"},
  {"source": "我爱学习。", "target": "I love learning."}
]
```

**TXT 格式**：
```
你好，世界！\tHello, world!
我爱学习。\tI love learning.
```

### 高级训练选项

#### 使用默认配置训练

```bash
python train.py --config config/training_config.json
```

#### 自定义训练参数

```bash
python train.py \
    --config config/training_config.json \
    --device cuda \
    --log-level INFO \
    --seed 42
```

#### 恢复训练

```bash
python train.py \
    --config config/training_config.json \
    --resume checkpoints/model_epoch_10.pt
```

### 3. 模型推理

#### 交互式翻译

```bash
python inference.py \
    --model checkpoints/final_model.pt \
    --tokenizer checkpoints/tokenizer \
    --mode interactive
```

#### 翻译单个文本

```bash
python inference.py \
    --model checkpoints/final_model.pt \
    --tokenizer checkpoints/tokenizer \
    --mode text \
    --input "你好，世界！"
```

#### 批量翻译文件

```bash
python inference.py \
    --model checkpoints/final_model.pt \
    --tokenizer checkpoints/tokenizer \
    --mode file \
    --input input.txt \
    --output output.txt
```

#### 评估模型性能

```bash
python inference.py \
    --model checkpoints/final_model.pt \
    --tokenizer checkpoints/tokenizer \
    --mode evaluate \
    --test-file test_data.json
```

### 4. 配置参数

主要配置参数说明：

#### 模型参数
- `d_model`: 模型维度 (默认: 512)
- `n_heads`: 注意力头数 (默认: 8)
- `n_layers`: 层数 (默认: 6)
- `d_ff`: 前馈网络维度 (默认: 2048)
- `max_seq_length`: 最大序列长度 (默认: 512)
- `dropout`: Dropout 率 (默认: 0.1)

#### 训练参数
- `batch_size`: 批大小 (默认: 32)
- `learning_rate`: 学习率 (默认: 1e-4)
- `num_epochs`: 训练轮数 (默认: 100)
- `warmup_steps`: 预热步数 (默认: 4000)
- `max_grad_norm`: 梯度裁剪 (默认: 1.0)

#### 数据参数
- `vocab_size`: 词汇表大小 (默认: 32000)
- `min_freq`: 最小词频 (默认: 2)
- `src_lang`: 源语言 (默认: "zh")
- `tgt_lang`: 目标语言 (默认: "en")

### 5. 解码策略

#### 贪心搜索
```bash
python inference.py \
    --strategy greedy \
    --max-length 100
```

#### 束搜索
```bash
python inference.py \
    --strategy beam_search \
    --beam-size 5 \
    --max-length 100 \
    --num-return 3
```

#### 采样生成
```bash
python inference.py \
    --strategy sampling \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --num-return 3
```

## 输出文件说明

### 训练输出

训练完成后，项目将生成以下文件：

```
data/models/
├── final_model.pt          # 最终训练模型
├── test_model.pth          # 测试检查点
├── config.json             # 模型配置文件
├── tokenizer/              # 分词器目录
│   ├── zh_tokenizer.json   # 中文分词器
│   └── en_tokenizer.json   # 英文分词器
└── training.log            # 训练日志
```

### 数据处理输出

```
data/processed/
├── train.json              # 训练集（80%数据）
├── val.json                # 验证集（10%数据）
└── test.json               # 测试集（10%数据）
```

## 常见问题和注意事项

### 常见问题

**Q: 训练时出现CUDA内存不足错误？**
A: 减小批次大小（batch_size）或使用CPU训练：
```python
# 在 config/training_config.py 中
batch_size = 16  # 或更小
```

**Q: 数据预处理失败？**
A: 确保CSV文件格式正确，包含中英文对照数据，使用UTF-8编码。

**Q: 模型推理结果不理想？**
A: 可能需要更多训练数据或调整模型参数，增加训练轮数。

**Q: 找不到模型文件？**
A: 确保先运行训练脚本生成模型文件。

### 注意事项

1. **数据质量**: 确保训练数据质量良好，中英文对照准确
2. **计算资源**: 训练需要一定的计算资源，建议使用GPU加速
3. **内存使用**: 大词汇表和长序列会消耗更多内存
4. **训练时间**: 完整训练可能需要数小时到数天
5. **模型评估**: 使用多种指标评估模型性能

### 性能优化建议

- 使用GPU加速训练
- 调整批次大小平衡速度和内存使用
- 使用混合精度训练（如果支持）
- 定期保存检查点避免训练中断
- 监控验证损失避免过拟合

## 模型架构

### Transformer 结构

本项目实现了标准的 Transformer 架构，包括：

1. **编码器 (Encoder)**
   - 多头自注意力机制
   - 位置前馈网络
   - 残差连接和层归一化
   - 位置编码

2. **解码器 (Decoder)**
   - 掩码多头自注意力
   - 编码器-解码器注意力
   - 位置前馈网络
   - 残差连接和层归一化

3. **注意力机制**
   - 多头注意力
   - 缩放点积注意力
   - 相对位置编码（可选）

### 关键特性

- **位置编码**：支持正弦位置编码和可学习位置嵌入
- **注意力掩码**：支持填充掩码和前瞻掩码
- **层归一化**：Pre-LN 和 Post-LN 两种方式
- **激活函数**：ReLU、GELU、Swish 等多种选择

## 评估指标

### 支持的指标

1. **BLEU**: 双语评估替补 (Bilingual Evaluation Understudy)
2. **METEOR**: 基于词干、同义词和释义的评估
3. **ROUGE**: 面向摘要的评估指标
4. **CER**: 字符错误率
5. **WER**: 词错误率
6. **困惑度**: 模型的困惑度指标

### 使用示例

```python
from utils.metrics import TranslationMetrics, calculate_bleu

# 计算 BLEU 分数
bleu = calculate_bleu(["Hello world"], [["Hello world"]])
print(f"BLEU: {bleu:.4f}")

# 使用完整评估指标
metrics = TranslationMetrics()
meteor = metrics.calculate_meteor("Hello world", "Hello world")
rouge = metrics.calculate_rouge("Hello world", "Hello world")
```

## 可视化功能

### 注意力可视化

```python
from utils.visualization import AttentionVisualizer

visualizer = AttentionVisualizer()

# 绘制注意力热力图
visualizer.plot_attention_heatmap(
    attention_weights,
    source_tokens,
    target_tokens,
    save_path="attention.png"
)

# 绘制多头注意力
visualizer.plot_multihead_attention(
    attention_weights,
    source_tokens,
    target_tokens,
    save_path="multihead_attention.png"
)
```

### 训练曲线可视化

```python
from utils.visualization import TrainingVisualizer

visualizer = TrainingVisualizer()

# 绘制损失曲线
visualizer.plot_loss_curves(
    train_losses,
    val_losses,
    save_path="loss_curves.png"
)

# 绘制学习率调度
visualizer.plot_learning_rate_schedule(
    learning_rates,
    save_path="lr_schedule.png"
)
```

## 高级功能

### 混合精度训练

在配置中启用混合精度训练：

```python
config.use_amp = True
config.amp_opt_level = "O1"
```

### 梯度累积

```python
config.gradient_accumulation_steps = 4
```

### 早停机制

```python
config.early_stopping = True
config.patience = 10
config.min_delta = 1e-4
```

### 学习率调度

支持多种学习率调度策略：

- `linear`: 线性衰减
- `cosine`: 余弦退火
- `exponential`: 指数衰减
- `step`: 阶梯衰减
- `transformer`: Transformer 原始调度

## 性能优化

### 训练优化

1. **数据并行**：支持多 GPU 训练
2. **混合精度**：减少显存使用，加速训练
3. **梯度累积**：模拟大批量训练
4. **动态填充**：减少计算浪费

### 推理优化

1. **批量推理**：提高吞吐量
2. **缓存机制**：加速解码过程
3. **束搜索优化**：高效的束搜索实现
4. **量化支持**：减少模型大小

## 故障排除

### 常见问题

1. **显存不足**
   - 减小 `batch_size`
   - 启用梯度累积
   - 使用混合精度训练

2. **训练不收敛**
   - 调整学习率
   - 增加预热步数
   - 检查数据质量

3. **翻译质量差**
   - 增加训练数据
   - 调整模型大小
   - 优化超参数

### 调试技巧

1. **启用详细日志**：
   ```bash
   python train.py --log-level DEBUG
   ```

2. **可视化注意力权重**：
   ```python
   # 在推理时启用注意力输出
   config.return_attention = True
   ```

3. **监控训练指标**：
   - 使用 TensorBoard 或 Weights & Biases
   - 定期评估验证集

## 扩展开发

### 添加新的注意力机制

1. 在 `models/attention.py` 中实现新的注意力类
2. 在 `models/transformer.py` 中集成
3. 更新配置选项

### 添加新的损失函数

1. 在 `training/loss.py` 中实现新的损失类
2. 在 `create_loss_function` 中注册
3. 更新配置文档

### 添加新的评估指标

1. 在 `utils/metrics.py` 中实现新的指标函数
2. 在 `TranslationMetrics` 类中集成
3. 更新评估脚本

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 致谢

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 优秀的 Transformer 教程
- PyTorch 团队 - 深度学习框架

## 项目特点

### 技术特色

- **完整实现**: 从零开始实现Transformer架构
- **端到端流程**: 数据处理→训练→推理→演示完整链路
- **实用性强**: 专注中译英任务，适合学习和研究
- **代码清晰**: 模块化设计，易于理解和扩展
- **文档完善**: 详细的使用说明和注释

### 学习价值

- 深入理解Transformer架构原理
- 掌握神经机器翻译完整流程
- 学习PyTorch深度学习框架使用
- 了解自然语言处理项目开发

## 贡献指南

欢迎贡献代码和改进建议！

### 贡献方式

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档
- 编写单元测试
- 确保代码可以正常运行

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- 感谢 PyTorch 团队提供优秀的深度学习框架
- 感谢 Hugging Face 提供的 Transformers 库参考
- 感谢开源社区的贡献和支持

---

**如果这个项目对您有帮助，请给个 ⭐ Star！**