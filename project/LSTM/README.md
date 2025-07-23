# LSTM 时间序列预测演示

这是一个完整的 LSTM（长短期记忆网络）演示项目，用于时间序列预测任务。项目包含了从数据生成、模型定义、训练到预测的完整流程。

## 项目结构

```
LSTM/
├── lstm_model.py      # LSTM 模型定义和训练器
├── data_utils.py      # 数据生成和预处理工具
├── lstm_demo.py       # 完整的演示脚本
├── quick_test.py      # 快速测试脚本
├── requirements.txt   # 依赖包列表
└── README.md         # 项目说明文档
```

## 功能特性

### 1. LSTM 模型 (`lstm_model.py`)
- **LSTMModel**: 可配置的 LSTM 神经网络模型
  - 支持多层 LSTM
  - 包含 Dropout 防止过拟合
  - 灵活的输入输出维度配置
- **LSTMTrainer**: 模型训练器
  - 完整的训练循环
  - 验证集评估
  - 训练过程监控

### 2. 数据工具 (`data_utils.py`)
- **TimeSeriesDataGenerator**: 时间序列数据生成器
  - 正弦波数据
  - 趋势数据
  - 季节性数据
  - 复杂时间序列（趋势+季节性+噪声）
- **TimeSeriesPreprocessor**: 数据预处理器
  - 数据归一化
  - 序列创建
  - 训练/测试集分割
  - PyTorch DataLoader 创建

### 3. 演示脚本
- **lstm_demo.py**: 完整演示
  - 数据生成和可视化
  - 模型训练
  - 性能评估
  - 预测结果可视化
- **quick_test.py**: 快速测试
  - 简化的训练流程
  - 交互式预测演示
  - 适合初学者理解

## 安装依赖

```bash
pip install torch numpy matplotlib scikit-learn
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速开始

运行快速测试脚本：

```bash
python quick_test.py
```

这将：
1. 生成简单的正弦波数据
2. 训练一个小型 LSTM 模型
3. 显示训练过程和预测结果
4. 演示交互式预测

### 完整演示

运行完整演示脚本：

```bash
python lstm_demo.py
```

这将：
1. 生成复杂的时间序列数据
2. 进行完整的数据预处理
3. 训练更复杂的 LSTM 模型
4. 详细的性能评估
5. 多种数据类型演示

### 自定义使用

```python
from lstm_model import LSTMModel, LSTMTrainer
from data_utils import TimeSeriesDataGenerator, TimeSeriesPreprocessor

# 1. 生成数据
data = TimeSeriesDataGenerator.generate_complex_series(1000)

# 2. 预处理
preprocessor = TimeSeriesPreprocessor(sequence_length=20, prediction_steps=1)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(data)
train_loader, test_loader = preprocessor.create_dataloaders(X_train, X_test, y_train, y_test)

# 3. 创建模型
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)

# 4. 训练
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = LSTMTrainer(model, criterion, optimizer)
train_losses, val_losses = trainer.train(train_loader, test_loader, epochs=50)

# 5. 预测
predictions = model.predict(X_test_tensor)
```

## 模型参数说明

### LSTMModel 参数
- `input_size`: 输入特征维度（通常为1，表示单变量时间序列）
- `hidden_size`: LSTM 隐藏层大小（控制模型容量）
- `num_layers`: LSTM 层数（增加模型深度）
- `output_size`: 输出维度（预测步数）
- `dropout`: Dropout 比率（防止过拟合）

### TimeSeriesPreprocessor 参数
- `sequence_length`: 输入序列长度（用多少个历史点预测）
- `prediction_steps`: 预测步数（预测未来多少步）

## 性能评估指标

项目使用以下指标评估模型性能：
- **MSE (Mean Squared Error)**: 均方误差
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **MAE (Mean Absolute Error)**: 平均绝对误差

## 可视化功能

- 原始时间序列数据可视化
- 训练损失曲线
- 预测结果对比图
- 不同类型时间序列数据展示

## 扩展建议

1. **多变量时间序列**: 修改 `input_size` 支持多个特征
2. **注意力机制**: 在 LSTM 基础上添加注意力层
3. **双向 LSTM**: 使用 `bidirectional=True`
4. **序列到序列**: 修改模型支持多步预测
5. **实时预测**: 添加在线学习功能

## 常见问题

### Q: 如何处理不同长度的时间序列？
A: 使用 padding 或者动态批处理，可以参考 PyTorch 的 `pad_sequence` 函数。

### Q: 如何提高模型性能？
A: 
- 增加模型复杂度（更多层数或隐藏单元）
- 调整学习率
- 使用更长的输入序列
- 添加正则化技术
- 尝试不同的优化器

### Q: 如何处理非平稳时间序列？
A: 
- 使用差分处理
- 添加趋势和季节性分解
- 使用更复杂的归一化方法

### Q: 模型训练很慢怎么办？
A: 
- 减少序列长度或批大小
- 使用 GPU 加速
- 减少模型复杂度
- 使用更高效的优化器如 AdamW

## 技术细节

- **框架**: PyTorch
- **Python 版本**: 3.7+
- **主要依赖**: torch, numpy, matplotlib, scikit-learn
- **模型类型**: 多层 LSTM + 全连接层
- **损失函数**: 均方误差 (MSE)
- **优化器**: Adam

## 许可证

本项目仅用于学习和演示目的。