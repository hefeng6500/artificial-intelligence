# RNN 时间序列预测演示项目

这是一个完整的RNN时间序列预测演示项目，展示了从数据处理到模型训练、预测和评估的整个机器学习流程。

## 📁 项目结构

```
RNN/
├── data_processor.py    # 数据处理模块
├── rnn_model.py        # RNN模型定义和训练器
├── rnn_demo.py         # 主演示脚本
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 运行完整演示
python rnn_demo.py
```

## 📊 项目特性

### 🔍 数据处理模块 (`data_processor.py`)

- **自动数据生成**: 生成复合正弦波时间序列数据
- **数据归一化**: MinMax标准化处理
- **序列创建**: 自动创建时间序列训练样本
- **数据分割**: 训练集/验证集/测试集分割
- **数据可视化**: 原始数据和处理后数据的可视化

**主要功能**:
```python
from data_processor import DataProcessor

# 创建数据处理器
processor = DataProcessor(sequence_length=20)

# 生成示例数据
data = processor.generate_sample_data(n_samples=2000)

# 数据预处理
scaled_data = processor.normalize_data()
X, y = processor.create_sequences()
```

### 🏗️ 模型模块 (`rnn_model.py`)

支持多种RNN架构：
- **Simple RNN**: 基础RNN模型
- **LSTM**: 长短期记忆网络
- **可扩展设计**: 易于添加GRU等其他变体

**模型特性**:
- 多层网络支持
- Dropout正则化
- 梯度裁剪
- 早停机制
- 学习率调度

**使用示例**:
```python
from rnn_model import SimpleRNN, LSTMModel, RNNTrainer

# 创建模型
model = SimpleRNN(input_size=1, hidden_size=64, num_layers=2)

# 创建训练器
trainer = RNNTrainer(model)

# 训练模型
trainer.train(X_train, y_train, X_val, y_val, epochs=100)
```

### 🎯 完整演示 (`rnn_demo.py`)

演示包含五个主要步骤：

1. **数据集梳理和预处理**
   - 生成时间序列数据
   - 数据归一化和序列化
   - 训练/验证/测试集分割

2. **模型构建和训练**
   - 比较Simple RNN和LSTM
   - 训练过程可视化
   - 模型参数统计

3. **模型预测**
   - 测试集预测
   - 批量预测处理

4. **模型评估和可视化**
   - 多种评估指标（MSE, RMSE, MAE, MAPE）
   - 预测结果可视化
   - 模型性能对比

5. **未来预测演示**
   - 多步预测
   - 预测结果可视化

## 📈 评估指标

项目使用多种评估指标来全面评估模型性能：

- **MSE (均方误差)**: 衡量预测值与真实值的平方差
- **RMSE (均方根误差)**: MSE的平方根，与原数据同量纲
- **MAE (平均绝对误差)**: 预测值与真实值的绝对差的平均值
- **MAPE (平均绝对百分比误差)**: 相对误差的百分比表示

## 🎨 可视化功能

项目提供丰富的可视化功能：

1. **数据可视化**
   - 原始时间序列数据
   - 归一化后的数据

2. **训练过程可视化**
   - 训练损失曲线
   - 验证损失曲线
   - 对数尺度损失曲线

3. **预测结果可视化**
   - 真实值vs预测值对比
   - 散点图分析
   - 残差分析
   - 残差分布直方图

4. **模型对比可视化**
   - 多模型性能对比柱状图
   - 各项指标对比

5. **未来预测可视化**
   - 历史数据与未来预测
   - 预测区间可视化

## ⚙️ 配置参数

### 数据处理参数
```python
sequence_length = 20      # 时间序列长度
n_samples = 2000         # 数据样本数量
test_size = 0.2          # 测试集比例
```

### 模型参数
```python
config = {
    'input_size': 1,         # 输入特征维度
    'hidden_size': 64,       # 隐藏层大小
    'num_layers': 2,         # RNN层数
    'output_size': 1,        # 输出维度
    'dropout': 0.2,          # Dropout比例
    'epochs': 100,           # 训练轮数
    'batch_size': 32,        # 批次大小
    'learning_rate': 0.001,  # 学习率
    'patience': 15           # 早停耐心值
}
```

## 🔧 自定义使用

### 使用自己的数据

```python
# 从CSV文件加载数据
processor = DataProcessor(sequence_length=20)
processor.load_data_from_csv('your_data.csv', 'target_column')

# 继续后续处理
scaled_data = processor.normalize_data()
X, y = processor.create_sequences()
```

### 调整模型架构

```python
# 创建更大的LSTM模型
model = LSTMModel(
    input_size=1,
    hidden_size=128,    # 增大隐藏层
    num_layers=3,       # 增加层数
    dropout=0.3         # 增加正则化
)
```

### 修改训练参数

```python
# 更长的训练和更小的学习率
trainer.train(
    X_train, y_train, X_val, y_val,
    epochs=200,
    learning_rate=0.0005,
    patience=20
)
```

## 📋 输出示例

运行演示后，你将看到：

```
============================================================
RNN 时间序列预测演示
============================================================

🔍 第一步：数据集的梳理和预处理
----------------------------------------
1.1 生成示例数据...
正在生成示例时间序列数据...
生成了 2000 个数据点

数据集基本信息:
  数据点数量: 2000
  数据形状: (2000, 1)
  最小值: -1.8234
  最大值: 1.9876
  均值: 0.0123
  标准差: 0.8765
  序列长度: 20

...

📋 演示总结
============================================================
✅ 数据处理: 生成了 2000 个时间序列数据点
✅ 模型训练: 训练了 2 个不同的RNN模型
✅ 模型评估: 最佳模型是 LSTM，RMSE: 0.045123
✅ 未来预测: 预测了未来 50 个时间步

🎉 RNN时间序列预测演示完成！
============================================================
```

## 🤝 扩展建议

1. **添加更多模型**: GRU, Transformer等
2. **多变量时间序列**: 支持多特征输入
3. **注意力机制**: 添加注意力层提升性能
4. **超参数优化**: 集成Optuna等优化工具
5. **模型解释性**: 添加SHAP等解释性分析
6. **实时预测**: 支持流式数据预测

## 📚 学习资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [时间序列分析教程](https://otexts.com/fpp3/)
- [RNN原理详解](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 📄 许可证

本项目仅供学习和研究使用。

---

**作者**: AI Assistant  
**日期**: 2024  
**版本**: 1.0.0