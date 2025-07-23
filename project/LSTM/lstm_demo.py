#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 时间序列预测演示

这个演示展示了如何使用 LSTM 神经网络进行时间序列预测。
包括数据生成、模型训练、预测和结果可视化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from lstm_model import LSTMModel, LSTMTrainer
from data_utils import TimeSeriesDataGenerator, TimeSeriesPreprocessor, plot_time_series, plot_predictions

def main():
    print("=" * 50)
    print("LSTM 时间序列预测演示")
    print("=" * 50)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 生成时间序列数据
    print("\n1. 生成时间序列数据...")
    data_length = 1000
    
    # 生成复杂时间序列数据（包含趋势、季节性和噪声）
    data = TimeSeriesDataGenerator.generate_complex_series(data_length)
    
    print(f"生成了 {len(data)} 个数据点")
    
    # 可视化原始数据
    plot_time_series(data, "原始时间序列数据")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    sequence_length = 20  # 使用过去20个时间步预测未来
    prediction_steps = 1   # 预测未来1步
    
    preprocessor = TimeSeriesPreprocessor(sequence_length=sequence_length, 
                                         prediction_steps=prediction_steps)
    
    # 准备训练和测试数据
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, train_ratio=0.8)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建 DataLoader
    batch_size = 32
    train_loader, test_loader = preprocessor.create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=batch_size
    )
    
    # 3. 创建 LSTM 模型
    print("\n3. 创建 LSTM 模型...")
    input_size = 1        # 每个时间步的特征数
    hidden_size = 50      # LSTM 隐藏层大小
    num_layers = 2        # LSTM 层数
    output_size = prediction_steps  # 输出大小
    dropout = 0.2         # Dropout 比率
    
    model = LSTMModel(input_size=input_size,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     output_size=output_size,
                     dropout=dropout)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 设置训练参数
    print("\n4. 设置训练参数...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建训练器
    trainer = LSTMTrainer(model, criterion, optimizer, device)
    
    # 5. 训练模型
    print("\n5. 开始训练模型...")
    epochs = 50
    
    train_losses, val_losses = trainer.train(train_loader, test_loader, epochs)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程中的损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 6. 模型预测
    print("\n6. 进行预测...")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # 反归一化预测结果
    predictions_denorm = preprocessor.denormalize_data(predictions)
    actuals_denorm = preprocessor.denormalize_data(actuals)
    
    # 7. 评估模型性能
    print("\n7. 模型性能评估...")
    mse = np.mean((predictions_denorm - actuals_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    
    # 8. 可视化预测结果
    print("\n8. 可视化预测结果...")
    
    # 只显示前200个预测点以便更好地观察
    display_length = min(200, len(predictions_denorm))
    
    plot_predictions(actuals_denorm[:display_length], 
                    predictions_denorm[:display_length],
                    "LSTM 预测结果对比（前200个点）")
    
    # 9. 单步预测演示
    print("\n9. 单步预测演示...")
    
    # 使用最后一个序列进行预测
    last_sequence = torch.FloatTensor(X_test[-1:]).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        prediction = model(last_sequence)
        prediction_denorm = preprocessor.denormalize_data(prediction.cpu().numpy())
    
    print(f"输入序列的最后几个值: {preprocessor.denormalize_data(X_test[-1][-5:])[:5]}")
    print(f"预测的下一个值: {prediction_denorm[0]:.4f}")
    print(f"实际的下一个值: {preprocessor.denormalize_data(y_test[-1:])[:1]}")
    
    print("\n=" * 50)
    print("演示完成！")
    print("=" * 50)

def demo_different_data_types():
    """
    演示不同类型的时间序列数据
    """
    print("\n" + "=" * 50)
    print("不同类型时间序列数据演示")
    print("=" * 50)
    
    # 生成不同类型的数据
    sine_data = TimeSeriesDataGenerator.generate_sine_wave(500, frequency=0.05)
    trend_data = TimeSeriesDataGenerator.generate_trend_data(500, trend=0.02)
    seasonal_data = TimeSeriesDataGenerator.generate_seasonal_data(500, period=30)
    
    # 可视化不同类型的数据
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(sine_data)
    axes[0, 0].set_title('正弦波数据')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(trend_data)
    axes[0, 1].set_title('趋势数据')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(seasonal_data)
    axes[1, 0].set_title('季节性数据')
    axes[1, 0].grid(True)
    
    complex_data = TimeSeriesDataGenerator.generate_complex_series(500)
    axes[1, 1].plot(complex_data)
    axes[1, 1].set_title('复杂时间序列')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行主演示
    main()
    
    # 运行不同数据类型演示
    demo_different_data_types()