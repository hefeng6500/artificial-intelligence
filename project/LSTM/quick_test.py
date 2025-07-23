#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 快速测试脚本

这是一个简化的测试脚本，用于快速验证 LSTM 模型的基本功能。
适合初学者理解 LSTM 的基本工作原理。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from lstm_model import LSTMModel
from data_utils import TimeSeriesDataGenerator, TimeSeriesPreprocessor

def quick_lstm_test():
    """
    快速 LSTM 测试函数
    """
    print("开始 LSTM 快速测试...")
    
    # 设置参数
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 生成简单的正弦波数据
    print("\n1. 生成测试数据...")
    data = TimeSeriesDataGenerator.generate_sine_wave(length=300, frequency=0.1, noise_level=0.05)
    
    # 2. 数据预处理
    print("2. 数据预处理...")
    preprocessor = TimeSeriesPreprocessor(sequence_length=10, prediction_steps=1)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data, train_ratio=0.8)
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"训练数据形状: {X_train_tensor.shape}")
    print(f"测试数据形状: {X_test_tensor.shape}")
    
    # 3. 创建模型
    print("\n3. 创建 LSTM 模型...")
    model = LSTMModel(input_size=1, hidden_size=20, num_layers=1, output_size=1, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. 简单训练
    print("4. 开始训练...")
    epochs = 20
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # 5. 测试预测
    print("\n5. 测试预测...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'测试损失: {test_loss.item():.6f}')
    
    # 6. 可视化结果
    print("6. 可视化结果...")
    
    # 绘制训练损失
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制预测结果
    plt.subplot(1, 2, 2)
    
    # 反归一化数据用于可视化
    actual_denorm = preprocessor.denormalize_data(y_test_tensor.numpy().flatten())
    pred_denorm = preprocessor.denormalize_data(test_predictions.numpy().flatten())
    
    plt.plot(actual_denorm[:50], label='实际值', marker='o', markersize=3)
    plt.plot(pred_denorm[:50], label='预测值', marker='s', markersize=3)
    plt.title('预测结果对比（前50个点）')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 7. 计算评估指标
    mse = np.mean((actual_denorm - pred_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_denorm - pred_denorm))
    
    print(f"\n模型性能:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print("\n快速测试完成！")
    
    return model, preprocessor

def interactive_prediction(model, preprocessor):
    """
    交互式预测演示
    """
    print("\n" + "="*30)
    print("交互式预测演示")
    print("="*30)
    
    # 生成一个新的测试序列
    test_data = TimeSeriesDataGenerator.generate_sine_wave(length=50, frequency=0.1)
    normalized_data = preprocessor.normalize_data(test_data)
    
    # 使用前10个点预测第11个点
    input_sequence = normalized_data[:10]
    actual_next = test_data[10]
    
    # 转换为模型输入格式
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).unsqueeze(-1)
    
    # 预测
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_value = preprocessor.denormalize_data(prediction.numpy())[0]
    
    print(f"输入序列（前10个值）: {test_data[:10]}")
    print(f"实际第11个值: {actual_next:.4f}")
    print(f"预测第11个值: {predicted_value:.4f}")
    print(f"预测误差: {abs(actual_next - predicted_value):.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    x_range = range(11)
    values = list(test_data[:10]) + [actual_next]
    
    plt.plot(x_range[:-1], values[:-1], 'bo-', label='输入序列', markersize=6)
    plt.plot(x_range[-1], actual_next, 'go', label='实际值', markersize=8)
    plt.plot(x_range[-1], predicted_value, 'ro', label='预测值', markersize=8)
    
    plt.title('LSTM 单步预测演示')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 运行快速测试
    model, preprocessor = quick_lstm_test()
    
    # 运行交互式预测演示
    interactive_prediction(model, preprocessor)
    
    print("\n所有测试完成！")