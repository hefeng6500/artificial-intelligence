#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN 时间序列预测演示

本演示展示了使用RNN进行时间序列预测的完整流程：
1. 数据集的梳理和预处理
2. 模型构建和训练
3. 模型预测
4. 模型评估和可视化

作者: AI Assistant
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_processor import DataProcessor
from rnn_model import SimpleRNN, LSTMModel, RNNTrainer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """
    主函数：执行完整的RNN演示流程
    """
    print("="*60)
    print("RNN 时间序列预测演示")
    print("="*60)
    
    # ========================================
    # 第一步：数据集的梳理和预处理
    # ========================================
    print("\n🔍 第一步：数据集的梳理和预处理")
    print("-" * 40)
    
    # 创建数据处理器
    sequence_length = 20  # 时间序列长度
    processor = DataProcessor(sequence_length=sequence_length)
    
    # 生成示例时间序列数据
    print("1.1 生成示例数据...")
    data = processor.generate_sample_data(n_samples=2000)
    
    # 显示数据基本信息
    info = processor.get_data_info()
    print("\n数据集基本信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 数据归一化
    print("\n1.2 数据归一化...")
    scaled_data = processor.normalize_data()
    
    # 创建时间序列数据集
    print("\n1.3 创建时间序列数据集...")
    X, y = processor.create_sequences()
    
    # 分割训练集、验证集和测试集
    print("\n1.4 分割数据集...")
    # 先分出测试集
    X_temp, X_test, y_temp, y_test = processor.split_data(X, y, test_size=0.2, random_state=42)
    # 再从剩余数据中分出验证集
    X_train, X_val, y_train, y_val = processor.split_data(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    # 可视化原始数据
    print("\n1.5 数据可视化...")
    processor.visualize_data(n_points=500)
    
    # ========================================
    # 第二步：模型构建和训练
    # ========================================
    print("\n🏗️ 第二步：模型构建和训练")
    print("-" * 40)
    
    # 训练配置
    config = {
        'input_size': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'patience': 15
    }
    
    print("\n2.1 模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 比较不同模型
    models_to_compare = {
        'Simple RNN': SimpleRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            dropout=config['dropout']
        ),
        'LSTM': LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size'],
            dropout=config['dropout']
        )
    }
    
    trained_models = {}
    model_metrics = {}
    
    for model_name, model in models_to_compare.items():
        print(f"\n2.2 训练 {model_name} 模型...")
        
        # 创建训练器
        trainer = RNNTrainer(model)
        
        # 显示模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数数量: {total_params:,}")
        
        # 训练模型
        trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            patience=config['patience'],
            save_path=f'best_{model_name.lower().replace(" ", "_")}_model.pth'
        )
        
        # 保存训练好的模型
        trained_models[model_name] = trainer
        
        # 绘制训练历史
        print(f"\n2.3 {model_name} 训练历史可视化...")
        trainer.plot_training_history()
    
    # ========================================
    # 第三步：模型预测
    # ========================================
    print("\n🔮 第三步：模型预测")
    print("-" * 40)
    
    predictions = {}
    
    for model_name, trainer in trained_models.items():
        print(f"\n3.1 使用 {model_name} 进行预测...")
        
        # 在测试集上进行预测
        pred = trainer.predict(X_test)
        predictions[model_name] = pred
        
        print(f"  预测完成，生成了 {len(pred)} 个预测值")
    
    # ========================================
    # 第四步：模型评估和可视化
    # ========================================
    print("\n📊 第四步：模型评估和可视化")
    print("-" * 40)
    
    # 评估所有模型
    for model_name, trainer in trained_models.items():
        print(f"\n4.1 评估 {model_name} 模型性能...")
        
        # 评估模型
        metrics, pred = trainer.evaluate(X_test, y_test)
        model_metrics[model_name] = metrics
        
        # 反归一化预测结果用于可视化
        y_test_original = processor.inverse_transform(y_test)
        pred_original = processor.inverse_transform(pred)
        
        # 绘制预测结果
        print(f"\n4.2 {model_name} 预测结果可视化...")
        trainer.plot_predictions(y_test_original.flatten(), pred_original.flatten(), n_points=200)
    
    # 模型性能对比
    print("\n4.3 模型性能对比")
    print("-" * 40)
    
    # 创建性能对比表
    metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
    
    print(f"{'模型':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE(%)':<12}")
    print("-" * 65)
    
    for model_name, metrics in model_metrics.items():
        print(f"{model_name:<15} {metrics['MSE']:<12.6f} {metrics['RMSE']:<12.6f} "
              f"{metrics['MAE']:<12.6f} {metrics['MAPE']:<12.2f}")
    
    # 绘制性能对比图
    plot_model_comparison(model_metrics)
    
    # ========================================
    # 第五步：未来预测演示
    # ========================================
    print("\n🚀 第五步：未来预测演示")
    print("-" * 40)
    
    # 选择最佳模型进行未来预测
    best_model_name = min(model_metrics.keys(), key=lambda x: model_metrics[x]['RMSE'])
    best_trainer = trained_models[best_model_name]
    
    print(f"\n5.1 使用最佳模型 ({best_model_name}) 进行未来预测...")
    
    # 进行多步预测
    future_steps = 50
    future_predictions = predict_future(best_trainer, X_test[-1:], future_steps)
    
    # 可视化未来预测
    plot_future_predictions(processor, y_test, future_predictions, future_steps)
    
    # ========================================
    # 总结
    # ========================================
    print("\n📋 演示总结")
    print("="*60)
    print(f"✅ 数据处理: 生成了 {len(data)} 个时间序列数据点")
    print(f"✅ 模型训练: 训练了 {len(trained_models)} 个不同的RNN模型")
    print(f"✅ 模型评估: 最佳模型是 {best_model_name}，RMSE: {model_metrics[best_model_name]['RMSE']:.6f}")
    print(f"✅ 未来预测: 预测了未来 {future_steps} 个时间步")
    print("\n🎉 RNN时间序列预测演示完成！")
    print("="*60)

def plot_model_comparison(model_metrics):
    """
    绘制模型性能对比图
    
    Args:
        model_metrics (dict): 模型评估指标字典
    """
    metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
    model_names = list(model_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_names):
        values = [model_metrics[model][metric] for model in model_names]
        
        bars = axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric} 对比')
        axes[i].set_ylabel(metric)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('模型性能对比', fontsize=16, y=1.02)
    plt.show()

def predict_future(trainer, last_sequence, steps):
    """
    进行多步未来预测
    
    Args:
        trainer (RNNTrainer): 训练好的模型
        last_sequence (np.array): 最后一个输入序列
        steps (int): 预测步数
        
    Returns:
        np.array: 未来预测值
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # 预测下一个值
        next_pred = trainer.predict(current_sequence)
        predictions.append(next_pred[0])
        
        # 更新序列：移除第一个元素，添加预测值
        new_sequence = np.zeros_like(current_sequence)
        new_sequence[0, :-1, :] = current_sequence[0, 1:, :]
        new_sequence[0, -1, 0] = next_pred[0]
        current_sequence = new_sequence
    
    return np.array(predictions)

def plot_future_predictions(processor, y_test, future_predictions, future_steps):
    """
    可视化未来预测结果
    
    Args:
        processor (DataProcessor): 数据处理器
        y_test (np.array): 测试集真实值
        future_predictions (np.array): 未来预测值
        future_steps (int): 预测步数
    """
    # 反归一化
    y_test_original = processor.inverse_transform(y_test)
    future_original = processor.inverse_transform(future_predictions)
    
    plt.figure(figsize=(15, 8))
    
    # 显示最后100个测试点和未来预测
    n_show = min(100, len(y_test_original))
    
    # 测试集数据
    test_x = range(len(y_test_original) - n_show, len(y_test_original))
    test_y = y_test_original[-n_show:].flatten()
    
    # 未来预测数据
    future_x = range(len(y_test_original), len(y_test_original) + future_steps)
    future_y = future_original.flatten()
    
    plt.plot(test_x, test_y, label='历史数据', color='blue', linewidth=2)
    plt.plot(future_x, future_y, label=f'未来预测 ({future_steps}步)', 
             color='red', linewidth=2, linestyle='--')
    
    # 添加分界线
    plt.axvline(x=len(y_test_original)-1, color='green', linestyle=':', 
                linewidth=2, label='预测起点')
    
    plt.title('时间序列未来预测', fontsize=16)
    plt.xlabel('时间步')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加预测区域阴影
    plt.fill_between(future_x, future_y, alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n未来 {future_steps} 步预测统计:")
    print(f"  预测均值: {np.mean(future_original):.4f}")
    print(f"  预测标准差: {np.std(future_original):.4f}")
    print(f"  预测范围: [{np.min(future_original):.4f}, {np.max(future_original):.4f}]")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 运行主程序
    main()