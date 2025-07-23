import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class TimeSeriesDataGenerator:
    """
    时间序列数据生成器
    用于生成模拟的时间序列数据
    """
    
    @staticmethod
    def generate_sine_wave(length=1000, frequency=0.1, noise_level=0.1):
        """生成正弦波数据"""
        x = np.linspace(0, length * frequency * 2 * np.pi, length)
        y = np.sin(x) + np.random.normal(0, noise_level, length)
        return y
    
    @staticmethod
    def generate_trend_data(length=1000, trend=0.01, noise_level=0.1):
        """生成带趋势的数据"""
        x = np.arange(length)
        y = trend * x + np.random.normal(0, noise_level, length)
        return y
    
    @staticmethod
    def generate_seasonal_data(length=1000, period=50, amplitude=1.0, noise_level=0.1):
        """生成季节性数据"""
        x = np.arange(length)
        y = amplitude * np.sin(2 * np.pi * x / period) + np.random.normal(0, noise_level, length)
        return y
    
    @staticmethod
    def generate_complex_series(length=1000):
        """生成复杂时间序列（包含趋势、季节性和噪声）"""
        x = np.arange(length)
        trend = 0.01 * x
        seasonal = 2 * np.sin(2 * np.pi * x / 50) + np.sin(2 * np.pi * x / 20)
        noise = np.random.normal(0, 0.5, length)
        return trend + seasonal + noise

class TimeSeriesPreprocessor:
    """
    时间序列数据预处理器
    """
    
    def __init__(self, sequence_length=10, prediction_steps=1):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scaler = MinMaxScaler()
        
    def normalize_data(self, data):
        """数据归一化"""
        data_reshaped = data.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(data_reshaped)
        return normalized_data.flatten()
    
    def denormalize_data(self, normalized_data):
        """反归一化"""
        data_reshaped = normalized_data.reshape(-1, 1)
        original_data = self.scaler.inverse_transform(data_reshaped)
        return original_data.flatten()
    
    def create_sequences(self, data):
        """创建序列数据"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_steps + 1):
            # 输入序列
            X.append(data[i:(i + self.sequence_length)])
            # 目标值（预测未来 prediction_steps 步）
            y.append(data[i + self.sequence_length:i + self.sequence_length + self.prediction_steps])
            
        return np.array(X), np.array(y)
    
    def prepare_data(self, data, train_ratio=0.8):
        """准备训练和测试数据"""
        # 归一化
        normalized_data = self.normalize_data(data)
        
        # 创建序列
        X, y = self.create_sequences(normalized_data)
        
        # 分割训练和测试集
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        """创建 DataLoader"""
        # 转换为 PyTorch 张量
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # 添加特征维度
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

def plot_time_series(data, title="时间序列数据", figsize=(12, 6)):
    """绘制时间序列数据"""
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.grid(True)
    plt.show()

def plot_predictions(actual, predicted, title="预测结果对比", figsize=(12, 6)):
    """绘制预测结果对比"""
    plt.figure(figsize=figsize)
    plt.plot(actual, label='实际值', alpha=0.7)
    plt.plot(predicted, label='预测值', alpha=0.7)
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.show()