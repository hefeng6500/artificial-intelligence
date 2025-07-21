import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataProcessor:
    """
    数据处理类，负责数据的加载、预处理和序列化
    """
    
    def __init__(self, sequence_length=10):
        """
        初始化数据处理器
        
        Args:
            sequence_length (int): 时间序列长度
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def generate_sample_data(self, n_samples=1000):
        """
        生成示例时间序列数据（正弦波 + 噪声）
        
        Args:
            n_samples (int): 样本数量
            
        Returns:
            np.array: 生成的时间序列数据
        """
        print("正在生成示例时间序列数据...")
        
        # 生成时间点
        t = np.linspace(0, 4*np.pi, n_samples)
        
        # 生成复合信号：正弦波 + 余弦波 + 噪声
        signal = (np.sin(t) + 0.5 * np.sin(3*t) + 0.3 * np.cos(2*t) + 
                 0.1 * np.random.normal(0, 1, n_samples))
        
        # 添加趋势
        trend = 0.001 * t
        signal += trend
        
        self.data = signal.reshape(-1, 1)
        print(f"生成了 {n_samples} 个数据点")
        
        return self.data
    
    def load_data_from_csv(self, file_path, column_name):
        """
        从CSV文件加载数据
        
        Args:
            file_path (str): CSV文件路径
            column_name (str): 目标列名
        """
        try:
            df = pd.read_csv(file_path)
            self.data = df[column_name].values.reshape(-1, 1)
            print(f"从 {file_path} 加载了 {len(self.data)} 个数据点")
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
            
    def normalize_data(self):
        """
        对数据进行归一化处理
        
        Returns:
            np.array: 归一化后的数据
        """
        if self.data is None:
            raise ValueError("请先加载或生成数据")
            
        print("正在对数据进行归一化...")
        self.scaled_data = self.scaler.fit_transform(self.data)
        print("数据归一化完成")
        
        return self.scaled_data
    
    def create_sequences(self, data=None):
        """
        创建时间序列数据集
        
        Args:
            data (np.array): 输入数据，如果为None则使用归一化后的数据
            
        Returns:
            tuple: (X, y) 特征和标签
        """
        if data is None:
            if self.scaled_data is None:
                raise ValueError("请先进行数据归一化")
            data = self.scaled_data
            
        print(f"正在创建长度为 {self.sequence_length} 的时间序列...")
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            # 输入序列
            X.append(data[i:(i + self.sequence_length), 0])
            # 目标值（下一个时间点的值）
            y.append(data[i + self.sequence_length, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # 重塑为RNN需要的格式 [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"创建了 {len(X)} 个序列")
        print(f"输入形状: {X.shape}, 输出形状: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        分割训练集和测试集
        
        Args:
            X (np.array): 特征数据
            y (np.array): 标签数据
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"正在分割数据集，测试集比例: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, data):
        """
        反归一化数据
        
        Args:
            data (np.array): 归一化的数据
            
        Returns:
            np.array: 原始尺度的数据
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1))
    
    def visualize_data(self, n_points=200):
        """
        可视化原始数据
        
        Args:
            n_points (int): 显示的数据点数量
        """
        if self.data is None:
            print("没有数据可以可视化")
            return
            
        plt.figure(figsize=(12, 6))
        
        # 显示原始数据
        plt.subplot(1, 2, 1)
        plt.plot(self.data[:n_points])
        plt.title('原始时间序列数据')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.grid(True)
        
        # 显示归一化数据（如果存在）
        if self.scaled_data is not None:
            plt.subplot(1, 2, 2)
            plt.plot(self.scaled_data[:n_points])
            plt.title('归一化后的数据')
            plt.xlabel('时间步')
            plt.ylabel('归一化数值')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def get_data_info(self):
        """
        获取数据集信息
        
        Returns:
            dict: 数据集统计信息
        """
        if self.data is None:
            return {"status": "无数据"}
            
        info = {
            "数据点数量": len(self.data),
            "数据形状": self.data.shape,
            "最小值": float(np.min(self.data)),
            "最大值": float(np.max(self.data)),
            "均值": float(np.mean(self.data)),
            "标准差": float(np.std(self.data)),
            "序列长度": self.sequence_length
        }
        
        return info

# 使用示例
if __name__ == "__main__":
    # 创建数据处理器
    processor = DataProcessor(sequence_length=10)
    
    # 生成示例数据
    data = processor.generate_sample_data(n_samples=1000)
    
    # 归一化数据
    scaled_data = processor.normalize_data()
    
    # 创建序列
    X, y = processor.create_sequences()
    
    # 分割数据集
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # 显示数据信息
    info = processor.get_data_info()
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 可视化数据
    processor.visualize_data()