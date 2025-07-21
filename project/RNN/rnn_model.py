import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class SimpleRNN(nn.Module):
    """
    简单的RNN模型类
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化RNN模型
        
        Args:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层大小
            num_layers (int): RNN层数
            output_size (int): 输出维度
            dropout (float): Dropout比例
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, output_size]
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN前向传播
        out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out

class LSTMModel(nn.Module):
    """
    LSTM模型类
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层大小
            num_layers (int): LSTM层数
            output_size (int): 输出维度
            dropout (float): Dropout比例
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, output_size]
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out

class RNNTrainer:
    """
    RNN训练器类
    """
    
    def __init__(self, model, device=None):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 要训练的模型
            device (str): 设备类型 ('cuda' 或 'cpu')
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        
        print(f"使用设备: {self.device}")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, learning_rate=0.001, 
              patience=10, save_path='best_model.pth'):
        """
        训练模型
        
        Args:
            X_train (np.array): 训练特征
            y_train (np.array): 训练标签
            X_val (np.array): 验证特征
            y_val (np.array): 验证标签
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            learning_rate (float): 学习率
            patience (int): 早停耐心值
            save_path (str): 模型保存路径
        """
        print(f"开始训练模型，共 {epochs} 轮...")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证集
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                
                # 学习率调度
                scheduler.step(avg_val_loss)
                
                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                
                # 打印进度
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {avg_val_loss:.6f}, '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                # 早停
                if patience_counter >= patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            else:
                # 没有验证集时的打印
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}')
        
        print("训练完成！")
        
        # 加载最佳模型
        if os.path.exists(save_path) and X_val is not None:
            self.model.load_state_dict(torch.load(save_path))
            print(f"已加载最佳模型: {save_path}")
    
    def predict(self, X):
        """
        模型预测
        
        Args:
            X (np.array): 输入特征
            
        Returns:
            np.array: 预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions.squeeze()
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test (np.array): 测试特征
            y_test (np.array): 测试标签
            
        Returns:
            dict: 评估指标
        """
        print("正在评估模型性能...")
        
        # 预测
        predictions = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        
        # 计算MAPE（平均绝对百分比误差）
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("\n模型评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics, predictions
    
    def plot_training_history(self):
        """
        绘制训练历史
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        if self.val_losses:
            plt.plot(self.val_losses, label='验证损失')
        plt.title('训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        if self.val_losses:
            plt.plot(self.val_losses, label='验证损失')
        plt.title('训练历史（对数尺度）')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, n_points=200):
        """
        绘制预测结果
        
        Args:
            y_true (np.array): 真实值
            y_pred (np.array): 预测值
            n_points (int): 显示的点数
        """
        plt.figure(figsize=(15, 8))
        
        # 限制显示点数
        n_points = min(n_points, len(y_true))
        
        plt.subplot(2, 2, 1)
        plt.plot(y_true[:n_points], label='真实值', alpha=0.7)
        plt.plot(y_pred[:n_points], label='预测值', alpha=0.7)
        plt.title('预测结果对比')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title('真实值 vs 预测值')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        residuals = y_true - y_pred
        plt.plot(residuals[:n_points])
        plt.title('残差图')
        plt.xlabel('时间步')
        plt.ylabel('残差')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title('残差分布')
        plt.xlabel('残差')
        plt.ylabel('频次')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = SimpleRNN(input_size=1, hidden_size=50, num_layers=2)
    
    # 创建训练器
    trainer = RNNTrainer(model)
    
    print("RNN模型创建完成！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")