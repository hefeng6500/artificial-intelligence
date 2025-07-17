# 导入PyTorch神经网络模块
from torch import nn
# 导入配置文件，包含模型超参数
import config


# 定义输入法模型类，继承自PyTorch的nn.Module
class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        """初始化输入法模型
        Args:
            vocab_size: 词汇表大小，即字符种类数量
        """
        # 调用父类构造函数
        super().__init__()
        
        # 创建词嵌入层，将字符ID转换为向量表示
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小
            embedding_dim=config.EMBEDDING_DIM  # 嵌入向量维度
        )
        
        # 创建RNN循环神经网络层，用于学习序列模式
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,  # 输入特征维度（等于嵌入维度）
            hidden_size=config.HIDDEN_SIZE,   # 隐藏层大小
            batch_first=True,  # 批次维度在第一维
        )
        
        # 创建线性输出层，将隐藏状态映射到词汇表概率分布
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,  # 输入特征数（等于RNN隐藏层大小）
            out_features=vocab_size  # 输出特征数（等于词汇表大小）
        )

    def forward(self, x):
        """前向传播函数
        Args:
            x: 输入张量，形状为[batch_size, seq_len]
        Returns:
            output: 预测结果，形状为[batch_size, vocab_size]
        """
        # x.shape: [batch_size, seq_len] - 输入的字符ID序列
        
        # 将字符ID转换为嵌入向量
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim] - 嵌入后的向量序列
        
        # 通过RNN处理序列，获取所有时间步的输出和最终隐藏状态
        output, hidden = self.rnn(embed)
        # output.shape: [batch_size, seq_len, hidden_dim] - 所有时间步的输出
        # hidden.shape: [1, batch_size, hidden_dim] - 最终隐藏状态

        # 取最后一个时间步的输出作为序列的表示
        last_hidden = output[:, -1, :]
        # 备选方案：last_hidden = hidden.squeeze(0) - 也可以直接使用最终隐藏状态

        # last_hidden.shape: [batch_size, hidden_dim] - 序列的最终表示
        
        # 通过线性层将隐藏状态映射到词汇表概率分布
        output = self.linear(last_hidden)
        # output.shape: [batch_size, vocab_size] - 每个字符的预测概率
        
        return output
