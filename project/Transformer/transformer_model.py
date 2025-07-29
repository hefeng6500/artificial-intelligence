import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    这是 Transformer 的核心组件之一。多头注意力允许模型同时关注不同位置的信息，
    每个"头"可以学习不同类型的依赖关系。
    """
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保可以平均分配到每个头
        
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换层，用于生成 Query、Key、Value
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        
        计算公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            Q: Query 矩阵 [batch_size, num_heads, seq_len, d_k]
            K: Key 矩阵 [batch_size, num_heads, seq_len, d_k]
            V: Value 矩阵 [batch_size, num_heads, seq_len, d_k]
            mask: 掩码矩阵，用于遮蔽某些位置
        
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用 softmax 获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 计算加权的 Value
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询序列 [batch_size, seq_len, d_model]
            key: 键序列 [batch_size, seq_len, d_model]
            value: 值序列 [batch_size, seq_len, d_model]
            mask: 掩码
        
        Returns:
            output: 多头注意力输出
            attention_weights: 注意力权重
        """
        batch_size = query.size(0)
        
        # 1. 线性变换生成 Q、K、V
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在形状为 [batch_size, num_heads, seq_len, d_k]
        
        # 3. 应用缩放点积注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 连接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 5. 最终线性变换
        output = self.W_o(attn_output)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络
    
    这是一个简单的两层全连接网络，用于对每个位置的表示进行非线性变换。
    结构：Linear -> ReLU -> Linear
    """
    
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层：扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层：恢复维度
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    """
    位置编码
    
    由于 Transformer 没有循环或卷积结构，需要添加位置信息来让模型理解序列中的位置关系。
    使用正弦和余弦函数的组合来编码位置信息。
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与梯度更新
        
    def forward(self, x):
        """
        添加位置编码到输入嵌入
        
        Args:
            x: 输入嵌入 [seq_len, batch_size, d_model]
        
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """
    Transformer 块
    
    包含多头注意力和前馈网络，以及残差连接和层归一化。
    这是 Transformer 编码器和解码器的基本构建块。
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        
        Returns:
            output: 输出张量
            attention_weights: 注意力权重
        """
        # 第一个子层：多头注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    由多个 TransformerBlock 堆叠而成，用于编码输入序列。
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer 块堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        编码器前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            mask: 注意力掩码
        
        Returns:
            output: 编码后的表示
            attention_weights: 各层的注意力权重
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放嵌入
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        
        attention_weights = []
        
        # 通过所有 Transformer 块
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    
    由多个解码器层堆叠而成，用于生成输出序列。
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        解码器前向传播
        
        Args:
            x: 目标序列 [batch_size, seq_len]
            encoder_output: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            output: 解码器输出
            attention_weights: 注意力权重
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        
        attention_weights = []
        
        # 通过所有解码器层
        for decoder_layer in self.decoder_layers:
            x, self_attn, cross_attn = decoder_layer(x, encoder_output, src_mask, tgt_mask)
            attention_weights.append((self_attn, cross_attn))
        
        # 输出投影
        output = self.output_projection(x)
        
        return output, attention_weights

class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    包含自注意力、编码器-解码器注意力和前馈网络。
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 自注意力（掩码多头注意力）
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # 编码器-解码器注意力
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        解码器层前向传播
        
        Args:
            x: 解码器输入
            encoder_output: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            output: 层输出
            self_attn: 自注意力权重
            cross_attn: 交叉注意力权重
        """
        # 第一个子层：掩码自注意力
        self_attn_output, self_attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 第二个子层：编码器-解码器注意力
        cross_attn_output, cross_attn = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 第三个子层：前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn, cross_attn

class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    包含编码器和解码器，用于序列到序列的任务。
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 编码器
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff, max_len, dropout
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff, max_len, dropout
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Transformer 前向传播
        
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            output: 模型输出 [batch_size, tgt_len, tgt_vocab_size]
            encoder_attention: 编码器注意力权重
            decoder_attention: 解码器注意力权重
        """
        # 编码
        encoder_output, encoder_attention = self.encoder(src, src_mask)
        
        # 解码（包含输出投影）
        output, decoder_attention = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return output, encoder_attention, decoder_attention
    
    def generate_square_subsequent_mask(self, sz):
        """
        生成下三角掩码，用于防止解码器看到未来的信息
        
        Args:
            sz: 序列长度
        
        Returns:
            mask: 掩码矩阵
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask == 0

def visualize_attention(attention_weights, input_tokens, output_tokens, layer=0, head=0):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重张量
        input_tokens: 输入词汇列表
        output_tokens: 输出词汇列表
        layer: 要可视化的层
        head: 要可视化的注意力头
    """
    # 提取指定层和头的注意力权重
    attn = attention_weights[layer][0, head].detach().cpu().numpy()
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, 
                xticklabels=input_tokens, 
                yticklabels=output_tokens,
                cmap='Blues',
                cbar=True)
    plt.title(f'注意力权重可视化 - 层 {layer}, 头 {head}')
    plt.xlabel('输入词汇')
    plt.ylabel('输出词汇')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 示例：创建一个小型 Transformer 模型
if __name__ == "__main__":
    # 模型参数
    src_vocab_size = 1000  # 源词汇表大小
    tgt_vocab_size = 1000  # 目标词汇表大小
    d_model = 512          # 模型维度
    num_heads = 8          # 注意力头数
    num_layers = 6         # 层数
    d_ff = 2048           # 前馈网络维度
    max_len = 100         # 最大序列长度
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print("Transformer 模型创建成功！")
    
    # 创建示例输入
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # 创建掩码
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    
    # 前向传播
    with torch.no_grad():
        output, enc_attn, dec_attn = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    print(f"编码器注意力层数: {len(enc_attn)}")
    print(f"解码器注意力层数: {len(dec_attn)}")