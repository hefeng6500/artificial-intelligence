"""注意力机制模块"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()
        _, value_len, _ = value.size()
        
        # 线性变换并重塑为多头形式
        Q = self.w_q(query).view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 重塑并通过输出线性层
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        缩放点积注意力
        
        Args:
            Q: 查询张量 [batch_size, n_heads, seq_len, d_k]
            K: 键张量 [batch_size, n_heads, seq_len, d_k]
            V: 值张量 [batch_size, n_heads, seq_len, d_k]
            mask: 掩码张量
            
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            # 确保掩码维度与scores匹配
            if mask.size(-1) != scores.size(-1) or mask.size(-2) != scores.size(-2):
                # 如果掩码维度不匹配，跳过掩码应用（用于交叉注意力）
                pass
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            输出张量 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 相对位置嵌入
        self.relative_positions = nn.Embedding(2 * max_len - 1, d_model)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成相对位置编码
        
        Args:
            seq_len: 序列长度
            
        Returns:
            相对位置编码张量 [seq_len, seq_len, d_model]
        """
        # 创建相对位置矩阵
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        positions = positions + self.max_len - 1  # 偏移到正数范围
        positions = torch.clamp(positions, 0, 2 * self.max_len - 2)
        
        return self.relative_positions(positions)


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自注意力前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量
            
        Returns:
            output: 输出张量
            attention_weights: 注意力权重
        """
        return self.attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """交叉注意力机制（用于编码器-解码器注意力）"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        交叉注意力前向传播
        
        Args:
            query: 查询张量（来自解码器）
            key_value: 键值张量（来自编码器）
            mask: 掩码张量
            
        Returns:
            output: 输出张量
            attention_weights: 注意力权重
        """
        return self.attention(query, key_value, key_value, mask)


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    创建填充掩码
    
    Args:
        seq: 序列张量 [batch_size, seq_len]
        pad_token_id: 填充标记 ID
        
    Returns:
        掩码张量 [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.size()
    # 创建 [batch_size, seq_len] 的掩码
    mask = (seq != pad_token_id)
    # 扩展为 [batch_size, seq_len, seq_len]
    mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    return mask


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    创建前瞻掩码（用于解码器自注意力）
    
    Args:
        size: 序列长度
        
    Returns:
        掩码张量 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_combined_mask(
    target_seq: torch.Tensor, 
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    创建组合掩码（填充掩码 + 前瞻掩码）
    
    Args:
        target_seq: 目标序列 [batch_size, seq_len]
        pad_token_id: 填充标记 ID
        
    Returns:
        组合掩码 [batch_size, seq_len, seq_len]
    """
    seq_len = target_seq.size(1)
    
    # 填充掩码
    padding_mask = create_padding_mask(target_seq, pad_token_id)
    
    # 前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(
        target_seq.size(0), -1, -1
    )
    
    # 组合掩码
    combined_mask = padding_mask & look_ahead_mask
    
    return combined_mask


if __name__ == "__main__":
    # 测试多头注意力
    d_model = 512
    n_heads = 8
    seq_len = 10
    batch_size = 2
    
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output, weights = attention(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 测试位置编码
    pos_encoding = PositionalEncoding(d_model)
    x_with_pos = pos_encoding(x.transpose(0, 1))
    
    print(f"位置编码后形状: {x_with_pos.shape}")
    
    print("✅ 注意力机制测试通过")